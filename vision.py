"""Landmarks and measured metrics. No LLM here — everything in this file is
deterministic, runs on CPU in well under a second, and costs nothing.
"""
import os
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

# MediaPipe 1.0 removed the legacy `mp.solutions` API. This uses the Tasks API,
# which works on both 0.10.x and 1.0.x. The Tasks API needs a model file, which
# is downloaded once on first run and cached next to this script.
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "face_landmarker.task")
_LANDMARKER = None


def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        print(f"downloading face_landmarker.task (~3.8 MB) -> {MODEL_PATH}")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def _landmarker():
    """Built once and reused — creating it per image is slow."""
    global _LANDMARKER
    if _LANDMARKER is None:
        _LANDMARKER = mp_vision.FaceLandmarker.create_from_options(
            mp_vision.FaceLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(model_asset_path=_ensure_model()),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
            )
        )
    return _LANDMARKER

# FaceMesh landmark sets, one per analysis region.
#
# These are SETS, not traced outlines: `region_polygon` takes the convex hull,
# so the order of the indices does not matter. That is deliberate. Hand-ordered
# outlines are silently fragile — get two indices out of sequence and you get a
# self-intersecting bowtie that still fills, still measures, and still renders,
# just over the wrong pixels. A hull cannot fail that way, and every region here
# is convex enough for one.
#
# LEFT AND RIGHT ARE THE SUBJECT'S, not the viewer's. In MediaPipe's canonical
# mesh, landmarks 33/133/…  are the subject's RIGHT eye and 362/263/… are the
# subject's LEFT, which is the opposite of the side they appear on in a
# front-facing photo. Clinical convention is the patient's own left and right,
# and the specialist report is read by someone who will act on it.
#
# VERIFY VISUALLY after any edit: `python run.py images/face.jpg --debug-mesh`.
REGION_INDICES = {
    # Hairline down to just above the brow line, temple to temple. The lower
    # edge is a hull chord, so it is drawn from the inner brow landmarks —
    # taking the outer ones (70, 300) drops that chord onto the brows, and
    # brow hair is not skin.
    "forehead": [10, 338, 297, 332, 284, 251, 301, 296, 334, 9,
                 107, 105, 66, 71, 21, 54, 103, 67, 109],
    # Between the brows, above the nasion.
    "glabella": [9, 8, 336, 285, 168, 55, 107],
    # Dorsum, tip and alae.
    "nose": [168, 6, 197, 195, 5, 4, 1, 2, 45, 275, 220, 440, 115, 344,
             48, 278, 64, 294, 98, 327, 129, 358],
    # Mid-cheek, below the lower lid and inside the nasolabial fold.
    "right_cheek": [117, 118, 119, 101, 100, 36, 205, 206, 207, 187, 123, 116, 50],
    "left_cheek":  [346, 347, 348, 330, 329, 266, 425, 426, 427, 411, 352, 345, 280],
    # Eye apertures. Dilated into an orbital ring and punched out below, so what
    # is measured is the skin around the eye and never the eye itself.
    "periorbital_right": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157,
                          158, 159, 160, 161, 246],
    "periorbital_left":  [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384,
                          385, 386, 387, 388, 466],
    # Outer lip line, dilated and punched out — the skin around the mouth.
    "perioral": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270,
                 269, 267, 0, 37, 39, 40, 185],
    # Below the lower lip, down to the jaw edge.
    "chin": [17, 18, 200, 199, 175, 152, 148, 176, 377, 400],
    "jawline_right": [172, 136, 150, 149, 176, 210, 214, 135, 138, 215, 177, 58],
    "jawline_left":  [397, 365, 379, 378, 400, 430, 434, 364, 367, 435, 401, 288],
}

# Regions defined by a feature outline rather than by the skin around it: the
# hull is grown about its centroid, and the feature itself becomes a hole.
REGION_DILATE = {"periorbital_right": 1.5, "periorbital_left": 1.5, "perioral": 1.7}
REGION_HOLES = {r: REGION_INDICES[r] for r in REGION_DILATE}

T_ZONE = ["forehead", "glabella", "nose"]


def detect_landmarks(bgr):
    """Returns Nx2 float array of pixel coordinates, or None if no face."""
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = _landmarker().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.face_landmarks:
        return None
    return np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]],
                    dtype=np.float32)


def face_width(pts):
    """Normalise every size measurement by this, or you are just measuring how
    close the person held their phone."""
    return float(np.linalg.norm(pts[454] - pts[234]))


def _hull(pts, indices, dilate=1.0):
    """Convex hull of a landmark set, optionally grown about its own centroid."""
    p = pts[indices].astype(np.float32)
    if dilate != 1.0:
        c = p.mean(axis=0)
        p = c + (p - c) * dilate
    return cv2.convexHull(p.astype(np.int32))[:, 0, :]


def region_polygon(pts, region):
    """Closed polygon for a region, in original-image pixel coordinates."""
    return _hull(pts, REGION_INDICES[region], REGION_DILATE.get(region, 1.0))


def region_mask(shape, pts, region):
    """Filled region, with any feature hole (eye aperture, lips) removed.

    Measuring an eyeball as periorbital skin or a lip as perioral skin would
    put a large, strongly coloured non-skin area into every redness and texture
    statistic for those regions.
    """
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [region_polygon(pts, region)], 255)
    hole = REGION_HOLES.get(region)
    if hole is not None:
        cv2.fillPoly(m, [_hull(pts, hole)], 0)
    return m


# ------------------------------------------------------------------ crops

def _square_box(cx, cy, side, img_w, img_h):
    """Square box centred on (cx, cy), shifted to stay inside the image rather
    than shrunk — a consistent crop size keeps the zoom factor honest across
    regions. Only shrinks if the image itself is smaller than the box."""
    side = int(min(side, img_w, img_h))
    x = int(round(cx - side / 2))
    y = int(round(cy - side / 2))
    x = max(0, min(x, img_w - side))
    y = max(0, min(y, img_h - side))
    return x, y, side


def crop_box(bgr, cx, cy, side, out_px=420):
    """Cut a square from the image and scale it to out_px.

    Returns the crop plus the magnification it represents, so the caller can
    state it rather than implying a zoom that isn't there. INTER_CUBIC on the
    way up, INTER_AREA on the way down — upscaling with AREA looks soft and
    these crops exist to be looked at closely.
    """
    h, w = bgr.shape[:2]
    x, y, side = _square_box(cx, cy, side, w, h)
    patch = bgr[y:y + side, x:x + side]
    if patch.size == 0:
        return None
    interp = cv2.INTER_CUBIC if out_px > side else cv2.INTER_AREA
    scaled = cv2.resize(patch, (out_px, out_px), interpolation=interp)
    return {"image": scaled, "zoom": round(out_px / side, 1),
            "source_px": side, "box": [x, y, side]}


def crop_region(bgr, pts, region, out_px=420, pad=1.0):
    """Zoomed crop of one analysis region.

    This is the evidence for a finding: the claim says 'redness, left cheek',
    and this is the left cheek at a size where you can actually see it.

    `pad` scales the region's bounding box. It defaults to 1.0 rather than
    something roomier because squaring the box already contributes surrounding
    skin on the narrow axis, and padding beyond that pulls an eye into every
    cheek crop — at which point the reader looks at the eye and not at the
    finding.
    """
    if region not in REGION_INDICES:
        return None
    poly = region_polygon(pts, region)
    x, y, w, h = cv2.boundingRect(poly)
    # Squaring on the long side is right for a roughly square region and wrong
    # for a wide flat one: the forehead spans temple to temple but is shallow,
    # and a square that wide reaches down over the eyes. Cap the square at twice
    # the short side and centre it, which keeps a wide region's crop on the
    # region.
    side = min(max(w, h), min(w, h) * 2) * pad
    return crop_box(bgr, x + w / 2, y + h / 2, side, out_px)


def crop_spots(bgr, pts, spots, out_px=420, pad=3.0):
    """Crop centred on the tightest cluster of blemishes, with the detected
    spots ringed. Drawn on the crop rather than described, because a count on
    its own ('5 spots') is a claim and this is the evidence for it.

    Picks the spot whose neighbours are closest, so the crop lands where the
    detections actually cluster instead of on an isolated outlier.
    """
    if not spots:
        return None
    pick, best = spots[0], None
    if len(spots) > 2:
        for s in spots:
            d = sorted(abs(complex(s["x"] - o["x"], s["y"] - o["y"]))
                       for o in spots if o is not s)
            spread = sum(d[:2])
            if best is None or spread < best:
                pick, best = s, spread

    side = max(face_width(pts) * 0.25, pick["r"] * 2 * pad)
    crop = crop_box(bgr, pick["x"], pick["y"], side, out_px)
    if crop is None:
        return None

    x0, y0, side = crop["box"]
    k = out_px / side
    shown = 0
    for s in spots:
        cx, cy = (s["x"] - x0) * k, (s["y"] - y0) * k
        # Centre must land inside the crop. Counting spots that merely overlap
        # the edge would put a number in the caption larger than the number of
        # rings the reader can actually count.
        if not (0 <= cx < out_px and 0 <= cy < out_px):
            continue
        cv2.circle(crop["image"], (int(cx), int(cy)), int(max(s["r"] * k, 4)),
                   (48, 48, 200), max(2, int(out_px / 180)), cv2.LINE_AA)
        shown += 1
    crop["marked"] = shown
    return crop


# ---------------------------------------------------------------- quality gate

def quality_gate(bgr, pts):
    """Cheap local checks. Runs before you spend a token."""
    issues = []
    h, w = bgr.shape[:2]

    if min(h, w) < 480:
        issues.append("resolution_too_low")

    blur = cv2.Laplacian(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    if blur < 60:
        issues.append("image_blurry")

    if pts is not None:
        fw = face_width(pts)
        if fw / w < 0.25:
            issues.append("face_too_small_in_frame")
        # Rough yaw check: nose tip should sit near the midpoint of the face edges.
        mid = (pts[234][0] + pts[454][0]) / 2
        if abs(pts[4][0] - mid) / fw > 0.13:
            issues.append("head_turned_too_far")

    return {"usable": not issues, "issues": issues, "blur_score": float(round(blur, 1))}


# ------------------------------------------------------------ measured metrics

def measure_oiliness(bgr, pts):
    """Specular highlights: oily skin reflects light differently. High value,
    low saturation. Returns fraction of each T-zone region that is shining."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    S, V = hsv[:, :, 1], hsv[:, :, 2]
    out = {}
    for region in T_ZONE + ["left_cheek", "right_cheek"]:
        m = region_mask(bgr.shape, pts, region)
        area = int(m.sum() / 255)
        if area == 0:
            continue
        v_thresh = np.percentile(V[m > 0], 92)
        shine = ((V > max(v_thresh, 180)) & (S < 60) & (m > 0)).sum()
        out[region] = round(float(shine) / area, 3)
    return out


def measure_pores(bgr, pts):
    """Local high-frequency contrast, normalised by face width so the score is
    scale-invariant. Maps to the same 0-10 scale as the LLM metrics."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    fw = face_width(pts)
    scores, hot = {}, []
    for region in ["nose", "left_cheek", "right_cheek"]:
        m = region_mask(bgr.shape, pts, region)
        if m.sum() == 0:
            continue
        # Band-pass: subtract a blur to isolate pore-scale detail.
        k = max(3, int(fw / 60) | 1)
        detail = cv2.absdiff(gray, cv2.GaussianBlur(gray, (k, k), 0))
        val = float(detail[m > 0].mean())
        s = int(np.clip(val * 2.2, 0, 10))
        scores[region] = s
        if s >= 5:
            hot.append(region)
    overall = int(round(np.mean(list(scores.values())))) if scores else None
    return {"score": overall, "per_region": scores, "regions": hot}


def measure_blemishes(bgr, pts):
    """Blob detection on the LAB a-channel. Output doubles as your marker
    layer — these coordinates are in original image pixels."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    a = lab[:, :, 1].astype(np.float32)
    fw = face_width(pts)

    face = np.zeros(bgr.shape[:2], np.uint8)
    for region in REGION_INDICES:
        face |= region_mask(bgr.shape, pts, region)
    # Exclude eyes and mouth — they are red and are not blemishes.
    for region in ["periorbital_left", "periorbital_right", "perioral"]:
        face &= ~region_mask(bgr.shape, pts, region)

    # Local redness excess: how much redder than the surrounding skin.
    bg = cv2.GaussianBlur(a, (0, 0), fw / 12)
    excess = a - bg
    excess[face == 0] = 0

    thresh = float(np.percentile(excess[face > 0], 99)) if (face > 0).any() else 0
    binary = ((excess > max(thresh, 3.0)) & (face > 0)).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lo, hi = (fw * 0.008) ** 2 * 3.14, (fw * 0.055) ** 2 * 3.14
    spots = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (lo < area < hi):
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        spots.append({"x": round(float(x), 1), "y": round(float(y), 1),
                      "r": round(float(max(r, fw * 0.008)), 1)})
    spots.sort(key=lambda s: -s["r"])
    return {"count": len(spots), "spots": spots[:60]}


def measure_all(bgr, pts):
    return {
        "oiliness": measure_oiliness(bgr, pts),
        "pore_size": measure_pores(bgr, pts),
        "blemishes": measure_blemishes(bgr, pts),
        "face_width_px": round(face_width(pts), 1),
    }
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

    The square is placed on the hull's centroid and sized to the square root of
    the hull's area — both rotation-invariant, which the alternative is not. A
    bounding box grows when the region it contains is tilted, so on a turned or
    tilted head a box-derived crop drifts off the region and pulls in whatever
    is next to it; on the forehead that is the eyes, and then the reader looks
    at the eyes instead of the finding. Equal-area sizing also adapts to shape
    on its own: a wide shallow band gets a square that sits inside it rather
    than one as wide as the band is long.

    `pad` scales that square. It defaults to 1.0 because the crop already picks
    up surrounding skin wherever the region is not itself square.
    """
    if region not in REGION_INDICES:
        return None
    poly = region_polygon(pts, region).astype(np.int32)
    m = cv2.moments(poly)
    if m["m00"] <= 0:
        return None
    area = cv2.contourArea(poly.astype(np.float32))
    return crop_box(bgr, m["m10"] / m["m00"], m["m01"] / m["m00"],
                    (area ** 0.5) * pad, out_px)


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

# Mirrored regions. A turned head foreshortens the far side of the face, and the
# ratio of a region's area to its mirror's is a cheap, deterministic measure of
# how much. On a frontal photo these sit near 1.0; past a moderate yaw the far
# jawline drops below 0.3.
MIRRORED = [("left_cheek", "right_cheek"),
            ("jawline_left", "jawline_right"),
            ("periorbital_left", "periorbital_right")]

# Below this, a region is too foreshortened to measure or to show. Chosen from
# the gap between a frontal face (~0.76 at worst) and a clearly turned one
# (~0.31). Recalibrate on real photos.
VISIBILITY_MIN = 0.60

YAW_ADVISORY = 0.18   # far side starts to foreshorten
YAW_FATAL = 0.35      # far side is gone; not worth a model call


def face_axes(pts):
    """The face's own left-right axis and its width along it.

    Yaw has to be measured in this frame, not in the image's. A rolled head
    rotates the nose-tip offset partly into the vertical, and an image-space
    check reads that as less yaw than there is.
    """
    v = pts[454] - pts[234]
    fw = float(np.linalg.norm(v))
    return v / fw, fw


def measure_yaw(pts):
    """Nose-tip offset from the midline as a fraction of face width, in the
    face's own frame, so head tilt does not contaminate it."""
    ex, fw = face_axes(pts)
    mid = (pts[234] + pts[454]) / 2
    return abs(float(np.dot(pts[4] - mid, ex))) / fw


def region_visibility(pts):
    """Per-region visibility, 1.0 meaning not foreshortened relative to its
    mirror. Unmirrored regions are always 1.0 — there is nothing to compare
    them against, and none of them sit on the edge of a turning face."""
    vis = {r: 1.0 for r in REGION_INDICES}
    for a, b in MIRRORED:
        aa = cv2.contourArea(region_polygon(pts, a).astype(np.float32))
        ab = cv2.contourArea(region_polygon(pts, b).astype(np.float32))
        if max(aa, ab) <= 0:
            continue
        ratio = min(aa, ab) / max(aa, ab)
        vis[a if aa < ab else b] = round(ratio, 2)
    return vis


def foreshortened(pts):
    """Regions too turned away to measure or to crop. Showing a reader a patch
    of hair captioned 'right cheek' costs more trust than the missing finding
    was ever worth."""
    return sorted(r for r, v in region_visibility(pts).items() if v < VISIBILITY_MIN)


def quality_gate(bgr, pts):
    """Cheap local checks. Runs before you spend a token.

    Two severities, because they call for different things. `issues` are fatal:
    the photo cannot support an assessment and the pipeline stops before paying
    for one. `advisories` are worth recording and worth telling the specialist,
    but the photo is still worth analysing — most real selfies carry some tilt
    or turn, and a gate that rejects them rejects the product.
    """
    issues, advisories = [], []
    h, w = bgr.shape[:2]

    if min(h, w) < 480:
        issues.append("resolution_too_low")

    blur = cv2.Laplacian(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    if blur < 60:
        issues.append("image_blurry")

    yaw, hidden = None, []
    if pts is not None:
        _, fw = face_axes(pts)
        if fw / w < 0.25:
            issues.append("face_too_small_in_frame")

        yaw = measure_yaw(pts)
        if yaw > YAW_FATAL:
            issues.append("head_turned_too_far")
        elif yaw > YAW_ADVISORY:
            advisories.append("head_turned")

        hidden = foreshortened(pts)
        if hidden:
            advisories.append("regions_foreshortened")

    return {
        "usable": not issues,
        "issues": issues,
        "advisories": advisories,
        "blur_score": float(round(blur, 1)),
        "yaw": None if yaw is None else round(yaw, 3),
        "foreshortened": hidden,
    }


# ------------------------------------------------------------ measured metrics

def face_mask(shape, pts, regions=None):
    """Union of the analysis regions — the whole measurable face."""
    m = np.zeros(shape[:2], np.uint8)
    for region in (regions or REGION_INDICES):
        m |= region_mask(shape, pts, region)
    return m


def measure_oiliness(bgr, pts, skip=()):
    """Where the shine sits on this face.

    Specular reflection bounces off the surface without meeting pigment, so it
    carries the colour of the light rather than of the skin: bright *and*
    desaturated. That conjunction is the signature, and it is what this looks
    for.

    Both thresholds come from the whole face's own distribution, which matters
    twice over. It cancels exposure and white balance, which no phone selfie
    controls for. And it removes a hard ceiling in the previous version, where
    the threshold was a per-region percentile — that made the answer 'is the
    top 8% of this region brighter than the rest of it', which by construction
    could never exceed 0.08 no matter how oily the skin was.

    What this is NOT is an absolute measure of sebum. A photograph cannot give
    you one: a directionally lit face has a bright side, and no amount of pixel
    work separates 'lit' from 'oily' in a single uncontrolled frame. Read it as
    a within-face distribution — which is the clinically interesting axis
    anyway, since an oily T-zone against dry cheeks is what a routine is built
    around. `distribution` is the honest headline; the per-region shares are
    the detail behind it.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    S, V = hsv[:, :, 1], hsv[:, :, 2]

    face = face_mask(bgr.shape, pts)
    if not face.any():
        return {"distribution": "not assessed", "shine_share": {}}

    s_lo = np.percentile(S[face > 0], 15)
    v_hi = np.percentile(V[face > 0], 85)
    spec = (S < s_lo) & (V > v_hi)

    shares = {}
    for region in T_ZONE + ["left_cheek", "right_cheek"]:
        if region in skip:
            continue
        m = region_mask(bgr.shape, pts, region)
        area = int(m.sum() / 255)
        if area == 0:
            continue
        shares[region] = round(float((spec & (m > 0)).sum()) / area, 3)

    t = [shares[r] for r in T_ZONE if r in shares]
    c = [shares[r] for r in ("left_cheek", "right_cheek") if r in shares]
    if t and c:
        # Guarded: matte cheeks make the raw ratio explode, and the answer to
        # "how many times shinier" is not a number anyone should read anyway.
        ratio = float(np.mean(t)) / max(float(np.mean(c)), 0.005)
        dist = "t_zone" if ratio >= 2 else "cheeks" if ratio < 0.5 else "even"
    else:
        dist = "not assessed"

    return {"distribution": dist, "shine_share": shares}


def measure_pores(bgr, pts, skip=()):
    """Local high-frequency contrast, normalised by face width so the score is
    scale-invariant. Maps to the same 0-10 scale as the LLM metrics."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    fw = face_width(pts)
    scores, hot = {}, []
    for region in ["nose", "left_cheek", "right_cheek"]:
        if region in skip:
            continue
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


def blemish_field(bgr, pts):
    """The skin the detector is allowed to look at, and local redness excess
    over it.

    Two exclusions beyond the eyes and mouth, both learned from false positives
    the evidence crops made visible. The region hulls overshoot onto hair when
    it falls across the face, so dark pixels are dropped. And the hulls' own
    edges are where the blurred background estimate is least reliable, so the
    mask is eroded before anything is measured there.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, a = lab[:, :, 0].astype(np.float32), lab[:, :, 1].astype(np.float32)
    fw = face_width(pts)

    face = face_mask(bgr.shape, pts)
    for region in ["periorbital_left", "periorbital_right", "perioral"]:
        face &= ~region_mask(bgr.shape, pts, region)

    k = max(3, int(fw * 0.012)) | 1
    face = cv2.erode(face, np.ones((k, k), np.uint8))
    if face.any():
        # Hair, nostril and deep shadow. All are much darker than lit skin and
        # all of them read as locally red against a blurred neighbourhood.
        face[L < np.median(L[face > 0]) * 0.72] = 0

    excess = a - cv2.GaussianBlur(a, (0, 0), fw / 12)
    excess[face == 0] = 0
    return face, excess, fw


def measure_blemishes(bgr, pts):
    """Blob detection on the LAB a-channel. Output doubles as your marker
    layer — these coordinates are in original image pixels.

    The threshold is a robust multiple of the image's own noise, not a fixed
    percentile. A percentile keeps a *constant fraction* of pixels whatever is
    on the face, so a clear face and a badly broken-out one returned nearly the
    same count. Median plus 2 MAD-sigma adapts to noise without adapting to how
    much there is to find.
    """
    face, excess, fw = blemish_field(bgr, pts)
    if not face.any():
        return {"count": 0, "spots": []}

    d = excess[face > 0]
    med = float(np.median(d))
    sigma = 1.4826 * float(np.median(np.abs(d - med)))

    binary = ((excess > med + 2.0 * sigma) & (face > 0)).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # The floor is the smallest blob worth calling a spot. It was 0.008 of face
    # width, which is larger than most real blemish cores once thresholded —
    # 68 of 72 candidates on the test face were being discarded by it alone.
    lo, hi = (fw * 0.004) ** 2 * 3.14, (fw * 0.055) ** 2 * 3.14
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
    """Every deterministic measurement, with foreshortened regions left out.

    A region turned away from the camera still has a hull and still has pixels,
    so it still produces a number — one that describes the shadow it is sitting
    in. Dropping it is more useful than reporting it, and the reviewing
    specialist gets told which ones were dropped.
    """
    skip = foreshortened(pts)
    return {
        "oiliness": measure_oiliness(bgr, pts, skip=skip),
        "pore_size": measure_pores(bgr, pts, skip=skip),
        "blemishes": measure_blemishes(bgr, pts),
        "face_width_px": round(face_width(pts), 1),
        "not_measured": skip,
    }
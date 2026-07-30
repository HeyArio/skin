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

# FaceMesh index sets tracing each analysis region.
# VERIFY THESE VISUALLY before trusting them: run `python run.py --debug-mesh
# images/yourface.jpg` to render the polygons and adjust. They are a working
# starting point, not gospel.
REGION_INDICES = {
    "forehead":          [10, 109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58, 172, 136, 150],
    "glabella":          [9, 336, 296, 334, 293, 300, 107, 66, 105, 63, 70],
    "nose":              [168, 6, 197, 195, 5, 4, 45, 220, 115, 48, 64, 98, 327, 294, 278, 344, 440, 275],
    "left_cheek":        [117, 118, 119, 120, 121, 128, 205, 36, 142, 126, 47, 100, 101, 50, 123],
    "right_cheek":       [346, 347, 348, 349, 350, 357, 425, 266, 371, 355, 277, 329, 330, 280, 352],
    "periorbital_left":  [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    "periorbital_right": [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466],
    "perioral":          [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185],
    "chin":              [18, 200, 199, 175, 152, 148, 176, 149, 150, 169, 210, 214],
    "jawline_left":      [172, 136, 150, 149, 176, 148, 152, 234, 227, 137, 177, 215, 138, 135, 169],
    "jawline_right":     [397, 365, 379, 378, 400, 377, 152, 454, 447, 366, 401, 435, 367, 364, 394],
}

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


def region_polygon(pts, region):
    return pts[REGION_INDICES[region]].astype(np.int32)


def region_mask(shape, pts, region):
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [region_polygon(pts, region)], 255)
    return m


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
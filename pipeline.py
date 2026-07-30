"""Orchestration: image in, findings + two reports + overlay out."""
import base64
import json
import os

import cv2

import prompts
import vision

# The model can only cite regions the prompt offers it, and the renderer can
# only draw regions vision.py knows. When those two lists drift apart the
# mismatch is silent — findings are filtered out and simply never appear. Fail
# at import instead.
_mismatch = set(prompts.REGIONS) ^ set(vision.REGION_INDICES)
if _mismatch:
    raise RuntimeError(
        f"region vocabulary mismatch between prompts.REGIONS and "
        f"vision.REGION_INDICES: {sorted(_mismatch)}")

RAMP = ["#E1F5EE", "#9FE1CB", "#FAC775", "#EF9F27", "#D85A30", "#A32D2D"]
BANDS = [(2, "minimal"), (4, "mild"), (6, "moderate"), (8, "notable"), (10, "significant")]


def band(score):
    if score is None:
        return "not assessed"
    return next(name for hi, name in BANDS if score <= hi)


def _colour(score):
    return RAMP[min(5, max(0, score) // 2)]


REGION_LABEL = {
    "forehead": "Forehead", "glabella": "Between the brows", "nose": "Nose",
    "left_cheek": "Left cheek", "right_cheek": "Right cheek",
    "periorbital_left": "Around the left eye",
    "periorbital_right": "Around the right eye",
    "perioral": "Around the mouth", "chin": "Chin",
    "jawline_left": "Left jawline", "jawline_right": "Right jawline",
}


def _jpeg_b64(bgr, quality=88):
    return base64.b64encode(
        cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])[1]).decode()


def build_receipts(bgr, pts, findings, limit=4):
    """Evidence crops — one per finding worth showing.

    Every claim the report makes is about a place on a face, and until you show
    that place at a size where it can be seen, the claim is just an assertion.
    These are the receipts: the actual pixels the finding came from.

    Selection is deterministic — no model involved. Regions are ranked by the
    strongest score citing them, so the crops lead with whatever is most worth
    looking at rather than walking the metrics in declaration order. A region
    cited by several metrics gets one crop carrying all of them, which is also
    what stops the strip from repeating the same patch of cheek three times.
    """
    cited = {}
    for name, data in findings.get("metrics", {}).items():
        score = data.get("score")
        if score is None:
            continue
        for region in data.get("regions") or []:
            if region in vision.REGION_INDICES:
                cited.setdefault(region, []).append((name, score))

    # Measured metrics earn a receipt on the same terms as judged ones.
    for region, score in (findings.get("pore_size", {}).get("per_region") or {}).items():
        if region in vision.REGION_INDICES and region in \
                (findings.get("pore_size", {}).get("regions") or []):
            cited.setdefault(region, []).append(("pore size", score))

    ranked = sorted(cited.items(),
                    key=lambda kv: (-max(s for _, s in kv[1]), -len(kv[1]), kv[0]))

    spots = findings.get("blemishes", {}).get("spots") or []
    room = limit - 1 if len(spots) >= 3 else limit

    receipts = []
    for region, metrics in ranked[:max(room, 0)]:
        crop = vision.crop_region(bgr, pts, region)
        if crop is None:
            continue
        metrics.sort(key=lambda m: -m[1])
        receipts.append({
            "kind": "region",
            "region": region,
            "label": REGION_LABEL.get(region, region.replace("_", " ").capitalize()),
            "claim": " · ".join(f"{n} {band(s)}" for n, s in metrics),
            "metrics": [{"metric": n, "score": s, "band": band(s)} for n, s in metrics],
            "zoom": crop["zoom"], "source_px": crop["source_px"], "box": crop["box"],
            "image_b64": _jpeg_b64(crop["image"]),
        })

    if len(spots) >= 3 and len(receipts) < limit:
        crop = vision.crop_spots(bgr, pts, spots)
        if crop is not None:
            n = crop.get("marked", 0)
            receipts.append({
                "kind": "spots",
                "region": None,
                "label": "Blemishes",
                "claim": f"{n} of {len(spots)} detected spots in view",
                "metrics": [],
                "zoom": crop["zoom"], "source_px": crop["source_px"], "box": crop["box"],
                "image_b64": _jpeg_b64(crop["image"]),
            })

    return receipts


def render_overlay(findings, pts, w, h):
    """Three layers, one SVG. viewBox is the image's own pixel space, so these
    coordinates work unchanged in the browser and in the specialist PDF."""
    heat, outlines, spots = [], [], []
    stroke = max(1.5, w / 400)

    for name, data in findings.get("metrics", {}).items():
        score = data.get("score")
        if score is None:
            continue
        for region in data.get("regions") or []:
            if region not in vision.REGION_INDICES:
                continue
            pl = vision.region_polygon(pts, region)
            pstr = " ".join(f"{x},{y}" for x, y in pl)
            c = _colour(score)
            heat.append(f'<polygon points="{pstr}" fill="{c}" fill-opacity="{score/10*0.40:.2f}"/>')
            outlines.append(
                f'<polygon points="{pstr}" fill="none" stroke="{c}" stroke-width="{stroke:.1f}"/>')

    for s in findings.get("blemishes", {}).get("spots", []):
        spots.append(f'<circle cx="{s["x"]}" cy="{s["y"]}" r="{s["r"]}" fill="none" '
                     f'stroke="#A32D2D" stroke-width="{max(1.2, w/500):.1f}"/>')

    return (f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" '
            f'style="position:absolute;inset:0;width:100%;height:100%">'
            f'<g class="layer" data-layer="heat">{"".join(heat)}</g>'
            f'<g class="layer" data-layer="regions">{"".join(outlines)}</g>'
            f'<g class="layer" data-layer="spots">{"".join(spots)}</g></svg>')


def analyse(path, mock=False, runs=None):
    bgr = cv2.imread(path)
    if bgr is None:
        raise RuntimeError(f"could not read {path}")
    h, w = bgr.shape[:2]

    pts = vision.detect_landmarks(bgr)
    if pts is None:
        return {"path": path, "rejected": "no_face_detected"}

    gate = vision.quality_gate(bgr, pts)
    measured = vision.measure_all(bgr, pts)

    if mock:
        scored = json.loads(open(os.path.join(os.path.dirname(__file__),
                                              "mock_scoring.json")).read())
    else:
        import llm
        n = runs if runs is not None else int(os.environ.get("SCORING_RUNS", "3"))
        with open(path, "rb") as f:
            raw = f.read()
        mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
        scored = llm.score_image_median(
            raw, prompts.SCORING_SYSTEM, prompts.SCORING_USER, n=n, mime=mime)

    findings = {
        **scored,
        "local_quality": gate,
        "oiliness": measured["oiliness"],
        "pore_size": measured["pore_size"],
        "blemishes": measured["blemishes"],
        "face_width_px": measured["face_width_px"],
    }
    # Bands are what the UI shows. Raw scores stay internal — a 6-to-7 wobble is
    # invisible in a band and glaring on a dial.
    findings["bands"] = {k: band(v.get("score"))
                         for k, v in findings.get("metrics", {}).items()}
    findings["bands"]["pore_size"] = band(measured["pore_size"]["score"])

    result = {
        "path": path, "width": w, "height": h,
        "findings": findings,
        "overlay_svg": render_overlay(findings, pts, w, h),
        "receipts": build_receipts(bgr, pts, findings),
        "image_b64": _jpeg_b64(bgr, 82),
    }

    if not mock:
        import llm
        slim = {k: v for k, v in findings.items() if k != "blemishes"}
        slim["blemish_count"] = findings["blemishes"]["count"]

        user_text = llm.write_report(prompts.USER_REPORT_SYSTEM, slim)
        safe, hit = prompts.check_user_text(user_text)
        result["user_report"] = user_text if safe else prompts.SAFE_FALLBACK
        result["denylist_trip"] = hit          # log this — a rising rate means drift
        result["user_report_raw"] = user_text  # kept for review only, never shipped

        result["specialist_report"] = llm.write_report(
            prompts.SPECIALIST_REPORT_SYSTEM, slim)

    return result

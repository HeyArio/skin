#!/usr/bin/env python3
"""Phase 0 harness.

    python run.py images/                 # full pipeline
    python run.py images/ --mock          # no API calls, tests CV + rendering
    python run.py images/ --runs 1        # single scoring pass while iterating
    python run.py --debug-mesh face.jpg   # verify the region polygons

Opens out/review.html — image, overlay, both reports, raw JSON, side by side.
That page is the deliverable: show it to your specialist and ask whether she
would act on it.
"""
import argparse
import html
import json
import os
import sys
import traceback

import cv2

import pipeline
import vision

EXTS = (".jpg", ".jpeg", ".png", ".webp")


def debug_mesh(path):
    bgr = cv2.imread(path)
    pts = vision.detect_landmarks(bgr)
    if pts is None:
        sys.exit("no face detected")
    for i, region in enumerate(vision.REGION_INDICES):
        col = [(0, 200, 255), (0, 255, 120), (255, 120, 0), (255, 0, 200)][i % 4]
        poly = vision.region_polygon(pts, region)
        cv2.polylines(bgr, [poly], True, col, 2)
        cv2.putText(bgr, region, tuple(poly[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
    out = "out/mesh_debug.jpg"
    cv2.imwrite(out, bgr)
    print(f"wrote {out} — check every polygon sits where you expect, then tune "
          f"REGION_INDICES in vision.py")


CSS = """
body{font:14px/1.55 system-ui,sans-serif;margin:0;padding:24px;background:#faf9f7;color:#1a1a19}
h1{font-size:19px;margin:0 0 20px}
.card{background:#fff;border:1px solid #e5e3de;border-radius:12px;padding:18px;margin-bottom:22px;display:grid;grid-template-columns:340px 1fr;gap:22px}
.imgwrap{position:relative;align-self:start}
.imgwrap img{width:100%;display:block;border-radius:8px}
.toggles{display:flex;gap:6px;margin-bottom:10px}
.toggles button{flex:1;font:12px system-ui;padding:5px;border:1px solid #d5d2cc;border-radius:6px;background:#fff;cursor:pointer}
.toggles button[data-on="0"]{opacity:.4}
h3{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:#7a7770;margin:16px 0 6px}
h3:first-child{margin-top:0}
pre{background:#f5f3ef;padding:11px;border-radius:7px;overflow:auto;font-size:11px;max-height:280px;margin:0}
.report{background:#f5f3ef;padding:12px;border-radius:7px;white-space:pre-wrap;margin:0}
.warn{background:#fdecea;border-left:3px solid #A32D2D;padding:9px 12px;border-radius:5px;margin:8px 0;font-size:13px}
.pills span{display:inline-block;background:#eceae5;border-radius:20px;padding:2px 10px;margin:0 5px 5px 0;font-size:12px}
"""

JS = """
document.querySelectorAll('.toggles button').forEach(b=>{
  b.onclick=()=>{
    const on=b.dataset.on!=='0'; b.dataset.on=on?'0':'1';
    b.closest('.card').querySelector(`[data-layer="${b.dataset.t}"]`)
      .style.opacity=on?'0':'1';
  };
});
"""


def card(r):
    if r.get("rejected"):
        return (f'<div class="card"><div>{html.escape(os.path.basename(r["path"]))}</div>'
                f'<div class="warn">Rejected: {r["rejected"]}</div></div>')

    f = r["findings"]
    pills = "".join(f"<span>{k}: {v}</span>" for k, v in f.get("bands", {}).items())
    issues = (f.get("local_quality", {}).get("issues") or []) + \
             (f.get("image_quality", {}).get("issues") or [])
    warn = f'<div class="warn">Quality: {", ".join(issues)}</div>' if issues else ""
    trip = (f'<div class="warn">Denylist tripped on "{r["denylist_trip"]}" — '
            f'fallback text shown to user</div>') if r.get("denylist_trip") else ""
    spread = f.get("_spread")
    spr = (f'<div class="warn">Score spread across runs: {spread} — '
           f'anything above 2 means the rubric needs tightening</div>'
           if spread and max(spread.values(), default=0) > 2 else "")

    def sec(title, body, cls="report"):
        return f"<h3>{title}</h3><div class='{cls}'>{html.escape(body)}</div>" if body else ""

    return f"""<div class="card">
<div>
  <div class="toggles">
    <button data-t="heat">heat</button>
    <button data-t="regions">regions</button>
    <button data-t="spots">spots</button>
  </div>
  <div class="imgwrap">
    <img src="data:image/jpeg;base64,{r['image_b64']}">
    {r['overlay_svg']}
  </div>
</div>
<div>
  <strong>{html.escape(os.path.basename(r['path']))}</strong>
  &nbsp;<span style="color:#7a7770">skin age {f.get('skin_age_band','—')} ·
  {f.get('blemishes',{}).get('count',0)} spots</span>
  {warn}{trip}{spr}
  <div class="pills" style="margin-top:10px">{pills}</div>
  {sec("User report", r.get("user_report",""))}
  {sec("Specialist report", r.get("specialist_report",""))}
  <h3>Findings JSON</h3>
  <pre>{html.escape(json.dumps(f, indent=2, ensure_ascii=False))}</pre>
</div></div>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("--mock", action="store_true", help="skip API calls")
    ap.add_argument("--runs", type=int, default=None, help="scoring passes (default 3)")
    ap.add_argument("--debug-mesh", action="store_true")
    a = ap.parse_args()

    os.makedirs("out", exist_ok=True)
    if a.debug_mesh:
        return debug_mesh(a.target)

    files = ([a.target] if os.path.isfile(a.target) else
             sorted(os.path.join(a.target, f) for f in os.listdir(a.target)
                    if f.lower().endswith(EXTS)))
    if not files:
        sys.exit(f"no images found in {a.target}")

    results = []
    for i, p in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {os.path.basename(p)} ... ", end="", flush=True)
        try:
            r = pipeline.analyse(p, mock=a.mock, runs=a.runs)
            results.append(r)
            print("rejected: " + r["rejected"] if r.get("rejected") else "ok")
        except Exception as e:
            print(f"FAILED: {e}")
            traceback.print_exc(limit=2)

    with open("out/results.json", "w") as fh:
        json.dump([{k: v for k, v in r.items() if k not in ("image_b64", "overlay_svg")}
                   for r in results], fh, indent=2, ensure_ascii=False)

    with open("out/review.html", "w") as fh:
        fh.write(f"<!doctype html><meta charset=utf-8><style>{CSS}</style>"
                 f"<h1>Skin analysis review — {len(results)} image(s)</h1>"
                 + "".join(card(r) for r in results)
                 + f"<script>{JS}</script>")

    print(f"\nout/review.html  ({len(results)} processed)\nout/results.json")


if __name__ == "__main__":
    main()

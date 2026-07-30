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
import shutil
import subprocess
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
.receipts{display:grid;grid-template-columns:repeat(auto-fill,minmax(148px,1fr));gap:10px;margin:0}
.receipt{border:1px solid #e5e3de;border-radius:8px;overflow:hidden;background:#fff}
.crop{position:relative;line-height:0}
.crop img{width:100%;display:block}
.crop b{position:absolute;right:5px;bottom:5px;background:rgba(26,26,25,.72);color:#fff;
  font:10px/1.5 system-ui;padding:1px 6px;border-radius:10px;letter-spacing:.02em}
.cap{padding:7px 9px;line-height:1.35}
.cap strong{display:block;font-size:12px}
.cap span{font-size:11px;color:#7a7770}
.cap i{font-style:normal;white-space:nowrap}
.cap i+i:before{content:" · ";white-space:normal}

/* Grid children default to min-width:auto, so one long unbreakable line in the
   JSON dump widens the whole column and pushes the evidence strip off the page.
   This is the fix for that, and it has to apply on screen as well as in print. */
.card>*{min-width:0}
/* Full width, so the dump does not stretch a column it shares with the
   evidence strip — and so it does not leave a page of blank space beside it. */
.json{grid-column:1/-1}
.json h3{margin-top:4px}

@media print{
  @page{size:A4;margin:13mm}
  body{padding:0;background:#fff;font-size:11px}
  h1{font-size:15px}
  /* Cards run longer than a page and are allowed to break. avoid-page here
     would push each one to a fresh page and then break it anyway. The blocks
     inside are what must stay intact. */
  .card{grid-template-columns:230px 1fr;gap:16px;margin-bottom:14px;border-radius:0}
  .toggles{display:none}
  .receipts{grid-template-columns:repeat(4,1fr);break-inside:avoid-page}
  .report,.receipt,.warn{break-inside:avoid-page}
  /* No scroll box to scroll in a PDF: let it flow, wrap it, and shrink it, so
     the JSON stays auditable without costing a page per image. */
  pre{max-height:none;overflow:visible;font-size:8px;line-height:1.35;
      white-space:pre-wrap;overflow-wrap:anywhere}
}
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


def receipts_html(receipts):
    """The evidence strip: every claim next to the pixels it came from.

    The zoom badge is only drawn when there is real magnification to report —
    labelling a 1.0x crop as a zoom would be the kind of small dishonesty that
    makes a reader stop trusting the rest of the page.
    """
    if not receipts:
        return ""
    out = []
    for rc in receipts:
        zoom = (f'<b>{rc["zoom"]:.1f}&times;</b>' if rc.get("zoom", 0) >= 1.2 else "")
        # Each metric is its own nowrap chunk so a narrow column breaks between
        # findings instead of stranding a separator at the end of a line.
        if rc.get("metrics"):
            claim = "".join(f'<i>{html.escape(m["metric"])} {html.escape(m["band"])}</i>'
                            for m in rc["metrics"])
        else:
            claim = f'<i>{html.escape(rc["claim"])}</i>'
        out.append(
            f'<div class="receipt"><div class="crop">'
            f'<img src="data:image/jpeg;base64,{rc["image_b64"]}" alt="">{zoom}</div>'
            f'<div class="cap"><strong>{html.escape(rc["label"])}</strong>'
            f'<span>{claim}</span></div></div>')
    return f'<h3>Evidence</h3><div class="receipts">{"".join(out)}</div>'


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
  {receipts_html(r.get("receipts"))}
  {sec("User report", r.get("user_report",""))}
  {sec("Specialist report", r.get("specialist_report",""))}
</div>
<div class="json">
  <h3>Findings JSON</h3>
  <pre>{html.escape(json.dumps(f, indent=2, ensure_ascii=False))}</pre>
</div></div>"""


# ------------------------------------------------------------------------ pdf

# Ordered by preference. The Playwright cache is first because if a machine has
# it, it is the one Chromium whose version is known.
CHROME_CANDIDATES = [
    os.environ.get("CHROME_BIN"),
    os.path.join(os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "/opt/pw-browsers"), "chromium"),
    "chromium", "chromium-browser", "google-chrome", "google-chrome-stable",
    "chrome", "microsoft-edge",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
]


def find_chrome():
    for c in CHROME_CANDIDATES:
        if not c:
            continue
        found = shutil.which(c) or (c if os.path.isfile(c) and os.access(c, os.X_OK) else None)
        if found:
            return found
    return None


def html_to_pdf(html_path, pdf_path):
    """Print the review page with headless Chromium.

    Rendering the same HTML rather than building the PDF separately is the
    whole point: there is one layout to maintain, and the SVG overlay, the
    crops and the print rules are all the ones already verified in a browser.
    A second renderer would be a second thing that can disagree.
    """
    chrome = find_chrome()
    if not chrome:
        raise RuntimeError(
            "no Chrome/Chromium found — install one, or set CHROME_BIN to its path.\n"
            "  macOS:  already there if you have Chrome\n"
            "  Debian: apt install chromium\n"
            "  any:    pip install playwright && playwright install chromium")

    cmd = [chrome, "--headless", "--disable-gpu", "--no-sandbox",
           "--no-pdf-header-footer", "--run-all-compositor-stages-before-draw",
           # Data-URI images decode asynchronously; without a budget Chromium
           # can print before the crops have painted.
           "--virtual-time-budget=10000",
           f"--print-to-pdf={os.path.abspath(pdf_path)}",
           "file://" + os.path.abspath(html_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if not os.path.exists(pdf_path) or os.path.getsize(pdf_path) == 0:
        raise RuntimeError(f"{os.path.basename(chrome)} produced no PDF\n"
                           f"{(proc.stderr or proc.stdout or '').strip()[-800:]}")
    return chrome


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("--mock", action="store_true", help="skip API calls")
    ap.add_argument("--runs", type=int, default=None, help="scoring passes (default 3)")
    ap.add_argument("--debug-mesh", action="store_true")
    ap.add_argument("--pdf", action="store_true",
                    help="also render out/review.pdf (needs Chrome/Chromium)")
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

    # Crops are base64 like the other images — keep their metadata in the JSON
    # (that is what you audit) and leave the pixels in the review page.
    def slim(r):
        out = {k: v for k, v in r.items() if k not in ("image_b64", "overlay_svg")}
        if out.get("receipts"):
            out["receipts"] = [{k: v for k, v in rc.items() if k != "image_b64"}
                               for rc in out["receipts"]]
        return out

    with open("out/results.json", "w") as fh:
        json.dump([slim(r) for r in results], fh, indent=2, ensure_ascii=False)

    with open("out/review.html", "w", encoding="utf-8") as fh:
        fh.write("<!doctype html><meta charset=utf-8>"
                 "<meta name=viewport content='width=device-width,initial-scale=1'>"
                 f"<title>Skin analysis review</title><style>{CSS}</style>"
                 f"<h1>Skin analysis review — {len(results)} image(s)</h1>"
                 + "".join(card(r) for r in results)
                 + f"<script>{JS}</script>")

    print(f"\nout/review.html  ({len(results)} processed)\nout/results.json")

    if a.pdf:
        try:
            chrome = html_to_pdf("out/review.html", "out/review.pdf")
            print(f"out/review.pdf   (via {os.path.basename(chrome)})")
        except Exception as e:
            print(f"\nPDF not written: {e}")


if __name__ == "__main__":
    main()

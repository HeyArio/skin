#!/usr/bin/env python3
"""Phase 0 harness.

    python run.py images/                 # full pipeline -> out/report.pdf
    python run.py images/ --mock          # no API calls, tests CV + rendering
    python run.py images/ --runs 1        # single scoring pass while iterating
    python run.py images/ --html          # skip the PDF, leave the HTML
    python run.py face.jpg --debug-mesh   # verify the region polygons

Writes out/report.pdf — the photo with its overlay, the findings, the evidence
crops and both reports. That document is the deliverable: put it in front of
your specialist and ask whether she would act on it.

out/results.json holds the same run in machine-readable form.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback

import cv2

import config
import pipeline
import report
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


# ------------------------------------------------------------------------ pdf

def _chrome_candidates():
    """Where a Chrome or Chromium lives, per platform.

    Windows and macOS do not put it on PATH, so the install locations have to be
    named. CHROME_BIN overrides everything, and a Playwright cache comes next
    because if a machine has one, that is the Chromium whose version is known.
    """
    env = os.environ
    pw = env.get("PLAYWRIGHT_BROWSERS_PATH", "/opt/pw-browsers")
    names = ["chrome", "chromium", "chromium-browser", "google-chrome",
             "google-chrome-stable", "msedge", "microsoft-edge"]
    paths = [env.get("CHROME_BIN"), os.path.join(pw, "chromium")]

    if sys.platform == "win32":
        for base in filter(None, [env.get("PROGRAMFILES"), env.get("PROGRAMFILES(X86)"),
                                  env.get("LOCALAPPDATA")]):
            paths += [
                os.path.join(base, "Google", "Chrome", "Application", "chrome.exe"),
                os.path.join(base, "Chromium", "Application", "chrome.exe"),
                os.path.join(base, "Microsoft", "Edge", "Application", "msedge.exe"),
            ]
    elif sys.platform == "darwin":
        paths += ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                  "/Applications/Chromium.app/Contents/MacOS/Chromium",
                  "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"]

    return paths + names


INSTALL_HINT = """no Chrome, Chromium or Edge found, so the PDF was not written.
  Windows : install Google Chrome, or set CHROME_BIN to chrome.exe
  macOS   : install Google Chrome
  Debian  : apt install chromium
  any     : pip install playwright && playwright install chromium
Then re-run. out/report.html is already written and opens in any browser."""


def find_chrome():
    for c in _chrome_candidates():
        if not c:
            continue
        found = shutil.which(c) or (c if os.path.isfile(c) and os.access(c, os.X_OK) else None)
        if found:
            return found
    return None


def html_to_pdf(html_path, pdf_path):
    """Print the report with headless Chromium.

    Rendering the same HTML rather than composing the PDF separately keeps one
    layout to maintain; the print rules live beside the screen rules in
    report.CSS. A second renderer would be a second thing that can disagree.
    """
    chrome = find_chrome()
    if not chrome:
        raise RuntimeError(INSTALL_HINT)

    pdf_path = os.path.abspath(pdf_path)
    if os.path.exists(pdf_path):
        os.remove(pdf_path)          # so a stale file cannot look like success

    cmd = [chrome, "--headless", "--disable-gpu", "--no-sandbox",
           "--no-pdf-header-footer", "--run-all-compositor-stages-before-draw",
           # Data-URI images decode asynchronously; without a budget Chromium
           # can print before the crops have painted.
           "--virtual-time-budget=15000",
           f"--print-to-pdf={pdf_path}",
           "file:///" + os.path.abspath(html_path).lstrip("/").replace("\\", "/")]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
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
    ap.add_argument("--html", action="store_true",
                    help="skip the PDF and leave the HTML (it has layer toggles)")
    ap.add_argument("--no-gate", action="store_true",
                    help="analyse images the quality gate would reject — use "
                         "this to calibrate the thresholds against real photos")
    a = ap.parse_args()

    os.makedirs("out", exist_ok=True)
    if a.debug_mesh:
        return debug_mesh(a.target)

    files = ([a.target] if os.path.isfile(a.target) else
             sorted(os.path.join(a.target, f) for f in os.listdir(a.target)
                    if f.lower().endswith(EXTS)))
    if not files:
        sys.exit(f"no images found in {a.target}")

    results, stopped = [], None
    for i, p in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {os.path.basename(p)} ... ", end="", flush=True)
        try:
            r = pipeline.analyse(p, mock=a.mock, runs=a.runs,
                                 gate_images=not a.no_gate)
            results.append(r)
            print("rejected: " + r["rejected"] if r.get("rejected") else "ok")
        except Exception as e:
            # A rejected credential is true for every remaining image. Stop, and
            # keep whatever already succeeded rather than burning the rest of
            # the batch on the same rejected call.
            if isinstance(e, config.AuthError):
                print("FAILED")
                stopped = str(e)
                break
            print(f"FAILED: {e}")
            traceback.print_exc(limit=2)

    if stopped:
        print(f"\n{stopped}\n")
    if not results:
        # An empty report is worse than none: it looks like a finished run.
        sys.exit("nothing was analysed — no report written.")

    # The images are base64 and would dwarf everything else. The report carries
    # the pixels; this file carries the numbers.
    def slim(r):
        out = {k: v for k, v in r.items() if k not in ("image_b64", "overlay_svg")}
        if out.get("receipts"):
            out["receipts"] = [{k: v for k, v in rc.items() if k != "image_b64"}
                               for rc in out["receipts"]]
        return out

    with open("out/results.json", "w", encoding="utf-8") as fh:
        json.dump([slim(r) for r in results], fh, indent=2, ensure_ascii=False)

    with open("out/report.html", "w", encoding="utf-8") as fh:
        fh.write(report.document(results))

    n = len(results)
    if a.html:
        print(f"\nout/report.html   ({n} image{'s'[:n != 1]})\nout/results.json")
        return

    try:
        chrome = html_to_pdf("out/report.html", "out/report.pdf")
        print(f"\nout/report.pdf    ({n} image{'s'[:n != 1]}, via "
              f"{os.path.basename(chrome)})\nout/results.json")
    except Exception as e:
        print(f"\n{e}")
        print("out/results.json")


if __name__ == "__main__":
    main()

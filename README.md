# Skin analysis — Phase 0 harness

Image in → findings JSON + user report + specialist report + toggleable overlay.
Batches a folder and builds one review page you can hand to your specialist.

**The question this answers is not "does the code work" — it's "would she act on
this report?"** Everything else is solvable engineering. Run 20 real photos
through it before you build a product around it.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env      # fill in, then: set -a; source .env; set +a
```

Add `.env` to `.gitignore`. Your gateway URL has the credential in its path, so
the whole URL is a secret — it leaks through browser history, server logs and
stack traces more easily than a header would.

## 1. Probe the endpoint first

```bash
python probe.py
```

Sends a two-colour test image four ways and reports which shapes the gateway
accepts. Do this before anything else — some gateways proxy only the text path,
and you want to find that out now rather than after building around it. Set
`GATEWAY_STYLE` from the result.

## 2. Check the region polygons

```bash
python run.py images/someface.jpg --debug-mesh
```

Writes `out/mesh_debug.jpg` with every region outlined and labelled. The index
sets in `vision.py` are a working starting point, not gospel — expect to spend
an hour nudging them. Wrong polygons mean the overlay lands in the wrong place
and every measured metric samples the wrong pixels.

## 3. Dry run, no API calls

```bash
python run.py images/ --mock
```

Uses `mock_scoring.json` instead of calling the model. Verifies the CV pipeline,
the overlay renderer and the review page without spending a token.

## 4. Real run

```bash
python run.py images/            # median-of-3 scoring
python run.py images/ --runs 1   # single pass, while iterating on the rubric
python run.py images/ --pdf      # also write out/review.pdf
python run.py images/ --no-gate  # analyse photos the quality gate would reject
```

Open `out/review.html`.

`--pdf` prints the same page with headless Chrome or Chromium, found via
`CHROME_BIN`, a Playwright cache, or a system install. The HTML is written
either way — it is what gets printed. Only the layer toggles are lost.

## What to look at in the review page

- **Score spread** — flagged when median-of-3 disagrees by more than 2. That's
  the rubric being ambiguous, not the model being bad. Tighten the anchors in
  `prompts.py`.
- **Denylist trips** — the user report tried to name a condition. Track the rate.
  A rise means the prompt has drifted or the model changed under you.
- **Rejections** — how many real photos fail the quality gate. If it's most of
  them, your gate is too strict and users will bounce before they ever see a
  result. Thresholds are in `vision.quality_gate`; run `--no-gate` to see what
  you would have got from the photos it turned away.
- **Advisories and `not_measured`** — the amber note on a card. The photo was
  analysed, but some regions were too turned away to measure and were dropped.
  A lot of these means your capture flow needs to coach people to face the
  camera, not that the analysis is failing.
- **The evidence crops.** Each finding is shown next to the zoomed patch of skin
  it came from. Can you see the thing the band claims is there? If you can't,
  the score is wrong, the region polygon is wrong, or the rubric anchor is too
  loose — and which of the three is usually obvious from the crop.
- **The reports themselves.** Would your specialist act on them? That's the
  whole point of Phase 0.

## Cost per image

Roughly 525 Toman at median-of-3 — about 390 for the three Gemini vision passes,
135 for the two text calls. Twenty test images is around 10,500 Toman.

## Layout

```
prompts.py    rubric, both report prompts, denylist   <- tune this most
vision.py     FaceMesh regions, quality gate, CV metrics
llm.py        gateway client, median-of-3
pipeline.py   orchestration + SVG overlay renderer
run.py        CLI + review page builder
probe.py      endpoint shape detector
```

## Notes

- **Oiliness, pores and blemishes never touch the model.** They're measured with
  OpenCV — deterministic, free, and they don't drift. The blemish coordinates
  double as the marker layer.
- **Oiliness is a distribution, not a quantity** — `t_zone`, `even` or `cheeks`.
  A photo can't tell you how oily a face is, only where the shine sits: a
  directionally lit face has a bright side and no pixel work separates "lit"
  from "oily" in one frame.
- **Regions turned away from the camera are dropped, not guessed at**, and
  listed in `not_measured`.
- **Everything is normalised by face width**, so scores don't change with how
  close the phone was held.
- **Every finding gets an evidence crop** — the actual pixels it came from, at
  2-3x, chosen deterministically from the scores. No tokens, no model.
- **Left and right are the subject's**, not the viewer's, because a specialist
  reading "left cheek" will examine the patient's left cheek.
- **The UI shows bands, not numbers.** Raw scores stay internal. A 6-to-7 wobble
  is invisible in a band and glaring on a dial.
- **The denylist is enforced in code**, not left to the prompt. Prompts drift;
  validators don't. On a trip the user sees a safe fallback and the incident is
  logged — it does not retry and hope.
- **For before/after, send both images in one call** and ask for the comparison
  directly. Do not diff two separately-scored sessions.

## Not yet built

Storage, auth, consent flow, specialist delivery, booking. Deliberately — none
of it matters if the reports aren't good enough, and all of it is cheap to add
once they are.

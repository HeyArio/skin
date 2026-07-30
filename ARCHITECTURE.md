# Skin analysis pipeline — how it works

Reference for the Phase 0 harness. Written so that someone joining the project
(or you, in three months) can understand not just what each piece does but why
it was built that way.

---

## 1. What the system does

A user uploads a selfie. They get back two things:

- a **written report** in plain language
- the **same photograph** with the findings drawn onto it, in three toggleable
  layers

A second, clinical version of the report goes to a skincare specialist, who
reviews it and can then be booked for a consultation.

The product is B2C, but the specialist in the loop is what makes the whole
design work. Because a qualified human reviews every result, the pipeline does
not need to be diagnostically accurate — it needs to be **accurate enough to
start a useful conversation**. That single fact drives most of the decisions
below.

---

## 2. The architecture in one picture

```
                      selfie (JPEG/PNG)
                             │
              ┌──────────────┴──────────────┐
              │                             │
      [quality gate]                        │        local, free, instant
       blur / size /                        │        rejects bad photos
       angle / face size                    │        BEFORE spending money
              │                             │
       reject ◄┘ (if it fails)              │
                                            │
              ┌─────────────────────────────┴──────────────┐
              │                                            │
      [CV PIPELINE]                                [CALL 1 — VISION]
      OpenCV, deterministic                       Gemini 3.1 Flash-Lite
      no model, no cost                           median of 3 runs
              │                                            │
      oiliness, pore size,                        dryness, texture, redness,
      blemish coordinates                         pigmentation, lines,
              │                                   skin age band, patterns
              │                                            │
              └────────────────┬───────────────────────────┘
                               │
                        findings.json
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
 [CALL 2 — TEXT]       [SVG RENDERER]         [CALL 3 — TEXT]
 Gemini Flash-Lite     local, free            Gemini Flash-Lite
 (text only, no image) (no model at all)      (text only, no image)
        │                      │                      │
 user report            3-layer overlay        specialist report
        │
 [DENYLIST CHECK] ──► fails ──► safe fallback text
        │
  shown to user
```

**Three model calls, not one.** Only Call 1 ever sees the image. Calls 2 and 3
receive the findings JSON as text. This matters for four reasons:

1. **Cost.** Text calls carry no image tokens, so they cost roughly half a
   vision pass each even on the same model. And because Call 1 is the only one
   worth running three times for a median, keeping the report calls separate
   means you pay for the median once, not three times over.
2. **Consistency.** A model re-reading pixels to write prose would reinterpret
   them. Reading structured numbers gives the same prose every time.
3. **Privacy.** Calls 2 and 3 carry region names and integers — no face, no
   identity. They can be routed to any provider without a data-protection
   question. Only Call 1 needs a provider you have a contract with. This
   currently costs you nothing to keep true, and it is the property that lets
   you take EU users later without re-architecting.
4. **Swappability.** Each stage has one input shape and one output shape. Any
   stage can be replaced without touching the others.

### One model or two?

All three calls currently run on **Gemini 3.1 Flash-Lite**. That is the right
call at this stage: one key, one gateway, one request shape, one thing to probe
and debug. Operational simplicity is worth more right now than token savings.

The cheaper alternative is a small text-only model such as gpt-oss-120b for
Calls 2 and 3, which would cut roughly 250 Toman per scan — about a third of the
total. Worth revisiting once volume is real, and it is a config change rather
than a code change, because those two calls already go through their own
`TEXT_URL` / `TEXT_MODEL` settings. Nothing in the pipeline assumes the three
calls share a model.

---

## 3. Measured versus judged

The most important design decision in the project: **some metrics are never
shown to a language model.**

| Metric | Who produces it | Why |
|---|---|---|
| Oiliness | OpenCV | Specular reflection never meets pigment, so it carries the colour of the light rather than the skin: bright *and* desaturated. That conjunction is measurable. **Its magnitude is not** — see below. |
| Pore size | OpenCV | High-frequency local contrast, band-passed and normalised by face width. |
| Blemishes | OpenCV | Blob detection on the LAB a-channel. Returns exact pixel coordinates — which also become the marker layer. |
| Dryness, texture, redness, pigmentation, lines | Gemini | Genuinely qualitative. This is what vision models are good at. |
| Skin age band | Gemini | A composite judgement, not a measurement. |
| Dehydration | **Neither** | Not reliably visible in a photograph. Dehydrated skin can look oily, dry, or normal. Belongs in a questionnaire, not a pixel analysis. Guessing it produces confident nonsense. |

Everything measured is **normalised by face width** (the distance between
landmarks 234 and 454). Without that, pore and blemish sizes would just be
measuring how close the person held their phone.

### Oiliness is a distribution, not a quantity

A single uncontrolled selfie cannot tell you how oily a face is. A directionally
lit face has a bright side, and nothing in the pixels separates "lit" from
"oily" in one frame. Any absolute number would be a number about the lighting.

So oiliness is reported as a **within-face distribution** — `t_zone`, `even` or
`cheeks` — with the per-region shares behind it. That is exposure-independent,
it is what survives the confound, and it happens to be the clinically
interesting axis anyway: an oily T-zone against dry cheeks is what a routine
gets built around.

Both thresholds are drawn from the whole face's own distribution. Taking them
per region, as the first version did, capped the answer at 0.08 by construction
— the question became "is the top 8% of this region brighter than the rest of
it", which has the same answer on every face.

Vision-language models are unreliable at precise pixel coordinates. Asking one
for bounding boxes around blemishes produces markers vaguely near the right
place, plus invented ones. Google's own model card for this family lists
spatial-localisation confusion as a known limitation. So the model is never
asked for a coordinate — it returns **region names** from a fixed vocabulary,
and the geometry comes from face landmarks.

---

## 4. Files

```
prompts.py       Rubric, both report prompts, denylist          <- tune most often
vision.py        Landmark detection, region polygons,
                 quality gate, all CV measurements
llm.py           Gateway client, median-of-3, JSON recovery
pipeline.py      Orchestration + the SVG overlay renderer
run.py           CLI, batching, review-page builder
probe.py         Endpoint shape detector (run this first)
mock_scoring.json  Canned scoring output for --mock runs
.env             Gateway URLs and settings (never commit)
face_landmarker.task   Downloaded on first run, ~3.8 MB
```

### prompts.py

Holds the rubric, both report prompts, and the denylist. This is the file you
will edit most.

**The rubric must never move to a knowledge base.** It has to arrive
byte-identical on every request, or scores drift between sessions. A knowledge
base retrieves varying chunks, which is exactly the wrong property here. Kept in
version control, a scoring change is a diff you can review and roll back.

Each metric's 0–10 scale has explicit written anchors — what a 4 looks like
versus a 7. Undefined scales are where drift is born.

### vision.py

Landmark detection uses MediaPipe's **Tasks API** (`FaceLandmarker`), not the
old `mp.solutions` API, which was removed in mediapipe 1.0. The Tasks API exists
in both 0.10.x and 1.0.x, so the code runs on either. The landmarker is built
once and reused — constructing it per image is slow.

`REGION_INDICES` maps each analysis region to a **set** of FaceMesh landmark
indices. `region_polygon` takes the convex hull of that set, so index order does
not matter. Because landmarks track the actual face, these polygons fit any face
at any size or angle automatically.

The hull is the point. Hand-ordered outlines fail silently: swap two indices and
you get a self-intersecting polygon that still fills, still measures, and still
renders — just over the wrong pixels, with nothing in the output to say so. A
hull cannot fail that way, and every region here is convex enough for one.

Two regions are defined by the feature at their centre rather than by the skin
around it. `REGION_DILATE` grows the periorbital and perioral hulls about their
own centroid, and `REGION_HOLES` punches the eye aperture or the lip line back
out, so what gets measured is a ring of skin and never an eyeball or a lip.

**Left and right are the subject's, not the viewer's.** In MediaPipe's canonical
mesh, landmarks 33/133 are the subject's *right* eye and 362/263 the subject's
*left* — the opposite of the side they appear on in a front-facing photo. The
specialist report is read by someone who will act on it, and clinical convention
is the patient's own left and right.

**These index sets need visual verification.** Run `--debug-mesh` and check that
each labelled polygon sits where it should. Wrong polygons mean the overlay
lands in the wrong place, every measured metric samples the wrong pixels, *and*
the evidence crops show the wrong patch of skin.

### The quality gate

Runs before any API call: resolution, Laplacian blur variance, face size in
frame, and yaw. It has **two severities**, because they call for different
things.

`issues` are fatal. The photo cannot support an assessment, and `analyse`
returns a rejection without paying for one. `advisories` are recorded and shown
but do not stop anything — most real selfies carry some tilt or turn, and a gate
that rejects those rejects the product. `run.py --no-gate` analyses everything
regardless, which is how you calibrate the thresholds: you cannot tune a gate
from the near side of it.

Yaw is measured **in the face's own frame**, not the image's. A rolled head
rotates the nose-tip offset partly into the vertical, and an image-space
comparison reads that as less yaw than there is — on the test photos it
understated a real turn by about a fifth.

Yaw does not only decide whether a photo is usable, though. Past a moderate
turn, the far side of the face is foreshortened, and a region that is turned
away still has a hull, still has pixels, and still yields a number — one that
describes the shadow it is sitting in. So `region_visibility` compares each
mirrored region's area against its twin, and anything below `VISIBILITY_MIN` is
dropped from the measurements and from the evidence crops, and listed in
`not_measured`. On a frontal photo these ratios sit near 1.0; on a turned one
the far jawline falls to about 0.3.

Dropping a region is better than reporting it, and saying which ones were
dropped is better than either. A crop of hair captioned "right cheek" costs more
trust than the missing finding was ever worth.

### llm.py

Supports both request shapes — OpenAI-compatible (`/chat/completions` with
`image_url`) and Gemini-native (`inline_data`) — selected by `GATEWAY_STYLE`,
because proxy gateways vary in which they implement. Both the vision call and
the two text calls build through the same `_payload`, so the reports cannot
quietly stay OpenAI-shaped on a gateway where `GATEWAY_STYLE=gemini`.

Includes retry with backoff, and `_parse_json`, which strips markdown fences and
falls back to a brace-matching regex, because models sometimes wrap JSON despite
being told not to.

`score_image_median` runs Call 1 N times and takes the per-metric median. A
region survives only if a majority of runs flagged it. It also records
`_spread` — the max-minus-min per metric — which is your drift signal.

### pipeline.py

Orchestrates everything and renders the overlay. Also converts raw scores to
**bands** (minimal / mild / moderate / notable / significant), which is what the
UI shows.

### run.py

Batches a folder and writes `out/review.html` — every image with its overlay,
toggle buttons, the evidence crops, both reports, and the raw JSON. Optionally
prints the same page to `out/review.pdf`. That page is the actual deliverable of
Phase 0.

---

## 5. The overlay renderer

The overlay is a **rendering** job, not a generation job. Using a generative
image model here would be wrong on every axis: it regenerates the face rather
than annotating it, it can't place markers accurately, it produces different
output for identical input (destroying progress tracking), it can't be toggled
because the result is a flat bitmap, and it costs roughly seven times the entire
rest of the pipeline combined.

Instead: the photo is an `<img>`, and an `<svg>` sits on top of it in a
`position: relative` container.

```html
<div style="position:relative">
  <img src="selfie.jpg" style="width:100%">
  <svg viewBox="0 0 1080 1440" style="position:absolute;inset:0;width:100%">
    <g data-layer="heat">…</g>
    <g data-layer="regions">…</g>
    <g data-layer="spots">…</g>
  </svg>
</div>
```

**The `viewBox` is set to the image's natural pixel dimensions.** That one
detail is what makes everything else simple: every coordinate drawn is in
original-image pixel space, matching exactly what the CV pipeline returns, and
the whole thing scales responsively with no arithmetic.

The three layers:

| Layer | Source | Rendering |
|---|---|---|
| heat | Region polygons + metric score | Filled polygon, `fill-opacity = score/10 × 0.4` |
| regions | Same polygons | Stroked outline, colour from a 6-step severity ramp |
| spots | Blemish blob centroids | `<circle>` per spot, radius from blob area |

Toggling is `opacity` on the `<g>` group. No re-render, no second API call,
instant.

For the specialist's PDF, the identical coordinates go through `resvg` or Pillow
server-side — so what the doctor sees is pixel-for-pixel what the user saw.

---

## 5a. Evidence crops

Every band the report shows is a claim about a place on a face. Until that place
is shown at a size where it can be seen, the claim is an assertion the reader has
to take on faith — and "moderate redness, left cheek" reads exactly the same on
every face that ever passes through the pipeline. The crops are the fix, and they
are the difference between a result that looks generated and one that looks read.

`vision.crop_region` cuts a square around a region's hull and scales it up;
`vision.crop_spots` does the same around the tightest cluster of detected
blemishes and rings each one. `pipeline.build_receipts` chooses which crops to
make.

Three properties matter:

- **Selection is deterministic.** No model involved. Regions are ranked by the
  strongest score citing them, so the strip leads with whatever is most worth
  looking at rather than walking the metrics in declaration order. A region cited
  by several metrics gets one crop carrying all of them, which is what stops the
  strip from showing the same cheek three times.
- **The magnification is stated, and only when it is real.** A crop badged 2.5x
  is 2.5x. Below 1.2x no badge is drawn, because labelling a 1.0x crop as a zoom
  is the kind of small dishonesty that costs you the reader's trust in
  everything else on the page.
- **The crop stays on the region.** Squaring on the long side is right for a
  roughly square region and wrong for a wide flat one — a square as wide as the
  forehead reaches down over the eyes, and then the reader looks at the eyes.
  The square is capped at twice the short side.

They cost nothing: no tokens, no model, a few milliseconds of OpenCV.

---

## 6. The regulatory design

This is not a legal footnote bolted on; it shapes the code.

**The app must never name a medical condition to a user.** The moment the UI
says "rosacea," it is making a medical-device claim, which carries a regulatory
burden in the EU, the UAE, and most other markets. Rosacea, eczema, dermatitis
and the rest are *diagnoses*, not severity scores.

The solution is **one analysis, two vocabularies**:

- **User sees observation** — "persistent redness concentrated across the cheeks
  and nose," "dry, irritated patches along the jawline"
- **Specialist sees clinical** — "centrofacial erythema with visible
  telangiectasia; consider rosacea vs. seborrheic dermatitis — recommend
  in-person assessment"

Same findings JSON, two prompts, two audiences. This isn't only a legal dodge —
it's correct epistemics. The model shouldn't be diagnosing; the specialist
should. It also makes the specialist's judgement the product's value, which is
precisely what the booking funnel sells. If the app diagnosed, why book anyone?

**Enforcement is in code, not in the prompt.** `check_user_text()` runs a hard
denylist over every user-facing string before it reaches the renderer. On a trip,
the user gets a safe fallback and the incident is logged — it does **not** retry
the model and hope. Prompts drift; validators don't. A rising trip rate means
the prompt has drifted or the model changed under you, and you want to learn
that from a log, not from a user.

Skin age is shown as a **decade band**, never a number. Telling a 32-year-old
her skin age is 41 drives bookings and also drives uninstalls.

---

## 7. Controlling drift

The known weakness of LLM scoring: the same photo, scored twice, gives 6 then 7.
Five mitigations, in order of effect:

1. **Explicit rubric anchors.** Define what each score band means. This is the
   single biggest lever.
2. **`temperature: 0`** on the scoring call.
3. **Median-of-3.** At Flash-Lite prices this costs a rounding error. The review
   page flags any metric where runs disagreed by more than 2 — that's a signal
   the *rubric* is ambiguous, not that the model is bad.
4. **Bands, not numbers, in the UI.** A 6-to-7 wobble is invisible in a band and
   glaring on a dial.
5. **For before/after, send both images in one call** and ask for the comparison
   directly. Vision models are far more reliable at relative judgement than at
   absolute scoring. Never diff two separately-scored sessions.

---

## 8. Cost

Approximate, per scan, all calls on Gemini 3.1 Flash-Lite at ArvanCloud rates
(105,000 Toman per 1M input tokens, 315,000 per 1M output).

| Stage | Tokens | Toman |
|---|---|---|
| Call 1 — vision, single pass | ~2,500 in / 400 out | ~390 |
| Call 2 — user report | ~900 in / 280 out | ~183 |
| Call 3 — specialist report | ~900 in / 330 out | ~199 |
| CV pipeline | — | 0 |
| Overlay rendering | — | 0 |
| **Total, `SCORING_RUNS=1`** | | **~770** |
| **Total, `SCORING_RUNS=3`** | (Call 1 ×3 = ~1,170) | **~1,550** |

Ten thousand scans a month is roughly 7.7M Toman single-pass, or 15.5M with the
median. Either way, cost is not a factor in any architecture decision — but the
gap is large enough that you should not leave median-of-3 on before the rubric
is stable enough to need it.

Set `SCORING_RUNS=1` while iterating. You will re-run often, and the median is
only worth paying for once the prompt has settled.

The image itself is roughly 1,100–1,300 of those input tokens. Downscaling
selfies to around 1024px on the long edge before sending costs you nothing in
assessment quality and keeps that number predictable.

## 9. Configuration

```
VISION_URL      Gateway URL for the vision model. On ArvanCloud the credential
                is in the URL path, so treat the whole URL as a secret.
VISION_MODEL    Model identifier
VISION_KEY      Leave empty if the token is in the URL
TEXT_URL        Gateway URL for the report calls. Leave unset and it falls
                back to VISION_URL — which is what you want while everything
                runs on one model.
TEXT_MODEL      Leave unset and it falls back to VISION_MODEL, which is what
                you want while everything runs on one model
GATEWAY_STYLE   openai | gemini — run probe.py to determine
SCORING_RUNS    3 in production, 1 while iterating
```

The model is a **Preview** release. Preview models change and get
deprecated. The model name lives in config and the scoring call sits behind one
function, so a swap is a config change plus a re-run of the eval set — half an
hour, not a rebuild.

---

## 10. Running it

```bash
python probe.py                             # 1. does the gateway forward images?
python run.py images/face.jpg --debug-mesh  # 2. are the regions in the right place?
python run.py images/ --mock                # 3. dry run, no tokens
python run.py images/ --runs 1              # 4. first real pass
python run.py images/                       # 5. median-of-3 once tuned
python run.py images/ --pdf                 # 6. same page as out/review.pdf
python run.py images/ --no-gate             # analyse what the gate would reject
```

Open `out/review.html`.

`--pdf` prints that same page with headless Chrome or Chromium — whichever of
`CHROME_BIN`, a Playwright cache, or a system install it finds first. Rendering
the HTML rather than building the PDF separately means one layout to maintain
and no second renderer to disagree with the first; the print rules live in the
same stylesheet, under `@media print`. The HTML is still written either way,
since it is what gets printed. Only the toggles are lost, for the obvious
reason.

---

## 11. What to look at in the review page

- **Score spread** — flagged above 2. Tighten the rubric anchors.
- **Denylist trips** — track the rate over time.
- **Rejection rate** — if most real-world selfies fail the quality gate, that's
  your actual product problem. Users will bounce before they ever see a result.
  Thresholds are in `vision.quality_gate`.
- **The reports themselves.** Would your specialist act on them? That is the
  only question Phase 0 exists to answer. Everything else is solvable
  engineering; this is the one that decides whether there's a product.

---

## 12. Not built yet

Storage, auth, consent flow, specialist delivery, booking, before/after
tracking, the web front end.

Deliberately. None of it matters if the reports aren't good enough, and all of
it is cheap to add once they are.

When you do build it:

- **Consent and retention must exist before the first real user selfie** hits a
  server. Facial images are at the sensitive end of what GDPR covers.
- **Storage lifecycle deletion** should be configured at bucket creation, not
  added later.
- **The capture flow should run MediaPipe in the browser**, so bad photos are
  rejected before upload — no wasted bandwidth, no wasted tokens, and the user
  gets instant feedback to retake.
- **Data residency**: only Call 1 carries a face. If you take EU users, route
  that one call to a provider with a DPA and keep Calls 2 and 3 wherever is
  cheapest.

---

## 13. Known tuning points

| What | Where | Note |
|---|---|---|
| Region polygons | `vision.REGION_INDICES` | Verify with `--debug-mesh` after any edit. |
| Ring width, periorbital / perioral | `vision.REGION_DILATE` | How far the hull grows before the feature is punched out |
| Crop framing | `vision.crop_region`, `pad` | 1.0 keeps cheek crops off the eye |
| Receipts per image | `pipeline.build_receipts`, `limit` | 4 fits one row at print width |
| Blur threshold | `vision.quality_gate`, currently 60 | Calibrate against `blur_score` on real photos |
| Face-size minimum | `vision.quality_gate`, 0.25 of frame width | |
| Yaw tolerance | `vision.YAW_ADVISORY` 0.18, `vision.YAW_FATAL` 0.35 | Set from three photos. Recalibrate on twenty |
| Region visibility floor | `vision.VISIBILITY_MIN`, 0.60 | Frontal photos sit above 0.76, turned ones below 0.31 |
| Blemish size limits | `vision.measure_blemishes` | Fractions of face width. The floor was 0.008 and discarded 68 of 72 candidates |
| Blemish sensitivity | `vision.measure_blemishes`, median + 2.0 sigma | MAD-based, so it adapts to noise but not to how much there is to find |
| Oiliness percentiles | `vision.measure_oiliness`, S below 15th, V above 85th | Face-wide, not per region |
| Pore score scaling | `vision.measure_pores`, ×2.2 | Arbitrary — calibrate against faces you can judge |
| Rubric anchors | `prompts.SCORING_SYSTEM` | The highest-leverage file in the project |

"""The report document.

run.py owns the CLI and the PDF call; this owns what the page actually says.

The findings JSON is not in here on purpose. A reviewer reading a report should
not have to parse a data structure to find out what was found — every field
worth reading is given a line of its own below, and the machine-readable copy
stays in out/results.json where a machine can have it.
"""
import datetime
import html
import os

import pipeline

# Ordinal severity ramp — one hue, light to dark, five steps for five bands.
# Validated against the page surface for monotone lightness, step separation
# and light-end contrast.
SEVERITY_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
BANDS = ["minimal", "mild", "moderate", "notable", "significant"]

# Every severity bar is the same colour. The bar's length already encodes how
# much; colouring it by that same value would spend the one free channel
# restating what the reader can already see, and the metrics themselves have no
# natural order to put in a ramp.
BAR = "#2a78d6"

DATE = datetime.date.today().strftime("%d %b %Y")

METRIC_LABEL = {
    "dryness": "Dryness", "texture": "Texture", "redness": "Redness",
    "pigmentation": "Pigmentation", "lines": "Lines", "pore_size": "Pore visibility",
}


def _regions(names):
    return ", ".join(pipeline.REGION_LABEL.get(n, n.replace("_", " ")).lower()
                     for n in names)


def _rows(f):
    """Metric rows, worst first.

    Leading with the strongest finding rather than with whatever order the
    metrics happen to be declared in is most of what stops two reports from
    reading identically.
    """
    metrics = dict(f.get("metrics") or {})
    rows = []
    for key, data in metrics.items():
        band = (f.get("bands") or {}).get(key)
        if band is None or band not in BANDS:
            continue
        rows.append((key, band, data.get("regions") or []))

    pore = f.get("pore_size") or {}
    if (f.get("bands") or {}).get("pore_size") in BANDS:
        rows.append(("pore_size", f["bands"]["pore_size"], pore.get("regions") or []))

    rows.sort(key=lambda r: (-BANDS.index(r[1]), r[0]))
    return rows


def _chart(f):
    if not (rows := _rows(f)):
        return ""
    out = []
    for key, band, regions in rows:
        pct = (BANDS.index(band) + 1) / len(BANDS) * 100
        out.append(
            f'<div class="mrow">'
            f'<div class="mname">{html.escape(METRIC_LABEL.get(key, key))}</div>'
            f'<div class="track"><div class="bar" style="width:{pct:.0f}%"></div></div>'
            f'<div class="mband">{html.escape(band)}</div>'
            f'<div class="mwhere">{html.escape(_regions(regions)) or "&mdash;"}</div>'
            f'</div>')
    # No axis ticks. Every bar is already labelled with its band, and the skill
    # here is direct labels before gridlines — five tick words under a 150px
    # track collide into one another anyway.
    return f'<div class="chart">{"".join(out)}</div>'


def _measured(f):
    """The deterministic measurements, as sentences rather than as fields."""
    facts = []
    n = (f.get("blemishes") or {}).get("count")
    if n is not None:
        facts.append(("Blemishes detected", str(n)))

    shine = (f.get("oiliness") or {}).get("distribution")
    if shine and shine != "not assessed":
        facts.append(("Shine", {"t_zone": "concentrated on the T-zone",
                                "cheeks": "concentrated on the cheeks",
                                "even": "spread evenly across the face"}[shine]))

    age = f.get("skin_age_band")
    if age:
        facts.append(("Skin age band", age))
    return "".join(f'<div class="fact"><dt>{html.escape(k)}</dt>'
                   f'<dd>{html.escape(v)}</dd></div>' for k, v in facts)


def _distribution(f):
    """Where the detected blemishes actually sit.

    A count is one number and two faces often share it; the distribution almost
    never matches. It is also exact — a point-in-polygon test against the same
    landmark regions everything else uses — so it says something specific about
    this face without anything being inferred.
    """
    per = (f.get("blemishes") or {}).get("per_region") or {}
    total = (f.get("blemishes") or {}).get("count") or 0
    if not per or total < 3:
        return ""
    # A region that was too turned away to measure still has spots detected in
    # whatever of it is visible. Listing that count unqualified would
    # contradict the Confidence section further down the same page.
    turned = set(f.get("not_measured") or [])
    top = max(per.values())
    bars = "".join(
        f'<div class="drow"><div class="dname">'
        f'{html.escape(pipeline.REGION_LABEL.get(k, k).capitalize())}'
        f'{"<em>partial</em>" if k in turned else ""}</div>'
        f'<div class="track"><div class="bar{" partial" if k in turned else ""}" '
        f'style="width:{v / top * 100:.0f}%"></div></div>'
        f'<div class="dnum">{v}</div></div>'
        for k, v in per.items())
    placed = sum(per.values())
    note = ("" if placed == total else
            f'<p class="lede">{total - placed} of {total} fell outside the '
            f'mapped regions and are not shown.</p>')
    return f'<h3>Where the blemishes are</h3>{note}<div class="dist">{bars}</div>'


def _asymmetry(f):
    """Left-right differences, where both sides were measurable.

    Nobody looks at their own face closely enough to know how its sides differ,
    which makes this one of the few things a report can say that the reader
    cannot already see in a mirror. Stated as the measurement it is, with no
    account of what causes it.
    """
    items = f.get("asymmetry") or []
    if not items:
        return ""
    out = []
    for it in items:
        side = pipeline.REGION_LABEL.get(it.get("redder") or it.get("more"), "").lower()
        if it["measure"] == "colour":
            out.append(f'Your <strong>{html.escape(side)}</strong> reads redder than '
                       f'the other side, by {it["delta"]} points of colour.')
        else:
            hi, lo = max(it["counts"]), min(it["counts"])
            out.append(f'Your <strong>{html.escape(side)}</strong> carries more '
                       f'detected blemishes than the other side &mdash; {hi} against {lo}.')
    return ('<h3>Left and right</h3><div class="conf">'
            + "".join(f"<p>{o}</p>" for o in out) + "</div>")


def _side_by_side(*blocks):
    """Lay out whichever of these blocks have content.

    Each has to be wrapped, or the grid would put a heading in one column and
    its own content in the next. An absent block takes no column at all.
    """
    present = [b for b in blocks if b]
    if not present:
        return ""
    if len(present) == 1:
        return present[0]
    return f'<div class="split">{"".join(f"<div>{b}</div>" for b in present)}</div>'


def _patterns(f):
    """Patterns the scoring pass flagged. These are the reason a specialist is
    in the loop at all, so they get their own block rather than a JSON key."""
    flagged = [p for p in (f.get("patterns") or []) if p.get("description")]
    if not flagged:
        return ""
    items = []
    for p in flagged:
        where = _regions(p.get("regions") or [])
        mark = ' <span class="flag">flagged for review</span>' if p.get("flag_for_review") else ""
        items.append(f'<li>{html.escape(p["description"])}'
                     f'{f" &mdash; {html.escape(where)}" if where else ""}{mark}</li>')
    return f'<h3>Patterns</h3><ul class="patterns">{"".join(items)}</ul>'


def _confidence(f):
    """What could not be assessed, and why.

    Stating the limits plainly is not a caveat bolted onto the end — a report
    that silently omits what it could not see reads as complete when it isn't.
    """
    bits = []
    q = f.get("local_quality") or {}
    iq = f.get("image_quality") or {}

    if skipped := (f.get("not_measured") or []):
        bits.append(f"Not assessed: {_regions(skipped)} &mdash; turned too far from "
                    f"the camera to measure. A straight-on photo would cover these.")
    for issue in (iq.get("issues") or []):
        bits.append(html.escape(issue.replace("_", " ")))
    if (conf := iq.get("confidence")) and conf != "high":
        bits.append(f"Scoring confidence reported as {html.escape(conf)}.")
    if not bits:
        return ""
    return ('<h3>Confidence</h3><div class="conf">'
            + "".join(f"<p>{b}</p>" for b in bits) + "</div>")


def _technical(r, f):
    """One discreet line of provenance. Not report content — the thing you
    check when a report looks wrong."""
    q = f.get("local_quality") or {}
    bits = []
    if q.get("blur_score") is not None:
        bits.append(f"blur {q['blur_score']}")
    if q.get("yaw") is not None:
        bits.append(f"yaw {q['yaw']}")
    if f.get("face_width_px"):
        bits.append(f"face width {f['face_width_px']}px")
    if runs := f.get("_runs"):
        bits.append(f"{runs} scoring runs")
    if (spread := f.get("_spread")) and max(spread.values(), default=0) > 2:
        worst = max(spread, key=spread.get)
        bits.append(f"score spread {spread[worst]} on {worst} — rubric needs tightening")
    if r.get("denylist_trip"):
        bits.append(f"denylist tripped on \"{r['denylist_trip']}\", fallback text shown")
    if q.get("advisories"):
        bits.append(", ".join(q["advisories"]).replace("_", " "))
    return f'<p class="tech">{html.escape(" · ".join(bits))}</p>' if bits else ""


def _legend():
    swatches = "".join(f'<i style="background:{c}"></i>' for c in SEVERITY_RAMP)
    return (f'<div class="legend"><div class="ramp">{swatches}</div>'
            f'<div class="ramplabels"><span>minimal</span><span>significant</span></div>'
            f'<div class="mk"><b></b> detected blemish</div></div>')


def _receipts(receipts):
    """Evidence crops: each finding beside the skin it was read from."""
    if not receipts:
        return ""
    out = []
    for rc in receipts:
        zoom = f'<b>{rc["zoom"]:.1f}&times;</b>' if rc.get("zoom", 0) >= 1.2 else ""
        if rc.get("metrics"):
            claim = "".join(f'<i>{html.escape(METRIC_LABEL.get(m["metric"], m["metric"]))}'
                            f' &mdash; {html.escape(m["band"])}</i>' for m in rc["metrics"])
        else:
            claim = f'<i>{html.escape(rc["claim"])}</i>'
        out.append(f'<figure class="receipt"><div class="crop">'
                   f'<img src="data:image/jpeg;base64,{rc["image_b64"]}" alt="">{zoom}</div>'
                   f'<figcaption><strong>{html.escape(rc["label"])}</strong>'
                   f'{claim}</figcaption></figure>')
    return (f'<h3>Evidence</h3><p class="lede">Each finding, shown at the patch of '
            f'skin it was read from.</p><div class="receipts">{"".join(out)}</div>')


def _prose(title, text, cls):
    if not text:
        return ""
    paras = "".join(f"<p>{html.escape(p)}</p>" for p in text.split("\n") if p.strip())
    return f'<h3>{title}</h3><div class="prose {cls}">{paras}</div>'


def card(r):
    name = os.path.basename(r["path"])

    if r.get("rejected"):
        why = r["rejected"].replace("_", " ")
        return (f'<section class="page rejected"><header><h2>{html.escape(name)}</h2>'
                f'<span class="stamp">not analysed</span></header>'
                f'<p class="rej">This photo could not be assessed: '
                f'<strong>{html.escape(why)}</strong>. Nothing was sent to the '
                f'scoring model.</p></section>')

    f = r["findings"]
    drivers = ", ".join(f.get("skin_age_drivers") or [])

    mock = ('<div class="mock"><strong>Mock run &mdash; the model was not called.</strong> '
            'Everything under Assessment and Patterns is canned output from '
            '<code>mock_scoring.json</code> and is identical for every image in '
            'the batch. Only the measured values — blemishes, shine, pore '
            'visibility, distribution and asymmetry — were read from this '
            'photograph. Drop <code>--mock</code> for a real analysis.</div>'
            ) if f.get("_mock") else ""

    return f"""<section class="page">
<header><h2>{html.escape(name)}</h2><span class="stamp">Skin analysis &middot; {DATE}</span></header>
{mock}
<div class="top">
  <div class="figure">
    <div class="imgwrap">
      <img src="data:image/jpeg;base64,{r['image_b64']}" alt="">
      {r['overlay_svg']}
    </div>
    {_legend()}
  </div>
  <div class="summary">
    <h3>Assessment</h3>
    {_chart(f)}
    <dl class="facts">{_measured(f)}</dl>
    {f'<p class="drivers">Age band read from {html.escape(drivers)}.</p>' if drivers else ''}
  </div>
</div>
{_patterns(f)}
{_side_by_side(_distribution(f), _asymmetry(f))}
{_receipts(r.get("receipts"))}
{_prose("Your result", r.get("user_report"), "user")}
{_prose("Specialist notes", r.get("specialist_report"), "spec")}
{_confidence(f)}
{_technical(r, f)}
</section>"""


CSS = """
*{box-sizing:border-box}
body{font:13px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif;margin:0;
  padding:28px;background:#f2f1ee;color:#1a1a19}
.page{background:#faf9f7;max-width:900px;margin:0 auto 26px;padding:30px 34px 26px;
  border:1px solid #e5e3de;border-radius:10px}
header{display:flex;align-items:baseline;justify-content:space-between;
  border-bottom:1.5px solid #1a1a19;padding-bottom:9px;margin-bottom:20px}
h2{font-size:17px;margin:0;letter-spacing:-.01em}
.stamp{font-size:10.5px;text-transform:uppercase;letter-spacing:.13em;color:#7a7770}
h3{font-size:10.5px;text-transform:uppercase;letter-spacing:.1em;color:#7a7770;
  margin:22px 0 9px;font-weight:600}
h3:first-child{margin-top:0}
.lede{margin:-4px 0 10px;color:#7a7770;font-size:12px}

.top{display:grid;grid-template-columns:260px 1fr;gap:26px;align-items:start}
.imgwrap{position:relative;line-height:0}
.imgwrap img{width:100%;display:block;border-radius:6px}

.legend{margin-top:10px;font-size:10.5px;color:#7a7770;line-height:1.4}
.ramp{display:flex;gap:2px}
.ramp i{flex:1;height:7px;border-radius:1px}
.ramplabels{display:flex;justify-content:space-between;margin-top:3px}
.mk{margin-top:5px;display:flex;align-items:center;gap:6px}
.mk b{width:9px;height:9px;border-radius:50%;border:1.8px solid #d03b3b;display:inline-block}

/* Severity chart. Bar length is the encoding; every bar is one hue. */
.chart{margin:0}
.mrow{display:grid;grid-template-columns:88px 1fr 74px 1fr;gap:10px;align-items:center;
  padding:3px 0}
.mname{font-size:12px}
.track{background:#e7e5e0;border-radius:2px;height:9px}
.bar{background:#2a78d6;height:9px;border-radius:2px 4px 4px 2px}
.mband{font-size:12px;color:#52514e}
.mwhere{font-size:11.5px;color:#7a7770}
.mrow>*{min-width:0}

.facts{display:flex;flex-wrap:wrap;gap:0 30px;margin:16px 0 0}
.fact dt{font-size:10.5px;text-transform:uppercase;letter-spacing:.08em;color:#7a7770}
.fact dd{margin:1px 0 0;font-size:14px}
.drivers{color:#7a7770;font-size:11.5px;margin:12px 0 0}

.mock{background:#fdf0e3;border-left:3px solid #EF9F27;padding:9px 13px;
  border-radius:0 5px 5px 0;margin:0 0 18px;font-size:12px;line-height:1.5}
.mock code{font:11px ui-monospace,Menlo,Consolas,monospace;background:#f6e6d0;
  padding:0 4px;border-radius:3px}

.split{display:grid;grid-template-columns:1fr 1fr;gap:0 30px;align-items:start}
.split>div>h3{margin-top:22px}
.dist{margin:0;max-width:440px}
.drow{display:grid;grid-template-columns:132px 1fr 26px;gap:9px;align-items:center;
  padding:2px 0}
.drow>*{min-width:0}
.dname{font-size:12px;white-space:nowrap}
.dname em{font-style:normal;font-size:9.5px;color:#a5a29b;background:#eceae5;
  border-radius:8px;padding:1px 5px;margin-left:5px}
.dnum{font-size:12px;color:#52514e;text-align:right}
/* A count from a partly hidden region is a floor, not a total. */
.bar.partial{background:repeating-linear-gradient(135deg,#2a78d6 0 5px,#7aa9e0 5px 10px)}

.patterns{margin:0;padding-left:17px}
.patterns li{margin-bottom:3px}
.flag{background:#fdf0e3;color:#8a5416;font-size:10px;text-transform:uppercase;
  letter-spacing:.07em;padding:1px 6px;border-radius:9px;white-space:nowrap}

.receipts{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
.receipt{margin:0;min-width:0;border:1px solid #e5e3de;border-radius:7px;
  overflow:hidden;background:#fff}
.crop{position:relative;line-height:0}
.crop img{width:100%;display:block}
.crop b{position:absolute;right:5px;bottom:5px;background:rgba(26,26,25,.72);color:#fff;
  font:9.5px/1.5 system-ui;padding:1px 6px;border-radius:9px}
figcaption{padding:7px 9px}
figcaption strong{display:block;font-size:11.5px;margin-bottom:2px}
figcaption i{font-style:normal;display:block;font-size:10.5px;color:#7a7770;
  overflow-wrap:anywhere}

.prose{font:14px/1.65 Georgia,"Times New Roman",serif;background:#fff;
  border:1px solid #e5e3de;border-left:2.5px solid #2a78d6;
  border-radius:0 7px 7px 0;padding:14px 18px}
.prose.spec{border-left-color:#7a7770;font-family:system-ui,sans-serif;font-size:12.5px}
.prose p{margin:0 0 9px}.prose p:last-child{margin:0}

.conf p{margin:0 0 5px;color:#52514e}
.tech{margin:20px 0 0;padding-top:9px;border-top:1px solid #eceae5;
  font-size:10px;color:#a5a29b;letter-spacing:.01em}

.rejected header{border-bottom-color:#A32D2D}
.rej{margin:0;color:#52514e}

.toggles{display:flex;gap:6px;margin-bottom:9px}
.toggles button{flex:1;font:11px system-ui;padding:4px;border:1px solid #d5d2cc;
  border-radius:5px;background:#fff;cursor:pointer}
.toggles button[data-on="0"]{opacity:.4}

@media print{
  @page{size:A4;margin:12mm}
  body{padding:0;background:#fff}
  .page{max-width:none;margin:0;border:0;border-radius:0;padding:0;
    break-after:page;background:#fff}
  .page:last-child{break-after:auto}
  .toggles{display:none}
  .top,.receipts,.prose,.receipt,.patterns{break-inside:avoid-page}
  /* A heading whose block breaks to the next page must go with it. Otherwise
     "Your result" sits alone at the foot of a page and the report reads as
     though the section is empty. */
  h3{break-after:avoid-page}
}
"""

JS = """
document.querySelectorAll('.toggles button').forEach(b=>{
  b.onclick=()=>{
    const on=b.dataset.on!=='0'; b.dataset.on=on?'0':'1';
    b.closest('.page').querySelector(`[data-layer="${b.dataset.t}"]`)
      .style.opacity=on?'0':'1';
  };
});
"""


def document(results):
    return ("<!doctype html><meta charset=utf-8>"
            "<meta name=viewport content='width=device-width,initial-scale=1'>"
            "<title>Skin analysis report</title>"
            f"<style>{CSS}</style>"
            + "".join(card(r) for r in results)
            + f"<script>{JS}</script>")

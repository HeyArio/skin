"""Gateway client.

The ArvanCloud endpoint ends in /v1, which usually means OpenAI-compatible,
but gateways vary and some only implement the text path. `probe.py` tells you
which shape yours accepts. Set GATEWAY_STYLE accordingly.
"""
import base64
import json
import os
import re
import time

import requests

import config  # importing it loads .env

VISION_URL = config.require("VISION_URL").rstrip("/")
VISION_KEY = os.environ.get("VISION_KEY", "")
TEXT_URL = os.environ.get("TEXT_URL", VISION_URL).rstrip("/")
TEXT_KEY = os.environ.get("TEXT_KEY", VISION_KEY)
STYLE = os.environ.get("GATEWAY_STYLE", "openai")  # openai | gemini
VISION_MODEL = os.environ.get("VISION_MODEL", "Gemini-3.1-Flash-Lite-Preview")
# Defaults to the vision model rather than to a second one. Everything runs on
# one model until there is a reason not to, and a default naming a model you
# are not using is a config bug waiting to be debugged.
TEXT_MODEL = os.environ.get("TEXT_MODEL", VISION_MODEL)

TIMEOUT = 90


def _headers(key):
    h = {"Content-Type": "application/json"}
    if key:
        h["Authorization"] = f"Bearer {key}"
    return h


def _post(url, payload, key, attempts=3):
    last = None
    for i in range(attempts):
        try:
            r = requests.post(url, headers=_headers(key), json=payload, timeout=TIMEOUT)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (401, 403):
                raise config.AuthError(config.AUTH_HELP.format(code=r.status_code))
            last = f"HTTP {r.status_code}: {r.text[:400]}"
            if r.status_code < 500:
                break
        except requests.RequestException as e:
            # Never let the URL into an error string — the credential is in it,
            # and this is exactly how it ends up in logs and pasted tracebacks.
            last = str(e).replace(url, "<gateway url>")
        time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"gateway call failed — {last}")


def _extract_text(data):
    """Pull the text out of either response shape."""
    if "choices" in data:
        return data["choices"][0]["message"]["content"]
    if "candidates" in data:
        parts = data["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts)
    raise RuntimeError(f"unrecognised response shape: {list(data)[:5]}")


def _parse_json(text):
    """Models sometimes wrap JSON in fences despite being told not to."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, re.S)
        if not m:
            raise
        return json.loads(m.group(0))


# ------------------------------------------------------------- request shapes

def _url(base, model):
    return (f"{base}/models/{model}:generateContent" if STYLE == "gemini"
            else f"{base}/chat/completions")


def _payload(model, system, parts, temperature, json_out=False):
    """Build a request in whichever shape the gateway speaks.

    `parts` is a list of ("text", str) or ("image", (mime, b64)). Both calls
    build through here so the text calls cannot quietly stay OpenAI-shaped when
    GATEWAY_STYLE says gemini — which is exactly what they used to do, leaving
    the reports broken on a gateway where scoring worked fine.
    """
    if STYLE == "gemini":
        cfg = {"temperature": temperature}
        if json_out:
            cfg["responseMimeType"] = "application/json"
        return {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [
                {"text": v} if k == "text"
                else {"inline_data": {"mime_type": v[0], "data": v[1]}}
                for k, v in parts
            ]}],
            "generationConfig": cfg,
        }

    if all(k == "text" for k, _ in parts):
        # Plain string, not a one-element list. Some gateways only accept the
        # simple form for text-only messages.
        content = "\n\n".join(v for _, v in parts)
    else:
        content = [
            {"type": "text", "text": v} if k == "text"
            else {"type": "image_url",
                  "image_url": {"url": f"data:{v[0]};base64,{v[1]}"}}
            for k, v in parts
        ]
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": content}],
    }
    if json_out:
        payload["response_format"] = {"type": "json_object"}
    return payload


# ------------------------------------------------------------- call 1 (vision)

def score_image(image_bytes, system, user, mime="image/jpeg", temperature=0.0):
    b64 = base64.b64encode(image_bytes).decode()
    payload = _payload(VISION_MODEL, system,
                       [("image", (mime, b64)), ("text", user)],
                       temperature, json_out=True)
    return _parse_json(_extract_text(
        _post(_url(VISION_URL, VISION_MODEL), payload, VISION_KEY)))


def score_image_median(image_bytes, system, user, n=3, **kw):
    """Median-of-N. At Flash-Lite prices this is cheap insurance against drift.
    Set SCORING_RUNS=1 while iterating on the prompt to save time."""
    runs = [score_image(image_bytes, system, user, **kw) for _ in range(n)]
    if n == 1:
        return runs[0]

    merged = json.loads(json.dumps(runs[0]))
    for metric in merged.get("metrics", {}):
        vals = [r["metrics"][metric]["score"] for r in runs
                if r.get("metrics", {}).get(metric, {}).get("score") is not None]
        merged["metrics"][metric]["score"] = (
            int(sorted(vals)[len(vals) // 2]) if vals else None
        )
        # Keep a region only if a majority of runs flagged it.
        tally = {}
        for r in runs:
            for reg in r.get("metrics", {}).get(metric, {}).get("regions") or []:
                tally[reg] = tally.get(reg, 0) + 1
        merged["metrics"][metric]["regions"] = [
            reg for reg, c in tally.items() if c > n / 2
        ]
    merged["_runs"] = n
    merged["_spread"] = {
        m: max(v) - min(v) for m in merged.get("metrics", {})
        if (v := [r["metrics"][m]["score"] for r in runs
                  if r.get("metrics", {}).get(m, {}).get("score") is not None])
    }
    return merged


# -------------------------------------------------------- calls 2 & 3 (text)

def write_report(system, findings, temperature=0.3):
    payload = _payload(TEXT_MODEL, system,
                       [("text", json.dumps(findings, ensure_ascii=False))],
                       temperature)
    return _extract_text(
        _post(_url(TEXT_URL, TEXT_MODEL), payload, TEXT_KEY)).strip()

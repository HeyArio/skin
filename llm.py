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

VISION_URL = os.environ["VISION_URL"].rstrip("/")
VISION_KEY = os.environ.get("VISION_KEY", "")
TEXT_URL = os.environ.get("TEXT_URL", VISION_URL).rstrip("/")
TEXT_KEY = os.environ.get("TEXT_KEY", VISION_KEY)
STYLE = os.environ.get("GATEWAY_STYLE", "openai")  # openai | gemini
TEXT_MODEL = os.environ.get("TEXT_MODEL", "gpt-oss-120b")

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
            last = f"HTTP {r.status_code}: {r.text[:400]}"
            if r.status_code < 500:
                break
        except requests.RequestException as e:
            last = str(e)
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


# ------------------------------------------------------------- call 1 (vision)

def score_image(image_bytes, system, user, mime="image/jpeg", temperature=0.0):
    b64 = base64.b64encode(image_bytes).decode()

    if STYLE == "gemini":
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [
                {"inline_data": {"mime_type": mime, "data": b64}},
                {"text": user},
            ]}],
            "generationConfig": {"temperature": temperature,
                                 "responseMimeType": "application/json"},
        }
        url = f"{VISION_URL}/models/gemini:generateContent"
    else:
        payload = {
            "model": os.environ.get("VISION_MODEL", "Gemini-3.1-Flash-Lite-Preview"),
            "temperature": temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text", "text": user},
                ]},
            ],
        }
        url = f"{VISION_URL}/chat/completions"

    return _parse_json(_extract_text(_post(url, payload, VISION_KEY)))


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
    payload = {
        "model": TEXT_MODEL,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(findings, ensure_ascii=False)},
        ],
    }
    return _extract_text(_post(f"{TEXT_URL}/chat/completions", payload, TEXT_KEY)).strip()

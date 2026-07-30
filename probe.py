#!/usr/bin/env python3
"""Run this FIRST, before building anything on the endpoint.

    python probe.py

Sends a tiny generated image four ways and reports which the gateway accepts.
Gateways that proxy Gemini sometimes implement only the text path, or want a
different image encoding than the native API. Five minutes here saves a day.
"""
import base64
import json
import os

import numpy as np
import cv2
import requests

URL = os.environ["VISION_URL"].rstrip("/")
KEY = os.environ.get("VISION_KEY", "")
MODEL = os.environ.get("VISION_MODEL", "Gemini-3.1-Flash-Lite-Preview")

# A 64x64 image: red square on the left, blue on the right. If the model can
# see, it will say so. If the gateway silently drops images, it will not.
img = np.zeros((64, 64, 3), np.uint8)
img[:, :32] = (0, 0, 255)
img[:, 32:] = (255, 0, 0)
B64 = base64.b64encode(cv2.imencode(".png", img)[1]).decode()
ASK = "Name the two colours in this image, left then right. Two words only."

HEAD = {"Content-Type": "application/json"}
if KEY:
    HEAD["Authorization"] = f"Bearer {KEY}"

CASES = {
    "openai / text only": (f"{URL}/chat/completions", {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
    }),
    "openai / image_url data-uri": (f"{URL}/chat/completions", {
        "model": MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{B64}"}},
            {"type": "text", "text": ASK}]}],
    }),
    # The model name belongs in the path here — probing a literal "gemini"
    # fails on every gateway, which reads as "this gateway has no native path"
    # and sends you to the wrong GATEWAY_STYLE.
    "gemini native / inline_data": (f"{URL}/models/{MODEL}:generateContent", {
        "contents": [{"role": "user", "parts": [
            {"inline_data": {"mime_type": "image/png", "data": B64}},
            {"text": ASK}]}],
    }),
    "openai / json_object mode": (f"{URL}/chat/completions", {
        "model": MODEL,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": 'Return {"ok":true} and nothing else.'}],
    }),
}


def text_of(d):
    if "choices" in d:
        return d["choices"][0]["message"]["content"]
    if "candidates" in d:
        return "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"])
    return json.dumps(d)[:200]


for name, (url, payload) in CASES.items():
    try:
        r = requests.post(url, headers=HEAD, json=payload, timeout=60)
        if r.status_code == 200:
            print(f"  PASS  {name}\n        -> {text_of(r.json())[:110].strip()}")
        else:
            print(f"  FAIL  {name}  [HTTP {r.status_code}] {r.text[:110]}")
    except Exception as e:
        print(f"  FAIL  {name}  {e}")

print("""
Read the results:
  - "text only" passes, image cases fail  -> gateway drops images. Blocker.
  - an image case passes but names the wrong colours -> image arrives corrupted.
  - "gemini native" passes -> set GATEWAY_STYLE=gemini
  - "json_object mode" fails -> remove response_format from llm.py and rely on
    the fence-stripping fallback in _parse_json.""")

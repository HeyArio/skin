#!/usr/bin/env python3
"""Run this FIRST, before building anything on the endpoint.

    python probe.py

Answers three questions in order, because each one is only worth asking if the
previous answered yes:

  1. Is the endpoint there at all?
  2. How does it want to be authenticated?
  3. Which request shapes does it accept, and do images survive the trip?

The auth step exists because "the key is right" and "the key is being sent the
way this gateway wants it" are different statements, and a 401 cannot tell them
apart. Gateways variously want a Bearer header, a bare header, a vendor-specific
header name, a query parameter, or the token baked into the URL path — and the
rejection looks identical for all of them.
"""
import base64
import json
import os

import numpy as np
import cv2
import requests

import config  # importing it loads .env

URL = config.require("VISION_URL").rstrip("/")
KEY = os.environ.get("VISION_KEY", "")
MODEL = os.environ.get("VISION_MODEL", "Gemini-3.1-Flash-Lite-Preview")
TIMEOUT = 45

# A 64x64 image: red square on the left, blue on the right. If the model can
# see, it will say so. If the gateway silently drops images, it will not.
img = np.zeros((64, 64, 3), np.uint8)
img[:, :32] = (0, 0, 255)
img[:, 32:] = (255, 0, 0)
B64 = base64.b64encode(cv2.imencode(".png", img)[1]).decode()
ASK = "Name the two colours in this image, left then right. Two words only."

PING = {"model": MODEL,
        "messages": [{"role": "user", "content": "Reply with the single word: ok"}]}


def mask(s, keep=3):
    """Never print a credential. Length and a few characters are enough to tell
    'not set' from 'set to the wrong thing' from 'set to something plausible'."""
    if not s:
        return "(empty)"
    return f"{s[:keep]}…{s[-keep:]} ({len(s)} chars)" if len(s) > keep * 2 else "(short)"


def endpoint_summary():
    """Show what is about to be used, with the secret parts masked.

    On a path-token gateway the URL itself is the credential, so it cannot be
    printed — but whether it *looks* like it carries one is the single most
    useful thing to see when a key from a dashboard is being pasted into the
    wrong slot.
    """
    from urllib.parse import urlsplit
    u = urlsplit(URL)
    segments = [s for s in u.path.split("/") if s]
    long_segments = [s for s in segments if len(s) > 40]

    print("Config")
    print(f"  host          {u.scheme}://{u.netloc}")
    print(f"  path          /{'/'.join(s if len(s) <= 40 else mask(s) for s in segments)}")
    print(f"  model         {MODEL}")
    print(f"  VISION_KEY    {mask(KEY)}")
    print(f"  token in URL  {'yes — path carries a long opaque segment' if long_segments else 'no'}")
    if not long_segments and not KEY:
        print("\n  ! No credential anywhere: the URL carries no token and VISION_KEY")
        print("    is empty. Every call will 401. Put your dashboard key in")
        print("    VISION_KEY, or use the full URL your gateway gave you.")
    print()


# How the credential might need to be presented, and what to put in .env when
# one of them works. The URL is unchanged for all but the query-parameter cases.
#
# The scheme word matters as much as the key. A gateway expecting "apikey" and
# handed "Bearer" answers 401 — identical to the response for a wrong key, which
# is why "my key is definitely right" and a 401 are not a contradiction.
def auth_variants():
    v = [("no credential sent", {}, {}, "the URL alone carries the credential")]
    if not KEY:
        return v
    for scheme in ("Bearer", "apikey", "Token", "Key"):
        v.append((f"Authorization: {scheme} <key>",
                  {"Authorization": f"{scheme} {KEY}"}, {},
                  f'VISION_KEY="<your key>" and AUTH_SCHEME="{scheme}"'))
    v += [
        ("Authorization: <key>, no scheme word", {"Authorization": KEY}, {},
         'VISION_KEY="<your key>" and AUTH_SCHEME="" (empty)'),
        ("api-key: <key>", {"api-key": KEY}, {},
         "a header name llm._headers does not send — see the note below"),
        ("x-api-key: <key>", {"x-api-key": KEY}, {},
         "a header name llm._headers does not send — see the note below"),
        ("?api_key=<key>", {}, {"api_key": KEY},
         "a query parameter llm._post does not send — see the note below"),
        ("?key=<key>", {}, {"key": KEY},
         "a query parameter llm._post does not send — see the note below"),
    ]
    return v


def post(url, payload, headers=None, params=None):
    h = {"Content-Type": "application/json", **(headers or {})}
    try:
        return requests.post(url, headers=h, params=params, json=payload, timeout=TIMEOUT)
    except requests.RequestException as e:
        return str(e).replace(URL, "<gateway url>")


def text_of(d):
    if "choices" in d:
        return d["choices"][0]["message"]["content"]
    if "candidates" in d:
        return "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"])
    return json.dumps(d)[:200]


def show(label, r):
    if isinstance(r, str):
        print(f"  FAIL  {label}  {r[:110]}")
        return False
    if r.status_code == 200:
        print(f"  PASS  {label}\n        -> {text_of(r.json())[:110].strip()}")
        return True
    print(f"  FAIL  {label}  [HTTP {r.status_code}] {r.text[:110].strip()}")
    return False


def find_auth():
    """Try each presentation of the credential against the simplest possible
    call. Returns the first that works, and what to put in .env for it."""
    print("Authentication — simplest text call, one variant at a time")
    for label, headers, params, advice in auth_variants():
        r = post(f"{URL}/chat/completions", PING, headers, params)
        if show(label, r):
            print(f"\n  -> in .env: {advice}\n")
            return headers, params
    print()
    return None


def probe_shapes(headers, params):
    cases = {
        "openai / text only": (f"{URL}/chat/completions", PING),
        "openai / image_url data-uri": (f"{URL}/chat/completions", {
            "model": MODEL,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{B64}"}},
                {"type": "text", "text": ASK}]}],
        }),
        # The model name belongs in the path here — probing a literal "gemini"
        # fails on every gateway and reads as "no native path".
        "gemini native / inline_data": (f"{URL}/models/{MODEL}:generateContent", {
            "contents": [{"role": "user", "parts": [
                {"inline_data": {"mime_type": "image/png", "data": B64}},
                {"text": ASK}]}],
        }),
        "openai / json_object mode": (f"{URL}/chat/completions", {
            "model": MODEL,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user",
                          "content": 'Return {"ok":true} and nothing else.'}],
        }),
    }
    print("Request shapes — using the authentication that worked")
    passed = {}
    for label, (url, payload) in cases.items():
        passed[label] = show(label, post(url, payload, headers, params))
    return passed


def main():
    endpoint_summary()

    auth = find_auth()
    if auth is None:
        print("Every authentication variant was rejected.\n")
        print("The endpoint is reachable — it answered — so this is the credential,")
        print("not the URL or the request shape. Check, in this order:\n")
        print("  1. WHICH credential. A key from an API-key dashboard goes in")
        print("     VISION_KEY. A gateway URL with a long opaque path segment")
        print("     already contains one, and the two are not interchangeable.")
        print("     Setting the wrong one leaves you with no valid credential.")
        print("  2. Whether the key is still live, and scoped to this model.")
        print("  3. Whether your account has access to VISION_MODEL as spelled")
        print("     above — some gateways return 401 rather than 404 for a model")
        print("     you are not entitled to.")
        print("  4. A working `curl` for this gateway, if you have one. The word")
        print("     before the token in its Authorization header is the value of")
        print("     AUTH_SCHEME, and it is not always Bearer.")
        raise SystemExit(1)

    passed = probe_shapes(*auth)
    print()

    if passed["gemini native / inline_data"]:
        print("  -> set GATEWAY_STYLE=gemini")
    elif passed["openai / image_url data-uri"]:
        print("  -> set GATEWAY_STYLE=openai")
        print("     Check the colours above read red then blue. Anything else and")
        print("     the image is arriving corrupted, which is worse than it failing.")
    elif passed["openai / text only"]:
        print("  -> BLOCKER: text works, images do not. This gateway proxies only")
        print("     the text path, so Call 1 cannot run through it. You need a")
        print("     different endpoint for the vision call.")

    if passed["openai / text only"] and not passed["openai / json_object mode"]:
        print("  -> no json_object mode: remove response_format from llm._payload")
        print("     and rely on the fence-stripping fallback in _parse_json.")

    print("\n  Note: llm.py authenticates with an Authorization header, whose")
    print("  scheme word comes from AUTH_SCHEME. If the variant that worked above")
    print("  was a different header name or a query parameter, llm._headers needs")
    print("  a small edit to match — the probe can find it, but cannot apply it.")


if __name__ == "__main__":
    main()

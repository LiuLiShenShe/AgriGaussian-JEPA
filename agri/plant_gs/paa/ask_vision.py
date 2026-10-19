#!/usr/bin/env python
"""Ask the Agnes vision model to describe a local image (see ~/.claude/CLAUDE.md).

Kept as a file rather than a heredoc so the API key is never mangled by shell
quoting. This project has no built-in vision, so any claim about what an image
contains has to go through here rather than be guessed.

    python agri/plant_gs/paa/ask_vision.py IMAGE.png "your question"
"""
import base64
import json
import os
import sys
import urllib.request

# The key is deliberately NOT defaulted here: this file is committed, and a
# credential in a repo outlives the machine it was written on. Export it
# instead:  AGNES_API_KEY=sk-... python agri/plant_gs/paa/ask_vision.py IMG "..."
KEY = os.environ.get("AGNES_API_KEY")
if not KEY:
    raise SystemExit("AGNES_API_KEY is not set; refusing to guess a credential")
ENDPOINTS = ["https://apihub.agnes-ai.cn/v1/chat/completions",
             "https://apihub.agnes-ai.com/v1/chat/completions"]


def ask(image_path: str, prompt: str, max_tokens: int = 1200) -> str:
    with open(image_path, "rb") as handle:
        data = "data:image/png;base64," + base64.b64encode(handle.read()).decode()
    body = {
        "model": "agnes-3.0-flash",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data}},
        ]}],
        "max_tokens": max_tokens,
    }
    payload = json.dumps(body).encode()
    last = None
    for endpoint in ENDPOINTS:
        try:
            request = urllib.request.Request(
                endpoint, data=payload,
                headers={"Authorization": "Bearer " + KEY,
                         "Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=300) as response:
                return json.load(response)["choices"][0]["message"]["content"]
        except Exception as exc:                                   # noqa: BLE001
            last = f"{endpoint}: {type(exc).__name__}: {exc}"
    raise SystemExit(f"vision API unavailable: {last}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    print(ask(sys.argv[1], sys.argv[2]))

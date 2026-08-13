#!/usr/bin/env python3
"""Local Gradio client for the tunneled ME344 vLLM server."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import gradio as gr
import requests

try:
    from final_project import BANKING77_INTENTS
except ModuleNotFoundError:
    from final_project import BANKING77_INTENTS


ROUTING_PREFIX = (
    "Route this retail-bank support request. Return exactly one label and no explanation.\n"
    f"Allowed labels: {' | '.join(BANKING77_INTENTS)}\n\nRequest: "
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create a temporary authenticated public link.")
    return parser.parse_args()


def prior_messages(history) -> list[dict[str, str]]:
    messages = []
    for item in history:
        if isinstance(item, dict) and item.get("role") in {"user", "assistant"}:
            messages.append({"role": item["role"], "content": str(item.get("content", ""))})
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            messages.extend(
                [
                    {"role": "user", "content": str(item[0])},
                    {"role": "assistant", "content": str(item[1])},
                ]
            )
    return messages


def main() -> None:
    args = parse_args()
    health = requests.get(f"{args.base_url}/health", timeout=5)
    health.raise_for_status()

    def respond(message, history, mode, temperature, max_tokens):
        routing = mode == "Intent router"
        payload = {
            "model": args.model,
            "messages": (
                [{"role": "user", "content": ROUTING_PREFIX + message}]
                if routing
                else [*prior_messages(history), {"role": "user", "content": message}]
            ),
            "temperature": 0 if routing else temperature,
            "max_tokens": 24 if routing else int(max_tokens),
            "stream": True,
        }
        response = requests.post(
            f"{args.base_url}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=(10, 300),
        )
        response.raise_for_status()
        answer = ""
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            choices = event.get("choices") or []
            answer += choices[0].get("delta", {}).get("content", "") if choices else ""
            yield answer

    mode = gr.Radio(["Intent router", "Chat"], value="Intent router", label="Mode")
    temperature = gr.Slider(0, 1, value=0.2, step=0.05, label="Temperature")
    max_tokens = gr.Slider(16, 256, value=128, step=16, label="Maximum output tokens")
    demo = gr.ChatInterface(
        fn=respond,
        title=f"ME344: {args.model}",
        additional_inputs=[mode, temperature, max_tokens],
        examples=[
            ["My cash withdrawal was declined.", "Intent router", 0.2, 128],
            ["Why has my transfer not reached them?", "Intent router", 0.2, 128],
            ["Explain tensor parallelism.", "Chat", 0.2, 128],
        ],
        analytics_enabled=False,
    )
    auth = None
    if args.share:
        password = os.environ.get("ME344_CHAT_PASSWORD")
        if not password:
            raise SystemExit("Set ME344_CHAT_PASSWORD before using --share.")
        auth = ("me344", password)
        share_cache = Path.home() / ".cache" / "me344" / "gradio"
        share_cache.mkdir(parents=True, exist_ok=True)
        os.chdir(share_cache)
    demo.queue(default_concurrency_limit=4, max_size=16).launch(
        server_name="127.0.0.1",
        server_port=args.port,
        show_error=True,
        share=args.share,
        auth=auth,
    )


if __name__ == "__main__":
    main()

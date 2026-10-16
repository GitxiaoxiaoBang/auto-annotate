#!/usr/bin/env python3
"""Qwen3.8-27B (BF16) captioner for referring expressions.

Run under jevcod env. Two modes:
  1) one-shot:  --request req.json
  2) serve:     --serve   (JSON-lines over stdin/stdout, model stays loaded)

Request JSON:
  {
    "image": "/path/to/rgb.jpg",
    "overlay": "/path/to/overlay_with_green_mask.jpg",
    "instruction": "user text",
    "model": "...",
    "max_new_tokens": 256,
    "enable_thinking": false
  }
Prints one JSON line: {"ok": true, "text": "..."}
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

DEFAULT_MODEL = "/home/zyserver/LLB/RS2-SAM2/your_weights/Qwen3.8-27B"
DEFAULT_INSTRUCTION = (
    "The target region is highlighted with a GREEN translucent mask "
    "(and may have a green outline). "
    "Write ONE short English caption / referring expression that uniquely describes "
    "the highlighted target using appearance, location, or relative position. "
    "Output ONLY the English sentence, no quotes, no explanation."
)


def clean_phrase(text: str) -> str:
    t = (text or "").strip()
    # Drop thinking blocks if present (Qwen3.x)
    t = re.sub(r"<think>[\s\S]*?</think>", "", t, flags=re.I)
    t = re.sub(r"</?think>", "", t, flags=re.I)
    t = t.strip().strip('"').strip("'")
    # Keep multi-line referring phrases (3-5 items); collapse blank lines
    lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
    t = "\n".join(lines) if len(lines) > 1 else (lines[0] if lines else "")
    t = re.sub(
        r"^(referring expression|caption|answer|output|描述)\s*[:：-]\s*",
        "",
        t,
        flags=re.I,
    )
    return t.strip()


class Captioner:
    def __init__(self, model_id: str):
        self.model_id = model_id
        # device_map=auto spreads BF16 ~52GB across visible GPUs (use 2x 32GB)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model.eval()

    def generate(self, req: dict) -> dict:
        image_path = req.get("overlay") or req.get("image")
        if not image_path or not Path(image_path).exists():
            return {"ok": False, "error": f"image not found: {image_path}"}

        instruction = (req.get("instruction") or "").strip() or DEFAULT_INSTRUCTION
        max_new_tokens = int(req.get("max_new_tokens") or 256)
        enable_thinking = bool(req.get("enable_thinking", False))

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        # device_map=auto 时取首层参数所在卡
        first_device = next(self.model.parameters()).device
        inputs = inputs.to(first_device)

        with torch.inference_mode():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        trimmed = [o[len(i) :] for i, o in zip(inputs.input_ids, out_ids)]
        answer = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        phrase = clean_phrase(answer)
        if not phrase:
            return {"ok": False, "error": "empty generation", "raw": answer}
        return {"ok": True, "text": phrase}


def _emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def serve(model_id: str) -> None:
    try:
        cap = Captioner(model_id)
    except Exception as e:
        _emit({"ok": False, "error": f"load failed: {e}", "traceback": traceback.format_exc()[-2000:]})
        sys.exit(2)
    _emit({"ok": True, "ready": True, "model": model_id})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line in {"quit", "exit"}:
            _emit({"ok": True, "bye": True})
            break
        try:
            req = json.loads(line)
            if req.get("ping"):
                _emit({"ok": True, "pong": True})
                continue
            _emit(cap.generate(req))
        except Exception as e:
            _emit({"ok": False, "error": str(e), "traceback": traceback.format_exc()[-1500:]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", default="", help="path to JSON request file; else read stdin")
    ap.add_argument("--serve", action="store_true", help="JSON-lines server; keep model loaded")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    if args.serve:
        serve(args.model)
        return

    if args.request:
        req = json.loads(Path(args.request).read_text(encoding="utf-8"))
    else:
        req = json.loads(sys.stdin.read())

    model_id = req.get("model") or args.model
    try:
        cap = Captioner(model_id)
        _emit(cap.generate(req))
    except Exception as e:
        _emit({"ok": False, "error": str(e), "traceback": traceback.format_exc()[-2000:]})
        sys.exit(2)


if __name__ == "__main__":
    main()

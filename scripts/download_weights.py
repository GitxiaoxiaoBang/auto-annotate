#!/usr/bin/env python3
"""Download Grounding DINO + SAM weights via HuggingFace (supports hf-mirror)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Download Grounded-SAM pretrained weights")
    p.add_argument(
        "--dino",
        default="IDEA-Research/grounding-dino-base",
        help="HF repo id for Grounding DINO",
    )
    p.add_argument(
        "--sam",
        default="facebook/sam-vit-large",
        help="HF repo id for SAM (sam-vit-base / large / huge)",
    )
    p.add_argument(
        "--model_dir",
        default="/home/zyserver/LLB/auto-annotate/models",
        help="Local directory to store models",
    )
    p.add_argument(
        "--hf_endpoint",
        default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"),
        help="HuggingFace endpoint (default: hf-mirror for China network)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["HF_ENDPOINT"] = args.hf_endpoint
    # huggingface_hub picks this up for downloads
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

    from huggingface_hub import snapshot_download

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    targets = [
        ("dino", args.dino),
        ("sam", args.sam),
    ]
    for tag, repo_id in targets:
        local = model_dir / repo_id.replace("/", "__")
        print(f"[{tag}] downloading {repo_id} -> {local}")
        path = snapshot_download(
            repo_id=repo_id,
            local_dir=str(local),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"[{tag}] ready: {path}")

    print("\nDone. Point configs/default.yaml model_dir to:", model_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Grounded-SAM batch auto-annotation: text prompt -> boxes -> pixel masks.

Pipeline:
  1) Grounding DINO (open-vocab detector) finds boxes from TEXT_PROMPT
  2) SAM (Segment Anything) refines each box into a binary mask
  3) Export combined PNG masks (+ optional overlays / COCO / meta JSON)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    SamModel,
    SamProcessor,
)


IMG_EXTS_DEFAULT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(description="Grounded-SAM batch auto annotation")
    p.add_argument("--config", default="/home/zyserver/LLB/auto-annotate/configs/default.yaml")
    p.add_argument("--input_dir", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--text_prompt", default=None)
    p.add_argument("--box_threshold", type=float, default=None)
    p.add_argument("--text_threshold", type=float, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--dino_model", default=None)
    p.add_argument("--sam_model", default=None)
    p.add_argument("--model_dir", default=None)
    p.add_argument("--max_images", type=int, default=-1, help="-1 = all")
    p.add_argument(
        "--hf_endpoint",
        default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"),
    )
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_model_path(model_id: str, model_dir: str | None) -> str:
    """Prefer local snapshot under model_dir; otherwise use HF repo id."""
    if model_dir:
        local = Path(model_dir) / model_id.replace("/", "__")
        if local.exists() and any(local.iterdir()):
            return str(local)
    return model_id


def list_images(input_dir: Path, exts: set[str]) -> list[Path]:
    files = []
    for p in sorted(input_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return files


def draw_overlay(image_rgb: np.ndarray, mask: np.ndarray, boxes_xyxy, phrases) -> np.ndarray:
    out = image_rgb.copy()
    color = np.array([0, 255, 128], dtype=np.float32)
    m = mask.astype(bool)
    if m.any():
        out = out.astype(np.float32)
        out[m] = out[m] * 0.55 + color * 0.45
        out = out.astype(np.uint8)
    for box, phrase in zip(boxes_xyxy, phrases):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 64, 64), 2)
        cv2.putText(
            out,
            phrase[:40],
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 64, 64),
            1,
            cv2.LINE_AA,
        )
    return out


def masks_to_coco(images_meta: list[dict], out_json: Path):
    """Minimal COCO instances export (binary category=1)."""
    coco = {
        "info": {"description": "Grounded-SAM auto labels", "version": "1.0"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "object", "supercategory": "object"}],
    }
    ann_id = 1
    for img in images_meta:
        coco["images"].append(
            {
                "id": img["image_id"],
                "file_name": img["file_name"],
                "width": img["width"],
                "height": img["height"],
            }
        )
        for inst in img["instances"]:
            x1, y1, x2, y2 = inst["bbox_xyxy"]
            w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": img["image_id"],
                    "category_id": 1,
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "area": float(inst["area"]),
                    "iscrowd": 0,
                    "segmentation": {
                        "size": [img["height"], img["width"]],
                        "counts": inst.get("rle", ""),
                    },
                    "score": float(inst.get("score", 0.0)),
                    "phrase": inst.get("phrase", ""),
                }
            )
            ann_id += 1
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(coco, f)
    print(f"COCO saved: {out_json}")


def encode_rle(mask: np.ndarray) -> str:
    """Simple uncompressed RLE as space-separated run lengths (for meta only)."""
    flat = mask.astype(np.uint8).ravel(order="F")
    # COCO-style counts string without pycocotools dependency
    counts = []
    prev = 0
    run = 0
    for v in flat:
        if v == prev:
            run += 1
        else:
            counts.append(str(run))
            run = 1
            prev = int(v)
    counts.append(str(run))
    # if first pixel is 1, COCO expects leading 0
    if mask.T.ravel()[0] == 1:  # Fortran order first pixel
        counts = ["0"] + counts
    return " ".join(counts)


@torch.inference_mode()
def detect_boxes(dino_model, dino_processor, image: Image.Image, prompt: str, box_thr: float, text_thr: float, device: str):
    # Grounding DINO expects lowercase + trailing dot style prompts
    caption = prompt.lower().strip()
    if not caption.endswith("."):
        caption = caption + "."

    inputs = dino_processor(images=image, text=caption, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = dino_model(**inputs)

    target_sizes = [image.size[::-1]]  # (h, w)
    # transformers API differs slightly across versions
    try:
        results = dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=box_thr,
            text_threshold=text_thr,
            target_sizes=target_sizes,
        )[0]
    except TypeError:
        results = dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=box_thr,
            text_threshold=text_thr,
            target_sizes=target_sizes,
        )[0]

    boxes = results["boxes"].detach().cpu()
    scores = results["scores"].detach().cpu()
    labels = results.get("labels", results.get("text_labels", []))
    if hasattr(labels, "tolist"):
        phrases = [str(x) for x in labels.tolist()]
    else:
        phrases = [str(x) for x in labels]
    return boxes, scores, phrases


def filter_boxes(boxes, scores, phrases, img_w: int, img_h: int, max_area_ratio: float):
    """Drop near-full-image boxes that often appear with vague open-vocab prompts."""
    if boxes is None or len(boxes) == 0:
        return boxes, scores, phrases
    keep_boxes, keep_scores, keep_phrases = [], [], []
    img_area = float(img_w * img_h)
    for box, score, phrase in zip(boxes, scores, phrases):
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if max_area_ratio > 0 and (area / img_area) > max_area_ratio:
            continue
        keep_boxes.append(box)
        keep_scores.append(score)
        keep_phrases.append(phrase)
    if not keep_boxes:
        return boxes.new_zeros((0, 4)), scores.new_zeros((0,)), []
    return torch.stack(keep_boxes, 0), torch.stack(keep_scores, 0), keep_phrases


@torch.inference_mode()
def segment_boxes(sam_model, sam_processor, image: Image.Image, boxes_xyxy: torch.Tensor, device: str):
    """Return list of HxW bool masks, one per box."""
    if boxes_xyxy.numel() == 0:
        return []

    # SamProcessor expects input_boxes as list[list[list[float]]] : batch x num_boxes x 4
    boxes_list = boxes_xyxy.cpu().tolist()
    inputs = sam_processor(
        image,
        input_boxes=[[boxes_list]],
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = sam_model(**inputs)

    # post_process_masks -> list of tensors shaped (batch, num_masks, H, W) or similar
    masks = sam_processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )
    # masks[0]: (num_boxes, num_multimask, H, W) — take first multimask channel / best iou
    iou = getattr(outputs, "iou_scores", None)
    batch_masks = masks[0]  # (N, M, H, W)
    out = []
    for i in range(batch_masks.shape[0]):
        if iou is not None:
            best = int(iou[0, i].argmax().item())
        else:
            best = 0
        out.append(batch_masks[i, best].numpy().astype(bool))
    return out


def main():
    args = parse_args()
    os.environ["HF_ENDPOINT"] = args.hf_endpoint

    cfg = load_config(args.config)
    # CLI overrides
    for key in [
        "input_dir",
        "output_dir",
        "text_prompt",
        "box_threshold",
        "text_threshold",
        "device",
        "dino_model",
        "sam_model",
        "model_dir",
    ]:
        val = getattr(args, key)
        if val is not None:
            cfg[key] = val

    device = cfg.get("device", "cuda:0")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable, fallback to CPU")
        device = "cpu"

    model_dir = cfg.get("model_dir")
    dino_id = resolve_model_path(cfg["dino_model"], model_dir)
    sam_id = resolve_model_path(cfg["sam_model"], model_dir)

    print(f"Device: {device}")
    print(f"Loading Grounding DINO: {dino_id}")
    dino_processor = AutoProcessor.from_pretrained(dino_id)
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_id).to(device)
    dino_model.eval()

    print(f"Loading SAM: {sam_id}")
    sam_processor = SamProcessor.from_pretrained(sam_id)
    sam_model = SamModel.from_pretrained(sam_id).to(device)
    sam_model.eval()

    input_dir = Path(cfg["input_dir"]).resolve()
    output_dir = Path(cfg["output_dir"]).resolve()
    mask_dir = output_dir / "masks"
    overlay_dir = output_dir / "overlays"
    meta_dir = output_dir / "meta"
    inst_dir = output_dir / "instances"
    for d in [mask_dir, overlay_dir, meta_dir, inst_dir]:
        d.mkdir(parents=True, exist_ok=True)

    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in cfg.get("image_exts", IMG_EXTS_DEFAULT)}
    images = list_images(input_dir, exts)
    if args.max_images > 0:
        images = images[: args.max_images]
    if not images:
        print(f"No images found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(images)} images. Prompt: {cfg['text_prompt']!r}")
    coco_meta = []

    for idx, img_path in enumerate(tqdm(images, desc="Annotating")):
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        image_np = np.array(image)

        boxes, scores, phrases = detect_boxes(
            dino_model,
            dino_processor,
            image,
            cfg["text_prompt"],
            float(cfg["box_threshold"]),
            float(cfg["text_threshold"]),
            device,
        )
        boxes, scores, phrases = filter_boxes(
            boxes,
            scores,
            phrases,
            w,
            h,
            float(cfg.get("max_box_area_ratio", 0.90)),
        )

        combined = np.zeros((h, w), dtype=np.uint8)
        instances = []

        if len(boxes) > 0:
            masks = segment_boxes(sam_model, sam_processor, image, boxes, device)
            for i, (mask, box, score, phrase) in enumerate(zip(masks, boxes, scores, phrases)):
                if mask.shape[:2] != (h, w):
                    mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                combined = np.logical_or(combined, mask)
                area = int(mask.sum())
                xyxy = [float(v) for v in box.tolist()]
                inst = {
                    "instance_id": i,
                    "bbox_xyxy": xyxy,
                    "score": float(score),
                    "phrase": phrase,
                    "area": area,
                }
                if cfg.get("export_coco"):
                    inst["rle"] = encode_rle(mask.astype(np.uint8))
                instances.append(inst)

                if cfg.get("save_instance_masks"):
                    inst_name = f"{img_path.stem}_inst{i:02d}.png"
                    cv2.imwrite(str(inst_dir / inst_name), (mask.astype(np.uint8) * 255))

        mask_u8 = combined.astype(np.uint8) * 255
        if cfg.get("save_combined_mask", True):
            cv2.imwrite(str(mask_dir / f"{img_path.stem}.png"), mask_u8)

        if cfg.get("save_overlay", True):
            overlay = draw_overlay(
                image_np,
                combined,
                [b.tolist() for b in boxes] if len(boxes) else [],
                phrases,
            )
            cv2.imwrite(str(overlay_dir / f"{img_path.stem}.jpg"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        meta = {
            "image": img_path.name,
            "width": w,
            "height": h,
            "prompt": cfg["text_prompt"],
            "num_detections": len(instances),
            "instances": instances,
        }
        if cfg.get("save_meta_json", True):
            with open(meta_dir / f"{img_path.stem}.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

        if cfg.get("export_coco"):
            coco_meta.append(
                {
                    "image_id": idx + 1,
                    "file_name": img_path.name,
                    "width": w,
                    "height": h,
                    "instances": instances,
                }
            )

        tqdm.write(f"[{idx + 1}/{len(images)}] {img_path.name}: {len(instances)} objects")

    if cfg.get("export_coco") and coco_meta:
        masks_to_coco(coco_meta, output_dir / "coco_instances.json")

    print(f"Done. Masks -> {mask_dir}")
    print("Reminder: auto masks are pseudo-labels; human-in-the-loop refine before using as test GT.")


if __name__ == "__main__":
    main()

"""Shared Grounded-SAM inference for the annotation workbench."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    SamModel,
    SamProcessor,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG = ROOT / "configs" / "default.yaml"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
QWEN_PY = Path("/home/zyserver/miniconda3/envs/jevcod/bin/python")
QWEN_SCRIPT = Path(__file__).resolve().parent / "qwen_caption.py"
QWEN_MODEL = Path("/home/zyserver/LLB/RS2-SAM2/your_weights/Qwen3.8-27B")
# BF16 ~52GB → 默认占用两张空闲卡（与主标注卡 cuda:0 错开）
QWEN_GPUS = os.environ.get("QWEN_GPUS", "1,2")
QWEN_LOAD_TIMEOUT_SEC = int(os.environ.get("QWEN_LOAD_TIMEOUT_SEC", "900"))
DEFAULT_AI_INSTRUCTION = (
    "任务：为指定目标生成“指代性描述（Referring Expressions）”。\n\n"
    "输入图像说明：\n"
    "图像中有一个被半透明绿色掩码覆盖的目标。\n\n"
    "核心指令：\n\n"
    "隐匿标注：在生成描述时，绝对不能提到“绿色”、“掩码”、“覆盖物”、“标注”、"
    "“高亮”、“半透明”、“涂层”、“色块”或任何与绿色遮罩相关的词汇。"
    "你必须假装这个绿色完全不存在，只关注它覆盖的底层真实物体本身的视觉属性。\n\n"
    "精准指代：该图像中存在多个同类目标（例如多只鸟）。"
    "你必须通过以下维度提供具有唯一确定性的描述，"
    "从而在不提到“绿色”的前提下，让读者能且仅能定位到这个被覆盖的目标：\n\n"
    "空间方位：它在场景中的具体位置，以及它与其他类似物体的相对位置关系"
    "（如：上方、下方、左侧、右侧、靠近水边、居中偏上等）。\n\n"
    "环境上下文：它周围的地面（多石、沙滩、泥地、水际线）或相邻物体的特征。\n\n"
    "姿态与行为：它的动作（如：低头觅食、站立、背对、面朝右下方、俯身等）。\n\n"
    "外观细节：它的羽毛花纹、体型大小、颜色（除绿色外，如灰褐色斑驳）"
    "以及与其他个体的形态差异。\n\n"
    "输出要求：\n\n"
    "语言：必须使用英文 (English)。\n\n"
    "格式：提供 3-5 个不同长度和侧重点的短句/短语，适用于训练和测试。\n\n"
    "语气：自然、客观、符合指代图像分割/检测 (RIS/REC) 数据集规范"
    "（不含“There is”, “I can see”等废话）。\n\n"
    "输出示例样式：\n\n"
    "[基于绝对方位与行为的短语]\n\n"
    "[基于相对邻近关系的描述]\n\n"
    "[基于细节特征的复杂指代句子]\n\n"
    "用英语。"
)


def load_cfg(path: Path | None = None) -> dict:
    with open(path or DEFAULT_CFG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_model_path(model_id: str, model_dir: str | Path | None) -> str:
    if model_dir:
        local = Path(model_dir) / model_id.replace("/", "__")
        if local.exists() and any(local.iterdir()):
            return str(local)
    return model_id


class AnnotatorEngine:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_cfg()
        self.device = self.cfg.get("device", "cuda:0")
        if str(self.device).startswith("cuda") and not torch.cuda.is_available():
            self.device = "cpu"
        self.root = ROOT
        self.input_dir = (self.root / self.cfg.get("input_dir", "raw_images")).resolve()
        self.output_dir = (self.root / self.cfg.get("output_dir", "outputs")).resolve()
        self._ensure_output_dirs()
        os.environ.setdefault("HF_ENDPOINT", os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
        self._loaded = False
        self.dino_model = None
        self.dino_processor = None
        self.sam_model = None
        self.sam_processor = None

    def _ensure_output_dirs(self):
        self.mask_dir = self.output_dir / "masks"
        self.overlay_dir = self.output_dir / "overlays"
        self.meta_dir = self.output_dir / "meta"
        self.corrected_dir = self.output_dir / "corrected"
        self.pairs_dir = self.output_dir / "pairs"  # 每图一份：image ↔ prompt ↔ mask
        # 兼容旧路径；新逻辑统一写入 output_dir/refs.txt，不再一图一个 txt
        self.texts_dir = self.output_dir / "txt"
        for d in [
            self.mask_dir,
            self.overlay_dir,
            self.meta_dir,
            self.corrected_dir,
            self.pairs_dir,
            self.texts_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)
        self.refs_txt = self.output_dir / "refs.txt"
        self._migrate_legacy_texts_if_needed()

    @property
    def refs_path(self) -> Path:
        """主 refs：优先用输入数据集旁的 refs.txt（如 train/refs.txt），否则 outputs/refs.txt。"""
        for p in self._refs_candidates(include_missing_output=False):
            if p.exists():
                return p
        return self.output_dir / "refs.txt"

    def _refs_candidates(self, *, include_missing_output: bool = True) -> list[Path]:
        """可能存放统一文本的 refs.txt：输出目录 + 输入目录 + Imgs 的上一级。"""
        seen: set[Path] = set()
        out: list[Path] = []

        def add(p: Path, *, require_exists: bool) -> None:
            try:
                rp = p.resolve()
            except OSError:
                rp = p
            if rp in seen:
                return
            if require_exists and not p.exists():
                return
            seen.add(rp)
            out.append(p)

        # 数据集侧（打开 Imgs/ 时，refs 常在上一级 train/refs.txt）
        if getattr(self, "input_dir", None) and self.input_dir:
            add(self.input_dir / "refs.txt", require_exists=True)
            add(self.input_dir.parent / "refs.txt", require_exists=True)
        # 平台输出侧：始终可写
        add(self.output_dir / "refs.txt", require_exists=not include_missing_output)
        return out

    def _parse_refs_line(self, line: str) -> tuple[str, str] | None:
        line = (line or "").strip()
        if not line:
            return None
        if "\t" in line:
            name, prompt = line.split("\t", 1)
        else:
            parts = line.split(None, 1)
            if not parts:
                return None
            name = parts[0]
            prompt = parts[1] if len(parts) > 1 else ""
        return name.strip(), prompt.strip()

    def _read_refs_file(self, path: Path) -> dict[str, str]:
        mapping: dict[str, str] = {}
        if not path.exists():
            return mapping
        for raw in path.read_text(encoding="utf-8").splitlines():
            parsed = self._parse_refs_line(raw)
            if not parsed:
                continue
            name, prompt = parsed
            mapping[name] = prompt
            mapping[Path(name).stem] = prompt
        return mapping

    def _read_refs_map(self) -> dict[str, str]:
        """合并所有 refs.txt；靠前的数据集侧优先。"""
        mapping: dict[str, str] = {}
        # 后写覆盖：先读 output，再读数据集侧，使数据集文本优先
        paths = list(reversed(self._refs_candidates(include_missing_output=False)))
        if not paths:
            paths = [self.output_dir / "refs.txt"]
        for path in paths:
            for k, v in self._read_refs_file(path).items():
                mapping[k] = v
        return mapping

    def _write_refs_map_to(self, path: Path, by_name: dict[str, str]) -> None:
        rows: dict[str, str] = {}
        for key, prompt in by_name.items():
            if "." in key:
                rows[key] = (prompt or "").replace("\n", " ").strip()
            else:
                if any(Path(k).stem == key for k in rows):
                    continue
                rows[f"{key}.jpg"] = (prompt or "").replace("\n", " ").strip()
        lines = [f"{name}\t{prompt}" for name, prompt in sorted(rows.items(), key=lambda x: x[0])]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(("\n".join(lines) + ("\n" if lines else "")), encoding="utf-8")

    def _write_refs_map(self, by_name: dict[str, str]) -> None:
        """写入主 refs，并同步到其它已存在的 refs.txt。"""
        targets = self._refs_candidates(include_missing_output=True)
        if not targets:
            targets = [self.output_dir / "refs.txt"]
        for path in targets:
            # 对「仅存在于某份 refs」的条目：以合并后的 by_name 为准整表重写
            self._write_refs_map_to(path, by_name)

    def _remove_stem_from_all_refs(self, stem: str, image_name: str | None = None) -> list[str]:
        """从所有相关 refs.txt（数据集侧 + outputs）删除该图对应行。"""
        touched = []
        # include_missing_output=True：保证 outputs/refs.txt 也会被更新
        for path in self._refs_candidates(include_missing_output=True):
            mapping = self._read_refs_file(path) if path.exists() else {}
            by_name = {k: v for k, v in mapping.items() if "." in k}
            before = len(by_name)
            by_name = {
                k: v
                for k, v in by_name.items()
                if Path(k).stem != stem and (not image_name or k != image_name)
            }
            if len(by_name) == before and path.exists():
                continue
            if len(by_name) != before:
                self._write_refs_map_to(path, by_name)
                touched.append(str(path))
        return touched

    def _migrate_legacy_texts_if_needed(self) -> None:
        """把旧的 txt/*.txt 合并进 refs.txt（只在有遗留文件时执行一次合并）。"""
        legacy = list(self.texts_dir.glob("*.txt")) if self.texts_dir.exists() else []
        if not legacy:
            if not self.refs_path.exists():
                self.refs_path.write_text("", encoding="utf-8")
            return
        mapping = self._read_refs_map()
        # 只保留带扩展名的键做写入
        by_name: dict[str, str] = {k: v for k, v in mapping.items() if "." in k}
        for p in legacy:
            raw = p.read_text(encoding="utf-8").strip()
            parsed = self._parse_refs_line(raw)
            if not parsed:
                continue
            name, prompt = parsed
            if "." not in name:
                name = f"{p.stem}.jpg"
            by_name[name] = prompt
        self._write_refs_map(by_name)
        # 迁移后清理单文件 txt，避免两套数据源
        for p in legacy:
            try:
                p.unlink()
            except OSError:
                pass

    def get_prompt_for_stem(self, stem: str, image_name: str | None = None) -> str:
        mapping = self._read_refs_map()
        if image_name and image_name in mapping:
            return mapping[image_name]
        if stem in mapping:
            return mapping[stem]
        for key, prompt in mapping.items():
            if Path(key).stem == stem:
                return prompt
        meta = self._read_meta(stem)
        return str(meta.get("prompt") or "")

    def set_input_dir(self, path: str | Path) -> Path:
        p = Path(path).expanduser().resolve()
        if not p.exists() or not p.is_dir():
            raise FileNotFoundError(f"输入文件夹不存在: {p}")
        self.input_dir = p
        return p

    def set_output_dir(self, path: str | Path) -> Path:
        p = Path(path).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        self.output_dir = p
        self._ensure_output_dirs()
        return p

    def browse_dirs(self, path: str | Path | None = None) -> dict[str, Any]:
        """服务器本地文件夹浏览（打开文件夹用）。"""
        if path:
            cur = Path(path).expanduser().resolve()
        else:
            cur = self.input_dir if self.input_dir.exists() else self.root
        if not cur.exists():
            cur = Path.home()
        if cur.is_file():
            cur = cur.parent
        parent = cur.parent if cur.parent != cur else None
        dirs = []
        files_img = []
        try:
            for child in sorted(cur.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
                if child.name.startswith("."):
                    continue
                if child.is_dir():
                    dirs.append({"name": child.name, "path": str(child)})
                elif child.is_file() and child.suffix.lower() in IMG_EXTS:
                    files_img.append({"name": child.name, "path": str(child)})
        except PermissionError:
            pass
        return {
            "cwd": str(cur),
            "parent": str(parent) if parent else None,
            "dirs": dirs,
            "images": files_img[:200],
            "image_count": len(files_img),
        }

    def ensure_loaded(self):
        if self._loaded:
            return
        model_dir = self.cfg.get("model_dir")
        if model_dir and not Path(model_dir).is_absolute():
            model_dir = str(self.root / model_dir)
        dino_id = resolve_model_path(self.cfg["dino_model"], model_dir)
        sam_id = resolve_model_path(self.cfg["sam_model"], model_dir)
        print(f"[engine] loading DINO: {dino_id}")
        self.dino_processor = AutoProcessor.from_pretrained(dino_id)
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_id).to(self.device)
        self.dino_model.eval()
        print(f"[engine] loading SAM: {sam_id}")
        self.sam_processor = SamProcessor.from_pretrained(sam_id)
        self.sam_model = SamModel.from_pretrained(sam_id).to(self.device)
        self.sam_model.eval()
        self._loaded = True
        print(f"[engine] ready on {self.device}")

    def list_images(self) -> list[dict[str, Any]]:
        items = []
        if not self.input_dir.exists():
            return items
        for p in sorted(self.input_dir.iterdir()):
            if not p.is_file() or p.suffix.lower() not in IMG_EXTS:
                continue
            mask_p = self.mask_dir / f"{p.stem}.png"
            corr_p = self.corrected_dir / f"{p.stem}.png"
            meta_p = self.meta_dir / f"{p.stem}.json"
            status = "raw"
            if corr_p.exists():
                status = "corrected"
            elif mask_p.exists():
                status = "auto"
            meta = {}
            if meta_p.exists():
                try:
                    meta = json.loads(meta_p.read_text(encoding="utf-8"))
                except Exception:
                    meta = {}
            items.append(
                {
                    "stem": p.stem,
                    "name": p.name,
                    "status": status,
                    "has_mask": mask_p.exists() or corr_p.exists(),
                    "has_text": bool(self.get_prompt_for_stem(p.stem, p.name)),
                    "human_edited": bool(meta.get("human_edited")) or corr_p.exists(),
                    "num_detections": meta.get("num_detections"),
                }
            )
        return items

    def list_outputs(self) -> list[dict[str, Any]]:
        """已保存到输出目录的掩码 + 对应文本（左侧下方列表）。"""
        items = []
        stems = set()
        for p in list(self.mask_dir.glob("*.png")) + list(self.corrected_dir.glob("*.png")):
            stems.add(p.stem)
        refs = self._read_refs_map()
        for stem in sorted(stems):
            mask_path = self.active_mask_path(stem)
            if mask_path is None:
                continue
            img_path = self.find_image(stem)
            img_name = img_path.name if img_path else f"{stem}.jpg"
            prompt = refs.get(img_name) or refs.get(stem) or self._read_meta(stem).get("prompt") or ""
            line = f"{img_name}\t{prompt}".strip() if prompt else ""
            corr = (self.corrected_dir / f"{stem}.png").exists()
            ov = self.overlay_dir / f"{stem}.jpg"
            if not ov.exists():
                ov = self.overlay_dir / f"{stem}.png"
            items.append(
                {
                    "stem": stem,
                    "image_name": img_name,
                    "mask_name": f"{stem}.png",
                    "mask_url": f"/api/mask/{stem}",
                    # 侧栏缩略图用叠图层，避免黑底白块看起来像「整片覆盖」
                    "thumb_url": f"/api/overlay/{stem}" if ov.exists() else f"/api/mask/{stem}",
                    "overlay_url": f"/api/overlay/{stem}" if ov.exists() else None,
                    "text_name": "refs.txt" if prompt else None,
                    "text_line": line,
                    "prompt": prompt,
                    "status": "corrected" if corr else "auto",
                    "mtime": mask_path.stat().st_mtime,
                }
            )
        items.sort(key=lambda x: x["mtime"], reverse=True)
        return items

    def list_texts(self) -> list[dict[str, Any]]:
        """统一从 outputs/refs.txt 读取全部文本行。"""
        items = []
        path = self.refs_path
        if not path.exists():
            return items
        mtime = path.stat().st_mtime
        seen = set()
        for raw in path.read_text(encoding="utf-8").splitlines():
            parsed = self._parse_refs_line(raw)
            if not parsed:
                continue
            img_name, prompt = parsed
            stem = Path(img_name).stem
            if stem in seen:
                continue
            seen.add(stem)
            items.append(
                {
                    "stem": stem,
                    "file_name": "refs.txt",
                    "text_line": f"{img_name}\t{prompt}",
                    "image_name": img_name,
                    "prompt": prompt,
                    "path": str(path),
                    "mtime": mtime,
                }
            )
        return items

    def delete_item(self, stem: str) -> dict[str, Any]:
        """删除一张图及其对应掩码/文本行/元数据（不可恢复）。"""
        removed = []
        img = self.find_image(stem)
        img_name = img.name if img else None
        if img and img.exists():
            img.unlink()
            removed.append(str(img))

        candidates = [
            self.mask_dir / f"{stem}.png",
            self.corrected_dir / f"{stem}.png",
            self.overlay_dir / f"{stem}.jpg",
            self.overlay_dir / f"{stem}.png",
            self.meta_dir / f"{stem}.json",
            self.pairs_dir / f"{stem}.json",
            self.texts_dir / f"{stem}.txt",
            self.output_dir / "ai_overlays" / f"{stem}_qwen.jpg",
            self.output_dir / "instances" / f"{stem}_inst00.png",
        ]
        if self.output_dir.joinpath("instances").exists():
            for p in self.output_dir.joinpath("instances").glob(f"{stem}_inst*.png"):
                candidates.append(p)

        for p in candidates:
            if p.exists():
                p.unlink()
                removed.append(str(p))

        # 从所有 refs.txt 删除该图对应文本行（含 train/refs.txt、outputs/refs.txt）
        for ref_path in self._remove_stem_from_all_refs(stem, img_name):
            removed.append(f"{ref_path}#line:{stem}")

        if hasattr(self, "_rewrite_pairs_index"):
            try:
                self._rewrite_pairs_index()
            except Exception:
                pass

        return {
            "stem": stem,
            "removed": removed,
            "count": len(removed),
            "refs_txt": str(self.refs_path),
        }

    def write_text_pair(self, stem: str, image_name: str, prompt: str) -> Path:
        """写入/更新统一 refs.txt 中该图的一行（追加或覆盖同名行）。"""
        prompt = (prompt or "").replace("\n", " ").strip()
        image_name = image_name or f"{stem}.jpg"
        mapping = self._read_refs_map()
        by_name = {k: v for k, v in mapping.items() if "." in k}
        # 去掉同 stem 的旧行再写入，避免重命名扩展名后残留
        by_name = {k: v for k, v in by_name.items() if Path(k).stem != stem}
        by_name[image_name] = prompt
        self._write_refs_map(by_name)
        return self.refs_path

    def _rewrite_refs_txt(self):
        """兼容旧调用：按当前 map 重写 refs.txt。"""
        mapping = self._read_refs_map()
        by_name = {k: v for k, v in mapping.items() if "." in k}
        self._write_refs_map(by_name)

    def find_image(self, stem: str) -> Path | None:
        for ext in IMG_EXTS:
            p = self.input_dir / f"{stem}{ext}"
            if p.exists():
                return p
            p2 = self.input_dir / f"{stem}{ext.upper()}"
            if p2.exists():
                return p2
        for p in self.input_dir.iterdir():
            if p.is_file() and p.stem == stem:
                return p
        return None

    def active_mask_path(self, stem: str) -> Path | None:
        """优先读更新时间更晚的掩码，避免「重新自动标注后仍被旧 corrected 盖住」。"""
        corr = self.corrected_dir / f"{stem}.png"
        auto = self.mask_dir / f"{stem}.png"
        if corr.exists() and auto.exists():
            return corr if corr.stat().st_mtime >= auto.stat().st_mtime else auto
        if corr.exists():
            return corr
        if auto.exists():
            return auto
        return None

    def load_mask(self, stem: str) -> np.ndarray | None:
        path = self.active_mask_path(stem)
        if path is None:
            return None
        arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if arr is None:
            return None
        return (arr > 127).astype(np.uint8) * 255

    def _read_meta(self, stem: str) -> dict:
        meta_path = self.meta_dir / f"{stem}.json"
        if not meta_path.exists():
            return {}
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def save_mask(
        self,
        stem: str,
        mask_u8: np.ndarray,
        *,
        human_edited: bool = True,
        source: str = "manual",
        instances: list | None = None,
        prompt: str | None = None,
        preserve_prompt: bool = True,
        preserve_instances: bool = False,
    ) -> dict:
        img_path = self.find_image(stem)
        if img_path is None:
            raise FileNotFoundError(f"image not found: {stem}")
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        if mask_u8.shape[:2] != (h, w):
            mask_u8 = cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_bin = ((mask_u8 > 127).astype(np.uint8) * 255)

        old = self._read_meta(stem)
        # 画布保存时若未传 prompt，保留该图已有文本，保证 image↔prompt↔mask 一一对应
        if prompt is None or (preserve_prompt and str(prompt).strip() == ""):
            prompt = old.get("prompt") or self.cfg.get("text_prompt", "")
        if instances is None:
            instances = (old.get("instances") or []) if preserve_instances else []

        # always update working mask
        mask_path = self.mask_dir / f"{stem}.png"
        cv2.imwrite(str(mask_path), mask_bin)
        corr_path = None
        if human_edited:
            corr_path = self.corrected_dir / f"{stem}.png"
            cv2.imwrite(str(corr_path), mask_bin)

        # overlay
        overlay = np.array(image).astype(np.float32)
        m = mask_bin > 127
        color = np.array([0, 220, 120], dtype=np.float32)
        overlay[m] = overlay[m] * 0.55 + color * 0.45
        overlay = overlay.astype(np.uint8)
        overlay_path = self.overlay_dir / f"{stem}.jpg"
        cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        meta = {
            "image": img_path.name,
            "image_path": str(img_path),
            "mask_path": str(mask_path),
            "corrected_mask_path": str(corr_path) if corr_path else None,
            "overlay_path": str(overlay_path),
            "width": w,
            "height": h,
            "prompt": prompt or "",
            "num_detections": len(instances or []),
            "instances": instances or [],
            "human_edited": human_edited,
            "source": source,
            "pipeline": {
                "detector": "Grounding DINO (text -> boxes)",
                "segmentor": "SAM (boxes/points -> mask)",
                "note": "SAM/SAM2 本身不接收自由文本，文本只进 Grounding DINO",
            },
        }
        with open(self.meta_dir / f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # 显式三元组记录，方便按图检索
        pair = {
            "stem": stem,
            "image": img_path.name,
            "prompt": meta["prompt"],
            "mask": f"masks/{stem}.png",
            "corrected_mask": f"corrected/{stem}.png" if human_edited else None,
            "human_edited": human_edited,
            "source": source,
        }
        with open(self.pairs_dir / f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump(pair, f, ensure_ascii=False, indent=2)
        self._rewrite_pairs_index()

        # 统一写入 refs.txt 一行「图片名\t文本描述」
        text_path = self.write_text_pair(stem, img_path.name, meta["prompt"])
        pair["text"] = "refs.txt"
        pair["text_line"] = f"{img_path.name}\t{meta['prompt']}".strip()

        return {
            "stem": stem,
            "status": "corrected" if human_edited else "auto",
            "mask_path": str(mask_path),
            "text_path": str(text_path),
            "prompt": meta["prompt"],
            "pair": pair,
        }

    def _rewrite_pairs_index(self):
        """汇总 outputs/pairs_index.jsonl ：每行 = 一图的 image/prompt/mask。"""
        rows = []
        for p in sorted(self.pairs_dir.glob("*.json")):
            try:
                rows.append(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                continue
        index_path = self.output_dir / "pairs_index.jsonl"
        with open(index_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    @torch.inference_mode()
    def detect(self, image: Image.Image, prompt: str, box_thr: float, text_thr: float, max_area_ratio: float = 0.9):
        self.ensure_loaded()
        caption = prompt.lower().strip()
        if not caption.endswith("."):
            caption += "."
        inputs = self.dino_processor(images=image, text=caption, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.dino_model(**inputs)
        target_sizes = [image.size[::-1]]
        try:
            results = self.dino_processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                threshold=box_thr,
                text_threshold=text_thr,
                target_sizes=target_sizes,
            )[0]
        except TypeError:
            results = self.dino_processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                box_threshold=box_thr,
                text_threshold=text_thr,
                target_sizes=target_sizes,
            )[0]
        boxes = results["boxes"].detach().cpu()
        scores = results["scores"].detach().cpu()
        labels = results.get("labels", results.get("text_labels", []))
        phrases = [str(x) for x in (labels.tolist() if hasattr(labels, "tolist") else labels)]

        w, h = image.size
        img_area = float(w * h)
        keep_b, keep_s, keep_p = [], [], []
        for box, score, phrase in zip(boxes, scores, phrases):
            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if max_area_ratio > 0 and area / img_area > max_area_ratio:
                continue
            keep_b.append(box)
            keep_s.append(score)
            keep_p.append(phrase)
        if not keep_b:
            return torch.zeros((0, 4)), torch.zeros((0,)), []
        return torch.stack(keep_b), torch.stack(keep_s), keep_p

    @torch.inference_mode()
    def segment_boxes(self, image: Image.Image, boxes_xyxy: torch.Tensor) -> list[np.ndarray]:
        self.ensure_loaded()
        if boxes_xyxy.numel() == 0:
            return []
        boxes_list = boxes_xyxy.cpu().tolist()
        inputs = self.sam_processor(image, input_boxes=[[boxes_list]], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.sam_model(**inputs)
        masks = self.sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )
        batch_masks = masks[0]
        iou = getattr(outputs, "iou_scores", None)
        out = []
        for i in range(batch_masks.shape[0]):
            best = int(iou[0, i].argmax().item()) if iou is not None else 0
            out.append(batch_masks[i, best].numpy().astype(bool))
        return out

    def auto_annotate_one(
        self,
        stem: str,
        prompt: str | None = None,
        box_threshold: float | None = None,
        text_threshold: float | None = None,
    ) -> dict:
        img_path = self.find_image(stem)
        if img_path is None:
            raise FileNotFoundError(stem)
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        prompt = prompt or self.cfg.get("text_prompt", "object")
        box_thr = float(box_threshold if box_threshold is not None else self.cfg.get("box_threshold", 0.35))
        text_thr = float(text_threshold if text_threshold is not None else self.cfg.get("text_threshold", 0.25))
        max_ratio = float(self.cfg.get("max_box_area_ratio", 0.9))

        boxes, scores, phrases = self.detect(image, prompt, box_thr, text_thr, max_ratio)
        combined = np.zeros((h, w), dtype=np.uint8)
        instances = []
        if len(boxes) > 0:
            masks = self.segment_boxes(image, boxes)
            for i, (mask, box, score, phrase) in enumerate(zip(masks, boxes, scores, phrases)):
                if mask.shape[:2] != (h, w):
                    mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                combined = np.logical_or(combined, mask)
                instances.append(
                    {
                        "instance_id": i,
                        "bbox_xyxy": [float(v) for v in box.tolist()],
                        "score": float(score),
                        "phrase": phrase,
                        "area": int(mask.sum()),
                    }
                )
        # auto overwrite working mask; remove old correction so status becomes auto
        corr = self.corrected_dir / f"{stem}.png"
        if corr.exists():
            corr.unlink()
        result = self.save_mask(
            stem,
            combined.astype(np.uint8) * 255,
            human_edited=False,
            source="auto",
            instances=instances,
            prompt=prompt,
        )
        result["num_detections"] = len(instances)
        result["instances"] = instances
        return result

    def refine_from_box(self, stem: str, box_xyxy: list[float], mode: str = "replace") -> dict:
        """User-drawn box -> SAM mask. mode: replace | add | subtract."""
        img_path = self.find_image(stem)
        if img_path is None:
            raise FileNotFoundError(stem)
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        box = torch.tensor([box_xyxy], dtype=torch.float32)
        masks = self.segment_boxes(image, box)
        if not masks:
            raise RuntimeError("SAM returned empty mask")
        new_m = masks[0]
        if new_m.shape[:2] != (h, w):
            new_m = cv2.resize(new_m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

        cur = self.load_mask(stem)
        if cur is None:
            cur = np.zeros((h, w), dtype=np.uint8)
        else:
            cur = (cur > 127).astype(bool)

        if mode == "add":
            out = np.logical_or(cur, new_m)
        elif mode == "subtract":
            out = np.logical_and(cur, np.logical_not(new_m))
        else:
            out = new_m

        return self.save_mask(
            stem,
            out.astype(np.uint8) * 255,
            human_edited=True,
            source=f"box_sam_{mode}",
            instances=[
                {
                    "instance_id": 0,
                    "bbox_xyxy": [float(v) for v in box_xyxy],
                    "score": 1.0,
                    "phrase": "user_box",
                    "area": int(out.sum()),
                }
            ],
            prompt="user_box",
        )

    def refine_from_point(self, stem: str, x: float, y: float, label: int = 1, mode: str = "add") -> dict:
        """Click point -> SAM. label 1=foreground, 0=background."""
        self.ensure_loaded()
        img_path = self.find_image(stem)
        if img_path is None:
            raise FileNotFoundError(stem)
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        inputs = self.sam_processor(
            image,
            input_points=[[[x, y]]],
            input_labels=[[[label]]],
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = self.sam_model(**inputs)
        masks = self.sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )
        batch = masks[0]  # (1, M, H, W) or (M, H, W)
        iou = getattr(outputs, "iou_scores", None)
        if batch.ndim == 4:
            best = int(iou[0, 0].argmax().item()) if iou is not None else 0
            new_m = batch[0, best].numpy().astype(bool)
        else:
            best = int(iou[0].argmax().item()) if iou is not None else 0
            new_m = batch[best].numpy().astype(bool)
        if new_m.shape[:2] != (h, w):
            new_m = cv2.resize(new_m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

        cur = self.load_mask(stem)
        cur = np.zeros((h, w), dtype=bool) if cur is None else (cur > 127).astype(bool)
        if mode == "subtract" or label == 0:
            out = np.logical_and(cur, np.logical_not(new_m)) if cur.any() else np.logical_not(new_m)
            # for bg click on empty: just use inverse of clicked region removal from empty = empty
            if not cur.any() and label == 0:
                out = np.zeros((h, w), dtype=bool)
            elif label == 0:
                out = np.logical_and(cur, np.logical_not(new_m))
            else:
                out = np.logical_or(cur, new_m) if mode == "add" else new_m
        else:
            out = np.logical_or(cur, new_m) if mode == "add" else new_m

        return self.save_mask(
            stem,
            out.astype(np.uint8) * 255,
            human_edited=True,
            source="point_sam",
            instances=[{"instance_id": 0, "point": [x, y], "label": label, "area": int(out.sum())}],
            prompt="user_point",
        )

    def build_green_overlay(self, stem: str, mask_u8: np.ndarray | None = None) -> Path:
        """合成「原图 + 绿色掩码」供 Qwen 看目标区域。"""
        img_path = self.find_image(stem)
        if img_path is None:
            raise FileNotFoundError(stem)
        image = np.array(Image.open(img_path).convert("RGB"))
        h, w = image.shape[:2]
        if mask_u8 is None:
            mask_u8 = self.load_mask(stem)
        if mask_u8 is None:
            raise ValueError("当前图没有掩码，请先自动标注或多边形手标")
        if mask_u8.shape[:2] != (h, w):
            mask_u8 = cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_NEAREST)
        m = mask_u8 > 127
        if not m.any():
            raise ValueError("掩码为空，请先标注目标区域")
        out = image.astype(np.float32)
        color = np.array([0, 220, 120], dtype=np.float32)
        out[m] = out[m] * 0.45 + color * 0.55
        out = out.astype(np.uint8)
        # bbox outline
        ys, xs = np.where(m)
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 80), max(2, min(h, w) // 250))

        ai_dir = self.output_dir / "ai_overlays"
        ai_dir.mkdir(parents=True, exist_ok=True)
        overlay_path = ai_dir / f"{stem}_qwen.jpg"
        cv2.imwrite(str(overlay_path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        return overlay_path

    _qwen_lock = threading.Lock()
    _qwen_proc: subprocess.Popen | None = None
    _qwen_gpus: str | None = None

    def _stop_qwen_worker(self) -> None:
        proc = AnnotatorEngine._qwen_proc
        AnnotatorEngine._qwen_proc = None
        AnnotatorEngine._qwen_gpus = None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.write("quit\n")
                proc.stdin.flush()
        except Exception:
            pass
        try:
            proc.wait(timeout=10)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _ensure_qwen_worker_locked(self, gpus: str) -> subprocess.Popen:
        """常驻加载 Qwen3.8-27B BF16。调用方须已持有 _qwen_lock。"""
        proc = AnnotatorEngine._qwen_proc
        if (
            proc is not None
            and proc.poll() is None
            and AnnotatorEngine._qwen_gpus == gpus
        ):
            return proc
        if proc is not None:
            self._stop_qwen_worker()

        if not QWEN_PY.exists():
            raise RuntimeError(f"找不到 Qwen Python: {QWEN_PY}")
        if not QWEN_SCRIPT.exists():
            raise RuntimeError(f"找不到脚本: {QWEN_SCRIPT}")
        if not QWEN_MODEL.exists():
            raise RuntimeError(f"找不到 Qwen 权重: {QWEN_MODEL}")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpus
        env["HF_ENDPOINT"] = env.get("HF_ENDPOINT", "https://hf-mirror.com")
        env["TOKENIZERS_PARALLELISM"] = "false"
        log_path = self.output_dir / "qwen_worker.log"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log_f = open(log_path, "ab", buffering=0)
        proc = subprocess.Popen(
            [
                str(QWEN_PY),
                str(QWEN_SCRIPT),
                "--serve",
                "--model",
                str(QWEN_MODEL),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log_f,
            text=True,
            bufsize=1,
            env=env,
        )
        AnnotatorEngine._qwen_proc = proc
        AnnotatorEngine._qwen_gpus = gpus

        assert proc.stdout is not None
        deadline = time.time() + QWEN_LOAD_TIMEOUT_SEC
        ready_line = None
        while time.time() < deadline:
            if proc.poll() is not None:
                err = ""
                try:
                    err = log_path.read_text(encoding="utf-8", errors="ignore")[-2000:]
                except Exception:
                    pass
                AnnotatorEngine._qwen_proc = None
                raise RuntimeError(f"Qwen worker 启动失败: {err or proc.returncode}")
            line = proc.stdout.readline()
            if not line:
                time.sleep(0.2)
                continue
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("ready") and payload.get("ok"):
                ready_line = payload
                break
            if payload.get("ok") is False:
                AnnotatorEngine._qwen_proc = None
                raise RuntimeError(payload.get("error") or "Qwen 加载失败")
        if ready_line is None:
            self._stop_qwen_worker()
            raise RuntimeError(
                f"Qwen worker 加载超时（>{QWEN_LOAD_TIMEOUT_SEC}s），详见 {log_path}"
            )
        return proc

    def generate_caption_with_qwen(
        self,
        stem: str,
        instruction: str | None = None,
        mask_u8: np.ndarray | None = None,
        gpu: str | None = None,
        timeout_sec: int = 600,
    ) -> dict[str, Any]:
        """调用常驻 Qwen3.8-27B（BF16，默认 GPU 1,2）。"""
        gpus = (gpu or QWEN_GPUS).strip() or QWEN_GPUS
        overlay = self.build_green_overlay(stem, mask_u8=mask_u8)
        img_path = self.find_image(stem)
        req = {
            "image": str(img_path),
            "overlay": str(overlay),
            "instruction": (instruction or "").strip() or DEFAULT_AI_INSTRUCTION,
            "model": str(QWEN_MODEL),
            "max_new_tokens": 256,
            "enable_thinking": False,
        }

        t0 = time.time()
        with AnnotatorEngine._qwen_lock:
            proc = self._ensure_qwen_worker_locked(gpus)
            assert proc.stdin is not None and proc.stdout is not None
            try:
                proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
                proc.stdin.flush()
            except Exception as e:
                self._stop_qwen_worker()
                raise RuntimeError(f"无法向 Qwen worker 发送请求: {e}") from e

            deadline = time.time() + timeout_sec
            payload = None
            while time.time() < deadline:
                if proc.poll() is not None:
                    AnnotatorEngine._qwen_proc = None
                    raise RuntimeError("Qwen worker 意外退出")
                line = proc.stdout.readline()
                if not line:
                    time.sleep(0.05)
                    continue
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as e:
                    raise RuntimeError(f"Qwen 输出不是 JSON: {line[:500]}") from e
                break
            if payload is None:
                raise RuntimeError(f"Qwen 推理超时（>{timeout_sec}s）")
            if not payload.get("ok"):
                raise RuntimeError(payload.get("error") or "Qwen 返回失败")
            text = (payload.get("text") or "").strip()
            if not text:
                raise RuntimeError("Qwen 返回空文本")
            elapsed = time.time() - t0
            return {
                "text": text,
                "instruction": req["instruction"],
                "overlay_path": str(overlay),
                "elapsed_sec": round(elapsed, 2),
                "gpu": gpus,
                "model": str(QWEN_MODEL.name),
            }


_engine: AnnotatorEngine | None = None


def get_engine() -> AnnotatorEngine:
    global _engine
    if _engine is None:
        _engine = AnnotatorEngine()
    return _engine

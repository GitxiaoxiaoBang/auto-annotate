#!/usr/bin/env python3
"""Human-in-the-loop annotation workbench for Grounded-SAM."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image
from starlette.middleware.base import BaseHTTPMiddleware

from engine import get_engine


class NoCacheApiMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
        return response

WEB_DIR = Path(__file__).resolve().parent
app = FastAPI(title="Grounded-SAM Annotator", version="1.0")
app.add_middleware(NoCacheApiMiddleware)
app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")


class AutoReq(BaseModel):
    stem: str
    prompt: str | None = None
    box_threshold: float | None = None
    text_threshold: float | None = None


class SaveMaskReq(BaseModel):
    stem: str
    # raw PNG base64 (data URL or plain)
    mask_png_b64: str
    human_edited: bool = True
    prompt: str | None = None  # 与该图一一对应保存的文本提示


class SaveTextReq(BaseModel):
    stem: str
    prompt: str = ""


class BoxReq(BaseModel):
    stem: str
    box_xyxy: list[float] = Field(..., min_length=4, max_length=4)
    mode: str = "replace"  # replace | add | subtract


class PointReq(BaseModel):
    stem: str
    x: float
    y: float
    label: int = 1  # 1 fg, 0 bg
    mode: str = "add"


class AICaptionReq(BaseModel):
    stem: str
    instruction: str | None = None
    # 可选：传当前画布掩码（未保存也能生成）；不传则读磁盘掩码
    mask_png_b64: str | None = None
    apply_to_prompt: bool = True
    auto_save_text: bool = False
    gpu: str = "1,2"


class SetDirsReq(BaseModel):
    input_dir: str | None = None
    output_dir: str | None = None


def _decode_mask_b64(b64: str) -> np.ndarray:
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("invalid mask image")
    if img.ndim == 3:
        # 前端 canvas 导出的 PNG 多为 RGBA，且 alpha 恒为 255。
        # 绝不能用 alpha 当掩码，否则一保存就变成「整图掩码」。
        if img.shape[2] == 4:
            img = np.max(img[:, :, :3], axis=2)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return (img > 127).astype(np.uint8) * 255


def _mask_to_png_b64(mask_u8: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", mask_u8)
    if not ok:
        raise RuntimeError("encode failed")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


@app.get("/", response_class=HTMLResponse)
def index():
    html = (WEB_DIR / "templates" / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.get("/api/health")
def health():
    eng = get_engine()
    return {
        "ok": True,
        "device": eng.device,
        "models_loaded": eng._loaded,
        "input_dir": str(eng.input_dir),
        "output_dir": str(eng.output_dir),
        "mask_dir": str(eng.mask_dir),
        "texts_dir": str(eng.refs_path),
        "refs_txt": str(eng.refs_path),
    }


@app.get("/api/images")
def list_images():
    eng = get_engine()
    return {
        "items": eng.list_images(),
        "input_dir": str(eng.input_dir),
        "output_dir": str(eng.output_dir),
        "mask_dir": str(eng.mask_dir),
        "texts_dir": str(eng.refs_path),
        "refs_txt": str(eng.refs_path),
    }


@app.get("/api/browse")
def browse(path: str | None = None):
    return get_engine().browse_dirs(path)


@app.post("/api/set_dirs")
def set_dirs(req: SetDirsReq):
    eng = get_engine()
    out = {}
    try:
        if req.input_dir:
            out["input_dir"] = str(eng.set_input_dir(req.input_dir))
        if req.output_dir:
            out["output_dir"] = str(eng.set_output_dir(req.output_dir))
            out["mask_dir"] = str(eng.mask_dir)
            out["texts_dir"] = str(eng.refs_path)
            out["refs_txt"] = str(eng.refs_path)
    except FileNotFoundError as e:
        raise HTTPException(400, str(e)) from e
    out.setdefault("input_dir", str(eng.input_dir))
    out.setdefault("output_dir", str(eng.output_dir))
    out.setdefault("mask_dir", str(eng.mask_dir))
    out.setdefault("texts_dir", str(eng.refs_path))
    out.setdefault("refs_txt", str(eng.refs_path))
    out["items"] = eng.list_images()
    return out


@app.get("/api/outputs")
def list_outputs():
    eng = get_engine()
    return {
        "items": eng.list_outputs(),
        "texts": eng.list_texts(),
        "output_dir": str(eng.output_dir),
        "mask_dir": str(eng.mask_dir),
        "texts_dir": str(eng.refs_path),
        "refs_txt": str(eng.refs_path),
    }


@app.get("/api/texts")
def list_texts():
    eng = get_engine()
    return {
        "items": eng.list_texts(),
        "texts_dir": str(eng.refs_path),
        "refs_txt": str(eng.refs_path),
    }


@app.get("/api/image/{stem}")
def get_image(stem: str):
    eng = get_engine()
    path = eng.find_image(stem)
    if path is None:
        raise HTTPException(404, "image not found")
    return FileResponse(path)


@app.get("/api/mask/{stem}")
def get_mask(stem: str):
    eng = get_engine()
    path = eng.active_mask_path(stem)
    if path is None:
        raise HTTPException(404, "mask not found")
    return FileResponse(path)


@app.get("/api/overlay/{stem}")
def get_overlay(stem: str):
    """原图 + 绿色半透明掩码叠层，供侧栏预览（不是二值掩码）。"""
    eng = get_engine()
    for ext in (".jpg", ".png"):
        path = eng.overlay_dir / f"{stem}{ext}"
        if path.exists():
            return FileResponse(path)
    # 没有现成 overlay 时现场生成一份
    try:
        path = eng.build_green_overlay(stem)
        return FileResponse(path)
    except Exception as e:
        raise HTTPException(404, f"overlay not found: {e}") from e


@app.get("/api/text/{stem}")
def get_text(stem: str):
    eng = get_engine()
    path = eng.find_image(stem)
    img_name = path.name if path else f"{stem}.jpg"
    prompt = eng.get_prompt_for_stem(stem, img_name)
    if not prompt:
        raise HTTPException(404, "text not found")
    from fastapi.responses import PlainTextResponse

    return PlainTextResponse(f"{img_name}\t{prompt}\n", media_type="text/plain; charset=utf-8")


@app.get("/api/item/{stem}")
def get_item(stem: str):
    eng = get_engine()
    path = eng.find_image(stem)
    if path is None:
        raise HTTPException(404, "image not found")
    im = Image.open(path)
    w, h = im.size
    mask = eng.load_mask(stem)
    meta_path = eng.meta_dir / f"{stem}.json"
    meta = {}
    if meta_path.exists():
        import json

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    # 文本以 refs.txt 为准，覆盖 meta.prompt
    prompt = eng.get_prompt_for_stem(stem, path.name)
    if prompt:
        meta = {**meta, "prompt": prompt}
    corr = (eng.corrected_dir / f"{stem}.png").exists()
    status = "corrected" if corr else ("auto" if mask is not None else "raw")
    return {
        "stem": stem,
        "name": path.name,
        "width": w,
        "height": h,
        "status": status,
        "mask_b64": _mask_to_png_b64(mask) if mask is not None else None,
        "meta": meta,
        "image_url": f"/api/image/{stem}",
    }


@app.post("/api/save_mask")
def save_mask(req: SaveMaskReq):
    eng = get_engine()
    if eng.find_image(req.stem) is None:
        raise HTTPException(404, "image not found")
    try:
        mask = _decode_mask_b64(req.mask_png_b64)
    except Exception as e:
        raise HTTPException(400, f"bad mask: {e}") from e
    result = eng.save_mask(
        req.stem,
        mask,
        human_edited=req.human_edited,
        source="canvas",
        prompt=req.prompt,
        preserve_prompt=True,
        preserve_instances=True,
    )
    return result


@app.post("/api/save_text")
def save_text(req: SaveTextReq):
    """只写入 refs.txt / meta.prompt，绝不改动掩码文件。"""
    eng = get_engine()
    img = eng.find_image(req.stem)
    if img is None:
        raise HTTPException(404, "image not found")
    text_path = eng.write_text_pair(req.stem, img.name, req.prompt or "")
    meta = eng._read_meta(req.stem)
    meta["prompt"] = (req.prompt or "").strip()
    meta["image"] = img.name
    eng.meta_dir.mkdir(parents=True, exist_ok=True)
    import json as _json

    with open(eng.meta_dir / f"{req.stem}.json", "w", encoding="utf-8") as f:
        _json.dump(meta, f, ensure_ascii=False, indent=2)
    return {
        "stem": req.stem,
        "prompt": meta["prompt"],
        "text_path": str(text_path),
        "mask_unchanged": True,
    }


@app.post("/api/auto_annotate")
def auto_annotate(req: AutoReq):
    eng = get_engine()
    try:
        result = eng.auto_annotate_one(
            req.stem,
            prompt=req.prompt,
            box_threshold=req.box_threshold,
            text_threshold=req.text_threshold,
        )
    except FileNotFoundError:
        raise HTTPException(404, "image not found") from None
    except Exception as e:
        raise HTTPException(500, str(e)) from e
    mask = eng.load_mask(req.stem)
    result["mask_b64"] = _mask_to_png_b64(mask) if mask is not None else None
    return result


@app.post("/api/refine_box")
def refine_box(req: BoxReq):
    eng = get_engine()
    try:
        result = eng.refine_from_box(req.stem, req.box_xyxy, mode=req.mode)
    except FileNotFoundError:
        raise HTTPException(404, "image not found") from None
    except Exception as e:
        raise HTTPException(500, str(e)) from e
    mask = eng.load_mask(req.stem)
    result["mask_b64"] = _mask_to_png_b64(mask) if mask is not None else None
    return result


@app.post("/api/refine_point")
def refine_point(req: PointReq):
    eng = get_engine()
    try:
        result = eng.refine_from_point(req.stem, req.x, req.y, label=req.label, mode=req.mode)
    except FileNotFoundError:
        raise HTTPException(404, "image not found") from None
    except Exception as e:
        raise HTTPException(500, str(e)) from e
    mask = eng.load_mask(req.stem)
    result["mask_b64"] = _mask_to_png_b64(mask) if mask is not None else None
    return result


@app.post("/api/ai_caption")
def ai_caption(req: AICaptionReq):
    """基于当前图 + 绿色掩码，用本地 Qwen3.8-27B（BF16）生成英文描述。"""
    eng = get_engine()
    if eng.find_image(req.stem) is None:
        raise HTTPException(404, "image not found")
    mask = None
    if req.mask_png_b64:
        try:
            mask = _decode_mask_b64(req.mask_png_b64)
        except Exception as e:
            raise HTTPException(400, f"bad mask: {e}") from e
    try:
        result = eng.generate_caption_with_qwen(
            req.stem,
            instruction=req.instruction,
            mask_u8=mask,
            gpu=req.gpu or "1,2",
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        raise HTTPException(500, str(e)) from e

    text = result["text"]
    # 可选：只写 txt，不改掩码
    if req.auto_save_text:
        img_path = eng.find_image(req.stem)
        eng.write_text_pair(req.stem, img_path.name, text)
        # 同步 meta.prompt
        meta = eng._read_meta(req.stem)
        meta["prompt"] = text
        meta["source"] = meta.get("source") or "ai_caption"
        meta["ai_caption"] = {
            "instruction": result["instruction"],
            "elapsed_sec": result["elapsed_sec"],
        }
        with open(eng.meta_dir / f"{req.stem}.json", "w", encoding="utf-8") as f:
            import json as _json

            _json.dump(meta, f, ensure_ascii=False, indent=2)
        result["text_path"] = str(eng.refs_path)

    result["apply_to_prompt"] = req.apply_to_prompt
    return result


@app.post("/api/delete_item")
def delete_item(stem: str):
    """删除当前输入图，并同步删除 outputs 下掩码、refs.txt 对应行与元数据。"""
    eng = get_engine()
    has_img = eng.find_image(stem) is not None
    has_mask = eng.active_mask_path(stem) is not None
    has_text = bool(eng.get_prompt_for_stem(stem))
    if not has_img and not has_mask and not has_text:
        raise HTTPException(404, "image not found")
    result = eng.delete_item(stem)
    return result


@app.post("/api/clear_mask")
def clear_mask(stem: str):
    eng = get_engine()
    path = eng.find_image(stem)
    if path is None:
        raise HTTPException(404, "image not found")
    im = Image.open(path)
    w, h = im.size
    empty = np.zeros((h, w), dtype=np.uint8)
    # 清空掩码时保留已有文本
    old_prompt = eng._read_meta(stem).get("prompt") or ""
    result = eng.save_mask(
        stem,
        empty,
        human_edited=True,
        source="clear",
        instances=[],
        prompt=old_prompt,
        preserve_prompt=True,
    )
    result["mask_b64"] = _mask_to_png_b64(empty)
    return result


def main():
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=7865, reload=False)


if __name__ == "__main__":
    main()

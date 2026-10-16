# Grounded-SAM 自动化预标注

开放词表检测（Grounding DINO）+ 像素级分割（SAM）的批量预标注工具包。

```
文本提示 ──► Grounding DINO ──► 边界框 ──► SAM ──► 二值 Mask PNG
                                              └─► overlay / meta JSON / COCO
```

## 目录

```
auto-annotate/
  configs/default.yaml     # 默认配置
  scripts/
    download_weights.py    # 下载预训练权重（走 hf-mirror）
    batch_annotate.py      # 批量自动标注
  raw_images/              # 放入待标注原图
  outputs/
    masks/                 # 合并后的二值掩码 (0/255 PNG)
    overlays/              # 人工抽检用叠图
    meta/                  # 每图 boxes / phrases / scores
  models/                  # 本地下载的 HF 模型快照
```

## 环境

本机 RTX 5090 需 **CUDA 12.8 + PyTorch sm_120** 支持。直接复用已有环境：

```bash
conda activate rs2sam2
cd /home/zyserver/LLB/auto-annotate
pip install -r requirements.txt   # 缺什么补什么即可
```

国内网络默认使用镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 1. 下载权重

```bash
conda activate rs2sam2
export HF_ENDPOINT=https://hf-mirror.com
python scripts/download_weights.py \
  --dino IDEA-Research/grounding-dino-base \
  --sam facebook/sam-vit-large \
  --model_dir ./models
```

可选更强分割：`--sam facebook/sam-vit-huge`（更大更慢）。

## 2. 批量标注

把图片放到 `raw_images/`，或通过参数指定目录：

```bash
python scripts/batch_annotate.py \
  --config configs/default.yaml \
  --input_dir ./raw_images \
  --output_dir ./outputs \
  --text_prompt "fish . frog . camouflaged animal" \
  --box_threshold 0.35 \
  --text_threshold 0.25
```

冒烟测试只跑几张：

```bash
python scripts/batch_annotate.py --max_images 3 --text_prompt "camouflaged animal"
```

## 3. 输出说明

| 路径 | 含义 |
|------|------|
| `outputs/masks/*.png` | 同名二值前景掩码（多实例已 OR 合并） |
| `outputs/overlays/*.jpg` | 红框 + 绿色半透明掩码，便于抽检 |
| `outputs/meta/*.json` | 每个框的坐标、分数、短语 |
| `outputs/coco_instances.json` | 配置里 `export_coco: true` 时生成 |

## 4. 质量把控（重要）

1. **伪标不能直接当测试集真值**，否则会把检测器/SAM 的系统偏差写进评测。
2. 将 `masks/` 导入 Labelme / CVAT / AnyLabeling，对伪装边缘、触角、反光盲区做二次精修。
3. 训练集可用伪标冷启动；测试集建议全量或分层人工复核。

## 5. 标注工作台（推荐：可改错 / 重标）

启动本地 Web 页面：

```bash
conda activate rs2sam2
export HF_ENDPOINT=https://hf-mirror.com
bash scripts/run_web.sh
# 浏览器打开 http://127.0.0.1:7865
```

页面能力：

| 操作 | 说明 |
|------|------|
| 自动标注 | 用当前文本提示跑 Grounding DINO + SAM |
| 画笔 / 橡皮 | 直接涂改掩码，支持撤销 |
| 框选 SAM | 自己拖框，让 SAM 按框重切（可替换/叠加/减去） |
| 点选 SAM | 点击目标，SAM 加点扩展前景 |
| 保存修正 | 写入 `outputs/masks` 与 `outputs/corrected`，列表状态变为 `corrected` |

快捷键：`Ctrl+S` 保存，`Ctrl+Z` 撤销，`B` 画笔，`E` 橡皮。

## 6. 指代文本（可选，衔接 RS2-SAM2）

若已有精修后的实例 Mask，可用仓库内现成脚本生成 Referring 描述：

```bash
# 见 RS2-SAM2/tools/generate_rrcod_referring_texts.py
# 本地 Qwen2.5-VL 按高亮框+掩码写指代句
```

本目录只负责 **检测框 + 分割掩码** 预标注；指代句生成复用 `RS2-SAM2/tools/`。

## 配置项速查

编辑 `configs/default.yaml`：

- `text_prompt`：多类别用 `"a . b . c"` 形式
- `box_threshold` / `text_threshold`：越高越保守（漏检↑，误检↓）
- `save_instance_masks`：是否额外保存每个实例单独掩码
- `export_coco`：是否导出 COCO json

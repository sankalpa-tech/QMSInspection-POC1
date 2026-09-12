# Manufacturing Defect Inspection — YOLO11 Instance Segmentation

A focused pipeline for **multiclass instance segmentation** of manufacturing
surface defects (scratch / line defect, crack, dent, discoloration, rust,
electroplating defects, incomplete embossing, serration, etc.).

Every individual defect gets its **own mask + class + confidence** — this is
instance segmentation, not plain classification or bounding-box detection.

Annotation is done in **Roboflow** (or CVAT); this repo owns the CV pipeline:
dataset validation, training (with day-by-day incremental learning), and
dataset/model versioning.

```
Annotate in Roboflow (instance masks)
        │  export "YOLOv11" segmentation
        ▼
dataset/  (train/valid, YOLO-seg polygon labels)
        │
        ▼
visualize_labels.py   → sanity-check every annotation visually
        │
        ▼
train_daily.py        → YOLO11-seg training (auto warm-start + versioning)
        │
        ▼
runs_daily/ + model_registry/  → weights, metrics, dataset snapshots
```

---

## 1. Setup

```bash
python -m pip install -r requirements.txt
# or:  bash docs/setup.sh            (CPU)
#      bash docs/setup.sh cu121      (GPU, CUDA 12.1)
```

Requires Python 3.9–3.11. Ultralytics pulls in PyTorch automatically (CPU build
by default; install a CUDA build for GPU training).

---

## 2. Dataset layout (YOLO-seg)

Export from Roboflow as **YOLOv11** (segmentation, under a `TXT` heading — NOT
"Oriented Bounding Boxes", and NOT anything under "Convert To Object Detection",
both of which discard your masks). Unzip into `dataset/`:

```
dataset/
├── data.yaml           # class names + split paths
├── train/
│   ├── images/
│   └── labels/         # one .txt per image: "class x1 y1 x2 y2 ..." (normalized polygon)
├── valid/
│   ├── images/
│   └── labels/
└── test/  (optional)
```

Each label line is a **polygon** (many points) per defect instance. A thin
scratch is traced as a slender polygon — that is correct.

`data.yaml` must point at the dataset with an absolute `path`, e.g.:
```yaml
path: C:/workspace-ai/QMSInspection-POC/dataset
train: train/images
val: valid/images
nc: 5
names: ['black mark after electroplating', 'electroplating defect',
        'incomplete embossing', 'line defect', 'serration']
```

---

## 3. Validate annotations

Render masks onto the images and flag malformed labels before training:

```bash
python visualize_labels.py --dataset dataset --out label_preview
```
Open `label_preview/` and confirm each mask hugs its defect. The script warns
about degenerate annotations (fewer than 3 points = not a real mask).

---

## 4. Train

As you collect and annotate more parts over time, `train_daily.py`
**warm-starts from the previous run's `best.pt`** so the model keeps what it
already learned, and it snapshots the dataset + metrics per run. On the very
first run it starts from the pretrained `yolo11n-seg.pt` automatically.

```bash
# Day 1 (auto-starts from pretrained yolo11n-seg.pt)
python train_daily.py --data dataset --epochs 100

# Day 2..N — add more images to dataset/ first, then just run the same command.
# It automatically warm-starts from the latest best.pt.
python train_daily.py --data dataset --epochs 100
```

Use `--device 0` on a GPU. Model sizes are set via `--model`:
`yolo11n/s/m/l/x-seg.pt` (n = fastest/edge, s = good default, m/l = higher
accuracy but slower).

Other flags: `--fresh` (ignore history, start from pretrained),
`--init path/to/best.pt` (explicit checkpoint), `--tag label`.

> ⚠️ Always keep **all** accumulated images in `dataset/`, not just each day's
> new ones — training on new data alone makes the model forget older defects
> (catastrophic forgetting).

Outputs:
- `runs_daily/day_<timestamp>/weights/best.pt` — trained weights
- `runs_daily/day_<timestamp>/dataset_snapshot/` — labels + counts for that run
- `model_registry/registry.json` — every run → weights → metrics → what it started from

---

## 5. Data quality > everything

Instance segmentation needs enough examples per class:

| Need                | Target                          |
|---------------------|---------------------------------|
| Images per class    | 50–100+                         |
| Total dataset       | 300–500+ images                 |
| Validation split    | balanced across ALL classes     |
| Hardware            | a GPU for fast 100+ epoch runs  |

A handful of images per class will train end-to-end but produce near-zero mAP —
the bottleneck is annotated data, not the code.

---

## 6. Files

| File                   | Purpose                                                   |
|------------------------|-----------------------------------------------------------|
| `visualize_labels.py`  | Render/validate YOLO-seg annotations                      |
| `train_daily.py`       | YOLO11-seg training (warm-start + dataset/model versioning)|
| `dataset/`             | YOLO-seg dataset (images + polygon labels + data.yaml)    |
| `requirements.txt`     | Python dependencies                                       |
| `docs/setup.sh`        | Dependency installer (CPU/GPU)                            |
| `docs/setup-pod.md`    | Optional: RunPod GPU environment setup                    |
| `docs/PRODUCTION.md`   | Production runbook                                        |
| `samples/`             | Loose sample image + original dataset export (.zip)       |

---

## 7. Next steps toward production

- Run inference on new parts → per-instance mask, class, area/location, PASS/FAIL.
- Trigger frames from a PLC/line-sensor signal instead of polling.
- Emit a reject signal when a defect above threshold is detected.
- Log detections (image + metadata) for traceability/quality reports.
- Export to ONNX/TensorRT for accelerated edge inference on the line.

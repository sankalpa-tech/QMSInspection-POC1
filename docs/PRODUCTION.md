# QMS Defect Inspector - Production Runbook

Self-learning industrial defect inspection with a **zero-token local hot path**
and **Claude escalation on uncertainty**, carrying all accumulated learnings.

## The loop

```
image ──► local kNN model (models/best.pt)
             ├─ RESOLVED   → verdict + annotated masks        (0 LLM tokens)
             └─ UNCERTAIN  → Claude (full learned context)
                             → verdict → teach → repack best.pt (self-improves)
```

- **Known / repeat parts** resolve locally in ms, no tokens, deterministic.
- **Only genuinely-new images** hit Claude - and Claude receives the full context
  pack (signatures, confirmed rulings, severity map, placement gotchas, the
  reject-paint rule, and the nearest confirmed examples), so it never starts cold.
- Every Claude verdict is folded back in, so the UNCERTAIN rate keeps dropping.

## Install

```powershell
pip install -r requirements.txt
```

## Everyday commands (single entry point: qms.py)

```powershell
# Inspect one image or a folder, locally (0 tokens)
python qms.py inspect <image|dir>

# Inspect and auto-escalate UNCERTAIN images to Claude, then learn from them
$env:ANTHROPIC_API_KEY = "sk-ant-..."
python qms.py inspect <image|dir> --escalate

# Teach a verdict manually (image + verdict JSON in the required schema)
python qms.py teach <image> --json verdict.json --part cup-collar

# Repack models/best.pt from the learning DB (after any teaching)
python qms.py build

# Inspect what Claude would be told for an image (debug the context pack)
python qms.py context <image>

# Model + DB summary
python qms.py stats

# REST API (see endpoints below)
python qms.py serve --port 8000
```

## Output

For every image, `inspect_out/<name>.json` + `<name>_annotated.jpg`:

```json
{
  "result": "DEFECT" | "OK" | "UNCERTAIN",
  "defects": [
    {"type": "...", "confidence": 0-100, "location": "...",
     "reason": "...", "severity_priority": 1-5, "primary": true}
  ]
}
```

## Configuration (env vars, all optional)

| Var | Purpose | Default |
|-----|---------|---------|
| `QMS_MODEL_PATH` | packed checkpoint | `models/best.pt` |
| `QMS_OUT_DIR` | annotated outputs | `inspect_out/` |
| `QMS_DB_PATH` | learning DB | `learning.db` |
| `QMS_REVIEW_PATH` | review geometry | `aes2_review.json` |
| `QMS_LOG_LEVEL` | logging | `INFO` |
| `ANTHROPIC_API_KEY` | enables escalation | (unset = escalation off) |
| `QMS_CLAUDE_MODEL` | Claude model id | `claude-opus-4-20250514` |
| `QMS_PHASH_RECALL_MAX` | recall strictness (lower = stricter) | `6` |

## What's inside models/best.pt

One portable file with **all learnings**: exemplar feature vectors + perceptual
hashes (kNN memory), normalization + centroids + thresholds, exact per-image
defect geometry (masks/bboxes + severity), and the full knowledge base
(signatures, rulings, severity map, gotchas). Runs standalone - verified to work
even with `learning.db` absent.

> Note: this is a **classical kNN / prototype model** in a `.pt`, not a deep CNN.
> It is the correct choice for the current small dataset (~4-6 images/defect).

## REST API

```
GET  /health    service status + counts
POST /inspect   multipart 'image' (or raw body, or JSON {"path": "..."}) -> verdict
POST /learn     multipart 'image' + 'label' (or JSON) -> learns at runtime
GET  /stats     counts + recent inspections
GET  /parts     learned parts + their categories
```

## Tests

```powershell
python test_smoke.py         # 4 fast checks, no network
```

## Retraining / graduating to YOLO

The active-learning loop (`teach` -> `build`) grows `learning.db`. When you have
~50-200+ labeled images per defect, train a real YOLO-seg model:

```powershell
python train_daily.py --data dataset --epochs 100   # warm-starts from last best.pt
```

Labels are already in YOLO-seg format, so the swap is drop-in. Until then the
kNN `best.pt` + Claude escalation is the recommended path.

## Files

| File                     | Role |
|--------------------------|------|
| `qms.py`                 | unified CLI (inspect/build/teach/context/serve/stats) |
| `config.py`              | env-driven paths + logging |
| `run_pt.py`              | load best.pt, local kNN inference + rendering |
| `r.py`                   | pack DB + knowledge into best.pt |
| `context_pack.py`        | assemble all learnings for Claude |
| `escalate.py`            | send image + context to Claude, parse verdict |
| `teach_loop.py`          | fold a verdict back in, repack best.pt |
| `teach_cups.py`          | bulk-teach from aes2_review.json |
| `annotate_aes2.py`       | mask/label rendering + severity priority |
| `defect_inspector/`      | feature extraction, kb, learn, REST api |
| `knowledge/lessons.json` | durable rulings + gotchas (source of truth) |

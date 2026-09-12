# Self-Learning Defect Inspector (`defect_inspector/`)

A persistent, self-improving inspection layer for the 3-hole plated link part.
It gives a **fast first-pass verdict** from a learned cache and **keeps learning**
from your feedback, so repeated/known parts are judged instantly and the knowledge
base grows over time.

## The six categories
`black mark after electroplating` · `electroplating defect` · `incomplete embossing`
· `line defect` · `serration` · `OK`

## How it works (honest design)
On this part the defects are subtle and the labeled set is tiny (19 images), so
**hand-crafted CV features cannot classify them reliably on their own**
(leave-one-out fast-path accuracy ≈ 40-50%; features like line/dark scores overlap
across all classes). The cache is therefore built around **recall + retrieval to
assist visual reasoning**, not a fake autonomous classifier:

1. **Perceptual-hash recall** — if a new image is a near-duplicate (dHash Hamming
   ≤ 6) of a previously *confirmed* image, its verdict is returned instantly with
   high confidence. This is the real "faster result" for repeated/known parts.
2. **Nearest-neighbour retrieval** — otherwise the tool surfaces the most similar
   confirmed examples (kNN over normalised features) as a *hint* and marks the case
   **NEEDS VISUAL REVIEW** (result `UNCERTAIN`) so a human/Claude makes the call.
3. **Learning loop** — every confirmation/correction is stored as a new exemplar and
   the cache (`knowledge/defect_kb.json`) is rebuilt, so next time is faster/better.

## Files
| File | Purpose |
|------|---------|
| `features.py` | Fast OpenCV feature vector + dHash perceptual hash |
| `kb.py` | SQLite learning DB + JSON knowledge cache (load/save/rebuild) |
| `inspect.py` | Inspect an image/folder → required JSON verdict |
| `learn.py` | Confirm/correct a label → updates the cache |
| `loo_eval.py` | Leave-one-out check of the autonomous fast path (transparency) |

Artifacts: `learning.db` (durable memory) and `knowledge/defect_kb.json` (the cache
file that is updated as the system learns).

## Usage
```bash
# 1. seed / teach exemplars (top-level tools)
python teach_cups.py                 # bulk-teach known verdicts
python qms.py teach img.jpg --json verdict.json --part <part>

# 2. inspect a new image (or a whole folder)
python -m defect_inspector.inspect path\to\image.jpg

# 3. teach it (grows the cache -> faster/better next time)
python -m defect_inspector.learn add     path\to\image.jpg "line defect"
python -m defect_inspector.learn correct path\to\image.jpg "serration" optional note

# transparency: how good is the autonomous fast path today?
python -m defect_inspector.loo_eval
```

Output is exactly the required schema:
```json
{ "result": "DEFECT|OK|UNCERTAIN",
  "defects": [ { "type": "...", "confidence": 0-100, "location": "...", "reason": "..." } ] }
```

## Recommended operating mode
Fast path auto-passes only **confident phash recalls**; everything else is routed to
visual review. As you confirm more parts, recall coverage rises and manual review
drops. To make the *autonomous* path accurate (no human in loop) you need far more
labeled data per class and/or a trained CNN/YOLO-seg model (the repo's `train_daily.py`).

Requires: Python 3.11, `opencv-python`, `numpy` (already installed).

## REST API (`api.py`)
Run the service:
```bash
python -m defect_inspector.api            # http://127.0.0.1:8000
# set INSPECTOR_PORT to change the port
```

| Method & path | Body | Returns |
|---|---|---|
| `GET /health` | - | status + cache counts |
| `POST /inspect` | multipart `image` file | verdict JSON + `status` (RESOLVED\|TEACH_NEEDED) + `hint` + `image_path` |
| `POST /learn` | multipart `image`+`label`, or JSON `{"path","label"}` | `LEARNED` + updated counts |
| `GET /stats` | - | counts + recent inspections |

**Policy:** only a confident perceptual-hash **recall** counts as a resolved
result (`status=RESOLVED`). Anything else returns `result=UNCERTAIN`,
`status=TEACH_NEEDED` with a suspected label + nearest-neighbour hint, so the
caller (or Claude) supplies the true label via `/learn`, which updates the cache
live. Next inspection of that part returns `RESOLVED` instantly.

Examples:
```bash
# inspect
curl -s -X POST http://127.0.0.1:8000/inspect -F "image=@part.jpg"

# teach (re-uses the server path returned by /inspect)
curl -s -X POST http://127.0.0.1:8000/learn \
     -H "Content-Type: application/json" \
     -d "{\"path\":\"...\\uploads\\123_part.jpg\",\"label\":\"serration\"}"

# or teach by uploading the file + label
curl -s -X POST http://127.0.0.1:8000/learn -F "image=@part.jpg" -F "label=line defect"
```


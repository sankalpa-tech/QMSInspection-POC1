# Offline Annotation — Existing Parts (0 Claude Tokens)

Annotate images of **already-taught parts** (cup-collar, link-3hole) fully offline.
No `ANTHROPIC_API_KEY`, no Claude, no tokens. The local kNN model in
`models/best.pt` recognises the images and renders masks + verdicts.

> Use this only for **existing/known parts**. Genuinely new parts would report
> `UNCERTAIN` and require escalation (a token) — out of scope here.

---

## Prerequisites (one-time)

1. Python 3.11 with deps installed:
   ```powershell
   cd C:\workspace-ai\QMSInspection-POC
   pip install -r requirements.txt
   ```
2. The `defect_inspector\` package must exist in the repo root
   (contains `features.py`, `kb.py`, `learn.py`, `api.py`, ...).
3. Learnings present: `aes2_review.json`, `knowledge\lessons.json`.

---

## One-time setup — build the local model

Seed the local memory from the confirmed reviews, then pack `models/best.pt`:

```powershell
cd C:\workspace-ai\QMSInspection-POC

# 1. Load all confirmed exemplars from aes2_review.json into the learning DB
python teach_cups.py

# 2. Pack everything (exemplars + geometry + lessons) into models\best.pt
python qms.py build
```

Expected after build:
```
exemplars : 21  ({'cup-collar': 18, 'link-3hole': 3})
geometry  : 21 images
lessons   : yes
```

Verify:
```powershell
python qms.py stats
```

You only repeat this if `aes2_review.json` / lessons change.

---

## Everyday — annotate a folder (offline, 0 tokens)

```powershell
cd C:\workspace-ai\QMSInspection-POC

# choose where annotated output goes
$env:QMS_OUT_DIR = "C:\workspace-ai\AES2-20260830T112553Z-1-001\ring_annotated"

# annotate every image in the folder (or pass a single image path)
python qms.py inspect "C:\workspace-ai\AES2-20260830T112553Z-1-001\ring"
```

Each resolved image prints:
```
STATUS: RESOLVED (recall 'IMG...' hamming=0, conf=96) -> 0 LLM tokens
```
`RESOLVED ... 0 LLM tokens` = handled fully offline. 

---

## Output

In `$env:QMS_OUT_DIR` you get, per image:

| File                     | Content                                   |
|--------------------------|-------------------------------------------|
| `<name>_annotated.jpg`   | image with defect masks + labels          |
| `<name>.json`            | verdict: `DEFECT` / `OK`, defects, severity |

Verdict JSON shape:
```json
{
  "result": "DEFECT",
  "defects": [
    {"type": "...", "confidence": 96, "location": "...",
     "severity_priority": 5, "primary": true}
  ]
}
```

---

## If an image comes back UNCERTAIN

`STATUS: TEACH_NEEDED` means the local model doesn't recognise it (a new/unseen
part). To keep it offline, teach it once, then rebuild — no tokens:

```powershell
python qms.py teach <image> --json verdict.json --part cup-collar
python qms.py build
```

(`verdict.json` = the correct verdict in the schema shown above.)

---

## Quick reference

| Step            | Command                                                    |
|-----------------|------------------------------------------------------------|
| Seed memory     | `python teach_cups.py`                                     |
| Build model     | `python qms.py build`                                      |
| Check model     | `python qms.py stats`                                      |
| Annotate folder | `python qms.py inspect <dir>`  (set `$env:QMS_OUT_DIR`)    |
| Teach one image | `python qms.py teach <img> --json verdict.json --part X`  |

All commands above are **offline / 0 Claude tokens** as long as images match
already-taught parts.

# QMS Defect Inspector — Command Reference

All commands run from the project root:

```powershell
cd c:\workspace-ai\QMSInspection-POC
```

The single entry point is `qms.py`. Run `python qms.py -h` to see all subcommands.

---

## 1. Setup (once)

```powershell
python -m pip install -r requirements.txt
```

Requires Python 3.9–3.11.

---

## 2. Inspect an image — local, zero tokens

```powershell
# single image
python qms.py inspect samples\IMG20260824163734.jpg

# a whole folder of images
python qms.py inspect C:\path\to\images

# save the annotated output to a chosen folder
python qms.py inspect myimage.jpg --out results\
```

- Known / repeat parts resolve **locally** → `RESOLVED`, **no API cost**.
- Genuinely novel images return `UNCERTAIN` (escalate them, see below).
- Annotated images are written to `inspect_out\` by default (or `--out`).

---

## 3. Inspect + escalate hard cases to Claude (uses tokens)

```powershell
# set the API key for this PowerShell session
$env:ANTHROPIC_API_KEY = "sk-ant-..."

python qms.py inspect myimage.jpg --escalate
```

- `--escalate` sends **only `UNCERTAIN`** images to Claude, then auto-teaches the
  result back into the learning DB and repacks `best.pt`.
- Add `--no-teach` to get Claude's verdict without folding it back in.

---

## 4. Teach a verdict manually

```powershell
python qms.py teach myimage.jpg --json verdict.json --part cup-collar
```

`verdict.json` holds the corrected defect verdict for that image.
`--part` defaults to `cup-collar` (use `link-3hole` for the plated link).

---

## 5. Rebuild the packed model

```powershell
python qms.py build          # rewrites models\best.pt from the learning DB
```

Run this after teaching new verdicts so `best.pt` reflects the latest learnings.

---

## 6. Run the REST API server

```powershell
python qms.py serve --port 8000
```

In another window:

```powershell
curl.exe -X POST -F "image=@myimage.jpg" http://localhost:8000/inspect
```

---

## 7. Status, context, tests

```powershell
# exemplar counts, feedback rows, model path, Claude on/off
python qms.py stats

# show exactly what Claude would be told about an image
python qms.py context myimage.jpg

# smoke tests (expect: 4/4 passed)
python test_smoke.py
```

---

## Typical daily loop

1. `python qms.py inspect <image>`
2. If `UNCERTAIN` → `python qms.py inspect <image> --escalate` (or `teach` manually)
3. `python qms.py build`
4. Done — future runs of the same/similar part resolve locally at zero cost.

---

## Optional environment variables

| Variable            | Default                     | Purpose                          |
|---------------------|-----------------------------|----------------------------------|
| `ANTHROPIC_API_KEY` | *(unset — escalation off)*  | Enables Claude escalation        |
| `QMS_CLAUDE_MODEL`  | `claude-opus-4-20250514`    | Claude model id                  |
| `QMS_MODEL_PATH`    | `models\best.pt`            | Packed model location            |
| `QMS_DB_PATH`       | `learning.db`               | Learning database                |
| `QMS_OUT_DIR`       | `inspect_out\`              | Annotated output folder          |
| `QMS_PHASH_RECALL_MAX` | `6`                      | Local-recall strictness (lower = stricter) |

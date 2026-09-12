# Verdict Prompt — get a teach-ready JSON from Claude

Use this when a **new part image** arrives and you want a ground-truth verdict to
teach the system with. Open Claude (web/app), **attach the part image**, paste the
prompt below, then save the reply as `verdict.json`.

---

## The prompt (copy everything in the box, attach the image)

```
You are an industrial defect-inspection annotator. I will give you ONE image of a
manufactured metal part. Inspect it and return ONLY a JSON verdict — no prose, no
markdown fences.

Allowed defect categories (use these exact strings):
- "Black mark after electroplating"
- "Electroplating defect"
- "Incomplete embossing"
- "Line defect"
- "Serration"
(If a real defect clearly fits none of these, use a short lowercase label.)

Rules:
1. Do NOT assume a defect exists. If the part looks clean, return OK.
2. Reject/marker paint (red/pink) is the operator's POINTER to a problem, not the
   defect itself. Find the actual cut / notch / dent / scratch / mark next to it and
   annotate THAT as the primary defect; the paint is at most a low-severity secondary.
3. List the most severe/real defect FIRST.
4. Distinguish true defects from reflections, shadows, lighting, and normal texture.
5. For each defect give a bbox as [x, y, w, h], NORMALIZED 0-1, top-left origin,
   tightly around the defect only (small box — do not include the whole rim/part).
6. If genuinely ambiguous, set "result": "UNCERTAIN" and explain in the reason.

Output EXACTLY this shape:

{
  "result": "DEFECT" | "OK" | "UNCERTAIN",
  "defects": [
    {
      "type": "<category>",
      "confidence": <0-100>,
      "location": "<plain-words location, e.g. 'upper-right flange'>",
      "reason": "<visual evidence you saw>",
      "bbox": [<x>, <y>, <w>, <h>]
    }
  ]
}

If no defect: {"result": "OK", "defects": []}
```

---

## Then teach it

Save Claude's JSON reply as `verdict.json` and run:

```powershell
python qms.py teach myimage.jpg --json verdict.json --part bracket-x
python qms.py build
python qms.py stats     # confirm the new part/exemplar count went up
```

Use a consistent `--part` name for each new part type (e.g. `bracket-x`,
`cup-collar`, `link-3hole`).

---

## Field notes

| Field        | Required? | Notes                                                        |
|--------------|-----------|--------------------------------------------------------------|
| `result`     | yes       | `DEFECT` \| `OK` \| `UNCERTAIN`                               |
| `type`       | yes*      | Exact category string; first defect = primary label          |
| `location`   | yes*      | Plain words, e.g. "upper-right flange"                        |
| `reason`     | yes*      | Visual evidence                                              |
| `bbox`       | yes*      | `[x, y, w, h]` normalized 0–1, top-left origin, TIGHT box     |
| `confidence` | optional  | Ignored by the teach step; harmless to keep                  |

\* required only when `result` is `DEFECT` (for `OK`, `defects` is an empty list).

**`bbox` is the field that places the mask correctly.** Estimate as fractions of the
image: a defect in the upper-right, roughly 8% wide ≈ `[0.60, 0.20, 0.08, 0.06]`.

**Reject-paint reminder:** red/pink paint marks *where* a problem is — annotate the
actual cut/notch/dent next to it as the primary defect, not the paint.

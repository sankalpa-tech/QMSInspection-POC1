"""Assemble ALL accumulated learnings into a compact context pack for Claude.

When an image is escalated to Claude (because the local kNN model could not
confidently recall it), Claude must NOT start cold. This module builds a single
text block carrying everything the system has learned so far:

  * defect categories per part
  * per-defect visual signatures
  * confirmed human rulings (e.g. "12 o'clock tab IS a defect", "tilted smooth
    rim = OK")
  * the severity-priority map (P5..P1)
  * segmentation gotchas + the reject-paint rule
  * recurring mistakes to avoid
  * the nearest confirmed exemplars for THIS image (kNN neighbours) and their
    known verdicts, as few-shot grounding

The pack is derived from the packed checkpoint (models/best.pt) when available,
so it always reflects the latest learnings, and never invents labels.
"""
from __future__ import annotations
import json

import numpy as np


REQUIRED_SCHEMA = """{
  "result": "DEFECT" | "OK" | "UNCERTAIN",
  "defects": [
    {"type": "<one of the known categories for this part>",
     "confidence": <0-100>,
     "location": "<approximate image location>",
     "reason": "<visual evidence>",
     "bbox": [x, y, w, h]  // normalized 0-1, tight around the defect
    }
  ]
}"""


def _fmt_list(items, bullet="  - "):
    return "\n".join(bullet + str(i) for i in items) if items else "  (none)"


def nearest_exemplars(model, feature_vec, k=5):
    """Return list of (name, label, distance) nearest confirmed exemplars."""
    E = model["exemplars"]
    X = E["X"].numpy().astype(np.float64) if hasattr(E["X"], "numpy") else np.asarray(E["X"], float)
    if len(X) == 0:
        return []
    mean = model["norm"]["mean"]
    std = model["norm"]["std"]
    mean = mean.numpy() if hasattr(mean, "numpy") else np.asarray(mean, float)
    std = std.numpy() if hasattr(std, "numpy") else np.asarray(std, float)
    std = np.where(std < 1e-6, 1e-6, std)
    keys = model["feature_keys"]
    x = np.array([float(feature_vec.get(kk, 0.0)) for kk in keys], dtype=np.float64)
    Xn = (X - mean) / std
    xn = (x - mean) / std
    d = np.sqrt(((Xn - xn) ** 2).sum(axis=1))
    order = np.argsort(d)[:k]
    return [(E["names"][i], E["labels"][i], float(d[i])) for i in order]


def build(model, feature_vec, part_hint=None, k=5):
    """Return a single string: the full learning context for Claude."""
    lessons = model.get("lessons", {}) or {}
    cats_by_part = model.get("categories_by_part", {})
    sigs = model.get("signatures", {})

    neigh = nearest_exemplars(model, feature_vec, k=k)
    # infer the most likely part from nearest neighbours if not given
    part = part_hint
    if not part and neigh:
        geo = model.get("geometry", {})
        for name, _, _ in neigh:
            p = geo.get(name, {}).get("part")
            if p:
                part = p
                break

    lines = []
    lines.append("You are an industrial defect-inspection expert. Inspect the ONE")
    lines.append("attached image of a manufactured metal part. Use ONLY the known")
    lines.append("defect categories below - never invent a new label. If evidence is")
    lines.append("weak/ambiguous, return UNCERTAIN rather than forcing a class.")
    lines.append("")

    # 1. categories
    lines.append("=== KNOWN DEFECT CATEGORIES (per part) ===")
    if cats_by_part:
        for p, cats in cats_by_part.items():
            mark = "  <-- most likely this part" if p == part else ""
            lines.append(f"[{p}]{mark}")
            lines.append(_fmt_list(cats))
    else:
        lines.append(_fmt_list(list(sigs.keys())))
    lines.append("")

    # 2. per-part learned signatures / rulings
    cup = (lessons.get("parts", {}) or {}).get("cup-collar", {})
    part_lessons = (lessons.get("parts", {}) or {}).get(part, cup)
    if part_lessons.get("signatures"):
        lines.append("=== VISUAL SIGNATURES (learned) ===")
        for name, desc in part_lessons["signatures"].items():
            lines.append(f"  - {name}: {desc}")
        lines.append("")
    elif sigs:
        lines.append("=== VISUAL SIGNATURES (learned) ===")
        for name, desc in sigs.items():
            lines.append(f"  - {name}: {desc}")
        lines.append("")

    if part_lessons.get("confirmed_rulings"):
        lines.append("=== CONFIRMED RULINGS (must follow) ===")
        lines.append(_fmt_list(part_lessons["confirmed_rulings"]))
        lines.append("")

    # 3. severity priority
    sev = part_lessons.get("severity_priority") or _severity_from_rules(model.get("severity_rules"))
    if sev:
        lines.append("=== SEVERITY PRIORITY (rank defects; mark highest as primary) ===")
        for lvl in ("P5", "P4", "P3", "P2", "P1"):
            if lvl in sev:
                lines.append(f"  {lvl} (highest={lvl=='P5'}): {', '.join(sev[lvl])}")
        lines.append("")

    # 4. segmentation gotchas + paint rule
    if lessons.get("segmentation_gotchas"):
        lines.append("=== PLACEMENT GOTCHAS (avoid these mistakes) ===")
        lines.append(_fmt_list(lessons["segmentation_gotchas"]))
        lines.append("")
    if lessons.get("reject_paint_rule"):
        rp = lessons["reject_paint_rule"]
        lines.append("=== REJECT-PAINT RULE ===")
        lines.append("  " + rp.get("principle", ""))
        if rp.get("scan_habit"):
            lines.append("  " + rp["scan_habit"])
        lines.append("")

    # 5. recurring mistakes
    if lessons.get("recurring_mistakes"):
        lines.append("=== RECURRING MISTAKES TO AVOID ===")
        lines.append(_fmt_list(lessons["recurring_mistakes"]))
        lines.append("")

    # 6. few-shot grounding: nearest confirmed examples + their verdicts
    if neigh:
        lines.append("=== NEAREST CONFIRMED EXAMPLES (for grounding) ===")
        geo = model.get("geometry", {})
        for name, label, dist in neigh:
            g = geo.get(name, {})
            dfx = g.get("defects", [])
            summary = "; ".join(d.get("category", "") for d in dfx) or label
            lines.append(f"  - {name} [{g.get('result', '?')}]: {summary} (feat-dist {dist:.1f})")
        lines.append("")

    # 7. output contract
    lines.append("=== SCAN INSTRUCTIONS ===")
    lines.append("Scan the ENTIRE part: full outer edge, inner rim, both flanges and")
    lines.append("faces. Capture EVERY visible defect (not just the first). Rank by the")
    lines.append("severity map; mark the highest-severity defect primary. Give tight")
    lines.append("normalized bboxes on the actual defect, not on paint or reflections.")
    lines.append("")
    lines.append("=== RETURN STRICT JSON ONLY (no prose) ===")
    lines.append(REQUIRED_SCHEMA)

    return "\n".join(lines), part, neigh


def _severity_from_rules(rules):
    if not rules:
        return {}
    out = {}
    for score, kws in rules:
        out[f"P{score}"] = list(kws)
    return out

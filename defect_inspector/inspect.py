"""Fast first-pass defect inspection from the learned cache.

Usage:
    python -m defect_inspector.inspect <image_or_dir> [--quiet]

Combines kNN over stored exemplars with interpretable rule signals to produce
the required JSON verdict, and logs every inspection to learning.db so the
system can learn from feedback.
"""
from __future__ import annotations
import os
import sys
import json
import glob
import numpy as np

from . import kb
from . import features as F


def _load_exemplar_matrix(part=None):
    rows = kb.all_exemplars(confirmed_only=True, part=part)
    labels, vecs, names, phashes = [], [], [], []
    for r in rows:
        v = json.loads(r["features_json"])
        vecs.append([float(v.get(k, 0.0)) for k in F.FEATURE_KEYS])
        labels.append(r["label"])
        names.append(r["name"])
        phashes.append(r["phash"] if "phash" in r.keys() else "")
    return names, labels, np.array(vecs, dtype=np.float64), phashes


def _knn_votes(x, labels, mat, mean, std, k, class_counts=None):
    if len(mat) == 0:
        return {}, []
    xn = (x - mean) / std
    matn = (mat - mean) / std
    d = np.sqrt(((matn - xn) ** 2).sum(axis=1))
    order = np.argsort(d)[:k]
    votes = {}
    neighbors = []
    for i in order:
        w = 1.0 / (d[i] + 1e-6)
        if class_counts:  # down-weight majority classes to counter imbalance
            w /= np.sqrt(class_counts.get(labels[i], 1))
        votes[labels[i]] = votes.get(labels[i], 0.0) + w
        neighbors.append((labels[i], float(d[i])))
    return votes, neighbors


def _rule_signals(vec, signals, th):
    """Return list of (category, strength 0..1, reason, location)."""
    out = []
    region = signals.get("region_of", {})

    lf = vec["max_line_frac"]
    if lf >= th["line_frac_weak"]:
        strong = lf >= th["line_frac_strong"]
        loc = region.get("line", "surface")
        ang = signals["lines"][0]["angle"] if signals.get("lines") else 0
        out.append((
            "line defect",
            min(1.0, lf / th["line_frac_strong"]),
            f"A straight line spanning ~{int(lf*100)}% of the part (angle {ang} deg) "
            f"stands out from the normal brushed texture.",
            loc,
        ))

    ds = vec["dark_score"]
    if ds >= th["dark_score_weak"]:
        loc = region.get("dark", "surface")
        out.append((
            "black mark after electroplating",
            min(1.0, ds / th["dark_score_strong"]),
            f"A localized dark mark (contrast {ds:.0f}) darker than its surroundings "
            f"sits on top of otherwise normal plating.",
            loc,
        ))

    if vec["center_contrast"] <= th["center_contrast_low"]:
        out.append((
            "incomplete embossing",
            min(1.0, (th["center_contrast_low"] - vec["center_contrast"]) / th["center_contrast_low"] + 0.3),
            f"Central relief contrast is low ({vec['center_contrast']:.1f}), consistent with a "
            f"faint / partially missing 'VA' emboss.",
            "center",
        ))

    if vec["hole_rough"] >= th["hole_rough_high"]:
        out.append((
            "serration",
            min(1.0, vec["hole_rough"] / (th["hole_rough_high"] * 1.6)),
            f"A round hole edge shows high roughness ({vec['hole_rough']:.0f}), suggesting a "
            f"jagged/serrated edge rather than a clean bore.",
            "hole edge",
        ))

    # electroplating defect: matte, low gloss, low colorfulness (dull uneven finish)
    if vec["gloss_hi_frac"] < 0.002 and vec["colorfulness"] < 12 and vec["v_std"] > 18:
        out.append((
            "electroplating defect",
            0.5,
            f"Finish looks dull/matte and uneven (low gloss, low colorfulness, brightness "
            f"spread {vec['v_std']:.0f}) - possible non-uniform plating.",
            "surface",
        ))
    return out


def decide(vec, signals, cache, labels, mat, log_note=None):
    """Core decision shared by inspect + evaluation.

    Returns (top, confidence, score, rule_by_cat, neighbors).
    """
    th = cache["thresholds"]
    mean = np.array(cache["norm"]["mean"])
    std = np.array(cache["norm"]["std"])
    counts = cache.get("counts", {})
    x = F.to_array(vec)
    votes, neighbors = _knn_votes(x, labels, mat, mean, std, th["knn_k"], counts)

    # categories are per-part: whatever labels exist for this part, plus OK.
    part_cats = set(labels) | {"OK"}
    score = {c: 0.0 for c in part_cats}
    total_v = sum(votes.values()) or 1.0
    for c, w in votes.items():
        score[c] += 0.6 * (w / total_v)

    # Rule signals reference specific defect names; only apply them if this part
    # actually uses that category (so new parts with new categories aren't polluted).
    rules = _rule_signals(vec, signals or {"region_of": {}, "lines": []}, th)
    rule_by_cat = {}
    for cat, strength, reason, loc in rules:
        if cat not in part_cats:
            continue
        score[cat] = score.get(cat, 0.0) + 0.6 * strength
        if strength > rule_by_cat.get(cat, (0,))[0]:
            rule_by_cat[cat] = (strength, reason, loc)

    if not score:
        return "OK", 1, {}, {}, neighbors
    top = max(score, key=score.get)
    total = sum(score.values()) or 1.0
    confidence = max(1, min(99, int(round(100 * score[top] / total))))
    return top, confidence, score, rule_by_cat, neighbors


def inspect_image(path, cache, exemplars, log=True):
    names, labels, mat, phashes = exemplars
    feat = F.extract(path)
    vec = feat["vector"]
    signals = feat["signals"]
    th = cache["thresholds"]

    # --- FAST PATH 1: perceptual-hash recall of a previously confirmed image ---
    ph = signals.get("phash", "")
    recall = None
    if ph:
        best_h, best_i = 999, -1
        for i, h in enumerate(phashes):
            if not h:
                continue
            dist = F.hamming(ph, h)
            if dist < best_h:
                best_h, best_i = dist, i
        if best_i >= 0 and best_h <= th.get("phash_recall_max", 6):
            recall = {"name": names[best_i], "label": labels[best_i], "hamming": best_h}

    top, confidence, score, rule_by_cat, neighbors = decide(vec, signals, cache, labels, mat)

    # nearest known examples (retrieval hint for visual review)
    mean = np.array(cache["norm"]["mean"]); std = np.array(cache["norm"]["std"])
    xn = (F.to_array(vec) - mean) / std
    dists = np.sqrt((((mat - mean) / std - xn) ** 2).sum(axis=1)) if len(mat) else np.array([])
    hint = []
    for i in np.argsort(dists)[:3] if len(dists) else []:
        hint.append({"name": names[i], "label": labels[i], "dist": round(float(dists[i]), 2)})

    if recall is not None:
        # trusted instant recall
        top = recall["label"]
        confidence = 96 if recall["hamming"] == 0 else 88
        result = "OK" if top == "OK" else "DEFECT"
        loc = signals.get("region_of", {}).get("line") or "surface"
        defects = [] if top == "OK" else [{
            "type": top, "confidence": confidence, "location": loc,
            "reason": f"Near-duplicate of confirmed sample '{recall['name']}' "
                      f"(phash hamming {recall['hamming']}); recalled from learning cache."}]
        needs_review = False
    else:
        defects = []
        if top != "OK":
            if top in rule_by_cat:
                _, reason, loc = rule_by_cat[top]
            else:
                reason = cache["signatures"].get(top, "")
                loc = signals.get("region_of", {}).get("line") or "surface"
            defects.append({"type": top, "confidence": confidence, "location": loc,
                            "reason": "SUSPECTED (not confirmed): " + reason})
        # HONEST POLICY: hand-crafted features are not reliable on this part, so
        # ONLY a confident perceptual-hash recall counts as a resolved result.
        # Anything not recalled is returned as UNCERTAIN and routed for teaching.
        needs_review = True
        result = "UNCERTAIN"

    verdict = {"result": result, "defects": defects if result != "OK" else []}

    if log:
        kb.log_inspection(os.path.basename(path), path, top, result, confidence, verdict["defects"], vec)

    return {"verdict": verdict, "top": top, "confidence": confidence,
            "needs_review": (recall is None) and needs_review,
            "recall": recall, "hint": hint,
            "neighbors": neighbors, "vector": vec, "signals": signals}


def main(argv):
    if not argv:
        print("usage: python -m defect_inspector.inspect <image_or_dir>")
        return 1
    target = argv[0]
    quiet = "--quiet" in argv
    cache = kb.load_cache()
    exemplars = _load_exemplar_matrix()

    paths = []
    if os.path.isdir(target):
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            paths += glob.glob(os.path.join(target, ext))
    else:
        paths = [target]
    paths.sort()

    for p in paths:
        res = inspect_image(p, cache, exemplars)
        print(f"\n=== {os.path.basename(p)} ===")
        print(json.dumps(res["verdict"], indent=2))
        if res["recall"]:
            print(f"STATUS: RESOLVED (cached '{res['recall']['name']}' "
                  f"{res['recall']['label']}, hamming={res['recall']['hamming']}) -> no teaching")
        elif res["needs_review"]:
            hint = ", ".join(f"{h['label']}({h['dist']})" for h in res["hint"])
            print(f"STATUS: TEACH_NEEDED (conf {res['confidence']}). "
                  f"nearest known: {hint}")
        else:
            print(f"STATUS: RESOLVED (rule/kNN conf {res['confidence']}) -> no teaching")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

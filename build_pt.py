"""Pack ALL learnings into a single portable checkpoint: models/best.pt

This is a classical learned-memory model (kNN / nearest-prototype over
hand-crafted CV features) + the full knowledge base, NOT a deep CNN. It bundles
everything the inspector has learned so the whole system runs from one file:

  * exemplars  : feature vectors + perceptual hashes + labels + part (kNN memory)
  * norm/centroids/thresholds : the fitted decision parameters
  * geometry   : exact per-image defect masks/bboxes (from aes2_review.json + GT)
  * lessons    : durable rulings, severity map, segmentation gotchas, paint rule
  * meta       : feature order, categories, version, build time

Run it later with:  python run_pt.py <image>   (loads models/best.pt, 0 LLM tokens)
"""
from __future__ import annotations
import os
import json
import glob
import datetime

import numpy as np
import torch

from defect_inspector import kb
from defect_inspector import features as F
import annotate_aes2 as A

ROOT = os.path.dirname(os.path.abspath(__file__))
try:
    import config
    OUT_PT = config.MODEL_PATH
except Exception:
    OUT_PT = os.path.join(ROOT, "models", "best.pt")


def _review():
    p = os.path.join(ROOT, "aes2_review.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def _lessons():
    p = os.path.join(ROOT, "knowledge", "lessons.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def _taxonomy():
    p = os.path.join(ROOT, "knowledge", "taxonomy.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def build():
    kb.init_db()
    cache = kb.rebuild_cache()          # make sure cache + lessons are current
    rows = kb.all_exemplars(confirmed_only=True)

    keys = F.FEATURE_KEYS
    vecs, labels, names, parts, phashes = [], [], [], [], []
    for r in rows:
        v = json.loads(r["features_json"])
        vecs.append([float(v.get(k, 0.0)) for k in keys])
        labels.append(r["label"])
        names.append(r["name"])
        parts.append(r["part"] if r["part"] else "default")
        phashes.append(r["phash"] or "")

    X = torch.tensor(np.array(vecs, dtype=np.float32)) if vecs else torch.zeros((0, len(keys)))
    mean = torch.tensor(np.array(cache["norm"]["mean"], dtype=np.float32))
    std = torch.tensor(np.array(cache["norm"]["std"], dtype=np.float32))

    # geometry: exact learned defect placement per image (from the review file)
    review = _review()
    geometry = {}
    for b, entry in review.items():
        geometry[b] = {"part": entry.get("part", "default"),
                       "result": entry.get("result"),
                       "defects": entry.get("defects", [])}

    ckpt = {
        "format": "qms-defect-knn",
        "version": 3,
        "built_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "feature_keys": keys,
        # kNN memory (the "trained weights" of a nearest-neighbour model)
        "exemplars": {
            "X": X,                     # (N, D) float tensor of features
            "labels": labels,
            "names": names,
            "parts": parts,
            "phash": phashes,
        },
        # fitted decision parameters
        "norm": {"mean": mean, "std": std},
        "centroids": cache.get("centroids", {}),
        "counts": cache.get("counts", {}),
        "thresholds": cache.get("thresholds", {}),
        "categories_by_part": _cats_by_part(labels, parts),
        # human-readable knowledge
        "signatures": cache.get("signatures", {}),
        "confusions": cache.get("confusions", []),
        "lessons": _lessons(),
        "severity_rules": A.SEVERITY_RULES,
        "taxonomy": _taxonomy(),
        # exact geometry so masks render identically from the checkpoint
        "geometry": geometry,
    }

    os.makedirs(os.path.dirname(OUT_PT), exist_ok=True)
    torch.save(ckpt, OUT_PT)
    size_kb = os.path.getsize(OUT_PT) / 1024.0
    print(f"wrote {OUT_PT}  ({size_kb:.1f} KB)")
    print(f"  exemplars : {len(labels)}  ({dict(_count(parts))})")
    print(f"  geometry  : {len(geometry)} images")
    print(f"  lessons   : {'yes' if ckpt['lessons'] else 'no'}  "
          f"(cup signatures={len(ckpt['lessons'].get('parts',{}).get('Bearing Cup',{}).get('signatures',{}))})")
    print(f"  categories: {sum(len(v) for v in ckpt['categories_by_part'].values())} across "
          f"{len(ckpt['categories_by_part'])} parts")
    return OUT_PT


def _cats_by_part(labels, parts):
    out = {}
    for lab, pt in zip(labels, parts):
        out.setdefault(pt, set()).add(lab)
    return {k: sorted(v) for k, v in out.items()}


def _count(xs):
    d = {}
    for x in xs:
        d[x] = d.get(x, 0) + 1
    return d


if __name__ == "__main__":
    build()

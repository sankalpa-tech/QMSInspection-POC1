"""Leave-one-out evaluation of the automated (cache-only) classifier.

For each seeded exemplar, hold it out, rebuild a temporary kNN from the rest, and
predict. This measures real generalization of the FAST path (no Claude in loop),
so we know when the system can auto-pass vs. when it must defer to visual review.
"""
from __future__ import annotations
import json
import numpy as np

from . import kb
from . import features as F
from . import inspect as INS


def run():
    cache = kb.load_cache()
    rows = kb.all_exemplars(confirmed_only=True)
    data = []
    for r in rows:
        v = json.loads(r["features_json"])
        data.append((r["name"], r["label"], v))

    mean = np.array(cache["norm"]["mean"])
    std = np.array(cache["norm"]["std"])
    th = cache["thresholds"]

    n_correct = 0
    n_defer = 0
    per_cat = {}
    confusion = {}
    for i, (name, truth, vec) in enumerate(data):
        labels, mat = [], []
        for j, (_, lab, v) in enumerate(data):
            if j == i:
                continue
            labels.append(lab)
            mat.append([float(v.get(k, 0.0)) for k in F.FEATURE_KEYS])
        mat = np.array(mat, dtype=np.float64)
        # full fast-path decision (kNN + rule signals), holding this sample out
        fold = dict(cache)
        pred, conf, _, _, _ = INS.decide(vec, None, fold, labels, mat)
        deferred = conf < th["uncertain_below"]

        ok = pred == truth
        n_correct += ok
        n_defer += deferred
        d = per_cat.setdefault(truth, [0, 0])
        d[1] += 1
        d[0] += ok
        confusion.setdefault(truth, {}).setdefault(pred, 0)
        confusion[truth][pred] += 1
        flag = "OK " if ok else "XX "
        print(f"{flag}{name:16s} truth={truth:32s} pred={pred:32s}"
              f"{' [defer]' if deferred else ''}")

    total = len(data)
    print(f"\nLOO accuracy (fast path): {n_correct}/{total} = {100*n_correct/total:.0f}%")
    print(f"borderline (auto-defer to review): {n_defer}/{total}")
    print("\nper-category:")
    for c, (ok, tot) in sorted(per_cat.items()):
        print(f"  {c:32s} {ok}/{tot}")
    print("\nconfusions (truth -> pred):")
    for t, preds in confusion.items():
        for p, c in preds.items():
            if p != t:
                print(f"  {t} -> {p}: {c}")


if __name__ == "__main__":
    run()

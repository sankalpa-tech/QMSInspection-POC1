"""Run the packed model: python run_pt.py <image_or_dir> [--model models/best.pt] [--out inspect_out]

Loads models/best.pt (all learnings in one file) and inspects images with ZERO
LLM tokens. For any image that matches a learned part by perceptual hash it emits
the confirmed verdict + the exact annotated masks pulled from the checkpoint.
Genuinely-new images come back UNCERTAIN (need a one-time visual pass).
"""
from __future__ import annotations
import os
import sys
import glob
import json

import numpy as np
import torch
import cv2

from defect_inspector import features as F
import annotate_aes2 as A
import config

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = config.MODEL_PATH
DEFAULT_OUT = config.OUT_DIR


def load_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    E = ckpt["exemplars"]
    ckpt["_X"] = E["X"].numpy().astype(np.float64)
    ckpt["_mean"] = ckpt["norm"]["mean"].numpy().astype(np.float64)
    ckpt["_std"] = np.where(ckpt["norm"]["std"].numpy() < 1e-6, 1e-6,
                            ckpt["norm"]["std"].numpy()).astype(np.float64)
    return ckpt


def _priority(cat, severity_rules):
    c = (cat or "").lower()
    for score, kws in severity_rules:
        if any(k in c for k in kws):
            return score
    return 2


def render_verdict(path, verdict, m, out_dir):
    """Render a verdict dict (e.g. from Claude) onto the image and save it.

    Converts verdict defects (type/bbox/location/reason) into annotate() dets so
    the escalated result gets the same masks/labels/priority styling. Returns the
    annotated image path.
    """
    img = cv2.imread(path)
    base = os.path.splitext(os.path.basename(path))[0]
    os.makedirs(out_dir, exist_ok=True)
    result = verdict.get("result", "UNCERTAIN")
    dets = []
    for d in verdict.get("defects", []):
        entry = {"category": d.get("type", "defect"),
                 "location": d.get("location", "see box"),
                 "reason": d.get("reason", "")}
        bbox = d.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            entry["bbox"] = [float(x) for x in bbox]
        else:
            entry["bbox"] = [0.4, 0.4, 0.2, 0.2]  # fallback center box
        dets.append(entry)
    if dets:
        A.annotate(img, dets, uncertain=(result == "UNCERTAIN"))
    elif result == "OK":
        A.draw_ok_banner(img)
    else:
        A.draw_label(img, 15, 45, "UNCERTAIN - needs review", (0, 140, 255))
    out_img = os.path.join(out_dir, base + "_annotated.jpg")
    cv2.imwrite(out_img, img)
    with open(os.path.join(out_dir, base + ".json"), "w") as fh:
        json.dump(verdict, fh, indent=2)
    return out_img


def inspect(path, m, out_dir):
    feat = F.extract(path)
    vec, signals = feat["vector"], feat["signals"]
    keys = m["feature_keys"]
    x = np.array([float(vec.get(k, 0.0)) for k in keys], dtype=np.float64)

    E = m["exemplars"]
    phashes, names, labels = E["phash"], E["names"], E["labels"]
    th = m["thresholds"]

    # zero-token perceptual-hash recall
    ph = signals.get("phash", "")
    best_h, best_i = 999, -1
    for i, h in enumerate(phashes):
        if h:
            d = F.hamming(ph, h)
            if d < best_h:
                best_h, best_i = d, i
    recalled = best_i >= 0 and best_h <= th.get("phash_recall_max", 6)

    img = cv2.imread(path)
    base = os.path.splitext(os.path.basename(path))[0]
    os.makedirs(out_dir, exist_ok=True)

    if recalled:
        ref = names[best_i]
        geo = m["geometry"].get(ref, {"defects": [], "result": labels[best_i]})
        dets = geo.get("defects", [])
        result = geo.get("result") or ("DEFECT" if dets else "OK")
        conf = 96 if best_h == 0 else 88
        if dets:
            A.annotate(img, dets, uncertain=(result == "UNCERTAIN"))
        elif result == "OK":
            A.draw_ok_banner(img)
        sdets = sorted(dets, key=lambda d: _priority(d.get("category", ""), m["severity_rules"]), reverse=True)
        verdict = {"result": result, "defects": [] if result == "OK" else [{
            "type": d["category"], "confidence": conf,
            "location": d.get("location", "see box"), "reason": d.get("reason", ""),
            "severity_priority": _priority(d["category"], m["severity_rules"]),
            "primary": (i == 0)} for i, d in enumerate(sdets)]}
        status = f"RESOLVED (recall '{ref}' hamming={best_h}, conf={conf}) -> 0 LLM tokens"
    else:
        # nearest-prototype hint (kNN), still local
        Xn = (m["_X"] - m["_mean"]) / m["_std"]
        xn = (x - m["_mean"]) / m["_std"]
        dists = np.sqrt(((Xn - xn) ** 2).sum(axis=1)) if len(Xn) else np.array([])
        hint = ", ".join(f"{labels[i]}({dists[i]:.1f})" for i in np.argsort(dists)[:3]) if len(dists) else ""
        verdict = {"result": "UNCERTAIN", "defects": [], "hint": hint}
        A.draw_label(img, 15, 45, "UNCERTAIN - needs review", (0, 140, 255))
        cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), (0, 140, 255), 6)
        status = f"TEACH_NEEDED (no recall; nearest: {hint})"

    out_img = os.path.join(out_dir, base + "_annotated.jpg")
    cv2.imwrite(out_img, img)
    with open(os.path.join(out_dir, base + ".json"), "w") as fh:
        json.dump(verdict, fh, indent=2)
    return verdict, status, out_img


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 1
    target = argv[0]
    model = argv[argv.index("--model") + 1] if "--model" in argv else DEFAULT_MODEL
    out_dir = argv[argv.index("--out") + 1] if "--out" in argv else DEFAULT_OUT

    if not os.path.exists(model):
        print(f"model not found: {model}\nbuild it first:  python build_pt.py")
        return 2
    m = load_model(model)
    print(f"loaded {os.path.basename(model)}  (built {m.get('built_at')}, "
          f"{len(m['exemplars']['labels'])} exemplars, {len(m['geometry'])} geometries)")

    paths = []
    if os.path.isdir(target):
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG"):
            paths += glob.glob(os.path.join(target, ext))
    else:
        paths = [target]
    for p in sorted(set(paths)):
        verdict, status, out_img = inspect(p, m, out_dir)
        print(f"\n=== {os.path.basename(p)} ===")
        print(json.dumps({k: v for k, v in verdict.items() if k != "hint"}, indent=2))
        print("STATUS:", status)
        print("annotated:", out_img)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

"""Close the active-learning loop: fold a confirmed verdict back into the model.

Given an image + a verdict (from Claude escalation or a human), this:
  1. writes/updates the per-image geometry in the review file,
  2. adds the image as a confirmed exemplar in learning.db (features + phash),
  3. rebuilds knowledge/defect_kb.json and repacks models/best.pt.

After this, the same (or a near-duplicate) image resolves locally with ZERO
tokens next time. Idempotent per image name.
"""
from __future__ import annotations
import json
import os

import config

log = config.get_logger("teach")


def _load_review():
    if os.path.exists(config.REVIEW_PATH):
        with open(config.REVIEW_PATH) as fh:
            return json.load(fh)
    return {}


def _save_review(review):
    with open(config.REVIEW_PATH, "w") as fh:
        json.dump(review, fh, indent=2)


def _primary_label(verdict):
    defs = verdict.get("defects", [])
    if verdict.get("result") == "OK" or not defs:
        return "OK"
    # trust the verdict's own ordering; first defect is primary
    return defs[0].get("type", "defect")


def teach(image_path, verdict, part="cup-collar", source="claude", rebuild=True):
    """Persist a verdict as a confirmed learning. Returns the primary label."""
    import re
    base = os.path.splitext(os.path.basename(image_path))[0]
    m = re.match(r"(IMG\d+)", base)
    key = m.group(1) if m else base

    # 1. review geometry (convert verdict defects -> review defect entries)
    review = _load_review()
    dets = []
    for d in verdict.get("defects", []):
        entry = {"category": d.get("type", "defect"),
                 "location": d.get("location", "see box"),
                 "reason": d.get("reason", "")}
        if isinstance(d.get("bbox"), (list, tuple)) and len(d["bbox"]) == 4:
            entry["bbox"] = [float(x) for x in d["bbox"]]
        dets.append(entry)
    review[key] = {"part": part, "result": verdict.get("result", "DEFECT"), "defects": dets}
    _save_review(review)
    log.info("review updated: %s -> %s (%d defects)", key, review[key]["result"], len(dets))

    # 2. confirmed exemplar in learning.db
    from defect_inspector import kb
    from defect_inspector import features as F
    kb.init_db()
    feat = F.extract(image_path)
    label = _primary_label(verdict)
    # replace any prior exemplar for this name+part to stay idempotent
    con = kb.connect()
    con.execute("DELETE FROM exemplars WHERE name=? AND part=?", (key, part))
    con.commit()
    con.close()
    kb.add_exemplar(key, image_path, label, feat["vector"],
                    phash=feat["signals"].get("phash", ""),
                    source=source, confirmed=1, part=part)
    log.info("exemplar added: %s -> %s [part=%s]", key, label, part)

    # 3. rebuild cache + repack best.pt
    if rebuild:
        kb.rebuild_cache()
        try:
            import build_pt
            build_pt.build()
            log.info("best.pt repacked")
        except Exception as e:  # noqa
            log.warning("best.pt repack skipped: %s", e)
    return label

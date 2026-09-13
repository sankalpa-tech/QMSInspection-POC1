"""Teach Bracket exemplars from the YOLO-seg dataset (dataset/train, dataset/valid).

Every annotated dataset image becomes a confirmed 'Bracket' exemplar: features +
perceptual hash for near-duplicate recall / kNN memory. The primary label is the
highest-severity annotated defect (canonicalised to knowledge/taxonomy.json), or
OK when the image's label file is empty. Then the knowledge cache is rebuilt.

Idempotent: an image already present as a Bracket exemplar is skipped.

Run:  python teach_bracket.py   (then: python qms.py build)
"""
import os
import glob
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from defect_inspector import kb
from defect_inspector import features as F
import annotate_aes2 as A
import taxonomy as TAX

PART = "Bracket"
IMG_DIRS = [r"dataset\train\images", r"dataset\valid\images"]


def primary_label(dets):
    """dets: list of (class_name, points). Return highest-severity canonical name, or OK."""
    if not dets:
        return "OK"
    ranked = sorted(dets, key=lambda nd: TAX.priority(nd[0]), reverse=True)
    return TAX.canonical(ranked[0][0])


def existing_names(part):
    con = kb.connect()
    rows = con.execute("SELECT name FROM exemplars WHERE part=?", (part,)).fetchall()
    con.close()
    return {r["name"] for r in rows}


def main():
    kb.init_db()
    gt = A.load_gt()  # {base: [(class_name, points), ...]}  ([] for OK)

    # Guard: every annotated class must be known in the central taxonomy.
    all_cats = [n for dets in gt.values() for n, _ in dets]
    ok, unknown = TAX.validate(all_cats)
    if not ok:
        print("ERROR: dataset classes not in knowledge/taxonomy.json:", sorted(set(unknown)))
        sys.exit(1)

    have = existing_names(PART)
    added, skipped = 0, 0
    for d in IMG_DIRS:
        for path in sorted(glob.glob(os.path.join(d, "*.jpg"))):
            base = A.base(path)
            if base in have:
                skipped += 1
                continue
            dets = gt.get(base, [])
            label = primary_label(dets)
            feat = F.extract(path)
            kb.add_exemplar(base, path, label, feat["vector"],
                            phash=feat["signals"].get("phash", ""),
                            source="dataset", confirmed=1, part=PART)
            have.add(base)
            added += 1
            print(f"  + {PART} {base} -> {label} ({len(dets)} defect(s))")

    cache = kb.rebuild_cache()
    print(f"\nexemplars: +{added} added, {skipped} already present")
    print("Bracket categories learned:", cache.get("parts", {}).get(PART, {}))


if __name__ == "__main__":
    main()

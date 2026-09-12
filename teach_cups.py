"""Teach the learning DB + cache from the confirmed cup-collar verdicts.

Loads every entry in aes2_review.json into the exemplars table (with features +
phash for near-duplicate recall and kNN), records the human corrections as
feedback rows, then rebuilds knowledge/defect_kb.json (which now also carries
knowledge/lessons.json). Idempotent: an image already present as an exemplar for
its part is skipped.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from defect_inspector import kb
from defect_inspector import features as F
import taxonomy as TAX

AES2_DIR = r"C:\workspace-ai\AES2-20260830T112553Z-1-001\AES2"
REVIEW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aes2_review.json")


def priority(cat):
    return TAX.priority(cat)


def primary_label(entry):
    if entry.get("result") == "OK" or not entry.get("defects"):
        return "OK"
    dets = sorted(entry["defects"], key=lambda d: priority(d.get("category", "")), reverse=True)
    return dets[0]["category"]


# Corrections the user taught me (image -> the defect I originally MISSED/mis-marked).
CORRECTIONS = {
    "IMG20260824162142": ("reject paint mark on bottom flange",
                          "Stain",
                          "Missed the dark boss mark on first pass; capture BOTH defects."),
    "IMG20260824162425": ("reject paint mark / rough inner-bore edge",
                          "Cut / Chip",
                          "Missed the top-flange cut (same defect type as 162340); scan whole edge."),
    "IMG20260824161920": ("reject paint mark on inner wall",
                          "Deformation",
                          "Missed the pinched bore; the deformation is the primary (higher-severity) defect."),
    "IMG20260824161530": ("notch mask traced whole rim",
                          "Notch",
                          "edge-seg locked onto the bright rim; use a tiny tight box, no edge-seg."),
    "IMG20260824162340": ("boxed red paint + inner rim (edge-seg traced wrong contour)",
                          "Cut / Chip",
                          "Paint is only the reject MARK; the real cut is the step on the outer flange silhouette nearby. Hunt the edge geometry, not the paint."),
}


def existing_names(part):
    con = kb.connect()
    rows = con.execute("SELECT name FROM exemplars WHERE part=?", (part,)).fetchall()
    con.close()
    return {r["name"] for r in rows}


def main():
    kb.init_db()
    review = json.load(open(REVIEW))

    # Central-taxonomy guard: every category in the review must be a known
    # defect defined in knowledge/taxonomy.json.
    all_cats = [d.get("category", "") for e in review.values() for d in e.get("defects", [])]
    ok, unknown = TAX.validate(all_cats)
    if not ok:
        print("ERROR: categories not in knowledge/taxonomy.json:",
              sorted(set(unknown)))
        print("Add them to taxonomy.json (or fix the spelling) and re-run.")
        sys.exit(1)

    added, skipped = 0, 0
    have = {}
    for base, entry in review.items():
        part = entry.get("part", "Bearing Cup")
        have.setdefault(part, existing_names(part))
        if base in have[part]:
            skipped += 1
            continue
        path = os.path.join(AES2_DIR, base + ".jpg")
        if not os.path.exists(path):
            print("MISSING IMAGE:", path)
            continue
        feat = F.extract(path)
        label = primary_label(entry)
        kb.add_exemplar(base, path, label, feat["vector"],
                        phash=feat["signals"].get("phash", ""),
                        source="aes2_review", confirmed=1, part=part)
        have[part].add(base)
        added += 1
        print(f"  + {part:11s} {base} -> {label}")

    # Record the human corrections as an auditable learning trail (idempotent:
    # one feedback row per image; no throwaway inspection rows).
    fb = 0
    con = kb.connect()
    for base, (wrong, true_label, note) in CORRECTIONS.items():
        con.execute("DELETE FROM feedback WHERE note LIKE ?", (f"%[{base}]%",))
        con.execute(
            "INSERT INTO feedback(inspection_id,true_label,note,created_at) VALUES(NULL,?,?,?)",
            (true_label, f"[{base}] (was: {wrong}) {note}", kb._now()),
        )
        fb += 1
    con.commit()
    con.close()

    cache = kb.rebuild_cache()
    print(f"\nexemplars: +{added} added, {skipped} already present")
    print(f"feedback:  +{fb} corrections recorded")
    print(f"parts in cache: {cache.get('parts', {}).keys()}")
    print("Bearing Cup categories learned:",
          json.dumps(cache.get("parts", {}).get("Bearing Cup", {}), indent=2))
    print("lessons embedded:", bool(cache.get("lessons")))


if __name__ == "__main__":
    main()

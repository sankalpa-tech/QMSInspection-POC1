"""Render YOLO-seg polygon labels onto images so you can visually inspect
how each defect instance was annotated.

Usage:
    python visualize_labels.py --dataset dataset --out label_preview
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

COLORS = [
    (230, 25, 75), (60, 180, 75), (67, 99, 216), (245, 130, 49),
    (145, 30, 180), (66, 212, 244), (240, 50, 230), (191, 239, 69),
]


def load_names(dataset: Path) -> list[str]:
    y = yaml.safe_load((dataset / "data.yaml").read_text())
    names = y["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return names


def draw_label(img_path: Path, label_path: Path, names: list[str]):
    img = cv2.imread(str(img_path))
    if img is None:
        return None, []
    h, w = img.shape[:2]
    overlay = img.copy()
    issues = []
    if not label_path.exists():
        return img, ["no label file"]
    for i, line in enumerate(label_path.read_text().splitlines(), 1):
        parts = [p for p in line.split() if p]
        if not parts:
            continue
        cls = int(float(parts[0]))
        coords = [float(x) for x in parts[1:]]
        color = COLORS[cls % len(COLORS)]
        if len(coords) < 6:  # fewer than 3 points -> not a polygon
            issues.append(f"line {i}: only {len(coords)//2} point(s) (not a mask)")
            # draw whatever we can as a marker
            if len(coords) >= 2:
                cx, cy = int(coords[0] * w), int(coords[1] * h)
                cv2.circle(img, (cx, cy), 8, color, 2)
            continue
        pts = np.array(
            [[int(coords[j] * w), int(coords[j + 1] * h)] for j in range(0, len(coords) - 1, 2)],
            dtype=np.int32,
        )
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(img, [pts], True, color, 2)
        label = names[cls] if cls < len(names) else str(cls)
        x0, y0 = pts[0]
        cv2.putText(img, label, (x0, max(0, y0 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
    return img, issues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--out", default="label_preview")
    args = ap.parse_args()

    dataset = Path(args.dataset)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = load_names(dataset)
    print(f"Classes: {names}")

    all_issues = []
    for split in ("train", "valid", "test"):
        img_dir = dataset / split / "images"
        lbl_dir = dataset / split / "labels"
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.glob("*")):
            lbl = lbl_dir / (img_path.stem + ".txt")
            vis, issues = draw_label(img_path, lbl, names)
            if vis is None:
                continue
            cv2.imwrite(str(out / f"{split}__{img_path.stem}.jpg"), vis)
            for msg in issues:
                all_issues.append(f"{split}/{img_path.name}: {msg}")

    print(f"\nPreviews written to: {out.resolve()}")
    if all_issues:
        print("\n=== Potential annotation issues ===")
        for s in all_issues:
            print("  -", s)
    else:
        print("\nNo obvious annotation issues found.")


if __name__ == "__main__":
    main()

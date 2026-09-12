"""
Annotate AES2 part images with defect position + category.

Two annotation sources:
  1. Ground-truth YOLO-seg polygons from dataset/*/labels (exact).
  2. Manual review boxes (Claude visual inspection) supplied via review.json.

Output: annotated JPGs + per-image JSON + a summary.json, in OUT_DIR.
"""
import os, glob, re, json, cv2, numpy as np
import taxonomy as TAX

try:
    import config
    AES = config.LEGACY_AES_DIR
    OUT = os.environ.get("QMS_ANNOTATED_OUT", AES + "_annotated")
except Exception:
    AES = os.environ.get("QMS_AES_DIR", r"C:\workspace-ai\AES2-20260830T112553Z-1-001\AES2")
    OUT = AES + "_annotated"
LABEL_DIRS = [r"dataset\train\labels", r"dataset\valid\labels"]
# Dataset (YOLO GT) class order. Defect names + colors +
# severity are centrally controlled in knowledge/taxonomy.json (via taxonomy.py).
CLASSES = ['black mark after electroplating', 'electroplating defect',
           'incomplete embossing', 'line defect', 'serration']
COLORS = TAX.colors_map()
FALLBACK = TAX.FALLBACK_COLOR


def base(fn):
    m = re.match(r"(IMG\d+)", os.path.basename(fn))
    return m.group(1) if m else os.path.splitext(os.path.basename(fn))[0]


def load_gt():
    """base-name -> list of (class_name, polygon_norm[list of (x,y)])"""
    gt = {}
    for d in LABEL_DIRS:
        for f in glob.glob(os.path.join(d, "*.txt")):
            b = base(f)
            polys = []
            with open(f) as fh:
                for line in fh:
                    p = line.split()
                    if not p:
                        continue
                    cid = int(float(p[0]))
                    vals = list(map(float, p[1:]))
                    pts = [(vals[i], vals[i + 1]) for i in range(0, len(vals) - 1, 2)]
                    name = CLASSES[cid] if 0 <= cid < len(CLASSES) else f"class{cid}"
                    polys.append((name, pts))
            gt[b] = polys  # may be [] for OK
    return gt


def color_for(name):
    return TAX.color_for(name)


# Severity is centrally controlled in knowledge/taxonomy.json (via taxonomy.py).
SEVERITY_RULES = TAX.severity_rules()


def defect_priority(category):
    return TAX.priority(category)


def sort_by_priority(dets):
    """Return dets sorted highest-severity first (stable)."""
    return sorted(dets, key=lambda d: defect_priority(d.get("category", "")), reverse=True)


def draw_label(img, x, y, text, color):
    font, scale, th = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    (tw, tht), bl = cv2.getTextSize(text, font, scale, th)
    y = max(y, tht + 6)
    cv2.rectangle(img, (x, y - tht - 6), (x + tw + 6, y + bl - 2), color, -1)
    cv2.putText(img, text, (x + 3, y - 3), font, scale, (255, 255, 255), th, cv2.LINE_AA)


def _bbox_px(bbox, w, h):
    bx, by, bw, bh = bbox
    x0, y0 = int(bx * w), int(by * h)
    x1, y1 = int((bx + bw) * w), int((by + bh) * h)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    return x0, y0, x1, y1


def _largest_contours(mask, min_area, top=3):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) >= min_area]
    cnts.sort(key=cv2.contourArea, reverse=True)
    return cnts[:top]


def segment_defect(img, bbox, method):
    """Return a list of pixel contours (Nx1x2 int32) segmenting the defect inside bbox.
    Real OpenCV masks: color thresholding for paint/stain/rust, edge/GrabCut for geometry.
    Returns [] if segmentation fails (caller falls back to the box)."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = _bbox_px(bbox, w, h)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return []
    roi = img[y0:y1, x0:x1]
    area = roi.shape[0] * roi.shape[1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = None

    if method == "red_paint":
        m1 = cv2.inRange(hsv, (0, 60, 60), (12, 255, 255))
        m2 = cv2.inRange(hsv, (160, 40, 60), (180, 255, 255))
        mask = cv2.bitwise_or(m1, m2)
    elif method == "rust":
        # brown/orange-red, lower saturation than fresh paint
        mask = cv2.inRange(hsv, (3, 40, 40), (25, 255, 230))
    elif method == "dark":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        thr = max(40, int(np.mean(gray) - 1.0 * np.std(gray)))
        mask = cv2.inRange(gray, 0, thr)
    elif method in ("edge", "geometry"):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(gray, 40, 120)
        edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
        mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    elif method == "grabcut":
        m = np.zeros(roi.shape[:2], np.uint8)
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        rect = (2, 2, roi.shape[1] - 4, roi.shape[0] - 4)
        try:
            cv2.grabCut(roi, m, rect, bgd, fgd, 4, cv2.GC_INIT_WITH_RECT)
            mask = np.where((m == 1) | (m == 3), 255, 0).astype(np.uint8)
        except Exception:
            return []
    if mask is None:
        return []
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts = _largest_contours(mask, min_area=max(30, 0.01 * area))
    out = []
    for c in cnts:
        c = c + np.array([[x0, y0]])  # shift to full-image coords
        out.append(c.astype(np.int32))
    return out


def annotate(img, dets, uncertain=False):
    """dets: list of dict {category, points(norm) | bbox(norm) [+ seg method], reason}.
    Draws every defect. The highest-severity defect is marked PRIMARY (thicker border)."""
    h, w = img.shape[:2]
    overlay = img.copy()
    dets = sort_by_priority(dets)
    for i, d in enumerate(dets):
        name = d["category"]
        is_primary = (i == 0) and not uncertain and len(dets) >= 1
        color = (0, 140, 255) if uncertain else color_for(name)
        prefix = "? " if uncertain else ("* " if is_primary else "")
        label = prefix + name + (f"  [P{defect_priority(name)}]" if not uncertain else "")
        seg_contours = []
        if d.get("points"):
            pts = np.array([[int(x * w), int(y * h)] for x, y in d["points"]], np.int32)
            seg_contours = [pts]
            x0, y0 = int(pts[:, 0].min()), int(pts[:, 1].min())
            x1, y1 = int(pts[:, 0].max()), int(pts[:, 1].max())
        else:
            bx, by, bw, bh = d["bbox"]
            x0, y0 = int(bx * w), int(by * h)
            x1, y1 = int((bx + bw) * w), int((by + bh) * h)
            if d.get("seg"):
                seg_contours = segment_defect(img, d["bbox"], d["seg"])
        if seg_contours:
            cv2.fillPoly(overlay, seg_contours, color)
            cv2.polylines(img, seg_contours, True, color, 2, cv2.LINE_AA)
            allpts = np.vstack([c.reshape(-1, 2) for c in seg_contours])
            x0, y0 = int(allpts[:, 0].min()), int(allpts[:, 1].min())
            x1, y1 = int(allpts[:, 0].max()), int(allpts[:, 1].max())
        else:
            cv2.rectangle(overlay, (x0, y0), (x1, y1), color, -1)
        # Draw a locator rectangle + defect name label. Primary gets a thicker border.
        pad = 6
        rx0, ry0 = max(0, x0 - pad), max(0, y0 - pad)
        rx1, ry1 = min(w - 1, x1 + pad), min(h - 1, y1 + pad)
        cv2.rectangle(img, (rx0, ry0), (rx1, ry1), color, 4 if is_primary else 2, cv2.LINE_AA)
        draw_label(img, rx0, ry0, label, color)
    cv2.addWeighted(overlay, 0.30, img, 0.70, 0, img)
    return img


def draw_ok_banner(img):
    h, w = img.shape[:2]
    color = COLORS['OK']
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), color, 8)
    draw_label(img, 15, 40, "OK - no defect", color)
    return img


def main():
    os.makedirs(OUT, exist_ok=True)
    gt = load_gt()
    review = {}
    rp = os.path.join(os.path.dirname(__file__), "aes2_review.json")
    if os.path.exists(rp):
        with open(rp) as fh:
            review = json.load(fh)

    summary = []
    for path in sorted(glob.glob(os.path.join(AES, "*.jpg"))):
        b = base(path)
        img = cv2.imread(path)
        if img is None:
            continue
        dets, src = [], None
        forced = None
        if b in gt:
            src = "ground_truth"
            for name, pts in gt[b]:
                dets.append({"category": name, "points": pts,
                             "reason": "dataset ground-truth annotation"})
        elif b in review:
            src = "visual_review"
            dets = review[b].get("defects", [])
            forced = review[b].get("result")
        else:
            src = "unreviewed"

        if forced:
            result = forced
        else:
            result = "OK" if (src != "unreviewed" and len(dets) == 0) else \
                     ("DEFECT" if dets else "UNREVIEWED")
        if dets:
            annotate(img, dets, uncertain=(result == "UNCERTAIN"))
        elif result == "OK":
            draw_ok_banner(img)

        out_img = os.path.join(OUT, b + "_annotated.jpg")
        cv2.imwrite(out_img, img)
        sorted_dets = sort_by_priority(dets)
        rec = {"image": b, "source": src, "result": result,
               "primary_defect": (sorted_dets[0]["category"] if sorted_dets else None),
               "defects": [{"type": d["category"],
                            "severity": defect_priority(d["category"]),
                            "primary": (i == 0),
                            "location": d.get("location", "see box"),
                            "reason": d.get("reason", "")}
                           for i, d in enumerate(sorted_dets)]}
        with open(os.path.join(OUT, b + ".json"), "w") as fh:
            json.dump(rec, fh, indent=2)
        summary.append(rec)
        print(f"{b}: {result:10s} [{src}] defects={len(dets)}")

    with open(os.path.join(OUT, "_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    n_def = sum(1 for s in summary if s["result"] == "DEFECT")
    n_ok = sum(1 for s in summary if s["result"] == "OK")
    n_un = sum(1 for s in summary if s["result"] == "UNREVIEWED")
    print(f"\nTotal={len(summary)}  DEFECT={n_def}  OK={n_ok}  UNREVIEWED={n_un}")
    print("Output:", OUT)


if __name__ == "__main__":
    main()

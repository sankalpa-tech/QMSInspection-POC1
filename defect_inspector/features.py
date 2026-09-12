"""Fast classical-CV feature extraction for the 3-hole plated link part.

Produces a small, fixed-order numeric feature vector plus rich signal metadata
(locations, evidence) used both for kNN matching and for human-readable reasons.
No sklearn/scipy dependency - pure OpenCV + NumPy.
"""
from __future__ import annotations
import math
import numpy as np
import cv2

# Fixed feature order used by the kNN model. Keep stable across versions.
FEATURE_KEYS = [
    "area_frac",       # part area / image area
    "v_mean",          # mean brightness inside part
    "v_std",           # brightness spread (texture/gloss)
    "s_mean",          # mean saturation (chromate rainbow tint)
    "colorfulness",    # Hasler-Susstrunk colorfulness
    "gloss_hi_frac",   # fraction of very bright highlight pixels
    "lap_var",         # Laplacian variance (fine texture / brushed vs matte)
    "max_line_frac",   # longest clean straight line / part major axis
    "n_long_lines",    # count of long straight lines away from the border
    "dark_score",      # strength of localized dark marks
    "dark_area_frac",  # area fraction of dark marks
    "center_contrast", # high-pass contrast in central region (emboss proxy)
    "hole_rough",      # roughness of round-hole edges (serration proxy)
]

MAX_DIM = 1024


def _resize(img):
    h, w = img.shape[:2]
    scale = MAX_DIM / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def segment_part(gray):
    """Return a binary mask of the metal part (largest central compact blob)."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w = gray.shape
    cx, cy = w / 2.0, h / 2.0
    best, best_score = None, -1.0
    for mask in (otsu, cv2.bitwise_not(otsu)):
        m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        af = area / (h * w)
        if af < 0.05 or af > 0.9:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        bx, by = M["m10"] / M["m00"], M["m01"] / M["m00"]
        centrality = 1.0 - (math.hypot(bx - cx, by - cy) / math.hypot(cx, cy))
        score = centrality * (1.0 - abs(af - 0.35))
        if score > best_score:
            filled = np.zeros_like(gray)
            cv2.drawContours(filled, [c], -1, 255, -1)
            best, best_score = (filled, c), score
    if best is None:
        return np.full_like(gray, 255), None
    return best


def _colorfulness(bgr, mask):
    b, g, r = cv2.split(bgr.astype(np.float32))
    m = mask > 0
    rg = (r - g)[m]
    yb = (0.5 * (r + g) - b)[m]
    if rg.size == 0:
        return 0.0
    std = math.sqrt(rg.std() ** 2 + yb.std() ** 2)
    mean = math.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    return float(std + 0.3 * mean)


def _part_axis(contour):
    if contour is None:
        return MAX_DIM
    rect = cv2.minAreaRect(contour)
    (w, h) = rect[1]
    return max(w, h) if max(w, h) > 0 else MAX_DIM


def _region_name(px, py, w, h):
    col = "left" if px < w / 3 else ("right" if px > 2 * w / 3 else "center")
    row = "upper" if py < h / 3 else ("lower" if py > 2 * h / 3 else "middle")
    if row == "middle" and col == "center":
        return "center"
    return f"{row}-{col}"


def _phash(gray):
    """64-bit dHash as a hex string for fast near-duplicate recall."""
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    bits = 0
    for b in diff.flatten():
        bits = (bits << 1) | int(b)
    return f"{bits:016x}"


def hamming(a, b):
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def extract(path):
    """Extract features + signals from an image path.

    Returns dict: {"vector": {k: float}, "signals": {...}, "size": (w,h)}.
    """
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"cannot read image: {path}")
    bgr = _resize(bgr)
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    mask, contour = segment_part(gray)
    part_axis = _part_axis(contour)
    m = mask > 0
    area_frac = float(m.sum()) / (h * w)

    # eroded mask keeps analysis away from the part border
    er = cv2.erode(mask, np.ones((max(3, int(part_axis * 0.03)),) * 2, np.uint8))
    em = er > 0

    V = hsv[:, :, 2].astype(np.float32)
    S = hsv[:, :, 1].astype(np.float32)
    v_vals = V[m]
    v_mean = float(v_vals.mean()) if v_vals.size else 0.0
    v_std = float(v_vals.std()) if v_vals.size else 0.0
    s_mean = float(S[m].mean()) if v_vals.size else 0.0
    gloss_hi_frac = float((v_vals > 235).mean()) if v_vals.size else 0.0
    colorfulness = _colorfulness(bgr, mask)

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = float(lap[em].var()) if em.any() else 0.0

    # --- line defect: long clean straight lines away from border ---
    edges = cv2.Canny(gray, 60, 160)
    edges[~em] = 0
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=45,
                            minLineLength=int(0.20 * part_axis), maxLineGap=12)
    max_line_frac, n_long_lines, line_meta = 0.0, 0, []
    if lines is not None:
        for l in np.asarray(lines).reshape(-1, 4):
            x1, y1, x2, y2 = map(int, l)
            length = math.hypot(x2 - x1, y2 - y1)
            frac = length / part_axis
            if frac < 0.20:
                continue
            n_long_lines += 1
            if frac > max_line_frac:
                max_line_frac = frac
            mxx, myy = (x1 + x2) // 2, (y1 + y2) // 2
            line_meta.append({"frac": round(frac, 2),
                              "mid": [mxx, myy],
                              "angle": round(math.degrees(math.atan2(y2 - y1, x2 - x1)), 1)})
    line_meta.sort(key=lambda d: -d["frac"])

    # --- dark marks: locally darker than surroundings but not holes ---
    local_bg = cv2.medianBlur(gray, 51)
    diff = local_bg.astype(np.int16) - gray.astype(np.int16)
    dark = ((diff > 22) & (gray > 55)).astype(np.uint8)
    dark[~em] = 0
    dark_area_frac, dark_score, dark_meta = 0.0, 0.0, []
    n, lab, stats, cent = cv2.connectedComponentsWithStats(dark, 8)
    part_area = max(1.0, float(m.sum()))
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        af = a / part_area
        if af < 0.0008 or af > 0.08:
            continue
        comp = lab == i
        strength = float(diff[comp].mean())
        dark_area_frac += af
        dark_score = max(dark_score, strength * min(1.0, af * 200))
        dark_meta.append({"area_frac": round(af, 4), "strength": round(strength, 1),
                          "centroid": [int(cent[i][0]), int(cent[i][1])]})
    dark_meta.sort(key=lambda d: -d["strength"])

    # --- center contrast (embossing proxy) ---
    cyy, cxx = h // 2, w // 2
    ry, rx = int(h * 0.18), int(w * 0.18)
    patch = gray[max(0, cyy - ry):cyy + ry, max(0, cxx - rx):cxx + rx].astype(np.float32)
    center_contrast = 0.0
    if patch.size:
        hp = patch - cv2.GaussianBlur(patch, (0, 0), 3)
        center_contrast = float(hp.std())

    # --- hole edge roughness (serration proxy) ---
    hole_rough = 0.0
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=int(part_axis * 0.15),
                               param1=120, param2=30,
                               minRadius=int(part_axis * 0.02), maxRadius=int(part_axis * 0.10))
    if circles is not None:
        for cx0, cy0, r0 in np.round(circles[0]).astype(int):
            ang = np.linspace(0, 2 * np.pi, 72, endpoint=False)
            xs = np.clip((cx0 + (r0 + 2) * np.cos(ang)).astype(int), 0, w - 1)
            ys = np.clip((cy0 + (r0 + 2) * np.sin(ang)).astype(int), 0, h - 1)
            ring = gray[ys, xs].astype(np.float32)
            hole_rough = max(hole_rough, float(ring.std()))

    vector = {
        "area_frac": round(area_frac, 4),
        "v_mean": round(v_mean, 2),
        "v_std": round(v_std, 2),
        "s_mean": round(s_mean, 2),
        "colorfulness": round(colorfulness, 2),
        "gloss_hi_frac": round(gloss_hi_frac, 4),
        "lap_var": round(lap_var, 2),
        "max_line_frac": round(max_line_frac, 3),
        "n_long_lines": float(n_long_lines),
        "dark_score": round(dark_score, 2),
        "dark_area_frac": round(dark_area_frac, 4),
        "center_contrast": round(center_contrast, 2),
        "hole_rough": round(hole_rough, 2),
    }
    signals = {
        "lines": line_meta[:5],
        "dark_marks": dark_meta[:5],
        "size": [w, h],
        "region_of": {},
        "phash": _phash(gray),
    }
    if line_meta:
        mx, my = line_meta[0]["mid"]
        signals["region_of"]["line"] = _region_name(mx, my, w, h)
    if dark_meta:
        dx, dy = dark_meta[0]["centroid"]
        signals["region_of"]["dark"] = _region_name(dx, dy, w, h)
    return {"vector": vector, "signals": signals, "size": (w, h)}


def to_array(vector):
    return np.array([float(vector[k]) for k in FEATURE_KEYS], dtype=np.float64)

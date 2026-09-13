"""Learning DB (SQLite) + knowledge cache (JSON) for the defect inspector.

- learning.db  : durable memory (exemplars, inspections, feedback, meta)
- knowledge/defect_kb.json : the human+machine readable cache that inspect.py reads.
  It holds per-category text signatures, decision thresholds, feature centroids
  and normalization stats. `rebuild_cache` regenerates it from the exemplars.
"""
from __future__ import annotations
import os
import json
import sqlite3
import datetime as _dt
import numpy as np

from . import features as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, "learning.db")
CACHE_PATH = os.path.join(ROOT, "knowledge", "defect_kb.json")
LESSONS_PATH = os.path.join(ROOT, "knowledge", "lessons.json")

CATEGORIES = [
    "Black Mark After Electroplating",
    "Electroplating Defect",
    "Incomplete Embossing",
    "Line Defect",
    "Serration",
    "OK",
]

# Human-authored visual signatures (seeded from the reference analysis).
SIGNATURES = {
    "Black Mark After Electroplating": "Discrete dark/black smudge or spot sitting on top of otherwise normal plating; local darkening not explained by shadow.",
    "Electroplating Defect": "Patch of abnormal/uneven plating - dull, matte or rough area where the finish did not deposit uniformly.",
    "Incomplete Embossing": "Stamped 'VA' logo is partial, faint or missing strokes; low central relief contrast.",
    "Line Defect": "A single distinct straight scratch/gouge line crossing the surface, standing out from the normal brushed micro-texture.",
    "Serration": "Jagged/toothed edge on a hole that should be smooth (NOT the by-design splined bottom hole).",
    "OK": "Uniform finish, complete emboss, no discrete mark/line, smooth round hole edges.",
}


def _now():
    return _dt.datetime.now().isoformat(timespec="seconds")


def connect():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    con = connect()
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS exemplars (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT, path TEXT, label TEXT, part TEXT DEFAULT 'default',
            features_json TEXT, phash TEXT, source TEXT,
            confirmed INTEGER DEFAULT 1, added_at TEXT
        );
        CREATE TABLE IF NOT EXISTS inspections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT, path TEXT, pred TEXT, result TEXT,
            confidence INTEGER, defects_json TEXT, features_json TEXT, created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            inspection_id INTEGER, true_label TEXT, note TEXT, created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
        """
    )
    con.commit()
    con.close()


def add_exemplar(name, path, label, vector, phash="", source="manual", confirmed=1, part="default"):
    con = connect()
    con.execute(
        "INSERT INTO exemplars(name,path,label,part,features_json,phash,source,confirmed,added_at)"
        " VALUES(?,?,?,?,?,?,?,?,?)",
        (name, path, label, part, json.dumps(vector), phash, source, confirmed, _now()),
    )
    con.commit()
    con.close()


def all_exemplars(confirmed_only=True, part=None):
    con = connect()
    q = "SELECT * FROM exemplars"
    conds = []
    args = []
    if confirmed_only:
        conds.append("confirmed=1")
    if part is not None:
        conds.append("part=?")
        args.append(part)
    if conds:
        q += " WHERE " + " AND ".join(conds)
    rows = con.execute(q, args).fetchall()
    con.close()
    return rows


def list_parts():
    con = connect()
    rows = con.execute(
        "SELECT part, COUNT(*) n FROM exemplars WHERE confirmed=1 GROUP BY part"
    ).fetchall()
    con.close()
    return {r["part"]: r["n"] for r in rows}


def categories_for_part(part):
    """Distinct defect categories learned so far for a given part (free-form)."""
    con = connect()
    rows = con.execute(
        "SELECT DISTINCT label FROM exemplars WHERE confirmed=1 AND part=? ORDER BY label",
        (part,),
    ).fetchall()
    con.close()
    return [r["label"] for r in rows]


def log_inspection(name, path, pred, result, confidence, defects, vector):
    con = connect()
    cur = con.execute(
        "INSERT INTO inspections(name,path,pred,result,confidence,defects_json,features_json,created_at)"
        " VALUES(?,?,?,?,?,?,?,?)",
        (name, path, pred, result, confidence, json.dumps(defects), json.dumps(vector), _now()),
    )
    con.commit()
    iid = cur.lastrowid
    con.close()
    return iid


def add_feedback(inspection_id, true_label, note=""):
    con = connect()
    con.execute(
        "INSERT INTO feedback(inspection_id,true_label,note,created_at) VALUES(?,?,?,?)",
        (inspection_id, true_label, note, _now()),
    )
    con.commit()
    con.close()


def load_lessons():
    """Durable, human+machine readable lessons learned from human corrections.

    Kept in knowledge/lessons.json so it survives every cache rebuild. Holds
    per-part visual signatures, confirmed rulings, the severity-priority map,
    segmentation gotchas and per-image confirmed defects for the free-form parts
    (e.g. the cup-collar) that are not covered by the fixed 6-category centroids.
    """
    if os.path.exists(LESSONS_PATH):
        try:
            with open(LESSONS_PATH) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def rebuild_cache():
    """Recompute per-category centroids + normalization from exemplars -> cache JSON."""
    rows = all_exemplars(confirmed_only=True)
    keys = F.FEATURE_KEYS
    by_cat = {c: [] for c in CATEGORIES}
    mat = []
    for r in rows:
        v = json.loads(r["features_json"])
        arr = [float(v.get(k, 0.0)) for k in keys]
        mat.append(arr)
        if r["label"] in by_cat:
            by_cat[r["label"]].append(arr)
    mat = np.array(mat, dtype=np.float64) if mat else np.zeros((0, len(keys)))
    mean = mat.mean(axis=0) if len(mat) else np.zeros(len(keys))
    std = mat.std(axis=0) if len(mat) else np.ones(len(keys))
    std[std < 1e-6] = 1e-6

    centroids = {}
    counts = {}
    for c, lst in by_cat.items():
        counts[c] = len(lst)
        if lst:
            centroids[c] = list(np.array(lst, dtype=np.float64).mean(axis=0))

    # per-part -> per-category counts, for scoped teaching/inspection
    parts = {}
    for r in rows:
        p = r["part"] if "part" in r.keys() and r["part"] else "default"
        parts.setdefault(p, {})
        parts[p][r["label"]] = parts[p].get(r["label"], 0) + 1

    cache = {
        "version": 2,
        "updated_at": _now(),
        "categories": CATEGORIES,
        "signatures": SIGNATURES,
        "feature_keys": keys,
        "norm": {"mean": list(mean), "std": list(std)},
        "centroids": centroids,
        "counts": counts,
        "parts": parts,
        "thresholds": _default_thresholds(),
        "confusions": [
            ["Electroplating Defect", "Incomplete Embossing", "OK"],
            ["Serration", "splined-hole (design, not a defect)"],
            ["Black Mark After Electroplating", "shadow/plating tint"],
        ],
        "lessons": load_lessons(),
    }
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)
    return cache


def _default_thresholds():
    # Rule-signal thresholds; tuned lightly, adjustable as the cache learns.
    return {
        "line_frac_strong": 0.45,
        "line_frac_weak": 0.30,
        "dark_score_strong": 30.0,
        "dark_score_weak": 15.0,
        "center_contrast_low": 6.0,
        # The splined bottom hole is a DESIGN feature present on every part, so raw
        # hole roughness is not discriminative. Keep the rule effectively off and let
        # kNN + human review decide serration. Feature stays in the vector for kNN.
        "hole_rough_high": 999.0,
        "uncertain_below": 50,
        "knn_k": 5,
        "phash_recall_max": 6,
    }


def load_cache():
    with open(CACHE_PATH) as f:
        return json.load(f)

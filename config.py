"""Central configuration for the QMS defect inspector.

All paths and tunables resolve from environment variables first, then fall back
to sensible in-repo defaults. This removes hardcoded absolute paths and makes
the project portable across machines / containers.

Environment variables (all optional):
  QMS_DATA_DIR      images to inspect            (default: <repo>/data)
  QMS_OUT_DIR       annotated outputs            (default: <repo>/inspect_out)
  QMS_MODEL_PATH    packed checkpoint            (default: <repo>/models/best.pt)
  QMS_DB_PATH       learning database            (default: <repo>/learning.db)
  QMS_REVIEW_PATH   human review geometry        (default: <repo>/aes2_review.json)
  QMS_DATASET_DIR   YOLO dataset root            (default: <repo>/dataset)
  QMS_LOG_LEVEL     logging level                (default: INFO)
  ANTHROPIC_API_KEY Claude API key for escalation (no default; escalation off if unset)
  QMS_CLAUDE_MODEL  Claude model id              (default: claude-opus-4-20250514)
"""
from __future__ import annotations
import os
import logging

ROOT = os.path.dirname(os.path.abspath(__file__))


def _p(env, *default_parts):
    v = os.environ.get(env)
    return v if v else os.path.join(ROOT, *default_parts)


DATA_DIR = _p("QMS_DATA_DIR", "data")
OUT_DIR = _p("QMS_OUT_DIR", "inspect_out")
MODEL_PATH = _p("QMS_MODEL_PATH", "models", "best.pt")
DB_PATH = _p("QMS_DB_PATH", "learning.db")
REVIEW_PATH = _p("QMS_REVIEW_PATH", "aes2_review.json")
DATASET_DIR = _p("QMS_DATASET_DIR", "dataset")
KNOWLEDGE_DIR = os.path.join(ROOT, "knowledge")
CACHE_PATH = os.path.join(KNOWLEDGE_DIR, "defect_kb.json")
LESSONS_PATH = os.path.join(KNOWLEDGE_DIR, "lessons.json")
YOLO_WEIGHTS = _p("QMS_YOLO_WEIGHTS", "yolo11n-seg.pt")

# Optional legacy dataset image dir used by the annotate helper. Kept overridable
# so the original AES2 folder still works but is no longer hardcoded.
LEGACY_AES_DIR = os.environ.get(
    "QMS_AES_DIR",
    r"C:\workspace-ai\AES2-20260830T112553Z-1-001\AES2")

# --- Claude escalation ---
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.environ.get("QMS_CLAUDE_MODEL", "claude-opus-4-20250514")
CLAUDE_MAX_TOKENS = int(os.environ.get("QMS_CLAUDE_MAX_TOKENS", "1500"))

# --- recall / decision ---
PHASH_RECALL_MAX = int(os.environ.get("QMS_PHASH_RECALL_MAX", "6"))


def has_claude() -> bool:
    return bool(ANTHROPIC_API_KEY)


_LOG_CONFIGURED = False


def get_logger(name="qms"):
    """Return a configured logger (idempotent)."""
    global _LOG_CONFIGURED
    if not _LOG_CONFIGURED:
        level = os.environ.get("QMS_LOG_LEVEL", "INFO").upper()
        logging.basicConfig(
            level=getattr(logging, level, logging.INFO),
            format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        _LOG_CONFIGURED = True
    return logging.getLogger(name)


def ensure_dirs():
    for d in (DATA_DIR, OUT_DIR, os.path.dirname(MODEL_PATH), KNOWLEDGE_DIR):
        os.makedirs(d, exist_ok=True)

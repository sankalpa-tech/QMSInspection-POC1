"""Escalate an UNCERTAIN image to Claude - carrying ALL learned context.

Only called when the local kNN model in best.pt cannot confidently recall the
image. Claude receives the image plus the full learning context pack (signatures,
rulings, severity map, gotchas, nearest confirmed examples) so it reasons WITH
the accumulated memory, not from scratch, and returns the strict JSON verdict.

Requires ANTHROPIC_API_KEY. If unset, escalation is skipped and the caller keeps
the UNCERTAIN result (still zero cost).
"""
from __future__ import annotations
import base64
import json
import mimetypes
import os

import config
import context_pack

log = config.get_logger("escalate")


def _img_block(path):
    mime, _ = mimetypes.guess_type(path)
    if mime not in ("image/jpeg", "image/png", "image/webp", "image/gif"):
        mime = "image/jpeg"
    with open(path, "rb") as fh:
        data = base64.standard_b64encode(fh.read()).decode("ascii")
    return {"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}}


def _parse_json(text):
    """Extract the first JSON object from Claude's response."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in response: {text[:200]}")
    return json.loads(text[start:end + 1])


def escalate(path, model, feature_vec, part_hint=None):
    """Send the image + context pack to Claude. Returns (verdict_dict, meta).

    Raises RuntimeError if no API key. `verdict` follows the required schema.
    """
    if not config.has_claude():
        raise RuntimeError("ANTHROPIC_API_KEY not set; cannot escalate to Claude")

    import anthropic  # imported lazily so the SDK is optional

    prompt, inferred_part, neigh = context_pack.build(model, feature_vec, part_hint=part_hint)
    log.info("escalating %s to Claude (part=%s, %d neighbours)",
             os.path.basename(path), inferred_part, len(neigh))

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    msg = client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=config.CLAUDE_MAX_TOKENS,
        messages=[{
            "role": "user",
            "content": [_img_block(path), {"type": "text", "text": prompt}],
        }],
    )
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    verdict = _parse_json(text)

    # normalize to the required schema
    verdict.setdefault("result", "UNCERTAIN")
    verdict.setdefault("defects", [])
    meta = {
        "source": "claude",
        "model": config.CLAUDE_MODEL,
        "part": inferred_part,
        "neighbours": [{"name": n, "label": l, "dist": round(d, 2)} for n, l, d in neigh],
        "usage": {"input_tokens": getattr(msg.usage, "input_tokens", None),
                  "output_tokens": getattr(msg.usage, "output_tokens", None)},
    }
    log.info("claude verdict: %s (%d defects, tokens in=%s out=%s)",
             verdict["result"], len(verdict["defects"]),
             meta["usage"]["input_tokens"], meta["usage"]["output_tokens"])
    return verdict, meta

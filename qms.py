"""QMS defect inspector - unified CLI.

Implements the production loop:

    image --> local kNN (best.pt) --> RESOLVED  (0 tokens)
                                  \\-> UNCERTAIN --> Claude (full learned context)
                                                    --> verdict --> teach --> repack best.pt

Commands:
    python qms.py inspect <image|dir> [--escalate] [--no-teach] [--out DIR]
    python qms.py build                       # repack models/best.pt from DB
    python qms.py teach <image> --json <verdict.json> [--part "Bearing Cup"]
    python qms.py context <image>             # print the Claude context pack (debug)
    python qms.py serve [--port 8000]         # REST API
    python qms.py stats                       # DB + model summary

'inspect --escalate' auto-calls Claude on UNCERTAIN (needs ANTHROPIC_API_KEY)
and folds the result back in. Without --escalate, UNCERTAIN images are reported
for a later manual/Claude pass.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys

import config

log = config.get_logger("qms")


def _images(target):
    if os.path.isdir(target):
        out = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"):
            out += glob.glob(os.path.join(target, ext))
        return sorted(set(out))
    return [target]


def cmd_inspect(args):
    import run_pt
    if not os.path.exists(config.MODEL_PATH):
        log.error("model missing: %s  (run: python qms.py build)", config.MODEL_PATH)
        return 2
    model = run_pt.load_model(config.MODEL_PATH)
    out_dir = args.out or config.OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    n_resolved = n_escalated = n_uncertain = 0
    for p in _images(args.target):
        if not os.path.exists(p):
            log.warning("skip missing: %s", p)
            continue
        verdict, status, out_img = run_pt.inspect(p, model, out_dir)
        resolved = status.startswith("RESOLVED")

        if not resolved and args.escalate:
            verdict, status, out_img, model = _do_escalate(p, model, out_dir, args)
            if status.startswith("RESOLVED"):
                n_escalated += 1
            else:
                n_uncertain += 1
        elif resolved:
            n_resolved += 1
        else:
            n_uncertain += 1

        print(f"\n=== {os.path.basename(p)} ===")
        print(json.dumps({k: v for k, v in verdict.items() if k != "hint"}, indent=2))
        print("STATUS:", status)
        print("annotated:", out_img)

    total = n_resolved + n_escalated + n_uncertain
    if total > 1:
        print(f"\n{n_resolved} local (0 tokens), {n_escalated} escalated+learned, "
              f"{n_uncertain} still uncertain / {total} total.")
    return 0


def _do_escalate(path, model, out_dir, args):
    """Escalate to Claude, teach, reload model. Returns (verdict, status, out_img, model)."""
    import run_pt
    import escalate as ESC
    import teach_loop
    from defect_inspector import features as F

    if not config.has_claude():
        log.warning("UNCERTAIN and --escalate set, but ANTHROPIC_API_KEY is missing; "
                    "leaving as UNCERTAIN")
        verdict, status, out_img = run_pt.inspect(path, model, out_dir)
        return verdict, status, out_img, model
    try:
        feat = F.extract(path)
        verdict, meta = ESC.escalate(path, model, feat["vector"])
        # render + persist annotated output for the Claude verdict
        run_pt.render_verdict(path, verdict, model, out_dir)
        if not args.no_teach and verdict.get("result") in ("DEFECT", "OK"):
            part = meta.get("part") or "Bearing Cup"
            teach_loop.teach(path, verdict, part=part, source="claude", rebuild=True)
            model = run_pt.load_model(config.MODEL_PATH)  # reload improved model
        toks = meta["usage"]
        status = (f"ESCALATED->RESOLVED via Claude "
                  f"(tokens in={toks['input_tokens']} out={toks['output_tokens']}); learned")
        out_img = os.path.join(out_dir, os.path.splitext(os.path.basename(path))[0] + "_annotated.jpg")
        return verdict, status, out_img, model
    except Exception as e:  # noqa
        log.error("escalation failed: %s", e)
        verdict, status, out_img = run_pt.inspect(path, model, out_dir)
        return verdict, f"UNCERTAIN (escalation error: {e})", out_img, model


def cmd_build(args):
    import build_pt
    build_pt.build()
    return 0


def cmd_teach(args):
    import teach_loop
    with open(args.json) as fh:
        verdict = json.load(fh)
    label = teach_loop.teach(args.target, verdict, part=args.part, source="manual", rebuild=True)
    print(f"learned {os.path.basename(args.target)} -> {label} [part={args.part}]")
    return 0


def cmd_context(args):
    import run_pt
    from defect_inspector import features as F
    model = run_pt.load_model(config.MODEL_PATH)
    feat = F.extract(args.target)
    import context_pack
    prompt, part, neigh = context_pack.build(model, feat["vector"])
    print(prompt)
    print(f"\n[inferred part: {part}; {len(neigh)} nearest exemplars]")
    return 0


def cmd_serve(args):
    os.environ["INSPECTOR_PORT"] = str(args.port)
    from defect_inspector import api
    api.main()
    return 0


def cmd_stats(args):
    from defect_inspector import kb
    kb.init_db()
    con = kb.connect()
    parts = dict((r[0], r[1]) for r in con.execute(
        "SELECT part, COUNT(*) FROM exemplars GROUP BY part"))
    fb = con.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
    insp = con.execute("SELECT COUNT(*) FROM inspections").fetchone()[0]
    con.close()
    print("model     :", config.MODEL_PATH, "(exists)" if os.path.exists(config.MODEL_PATH) else "(MISSING)")
    print("exemplars :", parts)
    print("feedback  :", fb, "| inspections:", insp)
    print("claude    :", "configured" if config.has_claude() else "NOT configured (set ANTHROPIC_API_KEY)")
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="qms", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("inspect", help="inspect image(s) locally, optionally escalate")
    pi.add_argument("target")
    pi.add_argument("--escalate", action="store_true", help="call Claude on UNCERTAIN and learn")
    pi.add_argument("--no-teach", action="store_true", help="do not fold Claude result back in")
    pi.add_argument("--out")
    pi.set_defaults(func=cmd_inspect)

    pb = sub.add_parser("build", help="repack models/best.pt from the learning DB")
    pb.set_defaults(func=cmd_build)

    pt = sub.add_parser("teach", help="teach a verdict for an image")
    pt.add_argument("target")
    pt.add_argument("--json", required=True, help="verdict JSON file")
    pt.add_argument("--part", default="Bearing Cup")
    pt.set_defaults(func=cmd_teach)

    pc = sub.add_parser("context", help="print the Claude learning context pack for an image")
    pc.add_argument("target")
    pc.set_defaults(func=cmd_context)

    ps = sub.add_parser("serve", help="run the REST API")
    ps.add_argument("--port", type=int, default=8000)
    ps.set_defaults(func=cmd_serve)

    pst = sub.add_parser("stats", help="show DB + model summary")
    pst.set_defaults(func=cmd_stats)
    return p


def main(argv):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

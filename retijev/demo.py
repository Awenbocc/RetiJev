"""Answer the questions of one patient case with a trained RetiJev checkpoint.

  python -m retijev.demo --run runs/retijev-0.8b --input examples/next_investigation.json

The input is one request: {"state": "<free-text case>", "questions": {qid: {"type", "instructions", "criteria"}}}
with no labels (see examples/). The case is encoded once and every question is scored from its cache; nothing is
generated. Output, per question: noul -> p(true); choice -> probability of each option; score -> probability of each
ordered level.
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from retijev.predict import load_model


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="checkpoint directory or Hugging Face path org/repo[/subfolder], e.g. BoKelvin/RetiJev/0.8B")
    ap.add_argument("--base", default=None, help="override the base model recorded in the checkpoint (local dir or Hub id)")
    ap.add_argument("--input", default="examples/next_investigation.json")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    from medjev.api import Request, to_answers, to_record
    from medjev.model import MAX_BRANCH, MAX_STATE

    tok, model = load_model(a.run, a.device, torch.bfloat16 if a.dtype == "bf16" else torch.float32, a.base)
    rec, meta = to_record(Request.model_validate(json.load(open(a.input))))
    enc = model.encode(tok, rec, max_state=MAX_STATE, max_branch=MAX_BRANCH)
    model.probs_and_prefix(enc)                       # warm-up, so the timing below excludes kernel compilation
    if a.device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    probs, _ = model.probs_and_prefix(enc)
    if a.device == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000
    print(json.dumps(to_answers(probs, meta), indent=2))
    print(f"{len(meta)} questions answered in {ms:.0f} ms ({a.device}, {a.dtype})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

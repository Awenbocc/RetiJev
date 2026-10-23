"""Train RetiJev: medjev.train with soft-label distillation and paraphrase-consistency terms. MedJev is not modified:
three of the names medjev.train imported are wrapped before its main() runs.

    python -m retijev.train --teacher data/sample/teacher_train.jsonl --groups data/sample/groups.json \
        --kd_w 0.5 --kd_T 2 --cons_w 0.5 -- <usual medjev.train arguments>

Per question, loss = MedJev loss (cross-entropy [+ ordinal term])
    + kd_w  * T^2 * KL(teacher_T || student_T)   teacher = Qwen3.8-27B zero-shot probabilities on this TRAINING question
                                                  (decision questions only; rule-computed scale questions are skipped)
    + cons_w * KL(bank || student)                bank = running mean (detached) of the student's own probabilities on
                                                  the other members of the question's paraphrase group
A paraphrase record `<idx>~p` borrows the teacher probabilities of its original (same facts, same labels). Questions
without a teacher row, or whose option keys do not match, get no distillation term.
"""
from __future__ import annotations

import argparse
import json
import sys

import torch
import torch.nn.functional as F

import medjev.train as mt

CFG: dict = {"teacher": {}, "group": {}, "kd_w": 0.0, "kd_T": 2.0, "cons_w": 0.0, "bank": {}, "stats": {"kd": 0, "cons": 0, "q": 0}}
_orig_permute, _orig_materialize, _orig_qloss = mt.permute_choice_options, mt.materialize, mt.question_loss


def permute_keep_idx(req, rng):
    out = _orig_permute(req, rng)
    out["idx"] = req.get("idx")
    return out


def materialize_keep_idx(req):
    rec = _orig_materialize(req)
    for q in rec["questions"]:
        q["_idx"] = req.get("idx")
    return rec


def question_loss(z, q, dev, ord_w):
    loss = _orig_qloss(z, q, dev, ord_w)
    idx, qid, keys = q.get("_idx"), q["qid"], [str(k).lower() for k in q["keys"]]
    CFG["stats"]["q"] += 1
    base = CFG["group"].get(idx, idx)
    t = CFG["teacher"].get((base, qid))
    # rule-computed scale labels are exact and the teacher is weaker there (27B 0.77-0.85): no distillation on scales
    if CFG["kd_w"] > 0 and t is not None and "scales_v" not in q.get("src", "") and all(k in t for k in keys):
        T = CFG["kd_T"]
        pt = torch.tensor([t[k] for k in keys], device=dev, dtype=z.dtype).clamp_min(1e-6)
        pt = pt ** (1.0 / T)
        pt = pt / pt.sum()
        loss = loss + CFG["kd_w"] * T * T * F.kl_div(F.log_softmax(z / T, -1), pt, reduction="sum")
        CFG["stats"]["kd"] += 1
    if CFG["cons_w"] > 0 and base in CFG["grouped"]:
        key = (base, qid)
        p = F.softmax(z, -1)
        prev = CFG["bank"].get(key)
        if prev is not None and prev[0] != idx and prev[1] == keys:
            target = torch.tensor(prev[2], device=dev, dtype=z.dtype)
            loss = loss + CFG["cons_w"] * F.kl_div(F.log_softmax(z, -1), target, reduction="sum")
            CFG["stats"]["cons"] += 1
        cur = p.detach().float().cpu().tolist()
        if prev is not None and prev[1] == keys:
            cur = [0.5 * a + 0.5 * b for a, b in zip(prev[2], cur)]
        CFG["bank"][key] = (idx, keys, cur)
    if CFG["stats"]["q"] % 20000 == 0:
        print(f"  [kd] questions {CFG['stats']['q']}: with distillation {CFG['stats']['kd']}, with consistency {CFG['stats']['cons']}", flush=True)
    return loss


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ours, rest = (argv[: argv.index("--")], argv[argv.index("--") + 1:]) if "--" in argv else (argv, [])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", nargs="*", default=[], help="per-question prediction files of the teacher on TRAINING data")
    ap.add_argument("--groups", default=None, help="JSON {paraphrase idx: original idx}")
    ap.add_argument("--kd_w", type=float, default=0.5)
    ap.add_argument("--kd_T", type=float, default=2.0)
    ap.add_argument("--cons_w", type=float, default=0.5)
    a = ap.parse_args(ours)
    for f in a.teacher:
        for p in map(json.loads, open(f)):
            CFG["teacher"][(p["idx"], p["qid"])] = {str(k).lower(): float(v) for k, v in p["probs"].items()}
    CFG["group"] = json.load(open(a.groups)) if a.groups else {}
    CFG["grouped"] = set(CFG["group"].values())
    CFG.update(kd_w=a.kd_w, kd_T=a.kd_T, cons_w=a.cons_w)
    print(f"[kd] teacher rows {len(CFG['teacher'])}, paraphrase partners {len(CFG['group'])}, kd_w {a.kd_w} T {a.kd_T} cons_w {a.cons_w}", flush=True)
    mt.permute_choice_options, mt.materialize, mt.question_loss = permute_keep_idx, materialize_keep_idx, question_loss
    sys.argv = ["medjev.train"] + rest
    return mt.main() or 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Per-question predictions from a trained RetiJev checkpoint, one JSONL row per question.

Mirrors the inference loop of `medjev.evaluate` (serving path, state encoded once per record, every question scored
from its cache) but keeps every prediction instead of only aggregates.

  python -m retijev.predict --run runs/retijev-0.8b --data data/sample --split development \
      --out runs/retijev-0.8b/preds_development.jsonl

`--run` is a local checkpoint directory or a Hugging Face path `org/repo[/subfolder][@revision]`, e.g. BoKelvin/RetiJev/0.8B.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

CKPT_FILES = ["*.json", "*.safetensors", "*.pt", "*.txt", "*.jinja"]


def resolve_run(run: str) -> str:
    """A local directory as is; otherwise `org/repo[/subfolder][@revision]` on the Hugging Face Hub, downloaded to the
    HF cache (only the checkpoint files of that subfolder)."""
    if os.path.isdir(run):
        return run
    from huggingface_hub import snapshot_download
    path, _, revision = run.partition("@")
    parts = path.split("/")
    repo, sub = "/".join(parts[:2]), "/".join(parts[2:])
    local = snapshot_download(repo, revision=revision or None,
                              allow_patterns=[f"{sub}/{p}" for p in CKPT_FILES] if sub else CKPT_FILES)
    return os.path.join(local, sub)


def load_model(run: str, device: str, dtype=torch.bfloat16, base: str | None = None):
    """-> (tokenizer, model). Same procedure as `medjev.checkpoint.load` (LoRA folded into the base in fp32, then cast),
    plus three things: `run` may point into a subfolder of a Hub repo, `base` overrides the base-model path recorded in
    head.pt (a local copy of the Qwen weights, or a Hub id), and a full fine-tune (`--lora 0`, no adapter) is loaded
    tensor by tensor."""
    import glob
    from safetensors.torch import load_file
    from medjev.model import DecisionModel, load_tokenizer
    run = resolve_run(run)
    meta = torch.load(os.path.join(run, "head.pt"), map_location="cpu", weights_only=False)
    base = base or meta["base"]
    adapter = os.path.join(run, "adapter_config.json")
    merge = not (os.path.exists(adapter) and json.load(open(adapter)).get("trainable_token_indices"))
    tok = load_tokenizer(base, revision=meta.get("base_revision"))
    m = DecisionModel(base, tok, device, lora=None, revision=meta.get("base_revision"), head_dim=meta.get("head_dim", 256),
                      option_isolation=meta.get("option_isolation", False), dtype=torch.float32 if merge else dtype)
    if os.path.exists(adapter):
        from peft import PeftModel
        m.lm = PeftModel.from_pretrained(m.lm, run).to(device)
        if merge:
            m.lm = m.lm.merge_and_unload()          # in fp32: exact
    else:
        sd = {}
        for f in sorted(glob.glob(os.path.join(run, "*.safetensors"))):
            sd.update(load_file(f))
        missing, unexpected = m.lm.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"full checkpoint does not match the backbone: missing {missing[:5]} unexpected {unexpected[:5]}")
    m.lm = m.lm.to(dtype)
    m.head.load_state_dict(meta["head"])
    m.lm.config.use_cache = True
    m.eval()
    return tok, m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="checkpoint directory or Hugging Face path org/repo[/subfolder], e.g. BoKelvin/RetiJev/0.8B")
    ap.add_argument("--base", default=None, help="override the base model recorded in the checkpoint (local dir or Hub id)")
    ap.add_argument("--data", required=True, help="directory holding <split>.jsonl")
    ap.add_argument("--split", default="development")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.split == "test" and not a.allow_test:
        sys.exit("refusing to read test without --allow-test")
    from medjev.evaluate import gold_key, keys_of
    from medjev.model import MAX_BRANCH, MAX_STATE
    from medjev.records import load_records, materialize

    tok, model = load_model(a.run, a.device, torch.bfloat16 if a.dtype == "bf16" else torch.float32, a.base)
    records = load_records(os.path.join(a.data, f"{a.split}.jsonl"), source="medjev")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    n = correct = 0
    with open(a.out, "w") as fp:
        for i, req in enumerate(records):
            enc = model.encode(tok, materialize(req), max_state=MAX_STATE, max_branch=MAX_BRANCH)
            with torch.no_grad():
                probs, _ = model.probs_and_prefix(enc)
            for p, (qid, q) in zip(probs, req["questions"].items()):
                keys, p = keys_of(q), np.asarray(p, dtype=float)
                j, g = int(p.argmax()), keys.index(gold_key(q))
                fp.write(json.dumps({"idx": req["idx"], "qid": qid, "type": q["type"], "gold": keys[g], "pred": keys[j],
                                     "correct": j == g, "probs": dict(zip(keys, np.round(p, 5).tolist()))}) + "\n")
                n += 1
                correct += j == g
            if (i + 1) % 200 == 0:
                print(f"{i + 1}/{len(records)} records", flush=True)
    print(json.dumps({"run": a.run, "split": a.split, "questions": n, "micro_accuracy": round(correct / max(1, n), 4)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

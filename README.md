# RetiJev

![RetiJev overview: Eye+ and an ophthalmologist, with 0.8B, 2B and 9B models supporting decision, diagnosis and grading tasks through noul (yes/no), choice (one of N) and score (ordered levels) answers. Patient case text is encoded once to produce answer probabilities locally. Research use only.](docs/assets/retijev-overview-v1.png)

RetiJev is a family of small decision models (0.8B, 2B, 9B) for ophthalmology. Given a free-text description of a
patient and a set of closed-form clinical questions, it returns a probability for every predefined answer option.
It does not generate text. The case is encoded once and all questions are scored from that encoding, so a whole
case is answered in a single short pass. The models are small enough to run locally, and patient text never has to leave
the machine.

## Timeline

- **2026-10-05 · v1.** First public release of RetiJev, built on [MedJev](https://github.com/JunMa11/MedJev) as an exploration of the Jev family of decision models, with training and inference code, a data sample, and the 0.8B, 2B and 9B models on [Hugging Face](https://huggingface.co/BoKelvin/RetiJev).

## What RetiJev answers

RetiJev answers three kinds of questions:

| Family | What is asked | Examples |
|---|---|---|
| **Decision** | the next step for the patient as currently described | most appropriate next investigation, management direction, urgency, referral, whether a proposed plan is safe |
| **Diagnosis** | how likely each candidate diagnosis is, and which one leads | `dx_<disease>` likelihood per candidate, leading diagnosis among a candidate list |
| **Grading** | the grade under a published clinical grading system, computed from the findings in the note | ICDR diabetic retinopathy and macular oedema, Beckman AMD stage and AREDS2 eligibility, ETROP type for ROP, clinical activity score for thyroid eye disease, EVS endophthalmitis management, Chandler orbital cellulitis group, WHO visual impairment category, SUN anterior chamber cells, hyphema grade, Hodapp–Parrish–Anderson visual field stage, target IOP |

Each question has one of three answer types:

- `noul`: yes/no. Returns p(true).
- `choice`: one option out of a named list. Returns a probability per option.
- `score`: an ordered level such as a severity grade. Returns a probability per level.

> **Research use only.** RetiJev is not a medical device and has not been validated for clinical use. Decision and
> diagnosis labels in the training data were produced and cross-checked by LLMs. Ophthalmologists did not review
> them case by case. Grading labels are computed by code from the published grading rules.

## How it works

- **Backbone.** A frozen Qwen3.5 base model (hybrid Gated DeltaNet + gated attention) adapted with LoRA.
- **Readout.** Each question is appended to the case as its own branch. Every answer option is written out in the
  input, and a pointer head scores the options against a final decision token. Option names and descriptions are
  ordinary text, so new option lists need no new parameters.
- **Inference.** The patient state is encoded once and cached, and all questions are scored from copies of that cache.
  Nothing is decoded.
- **Training loss.** Each question contributes three terms:
  1. Cross-entropy, plus a ranked-probability term for ordered (`score`) questions.
  2. Distillation towards the soft probabilities of a larger teacher LLM (Qwen3.8-27B, zero-shot, training questions
     only). This applies to decision and diagnosis questions, not to code-computed grading labels.
  3. A consistency term between paraphrases of the same case.

## Installation

Python 3.12 or later. Install a PyTorch build that matches your CUDA version first (see [pytorch.org](https://pytorch.org)), then:

```bash
git clone https://github.com/Awenbocc/RetiJev.git
cd RetiJev
pip install -r requirements.txt
```

`requirements.txt` installs [MedJev](https://github.com/JunMa11/MedJev) at a pinned commit. MedJev provides the
model, encoding and training loop that RetiJev builds on.

On GPU, also install `causal-conv1d`. Without it the Gated DeltaNet short convolution runs a slow reference path, and
training becomes several times slower. It usually has to be built against your local CUDA toolkit:

```bash
pip install causal-conv1d --no-build-isolation
python -c "import causal_conv1d, fla; print('ok')"
```

All commands below are run from the repository root.

## Inference

### Trained models

All three RetiJev models are hosted in
[BoKelvin/RetiJev](https://huggingface.co/BoKelvin/RetiJev), one subfolder per size:

| Model | Base model | `--run` |
|---|---|---|
| RetiJev-0.8B | [Qwen/Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base) | `BoKelvin/RetiJev/0.8B` |
| RetiJev-2B | [Qwen/Qwen3.5-2B-Base](https://huggingface.co/Qwen/Qwen3.5-2B-Base) | `BoKelvin/RetiJev/2B` |
| RetiJev-9B | [Qwen/Qwen3.5-9B-Base](https://huggingface.co/Qwen/Qwen3.5-9B-Base) | `BoKelvin/RetiJev/9B` |

Each model folder contains the LoRA adapter and the pointer head (`head.pt`). The base model is downloaded from Hugging Face
on first use. To use a local copy of the base weights instead, pass `--base /path/to/Qwen3.5-0.8B-Base`. `--run` also
accepts a local checkpoint directory produced by `scripts/train.sh`.

### One case

A request is a JSON object holding the case text and the questions, without labels:

```json
{
  "state": "A 38-year-old man ... presents with a 10-week history of progressive, painless, bilateral visual loss ...",
  "questions": {
    "inv_oct_macula_or_rnfl_acceptable": {
      "type": "noul",
      "instructions": "In the current state, is 'OCT of the macula and/or optic nerve head RNFL' an acceptable next investigation?",
      "criteria": {"true": "acceptable or preferred now", "false": "not appropriate now"}
    },
    "inv_preferred": {
      "type": "choice",
      "instructions": "What is the most appropriate next investigation for the decision at hand? Choose one.",
      "criteria": {
        "laboratory_or_serology_workup": "Laboratory / serology work-up (inflammatory and infectious markers)",
        "orbital_or_neuro_imaging_ct_mri": "Orbital or neuro-imaging (CT or MRI)",
        "oct_macula_or_rnfl": "OCT of the macula and/or optic nerve head RNFL",
        "outside_catalog_escalate": "Outside this catalog; escalate for a human decision"
      }
    }
  }
}
```

For a `score` question, `criteria` is the ordered list of levels, lowest first. See [`examples/`](examples) for two complete
requests, one decision case and one diabetic retinopathy grading case.

```bash
python -m retijev.demo --run BoKelvin/RetiJev/0.8B --input examples/next_investigation.json
python -m retijev.demo --run BoKelvin/RetiJev/9B --input examples/dr_grading.json
```

The output has one entry per question:

- `noul`: `{"noul": p_true}`
- `choice`: the arg-max option and the probability of every option
- `score`: the expected level and the probability of every level

### A labelled dataset

```bash
RUN=BoKelvin/RetiJev/0.8B DATA=data/sample SPLIT=development bash scripts/predict.sh
```

This writes one JSON line per question, with the gold answer, the predicted answer and the full probability vector,
and prints the micro accuracy. Reading a split named `test` requires `--allow-test`.

## Training

```bash
bash scripts/train.sh                                   # RetiJev-0.8B recipe on data/sample
BASE=Qwen/Qwen3.5-2B-Base OUT=runs/retijev-2b bash scripts/train.sh
BASE=Qwen/Qwen3.5-9B-Base OUT=runs/retijev-9b bash scripts/train.sh
NPROC=4 bash scripts/train.sh                           # data parallel over 4 GPUs (torchrun)
```

The script runs the RetiJev recipe for all three sizes:

- 3 epochs, batch 4, learning rate 3e-5, bf16, gradient checkpointing
- LoRA rank 64 on the attention, MLP and Gated DeltaNet projections
- ordinal term weight 0.3
- distillation weight 0.5 at temperature 2
- paraphrase-consistency weight 0.5

Every option can be changed with environment variables (`BASE`, `DATA`, `TEACHER`, `PARA_GROUPS`, `OUT`, `KD_W`,
`CONS_W`, `NPROC`). Extra arguments go to the underlying trainer, for example
`bash scripts/train.sh --epochs 1 --max_records 100`. Run `python -m medjev.train --help` for the full list.

Distillation and the consistency term are optional:

- To train without a teacher, set `KD_W=0`, or leave out `teacher_train.jsonl` from the data directory.
- To train without the consistency term, set `CONS_W=0`, or leave out `groups.json`.

The output directory holds the LoRA adapter, `head.pt` and the tokenizer, and can be passed to `--run` directly.

### Data

`data/sample/` is a small subset of the RetiJev training data, released so the full pipeline can be run end to end:

| File | Content |
|---|---|
| `train.jsonl` | 438 training cases (1,061 questions) covering all three question families, including 28 paraphrased cases |
| `development.jsonl` | 120 development cases (274 questions) |
| `teacher_train.jsonl` | teacher soft probabilities for the training questions, used by the distillation term |
| `groups.json` | `{paraphrase idx: original idx}`, used by the consistency term |

A subset this small only checks that the pipeline runs. It is not enough to train a useful model.

Records use the following format, one JSON object per line:

```json
{"idx": "SYN_1535_005",
 "state": "<free-text case>",
 "questions": {
   "<question id>": {"type": "noul | choice | score",
                     "instructions": "<question text>",
                     "criteria": "<options, see below>",
                     "label": "<gold answer>",
                     "src": "<provenance tag>"}}}
```

| Type | `criteria` | `label` |
|---|---|---|
| `noul` | `{"true": "...", "false": "..."}` | `true` or `false` |
| `choice` | `{"<option id>": "<description>", ...}` | an option id |
| `score` | `["<level 0>", "<level 1>", ...]` | a 0-based level index |

The model only sees `state`, `instructions`, `type` and `criteria`. `src` is a provenance tag. Distillation is skipped
for questions whose `src` contains `scales_v` (the code-computed grading labels). Records whose `idx` ends in `~p` are
paraphrases. They share the teacher probabilities of their original, which is identified through `groups.json`.

Teacher rows look like `{"idx": "...", "qid": "...", "probs": {"<option>": p, ...}}`. For `noul` questions the option
keys are `"True"` and `"False"`. Questions without a teacher row are trained without the distillation term.

To train on your own data, write `train.jsonl` and `development.jsonl` in this format into one directory and set `DATA`
to it.

## Acknowledgements

RetiJev builds on the decision-model architecture and training code of [MedJev](https://github.com/JunMa11/MedJev) and
[kev](https://github.com/jaredpalmer/kev). The base models are [Qwen3.5](https://huggingface.co/Qwen).

## License

The code is released under the [Apache License 2.0](LICENSE).

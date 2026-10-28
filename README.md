# DIAL-OPD

Probability-aware token selection for sampled-token on-policy distillation.
DIAL-OPD combines teacher–student disagreement with its probability scale:

$$
s_{\beta,t}=L(p_t,q_t)^\beta\left|\log(q_t/p_t)\right|,
\qquad L(p,q)=\frac{q-p}{\log q-\log p}.
$$

Within each response, the highest-scoring fraction of valid tokens contributes
to the original **signed** OPD loss. All response tokens remain in context.
`beta=0` selects by reward magnitude; `beta=1` selects by the absolute probability
gap. The implementation computes scores in log space for numerical stability.

## Status

This release restores the original DIAL selector and training logic and supplies
a new portable entry point. It has been tested on one NVIDIA H20 with a
Qwen3-4B teacher and Qwen3-0.6B-Base student. Tests include real rollouts,
teacher scoring, gradient updates, vLLM sleep/wake-up, periodic math evaluation,
weight synchronization, and save/reload. A 32-response batch with a 1,024-token
completion limit passes two optimizer updates.

These are runtime validation tests, **not a rerun of the paper's experiments**.
Multi-GPU training and all historical baselines are not included in the supported
entry point. See [validation](docs/VALIDATION.md) and
[source provenance](docs/PROVENANCE.md).

## Installation

Use Linux, Python 3.10, and a CUDA driver compatible with PyTorch's CUDA 12.8
build. Create a dedicated environment in your own workspace:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-gpu.txt
export PYTHONPATH="$PWD:$PWD/trl${PYTHONPATH:+:$PYTHONPATH}"
```

The vendored TRL framework is pinned; do not replace it with pip's latest TRL.
The tested stack uses PyTorch 2.10.0, vLLM 0.19.1, and Transformers 5.14.1.
`requirements-lock.txt` records the complete validated environment.

vLLM 0.19.1's NVML helper requires a small compatibility fix when
`CUDA_VISIBLE_DEVICES` contains a GPU UUID. Apply it to this dedicated venv:

```bash
python scripts/patch_vllm_uuid.py
```

The script refuses to patch a non-venv installation or a different vLLM version.
It records the before/after hashes in `recovery/vllm_uuid_patch.json`.

## Prepare models and data

```bash
python scripts/download_assets.py
```

This downloads the pinned Qwen3-0.6B-Base/Qwen3-4B model revisions and the
conversation-form DeepScaleR training/MATH evaluation files used by the restored
PowerOPD framework. The training file contains 40,313 examples; the evaluation
file contains the 5,000-example MATH test set. The quick test uses only two
evaluation examples. Dataset source hashes are verified during download.

Custom training JSONL should contain a `messages` list per example, for example:

```json
{"messages": [{"role": "user", "content": "What is 2 + 2?"}], "answer": "4"}
```

Teacher and student must share the same tokenizer vocabulary and token IDs.
For the tested base student, the teacher's chat template is used with thinking
disabled. Put any custom models, datasets, caches, and outputs in your own
authorized workspace.

## Run a small training test

Choose **one free GPU** and run from the repository root:

```bash
export CUDA_VISIBLE_DEVICES=0  # replace with your allocated GPU index or UUID
export VLLM_ENABLE_V1_MULTIPROCESSING=0
python -m dial_opd.train --config configs/smoke_vllm.json
```

The smoke config uses vLLM with sleep enabled and runs three optimizer updates.
It evaluates before training and at step 2. Results include `trainer_state.json`,
`selection_metrics.jsonl`, `rollouts.jsonl.gz`, evaluation details, and a loadable
checkpoint under `final/`. Use a **new `output_dir`** for each run; overwriting
an existing result directory is rejected.

`configs/train_qwen3_06b.json` is a longer-run starting configuration with
batch size 32 and 1,024-token rollouts. Set the model/data paths, output directory,
retention ratio, and selection exponent for your experiment. It is not a claim
that launching this file reproduces every paper table.

Key configuration fields:

| Field | Meaning |
|---|---|
| `score_beta` | DIAL probability-weighting exponent |
| `keep_ratio` | Fraction of valid response tokens retained per response |
| `training.max_steps` | Number of optimizer updates |
| `training.generation_batch_size` | Responses generated for a rollout batch |
| `training.gradient_accumulation_steps` | Microbatches per optimizer update |
| `training.vllm_enable_sleep_mode` | Release vLLM memory between rollout and training |
| `log_rollouts` | Save token IDs, log probabilities, and selected positions |
| `save_final` | Save the final student model and tokenizer |

The inherited GOLD `training.beta` is **not** DIAL's `score_beta`; the entry
point fixes GOLD's objective parameters. Selection does not turn rewards into
absolute values in the loss. The `0.01` low-low threshold is diagnostic only.

## Validate

```bash
python -m unittest discover -s tests -p 'test_selection.py' -v
CUDA_VISIBLE_DEVICES='' python tests/verify_integration.py
python scripts/audit_smoke.py outputs/smoke-vllm --report outputs/smoke-audit.json
```

The integration test imports the actual restored trainers and checks signed
gradients, stop-gradient behavior, response masks, and normalization across
unequal microbatches. The artifact audit recomputes token rankings independently
using 70-digit arithmetic. The optional `verify_runtime` config flag checks
Qwen3 weight synchronization, saved weights, and checkpoint reload/generation.

## Confined environments

If your environment uses Landlock ABI 1 and blocks global shared-memory files
and cross-directory renames, the tested single-process adapters are:

```bash
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export DIAL_CONFINED_SINGLE_PROCESS=1
export TRITON_CACHE_MANAGER=dial_opd.confined_cache:SameDirectoryCacheManager
```

They keep a single-process executor lock local and publish compiled cache files
atomically within one directory. They do not disable filesystem confinement or
enable access to other GPUs. These options are unnecessary on ordinary
unconfined installations and are not intended for multiprocessing engines.

## Code layout

```text
dial_opd/          DIAL selector, recovered trainer, new launcher/adapters
factorized_gold/   Shared rollout, scoring, and evaluation framework
opd_dynamics/      Sampled-token reward/loss and terminal-token handling
trl/              Pinned TRL runtime source and upstream license
configs/          Smoke and longer-run configurations
tests/            Selector and actual-trainer numerical checks
scripts/          Asset download, environment patch, artifact audit
docs/             Provenance and validation scope
```

The framework derives from [PowerOPD](https://github.com/EIT-NLP/PowerOPD) and
[Hugging Face TRL](https://github.com/huggingface/trl). Vendored TRL retains its
Apache-2.0 license and file-level notices.

# Running DIAL-OPD

[← Back to the README](../README.md)

## Installation

Use Linux, Python 3.10, and a CUDA driver compatible with PyTorch's CUDA 12.8
build. Create a dedicated environment in your own workspace:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-gpu.txt
python scripts/install_trl.py
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

The installer places the pinned TRL runtime inside your active venv, outside
this repository. It verifies the immutable source archive checksum. Do not
replace it with pip's latest TRL.
The pinned stack uses PyTorch 2.10.0, vLLM 0.19.1, and Transformers 5.14.1.
`requirements-lock.txt` records the dependency versions.

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
conversation-form DeepScaleR training/MATH evaluation files from
[PowerOPD](https://github.com/EIT-NLP/PowerOPD). The training file contains 40,313 examples; the evaluation
file contains the 5,000-example MATH test set. The quick test uses only two
evaluation examples. Dataset source hashes are verified during download.

Custom training JSONL should contain a `messages` list per example, for example:

```json
{"messages": [{"role": "user", "content": "What is 2 + 2?"}], "answer": "4"}
```

Teacher and student must share the same tokenizer vocabulary and token IDs.
For the Qwen3 base student, the teacher's chat template is used with thinking
disabled.

## Run a small training test

The provided launcher uses one GPU. Run from the repository root:

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
retention ratio, and selection exponent for your experiment. Its default evaluation
uses the full MATH test file; use a MATH-500 subset to evaluate that benchmark.

Key configuration fields:

| Field | Meaning |
|---|---|
| `score_beta` | DIAL probability-weighting exponent |
| `keep_ratio` | Fraction of valid response tokens retained per response |
| `training.max_steps` | Number of optimizer updates |
| `training.generation_batch_size` | Responses generated for a rollout batch |
| `training.gradient_accumulation_steps` | Microbatches per optimizer update |
| `training.vllm_enable_sleep_mode` | Release vLLM memory between rollout and training |
| `log_rollouts` | Optional token IDs, log probabilities, and selected positions (default: off) |
| `save_final` | Save the final student model and tokenizer |

The inherited GOLD `training.beta` is **not** DIAL's `score_beta`; the entry
point fixes GOLD's objective parameters. Selection does not turn rewards into
absolute values in the loss. The `0.01` low-low threshold is diagnostic only.

## Tests

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
CUDA_VISIBLE_DEVICES='' python tests/verify_integration.py
python scripts/audit_smoke.py outputs/smoke-vllm --report outputs/smoke-audit.json
```

The integration test imports the actual DIAL trainer and checks signed
gradients, stop-gradient behavior, response masks, and normalization across
unequal microbatches. The artifact audit recomputes token rankings independently
using 70-digit arithmetic. With `log_rollouts=true`, the artifact audit checks each selected token.
The optional `verify_runtime` config flag checks
Qwen3 weight synchronization, saved weights, and checkpoint reload/generation.

## Confined environments

If your environment uses Landlock ABI 1 and blocks global shared-memory files
and cross-directory renames, use these single-process settings:

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
dial_opd/          DIAL scoring/loss, student rollouts, evaluation, and runtime adapters
configs/          Smoke and longer-run configurations
tests/            Selector and actual-trainer numerical checks
scripts/          Asset download, environment patch, artifact audit
docs/             Installation and usage guide
```

The framework derives from [PowerOPD](https://github.com/EIT-NLP/PowerOPD) and
[Hugging Face TRL](https://github.com/huggingface/trl). The separately installed TRL snapshot retains its
Apache-2.0 license and file-level notices.

## Runtime dependency

The TRL dependency uses the PowerOPD fork at commit
`65819e14a49c940fa7b2977f07872cabd6229129`, with model helpers from Hugging Face
TRL v1.1.0 (`3179965b2d0621c1778f5a561302905837627b60`). The installer retrieves
this combination from a fixed archive and checks its SHA-256 before installing.
Upstream license files and notices are included in the installed dependency.

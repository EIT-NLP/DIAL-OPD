# Runtime validation

The DIAL-only cleanup was validated with the same pinned runtime as the original
release: Linux, Python 3.10, one NVIDIA H20, PyTorch 2.10.0+cu128, vLLM 0.19.1,
and Transformers 5.14.1. All GPU tests used a Qwen3-4B teacher and Qwen3-0.6B-Base
student on a single GPU.

## Numerical checks

- Selector and configuration checks cover score endpoints, extreme tails,
  probability equality, stable ties, per-response quotas, invalid inputs, and
  rejection of unsupported objectives and removed experiment options.
- Actual-trainer integration checks cover signed gradients, detached teacher
  rewards, both terminal EOS types, prompt/padding masks, EOS-only responses,
  and the shared denominator across unequal microbatches.
- 96 comparisons against the untouched first-release implementation span eight
  beta values, four retention ratios and three input seeds. Selected masks,
  rewards, denominator tensors, losses and student gradients are exactly equal
  (zero absolute and relative tolerance). Token dumps and selection metrics also
  match; disabling token dumps leaves losses and metrics unchanged.

See `cleanup-integration.json` and `cleanup-equivalence.json`.

## Real model execution

- vLLM: three optimizer updates, six responses, sleep/wake, evaluation before
  training and at step 2, and final checkpoint saving.
- Independent artifact audit: 70-digit selection arithmetic, quotas, shared loss
  denominator, finite losses/nonzero gradients, and changed saved weights.
- All 596,049,920 vLLM parameter elements match the current student at engine dtype.
  All serialized parameters match the trained student. Reload and generation pass.
- Transformers generation: two updates with token dumps disabled.

See `cleanup-gpu-smoke.json`, `cleanup-runtime.json`, and `cleanup-hf.json`.

These are functional and numerical checks, not a reproduction of the paper's
800-step training or benchmark accuracy. The cleanup has not been stress-tested
at the previous 32-response/1,024-token configuration or on multiple GPUs.
The pinned TRL snapshot emits an upstream vLLM-version compatibility warning;
the tests above cover the documented vLLM 0.19.1 path with the included adapters.

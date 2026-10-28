# Runtime validation

Validated on one NVIDIA H20, Linux, Python 3.10, PyTorch 2.10.0+cu128,
vLLM 0.19.1, Transformers 5.14.1. TRL is the pinned vendored source.

- Real-class CPU gradient/mask/normalization tests across eight beta values.
- Transformers generation and three optimizer updates.
- vLLM compilation, CUDA Graph capture, generation, and three optimizer updates.
- vLLM level-2 sleep/wake and math evaluation at steps 0 and 2.
- Two updates using batch size 32 and max completion length 1,024 (64 responses).
- Independent 70-digit token-ranking audit, token quota and denominator checks.
- All 596,049,920 vLLM parameter elements match the current student at engine dtype.
- All serialized model parameters match the trained student; checkpoint reload
  and generation succeed.

See `gpu-smoke-audit.json` and `runtime-verification.json` for machine-readable
results. The latter's generated text is a functional check, not a quality score.

The tests do not reproduce the paper's 800-step experiments or seven-benchmark
results. They do not validate multi-GPU execution. The bundled longer-run config
is a starting point; its larger evaluation context has not been stress-tested.

The curated public tree itself also passes selector tests, actual-trainer CPU
checks, data download/hash verification, and a three-update GPU smoke run. See
`public-release-smoke.json` for its independent artifact audit.

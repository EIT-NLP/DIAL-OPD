# DIAL-only cleanup

The complete pre-cleanup checkout and original recovery archive are preserved
separately and remain unchanged. The pre-cleanup public commit is `881d86f06ca908bf9deedfecbd29bea8c4fe650c`.

## Removed from the DIAL implementation

- Alternative clipped/tanh/probability/square-root/power rewards and their stats.
- Reward z-score and sign-wise normalization; alternate microbatch loss reduction.
- Full-vocabulary KL and REOPOLD configuration classes.
- Contrastive-filter, token-type-mask and diff/reward-mask experiment options.
- Teacher quantization/adapter experiment fields unused by the DIAL launcher.
- Teacher-generated and dataset-provided trajectory branches.
- Cross-EOS relabeling, terminal canonicalization, and unmasked-terminal alternatives.
- Reward-position plotting/tracking and redundant intermediate reward tensors.
- Separate `opd_dynamics/` and `factorized_gold/` package hierarchy.

These features are removed, not retained behind disabled switches. Corresponding
legacy configuration keys now fail rather than silently changing behavior.

## Retained

- The original numerically stable DIAL scoring function, byte-for-byte unchanged.
- Per-response top-k selection, original quota rounding and stable tie-breaking.
- Signed log-ratio rewards, detached teacher/reward tensors, and shared selected-token
  normalization across gradient accumulation. No token is removed from context.
- Student rollouts via vLLM or Transformers; teacher scoring; prompt/padding and
  terminal-EOS loss masking; empty-window and alignment checks.
- Optimizer/checkpoint support, vLLM weight synchronization and sleep/wake.
- Math evaluation, optional token dumps, compact selection metrics and CPU checks.
- Narrow environment adapters needed by the validated runtime.

`DIALTrainer -> OnPolicyTrainer -> GOLDTrainer` replaces the previous additional
OPDDynamics layer. Scoring and the actual signed loss are under `dial_opd/`.

## Dependency and logging changes

The complete TRL source is no longer vendored in this repository. An immutable,
checksum-verified copy of the same runtime is installed under the active venv.
This moves third-party code out of the repository; it does not claim to remove
other trainers from the TRL dependency or switch to a new TRL version.

Token dumps are opt-in (`log_rollouts`, default false); the smoke config enables
them for independent auditing. Disabling dumps skips serialization entirely.
Selection metrics use the clearer `dial_*` prefix instead of `score_family_*`.
Legacy per-variant reward diagnostics were removed; rollout/evaluation metrics,
selected counts, loss and gradient norms remain available.

## Size and validation

Application Python source decreased from 3,182 to 1,910 lines (40.0%). This count
excludes third-party TRL, tests, scripts and documentation, so it represents actual
application simplification rather than only moving a dependency.

See [validation](VALIDATION.md) for numerical equivalence and real-model tests.

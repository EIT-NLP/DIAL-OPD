# Source provenance

- `dial_opd/selection.py` is byte-identical to the recovered DIAL selector and
  the first public release (`881d86f06ca908bf9deedfecbd29bea8c4fe650c`).
- `dial_opd/trainer.py` consolidates the recovered selection and sampled-token
  loss classes. It removes alternative objectives and fixes the signed log-ratio
  reward, terminal-EOS masking, and window-wide selected-token normalization.
- `dial_opd/rollout.py` derives from the recovered FactorizedGOLD trainer, retaining
  student rollouts and evaluation integration. Teacher/dataset rollout branches
  were removed. Evaluation helpers and stop-token handling moved into `dial_opd/`.
- Configuration, launcher, environment adapters and verification utilities are
  maintained code; they are not presented as unmodified historical files.

## Pinned runtime dependency

The runtime derives from EIT-NLP/PowerOPD commit
`65819e14a49c940fa7b2977f07872cabd6229129`. Its missing TRL model helpers came from
Hugging Face TRL v1.1.0 commit `3179965b2d0621c1778f5a561302905837627b60`.
The complete validated combination is archived in the immutable first public
DIAL-OPD release. `scripts/install_trl.py` extracts only its `trl/` directory,
verifies the archive SHA-256, and registers that source using a `.pth` file in
the active virtual environment. It does not install stock/newer TRL, change
other dependencies, or write into base Python. Runtime licenses and upstream
headers remain intact in the installed dependency.

Other TRL trainers remain part of that third-party dependency, not part of the
DIAL algorithm or the source tree users navigate in this repository.

Original recovery archives and the complete pre-cleanup checkout are preserved
outside this working tree. See [cleanup](CLEANUP.md) and [validation](VALIDATION.md).

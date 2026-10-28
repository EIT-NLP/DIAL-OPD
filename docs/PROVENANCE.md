# Source provenance

- `dial_opd/selection.py`: byte-identical recovered DIAL selector.
- `dial_opd/trainer.py`: original ScoreFamilySelectionTrainer class, with import glue adapted.
- `opd_dynamics/trainer.py` and `factorized_gold/trainer.py`: unchanged recovered classes.
- Shared framework/TRL: EIT-NLP/PowerOPD commit
  `65819e14a49c940fa7b2977f07872cabd6229129`.
- Missing `trl/trl/models/*.py`: Hugging Face TRL v1.1.0 commit
  `3179965b2d0621c1778f5a561302905837627b60`.
- `factorized_gold/math_eval_utils.py`: complete PowerOPD helper plus the
  recovered process-isolated MathVerifier; IPC uses pipes for confined hosts.
- Evaluation callback changes: wake KV cache after weight sync for vLLM 0.19;
  preserve pending train logs and avoid consuming training metric buffers.
- New: training entry point, environment/cache adapters, download scripts,
  optional runtime verification, and release documentation/configurations.

The selection/loss algorithm was not rewritten to fit the new environment.
Historical machine-specific launchers, recovered private history, model weights,
and experiment trajectories are excluded from this export.

TRL files retain their upstream headers and Apache-2.0 license in `trl/LICENSE`.

The public dependency lock records `pwinput==1.0.3` by version instead of the
local wheel path used during installation. No local wheel URL is required.

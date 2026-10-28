"""Training and evaluation configuration for sampled-token DIAL-OPD."""
from dataclasses import dataclass, field
from trl.experimental.gold.gold_config import GOLDConfig


@dataclass
class DIALConfig(GOLDConfig):
    beta: float = 1.0
    lmbda: float = 1.0
    eval_test_names: str = field(
        default="",
        metadata={"help": "Comma-separated evaluation dataset names."},
    )
    eval_test_paths: str = field(
        default="",
        metadata={"help": "Comma-separated JSON/JSONL paths for evaluation datasets."},
    )
    eval_test_samples: str = field(
        default="all",
        metadata={"help": "Sample limits per evaluation dataset. Single value or comma-separated list."},
    )
    eval_temperature: float = field(
        default=0.0,
        metadata={"help": "Sampling temperature for evaluation generations."},
    )
    eval_n: int = field(
        default=1,
        metadata={"help": "How many responses to sample per prompt during evaluation."},
    )
    eval_k: int = field(
        default=1,
        metadata={
            "help": "How many of the sampled responses are used for metrics. The evaluator reports acc=avg@k and pass=pass@k, and requires eval_n >= eval_k."
        },
    )
    eval_max_new_tokens: int | None = field(
        default=None,
        metadata={"help": "Max new tokens during eval. Falls back to max_completion_length when unset."},
    )
    enable_thinking: bool | None = field(
        default=False,
        metadata={"help": "Whether chat template rendering should enable thinking mode."},
    )

    def __post_init__(self):
        # These inherited GOLD options would change the DIAL training objective.
        for name, expected in {'beta': 1.0, 'lmbda': 1.0, 'num_generations': 1,
                               'use_uld_loss': False, 'seq_kd': False,
                               'use_liger_kernel': False}.items():
            if getattr(self, name) != expected:
                raise ValueError(f"DIAL requires {name}={expected}")
        super().__post_init__()
        if not 1 <= self.eval_k <= self.eval_n:
            raise ValueError("Evaluation requires 1 <= eval_k <= eval_n")

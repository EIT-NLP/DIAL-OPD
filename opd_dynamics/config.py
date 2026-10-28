from __future__ import annotations

from dataclasses import dataclass, field

from factorized_gold.config import FactorizedGOLDConfig

from .terminal_stop import VALID_TERMINAL_STOP_TARGETS


@dataclass
class OPDDynamicsConfig(FactorizedGOLDConfig):
    reward_mode: str = field(
        default="sampled_log_ratio",
        metadata={
            "help": (
                "Sampled-token reward: `sampled_log_ratio`, `sampled_log_clip`, `sampled_log_tanh`, "
                "`sampled_prob_diff`, `sampled_sqrt_diff`, or `sampled_power_diff`."
            )
        },
    )
    log_reward_clip_min: float = field(
        default=-5.0,
        metadata={"help": "Lower bound for sampled_log_clip reward."},
    )
    log_reward_clip_max: float = field(
        default=5.0,
        metadata={"help": "Upper bound for sampled_log_clip reward."},
    )
    log_reward_tanh_temperature: float = field(
        default=5.0,
        metadata={"help": "Temperature divisor for sampled_log_tanh reward."},
    )
    power_reward_alpha: float = field(
        default=0.1,
        metadata={"help": "Alpha for sampled_power_diff reward: p_T(o_t)^alpha - p_S(o_t)^alpha."},
    )
    token_loss_normalization: str = field(
        default="microbatch_mean",
        metadata={"help": "Token loss denominator: `microbatch_mean` or `window_token_mean`."},
    )
    reward_normalization: str = field(
        default="none",
        metadata={"help": "Reward normalization over the accumulation window: `none`, `sign_max_abs`, or `zscore`."},
    )
    terminal_stop_target: str = field(
        default="im_end",
        metadata={"help": "Terminal stop handling: `raw`, `mask`, `im_end`, or `cross_eos`."},
    )
    reward_position_plot_steps: int = field(
        default=0,
        metadata={"help": "Save reward-by-token-index plots every N optimizer steps. 0 disables."},
    )
    reward_position_max_index: int = field(
        default=0,
        metadata={"help": "Maximum completion token index count to plot. 0 uses max_completion_length."},
    )
    reward_position_smooth_window: int = field(
        default=16,
        metadata={"help": "Token-index smoothing window for reward position plots."},
    )
    teacher_load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Load the teacher model with bitsandbytes 4-bit weights for scoring-only forward passes."},
    )
    teacher_load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Load the teacher model with bitsandbytes 8-bit weights for scoring-only forward passes."},
    )
    teacher_bnb_4bit_quant_type: str = field(
        default="nf4",
        metadata={"help": "bitsandbytes 4-bit quantization type for the teacher model, e.g. `nf4` or `fp4`."},
    )
    teacher_bnb_4bit_compute_dtype: str = field(
        default="bfloat16",
        metadata={"help": "Compute dtype for teacher bitsandbytes 4-bit layers."},
    )
    teacher_use_bnb_nested_quant: bool = field(
        default=False,
        metadata={"help": "Whether to use bitsandbytes nested/double quantization for the teacher model."},
    )
    teacher_bnb_8bit_threshold: float = field(
        default=6.0,
        metadata={"help": "bitsandbytes llm_int8_threshold for teacher 8-bit loading."},
    )
    teacher_lora_adapter_path: str | None = field(
        default=None,
        metadata={"help": "Optional PEFT LoRA adapter path to attach to the teacher model for scoring."},
    )
    selective_filter_enabled: bool = field(
        default=False,
        metadata={"help": "Enable sampled-token selective probability-difference filtering."},
    )
    selective_filter_teacher_model_name_or_path: str | None = field(
        default=None,
        metadata={"help": "Teacher model used only for selective token filtering."},
    )
    selective_filter_threshold: float = field(
        default=0.25,
        metadata={"help": "Drop tokens with abs(p_reward_teacher - p_filter_teacher) below this threshold."},
    )
    selective_filter_mode: str = field(
        default="prob_diff_lt",
        metadata={
            "help": "`prob_diff_lt` keeps tokens whose |p_reward - p_filter| clears "
            "selective_filter_threshold; `prob_diff_topk` instead keeps the top "
            "selective_filter_keep_rate fraction by that same score."
        },
    )
    selective_filter_keep_rate: float = field(
        default=0.4,
        metadata={"help": "prob_diff_topk only: fraction of completion tokens kept per sequence."},
    )
    selective_filter_debug_tokens: int = field(
        default=20,
        metadata={"help": "Number of first completion tokens to print for selective filter prompt/mask debug."},
    )
    token_type_mask_mode: str = field(
        default="vanilla",
        metadata={
            "help": (
                "Token-type loss mask mode: `vanilla`, `mask_artifact`, `mask_markdown`, "
                "`mask_both`, or `mask_random`."
            )
        },
    )
    token_type_mask_random_seed: int = field(
        default=42,
        metadata={"help": "Random seed for count-matched token-type random mask."},
    )
    token_type_mask_debug_tokens: int = field(
        default=20,
        metadata={"help": "Number of first completion tokens to print for token-type mask debug."},
    )
    diff_reward_mask_mode: str = field(
        default="vanilla",
        metadata={
            "help": (
                "Diff-vs-reward mask ablation mode: `vanilla`, `mask_low_diff`, "
                "or `mask_extreme_reward`."
            )
        },
    )
    diff_reward_mask_threshold: float = field(
        default=0.01,
        metadata={"help": "Low-diff threshold for diff-vs-reward mask ablation."},
    )
    diff_reward_mask_debug_tokens: int = field(
        default=20,
        metadata={"help": "Number of first completion tokens to print for diff-vs-reward mask debug."},
    )

    def __post_init__(self):
        super().__post_init__()
        valid = {
            "sampled_log_ratio",
            "sampled_log_clip",
            "sampled_log_tanh",
            "sampled_prob_diff",
            "sampled_sqrt_diff",
            "sampled_power_diff",
        }
        if self.reward_mode not in valid:
            raise ValueError(f"reward_mode must be one of {sorted(valid)}, got {self.reward_mode!r}.")
        if self.log_reward_clip_min >= self.log_reward_clip_max:
            raise ValueError("log_reward_clip_min must be smaller than log_reward_clip_max.")
        if self.log_reward_tanh_temperature <= 0:
            raise ValueError("log_reward_tanh_temperature must be > 0.")
        if self.power_reward_alpha <= 0:
            raise ValueError("power_reward_alpha must be > 0.")
        valid_token_loss_normalization = {"microbatch_mean", "window_token_mean"}
        if self.token_loss_normalization not in valid_token_loss_normalization:
            raise ValueError(
                f"token_loss_normalization must be one of {sorted(valid_token_loss_normalization)}, "
                f"got {self.token_loss_normalization!r}."
            )
        valid_reward_normalization = {"none", "sign_max_abs", "zscore"}
        if self.reward_normalization not in valid_reward_normalization:
            raise ValueError(
                f"reward_normalization must be one of {sorted(valid_reward_normalization)}, "
                f"got {self.reward_normalization!r}."
            )
        if self.terminal_stop_target not in VALID_TERMINAL_STOP_TARGETS:
            raise ValueError(
                f"terminal_stop_target must be one of {sorted(VALID_TERMINAL_STOP_TARGETS)}, "
                f"got {self.terminal_stop_target!r}."
            )
        if self.reward_position_plot_steps < 0:
            raise ValueError("reward_position_plot_steps must be >= 0.")
        if self.reward_position_max_index < 0:
            raise ValueError("reward_position_max_index must be >= 0.")
        if self.reward_position_smooth_window < 1:
            raise ValueError("reward_position_smooth_window must be >= 1.")
        if self.teacher_load_in_4bit and self.teacher_load_in_8bit:
            raise ValueError("Only one of teacher_load_in_4bit and teacher_load_in_8bit can be enabled.")
        if self.selective_filter_threshold < 0:
            raise ValueError("selective_filter_threshold must be >= 0.")
        if self.selective_filter_mode not in ("prob_diff_lt", "prob_diff_topk"):
            raise ValueError(
                "selective_filter_mode must be `prob_diff_lt` or `prob_diff_topk`, "
                f"got {self.selective_filter_mode!r}."
            )
        if self.selective_filter_mode == "prob_diff_topk" and not (0.0 < self.selective_filter_keep_rate <= 1.0):
            raise ValueError("selective_filter_keep_rate must be in (0, 1].")
        if self.selective_filter_debug_tokens < 0:
            raise ValueError("selective_filter_debug_tokens must be >= 0.")
        valid_token_type_mask_modes = {"vanilla", "mask_artifact", "mask_markdown", "mask_both", "mask_random"}
        if self.token_type_mask_mode not in valid_token_type_mask_modes:
            raise ValueError(
                f"token_type_mask_mode must be one of {sorted(valid_token_type_mask_modes)}, "
                f"got {self.token_type_mask_mode!r}."
            )
        if self.token_type_mask_debug_tokens < 0:
            raise ValueError("token_type_mask_debug_tokens must be >= 0.")
        valid_diff_reward_mask_modes = {"vanilla", "mask_low_diff", "mask_extreme_reward"}
        if self.diff_reward_mask_mode not in valid_diff_reward_mask_modes:
            raise ValueError(
                f"diff_reward_mask_mode must be one of {sorted(valid_diff_reward_mask_modes)}, "
                f"got {self.diff_reward_mask_mode!r}."
            )
        if self.diff_reward_mask_threshold < 0:
            raise ValueError("diff_reward_mask_threshold must be >= 0.")
        if self.diff_reward_mask_debug_tokens < 0:
            raise ValueError("diff_reward_mask_debug_tokens must be >= 0.")


@dataclass
class FullVocabKLConfig(OPDDynamicsConfig):
    kl_direction: str = field(
        default="reverse",
        metadata={"help": "Full-vocab KL direction: `reverse` for KL(student||teacher), or `forward` for KL(teacher||student)."},
    )
    fullvocab_kl_temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature used for full-vocab KL logits. The KL loss is multiplied by T^2."},
    )
    fullvocab_kl_debug_tokens: int = field(
        default=0,
        metadata={"help": "Print this many decoded completion labels from the first KL batch. 0 disables."},
    )

    def __post_init__(self):
        super().__post_init__()
        if self.kl_direction not in {"reverse", "forward"}:
            raise ValueError(f"kl_direction must be `reverse` or `forward`, got {self.kl_direction!r}.")
        if self.fullvocab_kl_temperature <= 0:
            raise ValueError("fullvocab_kl_temperature must be > 0.")
        if self.fullvocab_kl_debug_tokens < 0:
            raise ValueError("fullvocab_kl_debug_tokens must be >= 0.")


@dataclass
class REOPOLDNoRhoConfig(FactorizedGOLDConfig):
    reopold_lambda: float = field(
        default=0.3,
        metadata={"help": "REOPOLD clipping coefficient lambda. Paper default: 0.3."},
    )
    reopold_entropy_top_fraction: float = field(
        default=0.2,
        metadata={"help": "Keep top beta fraction of token entropies in phase II. Paper default beta: 0.2."},
    )
    reopold_switch_step: int = field(
        default=500,
        metadata={"help": "Switch from reward filtering to entropy filtering."},
    )
    terminal_stop_target: str = field(
        default="im_end",
        metadata={"help": "Terminal stop handling: `raw`, `mask`, `im_end`, or `cross_eos`."},
    )
    reward_position_plot_steps: int = field(
        default=0,
        metadata={"help": "Save reward-by-token-index plots every N optimizer steps. 0 disables."},
    )
    reward_position_max_index: int = field(
        default=0,
        metadata={"help": "Maximum completion token index count to plot. 0 uses max_completion_length."},
    )
    reward_position_smooth_window: int = field(
        default=16,
        metadata={"help": "Token-index smoothing window for reward position plots."},
    )
    teacher_load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Load the teacher model with bitsandbytes 4-bit weights for scoring-only forward passes."},
    )
    teacher_load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Load the teacher model with bitsandbytes 8-bit weights for scoring-only forward passes."},
    )
    teacher_bnb_4bit_quant_type: str = field(
        default="nf4",
        metadata={"help": "bitsandbytes 4-bit quantization type for the teacher model, e.g. `nf4` or `fp4`."},
    )
    teacher_bnb_4bit_compute_dtype: str = field(
        default="bfloat16",
        metadata={"help": "Compute dtype for teacher bitsandbytes 4-bit layers."},
    )
    teacher_use_bnb_nested_quant: bool = field(
        default=False,
        metadata={"help": "Whether to use bitsandbytes nested/double quantization for the teacher model."},
    )
    teacher_bnb_8bit_threshold: float = field(
        default=6.0,
        metadata={"help": "bitsandbytes llm_int8_threshold for teacher 8-bit loading."},
    )
    teacher_lora_adapter_path: str | None = field(
        default=None,
        metadata={"help": "Optional PEFT LoRA adapter path to attach to the teacher model for scoring."},
    )

    def __post_init__(self):
        super().__post_init__()
        if not (0.0 < self.reopold_lambda < 1.0):
            raise ValueError("reopold_lambda must be in (0, 1).")
        if not (0.0 < self.reopold_entropy_top_fraction <= 1.0):
            raise ValueError("reopold_entropy_top_fraction must be in (0, 1].")
        if self.reopold_switch_step < 0:
            raise ValueError("reopold_switch_step must be >= 0.")
        if self.terminal_stop_target not in VALID_TERMINAL_STOP_TARGETS:
            raise ValueError(
                f"terminal_stop_target must be one of {sorted(VALID_TERMINAL_STOP_TARGETS)}, "
                f"got {self.terminal_stop_target!r}."
            )
        if self.reward_position_plot_steps < 0:
            raise ValueError("reward_position_plot_steps must be >= 0.")
        if self.reward_position_max_index < 0:
            raise ValueError("reward_position_max_index must be >= 0.")
        if self.reward_position_smooth_window < 1:
            raise ValueError("reward_position_smooth_window must be >= 1.")
        if self.teacher_load_in_4bit and self.teacher_load_in_8bit:
            raise ValueError("Only one of teacher_load_in_4bit and teacher_load_in_8bit can be enabled.")


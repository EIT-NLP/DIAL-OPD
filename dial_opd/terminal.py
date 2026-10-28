from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase



@dataclass
class TerminalStopResult:
    labels: torch.Tensor
    mask: torch.Tensor
    stats: dict[str, int]


def _token_id(tokenizer: PreTrainedTokenizerBase, token: str) -> int | None:
    value = tokenizer.convert_tokens_to_ids(token)
    if isinstance(value, int) and value >= 0:
        return value
    return None


def _empty_stats() -> dict[str, int]:
    return {
        "terminal_stop_endoftext_count": 0,
        "terminal_stop_im_end_count": 0,
        "terminal_stop_missing_count": 0,
        "terminal_stop_masked_count": 0,
    }


def resolve_terminal_stop_ids(tokenizer: PreTrainedTokenizerBase) -> tuple[int | None, int | None]:
    return _token_id(tokenizer, "<|endoftext|>"), _token_id(tokenizer, "<|im_end|>")


def mask_terminal_eos(
    shifted_labels: torch.Tensor,
    mask: torch.Tensor,
    tokenizer: PreTrainedTokenizerBase,
) -> TerminalStopResult:
    """Exclude a terminal Qwen EOS from loss without changing context tokens."""

    effective_labels = shifted_labels.clone()
    effective_mask = mask.clone()
    endoftext_id, im_end_id = resolve_terminal_stop_ids(tokenizer)
    stats = _empty_stats()

    for row_idx in range(shifted_labels.shape[0]):
        valid_positions = mask[row_idx].nonzero(as_tuple=True)[0]
        if valid_positions.numel() == 0:
            stats["terminal_stop_missing_count"] += 1
            continue

        last_pos = valid_positions[-1]
        last_id = int(shifted_labels[row_idx, last_pos].detach().item())
        is_endoftext = endoftext_id is not None and last_id == endoftext_id
        is_im_end = im_end_id is not None and last_id == im_end_id

        if is_endoftext:
            stats["terminal_stop_endoftext_count"] += 1
        elif is_im_end:
            stats["terminal_stop_im_end_count"] += 1
        else:
            stats["terminal_stop_missing_count"] += 1
            continue

        effective_mask[row_idx, last_pos] = False
        stats["terminal_stop_masked_count"] += 1

    return TerminalStopResult(labels=effective_labels, mask=effective_mask, stats=stats)




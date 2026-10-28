"""Stable per-response selection for the probability-mass score family.

s_beta = L(p, q)**beta * abs(log(q/p)); only the mask changes the OPD loss.
All effective response tokens are candidates. Low-low is diagnostic only.
"""
from __future__ import annotations

import math
from numbers import Real

import torch


def _number(value, name, *, positive=False, at_most_one=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    value = float(value)
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError(f"{name} must be finite and {'> 0' if positive else '>= 0'}")
    if at_most_one and value > 1:
        raise ValueError(f"{name} must be <= 1")
    return value


def _log_scores(student: torch.Tensor, teacher: torch.Tensor, beta: float) -> torch.Tensor:
    """Internal helper on detached float64 valid logp vectors.

    delta = abs(logp-logq) is the reward magnitude.
    For unequal probabilities:
      log(d) = max(logp,logq) + log(-expm1(-delta))
      log(L) = log(d) - log(delta)
      log(s_beta) = log(delta) + beta*log(L).
    expm1 avoids cancellation near equality, and retaining log scores avoids
    underflow for small probabilities. At equality s_beta=0 for every beta,
    hence log(s_beta)=-inf (including beta=0 and beta=1).
    """
    delta = (teacher - student).abs()
    different = delta > 0
    scores = torch.full_like(delta, -torch.inf)
    if not bool(different.any()):
        return scores
    log_reward = delta[different].log()
    if beta == 0.0:
        values = log_reward
    else:
        log_difference = torch.maximum(student[different], teacher[different]) + (
            -torch.expm1(-delta[different])).log()
        if beta == 1.0:
            # Exact APD in log form: sorting log(d) is sorting d, including
            # probabilities too small for an explicit exp(logp) subtraction.
            values = log_difference
        else:
            log_mean = log_difference - log_reward
            values = log_reward + beta * log_mean
    if not bool(torch.isfinite(values).all()):
        raise ValueError("finite inputs exceeded float64 log-score arithmetic range")
    scores[different] = values
    return scores


@torch.no_grad()
def select_score_tokens(
    student_logp: torch.Tensor,
    teacher_logp: torch.Tensor,
    effective_mask: torch.Tensor,
    keep_ratio: float = 0.2,
    score_beta: float = 0.5,
    probability_threshold: float = 0.01,
) -> tuple[torch.Tensor, dict[str, int | float]]:
    """Return a bool selection mask and pooled scalar diagnostics.

    Inputs are [responses, token_positions] tensors on one device. Mask must be
    bool or exactly binary; only valid logp must be finite and <=0. Invalid
    prompt, pad and masked EOS positions may contain NaN and never participate.

    Detach and convert logp to float64 only for scoring/diagnostics; the caller
    keeps its original differentiable tensors for the unchanged OPD loss.
    Each nonempty row keeps max(1, int(round(keep_ratio*n_valid))) tokens
    (Python ties-to-even). Stable sorting breaks ties by original position.
    beta=0 sorts reward magnitude directly; beta=1 sorts stable log(APD).

    probability_threshold changes only low-low diagnostics, never selection.
    Counts and reward-magnitude sums are pooled, additive Python int/float.
    skipped_* are always zero: no response is skipped for low-low content.
    """
    ratio = _number(keep_ratio, "keep_ratio", positive=True, at_most_one=True)
    beta = _number(score_beta, "score_beta")
    threshold = _number(probability_threshold, "probability_threshold",
                        positive=True, at_most_one=True)
    tensors = (student_logp, teacher_logp, effective_mask)
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        raise ValueError("logp and effective_mask must be torch tensors")
    if not all(value.ndim == 2 for value in tensors):
        raise ValueError("inputs must have shape [responses, token_positions]")
    if not all(value.shape == student_logp.shape for value in tensors):
        raise ValueError("input shapes must match")
    if not all(value.device == student_logp.device for value in tensors):
        raise ValueError("inputs must share a device")
    if not all(value.is_floating_point() for value in tensors[:2]):
        raise ValueError("logp tensors must be floating point")
    if effective_mask.dtype != torch.bool:
        if effective_mask.is_complex() or not bool(
            ((effective_mask == 0) | (effective_mask == 1)).all()
        ):
            raise ValueError("effective_mask must be bool or exactly binary")
    valid = effective_mask.bool()
    student = student_logp.detach().to(torch.float64)
    teacher = teacher_logp.detach().to(torch.float64)
    for name, values in (("student_logp", student), ("teacher_logp", teacher)):
        invalid = valid & (~torch.isfinite(values) | (values > 0))
        if bool(invalid.any()):
            row, position = invalid.nonzero(as_tuple=False)[0].tolist()
            raise ValueError(f"{name}: invalid logp at response {row}, position {position}; "
                             "expected finite and <= 0")
    reward_abs = torch.zeros_like(student)
    reward_abs[valid] = (teacher[valid] - student[valid]).abs()
    scores = torch.full_like(student, -torch.inf)
    scores[valid] = _log_scores(student[valid], teacher[valid], beta)
    # For beta=0, avoid collapsing distinct rewards through rounded log values.
    rank_values = reward_abs if beta == 0.0 else scores
    selected = torch.zeros_like(valid)
    for row in range(valid.shape[0]):
        positions = valid[row].nonzero(as_tuple=False).flatten()
        n_valid = positions.numel()
        if n_valid == 0:
            continue
        k = max(1, int(round(ratio * n_valid)))
        order = torch.argsort(rank_values[row, positions], descending=True, stable=True)
        selected[row, positions[order[:k]]] = True
    cutoff = math.log(threshold)
    low_low = valid & (student <= cutoff) & (teacher <= cutoff)
    stats = {
        "valid": int(valid.sum().item()),
        "selected": int(selected.sum().item()),
        "low_low_valid": int(low_low.sum().item()),
        "low_low_selected": int((selected & low_low).sum().item()),
        "r_abs_total": float(reward_abs[valid].sum().item()),
        "r_abs_low_total": float(reward_abs[low_low].sum().item()),
        "r_abs_selected": float(reward_abs[selected].sum().item()),
        "r_abs_low_selected": float(reward_abs[selected & low_low].sum().item()),
        "skipped_responses": 0,
        "skipped_valid_tokens": 0,
        "skipped_quota": 0,
    }
    if not all(math.isfinite(value) for value in stats.values()):
        raise ValueError("non-finite pooled selection diagnostics")
    return selected, stats


"""Recovered selection trainer; launch/import glue adapted on 2026-10-28.

Class body is unchanged from the recovered score-family experiment.
"""
from __future__ import annotations
import json
import math
import torch
from opd_dynamics.trainer import OPDDynamicsTrainer
from .selection import select_score_tokens

class ScoreFamilySelectionTrainer(OPDDynamicsTrainer):
    def __init__(self, *args, selection_config, score_beta,
                 rollout_stream, metrics_stream, run_id, **kwargs):
        self.selection_config = selection_config
        self.score_beta = float(score_beta)
        self.rollout_stream = rollout_stream
        self.metrics_stream = metrics_stream
        self.selection_run_id = run_id
        self._window_stats = {}
        self._window_rows = 0
        super().__init__(*args, **kwargs)

    def _generate_on_policy_for_slices(self, slices, on_policy_indices):
        limit = int(self.args.max_length) - int(self.args.max_completion_length)
        for slice_index in on_policy_indices:
            batch = slices[slice_index]
            mask = batch.get("prompt_attention_mask")
            lengths = mask.sum(dim=1) if mask is not None else torch.full(
                (batch["prompts"].shape[0],), batch["prompts"].shape[1])
            if bool((lengths > limit).any()):
                raise ValueError(
                    f"Prompt exceeds {limit} tokens before generation; refusing "
                    "to generate with a different context from scoring. No sample was dropped.")
        return super()._generate_on_policy_for_slices(slices, on_policy_indices)

    def _build_reward_components(self, inputs, shifted_student_logits, shifted_teacher_logits):
        prompt_length = inputs["prompts"].shape[1]
        labels, ids = inputs["labels"], inputs["input_ids"]
        attention = inputs["attention_mask"].bool()
        if labels.shape != ids.shape or attention.shape != ids.shape:
            raise ValueError("Input IDs, labels and attention mask shapes differ")
        if bool((labels[:, :prompt_length] != -100).any()):
            raise ValueError("Prompt labels must all be -100")
        if bool((labels[~attention] != -100).any()):
            raise ValueError("Padding labels must all be -100")
        valid_labels = labels != -100
        if not torch.equal(labels[valid_labels], ids[valid_labels]):
            raise ValueError("Unmasked target labels differ from the generated token IDs")
        if shifted_student_logits.shape[:2] != labels[:, prompt_length:].shape:
            raise ValueError("Causal response scoring shape mismatch")
        if shifted_teacher_logits.shape[:2] != shifted_student_logits.shape[:2]:
            raise ValueError("Student and teacher scoring shapes differ")
        components = super()._build_reward_components(
            inputs, shifted_student_logits, shifted_teacher_logits)
        if bool((components["effective_mask"] & ~valid_labels[:, prompt_length:]).any()):
            raise ValueError("Terminal mask introduced an invalid response position")
        return components

    def _write_rollouts(self, inputs, components, selected, error=None):
        prompt_width = inputs["prompts"].shape[1]
        prompt_mask = inputs.get("prompt_attention_mask")
        for row in range(inputs["input_ids"].shape[0]):
            valid = components["effective_mask"][row]
            positions = valid.nonzero(as_tuple=False).flatten()
            n_valid = int(positions.numel())
            quota = max(1, int(round(self.selection_config["keep_ratio"] * n_valid))) if n_valid else 0
            cutoff = math.log(self.selection_config["probability_threshold"])
            low_low = valid & (components["student_logp_full"][row].detach().double() <= cutoff) & (
                components["teacher_logp_full"][row].detach().double() <= cutoff)
            prompts = inputs["prompts"][row]
            if prompt_mask is not None:
                prompts = prompts[prompt_mask[row].bool()]
            response_mask = inputs["attention_mask"][row, prompt_width:].bool()
            response = inputs["input_ids"][row, prompt_width:][response_mask]
            record = {
                "run_id": self.selection_run_id,
                "step_before_update": int(self.state.global_step),
                "window_response_index": self._window_rows,
                "prompt_ids": prompts.detach().cpu().tolist(),
                "response_ids": response.detach().cpu().tolist(),
                "valid_positions": positions.detach().cpu().tolist(),
                "valid_token_count": n_valid,
                "score_beta": self.score_beta,
                "low_low_token_count": int(low_low.sum().item()),
                "original_selection_quota": quota,
                "has_effective_tokens": bool(n_valid),
                "selected_positions": (
                    selected[row].nonzero(as_tuple=False).flatten().detach().cpu().tolist()
                    if selected is not None else None),
                "student_logp": components["student_logp_full"][row, valid].detach().float().cpu().tolist(),
                "teacher_logp": components["teacher_logp_full"][row, valid].detach().float().cpu().tolist(),
            }
            if error is not None:
                record["selection_error"] = str(error)
            self.rollout_stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            self._window_rows += 1

    def _build_sampled_reward_loss_mask(self, inputs, components):
        try:
            selected, stats = select_score_tokens(
                components["student_logp_full"], components["teacher_logp_full"],
                components["effective_mask"],
                keep_ratio=self.selection_config["keep_ratio"],
                score_beta=self.score_beta,
                probability_threshold=self.selection_config["probability_threshold"])
        except Exception as exc:
            # Preserve the concrete failed example; never silently skip or refill.
            self._write_rollouts(inputs, components, None, error=exc)
            self.rollout_stream.flush()
            raise
        if bool((selected & ~components["effective_mask"]).any()):
            raise ValueError("Selection included prompt, padding or masked terminal EOS")
        self._write_rollouts(inputs, components, selected)
        for key, value in stats.items():
            self._window_stats[key] = self._window_stats.get(key, 0) + value
        return selected

    def _precompute_sampled_reward_window(self):
        self._window_stats = {}
        self._window_rows = 0
        try:
            super()._precompute_sampled_reward_window()
            if self._window_stats.get("selected", 0) == 0:
                raise ValueError("Entire optimizer window has zero selected tokens; stopping before optimizer.step")
            record = {
                "run_id": self.selection_run_id,
                "step_before_update": int(self.state.global_step),
                "responses": self._window_rows,
                "score_beta": self.score_beta,
                **self._window_stats,
            }
            self.metrics_stream.write(json.dumps(record, allow_nan=False) + "\n")
            self.metrics_stream.flush()
            for key in ("valid", "selected", "low_low_valid", "low_low_selected"):
                self._record_train_scalar("score_family_" + key, self._window_stats[key])
            self._record_train_scalar("score_family_score_beta", self.score_beta)
        finally:
            self.rollout_stream.flush()

    def _empty_window_reward_records(self, records):
        raise ValueError("Entire optimizer window has zero selected tokens; stopping before optimizer.step")

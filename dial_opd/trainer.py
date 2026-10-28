"""DIAL selection and the signed sampled-token OPD objective.

Scores only choose positions. The detached reward keeps its sign, and all
microbatches share the selected-token denominator for the optimizer window.
"""
from __future__ import annotations
from collections import Counter
import json
import math
import torch
from trl.experimental.utils import empty_cache
from .rollout import OnPolicyTrainer
from .selection import select_score_tokens
from .stop_tokens import apply_dual_eos_stop_token_ids
from .terminal import mask_terminal_eos


class DIALTrainer(OnPolicyTrainer):
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
        apply_dual_eos_stop_token_ids(self)

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

    def _write_rollouts(self, inputs, components, selected, error=None):
        if self.rollout_stream is None:
            self._window_rows += inputs["input_ids"].shape[0]
            return
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
            if self.rollout_stream is not None:
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
            self._cache_window_rewards()
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
                self._record_train_scalar("dial_" + key, self._window_stats[key])
            self._record_train_scalar("dial_score_beta", self.score_beta)
        finally:
            if self.rollout_stream is not None:
                self.rollout_stream.flush()

    def _empty_window_reward_records(self, records):
        raise ValueError("Entire optimizer window has zero selected tokens; stopping before optimizer.step")

    def _record_train_scalar(self, key: str, value: torch.Tensor | float | int) -> None:
        if isinstance(value, torch.Tensor):
            value = float(value.detach().float().cpu().item())
        self._metrics["train"][key].append(float(value))

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
        terminal = mask_terminal_eos(
            labels[:, prompt_length:], valid_labels[:, prompt_length:], self.processing_class
        )
        safe_labels = terminal.labels.masked_fill(~terminal.mask, 0).long()
        student_logp = torch.log_softmax(shifted_student_logits.float(), -1).gather(
            -1, safe_labels.unsqueeze(-1)
        ).squeeze(-1)
        teacher_logp = torch.log_softmax(shifted_teacher_logits.float(), -1).gather(
            -1, safe_labels.unsqueeze(-1)
        ).squeeze(-1)
        return {
            "effective_mask": terminal.mask, "loss_labels": terminal.labels,
            "student_logp_full": student_logp, "teacher_logp_full": teacher_logp,
            "reward_full": teacher_logp - student_logp, "terminal_stats": terminal.stats,
        }

    def _fill_buffer(self, generation_batch, buffer_steps):
        super()._fill_buffer(generation_batch, buffer_steps)
        self._precompute_sampled_reward_window()

    def _cache_window_rewards(self):
        records = []
        terminal_totals = Counter()
        selected_count = 0
        model_was_training = self.model.training
        self.model.eval()
        self.teacher_model.eval()
        try:
            with torch.no_grad():
                for idx, inputs in enumerate(self._buffered_inputs):
                    if inputs is None or "labels" not in inputs:
                        continue
                    student = self.model(
                        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], use_cache=False
                    )
                    teacher = self.teacher_model(
                        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], use_cache=False
                    )
                    prompt_length = inputs["prompts"].shape[1]
                    components = self._build_reward_components(inputs,
                        student.logits[:, prompt_length - 1:-1, :],
                        teacher.logits[:, prompt_length - 1:-1, :])
                    mask = self._build_sampled_reward_loss_mask(inputs, components).detach()
                    reward = components["reward_full"].detach()
                    records.append((idx, inputs, mask, components["loss_labels"].detach(), reward))
                    selected_count += int(mask.sum().item())
                    terminal_totals.update(components["terminal_stats"])
                    del student, teacher, components
        finally:
            self.model.train(model_was_training)
        if not selected_count:
            self._empty_window_reward_records(records)
        for key, value in terminal_totals.items():
            self._record_train_scalar(key, value)
        self._record_train_scalar("train_window_token_count", selected_count)
        denominator = torch.tensor(float(selected_count), dtype=torch.float32)
        for idx, inputs, mask, labels, reward in records:
            self._buffered_inputs[idx] = dict(inputs,
                sampled_window_reward=reward, sampled_window_mask=mask,
                sampled_window_loss_labels=labels,
                sampled_window_token_count=denominator.to(reward.device))
        empty_cache()

    def _compute_loss_from_window_cache(self, model, inputs, return_outputs=False):
        for key in (
            "sampled_window_reward",
            "sampled_window_mask",
            "sampled_window_loss_labels",
            "sampled_window_token_count",
        ):
            if key not in inputs:
                raise ValueError(f"Missing {key}; sampled reward window precompute did not run.")

        outputs_student = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], use_cache=False)
        prompt_length = inputs["prompts"].shape[1]
        shifted_student_logits = outputs_student.logits[:, prompt_length - 1 : -1, :]
        effective_mask = inputs["sampled_window_mask"].to(shifted_student_logits.device).bool()
        loss_labels = inputs["sampled_window_loss_labels"].to(shifted_student_logits.device)
        safe_loss_labels = loss_labels.masked_fill(~effective_mask, 0).long()
        student_log_probs = torch.log_softmax(shifted_student_logits.float(), dim=-1)
        loss_logp_full = student_log_probs.gather(-1, safe_loss_labels.unsqueeze(-1)).squeeze(-1)
        reward_full = inputs["sampled_window_reward"].to(shifted_student_logits.device).detach()

        if not effective_mask.any():
            loss = outputs_student.logits.sum() * 0.0
        else:
            denominator = inputs["sampled_window_token_count"].to(shifted_student_logits.device).float().clamp_min(1.0)
            loss = -(reward_full[effective_mask] * loss_logp_full[effective_mask]).sum() / denominator
            ga_steps = float(getattr(self, "current_gradient_accumulation_steps", self.args.gradient_accumulation_steps))
            loss = loss * ga_steps

        empty_cache()
        return (loss, outputs_student) if return_outputs else loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        return self._compute_loss_from_window_cache(model, inputs, return_outputs)

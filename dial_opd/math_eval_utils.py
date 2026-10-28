from __future__ import annotations

import inspect
import multiprocessing
import queue as _queue
import time
import re
from pathlib import Path
from typing import Any

from datasets import load_dataset


def extract_last_boxed(text: str) -> str:
    if not text:
        return ""
    pos = 0
    last = ""
    while True:
        idx = text.find(r"\boxed", pos)
        if idx < 0:
            break
        brace_start = text.find("{", idx)
        if brace_start < 0:
            pos = idx + 6
            continue
        depth = 0
        for i in range(brace_start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    last = text[brace_start + 1 : i].strip()
                    break
        pos = idx + 6
    return last


def extract_after_hashes(text: str) -> str:
    if not text:
        return ""
    if "####" in text:
        return text.split("####")[-1].strip()
    return ""


def extract_candidate_answer(text: str) -> str:
    return extract_last_boxed(text) or extract_after_hashes(text) or (text or "").strip()


def _normalize_interval_text(text: str) -> str:
    if not text:
        return ""
    normalized = str(text).strip().replace("$", "")
    normalized = normalized.replace(r"\left", "").replace(r"\right", "")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _split_top_level_comma(text: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    for idx, ch in enumerate(text):
        if ch in "{[(":
            depth += 1
        elif ch in "})]":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            parts.append(text[start:idx].strip())
            start = idx + 1
    parts.append(text[start:].strip())
    return parts


def parse_interval_bounds(text: str) -> tuple[str, str, str, str] | None:
    normalized = _normalize_interval_text(text)
    if len(normalized) < 3:
        return None
    if normalized[0] not in "[(" or normalized[-1] not in ")]":
        return None
    inner = normalized[1:-1].strip()
    if not inner:
        return None
    parts = _split_top_level_comma(inner)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return None
    return normalized[0], parts[0], parts[1], normalized[-1]


def supports_enable_thinking(tokenizer) -> bool:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if apply_chat_template is None:
        return False
    try:
        signature = inspect.signature(apply_chat_template)
    except (TypeError, ValueError):
        return False
    has_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
    return "enable_thinking" in signature.parameters or has_var_kwargs


def build_prompt(tokenizer, question: str, enable_thinking: bool | None = False) -> str:
    messages = [{"role": "user", "content": question}]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if supports_enable_thinking(tokenizer) and enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    prompt = tokenizer.apply_chat_template(messages, **kwargs)
    bos_token = getattr(tokenizer, "bos_token", None)
    if isinstance(bos_token, str) and bos_token and prompt.startswith(bos_token):
        prompt = prompt[len(bos_token) :]
    return prompt


def prepare_eval_dataset(name: str, path: str, limit: str) -> list[dict]:
    raw_dataset = load_dataset("json", data_files=path, split="train")
    if limit.lower() != "all":
        max_n = min(len(raw_dataset), int(limit))
        raw_dataset = raw_dataset.shuffle(seed=42).select(range(max_n))

    processed = []
    lowered = name.lower()
    for item in raw_dataset:
        if lowered == "gsm8k":
            question = (item.get("question") or "").strip()
            answer = (item.get("answer") or "").strip()
            gold = extract_after_hashes(answer) or answer
        else:
            question = ((item.get("problem") or item.get("question") or item.get("prompt") or "")).strip()
            gold_raw = item.get("answer") or item.get("gold") or item.get("solution") or ""
            gold = extract_after_hashes(gold_raw) or str(gold_raw).strip()
        if question and gold:
            processed.append({"question": question, "gold": gold})
    return processed


class _InProcessMathVerifier:
    """直接在当前进程调 math_verify；只应在下面的隔离 worker 进程里使用。"""

    def __init__(self):
        from math_verify import parse, verify

        self._parse = parse
        self._verify = verify

    def _verify_parse_equivalence(self, gold: str, pred: str) -> bool:
        gold_parsed = self._parse(gold)
        pred_parsed = self._parse(pred)
        if len(gold_parsed) == 0 or len(pred_parsed) == 0:
            return False
        return bool(self._verify(gold_parsed, pred_parsed))

    def _intervals_match(self, gold: str, pred: str) -> bool:
        gold_bounds = parse_interval_bounds(gold)
        pred_bounds = parse_interval_bounds(pred)
        if gold_bounds is None or pred_bounds is None:
            return False
        gold_left, gold_start, gold_end, gold_right = gold_bounds
        pred_left, pred_start, pred_end, pred_right = pred_bounds
        if gold_left != pred_left or gold_right != pred_right:
            return False
        return self._verify_parse_equivalence(gold_start, pred_start) and self._verify_parse_equivalence(gold_end, pred_end)

    def is_correct(self, gold: str, pred_text: str) -> bool:
        candidate = extract_candidate_answer(pred_text)
        if self._verify_parse_equivalence(gold, candidate):
            return True
        if self._intervals_match(gold, candidate):
            return True
        return False


def _math_verifier_worker_main(task_queue, result_queue):
    result_queue.put("__ready__")
    verifier = _InProcessMathVerifier()
    while True:
        task = task_queue.get()
        if task is None:
            return
        gold, pred_text = task
        try:
            ok = bool(verifier.is_correct(gold, pred_text))
        except BaseException:
            ok = False
        result_queue.put(ok)


class _PipeChannel:
    """Queue-like interface on a process pipe; no filesystem-backed semaphores."""
    def __init__(self, connection):
        self.connection = connection

    def put(self, value):
        self.connection.send(value)

    def get(self, timeout=None):
        if timeout is not None and not self.connection.poll(timeout):
            raise _queue.Empty
        return self.connection.recv()

    def close(self):
        self.connection.close()


class MathVerifier:
    """进程隔离的 math_verify 打分（公开接口与原类一致：is_correct）。

    math_verify 的 parsing_timeout 基于 signal.alarm，signal handler 只能在
    Python 字节码之间执行，挡不住 C 层正则/sympy 的灾难性回溯——表现为主进程
    100% CPU 用户态死循环、GPU 0%、永久挂死（2026-08-10 两个 layer_contrast
    训练先后在 rollout 打分处挂死）。这里把 parse/verify 放进可被 SIGKILL 的
    spawn 子进程：超时→杀 worker→判错→下次调用重建，与原来"异常→False"的
    语义一致；此外 signal.alarm 泄漏的 TimeoutException 也被限制在 worker 内。
    """

    _SPAWN_TIMEOUT_S = 300.0  # worker 首次 import datasets/math_verify 可能较慢

    def __init__(self, timeout_s: float = 20.0):
        self._timeout_s = float(timeout_s)
        self._ctx = multiprocessing.get_context("spawn")
        self._worker = None
        self._task_queue = None
        self._result_queue = None
        self._timeout_count = 0

    def _ensure_worker(self) -> bool:
        if self._worker is not None and self._worker.is_alive():
            return True
        parent, child = self._ctx.Pipe(duplex=True)
        self._task_queue = self._result_queue = _PipeChannel(parent)
        child_channel = _PipeChannel(child)
        self._worker = self._ctx.Process(
            target=_math_verifier_worker_main,
            args=(child_channel, child_channel),
            daemon=True,
        )
        self._worker.start()
        child.close()
        deadline = time.monotonic() + self._SPAWN_TIMEOUT_S
        while time.monotonic() < deadline:
            try:
                return self._result_queue.get(timeout=1.0) == "__ready__"
            except _queue.Empty:
                if not self._worker.is_alive():  # worker 启动即崩（如 import 失败）
                    break
        self._kill_worker()
        return False

    def _kill_worker(self) -> None:
        if self._worker is not None:
            try:
                self._worker.kill()
                self._worker.join(timeout=5)
            except Exception:
                pass
        if self._task_queue is not None:
            self._task_queue.close()
        self._worker = None
        self._task_queue = None
        self._result_queue = None

    def is_correct(self, gold: str, pred_text: str) -> bool:
        try:
            if not self._ensure_worker():
                return False
            self._task_queue.put((gold, pred_text))
            return bool(self._result_queue.get(timeout=self._timeout_s))
        except _queue.Empty:
            self._timeout_count += 1
            print(
                f"[math_verifier] scoring timed out after {self._timeout_s}s; "
                f"killed worker, treating as incorrect (n={self._timeout_count})",
                flush=True,
            )
            self._kill_worker()
            return False
        except Exception:
            self._kill_worker()
            return False

    def close(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            try:
                self._task_queue.put(None)
                self._worker.join(timeout=2)
            except Exception:
                pass
        self._kill_worker()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def score_candidates(verifier: MathVerifier, gold: str, candidates: list[dict], eval_k: int) -> dict:
    k_used = max(1, min(eval_k, len(candidates))) if candidates else 0
    selected_text = candidates[0]["text"] if candidates else ""
    selected_len = candidates[0]["token_len"] if candidates else 0
    correct_flags = []
    any_correct = False

    for candidate in candidates:
        try:
            is_correct = verifier.is_correct(gold, candidate["text"])
        except Exception:
            is_correct = False
        correct_flags.append(bool(is_correct))
        if is_correct and not any_correct:
            any_correct = True
            selected_text = candidate["text"]
            selected_len = candidate["token_len"]

    correct_in_k = sum(correct_flags[:k_used]) if k_used > 0 else 0
    avg_at_k = (correct_in_k / k_used) if k_used > 0 else 0.0
    pass_at_k = 1.0 if any(correct_flags[:k_used]) else 0.0
    return {
        "selected_text": selected_text,
        "selected_len": selected_len,
        "is_correct": any_correct,
        "avg_at_k": avg_at_k,
        "pass_at_k": pass_at_k,
        "k_used": k_used,
    }


def compute_response_entropy_totals(model, tokenizer, prompt_texts: list[str], grouped_candidates: list[list[dict]], k: int) -> tuple[float, int]:
    import torch

    try:
        device = next(model.parameters()).device
    except StopIteration:
        return 0.0, 0

    sequences: list[list[int]] = []
    spans: list[tuple[int, int]] = []
    for prompt_text, candidates in zip(prompt_texts, grouped_candidates, strict=True):
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
        for candidate in candidates[:k]:
            completion_ids = list(candidate.get("token_ids", []))
            if not completion_ids:
                continue
            full_ids = list(prompt_ids) + completion_ids
            if len(full_ids) < 2:
                continue
            start = max(len(prompt_ids) - 1, 0)
            end = start + len(completion_ids)
            sequences.append(full_ids)
            spans.append((start, end))

    if not sequences:
        return 0.0, 0

    was_training = model.training
    model.eval()
    total_entropy = 0.0
    total_tokens = 0
    try:
        with torch.no_grad():
            for full_ids, (start, end) in zip(sequences, spans, strict=True):
                input_ids = torch.tensor([full_ids], device=device, dtype=torch.long)
                attention_mask = torch.ones_like(input_ids)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                shift_logits = outputs.logits[:, :-1, :].float()
                log_probs = torch.log_softmax(shift_logits, dim=-1)
                token_entropy = -(log_probs.exp() * log_probs).sum(dim=-1)[0]
                response_entropy = token_entropy[start:end]
                total_entropy += float(response_entropy.sum().item())
                total_tokens += int(response_entropy.numel())
    finally:
        if was_training:
            model.train()

    return total_entropy, total_tokens


def compute_mean_response_entropy(model, tokenizer, prompt_texts: list[str], grouped_candidates: list[list[dict]], eval_k: int) -> float:
    total_entropy, total_tokens = compute_response_entropy_totals(
        model,
        tokenizer,
        prompt_texts,
        grouped_candidates,
        eval_k,
    )
    return total_entropy / max(1, total_tokens)


def group_flat_rollout_samples(
    prompt_texts: list[str],
    completion_token_ids: list[list[int]],
    tokenizer,
    group_size: int,
    gold_answers: list[str | None] | None = None,
) -> tuple[list[str], list[str | None], list[list[dict[str, Any]]]]:
    if group_size < 1:
        raise ValueError(f"group_size must be >= 1, got {group_size}.")

    grouped_prompts: list[str] = []
    grouped_golds: list[str | None] = []
    grouped_candidates: list[list[dict[str, Any]]] = []

    total = len(prompt_texts)
    if gold_answers is None:
        gold_answers = [None] * total
    if not (len(prompt_texts) == len(completion_token_ids) == len(gold_answers)):
        raise ValueError("prompt_texts, completion_token_ids, and gold_answers must have matching lengths.")

    idx = 0
    while idx < total:
        prompt_text = prompt_texts[idx]
        gold_answer = gold_answers[idx]
        candidates: list[dict[str, Any]] = []
        cursor = idx
        while cursor < total and len(candidates) < group_size:
            if cursor > idx and (prompt_texts[cursor] != prompt_text or gold_answers[cursor] != gold_answer):
                break
            token_ids = list(completion_token_ids[cursor])
            candidates.append(
                {
                    "text": tokenizer.decode(
                        token_ids,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    ),
                    "token_ids": token_ids,
                    "token_len": len(token_ids),
                }
            )
            cursor += 1

        if not candidates:
            cursor = idx + 1

        grouped_prompts.append(prompt_text)
        grouped_golds.append(gold_answer)
        grouped_candidates.append(candidates)
        idx = cursor

    return grouped_prompts, grouped_golds, grouped_candidates


def empty_rollout_metric_totals() -> dict[str, float]:
    return {
        "group_count": 0.0,
        "scored_group_count": 0.0,
        "candidate_count": 0.0,
        "acc_sum": 0.0,
        "pass_sum": 0.0,
        "len_sum": 0.0,
        "clip_count": 0.0,
        "entropy_sum": 0.0,
        "entropy_token_count": 0.0,
    }


def compute_rollout_metric_totals(
    verifier: MathVerifier,
    gold_answers: list[str | None],
    grouped_candidates: list[list[dict[str, Any]]],
    metric_k: int,
    clip_max_tokens: int,
    *,
    model=None,
    tokenizer=None,
    prompt_texts: list[str] | None = None,
    include_accuracy: bool = True,
) -> dict[str, float]:
    totals = empty_rollout_metric_totals()
    if metric_k < 1:
        raise ValueError(f"metric_k must be >= 1, got {metric_k}.")

    totals["group_count"] = float(len(grouped_candidates))
    for gold_answer, candidates in zip(gold_answers, grouped_candidates, strict=True):
        k_used = max(1, min(metric_k, len(candidates))) if candidates else 0
        considered = candidates[:k_used]
        totals["candidate_count"] += float(k_used)
        totals["len_sum"] += float(sum(candidate["token_len"] for candidate in considered))
        totals["clip_count"] += float(sum(candidate["token_len"] >= clip_max_tokens for candidate in considered))

        if include_accuracy and gold_answer:
            scored = score_candidates(verifier, gold_answer, candidates, metric_k)
            totals["acc_sum"] += float(scored["avg_at_k"])
            totals["pass_sum"] += float(scored["pass_at_k"])
            totals["scored_group_count"] += 1.0

    if model is not None and tokenizer is not None and prompt_texts is not None and grouped_candidates:
        entropy_sum, entropy_token_count = compute_response_entropy_totals(
            model,
            tokenizer,
            prompt_texts,
            grouped_candidates,
            metric_k,
        )
        totals["entropy_sum"] = float(entropy_sum)
        totals["entropy_token_count"] = float(entropy_token_count)

    return totals


def finalize_rollout_metric_totals(totals: dict[str, float], include_accuracy: bool = True) -> dict[str, float]:
    metrics: dict[str, float] = {}
    scored_group_count = totals.get("scored_group_count", 0.0)
    candidate_count = totals.get("candidate_count", 0.0)
    entropy_token_count = totals.get("entropy_token_count", 0.0)

    if include_accuracy and scored_group_count > 0:
        metrics["rollout_acc"] = totals["acc_sum"] / scored_group_count
        metrics["rollout_pass_at_k"] = totals["pass_sum"] / scored_group_count
    if candidate_count > 0:
        metrics["rollout_avg_len"] = totals["len_sum"] / candidate_count
        metrics["rollout_clip_ratio"] = totals["clip_count"] / candidate_count
    if entropy_token_count > 0:
        metrics["rollout_entropy"] = totals["entropy_sum"] / entropy_token_count

    return metrics


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

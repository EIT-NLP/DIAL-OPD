"""Recovered numerical fixtures, imported against the actual restored trainer."""
import io
import json
import math
from types import SimpleNamespace
import torch
from dial_opd.trainer import DIALTrainer
from dial_opd.math_eval_utils import MathVerifier
BETAS=(0., .25, .5, .75, 1., 1.2, 1.5, 2.)

class TinyWindowModel(torch.nn.Module):
    """Two input-dependent microbatches with differentiable output logits."""

    def __init__(self, batches, values):
        super().__init__()
        self.batches = batches
        self.values = torch.nn.ParameterList([torch.nn.Parameter(value.clone()) for value in values])

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        for (index, batch) in enumerate(self.batches):
            if torch.equal(input_ids, batch['input_ids']):
                return SimpleNamespace(logits=self.values[index])
        raise AssertionError('Unexpected input batch')

def fixture():
    ids = torch.tensor([[1, 2, 3, 1, 2, 7, 7, 7], [7, 7, 4, 3, 4, 5, 6, 0], [5, 6, 2, 4, 5, 7, 7, 7], [1, 2, 3, 8, 7, 7, 7, 7]])
    attention = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0]])
    labels = ids.clone()
    labels[:, :3] = -100
    labels[attention == 0] = -100
    batches = [dict(input_ids=ids[part], attention_mask=attention[part], labels=labels[part], prompts=ids[part, :3], prompt_attention_mask=attention[part, :3]) for part in (slice(0, 2), slice(2, 4))]
    student = torch.linspace(-0.4, 0.6, 4 * 8 * 9).reshape(4, 8, 9)
    teacher = student + torch.linspace(-0.6, 0.8, 9)
    (student[0, 2, 1], teacher[0, 2, 1]) = (-6.0, -12.0)
    (student[1, 2, 3], teacher[1, 2, 3]) = (-2.0, 6.0)
    (student[2, 2, 4], teacher[2, 2, 4]) = (-7.0, -10.0)
    (student[2, 3, 5], teacher[2, 3, 5]) = (-8.0, -11.0)
    return (batches, [student[:2], student[2:]], [teacher[:2], teacher[2:]])

def audit_actual_window():
    (batches, students, teachers) = fixture()
    expected_effective = [[[True, True, False, False, False], [True] * 5], [[True, True, False, False, False], [False] * 5]]
    previous_rewards = None
    for beta in BETAS:
        obj = DIALTrainer.__new__(DIALTrainer)
        obj.args = SimpleNamespace(gradient_accumulation_steps=32, beta=1)
        obj.state = SimpleNamespace(global_step=0)
        obj.processing_class = SimpleNamespace(convert_tokens_to_ids=lambda token: {'<|endoftext|>': 7, '<|im_end|>': 8}.get(token))
        obj._record_train_scalar = lambda *a, **kw: None
        obj.selection_config = {'keep_ratio': 0.2, 'probability_threshold': 0.01}
        (obj.selection_run_id, obj.score_beta) = ('cpu_independent_preflight', beta)
        (obj._window_stats, obj._window_rows) = ({}, 0)
        (obj.rollout_stream, obj.metrics_stream) = (io.StringIO(), io.StringIO())
        obj.model = TinyWindowModel(batches, students)
        obj.teacher_model = TinyWindowModel(batches, teachers)
        obj._buffered_inputs = [dict(batch) for batch in batches]
        obj._precompute_sampled_reward_window()
        cached_batches = obj._buffered_inputs
        assert [int(x['sampled_window_mask'].sum()) for x in cached_batches] == [2, 1]
        assert all((x['sampled_window_token_count'].item() == 3 for x in cached_batches))
        rewards = [x['sampled_window_reward'] for x in cached_batches]
        if previous_rewards is not None:
            for (current, previous) in zip(rewards, previous_rewards):
                torch.testing.assert_close(current, previous, rtol=0, atol=0)
        previous_rewards = [x.clone() for x in rewards]
        assert all((not x.requires_grad for x in rewards))
        chosen_rewards = torch.cat([batch['sampled_window_reward'][batch['sampled_window_mask']] for batch in cached_batches])
        assert bool((chosen_rewards > 0).any()) and bool((chosen_rewards < 0).any())
        assert obj.args.beta == 1
        (losses, references, expected_gradients) = ([], [], [])
        for (index, batch) in enumerate(cached_batches):
            current = obj.model.values[index]
            target = batch['input_ids'][:, 3:]
            student_lp = torch.log_softmax(current[:, 2:-1].float(), -1)
            teacher_lp = torch.log_softmax(obj.teacher_model.values[index][:, 2:-1].float(), -1)
            signed = (teacher_lp.gather(-1, target.unsqueeze(-1)).squeeze(-1) - student_lp.gather(-1, target.unsqueeze(-1)).squeeze(-1)).detach()
            selected = batch['sampled_window_mask']
            effective = torch.tensor(expected_effective[index])
            assert not bool((selected & ~effective).any())
            torch.testing.assert_close(batch['sampled_window_reward'][effective], signed[effective], rtol=0, atol=0)
            losses.append(obj._compute_loss_from_window_cache(obj.model, batch) / 32.0)
            chosen_lp = student_lp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
            references.append(-(signed[selected] * chosen_lp[selected]).sum() / 3.0)
            grad = torch.zeros_like(current)
            local = -signed.unsqueeze(-1) * (torch.nn.functional.one_hot(target, 9) - student_lp.detach().exp()) / 3.0
            grad[:, 2:-1] = local * selected.unsqueeze(-1)
            expected_gradients.append(grad)
        (loss, reference) = (sum(losses), sum(references))
        torch.testing.assert_close(loss, reference, rtol=1e-06, atol=1e-07)
        loss.backward()
        for (value, expected) in zip(obj.model.values, expected_gradients):
            torch.testing.assert_close(value.grad, expected, rtol=1e-05, atol=1e-07)
            assert torch.equal(value.grad.abs().sum(-1) > 0, expected.abs().sum(-1) > 0)
        assert all((value.grad is None for value in obj.teacher_model.values))
        records = [json.loads(line) for line in obj.rollout_stream.getvalue().splitlines()]
        assert len(records) == 4
        assert records[2]['low_low_token_count'] == 2 and len(records[2]['selected_positions']) == 1
        assert records[3]['valid_positions'] == records[3]['selected_positions'] == []
        assert records[3]['response_ids'] == [8]
        assert all(('loss_skipped' not in row for row in records))
        metrics = json.loads(obj.metrics_stream.getvalue())
        assert metrics['responses'] == 4 and metrics['selected'] == 3
    return {'status': 'passed', 'betas': list(BETAS), 'checks': ['real_precompute_components_selector_cached_loss_chain', 'unequal_microbatch_selected_counts_share_window_denominator', 'signed_reward_invariant_across_score_betas', 'positive_and_negative_reward_closed_form_gradients', 'teacher_and_reward_stop_gradient', 'prompt_padding_unselected_and_both_terminal_EOS_targets_have_zero_direct_loss', 'nonempty_low_low_only_response_keeps_original_quota', 'EOS_only_response_recorded_with_zero_targets', 'OPD_beta_remains_1']}

if __name__ == '__main__':
    report=audit_actual_window()
    verifier=MathVerifier()
    try:
        assert verifier.is_correct('4', r'$\boxed{4}$')
        assert not verifier.is_correct('4', r'$\boxed{5}$')
    finally:
        verifier.close()
    report['math_verifier_worker']=True
    print(json.dumps(report,indent=2))

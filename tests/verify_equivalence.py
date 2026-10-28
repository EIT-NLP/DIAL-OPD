"""Compare refactored training against an untouched release checkout on CPU.

Usage: PYTHONPATH=. python tests/verify_equivalence.py --original /path/to/release
The reference is supplied externally; historical implementations are not shipped.
"""
import argparse
from collections import defaultdict
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from dial_opd.trainer import DIALTrainer
from verify_integration import fixture, TinyWindowModel, BETAS


def reference_class(root):
    sys.path.append(str(root))
    spec = importlib.util.spec_from_file_location('reference_dial', root/'dial_opd/__init__.py',
                                                 submodule_search_locations=[str(root/'dial_opd')])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    spec = importlib.util.spec_from_file_location('reference_dial.trainer', root/'dial_opd/trainer.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.ScoreFamilySelectionTrainer


def make_trainer(cls, beta, ratio, seed, log_rollouts=True):
    batches, students, teachers = fixture()
    generator = torch.Generator().manual_seed(seed)
    students = [x + torch.randn(x.shape, generator=generator) * .2 for x in students]
    teachers = [x + torch.randn(x.shape, generator=generator) * .2 for x in teachers]
    obj = cls.__new__(cls)
    obj.args = SimpleNamespace(gradient_accumulation_steps=2, beta=1)
    obj.current_gradient_accumulation_steps = 2
    obj.state = SimpleNamespace(global_step=0)
    obj.processing_class = SimpleNamespace(convert_tokens_to_ids=lambda t: {'<|endoftext|>':7,'<|im_end|>':8}.get(t))
    obj.terminal_stop_target = 'mask'
    obj.reward_mode = 'sampled_log_ratio'
    obj.log_reward_clip_min, obj.log_reward_clip_max = -5., 5.
    obj.log_reward_tanh_temperature, obj.power_reward_alpha = 1., .1
    obj.token_loss_normalization, obj.reward_normalization = 'window_token_mean', 'none'
    obj.reward_position_tracker = SimpleNamespace(enabled=False)
    obj._maybe_dump_prefill_debug = lambda *a, **k: None
    obj._metrics = {'train': defaultdict(list)}
    obj.selection_config = {'keep_ratio':ratio, 'probability_threshold':.01}
    obj.selection_run_id, obj.score_beta = 'equivalence', beta
    obj._window_stats, obj._window_rows = {}, 0
    obj.rollout_stream = io.StringIO() if log_rollouts else None
    obj.metrics_stream = io.StringIO()
    obj.model = TinyWindowModel(batches, students)
    obj.teacher_model = TinyWindowModel(batches, teachers)
    obj._buffered_inputs = [dict(b) for b in batches]
    return obj


def run(obj):
    obj._precompute_sampled_reward_window()
    loss = sum(obj._compute_loss_from_window_cache(obj.model,b) / 2 for b in obj._buffered_inputs)
    loss.backward()
    return loss.detach()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', required=True, type=Path)
    args = parser.parse_args()
    original = reference_class(args.original.resolve())
    torch.set_num_threads(1)
    cases = 0
    for beta in BETAS:
        for ratio in (.1,.2,.4,1.):
            for seed in (0,17,42):
                before = make_trainer(original,beta,ratio,seed)
                after = make_trainer(DIALTrainer,beta,ratio,seed)
                torch.testing.assert_close(run(before),run(after),rtol=0,atol=0)
                for old,new in zip(before._buffered_inputs,after._buffered_inputs):
                    for name in ('sampled_window_mask','sampled_window_reward','sampled_window_loss_labels','sampled_window_token_count'):
                        torch.testing.assert_close(old[name],new[name],rtol=0,atol=0)
                for old,new in zip(before.model.parameters(),after.model.parameters()):
                    torch.testing.assert_close(old.grad,new.grad,rtol=0,atol=0)
                assert before.rollout_stream.getvalue() == after.rollout_stream.getvalue()
                assert before.metrics_stream.getvalue() == after.metrics_stream.getvalue()
                assert all(p.grad is None for p in after.teacher_model.parameters())
                # Turning off token dumps must not change training.
                quiet = make_trainer(DIALTrainer,beta,ratio,seed,False)
                torch.testing.assert_close(run(quiet),run_frozen_loss(after),rtol=0,atol=0)
                assert quiet.metrics_stream.getvalue() == after.metrics_stream.getvalue()
                cases += 1
    # Both implementations must reject windows with no trainable target tokens.
    for cls in (original,DIALTrainer):
        empty = make_trainer(cls,1.,.2,42)
        for batch in empty._buffered_inputs:
            batch['labels'].fill_(-100)
        try:
            empty._precompute_sampled_reward_window()
        except ValueError as exc:
            assert 'zero selected tokens' in str(exc)
        else:
            raise AssertionError('Empty window was silently accepted')
    print(json.dumps({'status':'passed','exact_equivalence_cases':cases,
        'betas':list(BETAS),'ratios':[.1,.2,.4,1.],
        'checks':['selected_masks','signed_rewards','window_denominators','losses',
                  'student_gradients','teacher_stop_gradient','rollout_diagnostics',
                  'optional_logging','empty_window_rejection']},indent=2))


def run_frozen_loss(obj):
    with torch.no_grad():
        return sum(obj._compute_loss_from_window_cache(obj.model,b)/2 for b in obj._buffered_inputs)


if __name__ == '__main__':
    main()

"""Audit actual training artifacts against independent selection arithmetic."""
import argparse
from decimal import Decimal, localcontext
import gzip
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open


def audit(root):
    cfg = json.loads((root/'launch_config.json').read_text())
    state = json.loads((root/'trainer_state.json').read_text())
    result = json.loads((root/'result.json').read_text())
    steps = cfg['training']['max_steps']
    assert state['global_step'] == steps
    assert all(math.isfinite(v) for v in result.values())
    losses = [x for x in state['log_history'] if 'loss' in x]
    assert len(losses) == steps
    assert all(math.isfinite(x['loss']) and math.isfinite(x['grad_norm']) and x['grad_norm'] > 0 for x in losses)
    with gzip.open(root/'rollouts.jsonl.gz', 'rt') as f:
        rows = [json.loads(line) for line in f]
    metrics = [json.loads(line) for line in (root/'selection_metrics.jsonl').read_text().splitlines()]
    expected_responses = steps * cfg['training']['generation_batch_size']
    assert len(rows) == expected_responses
    assert len(metrics) == steps
    signed = set()
    for row in rows:
        positions = row['valid_positions']
        k = max(1, round(len(positions)*cfg['keep_ratio'])) if positions else 0
        assert k == row['original_selection_quota'] == len(row['selected_positions'])
        assert len(positions) == row['valid_token_count'] == len(row['student_logp']) == len(row['teacher_logp'])
        assert all(0 <= p < len(row['response_ids']) for p in positions)
        # Independent high-precision definition: L=(q-p)/(log(q)-log(p)).
        # Does not invoke the selector or its numerically stabilized helper.
        scores = []
        with localcontext() as ctx:
            ctx.prec = 70
            beta = Decimal(str(cfg['score_beta']))
            for p, lp, lq in zip(positions, row['student_logp'], row['teacher_logp']):
                a, b = Decimal(str(lp)), Decimal(str(lq))
                assert a.is_finite() and b.is_finite() and a <= 0 and b <= 0
                d = abs(b-a)
                signed.add(1 if b>a else -1 if b<a else 0)
                score = (((b.exp()-a.exp())/(b-a))**beta)*d if d else Decimal(0)
                scores.append((p, score))
        expected = {p for p, _ in sorted(scores, key=lambda pair: -pair[1])[:k]}
        assert expected == set(row['selected_positions']), f'Selection mismatch in step {row["step_before_update"]}'
    assert {-1, 1}.issubset(signed), 'Smoke test must exercise both reward signs'
    for m in metrics:
        group = [r for r in rows if r['step_before_update'] == m['step_before_update']]
        assert len(group) == m['responses']
        assert sum(len(r['selected_positions']) for r in group) == m['selected']
        assert sum(r['valid_token_count'] for r in group) == m['valid']
        step_loss = next(x for x in losses if x['step'] == m['step_before_update']+1)
        assert step_loss['train_window_token_count'] == m['selected']
    name = 'model.layers.0.self_attn.q_proj.weight'
    initial = Path(cfg['student'])/'model.safetensors'
    with safe_open(initial, framework='pt', device='cpu') as source:
        before = source.get_tensor(name).float()
    with safe_open(root/'final/model.safetensors', framework='pt', device='cpu') as saved:
        after = saved.get_tensor(name).float()
    delta = (after-before).abs()
    assert torch.isfinite(after).all() and delta.max() > 0
    eval_rows = [x for x in state['log_history'] if 'eval_math_rollout_acc' in x]
    if cfg['training'].get('eval_test_paths'):
        assert {0,2}.issubset({x['step'] for x in eval_rows})
        assert all(x['eval_math_count']==2 and 0<=x['eval_math_rollout_acc']<=1 for x in eval_rows)
    return dict(output=str(root), optimizer_steps=steps, responses=len(rows),
                independent_selection_match=True, signed_rewards=True,
                shared_window_denominator=True, finite_losses_and_gradients=True,
                saved_weight_max_change=delta.max().item(),
                evaluation_steps=[x['step'] for x in eval_rows],
                train_runtime_seconds=result['train_runtime'])


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('outputs', nargs='+', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    args=parser.parse_args()
    report={'status':'passed', 'runs':[audit(p) for p in args.outputs]}
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))

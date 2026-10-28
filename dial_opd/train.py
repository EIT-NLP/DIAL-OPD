"""Train DIAL-OPD with student rollouts and per-response token selection."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import gzip
import json
import math
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer, set_seed

from .compat import enable_vllm_gpu_uuid_support, configure_confined_single_process_vllm

enable_vllm_gpu_uuid_support()
configure_confined_single_process_vllm()

from .config import DIALConfig
from .trainer import DIALTrainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='JSON configuration file')
    cli = parser.parse_args()
    cfg = json.loads(Path(cli.config).read_text())
    required = {'student', 'teacher', 'train_data', 'output_dir', 'score_beta', 'keep_ratio', 'training'}
    if set(cfg) - (required | {'max_samples', 'save_final', 'log_rollouts', 'verify_runtime'}):
        raise ValueError('Unknown top-level configuration fields')
    if required - set(cfg):
        raise ValueError(f'Missing fields: {sorted(required-set(cfg))}')
    beta, ratio = cfg['score_beta'], cfg['keep_ratio']
    if isinstance(beta, bool) or not math.isfinite(beta) or beta < 0:
        raise ValueError('score_beta must be finite and nonnegative')
    if isinstance(ratio, bool) or not math.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValueError('keep_ratio must lie in (0,1]')
    output = Path(cfg['output_dir']).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(f'Refusing to overwrite nonempty output directory: {output}')
    settings = dict(cfg['training'])
    invariants = dict(beta=1., lmbda=1., num_generations=1,
                      use_uld_loss=False, seq_kd=False, use_liger_kernel=False)
    for key, value in invariants.items():
        if key in settings and settings[key] != value:
            raise ValueError(f'DIAL requires {key}={value}; score_beta is the selection exponent')
        settings[key] = value
    settings.update(output_dir=str(output), teacher_model_name_or_path=cfg['teacher'])
    settings.setdefault('model_init_kwargs', {'torch_dtype': 'float32', 'attn_implementation': 'sdpa'})
    settings.setdefault('teacher_model_init_kwargs', {'torch_dtype': 'bfloat16', 'attn_implementation': 'sdpa'})
    settings.setdefault('report_to', 'none')
    settings.setdefault('enable_thinking', False)
    args = DIALConfig(**settings)
    if args.world_size != 1:
        raise ValueError('This training entry point requires a single GPU')
    if args.use_vllm and args.vllm_tensor_parallel_size != 1:
        raise ValueError('This release entry point requires vLLM tensor parallel size 1')
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(cfg['student'])
    teacher_tokenizer = AutoTokenizer.from_pretrained(cfg['teacher'])
    if tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise ValueError('Sampled-token scoring requires identical student/teacher token IDs')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.chat_template:
        tokenizer.chat_template = teacher_tokenizer.chat_template
    if not tokenizer.chat_template:
        raise ValueError('A chat template is required for non-thinking OPD prompts')
    dataset = load_dataset('json', data_files=cfg['train_data'], split='train')
    if 'messages' not in dataset.column_names:
        raise ValueError('Expected conversation JSONL with a messages field; see data preparation documentation')
    if cfg.get('max_samples'):
        dataset = dataset.select(range(min(int(cfg['max_samples']), len(dataset))))
    # GOLD reads per-example template kwargs. Keep thinking mode consistent
    # between preparation, rollout, and scoring.
    dataset = dataset.map(lambda _: {'chat_template_kwargs': {'enable_thinking': args.enable_thinking}})
    (output/'launch_config.json').write_text(json.dumps(cfg, indent=2)+'\n')
    with ExitStack() as stack:
        if cfg.get('log_rollouts', False):
            raw = stack.enter_context(gzip.open(output/'rollouts.jsonl.gz', 'wt'))
        else:
            raw = None
        metrics = stack.enter_context((output/'selection_metrics.jsonl').open('w'))
        trainer = DIALTrainer(
            model=cfg['student'], teacher_model=cfg['teacher'], args=args,
            train_dataset=dataset, processing_class=tokenizer,
            selection_config={'keep_ratio': ratio, 'probability_threshold': .01},
            score_beta=beta, rollout_stream=raw, metrics_stream=metrics, run_id=output.name,
        )
        try:
            result = trainer.train()
            trainer.save_state()
            if cfg.get('save_final', True):
                trainer.save_model(str(output/'final'))
                tokenizer.save_pretrained(output/'final')
            if cfg.get('verify_runtime', False):
                if not cfg.get('save_final', True):
                    raise ValueError('verify_runtime requires save_final')
                from .verify_runtime import verify_runtime
                verify_runtime(trainer, output)
            (output/'result.json').write_text(json.dumps(result.metrics, indent=2)+'\n')
        finally:
            trainer._rollout_verifier.close()
            # The colocated single-process engine initializes a process group.
            # Release it explicitly after all training/evaluation work completes.
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()


if __name__ == '__main__':
    main()

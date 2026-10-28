"""Optional end-to-end checks for the Qwen3 smoke configuration, not training logic."""
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.no_grad()
def verify_runtime(trainer, output):
    output = Path(output)
    student = trainer.accelerator.unwrap_model(trainer.model)
    assert student.config.model_type == 'qwen3', 'This audit covers Qwen3 only'
    report = {}
    if trainer.use_vllm:
        vg = trainer.vllm_generation
        vg.sync_weights()
        engine = vg.llm.llm_engine.model_executor.driver_worker.model_runner.model
        source = dict(student.named_parameters())
        checked = 0
        for name, weight in engine.named_parameters():
            if '.qkv_proj.' in name:
                expected = torch.cat([source[name.replace('.qkv_proj.', f'.{p}_proj.')]
                                      for p in ('q', 'k', 'v')], dim=0)
            elif '.gate_up_proj.' in name:
                expected = torch.cat([source[name.replace('.gate_up_proj.', f'.{p}_proj.')]
                                      for p in ('gate', 'up')], dim=0)
            else:
                expected = source[name]
            assert weight.shape == expected.shape, (name, weight.shape, expected.shape)
            assert torch.equal(weight, expected.to(weight.dtype)), f'vLLM stale or mismatched weight: {name}'
            checked += weight.numel()
        report['vllm_synced_parameter_elements'] = checked
        report['vllm_sync_exact_at_engine_dtype'] = True
        if vg.enable_sleep_mode:
            vg.llm.wake_up(tags=['kv_cache'])
            vg.llm.sleep(level=2)
    # Every serialized parameter must exactly match the live trained model.
    source = student.state_dict()
    checked = 0
    for shard in sorted((output/'final').glob('*.safetensors')):
        with safe_open(shard, framework='pt', device='cpu') as saved:
            for name in saved.keys():
                weight = saved.get_tensor(name)
                assert torch.equal(weight, source[name].cpu()), f'Save mismatch: {name}'
                checked += weight.numel()
    assert checked > 0
    report['saved_parameter_elements_checked'] = checked
    # Exercise normal from_pretrained loading and autoregressive inference.
    restored = AutoModelForCausalLM.from_pretrained(
        output/'final', dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True,
    ).to(trainer.accelerator.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(output/'final', local_files_only=True)
    ids = tokenizer.apply_chat_template(
        [{'role':'user', 'content':'What is 2 + 2?'}], add_generation_prompt=True,
        enable_thinking=False, tokenize=True, return_dict=False,
    )
    inputs = torch.tensor([ids], device=trainer.accelerator.device)
    generated = restored.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                  max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    completion = generated[0, len(ids):].tolist()
    assert 0 < len(completion) <= 8
    report['reloaded_generation_token_count'] = len(completion)
    report['reloaded_generation_text'] = tokenizer.decode(completion)
    report['status'] = 'passed'
    (output/'runtime_verification.json').write_text(json.dumps(report, indent=2)+'\n')
    return report

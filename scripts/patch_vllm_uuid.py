"""Add UUID support to vLLM 0.19.1's NVML lookup, including child processes.

Run with the project's venv Python. Never use system/base Python.
"""
import ast
import hashlib
from importlib.metadata import distribution
import json
from pathlib import Path
import sys

assert sys.prefix != sys.base_prefix, 'An isolated venv is required'
dist = distribution('vllm')
assert dist.version == '0.19.1', 'Patch targets vLLM 0.19.1 only'
path = Path(dist.locate_file('vllm/platforms/interface.py')).resolve()
assert path.is_relative_to(Path(sys.prefix).resolve())
text = path.read_text()
marker = '# DIAL compatibility: CUDA accepts GPU UUIDs; NVML expects its own index.'
before = hashlib.sha256(path.read_bytes()).hexdigest()
if marker not in text:
    old = '            return int(physical_device_id)'
    assert text.count(old) == 1
    new = '''            # DIAL compatibility: CUDA accepts GPU UUIDs; NVML expects its own index.
            if physical_device_id.startswith("GPU-"):
                import pynvml
                pynvml.nvmlInit()
                try:
                    handle = pynvml.nvmlDeviceGetHandleByUUID(physical_device_id)
                    return pynvml.nvmlDeviceGetIndex(handle)
                finally:
                    pynvml.nvmlShutdown()
            return int(physical_device_id)'''
    text = text.replace(old, new)
    ast.parse(text)
    path.write_text(text)
report = {'file': str(path), 'version': dist.version, 'before_sha256': before,
          'after_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
record = Path(__file__).resolve().parents[1]/'recovery/vllm_uuid_patch.json'
record.parent.mkdir(exist_ok=True)
if not record.exists():
    record.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report))

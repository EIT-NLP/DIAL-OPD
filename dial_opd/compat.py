"""Narrow runtime compatibility fixes; no changes to the learning algorithm."""
import os


def enable_vllm_gpu_uuid_support():
    """vLLM 0.19.1's NVML helper assumes numeric CUDA_VISIBLE_DEVICES.

    CUDA itself accepts UUIDs, which avoid container/NVML index ambiguity.
    Preserve the UUID mask and resolve only NVML metadata lookups by UUID.
    """
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if not any(item.startswith('GPU-') for item in visible):
        return
    import pynvml
    from vllm.platforms.interface import Platform
    previous = Platform.device_id_to_physical_device_id.__func__

    def physical_index(cls, device_id):
        ids = os.environ.get(cls.device_control_env_var, '').split(',')
        if ids and ids[device_id].startswith('GPU-'):
            pynvml.nvmlInit()
            try:
                handle = pynvml.nvmlDeviceGetHandleByUUID(ids[device_id])
                return pynvml.nvmlDeviceGetIndex(handle)
            finally:
                pynvml.nvmlShutdown()
        return previous(cls, device_id)

    Platform.device_id_to_physical_device_id = classmethod(physical_index)


def configure_confined_single_process_vllm():
    """Use an in-process lock when the entire engine runs in one process.

    This opt-in avoids creating semaphore files in global /dev/shm. It must
    never be used for a multiprocessing or tensor-parallel engine.
    """
    if os.environ.get('DIAL_CONFINED_SINGLE_PROCESS') != '1':
        return
    if os.environ.get('VLLM_ENABLE_V1_MULTIPROCESSING') != '0':
        raise ValueError('Confined lock adapter requires in-process vLLM')
    import threading
    import vllm.v1.executor.uniproc_executor as executor
    executor.Lock = threading.Lock

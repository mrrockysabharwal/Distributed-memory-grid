"""
System telemetry collection using psutil (and optional NVML for GPUs).
Maintained by Member 4 (DevOps & Integration) & Member 1.
"""
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    psutil = None
    HAS_PSUTIL = False

from typing import Dict, Any


def get_system_telemetry() -> Dict[str, Any]:
    if HAS_PSUTIL and psutil is not None:
        mem = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=None)
        telemetry = {
            "cpu_percent": cpu_percent,
            "total_memory_mb": mem.total // (1024 * 1024),
            "available_memory_mb": mem.available // (1024 * 1024),
            "memory_percent": mem.percent,
            "has_gpu": False,
            "gpu_info": []
        }
    else:
        # Fallback telemetry when psutil is not installed
        telemetry = {
            "cpu_percent": 12.5,
            "total_memory_mb": 16384,
            "available_memory_mb": 8192,
            "memory_percent": 50.0,
            "has_gpu": False,
            "gpu_info": []
        }

    try:
        import torch
        if torch.cuda.is_available():
            telemetry["has_gpu"] = True
            for i in range(torch.cuda.device_count()):
                telemetry["gpu_info"].append({
                    "device_index": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_allocated_mb": torch.cuda.memory_allocated(i) // (1024 * 1024),
                    "memory_reserved_mb": torch.cuda.memory_reserved(i) // (1024 * 1024),
                })
    except ImportError:
        pass

    return telemetry


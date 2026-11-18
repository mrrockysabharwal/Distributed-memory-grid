"""
System telemetry collection using psutil (and optional NVML for GPUs).
Maintained by Member 4 (DevOps & Integration) & Member 1.
"""
import psutil
from typing import Dict, Any


def get_system_telemetry() -> Dict[str, Any]:
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


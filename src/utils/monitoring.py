"""Resource monitoring utilities for CPU, RAM, and GPU usage.

NOTE: This monitors PROCESS-LEVEL resources (CPU/RAM for this Python process only),
not system-wide usage. GPU metrics are device-level (entire GPU).
"""

import threading
import time
import psutil
import torch


class ResourceMonitor:
    """Context manager for monitoring process-level CPU/RAM and device-level GPU resources.
    
    CPU & RAM: Tracks only this Python process (not system-wide).
    GPU: Tracks entire GPU device (memory and utilization).
    """
    
    def __init__(self):
        self.cpu_samples = []
        self.ram_samples = []
        self.gpu_util_samples = []
        self._stop_event = threading.Event()
        self._monitor_thread = None
        self._gpu_start_ts_us = 0
        
    def _poll_resources(self):
        """Background thread: poll process-level CPU and RAM usage."""
        proc = psutil.Process()
        while not self._stop_event.is_set():
            self.cpu_samples.append(proc.cpu_percent(interval=None))
            self.ram_samples.append(proc.memory_info().rss / 1e6)
            self._stop_event.wait(timeout=0.1)
    
    def __enter__(self):
        """Start resource monitoring."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        self._gpu_start_ts_us = time.time_ns() // 1000
        
        self._monitor_thread = threading.Thread(target=self._poll_resources, daemon=True)
        self._monitor_thread.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop resource monitoring and collect GPU utilization."""
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join()
        
        # Collect GPU utilization samples if available
        if torch.cuda.is_available():
            try:
                import pynvml  # type: ignore[import]
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                _, samples = pynvml.nvmlDeviceGetSamples(
                    handle, pynvml.NVML_GPU_UTILIZATION_SAMPLES, self._gpu_start_ts_us
                )
                self.gpu_util_samples = [s.sampleValue.uiVal for s in samples]
            except Exception:
                self.gpu_util_samples = []
    
    def get_stats(self) -> dict:
        """Get collected resource statistics.
        
        Returns:
            Dict with:
            - peak_gpu_mem_mb: Peak GPU memory allocated by this process (MB), None if no GPU
            - peak_gpu_util_pct: Peak GPU utilization across entire device (%), None if no GPU
            - avg_gpu_util_pct: Average GPU utilization across entire device (%), None if no GPU
            - peak_ram_mb: Peak RAM used by this process (MB)
            - avg_cpu_percent: Average CPU usage by this process (%)
        """
        gpu_available = torch.cuda.is_available()
        return {
            'peak_gpu_mem_mb': torch.cuda.max_memory_allocated() / 1e6 if gpu_available else None,
            'peak_gpu_util_pct': max(self.gpu_util_samples, default=None) if self.gpu_util_samples else None,
            'avg_gpu_util_pct': (sum(self.gpu_util_samples) / len(self.gpu_util_samples)) if self.gpu_util_samples else None,
            'peak_ram_mb': max(self.ram_samples, default=0.0),
            'avg_cpu_percent': sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0.0,
        }

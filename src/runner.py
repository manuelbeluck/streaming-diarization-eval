"""Main orchestrator for streaming diarization evaluation pipeline."""

import os
# Force PyTorch to disable weights_only=True for model loading compatibility.
# https://github.com/m-bain/whisperX/issues/1304
# Only official models are used
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import warnings

# Suppress deprecation noise from third-party packages before they are imported
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda")
warnings.filterwarnings("ignore", message=".*torchaudio.*list_audio_backends.*", category=UserWarning)

import shutil
import threading
import time

import numpy as np
import psutil
import torch

from src.config import DatasetConfig, SystemConfig, load_config
from src.dataset.base import DatasetProvider, Recording, Segment
from src.dataset.utils import write_rttm
from src.systems.base import StreamingDiarizationSystem
from src.evaluator.metrics import compute_der, compute_jer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Silence noisy third-party loggers
# for _noisy in [
#     "speechbrain", "pytorch_lightning", "lightning",
#     "nv_one_logger", "nemo_logger", "nemo",
# ]:
#     logging.getLogger(_noisy).setLevel(logging.ERROR)

# Suppress root-logger warnings from third-party libs (e.g. diart sample-rate warning)
# We filter only records that actually originate from the root logger (name == "root"),
# leaving propagated records from our own named loggers (e.g. __main__, src.*) untouched.
class _RootNoise(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.name != "root" or record.levelno >= logging.ERROR

logging.getLogger().addFilter(_RootNoise())


def _poll_resources(stop_event: threading.Event, stats: dict) -> None:
    """Background thread: poll process-level CPU and RAM usage."""
    proc = psutil.Process()
    while not stop_event.is_set():
        stats['cpu_percent'].append(proc.cpu_percent(interval=None))
        stats['ram_mb'].append(proc.memory_info().rss / 1e6)
        stop_event.wait(timeout=0.1)


def _get_gpu_util_samples(start_ts_us: int) -> list[int]:
    """Return GPU compute utilization samples (%) collected since start_ts_us.
    
    Uses NVML's internal sample buffer (~167 ms resolution).
    Returns an empty list if NVML / pynvml is unavailable.
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        _, samples = pynvml.nvmlDeviceGetSamples(
            handle, pynvml.NVML_GPU_UTILIZATION_SAMPLES, start_ts_us
        )
        return [s.sampleValue.uiVal for s in samples]
    except Exception:
        return []


@dataclass
class EvaluationResult:
    """Results from evaluating a system on a recording."""
    recording_id: str
    system: str
    DER: float
    false_alarm: float
    missed_detection: float
    confusion: float
    JER: float
    latency_mean_ms: float
    latency_std_ms: float
    peak_latency_ms: float
    peak_latency_chunk_idx: int
    num_chunks: int
    step_size_ms: float
    duration: float
    peak_gpu_mem_mb: float
    peak_gpu_util_pct: float
    avg_gpu_util_pct: float
    peak_ram_mb: float
    avg_cpu_percent: float
    wall_time_s: float
    total_speech_time: float

    def to_dict(self) -> dict[str, str | float | int]:
        """Convert to dictionary for CSV writing."""
        return asdict(self)
    
    @classmethod
    def get_csv_headers(cls) -> list[str]:
        """Get CSV column headers."""
        return [
            'recording_id', 'system', 'DER', 'false_alarm', 'missed_detection',
            'confusion', 'JER', 'latency_mean_ms', 'latency_std_ms',
            'peak_latency_ms', 'peak_latency_chunk_idx', 'num_chunks', 'step_size_ms', 'duration',
            'peak_gpu_mem_mb', 'peak_gpu_util_pct', 'avg_gpu_util_pct', 'peak_ram_mb', 'avg_cpu_percent', 'wall_time_s', 'total_speech_time'
        ]


@dataclass
class EvaluationSummary:
    """Summary statistics across multiple evaluation results."""
    system: str
    num_recordings: int
    avg_der: float
    avg_false_alarm: float
    avg_missed_detection: float
    avg_confusion: float
    avg_jer: float
    avg_latency_ms: float
    avg_latency_std_ms: float
    avg_peak_latency_ms: float
    total_chunks: int
    avg_wall_time_s: float

    @classmethod
    def from_results(cls, system_name: str, results: list[EvaluationResult]) -> 'EvaluationSummary':
        """Compute summary statistics from a list of results."""
        if not results:
            raise ValueError("Cannot create summary from empty results list")
        n = len(results)
        return cls(
            system=system_name,
            num_recordings=n,
            avg_der=sum(r.DER for r in results) / n,
            avg_false_alarm=sum(r.false_alarm for r in results) / n,
            avg_missed_detection=sum(r.missed_detection for r in results) / n,
            avg_confusion=sum(r.confusion for r in results) / n,
            avg_jer=sum(r.JER for r in results) / n,
            avg_latency_ms=sum(r.latency_mean_ms for r in results) / n,
            avg_latency_std_ms=sum(r.latency_std_ms for r in results) / n,
            avg_peak_latency_ms=sum(r.peak_latency_ms for r in results) / n,
            total_chunks=sum(r.num_chunks for r in results),
            avg_wall_time_s=sum(r.wall_time_s for r in results) / n
        )

    def log_summary(self, logger_func) -> None:
        """Log summary statistics."""
        logger_func(f"{self.system}:")
        logger_func(f"  DER: {self.avg_der:.3f}  (FA: {self.avg_false_alarm:.3f}  "
                    f"Miss: {self.avg_missed_detection:.3f}  Conf: {self.avg_confusion:.3f})")
        logger_func(f"  JER: {self.avg_jer:.3f}")
        logger_func(f"  Avg wall time: {self.avg_wall_time_s:.2f}s per recording")
        logger_func(f"  Chunk Latency: {self.avg_latency_ms:.2f}±{self.avg_latency_std_ms:.2f}ms  "
                    f"| avg peak chunk: {self.avg_peak_latency_ms:.2f}ms")
        logger_func(f"  Total Chunks: {self.total_chunks}")
        logger_func(f"  Processed: {self.num_recordings} recordings")


def create_dataset(config: DatasetConfig) -> DatasetProvider:
    """Create dataset provider from configuration."""
    dataset_type = config.name.lower()
    
    if dataset_type == 'test':
        from src.dataset.testdataset import TestDataset
        return TestDataset(
            data_dir=config.path,
            max_duration=config.max_duration
        )
    elif dataset_type == 'callhome':
        from src.dataset.callhome import CallHomeDataset
        return CallHomeDataset(
            language=config.language,
            data_dir=config.path,
            recordings=config.recordings,
            max_duration=config.max_duration
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def create_system(config: SystemConfig) -> StreamingDiarizationSystem:
    """Create system from configuration."""
    system_name = config.name.lower()
    
    if system_name == 'diart_default':
        from src.systems.diart.system import DiartSystem
        return DiartSystem(
            name='diart_default',
            duration=config.duration or 5.0,
            step=config.step or 0.5
            # Uses DIART default models
        )
    elif system_name == 'diart_custom':
        from src.systems.diart.system import DiartSystem
        return DiartSystem(
            name='diart_custom',
            duration=config.duration or 5.0,
            step=config.step or 0.5,
            segmentation_model='pyannote/segmentation-3.0',
            embedding_model='pyannote/wespeaker-voxceleb-resnet34-LM'
        )
    elif system_name == 'streaming_sortformer':
        from src.systems.sortformer.system import SortformerSystem
        return SortformerSystem(
            chunk_len=config.chunk_len or 10,
            subsampling_factor=config.subsampling_factor or 10,
            chunk_right_context=config.chunk_right_context or 0,
            chunk_left_context=config.chunk_left_context or 10,
            spkcache_len=config.spkcache_len or 188,
            fifo_len=config.fifo_len or 188,
            spkcache_update_period=config.spkcache_update_period or 144,
            log=config.log or False,
            chunk_size=config.chunk_size,  # Can be None
            overlap_aware=config.overlap_aware or False
        )
    else:
        raise ValueError(f"Unknown system: {system_name}")


def run_system_evaluation(
    system: StreamingDiarizationSystem,
    recording: Recording,
    audio: np.ndarray
) -> list[Segment]:
    """Run system evaluation for one recording."""
    logger.info(f"Processing {recording.recording_id} with {system.name}")
    prediction = system.run(
        audio=audio,
        sample_rate=recording.sample_rate
    )
    logger.info(f"  Completed: {len(prediction)} segments detected")
    return prediction


def evaluate_system(
    system: StreamingDiarizationSystem,
    dataset: DatasetProvider,
    output_dir: Path,
    collar: float
) -> list[EvaluationResult]:
    """
    Evaluate system on entire dataset.
    
    Args:
        system: Diarization system
        dataset: Dataset provider
        output_dir: Directory for outputs
        collar: Collar for DER calculation
        
    Returns:
        List of EvaluationResult objects
    """
    system_dir = output_dir / system.name
    system_dir.mkdir(parents=True, exist_ok=True)
    ground_truth_dir = output_dir / "ground_truth"
    ground_truth_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for recording in dataset.list_recordings():
        # Load audio and ground truth
        audio = dataset.get_audio(recording.recording_id)
        ground_truth = dataset.get_ground_truth(recording.recording_id)

        # Save ground truth RTTM (shared across systems; overwrite is fine)
        gt_path = ground_truth_dir / f"{recording.recording_id}.rttm"
        write_rttm(ground_truth, str(gt_path), recording.recording_id)
        
        # Run system with resource monitoring
        _stop = threading.Event()
        _stats: dict = {'cpu_percent': [], 'ram_mb': []}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        gpu_start_ts_us = time.time_ns() // 1000  # microseconds for NVML
        _monitor = threading.Thread(target=_poll_resources, args=(_stop, _stats), daemon=True)
        _monitor.start()

        _run_start = time.perf_counter()
        prediction = run_system_evaluation(system, recording, audio)
        wall_time_s = time.perf_counter() - _run_start

        _stop.set()
        _monitor.join()

        # Use torch's precise peak tracker (reset before run) instead of polling
        peak_gpu = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0
        gpu_util_samples = _get_gpu_util_samples(gpu_start_ts_us) if torch.cuda.is_available() else []
        peak_gpu_util = max(gpu_util_samples, default=0.0)
        avg_gpu_util = sum(gpu_util_samples) / len(gpu_util_samples) if gpu_util_samples else 0.0
        peak_ram = max(_stats['ram_mb'], default=0.0)
        avg_cpu = sum(_stats['cpu_percent']) / len(_stats['cpu_percent']) if _stats['cpu_percent'] else 0.0
        
        # Save prediction
        pred_path = system_dir / f"{recording.recording_id}.rttm"
        write_rttm(prediction, str(pred_path), recording.recording_id)
        
        # Compute metrics
        der_result = compute_der(prediction, ground_truth, collar=collar)
        jer = compute_jer(prediction, ground_truth, collar=collar)
        
        # Get chunk processing latency statistics
        # Get step size for reporting (DIART uses 'step', Sortformer may have 'chunk_size')
        step_or_chunk = getattr(system, 'step', None) or getattr(system, 'chunk_size', 0.0) or 0.0
        latency_stats = system.get_latency_stats()

        # Find the chunk with the highest latency
        latencies_ms = [lat * 1000 for lat in system.latencies]
        if latencies_ms:
            peak_lat_idx = int(np.argmax(latencies_ms))
            peak_lat_ms = float(latencies_ms[peak_lat_idx])
        else:
            peak_lat_idx, peak_lat_ms = 0, 0.0

        result = EvaluationResult(
            recording_id=recording.recording_id,
            system=system.name,
            DER=der_result['DER'],
            false_alarm=der_result['false_alarm'],
            missed_detection=der_result['missed_detection'],
            confusion=der_result['confusion'],
            JER=jer,
            latency_mean_ms=latency_stats['latency_mean_ms'],
            latency_std_ms=latency_stats['latency_std_ms'],
            peak_latency_ms=peak_lat_ms,
            peak_latency_chunk_idx=peak_lat_idx,
            num_chunks=int(latency_stats['num_chunks']),
            step_size_ms=step_or_chunk * 1000 if step_or_chunk else 0.0,
            duration=recording.duration,
            peak_gpu_mem_mb=peak_gpu,
            peak_gpu_util_pct=peak_gpu_util,
            avg_gpu_util_pct=avg_gpu_util,
            peak_ram_mb=peak_ram,
            avg_cpu_percent=avg_cpu,
            wall_time_s=wall_time_s,
            total_speech_time=der_result['total']
        )
        
        logger.info(f"  DER: {result.DER:.3f}  (FA: {result.false_alarm:.3f}  Miss: {result.missed_detection:.3f}  Conf: {result.confusion:.3f})")
        logger.info(f"  total speech time: {result.total_speech_time:.2f}s")
        logger.info(f"  JER: {result.JER:.3f}")
        rtf_str = f"{result.wall_time_s / result.duration:.3f}x" if result.duration else "n/a"
        logger.info(f"  Wall time: {result.wall_time_s:.2f}s  (audio: {result.duration:.2f}s | RTF: {rtf_str})")
        logger.info(f"  Chunk Latency: {result.latency_mean_ms:.2f}±{result.latency_std_ms:.2f}ms  "
                   f"| peak: {result.peak_latency_ms:.2f}ms at chunk #{result.peak_latency_chunk_idx}  "
                   f"| total chunks: {result.num_chunks}")
        logger.info(f"  GPU memory: {result.peak_gpu_mem_mb:.0f} MB peak"
                   f"  |  RAM: {result.peak_ram_mb:.0f} MB peak  |  CPU: {result.avg_cpu_percent:.1f}% avg")

        # Saturation warnings
        if torch.cuda.is_available():
            total_gpu_mem_mb = torch.cuda.get_device_properties(0).total_memory / 1e6
            gpu_mem_usage_pct = (peak_gpu / total_gpu_mem_mb * 100) if total_gpu_mem_mb else 0.0
            logger.info(f"  GPU memory usage: {gpu_mem_usage_pct:.1f}% of total ({peak_gpu:.0f} / {total_gpu_mem_mb:.0f} MB)")
            if gpu_util_samples:
                logger.info(f"  GPU compute utilization: {peak_gpu_util:.0f}% peak / {avg_gpu_util:.0f}% avg "
                           f"({len(gpu_util_samples)} samples)")
            else:
                logger.info("  GPU compute utilization: n/a (pynvml unavailable)")
            if gpu_mem_usage_pct >= 95:
                logger.warning(
                    f"  !! GPU memory saturated: {peak_gpu:.0f} / {total_gpu_mem_mb:.0f} MB "
                    f"({gpu_mem_usage_pct:.1f}%) — risk of OOM"
                )
            if peak_gpu_util >= 95:
                logger.warning(
                    f"  !! GPU compute saturated: peak {peak_gpu_util:.0f}% "
                    f"— system is compute-bound on GPU"
                )
        if avg_cpu >= 95:
            logger.warning(
                f"  !! CPU saturated: avg {avg_cpu:.1f}% "
                f"— system is likely CPU-bound"
            )
            
        
        results.append(result)
            
    return results


def save_results(results: list[EvaluationResult], output_path: Path):
    """Save results to CSV file."""
    import csv
    
    if not results:
        logger.warning("No results to save")
        return
    
    # Use EvaluationResult methods for CSV handling
    fieldnames = EvaluationResult.get_csv_headers()
    result_dicts = [result.to_dict() for result in results]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result_dicts)
    
    logger.info(f"Results saved to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Evaluate streaming diarization systems'
    )
    parser.add_argument(
        'config',
        type=str,
        nargs='?',
        default='config.yaml',
        help='Path to configuration YAML file. Default: config.yaml'
    )
    args = parser.parse_args()
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Log GPU / CPU info
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total_gpu_mem = props.total_memory / 1e9
        logger.info(
            f"GPU: {props.name}  |  "
            f"Total memory: {total_gpu_mem:.1f} GB  |  "
            f"SM count: {props.multi_processor_count}  |  "
            f"Compute capability: {props.major}.{props.minor}"
        )
    else:
        logger.info("GPU: not available, running on CPU")

    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Store a copy of the config used for this run
    config_copy_path = output_dir / "config.yaml"
    shutil.copy(args.config, config_copy_path)
    logger.info(f"Config saved to {config_copy_path}")
    
    # Create dataset
    logger.info("Initializing dataset...")
    dataset = create_dataset(config.dataset)
    recordings = dataset.list_recordings()
    logger.info(f"Found {len(recordings)} recordings")
    
    # Create systems
    logger.info("Initializing systems...")
    systems = [create_system(sys_config) for sys_config in config.systems]
    logger.info(f"Loaded {len(systems)} systems: {[s.name for s in systems]}")
    
    # Evaluation parameters
    collar = config.evaluation.collar
    
    # Evaluate each system
    all_results: list[EvaluationResult] = []
    for system in systems: # incase there are multiple systems, what i doubt, but just in case
        logger.info(f"{'='*30}")
        logger.info(f"Evaluating {system.name}")
        logger.info(f"{'='*30}")
        
        results = evaluate_system(
            system=system,
            dataset=dataset,
            output_dir=output_dir,
            collar=collar
        )
        all_results.extend(results)
    
    # Save results
    results_path = output_dir / 'metrics.csv'
    save_results(all_results, results_path)
    
    # Print summary
    logger.info(f"{'='*30}")
    logger.info("SUMMARY")
    logger.info(f"{'='*30}")
    
    for system in systems:
        system_results = [r for r in all_results if r.system == system.name]
        
        if system_results:
            summary = EvaluationSummary.from_results(system.name, system_results)
            summary.log_summary(logger.info)
        else:
            logger.warning(f"{system.name}: No valid results")
    
    logger.info(f"\nDone! Results saved to {output_dir}")


if __name__ == '__main__':
    main()

"""Core evaluation logic for computing results from predictions and ground truth."""

import numpy as np
from src.dataset.base import Recording, Segment
from src.systems.base import StreamingDiarizationSystem
from .metrics import compute_der, compute_jer, compute_speaker_overlap_stats
from .results import EvaluationResult


def evaluate_recording(
    recording: Recording,
    prediction: list[Segment],
    ground_truth: list[Segment],
    system: StreamingDiarizationSystem,
    resource_stats: dict,
    wall_time_s: float,
    collar: float = 0.25
) -> EvaluationResult:
    """Compute complete evaluation result for a recording.
    
    Args:
        recording: Recording metadata
        prediction: Predicted diarization segments
        ground_truth: Ground truth segments
        system: Diarization system used
        resource_stats: Dict with resource usage stats (from ResourceMonitor.get_stats())
        wall_time_s: Total wall clock time for processing
        collar: Forgiveness collar for metrics (default 0.25s)
        
    Returns:
        Complete EvaluationResult object
    """
    # Compute diarization metrics
    der_result = compute_der(prediction, ground_truth, collar=collar)
    jer = compute_jer(prediction, ground_truth, collar=collar)
    
    # Compute ground-truth-level stats (num_speakers, overlap_rate)
    gt_stats = compute_speaker_overlap_stats(ground_truth, recording.duration)
    
    # Mean and min segment confidence from prediction (None = unavailable, e.g. DIART)
    conf_values = [s.confidence for s in prediction if s.confidence is not None]
    avg_segment_confidence = float(np.mean(conf_values)) if conf_values else None
    min_segment_confidence = float(np.min(conf_values)) if conf_values else None
    
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
    
    return EvaluationResult(
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
        peak_gpu_mem_mb=resource_stats['peak_gpu_mem_mb'],
        peak_gpu_util_pct=resource_stats['peak_gpu_util_pct'],
        avg_gpu_util_pct=resource_stats['avg_gpu_util_pct'],
        peak_ram_mb=resource_stats['peak_ram_mb'],
        avg_cpu_percent=resource_stats['avg_cpu_percent'],
        wall_time_s=wall_time_s,
        total_speech_time=der_result['total'],
        gt_num_speakers=gt_stats['num_speakers'],
        gt_overlap_rate=gt_stats['overlap_rate'],
        avg_segment_confidence=avg_segment_confidence,
        min_segment_confidence=min_segment_confidence
    )

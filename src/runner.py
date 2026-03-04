"""Main orchestrator for streaming diarization evaluation pipeline."""

import os
# Force PyTorch to disable weights_only=True for model loading compatibility.
# https://github.com/m-bain/whisperX/issues/1304
# Only official models are used
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'

import argparse
from pathlib import Path
import logging

import shutil
import time

import numpy as np
import torch

from src.config import load_config
from src.dataset.base import DatasetProvider, Recording, Segment
from src.dataset.factory import create_dataset
from src.dataset.utils import write_rttm
from src.systems.base import StreamingDiarizationSystem
from src.systems.factory import create_system
from src.evaluator import evaluate_recording, EvaluationResult, EvaluationSummary
from src.utils.monitoring import ResourceMonitor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
        with ResourceMonitor() as monitor:
            _run_start = time.perf_counter()
            prediction = run_system_evaluation(system, recording, audio)
            wall_time_s = time.perf_counter() - _run_start
        
        # Get collected resource stats
        resource_stats = monitor.get_stats()
        
        # Save prediction
        pred_path = system_dir / f"{recording.recording_id}.rttm"
        write_rttm(prediction, str(pred_path), recording.recording_id)
        
        # Evaluate recording
        result = evaluate_recording(
            recording=recording,
            prediction=prediction,
            ground_truth=ground_truth,
            system=system,
            resource_stats=resource_stats,
            wall_time_s=wall_time_s,
            collar=collar
        )
        
        logger.info(f"  DER: {result.DER:.3f}  (FA: {result.false_alarm:.3f}  Miss: {result.missed_detection:.3f}  Conf: {result.confusion:.3f})")
        logger.info(f"  total speech time: {result.total_speech_time:.2f}s")
        confidence_info = ""
        if result.avg_segment_confidence is not None:
            confidence_info = f"  |  Segment confidence: avg={result.avg_segment_confidence:.3f}, min={result.min_segment_confidence:.3f}"
        logger.info(f"  GT speakers: {result.gt_num_speakers}  |  "
                   f"GT overlap rate: {result.gt_overlap_rate*100:.1f}%" +
                   confidence_info)
        logger.info(f"  JER: {result.JER:.3f}")
        rtf_str = f"{result.wall_time_s / result.duration:.3f}x" if result.duration else "n/a"
        logger.info(f"  Wall time: {result.wall_time_s:.2f}s  (audio: {result.duration:.2f}s | RTF: {rtf_str})")
        logger.info(f"  Chunk Latency: {result.latency_mean_ms:.2f}±{result.latency_std_ms:.2f}ms  "
                   f"| peak: {result.peak_latency_ms:.2f}ms at chunk #{result.peak_latency_chunk_idx}  "
                   f"| total chunks: {result.num_chunks}")
        gpu_info = ""
        if result.peak_gpu_mem_mb is not None:
            gpu_info = f"  GPU memory: {result.peak_gpu_mem_mb:.0f} MB peak  |  "
        logger.info(f"{gpu_info}RAM: {result.peak_ram_mb:.0f} MB peak  |  CPU: {result.avg_cpu_percent:.1f}% avg")

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

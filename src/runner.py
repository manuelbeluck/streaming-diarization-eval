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

import numpy as np

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
    latency_first_chunk_ms: float
    num_chunks: int
    step_size_ms: float
    duration: float
    
    def to_dict(self) -> dict[str, str | float | int]:
        """Convert to dictionary for CSV writing."""
        return asdict(self)
    
    @classmethod
    def get_csv_headers(cls) -> list[str]:
        """Get CSV column headers."""
        return [
            'recording_id', 'system', 'DER', 'false_alarm', 'missed_detection', 
            'confusion', 'JER', 'latency_mean_ms', 'latency_std_ms', 
            'latency_first_chunk_ms', 'num_chunks', 'step_size_ms', 'duration'
        ]


@dataclass
class EvaluationSummary:
    """Summary statistics across multiple evaluation results."""
    system: str
    num_recordings: int
    avg_der: float
    avg_jer: float
    avg_latency_ms: float
    avg_latency_std_ms: float
    avg_first_chunk_ms: float
    total_chunks: int
    
    @classmethod
    def from_results(cls, system_name: str, results: list[EvaluationResult]) -> 'EvaluationSummary':
        """Compute summary statistics from a list of results."""
        if not results:
            raise ValueError("Cannot create summary from empty results list")
        
        return cls(
            system=system_name,
            num_recordings=len(results),
            avg_der=sum(r.DER for r in results) / len(results),
            avg_jer=sum(r.JER for r in results) / len(results),
            avg_latency_ms=sum(r.latency_mean_ms for r in results) / len(results),
            avg_latency_std_ms=sum(r.latency_std_ms for r in results) / len(results),
            avg_first_chunk_ms=sum(r.latency_first_chunk_ms for r in results) / len(results),
            total_chunks=sum(r.num_chunks for r in results)
        )
    
    def log_summary(self, logger_func) -> None:
        """Log summary statistics."""
        logger_func(f"{self.system}:")
        logger_func(f"  Average DER: {self.avg_der:.3f}")
        logger_func(f"  Average JER: {self.avg_jer:.3f}")
        logger_func(f"  Average Chunk Processing Latency: {self.avg_latency_ms:.2f}±{self.avg_latency_std_ms:.2f}ms (first: {self.avg_first_chunk_ms:.2f}ms)")
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
            chunk_size=config.chunk_size  # Can be None
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
    
    results = []
    
    for recording in dataset.list_recordings():
        # Load audio and ground truth
        audio = dataset.get_audio(recording.recording_id)
        ground_truth = dataset.get_ground_truth(recording.recording_id)
        
        # Run system
        prediction = run_system_evaluation(system, recording, audio)
        
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
            latency_first_chunk_ms=latency_stats['latency_first_chunk_ms'],
            num_chunks=int(latency_stats['num_chunks']),
            step_size_ms=step_or_chunk * 1000 if step_or_chunk else 0.0,
            duration=recording.duration
        )
        
        logger.info(f"  DER: {result.DER:.3f}, JER: {result.JER:.3f}")
        logger.info(f"  Chunk Processing Latency: {result.latency_mean_ms:.2f}±{result.latency_std_ms:.2f}ms "
                   f"(first: {result.latency_first_chunk_ms:.2f}ms, {result.num_chunks} chunks)")
            
        
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
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
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

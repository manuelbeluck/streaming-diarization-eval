"""Main orchestrator for streaming diarization evaluation pipeline."""

import os
# Force PyTorch to disable weights_only=True for model loading compatibility.
# https://github.com/m-bain/whisperX/issues/1304
# Only official models are used
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'

import argparse
from pathlib import Path
import logging

import numpy as np

from config import DatasetConfig, SystemConfig, load_config
from dataset.base import DatasetProvider, Recording, Segment
from dataset.utils import write_rttm
from systems.base import StreamingDiarizationSystem
from evaluator.metrics import compute_der, compute_jer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_dataset(config: DatasetConfig) -> DatasetProvider:
    """Create dataset provider from configuration."""
    dataset_type = config.name.lower()
    
    if dataset_type == 'test':
        from dataset.testdataset import TestDataset
        return TestDataset(
            data_dir=config.path,
            max_duration=config.max_duration
        )
    elif dataset_type == 'callhome':
        from dataset.callhome import CallHomeDataset
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
        from systems.diart.system import DiartSystem
        return DiartSystem(
            name='diart_default',
            duration=config.duration or 5.0,
            step=config.step or 0.5
            # Uses DIART default models
        )
    elif system_name == 'diart_custom':
        from systems.diart.system import DiartSystem
        return DiartSystem(
            name='diart_custom',
            duration=config.duration or 5.0,
            step=config.step or 0.5,
            segmentation_model='pyannote/segmentation-3.0',
            embedding_model='pyannote/wespeaker-voxceleb-resnet34-LM'
        )
    elif system_name == 'streaming_sortformer':
        from systems.sortformer.system import SortformerSystem
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
) -> list[dict]:
    """
    Evaluate system on entire dataset.
    
    Args:
        system: Diarization system
        dataset: Dataset provider
        output_dir: Directory for outputs
        collar: Collar for DER calculation
        
    Returns:
        Dictionary of results
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
        
        result = {
            'recording_id': recording.recording_id,
            'system': system.name,
            'DER': der_result['DER'],
            'false_alarm': der_result['false_alarm'],
            'missed_detection': der_result['missed_detection'],
            'confusion': der_result['confusion'],
            'JER': jer,
            'latency_mean_ms': latency_stats['latency_mean_ms'],
            'latency_std_ms': latency_stats['latency_std_ms'],
            'latency_first_chunk_ms': latency_stats['latency_first_chunk_ms'],
            'num_chunks': latency_stats['num_chunks'],
            'step_size_ms': step_or_chunk * 1000 if step_or_chunk else 0.0,
            'duration': recording.duration
        }
        
        logger.info(f"  DER: {der_result['DER']:.3f}, JER: {jer:.3f}")
        logger.info(f"  Chunk Processing Latency: {latency_stats['latency_mean_ms']:.2f}±{latency_stats['latency_std_ms']:.2f}ms "
                   f"(first: {latency_stats['latency_first_chunk_ms']:.2f}ms, {latency_stats['num_chunks']} chunks)")
            
        
        results.append(result)
            
    return results


def save_results(results: list[dict], output_path: Path):
    """Save results to CSV file."""
    import csv
    
    if not results:
        logger.warning("No results to save")
        return
    
    # Get all keys from all results
    fieldnames = set()
    for result in results:
        fieldnames.update(result.keys())
    fieldnames = sorted(fieldnames)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
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
    all_results = []
    for system in systems: # incase there are multiple systems, what i doubt, but just in case
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating {system.name}")
        logger.info(f"{'='*60}")
        
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
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    
    for system in systems:
        system_results = [r for r in all_results if r.get('system') == system.name]
        valid_results = [r for r in system_results if 'DER' in r]
        
        if valid_results:
            avg_der = sum(r['DER'] for r in valid_results) / len(valid_results)
            avg_jer = sum(r['JER'] for r in valid_results) / len(valid_results)
            avg_latency = sum(r.get('latency_mean_ms', 0) for r in valid_results) / len(valid_results)
            avg_latency_std = sum(r.get('latency_std_ms', 0) for r in valid_results) / len(valid_results)
            avg_first_chunk = sum(r.get('latency_first_chunk_ms', 0) for r in valid_results) / len(valid_results)
            total_chunks = sum(r.get('num_chunks', 0) for r in valid_results)
            
            logger.info(f"{system.name}:")
            logger.info(f"  Average DER: {avg_der:.3f}")
            logger.info(f"  Average JER: {avg_jer:.3f}")
            logger.info(f"  Average Chunk Processing Latency: {avg_latency:.2f}±{avg_latency_std:.2f}ms (first: {avg_first_chunk:.2f}ms)")
            logger.info(f"  Total Chunks: {total_chunks}")
            logger.info(f"  Processed: {len(valid_results)}/{len(system_results)} recordings")
        else:
            logger.warning(f"{system.name}: No valid results")
    
    logger.info(f"\nDone! Results saved to {output_dir}")


if __name__ == '__main__':
    main()

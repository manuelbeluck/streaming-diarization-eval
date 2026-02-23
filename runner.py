"""Main orchestrator for streaming diarization evaluation pipeline."""

import argparse
import yaml
from pathlib import Path
from typing import Dict, List
import logging

import numpy as np

from dataset.base import DatasetProvider, Recording, Segment
from dataset.utils import write_rttm
from systems.base import StreamingDiarizationSystem
from systems.diart.system import DiartSystem
from systems.sortformer.system import SortformerSystem
from evaluator.metrics import compute_der, compute_jer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dataset(config: Dict) -> DatasetProvider:
    """Create dataset provider from configuration."""
    dataset_type = config['name'].lower()
    
    if dataset_type == 'test':
        from dataset.testdataset import TestDataset
        return TestDataset(data_dir=config.get('path', 'data'))
    elif dataset_type == 'callhome':
        from dataset.callhome import CallHomeDataset
        return CallHomeDataset(
            language=config.get('language', 'eng'),
            data_dir=config.get('path', 'data/callhome'),
            recordings=config.get('recordings')
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def create_system(config: Dict) -> StreamingDiarizationSystem:
    """Create system from configuration."""
    system_name = config['name'].lower()
    chunk_size = config.get('chunk_size', 0.5)
    
    if system_name == 'diart':
        return DiartSystem(chunk_size=chunk_size)
    elif system_name == 'sortformer':
        return SortformerSystem(chunk_size=chunk_size)
    else:
        raise ValueError(f"Unknown system: {system_name}")


def run_system_evaluation(
    system: StreamingDiarizationSystem,
    recording: Recording,
    audio: np.ndarray
) -> List[Segment]:
    """Run system evaluation for one recording."""
    logger.info(f"Processing {recording.recording_id} with {system.name}")
    prediction = system.run(
        audio=audio,
        sample_rate=recording.sample_rate,
        num_speakers=recording.num_speakers
    )
    logger.info(f"  Completed: {len(prediction)} segments detected")
    return prediction


def evaluate_system(
    system: StreamingDiarizationSystem,
    dataset: DatasetProvider,
    output_dir: Path,
    collar: float
) -> List[Dict]:
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
        
        result = {
            'recording_id': recording.recording_id,
            'system': system.name,
            'DER': der_result['DER'],
            'false_alarm': der_result['false_alarm'],
            'missed_detection': der_result['missed_detection'],
            'confusion': der_result['confusion'],
            'JER': jer,
            'duration': recording.duration,
            'num_speakers': recording.num_speakers
        }
        
        logger.info(f"  DER: {der_result['DER']:.3f}, JER: {jer:.3f}")
            
        
        results.append(result)
            
    return results


def save_results(results: List[Dict], output_path: Path):
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
    output_dir = Path(config.get('output_dir', './results'))
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Create dataset
    logger.info("Initializing dataset...")
    dataset = create_dataset(config['dataset'])
    recordings = dataset.list_recordings()
    logger.info(f"Found {len(recordings)} recordings")
    
    # Create systems
    logger.info("Initializing systems...")
    systems = [create_system(sys_config) for sys_config in config['systems']]
    logger.info(f"Loaded {len(systems)} systems: {[s.name for s in systems]}")
    
    # Evaluation parameters
    collar = config.get('evaluation', {}).get('collar', 0.25)
    
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
            logger.info(f"{system.name}:")
            logger.info(f"  Average DER: {avg_der:.3f}")
            logger.info(f"  Average JER: {avg_jer:.3f}")
            logger.info(f"  Processed: {len(valid_results)}/{len(system_results)} recordings")
        else:
            logger.warning(f"{system.name}: No valid results")
    
    logger.info(f"\nDone! Results saved to {output_dir}")


if __name__ == '__main__':
    main()

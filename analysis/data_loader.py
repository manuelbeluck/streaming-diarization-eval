"""
Data loading and preprocessing utilities for diarization evaluation results.
"""

import pandas as pd
from pathlib import Path
from typing import Union


def load_metrics(csv_path: Union[str, Path], verbose: bool = True) -> pd.DataFrame:
    """
    Load metrics CSV file and create derived metrics.
    
    Args:
        csv_path: Path to the metrics CSV file
        verbose: Whether to print summary information
        
    Returns:
        DataFrame with metrics and derived columns
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    
    # Create derived metrics
    # Real-Time Factor (RTF): processing_time / audio_duration
    # RTF < 1.0 means faster than real-time
    df['rtf'] = df['wall_time_s'] / df['duration']
    
    if verbose:
        print(f"✓ Loaded {len(df)} results from {csv_path}")
        print(f"  Number of recordings: {df['recording_id'].nunique()}")
        print(f"  Systems evaluated: {', '.join(df['system'].unique())}")
        print(f"  Metrics captured: {len(df.columns)} columns")
        print(f"  Created derived metrics (RTF)")
    
    return df


def prepare_ground_truth_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare ground truth dataset characteristics (unique recordings only).
    
    Args:
        df: DataFrame with full metrics
        
    Returns:
        DataFrame with ground truth characteristics per recording
    """
    gt_data = df.groupby('recording_id').first()[
        ['gt_num_speakers', 'gt_overlap_rate', 'duration', 'total_speech_time']
    ].reset_index()
    
    # Convert overlap rate to percentage for visualization
    gt_data['overlap_rate_pct'] = gt_data['gt_overlap_rate'] * 100
    
    return gt_data

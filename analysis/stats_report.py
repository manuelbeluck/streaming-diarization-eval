"""
Statistical analysis functions for diarization evaluation results.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


def print_dataset_summary(df: pd.DataFrame) -> None:
    """
    Print a summary of the dataset and data overview.
    
    Args:
        df: DataFrame with metrics
    """
    print("\n" + "="*80)
    print("DATASET SUMMARY")
    print("="*80)
    print(f"Total recordings: {df['recording_id'].nunique()}")
    print(f"Systems evaluated: {', '.join(df['system'].unique())}")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print("="*80)


def print_der_component_analysis(df: pd.DataFrame) -> None:
    """
    Print detailed DER component analysis.
    
    Args:
        df: DataFrame with metrics
    """
    # Calculate DER components as percentage of total speech time
    der_components = df.groupby('system')[['false_alarm', 'missed_detection', 'confusion']].mean()
    total_speech = df.groupby('system')['total_speech_time'].mean()
    der_components_pct = (der_components.div(total_speech, axis=0) * 100)
    
    # Also get the raw DER for comparison
    der_values = df.groupby('system')['DER'].mean() * 100  # Convert to percentage
    
    # Print component analysis
    print("\n" + "="*80)
    print("DER COMPONENT ANALYSIS")
    print("="*80)
    for system in der_values.sort_values().index:
        fa = der_components_pct.loc[system, 'false_alarm']
        md = der_components_pct.loc[system, 'missed_detection']
        conf = der_components_pct.loc[system, 'confusion']
        total = der_values[system]
        
        print(f"\n{system}:")
        print(f"  Total DER:         {total:6.2f}%")
        print(f"  ├─ False Alarm:    {fa:6.2f}% ({fa/total*100:5.1f}% of DER)")
        print(f"  ├─ Missed Detect:  {md:6.2f}% ({md/total*100:5.1f}% of DER)")
        print(f"  └─ Confusion:      {conf:6.2f}% ({conf/total*100:5.1f}% of DER)")
    print("="*80)


def calculate_system_correlations(df: pd.DataFrame, metric: str = 'DER', 
                                  variable: str = 'gt_overlap_rate') -> Dict[str, float]:
    """
    Calculate correlation between a metric and a variable for each system.
    
    Args:
        df: DataFrame with metrics
        metric: Metric column name to correlate
        variable: Variable column name to correlate with
        
    Returns:
        Dictionary mapping system names to correlation values
    """
    correlations = {}
    for system in df['system'].unique():
        system_data = df[df['system'] == system]
        corr = system_data[metric].corr(system_data[variable])
        correlations[system] = corr
    
    return correlations


def print_correlation_analysis(df: pd.DataFrame, metric: str = 'DER', 
                               variable: str = 'gt_overlap_rate', 
                               variable_display_name: Optional[str] = None) -> None:
    """
    Print correlation analysis between a metric and a variable.
    
    Args:
        df: DataFrame with metrics
        metric: Metric column name to correlate
        variable: Variable column name to correlate with
        variable_display_name: Display name for the variable (default: variable name)
    """
    if variable_display_name is None:
        variable_display_name = variable
    
    correlations = calculate_system_correlations(df, metric, variable)
    
    print(f"\nCorrelation: {metric} vs {variable_display_name}")
    print("-" * 60)
    for system, corr in correlations.items():
        print(f"{system:25s}: {corr:+.3f}")


def print_rtf_statistics(df: pd.DataFrame) -> None:
    """
    Print Real-Time Factor (RTF) statistics.
    
    Args:
        df: DataFrame with metrics (must have 'rtf' column)
    """
    rtf_stats = df.groupby('system')['rtf'].agg(['mean', 'std', 'min', 'max'])
    
    print("\n" + "="*80)
    print("REAL-TIME FACTOR (RTF) STATISTICS")
    print("="*80)
    print("(RTF < 1.0 means faster than real-time)")
    print()
    print(rtf_stats)
    print()
    
    print("Processing Speed (relative to real-time):")
    print("-" * 60)
    for system in df['system'].unique():
        mean_rtf = float(rtf_stats.loc[system, 'mean'])
        speed = 1.0 / mean_rtf
        print(f"{system:25s}: {speed:.2f}x real-time speed")
    print("="*80)


def print_latency_statistics(df: pd.DataFrame) -> None:
    """
    Print latency statistics.
    
    Args:
        df: DataFrame with metrics
    """
    latency_stats = df.groupby('system')[['latency_mean_ms', 'latency_std_ms', 
                                          'peak_latency_ms']].mean()
    
    print("\n" + "="*80)
    print("LATENCY STATISTICS (milliseconds)")
    print("="*80)
    print(latency_stats)
    print("="*80)


# Buffer fill times (ms) by system family.
# DiArt steady-state = one step (500 ms); first output requires filling the
# full 5 s rolling window.  Sortformer always accumulates one chunk (1040 ms).
_BUFFER_FILL_MS = {
    'diart_default':        500.0,
    'diart_custom':         500.0,
    'streaming_sortformer': 1040.0,
}
_FIRST_FILL_MS = {
    'diart_default':        5000.0,
    'diart_custom':         5000.0,
    'streaming_sortformer': 1040.0,
}


def compute_e2e_latency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute end-to-end latency metrics per recording.

    Two E2E metrics are derived:
    - ``e2e_steady_ms``      = steady-state buffer fill + mean computational latency
    - ``e2e_first_output_ms``= first-output buffer fill + peak computational latency
                               (worst-case time until the very first result)

    Returns a copy of *df* with the two new columns added.
    """
    df = df.copy()

    df['buffer_fill_ms'] = df['system'].map(_BUFFER_FILL_MS).fillna(
        df['step_size_ms'].where(df['step_size_ms'] > 0, 1040.0)
    )
    df['first_fill_ms'] = df['system'].map(_FIRST_FILL_MS).fillna(
        df['buffer_fill_ms'] * 10  # fallback: assume 10× step for window
    )

    df['e2e_steady_ms']       = df['buffer_fill_ms'] + df['latency_mean_ms']
    df['e2e_first_output_ms'] = df['first_fill_ms']  + df['peak_latency_ms']

    return df


def print_e2e_latency_statistics(df: pd.DataFrame) -> None:
    """
    Print end-to-end latency statistics broken down by system.

    Covers three quantities:
    - Mean computational latency  (pure model forward-pass time)
    - Steady-state E2E latency    (buffer fill + mean computational)
    - First-output E2E latency    (window fill + peak computational)
    """
    df = compute_e2e_latency(df)

    stats = df.groupby('system').agg(
        comp_mean_ms=('latency_mean_ms',       'mean'),
        comp_std_ms= ('latency_std_ms',        'mean'),
        comp_peak_ms=('peak_latency_ms',       'mean'),
        e2e_steady_ms      =('e2e_steady_ms',       'mean'),
        e2e_first_output_ms=('e2e_first_output_ms', 'mean'),
        buffer_fill_ms     =('buffer_fill_ms',       'first'),
        first_fill_ms      =('first_fill_ms',         'first'),
    )

    print("\n" + "="*80)
    print("END-TO-END LATENCY STATISTICS (milliseconds)")
    print("="*80)
    print("Computational latency = wall-clock time for one model forward pass")
    print("Steady-state E2E      = buffer fill time + mean computational latency")
    print("First-output E2E      = window fill time + peak computational latency")
    print("-"*80)

    for system, row in stats.iterrows():
        print(f"\n{system}:")
        print(f"  Buffer fill (steady-state):   {row['buffer_fill_ms']:7.0f} ms")
        print(f"  Window fill (first output):   {row['first_fill_ms']:7.0f} ms")
        print(f"  Computational latency (mean): {row['comp_mean_ms']:7.1f} ms  "
              f"(±{row['comp_std_ms']:.1f} ms, peak {row['comp_peak_ms']:.1f} ms)")
        print(f"  ─────────────────────────────────────────────────────")
        print(f"  Steady-state E2E:             {row['e2e_steady_ms']:7.1f} ms")
        print(f"  First-output E2E:             {row['e2e_first_output_ms']:7.1f} ms")

    print("="*80)


def print_resource_statistics(df: pd.DataFrame) -> None:
    """
    Print resource usage statistics.
    
    Args:
        df: DataFrame with metrics
    """
    resource_stats = df.groupby('system')[['peak_gpu_mem_mb', 'avg_gpu_util_pct', 
                                            'peak_ram_mb', 'avg_cpu_percent']].mean()
    
    print("\n" + "="*80)
    print("RESOURCE USAGE STATISTICS")
    print("="*80)
    print(resource_stats)
    print("="*80)


def print_best_system_per_recording(der_pivot: pd.DataFrame) -> None:
    """
    Print which system performs best for each recording.
    
    Args:
        der_pivot: Pivot table of DER values (from plot_der_heatmap)
    """
    print("\n" + "="*80)
    print("BEST PERFORMING SYSTEM PER RECORDING")
    print("="*80)
    best_system = der_pivot.idxmin(axis=1)
    print(best_system.value_counts())
    print("="*80)

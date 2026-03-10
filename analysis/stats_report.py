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

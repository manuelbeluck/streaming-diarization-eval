"""
Visualization functions for diarization evaluation results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional


def plot_ground_truth_characteristics(gt_data: pd.DataFrame, figsize=(14, 5)) -> None:
    """
    Plot ground truth dataset characteristics.
    
    Args:
        gt_data: DataFrame with ground truth data (from prepare_ground_truth_data)
        figsize: Figure size as (width, height) tuple
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # 1. Number of speakers distribution
    speaker_counts = gt_data['gt_num_speakers'].value_counts().sort_index()
    axes[0].bar(speaker_counts.index, speaker_counts.values, 
                color='#3498db', edgecolor='black', linewidth=1.5, alpha=0.8)
    axes[0].set_xlabel('Number of Speakers', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Number of Recordings', fontsize=12, fontweight='bold')
    axes[0].set_title('Distribution of Number of Speakers', fontsize=14, fontweight='bold')
    axes[0].set_xticks(speaker_counts.index)
    axes[0].grid(axis='y', alpha=0.3)
    # Add value labels
    for x, y in zip(speaker_counts.index, speaker_counts.values):
        axes[0].text(x, y + 0.3, str(y), ha='center', va='bottom', 
                     fontsize=11, fontweight='bold')
    
    # 2. Overlap rate histogram
    axes[1].hist(gt_data['overlap_rate_pct'], bins=15, 
                 color='#e74c3c', edgecolor='black', linewidth=1.5, alpha=0.8)
    axes[1].axvline(gt_data['overlap_rate_pct'].mean(), color='darkred', 
                    linestyle='--', linewidth=2, 
                    label=f'Mean: {gt_data["overlap_rate_pct"].mean():.1f}%')
    axes[1].set_xlabel('Overlap Rate (%)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Number of Recordings', fontsize=12, fontweight='bold')
    axes[1].set_title('Distribution of Speaker Overlap Rate', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_der_components(df: pd.DataFrame, figsize=(16, 6)) -> None:
    """
    Plot DER components analysis with stacked and side-by-side views.
    
    Args:
        df: DataFrame with metrics
        figsize: Figure size as (width, height) tuple
    """
    # Calculate DER components as percentage of total speech time
    der_components = df.groupby('system')[['false_alarm', 'missed_detection', 'confusion']].mean()
    total_speech = df.groupby('system')['total_speech_time'].mean()
    der_components_pct = (der_components.div(total_speech, axis=0) * 100)
    
    # Also get the raw DER for comparison
    der_values = df.groupby('system')['DER'].mean() * 100  # Convert to percentage
    
    # Calculate Detection Error Rate (FA + Miss, excludes confusion)
    detection_error_rate = (der_components_pct['false_alarm'] + 
                           der_components_pct['missed_detection'])
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Stacked bar chart - Absolute contribution
    der_components_sorted = der_components_pct.loc[der_values.sort_values().index]
    der_components_sorted.plot(kind='barh', stacked=True, ax=axes[0],
                               color=['#e74c3c', '#f39c12', '#9b59b6'],
                               edgecolor='black', linewidth=1.5, alpha=0.85)
    axes[0].set_xlabel('Error Rate (% of speech time)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('System', fontsize=12, fontweight='bold')
    axes[0].set_title('DER Components - Stacked View', fontsize=14, fontweight='bold')
    axes[0].legend(['False Alarm', 'Missed Detection', 'Confusion'], 
                   loc='lower right', fontsize=10, framealpha=0.9)
    axes[0].grid(axis='x', alpha=0.3)
    
    # Add detection error rate markers (FA + Miss)
    for i, system in enumerate(der_values.sort_values().index):
        det_err = detection_error_rate.loc[system]
        axes[0].plot(det_err, i, marker='D', markersize=8, color='#2c3e50',
                    markeredgecolor='white', markeredgewidth=1.5, zorder=10)
    
    # Add total DER labels and detection error rate
    for i, (system, der) in enumerate(der_values.sort_values().items()):
        total_width = der_components_pct.loc[system].sum()
        det_err = detection_error_rate.loc[system]
        axes[0].text(total_width + 0.5, i, f'DER: {der:.2f}%', 
                    va='center', fontsize=10, fontweight='bold')
        axes[0].text(det_err, i - 0.3, f'Det: {det_err:.2f}%', 
                    va='center', ha='center', fontsize=8, 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                             edgecolor='#2c3e50', linewidth=1))
    
    # Grouped bar chart - Component comparison
    x = np.arange(len(der_components_pct))
    width = 0.20
    systems_sorted = der_values.sort_values().index
    
    axes[1].barh(x - 1.5*width, der_components_pct.loc[systems_sorted, 'false_alarm'], 
                width, label='False Alarm', color='#e74c3c', 
                edgecolor='black', linewidth=1.5, alpha=0.85)
    axes[1].barh(x - 0.5*width, der_components_pct.loc[systems_sorted, 'missed_detection'], 
                width, label='Missed Detection', color='#f39c12',
                edgecolor='black', linewidth=1.5, alpha=0.85)
    axes[1].barh(x + 0.5*width, der_components_pct.loc[systems_sorted, 'confusion'], 
                width, label='Confusion', color='#9b59b6',
                edgecolor='black', linewidth=1.5, alpha=0.85)
    axes[1].barh(x + 1.5*width, detection_error_rate.loc[systems_sorted], 
                width, label='Detection Error (FA+Miss)', color='#2c3e50',
                edgecolor='white', linewidth=1.5, alpha=0.85)
    
    axes[1].set_yticks(x)
    axes[1].set_yticklabels(systems_sorted)
    axes[1].set_xlabel('Error Rate (% of speech time)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('System', fontsize=12, fontweight='bold')
    axes[1].set_title('DER Components - Side-by-Side Comparison', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10, framealpha=0.9, loc='lower right')
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_der_heatmap(df: pd.DataFrame, sort_by_system: Optional[str] = None, 
                     figsize=(10, 8)) -> pd.DataFrame:
    """
    Plot DER heatmap by recording and system.
    
    Args:
        df: DataFrame with metrics
        sort_by_system: System name to sort recordings by (default: first system)
        figsize: Figure size as (width, height) tuple
        
    Returns:
        Pivot table of DER values
    """
    # Create pivot table for DER by recording
    der_pivot = df.pivot(index='recording_id', columns='system', values='DER')
    
    # Sort by specified system or first system
    if sort_by_system is None:
        sort_by_system = der_pivot.columns[0]
    der_pivot = der_pivot.sort_values(by=sort_by_system)
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(der_pivot, annot=True, fmt='.3f', cmap='RdYlGn_r', 
               cbar_kws={'label': 'DER'}, linewidths=0.5, ax=ax)
    ax.set_title('DER Heatmap by Recording and System', fontsize=14, fontweight='bold')
    ax.set_xlabel('System', fontsize=12)
    ax.set_ylabel('Recording ID', fontsize=12)
    plt.tight_layout()
    plt.show()
    
    return der_pivot


def plot_der_vs_overlap(df: pd.DataFrame, figsize=(12, 6)) -> None:
    """
    Scatter plot of DER vs speaker overlap rate.
    
    Args:
        df: DataFrame with metrics
        figsize: Figure size as (width, height) tuple
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for system in df['system'].unique():
        system_data = df[df['system'] == system]
        ax.scatter(system_data['gt_overlap_rate'] * 100, system_data['DER'], 
                  label=system, alpha=0.7, s=100, edgecolors='black', linewidth=1)
    
    ax.set_title('DER vs Speaker Overlap Rate', fontsize=14, fontweight='bold')
    ax.set_xlabel('Ground Truth Overlap Rate (%)', fontsize=12)
    ax.set_ylabel('Diarization Error Rate (DER)', fontsize=12)
    ax.legend(title='System', fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_processing_efficiency(df: pd.DataFrame, figsize=(12, 6)) -> None:
    """
    Plot processing efficiency metrics (RTF and processing time).
    
    Args:
        df: DataFrame with metrics
        figsize: Figure size as (width, height) tuple
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot: duration vs wall time
    for system in df['system'].unique():
        system_data = df[df['system'] == system]
        ax.scatter(system_data['duration'], system_data['wall_time_s'], 
                   label=system, alpha=0.6, s=100, edgecolors='black', linewidth=1)
    
    # Add y=x line (real-time line)
    max_duration = df['duration'].max()
    ax.plot([0, max_duration], [0, max_duration], 'r--', linewidth=2, 
            label='Real-time (1:1)')
    ax.set_title('Processing Time vs Audio Duration', fontsize=14, fontweight='bold')
    ax.set_xlabel('Audio Duration (s)', fontsize=12)
    ax.set_ylabel('Wall Time (s)', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_resource_usage(df: pd.DataFrame, figsize=(15, 10)) -> None:
    """
    Plot resource usage statistics (GPU/RAM/CPU).
    
    Args:
        df: DataFrame with metrics
        figsize: Figure size as (width, height) tuple
    """
    resource_stats = df.groupby('system')[['peak_gpu_mem_mb', 'avg_gpu_util_pct', 
                                            'peak_ram_mb', 'avg_cpu_percent']].mean()
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    colors = sns.color_palette('Set2', n_colors=len(resource_stats))
    
    # GPU Memory
    axes[0, 0].bar(range(len(resource_stats)), resource_stats['peak_gpu_mem_mb'],
                  color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
    axes[0, 0].set_xticks(range(len(resource_stats)))
    axes[0, 0].set_xticklabels(resource_stats.index, rotation=15, ha='right')
    axes[0, 0].set_title('Peak GPU Memory Usage', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Memory (MB)', fontsize=12)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # GPU Utilization
    axes[0, 1].bar(range(len(resource_stats)), resource_stats['avg_gpu_util_pct'],
                  color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
    axes[0, 1].set_xticks(range(len(resource_stats)))
    axes[0, 1].set_xticklabels(resource_stats.index, rotation=15, ha='right')
    axes[0, 1].set_title('Average GPU Utilization', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Utilization (%)', fontsize=12)
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # RAM Usage
    axes[1, 0].bar(range(len(resource_stats)), resource_stats['peak_ram_mb'],
                  color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
    axes[1, 0].set_xticks(range(len(resource_stats)))
    axes[1, 0].set_xticklabels(resource_stats.index, rotation=15, ha='right')
    axes[1, 0].set_title('Peak RAM Usage', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Memory (MB)', fontsize=12)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # CPU Usage
    axes[1, 1].bar(range(len(resource_stats)), resource_stats['avg_cpu_percent'],
                  color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
    axes[1, 1].set_xticks(range(len(resource_stats)))
    axes[1, 1].set_xticklabels(resource_stats.index, rotation=15, ha='right')
    axes[1, 1].set_title('Average CPU Usage', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('CPU (%)', fontsize=12)
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_latency_analysis(df: pd.DataFrame, figsize=(10, 5)) -> None:
    """
    Plot latency analysis (peak latency by system).
    
    Args:
        df: DataFrame with metrics
        figsize: Figure size as (width, height) tuple
    """
    latency_stats = df.groupby('system')[['latency_mean_ms', 'latency_std_ms', 
                                          'peak_latency_ms']].mean()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    systems = latency_stats.index
    x_pos = np.arange(len(systems))
    ax.bar(x_pos, latency_stats['peak_latency_ms'],
           color=sns.color_palette('Set2', n_colors=len(systems)),
           edgecolor='black', linewidth=1.5, alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(systems, rotation=15, ha='right')
    ax.set_title('Peak Latency', fontsize=14, fontweight='bold')
    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

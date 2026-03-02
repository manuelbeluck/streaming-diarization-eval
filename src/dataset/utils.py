"""Utility functions for dataset handling."""

import numpy as np

from .base import Segment


def load_audio(path: str, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """
    Load audio file and resample if needed.
    
    Args:
        path: Path to audio file
        target_sr: Target sample rate
        
    Returns:
        (audio, sample_rate) tuple
        Audio is normalized float32 mono, [-1, 1]
    """
    try:
        import soundfile as sf
    except ImportError:
        raise ImportError("soundfile required for audio loading. Install: pip install soundfile")
    
    audio, sr = sf.read(path, dtype='float32')
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    # Resample if needed
    if sr != target_sr:
        try:
            import scipy.signal
            # Calculate resampling ratio
            num_samples = int(len(audio) * target_sr / sr)
            audio = scipy.signal.resample(audio, num_samples)
            sr = target_sr
        except ImportError:
            raise ImportError("scipy required for resampling. Install: pip install scipy")
        
    if not isinstance(audio, np.ndarray):
        raise ValueError(f"Audio loading failed for {path}, Got type {type(audio)}")
    
    return audio, sr


def parse_rttm(path: str) -> list[Segment]:
    """
    Parse RTTM (Rich Transcription Time Marked) file.
    
    RTTM format:
    SPEAKER <recording-id> 1 <start> <duration> <NA> <NA> <speaker-id> <NA> <NA>
    
    Args:
        path: Path to RTTM file
        
    Returns:
        list of segments
    """
    segments = []
    
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) < 8 or parts[0] != 'SPEAKER':
                continue
            
            start = float(parts[3])
            duration = float(parts[4])
            speaker = parts[7]
            
            segments.append(Segment(
                start=start,
                end=start + duration,
                speaker=speaker
            ))
    
    # Sort by start time
    segments.sort(key=lambda s: s.start)
    return segments


def write_rttm(segments: list[Segment], path: str, recording_id: str = "recording"):
    """
    Write segments to RTTM format.
    
    Args:
        segments: list of segments to write
        path: Output path
        recording_id: Recording identifier for RTTM file
    
    Note:
        Timestamps are rounded to 10ms precision (0.01s) to match standard 
        diarization annotation conventions and ground truth precision.
    """
    with open(path, 'w') as f:
        for seg in segments:
            # Round to 10ms precision (0.01 second increments)
            start_rounded = round(seg.start, 2)
            end_rounded = round(seg.end, 2)
            duration = end_rounded - start_rounded
            
            # RTTM format: SPEAKER <file> 1 <start> <duration> <NA> <NA> <speaker> <NA> <NA>
            f.write(f"SPEAKER {recording_id} 1 {start_rounded:.3f} {duration:.3f} <NA> <NA> {seg.speaker} <NA> <NA>\n")

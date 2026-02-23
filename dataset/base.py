"""Abstract base class for dataset providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Iterator, Optional
import numpy as np


@dataclass
class Segment:
    """Represents a single speaker segment."""
    start: float  # Start time in seconds
    end: float    # End time in seconds
    speaker: str  # Speaker label


@dataclass
class Recording:
    """Metadata for a single recording."""
    recording_id: str
    duration: float  # Total duration in seconds
    sample_rate: int
    num_speakers: Optional[int] = None


class DatasetProvider(ABC):
    """Abstract interface for dataset access."""
    
    @abstractmethod
    def list_recordings(self) -> List[Recording]:
        """Return list of all available recordings."""
        pass
    
    @abstractmethod
    def get_audio(self, recording_id: str) -> np.ndarray:
        """
        Load audio for a recording.
        
        Returns:
            Audio as float32 numpy array, mono, normalized to [-1, 1]
        """
        pass
    
    @abstractmethod
    def get_ground_truth(self, recording_id: str) -> List[Segment]:
        """
        Load reference diarization (ground truth).
        
        Returns:
            List of segments with speaker labels
        """
        pass
    
    def chunk_audio(self, audio: np.ndarray, sample_rate: int, 
                    chunk_size: float, overlap: float = 0.0) -> Iterator[tuple[np.ndarray, float]]:
        """
        Split audio into chunks for streaming simulation.
        
        Args:
            audio: Audio array
            sample_rate: Sample rate in Hz
            chunk_size: Chunk duration in seconds
            overlap: Overlap between chunks in seconds
            
        Yields:
            (audio_chunk, timestamp) tuples
        """
        chunk_samples = int(chunk_size * sample_rate)
        step_samples = int((chunk_size - overlap) * sample_rate)
        
        for start in range(0, len(audio), step_samples):
            end = min(start + chunk_samples, len(audio))
            chunk = audio[start:end]
            timestamp = start / sample_rate
            
            yield chunk, timestamp
            
            if end >= len(audio):
                break

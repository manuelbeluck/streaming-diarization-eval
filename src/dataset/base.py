"""Abstract base class for dataset providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator
import numpy as np


@dataclass
class Segment:
    """Represents a single speaker segment."""
    start: float    # Start time in seconds
    end: float      # End time in seconds
    speaker: str    # Speaker label
    confidence: float | None = None  # Model confidence (None = unavailable)


@dataclass
class Recording:
    """Metadata for a single recording."""
    recording_id: str
    duration: float  # Total duration in seconds
    sample_rate: int


class DatasetProvider(ABC):
    """Abstract interface for dataset access."""
    
    @abstractmethod
    def list_recordings(self) -> list[Recording]:
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
    def get_ground_truth(self, recording_id: str) -> list[Segment]:
        """
        Load reference diarization (ground truth).
        
        Returns:
            list of segments with speaker labels
        """
        pass

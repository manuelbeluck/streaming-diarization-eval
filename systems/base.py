"""Abstract base class for streaming diarization systems."""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np

from dataset.base import Segment


class StreamingDiarizationSystem(ABC):
    """Abstract interface for streaming speaker diarization systems."""
    
    def __init__(self, name: str):
        """
        Initialize system.
        
        Args:
            name: System identifier
        """
        self.name = name
    
    @abstractmethod
    def run(self, audio: np.ndarray, sample_rate: int, num_speakers: Optional[int] = None) -> List[Segment]:
        """
        Run diarization on full audio. Systems handle their own chunking.
        
        Args:
            audio: Audio data (float32, mono)
            sample_rate: Audio sample rate in Hz
            num_speakers: Expected number of speakers (if known)
        
        Returns:
            Complete list of segments for the entire recording
        """
        pass

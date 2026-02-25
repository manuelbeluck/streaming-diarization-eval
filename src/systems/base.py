"""Abstract base class for streaming diarization systems."""

from abc import ABC, abstractmethod
import numpy as np

from src.dataset.base import Segment


class StreamingDiarizationSystem(ABC):
    """Abstract interface for streaming speaker diarization systems."""
    
    def __init__(self, name: str):
        """
        Initialize system.
        
        Args:
            name: System identifier
        """
        self.name = name
        self.latencies: list[float] = []  # Per-chunk latencies in seconds
        self.first_chunk_latency: float | None = None  # First chunk latency (often higher)
    
    def reset_latencies(self) -> None:
        """Reset latency measurements for a new recording."""
        self.latencies.clear()
        self.first_chunk_latency = None
    
    def get_latency_stats(self) -> dict[str, float]:
        """
        Get measured latency statistics.
        
        Returns:
            Dictionary with mean, std, first chunk latency in milliseconds,
            and number of chunks. Returns zero values if no latencies recorded.
        """
        if not self.latencies:
            return {
                'latency_mean_ms': 0.0,
                'latency_std_ms': 0.0,
                'latency_first_chunk_ms': 0.0,
                'num_chunks': 0
            }
        
        # Convert measured latencies (in seconds) to milliseconds
        latencies_ms = [lat * 1000 for lat in self.latencies]
        
        mean_lat = np.mean(latencies_ms)
        std_lat = np.std(latencies_ms) if len(latencies_ms) > 1 else 0.0
        first_chunk_ms = (self.first_chunk_latency * 1000) if self.first_chunk_latency is not None else 0.0
        
        return {
            'latency_mean_ms': float(mean_lat),
            'latency_std_ms': float(std_lat),
            'latency_first_chunk_ms': float(first_chunk_ms),
            'num_chunks': len(self.latencies)
        }
    
    @abstractmethod
    def run(self, audio: np.ndarray, sample_rate: int) -> list[Segment]:
        """
        Run diarization on full audio. Systems handle their own chunking.
        
        Args:
            audio: Audio data (float32, mono)
            sample_rate: Audio sample rate in Hz
        
        Returns:
            Complete list of segments for the entire recording
        """
        pass

"""Adapter for Sortformer streaming diarization system."""

from typing import List, Optional
import numpy as np

from systems.base import StreamingDiarizationSystem
from datasets.base import Segment


class SortformerSystem(StreamingDiarizationSystem):
    """Wrapper for Sortformer streaming diarization."""
    
    def __init__(self, chunk_size: float = 0.5):
        """
        Initialize Sortformer system.
        
        Args:
            chunk_size: Audio chunk duration in seconds
        """
        super().__init__(name="sortformer")
        self.chunk_size = chunk_size
        self.model = None
        self.sample_rate = None
        self.accumulated_segments = []
    
    def run(self, audio: np.ndarray, sample_rate: int, num_speakers: Optional[int] = None) -> List[Segment]:
        """Run Sortformer on full audio, handling chunking internally."""
        self.sample_rate = sample_rate
        self.accumulated_segments = []
        # TODO: Initialize Sortformer model here
        # Example (adjust based on actual Sortformer API):
        # from sortformer import StreamingSortformer
        # self.model = StreamingSortformer(
        #     checkpoint_path="path/to/checkpoint",
        #     chunk_size=self.chunk_size,
        #     sample_rate=sample_rate
        # )
        raise NotImplementedError(
            "Sortformer run() not yet implemented. "
            "Configure the model based on Sortformer's API."
        )

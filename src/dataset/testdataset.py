"""Test dataset provider for a single audio file and RTTM."""

import os
from pathlib import Path
import numpy as np
import urllib.request

from src.dataset.base import DatasetProvider, Recording, Segment
from src.dataset.utils import load_audio, parse_rttm

class TestDataset(DatasetProvider):
    """Dataset provider for a single test file (an4_diarize_test.wav)."""
    
    AUDIO_URL = "https://nemo-public.s3.us-east-2.amazonaws.com/an4_diarize_test.wav"
    RTTM_URL = "https://nemo-public.s3.us-east-2.amazonaws.com/an4_diarize_test.rttm"
    
    def __init__(self, data_dir: str = "data", max_duration: float | None = None):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)
        self.audio_file = self.data_dir / "an4_diarize_test.wav"
        self.rttm_file = self.data_dir / "an4_diarize_test.rttm"
        self.recording_id = "an4_diarize_test"
        self.max_duration = max_duration
        self._metadata_cache = None
        
        # Auto-download if files don't exist
        self._ensure_files_exist()

    def list_recordings(self) -> list[Recording]:
        if self._metadata_cache is None:
            audio, sample_rate = load_audio(str(self.audio_file))
            duration = len(audio) / sample_rate
            
            # Apply duration limit if specified
            if self.max_duration is not None and duration > self.max_duration:
                duration = self.max_duration
            
            self._metadata_cache = [Recording(
                recording_id=self.recording_id,
                duration=duration,
                sample_rate=sample_rate
            )]
        return self._metadata_cache

    def get_audio(self, recording_id: str) -> np.ndarray:
        assert recording_id == self.recording_id, f"Unknown recording: {recording_id}"
        audio, sample_rate = load_audio(str(self.audio_file))
        
        # Apply duration limit if specified
        if self.max_duration is not None:
            max_samples = int(self.max_duration * sample_rate)
            audio = audio[:max_samples]
        
        return audio

    def get_ground_truth(self, recording_id: str) -> list[Segment]:
        assert recording_id == self.recording_id, f"Unknown recording: {recording_id}"
        segments = parse_rttm(str(self.rttm_file))
        
        # Apply duration limit if specified
        if self.max_duration is not None:
            filtered_segments = []
            for seg in segments:
                # Skip segments that start after the duration limit
                if seg.start >= self.max_duration:
                    continue
                
                # Truncate segments that extend beyond the duration limit
                segment_end = min(seg.end, self.max_duration)
                
                filtered_segments.append(Segment(
                    start=seg.start,
                    end=segment_end,
                    speaker=seg.speaker
                ))
            segments = filtered_segments
        
        return segments
    
    def _ensure_files_exist(self):
        """Download test files if they don't exist (HuggingFace-style auto-download)."""
        if not self.audio_file.exists():
            print(f"Downloading test audio to {self.audio_file}...")
            urllib.request.urlretrieve(self.AUDIO_URL, self.audio_file)
            print(f"✓ Downloaded {self.audio_file.name}")
        
        if not self.rttm_file.exists():
            print(f"Downloading test RTTM to {self.rttm_file}...")
            urllib.request.urlretrieve(self.RTTM_URL, self.rttm_file)
            print(f"✓ Downloaded {self.rttm_file.name}")

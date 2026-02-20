"""Test dataset provider for a single audio file and RTTM."""

import os
from pathlib import Path
from typing import List, Optional
import numpy as np
import urllib.request

from datasets.base import DatasetProvider, Recording, Segment
from datasets.utils import load_audio, parse_rttm

class TestDataset(DatasetProvider):
    """Dataset provider for a single test file (an4_diarize_test.wav)."""
    
    AUDIO_URL = "https://nemo-public.s3.us-east-2.amazonaws.com/an4_diarize_test.wav"
    RTTM_URL = "https://nemo-public.s3.us-east-2.amazonaws.com/an4_diarize_test.rttm"
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)
        self.audio_file = self.data_dir / "an4_diarize_test.wav"
        self.rttm_file = self.data_dir / "an4_diarize_test.rttm"
        self.recording_id = "an4_diarize_test"
        self._metadata_cache = None
        
        # Auto-download if files don't exist
        self._ensure_files_exist()

    def list_recordings(self) -> List[Recording]:
        if self._metadata_cache is None:
            audio, sample_rate = load_audio(str(self.audio_file))
            duration = len(audio) / sample_rate
            segments = self.get_ground_truth(self.recording_id)
            num_speakers = len(set(seg.speaker for seg in segments)) if segments else None
            self._metadata_cache = [Recording(
                recording_id=self.recording_id,
                duration=duration,
                sample_rate=sample_rate,
                num_speakers=num_speakers
            )]
        return self._metadata_cache

    def get_audio(self, recording_id: str) -> np.ndarray:
        assert recording_id == self.recording_id, f"Unknown recording: {recording_id}"
        audio, _ = load_audio(str(self.audio_file))
        return audio

    def get_ground_truth(self, recording_id: str) -> List[Segment]:
        assert recording_id == self.recording_id, f"Unknown recording: {recording_id}"
        return parse_rttm(str(self.rttm_file))
    
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

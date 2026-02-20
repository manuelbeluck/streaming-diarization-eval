"""CallHome dataset provider using HuggingFace datasets."""

from pathlib import Path
from typing import List, Optional
import numpy as np

from datasets.base import DatasetProvider, Recording, Segment


class CallHomeDataset(DatasetProvider):
    """Provider for CallHome corpus from HuggingFace datasets."""
    
    def __init__(self, language: str = "eng", data_dir: str = "data/callhome", recordings: Optional[List[int]] = None):
        """
        Initialize CallHome dataset.
        
        Args:
            language: Language code ('eng', 'deu', etc.)
            data_dir: Cache directory for downloaded data
            recordings: Specific recording indices to use (None = all)
        """
        self.language = language
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._recordings_filter = recordings
        self._dataset = None
        self._metadata_cache = None
    
    def _load_dataset(self):
        """Load dataset from HuggingFace (lazy loading with caching)."""
        if self._dataset is None:
            try:
                from datasets import load_dataset
            except ImportError:
                raise ImportError(
                    "HuggingFace datasets required for CallHome. "
                    "Install: pip install datasets"
                )
            print(f"Loading CallHome ({self.language}) from HuggingFace...")
            self._dataset = load_dataset(
                "talkbank/callhome", 
                self.language, 
                split="data",
                cache_dir=str(self.data_dir)
            )
            print(f"Loaded {len(self._dataset)} recordings")
        return self._dataset
    
    def list_recordings(self) -> List[Recording]:
        """Return list of available CallHome recordings."""
        if self._metadata_cache is not None:
            return self._metadata_cache
        
        dataset = self._load_dataset()
        
        # Filter recordings if specified
        indices = self._recordings_filter if self._recordings_filter else range(len(dataset))
        
        recordings = []
        for idx in indices:
            if idx >= len(dataset):
                continue
            
            sample = dataset[idx]
            
            # Get duration from timestamps
            if sample["timestamps_end"]:
                duration = max(sample["timestamps_end"])
            else:
                duration = 0.0
            
            # Get number of unique speakers
            num_speakers = len(set(sample["speakers"])) if sample["speakers"] else None
            
            # Sample rate from audio
            sample_rate = sample["audio"]["sampling_rate"] if "audio" in sample else 16000
            
            recording_id = f"callhome_{self.language}_{idx:04d}"
            
            recordings.append(Recording(
                recording_id=recording_id,
                duration=duration,
                sample_rate=sample_rate,
                num_speakers=num_speakers
            ))
        
        self._metadata_cache = recordings
        return recordings
    
    def get_audio(self, recording_id: str) -> np.ndarray:
        """Load audio for a recording."""
        idx = self._parse_recording_id(recording_id)
        dataset = self._load_dataset()
        
        sample = dataset[idx]
        audio_data = sample["audio"]
        
        # Extract audio array
        audio = np.array(audio_data["array"], dtype=np.float32)
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        return audio
    
    def get_ground_truth(self, recording_id: str) -> List[Segment]:
        """Load reference annotations."""
        idx = self._parse_recording_id(recording_id)
        dataset = self._load_dataset()
        
        sample = dataset[idx]
        
        starts = sample["timestamps_start"]
        ends = sample["timestamps_end"]
        speakers = sample["speakers"]
        
        segments = []
        for start, end, speaker in zip(starts, ends, speakers):
            segments.append(Segment(
                start=float(start),
                end=float(end),
                speaker=str(speaker)
            ))
        
        # Sort by start time
        segments.sort(key=lambda s: s.start)
        return segments
    
    def _parse_recording_id(self, recording_id: str) -> int:
        """Extract index from recording ID."""
        try:
            # Format: callhome_eng_0000
            parts = recording_id.split("_")
            return int(parts[-1])
        except (ValueError, IndexError):
            raise ValueError(f"Invalid recording ID format: {recording_id}")

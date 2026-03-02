"""CallHome dataset provider using HuggingFace datasets."""

from pathlib import Path
import numpy as np

from src.dataset.base import DatasetProvider, Recording, Segment


class CallHomeDataset(DatasetProvider):
    """Provider for CallHome corpus from HuggingFace datasets."""
    
    def __init__(self, language: str = "eng", data_dir: str = "data/callhome",
                 recordings: list[int] | None = None, max_duration: float | None = None):
        """
        Initialize CallHome dataset.
        
        Args:
            language: Language code ('eng', 'deu', etc.)
            data_dir: Cache directory for downloaded data
            recordings: Specific recording indices to use (None = all)
            max_duration: Limit each recording to this many seconds (None = no limit)
        """
        self.language = language
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._recordings_filter = recordings
        self.max_duration = max_duration
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
                split="data"
            )
            print(f"Loaded {len(self._dataset)} recordings")
        return self._dataset
    
    def list_recordings(self) -> list[Recording]:
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
            
            # Apply duration limit if specified
            if self.max_duration is not None and duration > self.max_duration:
                duration = self.max_duration
            
            # HuggingFace datasets decodes CallHome audio at 16 kHz
            sample_rate = 16000
            
            recording_id = f"callhome_{self.language}_{idx:04d}"
            
            recordings.append(Recording(
                recording_id=recording_id,
                duration=duration,
                sample_rate=sample_rate
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
        
        # Apply duration limit if specified
        if self.max_duration is not None:
            sample_rate = audio_data["sampling_rate"]
            max_samples = int(self.max_duration * sample_rate)
            audio = audio[:max_samples]
        
        return audio
    
    def get_ground_truth(self, recording_id: str) -> list[Segment]:
        """Load reference annotations."""
        idx = self._parse_recording_id(recording_id)
        dataset = self._load_dataset()
        
        sample = dataset[idx]
        
        starts = sample["timestamps_start"]
        ends = sample["timestamps_end"]
        speakers = sample["speakers"]
        
        segments = []
        for start, end, speaker in zip(starts, ends, speakers):
            # Skip segments that start after the duration limit
            if self.max_duration is not None and start >= self.max_duration:
                continue
            
            # Truncate segments that extend beyond the duration limit
            segment_end = end
            if self.max_duration is not None and end > self.max_duration:
                segment_end = self.max_duration
            
            segments.append(Segment(
                start=float(start),
                end=float(segment_end),
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

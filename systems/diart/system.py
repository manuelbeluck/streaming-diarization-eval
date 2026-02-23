"""Adapter for DIART streaming diarization system."""

from typing import List, Optional
import numpy as np
from pathlib import Path
import tempfile

from systems.base import StreamingDiarizationSystem
from dataset.base import Segment
from dataset.utils import parse_rttm


class DiartSystem(StreamingDiarizationSystem):
    """Wrapper for DIART (https://github.com/juanmc2005/diart)."""
    
    def __init__(self, chunk_size: float = 0.5):
        """
        Initialize DIART system.
        
        Args:
            chunk_size: Audio chunk duration in seconds
        """
        super().__init__(name="diart")
        self.chunk_size = chunk_size
    
    def run(self, audio: np.ndarray, sample_rate: int, num_speakers: Optional[int] = None) -> List[Segment]:
        """Run DIART on full audio using DIART streaming inference."""
        try:
            from diart import SpeakerDiarization
            from diart.sources import FileAudioSource
            from diart.inference import StreamingInference
            from diart.sinks import RTTMWriter
        except ImportError as e:
            raise ImportError("diart is not installed. Please install it to use DiartSystem.") from e
        try:
            import soundfile as sf
        except ImportError as e:
            raise ImportError("soundfile is required for DIART audio input.") from e
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            audio_path = tmp_path / "input.wav"
            rttm_path = tmp_path / "output.rttm"
            sf.write(str(audio_path), audio.astype(np.float32), sample_rate)
            audio_source = FileAudioSource(
                file=str(audio_path),
                sample_rate=sample_rate,
                block_duration=self.chunk_size
            )
            pipeline = SpeakerDiarization()
            inference = StreamingInference(
                pipeline,
                audio_source,
                do_plot=False,
                show_progress=False
            )
            inference.attach_observers(RTTMWriter(audio_source.uri, str(rttm_path)))
            try:
                inference()
            except Exception as e:
                raise RuntimeError(f"Error during DIART processing: {e}")
            return parse_rttm(str(rttm_path))

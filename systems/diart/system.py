"""Adapter for DIART streaming diarization system."""

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
        
        # Import dependencies
        try:
            from diart import SpeakerDiarization
            from diart.sources import FileAudioSource
            from diart.inference import StreamingInference
            from diart.sinks import RTTMWriter
            
            self.SpeakerDiarization = SpeakerDiarization
            self.FileAudioSource = FileAudioSource
            self.StreamingInference = StreamingInference
            self.RTTMWriter = RTTMWriter
        except ImportError as e:
            raise ImportError("diart is not installed. Please install it to use DiartSystem.") from e
        
        try:
            import soundfile as sf
            self.sf = sf
        except ImportError as e:
            raise ImportError("soundfile is required for DIART audio input.") from e
        
        # Initialize pipeline once
        self.pipeline = self.SpeakerDiarization()
    
    def run(self, audio: np.ndarray, sample_rate: int) -> list[Segment]:
        """Run DIART on full audio using DIART streaming inference."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            audio_path = tmp_path / "input.wav"
            rttm_path = tmp_path / "output.rttm"
            
            # Write audio to temporary file
            self.sf.write(str(audio_path), audio.astype(np.float32), sample_rate)
            
            # Create audio source
            audio_source = self.FileAudioSource(
                file=str(audio_path),
                sample_rate=sample_rate,
                block_duration=self.chunk_size
            )
            
            # Create streaming inference
            inference = self.StreamingInference(
                self.pipeline,
                audio_source,
                do_plot=False,
                show_progress=False
            )
            
            # Attach RTTM writer
            inference.attach_observers(self.RTTMWriter(audio_source.uri, str(rttm_path)))
            
            # Run inference
            try:
                inference()
            except Exception as e:
                raise RuntimeError(f"Error during DIART processing: {e}")
            
            # Parse and return results
            return parse_rttm(str(rttm_path))

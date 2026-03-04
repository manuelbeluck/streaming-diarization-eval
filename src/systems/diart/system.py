"""Adapter for DIART streaming diarization system."""

import logging
import numpy as np
from pathlib import Path
import tempfile
import time

logger = logging.getLogger(__name__)

import torch
import soundfile as sf

from diart import models as diart_models, SpeakerDiarization
from diart import SpeakerDiarizationConfig
from diart.sources import FileAudioSource
from diart.inference import StreamingInference
from diart.sinks import RTTMWriter

from src.systems.base import StreamingDiarizationSystem
from src.dataset.base import Segment
from src.dataset.utils import parse_rttm


class PerfCounterChronometer:
    """Chronometer using perf_counter for high-precision latency measurement.
    
    Compatible with DIART's Chronometer interface but uses time.perf_counter()
    instead of time.monotonic() for higher precision timing.
    """
    
    def __init__(self, unit: str = "seconds", progress_bar=None):
        self.unit = unit
        self.progress_bar = progress_bar
        self.history: list[float] = []
        self.current_start_time: float | None = None
    
    @property
    def is_running(self) -> bool:
        """Check if chronometer is currently running."""
        return self.current_start_time is not None
    
    def start(self) -> None:
        """Start timing a chunk."""
        self.current_start_time = time.perf_counter()
    
    def stop(self, do_count: bool = True) -> None:
        """Stop timing and record the latency."""
        msg = "No start time available, Did you call stop() before start()?"
        assert self.current_start_time is not None, msg
        end_time = time.perf_counter() - self.current_start_time
        self.current_start_time = None
        if do_count:
            self.history.append(end_time)
    
    def report(self) -> None:
        """Report timing statistics."""
        import numpy as np
        print_fn = print
        if self.progress_bar is not None:
            print_fn = self.progress_bar.write
        print_fn(f"Took {np.mean(self.history).item():.3f} {self.unit} per chunk")


class DiartSystem(StreamingDiarizationSystem):
    """Wrapper for DIART (https://github.com/juanmc2005/diart)."""
    
    def __init__(
        self, 
        name: str = "diart",
        duration: float = 5.0,
        step: float = 0.5,
        segmentation_model: str = "pyannote/segmentation",
        embedding_model: str = "pyannote/embedding"
    ):
        """
        Initialize DIART system.
        
        Args:
            name: System identifier name
            duration: Processing window duration in seconds (default: 5.0)
            step: Step size between windows in seconds (default: 0.5)
            segmentation_model: Segmentation model name (e.g., "pyannote/segmentation-3.0")
            embedding_model: Embedding model name (e.g., "pyannote/wespeaker-voxceleb-resnet34-LM")
        """
        super().__init__(name=name)
        self.duration = duration
        self.step = step
        self.segmentation_model = segmentation_model
        self.embedding_model = embedding_model
        
        # Initialize pipeline with custom models and window parameters
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = SpeakerDiarizationConfig(
            segmentation=diart_models.SegmentationModel.from_pretrained(self.segmentation_model),
            embedding=diart_models.EmbeddingModel.from_pretrained(self.embedding_model),
            duration=duration,
            step=step,
            device=device,
        )
        self.pipeline = SpeakerDiarization(config)
        logger.info("DiartSystem running on %s", device)
    
    def run(self, audio: np.ndarray, sample_rate: int) -> list[Segment]:
        """Run DIART on full audio using DIART streaming inference."""
        # Reset latencies for this recording
        self.reset_latencies()
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            audio_path = tmp_path / "input.wav"
            rttm_path = tmp_path / "output.rttm"
            
            # Write audio to temporary file
            sf.write(str(audio_path), audio.astype(np.float32), sample_rate)
            
            # Create audio source
            audio_source = FileAudioSource(
                file=str(audio_path),
                sample_rate=sample_rate,
                block_duration=self.step
            )
            
            # Create streaming inference with profiling enabled
            inference = StreamingInference(
                self.pipeline,
                audio_source,
                do_profile=True,  # Enable profiling to get timing structure
                do_plot=False,
                show_progress=False
            )
            
            # Replace DIART's chronometer (monotonic) with our perf_counter version
            chrono = PerfCounterChronometer()
            #chrono = inference._chrono # type: ignore[assignment] test
            inference._chrono = chrono  # type: ignore[assignment]
            
            # Attach RTTM writer
            inference.attach_observers(RTTMWriter(audio_source.uri, str(rttm_path)))
            
            # TODO: Extract confidence scores from DIART
            
            # Run inference
            try:
                inference()
                
                # Copy latencies from our chronometer
                self.latencies = chrono.history.copy()
                if self.latencies:
                    self.first_chunk_latency = self.latencies[0]
            except Exception as e:
                raise RuntimeError(f"Error during DIART processing: {e}")
            
            # Parse and return results
            return parse_rttm(str(rttm_path))

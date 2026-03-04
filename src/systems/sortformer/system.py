"""Adapter for Sortformer streaming diarization system.
Adopted from WhisperLiveKit implementation. https://github.com/QuentinFuxa/WhisperLiveKit"""

import logging
import time
from dataclasses import dataclass

import numpy as np
import torch

from nemo.collections.asr.models import SortformerEncLabelModel
from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor

from src.systems.base import StreamingDiarizationSystem
from src.dataset.base import Segment

logger = logging.getLogger(__name__)

@dataclass
class Timed:
    start: float = 0.0
    end: float = 0.0


@dataclass
class SpeakerSegment(Timed):
    """Represents a segment of audio attributed to a specific speaker."""
    speaker: int = -1
    confidence: float = 0.0  # Mean model confidence (max-speaker probability) across frames in this segment


class StreamingSortformerState:
    """
    This class creates a class instance that will be used to store the state of the
    streaming Sortformer model.

    Attributes:
        spkcache (torch.Tensor): Speaker cache to store embeddings from start
        spkcache_lengths (torch.Tensor): Lengths of the speaker cache
        spkcache_preds (torch.Tensor): The speaker predictions for the speaker cache parts
        fifo (torch.Tensor): FIFO queue to save the embedding from the latest chunks
        fifo_lengths (torch.Tensor): Lengths of the FIFO queue
        fifo_preds (torch.Tensor): The speaker predictions for the FIFO queue parts
        spk_perm (torch.Tensor): Speaker permutation information for the speaker cache
        mean_sil_emb (torch.Tensor): Mean silence embedding
        n_sil_frames (torch.Tensor): Number of silence frames
    """

    def __init__(self):
        self.spkcache: torch.Tensor | None = None  # Speaker cache to store embeddings from start
        self.spkcache_lengths: torch.Tensor | None = None
        self.spkcache_preds: torch.Tensor | None = None  # speaker cache predictions
        self.fifo: torch.Tensor | None = None  # to save the embedding from the latest chunks
        self.fifo_lengths: torch.Tensor | None = None
        self.fifo_preds: torch.Tensor | None = None
        self.spk_perm: torch.Tensor | None = None
        self.mean_sil_emb: torch.Tensor | None = None
        self.n_sil_frames: torch.Tensor | None = None


class SortformerDiarization:
    def __init__(
        self, 
        model_name: str = "nvidia/diar_streaming_sortformer_4spk-v2",
        chunk_len: int = 10,
        subsampling_factor: int = 10,
        chunk_right_context: int = 0,
        chunk_left_context: int = 10,
        spkcache_len: int = 188,
        fifo_len: int = 188,
        spkcache_update_period: int = 144,
        log: bool = False
    ):
        """
        Stores the shared streaming Sortformer diarization model. Used when a new online_diarization is initialized.
        
        Args:
            model_name: Pre-trained model name
            chunk_len: Number of frames in a processing chunk
            subsampling_factor: Subsampling factor for processing
            chunk_right_context: Number of right context frames
            chunk_left_context: Number of left context frames
            spkcache_len: Speaker cache length
            fifo_len: FIFO buffer length
            spkcache_update_period: Speaker cache update period
            log: Enable logging
        """
        self.chunk_len = chunk_len
        self.subsampling_factor = subsampling_factor
        self.chunk_right_context = chunk_right_context
        self.chunk_left_context = chunk_left_context
        self.spkcache_len = spkcache_len
        self.fifo_len = fifo_len
        self.spkcache_update_period = spkcache_update_period
        self.log = log
        self._load_model(model_name)
    
    def _load_model(self, model_name: str):
        """Load and configure the Sortformer model for streaming."""
        try:
            self.diar_model: SortformerEncLabelModel = SortformerEncLabelModel.from_pretrained(model_name) # type: ignore
            self.diar_model.eval()

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.diar_model.to(device) # type: ignore
            
            logger.info(f"Using {device.type.upper()} for Sortformer model")

            intressting_params = ["chunk_len", "subsampling_factor", "chunk_right_context", "chunk_left_context", "spkcache_len", "fifo_len", "spkcache_update_period", "log"]

            logger.info(f"Default values")
            for param in intressting_params:
                value = getattr(self.diar_model.sortformer_modules, param, None) # type: ignore
                logger.info(f"  {param}: {value}")


            self.diar_model.sortformer_modules.chunk_len = self.chunk_len                   # type: ignore
            self.diar_model.sortformer_modules.subsampling_factor = self.subsampling_factor          # type: ignore
            self.diar_model.sortformer_modules.chunk_right_context = self.chunk_right_context          # type: ignore
            self.diar_model.sortformer_modules.chunk_left_context = self.chunk_left_context          # type: ignore
            self.diar_model.sortformer_modules.spkcache_len = self.spkcache_len               # type: ignore
            self.diar_model.sortformer_modules.fifo_len = self.fifo_len                   # type: ignore
            self.diar_model.sortformer_modules.spkcache_update_period = self.spkcache_update_period     # type: ignore
            self.diar_model.sortformer_modules.log = self.log                      # type: ignore
            self.diar_model.sortformer_modules._check_streaming_parameters()    # type: ignore
                        
        except Exception as e:
            logger.error(f"Failed to load Sortformer model: {e}")
            raise
 

class SortformerDiarizationOnline:
    def __init__(self, shared_model, sample_rate: int = 16000, latencies: list[float] | None = None):
        """
        Initialize the streaming Sortformer diarization system.
        
        Args:
            sample_rate: Audio sample rate (default: 16000)
            model_name: Pre-trained model name (default: "nvidia/diar_streaming_sortformer_4spk-v2")
            latencies: Optional list to store per-chunk latencies
        """
        self.sample_rate = sample_rate
        self.buffer_audio = np.array([], dtype=np.float32)
        self.global_time_offset = 0.0
        self.debug = False
        self.latencies = latencies if latencies is not None else []
        self.first_chunk_latency: float | None = None
        self._is_first_chunk = True
                
        self.diar_model: SortformerEncLabelModel = shared_model.diar_model
        
        self.audio2mel = AudioToMelSpectrogramPreprocessor(
            window_size=0.025,
            normalize="NA",
            n_fft=512,
            features=128,
            pad_to=0
        )
        self.audio2mel.to(self.diar_model.device)
        
        self.chunk_duration_seconds = (
            self.diar_model.sortformer_modules.chunk_len * # type: ignore
            self.diar_model.sortformer_modules.subsampling_factor *
            self.diar_model.preprocessor._cfg.window_stride
        )
        
        self._init_streaming_state()
        
        self._previous_chunk_features = None
        self._chunk_index = 0
        self._len_prediction = None
        self._speaker_active_state: np.ndarray | None = None  # (n_spk,) bool — onset/offset state carried across chunks
        
        # Audio buffer to store PCM chunks for debugging
        self.audio_buffer = []

        # Wall-clock time when the first diarize() call was made (for elapsed time reporting)
        self._diarize_start_wall_time: float | None = None
        
        logger.info("SortformerDiarization initialized successfully")


    def _init_streaming_state(self):
        """Initialize the streaming state for the model."""
        batch_size = 1
        device = self.diar_model.device

        self.streaming_state = StreamingSortformerState()
        self.streaming_state.spkcache = torch.zeros(
            (batch_size, self.diar_model.sortformer_modules.spkcache_len, self.diar_model.sortformer_modules.fc_d_model), # type: ignore
            device=device
        )
        self.streaming_state.spkcache_preds = torch.zeros(
            (batch_size, self.diar_model.sortformer_modules.spkcache_len, self.diar_model.sortformer_modules.n_spk), # type: ignore
            device=device
        ) 
        self.streaming_state.spkcache_lengths = torch.zeros((batch_size,), dtype=torch.long, device=device)
        self.streaming_state.fifo = torch.zeros(
            (batch_size, self.diar_model.sortformer_modules.fifo_len, self.diar_model.sortformer_modules.fc_d_model), # type: ignore
            device=device
        )
        self.streaming_state.fifo_lengths = torch.zeros((batch_size,), dtype=torch.long, device=device)
        self.streaming_state.mean_sil_emb = torch.zeros((batch_size, self.diar_model.sortformer_modules.fc_d_model), device=device) # type: ignore
        self.streaming_state.n_sil_frames = torch.zeros((batch_size,), dtype=torch.long, device=device)        
        self.total_preds = torch.zeros((batch_size, 0, self.diar_model.sortformer_modules.n_spk), device=device) # type: ignore


    def insert_audio_chunk(self, pcm_array: np.ndarray):
        if self.debug:
            self.audio_buffer.append(pcm_array.copy())
        self.buffer_audio = np.concatenate([self.buffer_audio, pcm_array.copy()])
  

    def diarize(self):
        """
        Process audio data for diarization in streaming fashion.
        """
        threshold = int(self.chunk_duration_seconds * self.sample_rate)
        
        if not len(self.buffer_audio) >= threshold:
            return []

        # Track wall-clock start of first chunk for elapsed-time reporting
        if self._diarize_start_wall_time is None:
            self._diarize_start_wall_time = time.perf_counter()
        
        # Start timing chunk processing
        chunk_start_time = time.perf_counter()
        
        audio = self.buffer_audio[:threshold]
        self.buffer_audio = self.buffer_audio[threshold:]
        
        device = self.diar_model.device
        audio_signal_chunk = torch.tensor(audio, device=device).unsqueeze(0)
        audio_signal_length_chunk = torch.tensor([audio_signal_chunk.shape[1]], device=device)
        
        processed_signal_chunk, processed_signal_length_chunk = self.audio2mel.get_features(
            audio_signal_chunk, audio_signal_length_chunk
        )
        processed_signal_chunk = processed_signal_chunk.to(device)
        processed_signal_length_chunk = processed_signal_length_chunk.to(device)
        
        if self._previous_chunk_features is not None:
            to_add = self._previous_chunk_features[:, :, -99:].to(device)
            total_features = torch.concat([to_add, processed_signal_chunk], dim=2).to(device)
        else:
            total_features = processed_signal_chunk.to(device)
        
        self._previous_chunk_features = processed_signal_chunk.to(device)
        
        chunk_feat_seq_t = torch.transpose(total_features, 1, 2).to(device)
        
        with torch.inference_mode():
            left_offset = 8 if self._chunk_index > 0 else 0
            right_offset = 8

            self.streaming_state, self.total_preds = self.diar_model.forward_streaming_step(
                processed_signal=chunk_feat_seq_t,
                processed_signal_length=torch.tensor([chunk_feat_seq_t.shape[1]]).to(device),
                streaming_state=self.streaming_state,
                total_preds=self.total_preds,
                left_offset=left_offset,
                right_offset=right_offset,
            )                
        new_segments = self._process_predictions()
        
        # Record chunk processing latency (time from start to end of processing)
        chunk_end_time = time.perf_counter()
        latency = chunk_end_time - chunk_start_time
        self.latencies.append(latency)
        
        # Track first chunk separately
        if self._is_first_chunk:
            self.first_chunk_latency = latency
            self._is_first_chunk = False

        self._chunk_index += 1
        return new_segments

    def _process_predictions(self, onset_threshold=0.5, offset_threshold=0.5):
        """Process model predictions and convert to speaker segments.
        
        Args:
            onset_threshold: Minimum confidence to start a new speaker segment
            offset_threshold: Minimum confidence to continue an existing speaker segment
        
        Note: This was initially taken from the WhisperLiveKit implementation,
            but adapted to work with overlaps and to compute mean confidence per segment.
        """
        # Use the first chunk's frame count as the fixed step size.
        if self._len_prediction is None:
            self._len_prediction = self.total_preds.shape[1]

        n = self._len_prediction
        n_spk = self.total_preds.shape[2]
        frame_duration = self.chunk_duration_seconds / n

        base_time = self._chunk_index * self.chunk_duration_seconds + self.global_time_offset
        chunk_end_time = base_time + self.chunk_duration_seconds

        # Slice on GPU before transferring to CPU — avoids copying the full accumulated tensor.
        chunk_preds_slice = self.total_preds[0, -n:, :].cpu().numpy()  # (n, n_spk)

        if self._speaker_active_state is None:
            self._speaker_active_state = np.zeros(n_spk, dtype=bool)

        # Get active / inactive state for each speaker per frame.
        chunk_active = np.zeros((n, n_spk), dtype=bool)
        speaker_state = self._speaker_active_state.copy()
        for s in range(n_spk):
            is_active = bool(speaker_state[s])
            for i, c in enumerate(chunk_preds_slice[:, s]):
                if not is_active:
                    if c >= onset_threshold:
                        is_active = True
                else:
                    if c < offset_threshold:
                        is_active = False
                chunk_active[i, s] = is_active
            speaker_state[s] = is_active # the speaker state of the last frame is stored for the next chunk

        # iterate over each speaker and create segments based on active/inactive state changes
        new_segments = []
        for s in range(n_spk):
            is_in_segment = False
            seg_start = base_time
            seg_start_idx = 0

            # Iterate over frames for this speaker and create segments,
            # when active state changes from True to False (segment end).
            for frame_idx, act in enumerate(chunk_active[:, s]):
                current_time = base_time + frame_idx * frame_duration
                if act and not is_in_segment:
                    seg_start = current_time
                    seg_start_idx = frame_idx
                    is_in_segment = True
                elif not act and is_in_segment:
                    # calculate mean confidence for this segment and create a SpeakerSegment
                    seg_conf = float(np.mean(chunk_preds_slice[seg_start_idx:frame_idx, s]))
                    new_segments.append(SpeakerSegment(
                        speaker=s, start=seg_start, end=current_time,
                        confidence=seg_conf
                    ))
                    is_in_segment = False
            # Handle case where segment is still active at the end of the chunk
            if is_in_segment:
                seg_conf = float(np.mean(chunk_preds_slice[seg_start_idx:, s]))
                new_segments.append(SpeakerSegment(
                    speaker=s, start=seg_start, end=chunk_end_time,
                    confidence=seg_conf
                ))

        new_segments.sort(key=lambda seg: seg.start)
        return new_segments
                
    def close(self):
        """Close the diarization system and clean up resources."""
        logger.info("Closing SortformerDiarization")


# Shared model instance to avoid reloading for each audio file
_shared_sortformer_model = None


class SortformerSystem(StreamingDiarizationSystem):
    """Wrapper for Sortformer streaming diarization."""
    
    def __init__(
        self, 
        model_name: str = "nvidia/diar_streaming_sortformer_4spk-v2",
        chunk_len: int = 10,
        subsampling_factor: int = 10,
        chunk_right_context: int = 0,
        chunk_left_context: int = 10,
        spkcache_len: int = 188,
        fifo_len: int = 188,
        spkcache_update_period: int = 144,
        log: bool = False,
        chunk_size: float | None = None,
    ):
        """
        Initialize Sortformer system.
        
        Args:
            model_name: Pre-trained model name
            chunk_len: Number of frames in a processing chunk (affects internal chunk duration)
            subsampling_factor: Subsampling factor (affects internal chunk duration)
            chunk_right_context: Number of right context frames
            chunk_left_context: Number of left context frames
            spkcache_len: Speaker cache length
            fifo_len: FIFO buffer length
            spkcache_update_period: Speaker cache update period
            log: Enable logging
            chunk_size: Optional audio input chunk size in seconds. If None, uses model's natural chunk duration.
        """
        super().__init__(name="streaming_sortformer")
        self.model_name = model_name
        self.chunk_len = chunk_len
        self.subsampling_factor = subsampling_factor
        self.chunk_right_context = chunk_right_context
        self.chunk_left_context = chunk_left_context
        self.spkcache_len = spkcache_len
        self.fifo_len = fifo_len
        self.spkcache_update_period = spkcache_update_period
        self.log = log
        self.chunk_size = chunk_size  # Can be None, will be set after model is loaded
        
        # Initialize shared model on first use
        global _shared_sortformer_model
        if _shared_sortformer_model is None:
            logger.info(f"Loading Sortformer model: {model_name}")
            _shared_sortformer_model = SortformerDiarization(
                model_name=model_name,
                chunk_len=chunk_len,
                subsampling_factor=subsampling_factor,
                chunk_right_context=chunk_right_context,
                chunk_left_context=chunk_left_context,
                spkcache_len=spkcache_len,
                fifo_len=fifo_len,
                spkcache_update_period=spkcache_update_period,
                log=log
            )
        
        self.shared_model = _shared_sortformer_model
    
    def run(self, audio: np.ndarray, sample_rate: int) -> list[Segment]:
        """Run Sortformer on full audio, handling chunking internally."""
        # Reset latencies for this recording
        self.reset_latencies()
        
        logger.info(f"Processing audio of length {len(audio)/sample_rate:.2f}s at {sample_rate}Hz")
        
        # Initialize online diarization instance with latency tracking
        diarization = SortformerDiarizationOnline(
            shared_model=self.shared_model,
            sample_rate=sample_rate,
            latencies=self.latencies,
        )
        
        # Use model's natural chunk duration if chunk_size not specified
        if self.chunk_size is None:
            input_chunk_size = diarization.chunk_duration_seconds
            logger.info(f"Using model's natural chunk duration: {input_chunk_size:.3f}s")
        else:
            input_chunk_size = self.chunk_size
        
        # Process audio in chunks
        chunk_samples = int(input_chunk_size * sample_rate)
        all_segments = []
        
        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i:i+chunk_samples]
            diarization.insert_audio_chunk(chunk)
            new_segments = diarization.diarize()
            all_segments.extend(new_segments)
        
        # Merge consecutive segments with same speaker.
        # Per-speaker merge: group by speaker ID, merge overlapping/adjacent
        # segments within each speaker independently.
        merged_segments = []
        if all_segments:
            from collections import defaultdict
            by_speaker: dict[int, list[SpeakerSegment]] = defaultdict(list)
            for seg in all_segments:
                by_speaker[seg.speaker].append(seg)  # type: ignore[arg-type]

            for spk, segs in by_speaker.items():
                segs.sort(key=lambda s: s.start)
                cur = segs[0]
                for seg in segs[1:]:
                    if seg.start <= cur.end:
                        merged_conf = (cur.confidence + seg.confidence) / 2
                        cur = SpeakerSegment(speaker=cur.speaker, start=cur.start,
                                             end=max(cur.end, seg.end),
                                             confidence=merged_conf)
                    else:
                        merged_segments.append(cur)
                        cur = seg
                merged_segments.append(cur)

            merged_segments.sort(key=lambda s: s.start)
        
        # Convert to Segment objects with string speaker labels
        result_segments = [
            Segment(
                start=seg.start,
                end=seg.end,
                speaker=f"speaker_{seg.speaker}",
                confidence=seg.confidence
            )
            for seg in merged_segments
        ]
        
        logger.info(f"Generated {len(result_segments)} speaker segments")
        
        # Copy first chunk latency from diarization instance
        if diarization.first_chunk_latency is not None:
            self.first_chunk_latency = diarization.first_chunk_latency
        
        diarization.close()
        
        return result_segments

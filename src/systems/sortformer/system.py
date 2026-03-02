"""Adapter for Sortformer streaming diarization system.
Adopted from WhisperLiveKit implementation. https://github.com/QuentinFuxa/WhisperLiveKit"""

import logging
import threading
import time
from dataclasses import dataclass

import numpy as np
import torch

from src.systems.base import StreamingDiarizationSystem
from src.dataset.base import Segment

logger = logging.getLogger(__name__)

# Lazy import - only fail when actually trying to use Sortformer
_nemo_import_error = None
try:
    from nemo.collections.asr.models import SortformerEncLabelModel
    from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor
    
except ImportError as e:
    _nemo_import_error = e
    SortformerEncLabelModel = None  # type: ignore
    AudioToMelSpectrogramPreprocessor = None  # type: ignore


@dataclass
class Timed:
    start: float | None = 0
    end: float | None = 0


@dataclass
class SpeakerSegment(Timed):
    """Represents a segment of audio attributed to a specific speaker.
    No text nor probability is associated with this segment.
    """
    speaker: int | None = -1
    pass


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
        if _nemo_import_error is not None:
            raise ImportError(
                'NeMo is required for Sortformer. '
                'Please install it with: pip install "git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]"'
            ) from _nemo_import_error
        
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
            if SortformerEncLabelModel is None:
                raise ImportError("SortformerEncLabelModel is not available. NeMo may not be installed correctly.")
            self.diar_model = SortformerEncLabelModel.from_pretrained(model_name)
            self.diar_model.eval() # type: ignore

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
    def __init__(self, shared_model, sample_rate: int = 16000, latencies: list[float] | None = None,
                 overlap_aware: bool = False):
        """
        Initialize the streaming Sortformer diarization system.
        
        Args:
            sample_rate: Audio sample rate (default: 16000)
            model_name: Pre-trained model name (default: "nvidia/diar_streaming_sortformer_4spk-v2")
            latencies: Optional list to store per-chunk latencies
            overlap_aware: If True, emit overlapping segments when multiple speakers exceed threshold
        """
        self.sample_rate = sample_rate
        self.diarization_segments = []
        self.diar_segments = []
        self.buffer_audio = np.array([], dtype=np.float32)
        self.segment_lock = threading.Lock()
        self.global_time_offset = 0.0
        self.debug = False
        self.latencies = latencies if latencies is not None else []
        self.first_chunk_latency: float | None = None
        self._is_first_chunk = True
        self.overlap_aware = overlap_aware
                
        self.diar_model = shared_model.diar_model
        
        if AudioToMelSpectrogramPreprocessor is None:
            raise ImportError("AudioToMelSpectrogramPreprocessor is not available. NeMo may not be installed correctly.")

        self.audio2mel = AudioToMelSpectrogramPreprocessor(
            window_size=0.025,
            normalize="NA",
            n_fft=512,
            features=128,
            pad_to=0
        )
        self.audio2mel.to(self.diar_model.device)
        
        self.chunk_duration_seconds = (
            self.diar_model.sortformer_modules.chunk_len * 
            self.diar_model.sortformer_modules.subsampling_factor * 
            self.diar_model.preprocessor._cfg.window_stride
        )
        
        self._init_streaming_state()
        
        self._previous_chunk_features = None
        self._chunk_index = 0
        self._len_prediction = None
        
        # Audio buffer to store PCM chunks for debugging
        self.audio_buffer = []
        
        # Buffer for accumulating audio chunks until reaching chunk_duration_seconds
        self.audio_chunk_buffer = []
        self.accumulated_duration = 0.0
        
        logger.info("SortformerDiarization initialized successfully")


    def _init_streaming_state(self):
        """Initialize the streaming state for the model."""
        batch_size = 1
        device = self.diar_model.device
        
        self.streaming_state = StreamingSortformerState()
        self.streaming_state.spkcache = torch.zeros(
            (batch_size, self.diar_model.sortformer_modules.spkcache_len, self.diar_model.sortformer_modules.fc_d_model), 
            device=device
        )
        self.streaming_state.spkcache_preds = torch.zeros(
            (batch_size, self.diar_model.sortformer_modules.spkcache_len, self.diar_model.sortformer_modules.n_spk), 
            device=device
        )
        self.streaming_state.spkcache_lengths = torch.zeros((batch_size,), dtype=torch.long, device=device)
        self.streaming_state.fifo = torch.zeros(
            (batch_size, self.diar_model.sortformer_modules.fifo_len, self.diar_model.sortformer_modules.fc_d_model), 
            device=device
        )
        self.streaming_state.fifo_lengths = torch.zeros((batch_size,), dtype=torch.long, device=device)
        self.streaming_state.mean_sil_emb = torch.zeros((batch_size, self.diar_model.sortformer_modules.fc_d_model), device=device)
        self.streaming_state.n_sil_frames = torch.zeros((batch_size,), dtype=torch.long, device=device)        
        self.total_preds = torch.zeros((batch_size, 0, self.diar_model.sortformer_modules.n_spk), device=device)

    def insert_silence(self, silence_duration: float):
        """
        Insert silence period by adjusting the global time offset.
        
        Args:
            silence_duration: Duration of silence in seconds
        """
        with self.segment_lock:
            self.global_time_offset += silence_duration
        logger.debug(f"Inserted silence of {silence_duration:.2f}s, new offset: {self.global_time_offset:.2f}s")

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
        """
        preds_np = self.total_preds[0].cpu().numpy()  # (T_total, n_spk)

        if self._len_prediction is None:
            self._len_prediction = preds_np.shape[0]

        n = self._len_prediction
        frame_duration = self.chunk_duration_seconds / n

        with self.segment_lock:
            base_time = self._chunk_index * self.chunk_duration_seconds + self.global_time_offset
            chunk_end_time = base_time + self.chunk_duration_seconds

            if not self.overlap_aware:
                # ── single-speaker per frame (argmax) ────────────────────────
                active_speakers = np.argmax(preds_np, axis=1)
                max_confidences = np.max(preds_np, axis=1)

                current_speaker = -1
                for i in range(len(active_speakers)):
                    if current_speaker == -1:
                        if max_confidences[i] >= onset_threshold:
                            current_speaker = active_speakers[i]
                        else:
                            active_speakers[i] = -1
                    else:
                        if max_confidences[i] >= offset_threshold and active_speakers[i] == current_speaker:
                            pass
                        elif max_confidences[i] >= onset_threshold:
                            current_speaker = active_speakers[i]
                        else:
                            active_speakers[i] = -1
                            current_speaker = -1

                current_chunk_preds = active_speakers[-n:]
                new_segments = []
                current_spk = current_chunk_preds[0]
                start_time = base_time

                for idx, spk in enumerate(current_chunk_preds):
                    current_time = base_time + idx * frame_duration
                    if spk != current_spk:
                        if current_spk != -1:
                            new_segments.append(SpeakerSegment(
                                speaker=current_spk, start=start_time, end=current_time
                            ))
                        start_time = current_time
                        current_spk = spk

                if current_spk != -1:
                    new_segments.append(SpeakerSegment(
                        speaker=current_spk, start=start_time, end=chunk_end_time
                    ))
                return new_segments

            else:
                # ── overlap-aware: independent onset/offset per speaker ───────
                # Each speaker's activity is tracked independently; multiple
                # speakers can be active simultaneously → overlapping segments.
                n_spk = preds_np.shape[1]
                speaker_active = np.zeros((preds_np.shape[0], n_spk), dtype=bool)
                for s in range(n_spk):
                    conf = preds_np[:, s]
                    active = False
                    for i, c in enumerate(conf):
                        if not active:
                            if c >= onset_threshold:
                                active = True
                        else:
                            if c < offset_threshold:
                                active = False
                        speaker_active[i, s] = active

                # Only emit segments for the current chunk slice
                chunk_active = speaker_active[-n:, :]  # (n, n_spk)
                new_segments = []

                for s in range(n_spk):
                    in_seg = False
                    seg_start = base_time
                    for idx, act in enumerate(chunk_active[:, s]):
                        current_time = base_time + idx * frame_duration
                        if act and not in_seg:
                            seg_start = current_time
                            in_seg = True
                        elif not act and in_seg:
                            new_segments.append(SpeakerSegment(
                                speaker=s, start=seg_start, end=current_time
                            ))
                            in_seg = False
                    if in_seg:
                        new_segments.append(SpeakerSegment(
                            speaker=s, start=seg_start, end=chunk_end_time
                        ))

                new_segments.sort(key=lambda seg: seg.start)
                return new_segments
                
    def get_segments(self) -> list[SpeakerSegment]:
        """Get a copy of the current speaker segments."""
        with self.segment_lock:
            return self.diarization_segments.copy()

    def close(self):
        """Close the diarization system and clean up resources."""
        logger.info("Closing SortformerDiarization")
        with self.segment_lock:
            self.diarization_segments.clear()


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
        overlap_aware: bool = False
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
            overlap_aware: If True, each speaker's activity is tracked independently and overlapping
                           segments are emitted when multiple speakers exceed the threshold simultaneously.
                           If False (default), only the highest-confidence speaker wins each frame.
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
        self.overlap_aware = overlap_aware
        
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
            overlap_aware=self.overlap_aware
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
        
        # Merge consecutive segments with same speaker
        merged_segments = []
        if all_segments:
            if self.overlap_aware:
                # Per-speaker merge: group by speaker ID, merge within each
                # speaker independently (handles overlapping output correctly)
                from collections import defaultdict
                by_speaker: dict[int, list[SpeakerSegment]] = defaultdict(list)
                for seg in all_segments:
                    by_speaker[seg.speaker].append(seg)  # type: ignore[arg-type]

                for spk, segs in by_speaker.items():
                    segs.sort(key=lambda s: s.start)
                    cur = segs[0]
                    for seg in segs[1:]:
                        if seg.start <= cur.end:
                            cur = SpeakerSegment(speaker=cur.speaker, start=cur.start,
                                                 end=max(cur.end, seg.end))
                        else:
                            merged_segments.append(cur)
                            cur = seg
                    merged_segments.append(cur)

                merged_segments.sort(key=lambda s: s.start)
            else:
                current_speaker = all_segments[0].speaker
                current_start = all_segments[0].start
                current_end = all_segments[0].end

                for segment in all_segments[1:]:
                    if segment.speaker == current_speaker:
                        current_end = segment.end
                    else:
                        merged_segments.append(SpeakerSegment(
                            speaker=current_speaker,
                            start=current_start,
                            end=current_end
                        ))
                        current_speaker = segment.speaker
                        current_start = segment.start
                        current_end = segment.end

                merged_segments.append(SpeakerSegment(
                    speaker=current_speaker,
                    start=current_start,
                    end=current_end
                ))
        
        # Convert to Segment objects with string speaker labels
        result_segments = [
            Segment(
                start=seg.start,
                end=seg.end,
                speaker=f"speaker_{seg.speaker}"
            )
            for seg in merged_segments
        ]
        
        logger.info(f"Generated {len(result_segments)} speaker segments")
        
        # Copy first chunk latency from diarization instance
        if diarization.first_chunk_latency is not None:
            self.first_chunk_latency = diarization.first_chunk_latency
        
        diarization.close()
        
        return result_segments

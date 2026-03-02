"""
Offline diarization with pyannote.audio pipeline.

Runs pyannote/speaker-diarization-3.1 on one (or more) CallHome recordings,
prints the predicted RTTM to stdout, and reports DER / JER.

Requirements:
    pip install pyannote.audio
    A valid HuggingFace token with access to pyannote gated models must be set via
    the HF_TOKEN environment variable or stored in ~/.huggingface/token.
"""

import os
# Force PyTorch to disable weights_only=True for model loading compatibility.
# https://github.com/m-bain/whisperX/issues/1304
# Only official models are used
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
import sys
from io import StringIO
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.dataset.callhome import CallHomeDataset
from src.dataset.base import Segment
from src.dataset.utils import write_rttm
from src.evaluator.metrics import compute_der, compute_jer


# ── helpers ──────────────────────────────────────────────────────────────────

def annotation_to_segments(annotation) -> list[Segment]:
    """Convert a pyannote Annotation to our internal Segment list."""
    segments: list[Segment] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append(Segment(start=turn.start, end=turn.end, speaker=speaker))
    segments.sort(key=lambda s: s.start)
    return segments


def segments_to_rttm_str(segments: list[Segment], recording_id: str) -> str:
    """Render segments as an RTTM string (without writing to disk)."""
    buf = StringIO()
    for seg in segments:
        duration = seg.end - seg.start
        buf.write(
            f"SPEAKER {recording_id} 1 {seg.start:.3f} {duration:.3f}"
            f" <NA> <NA> {seg.speaker} <NA> <NA>\n"
        )
    return buf.getvalue()


# ── main logic ────────────────────────────────────────────────────────────────

def run(recordings: list[int], collar: float, save_dir: Path | None) -> None:
    # ── load pipeline ────────────────────────────────────────────────────────
    hf_token = os.environ.get("HF_TOKEN")

    print("Loading pyannote speaker-diarization-3.1 pipeline …")
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")
    pipeline = pipeline.to(torch.device(device))

    # ── load dataset ─────────────────────────────────────────────────────────
    print("Loading CallHome (eng) …")
    dataset = CallHomeDataset(
        language="eng",
        data_dir="data/callhome",
        recordings=recordings,
    )

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    all_ders: list[float] = []
    all_jers: list[float] = []

    for recording in dataset.list_recordings():
        rid = recording.recording_id
        print(f"\n{'─'*60}")
        print(f"Recording: {rid}  ({recording.duration:.1f}s)")
        print(f"{'─'*60}")

        # Load audio as waveform tensor expected by pyannote
        audio_np = dataset.get_audio(rid)
        sr = recording.sample_rate  # actual SR read from the HF dataset (e.g. 16 kHz)
        # pyannote pipeline expects {"waveform": (1, T) float32, "sample_rate": int}
        waveform = torch.from_numpy(audio_np).unsqueeze(0)  # (1, T)
        audio_input = {"waveform": waveform, "sample_rate": sr}

        ground_truth = dataset.get_ground_truth(rid)

        # ── run diarization ──────────────────────────────────────────────────
        diarization = pipeline(audio_input)

        # ── convert to our format ────────────────────────────────────────────
        prediction = annotation_to_segments(diarization)
        

        # ── print RTTM ───────────────────────────────────────────────────────
        print("\n── Predicted RTTM ──")
        rttm_str = segments_to_rttm_str(prediction, rid)
        print(rttm_str.rstrip())

        # ── save RTTMs to disk ───────────────────────────────────────────────
        if save_dir:
            pred_path = save_dir / f"{rid}_pred.rttm"
            gt_path   = save_dir / f"{rid}_gt.rttm"
            write_rttm(prediction,   str(pred_path), rid)
            write_rttm(ground_truth, str(gt_path),   rid)
            print(f"\nSaved → {pred_path}")
            print(f"Saved → {gt_path}")

        # ── metrics ──────────────────────────────────────────────────────────
        der_result = compute_der(prediction, ground_truth, collar=collar)
        jer        = compute_jer(prediction, ground_truth, collar=collar)

        print(f"\n── Metrics (collar={collar}s) ──")
        print(f"  DER  : {der_result['DER']:.3f}")
        print(f"    False Alarm      : {der_result['false_alarm']:.3f}")
        print(f"    Missed Detection : {der_result['missed_detection']:.3f}")
        print(f"    Confusion        : {der_result['confusion']:.3f}")
        print(f"  JER  : {jer:.3f}")

        all_ders.append(der_result['DER'])
        all_jers.append(jer)

    # ── aggregate summary ─────────────────────────────────────────────────────
    if len(all_ders) > 1:
        print(f"\n{'='*60}")
        print("AGGREGATE SUMMARY")
        print(f"{'='*60}")
        print(f"  Recordings : {len(all_ders)}")
        print(f"  Avg DER    : {sum(all_ders)/len(all_ders):.3f}")
        print(f"  Avg JER    : {sum(all_jers)/len(all_jers):.3f}")


# ── Configuration ────────────────────────────────────────────────────────────

RECORDINGS = [0]          # CallHome recording indices to evaluate
COLLAR     = 0.0          # DER/JER forgiveness collar in seconds
SAVE_DIR   = "results/pyannote_offline"         # Set to a path string to save RTTMs, e.g. "results/pyannote_offline"

# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    save_dir = Path(SAVE_DIR) if SAVE_DIR else None
    run(recordings=RECORDINGS, collar=COLLAR, save_dir=save_dir)

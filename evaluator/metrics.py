"""Diarization evaluation metrics: DER and JER."""

from typing import List, Dict, Tuple
import numpy as np

from datasets.base import Segment


def compute_der(hypothesis: List[Segment], reference: List[Segment], 
                collar: float = 0.25) -> Dict[str, float]:
    """
    Compute Diarization Error Rate (DER).
    
    DER = (False Alarm + Missed Speech + Speaker Error) / Total Speech Time
    
    Args:
        hypothesis: Predicted segments
        reference: Ground truth segments
        collar: Forgiveness collar in seconds (default 0.25s)
        
    Returns:
        Dictionary with DER components and total
    """
    try:
        from pyannote.core import Annotation, Segment as PySegment
        from pyannote.metrics.diarization import DiarizationErrorRate
    except ImportError:
        raise ImportError(
            "pyannote.metrics required for DER calculation. "
            "Install: pip install pyannote.metrics"
        )
    
    # Convert to pyannote format
    ref_annotation = Annotation()
    for seg in reference:
        ref_annotation[PySegment(seg.start, seg.end)] = seg.speaker
    
    hyp_annotation = Annotation()
    for seg in hypothesis:
        hyp_annotation[PySegment(seg.start, seg.end)] = seg.speaker
    
    # Compute DER
    metric = DiarizationErrorRate(collar=collar)
    der = metric(ref_annotation, hyp_annotation)
    
    # Get detailed components
    components = metric.compute_components(ref_annotation, hyp_annotation)
    
    return {
        'DER': der,
        'false_alarm': components.get('false alarm', 0.0),
        'missed_detection': components.get('missed detection', 0.0),
        'confusion': components.get('confusion', 0.0),
        'total': components.get('total', 0.0)
    }


def compute_jer(hypothesis: List[Segment], reference: List[Segment], 
                collar: float = 0.25) -> float:
    """
    Compute Jaccard Error Rate (JER).
    
    JER measures the overlap between hypothesis and reference.
    
    Args:
        hypothesis: Predicted segments
        reference: Ground truth segments
        collar: Forgiveness collar in seconds
        
    Returns:
        JER value (0 = perfect, 1 = completely wrong)
    """
    try:
        from pyannote.core import Annotation, Segment as PySegment
        from pyannote.metrics.diarization import JaccardErrorRate
    except ImportError:
        raise ImportError(
            "pyannote.metrics required for JER calculation. "
            "Install: pip install pyannote.metrics"
        )
    
    # Convert to pyannote format
    ref_annotation = Annotation()
    for seg in reference:
        ref_annotation[PySegment(seg.start, seg.end)] = seg.speaker
    
    hyp_annotation = Annotation()
    for seg in hypothesis:
        hyp_annotation[PySegment(seg.start, seg.end)] = seg.speaker
    
    # Compute JER
    metric = JaccardErrorRate(collar=collar)
    jer = metric(ref_annotation, hyp_annotation)
    
    return jer


def simple_der(hypothesis: List[Segment], reference: List[Segment],
               step: float = 0.01) -> float:
    """
    Simple DER implementation without pyannote dependency.
    
    Uses frame-based calculation with given time step.
    Less accurate than pyannote but useful for quick checks.
    
    Args:
        hypothesis: Predicted segments
        reference: Ground truth segments
        step: Time step for frame-based calculation
        
    Returns:
        DER value
    """
    if not reference:
        return 1.0 if hypothesis else 0.0
    
    # Find time range
    all_segments = reference + hypothesis
    start_time = min(seg.start for seg in all_segments)
    end_time = max(seg.end for seg in all_segments)
    
    # Create frame-based representation
    num_frames = int((end_time - start_time) / step) + 1
    ref_frames = {}  # frame_idx -> set of speakers
    hyp_frames = {}  # frame_idx -> set of speakers
    
    for seg in reference:
        start_idx = int((seg.start - start_time) / step)
        end_idx = int((seg.end - start_time) / step)
        for i in range(start_idx, end_idx + 1):
            if i not in ref_frames:
                ref_frames[i] = set()
            ref_frames[i].add(seg.speaker)
    
    for seg in hypothesis:
        start_idx = int((seg.start - start_time) / step)
        end_idx = int((seg.end - start_time) / step)
        for i in range(start_idx, end_idx + 1):
            if i not in hyp_frames:
                hyp_frames[i] = set()
            hyp_frames[i].add(seg.speaker)
    
    # Calculate errors
    false_alarm = 0
    missed = 0
    confusion = 0
    total_ref = 0
    
    for frame_idx in range(num_frames):
        ref_speakers = ref_frames.get(frame_idx, set())
        hyp_speakers = hyp_frames.get(frame_idx, set())
        
        total_ref += len(ref_speakers)
        
        if not ref_speakers and hyp_speakers:
            false_alarm += len(hyp_speakers)
        elif ref_speakers and not hyp_speakers:
            missed += len(ref_speakers)
        elif ref_speakers and hyp_speakers:
            # Simple matching (not optimal, but reasonable approximation)
            matched = min(len(ref_speakers), len(hyp_speakers))
            confusion += len(ref_speakers) - matched
            false_alarm += len(hyp_speakers) - matched
    
    if total_ref == 0:
        return 0.0
    
    der = (false_alarm + missed + confusion) / total_ref
    return der

"""Diarization evaluation metrics: DER and JER."""

from src.dataset.base import Segment
from pyannote.core import Annotation, Segment as PySegment
from pyannote.metrics.diarization import DiarizationErrorRate


def compute_der(hypothesis: list[Segment], reference: list[Segment], 
                collar: float = 0.25) -> dict[str, float]:
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
    
    # Convert to pyannote format
    ref_annotation = Annotation()
    for seg in reference:
        ref_annotation[PySegment(seg.start, seg.end)] = seg.speaker
    
    hyp_annotation = Annotation()
    for seg in hypothesis:
        hyp_annotation[PySegment(seg.start, seg.end)] = seg.speaker
    
    # Compute DER
    metric = DiarizationErrorRate(collar=collar)
    der: float = metric(ref_annotation, hyp_annotation) # type: ignore
    
    # Get detailed components
    components = metric.compute_components(ref_annotation, hyp_annotation)
    
    # Ensure all expected components are present
    required_keys = ['false alarm', 'missed detection', 'confusion', 'total']
    missing_keys = [key for key in required_keys if key not in components]
    if missing_keys:
        raise ValueError(f"Missing DER components: {missing_keys}. Got: {list(components.keys())}")
    
    return {
        'DER': der,
        'false_alarm': components['false alarm'],
        'missed_detection': components['missed detection'],
        'confusion': components['confusion'],
        'total': components['total']
    }


def compute_jer(hypothesis: list[Segment], reference: list[Segment], 
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
    jer: float = metric(ref_annotation, hyp_annotation) # type: ignore
    
    return jer

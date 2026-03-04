"""Data structures for evaluation results and summaries."""

from dataclasses import dataclass, asdict


@dataclass
class EvaluationResult:
    """Results from evaluating a system on a recording."""
    recording_id: str
    system: str
    DER: float
    false_alarm: float
    missed_detection: float
    confusion: float
    JER: float
    latency_mean_ms: float
    latency_std_ms: float
    peak_latency_ms: float
    peak_latency_chunk_idx: int
    num_chunks: int
    step_size_ms: float
    duration: float
    peak_gpu_mem_mb: float | None
    peak_gpu_util_pct: float | None
    avg_gpu_util_pct: float | None
    peak_ram_mb: float
    avg_cpu_percent: float
    wall_time_s: float
    total_speech_time: float
    gt_num_speakers: int = 0
    gt_overlap_rate: float = 0.0
    avg_segment_confidence: float | None = None
    min_segment_confidence: float | None = None

    def to_dict(self) -> dict[str, str | float | int]:
        """Convert to dictionary for CSV writing."""
        return asdict(self)
    
    @classmethod
    def get_csv_headers(cls) -> list[str]:
        """Get CSV column headers."""
        return [
            'recording_id', 'system', 'DER', 'false_alarm', 'missed_detection',
            'confusion', 'JER', 'latency_mean_ms', 'latency_std_ms',
            'peak_latency_ms', 'peak_latency_chunk_idx', 'num_chunks', 'step_size_ms', 'duration',
            'peak_gpu_mem_mb', 'peak_gpu_util_pct', 'avg_gpu_util_pct', 'peak_ram_mb', 'avg_cpu_percent',
            'wall_time_s', 'total_speech_time',
            'gt_num_speakers', 'gt_overlap_rate', 'avg_segment_confidence', 'min_segment_confidence'
        ]


@dataclass
class EvaluationSummary:
    """Summary statistics across multiple evaluation results."""
    system: str
    num_recordings: int
    avg_der: float
    avg_false_alarm: float
    avg_missed_detection: float
    avg_confusion: float
    avg_jer: float
    avg_latency_ms: float
    avg_latency_std_ms: float
    avg_peak_latency_ms: float
    total_chunks: int
    avg_wall_time_s: float

    @classmethod
    def from_results(cls, system_name: str, results: list[EvaluationResult]) -> 'EvaluationSummary':
        """Compute summary statistics from a list of results."""
        if not results:
            raise ValueError("Cannot create summary from empty results list")
        n = len(results)
        return cls(
            system=system_name,
            num_recordings=n,
            avg_der=sum(r.DER for r in results) / n,
            avg_false_alarm=sum(r.false_alarm for r in results) / n,
            avg_missed_detection=sum(r.missed_detection for r in results) / n,
            avg_confusion=sum(r.confusion for r in results) / n,
            avg_jer=sum(r.JER for r in results) / n,
            avg_latency_ms=sum(r.latency_mean_ms for r in results) / n,
            avg_latency_std_ms=sum(r.latency_std_ms for r in results) / n,
            avg_peak_latency_ms=sum(r.peak_latency_ms for r in results) / n,
            total_chunks=sum(r.num_chunks for r in results),
            avg_wall_time_s=sum(r.wall_time_s for r in results) / n
        )

    def log_summary(self, logger_func) -> None:
        """Log summary statistics."""
        logger_func(f"{self.system}:")
        logger_func(f"  DER: {self.avg_der:.3f}  (FA: {self.avg_false_alarm:.3f}  "
                    f"Miss: {self.avg_missed_detection:.3f}  Conf: {self.avg_confusion:.3f})")
        logger_func(f"  JER: {self.avg_jer:.3f}")
        logger_func(f"  Avg wall time: {self.avg_wall_time_s:.2f}s per recording")
        logger_func(f"  Chunk Latency: {self.avg_latency_ms:.2f}±{self.avg_latency_std_ms:.2f}ms  "
                    f"| avg peak chunk: {self.avg_peak_latency_ms:.2f}ms")
        logger_func(f"  Total Chunks: {self.total_chunks}")
        logger_func(f"  Processed: {self.num_recordings} recordings")

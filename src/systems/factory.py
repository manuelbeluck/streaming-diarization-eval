"""Factory function for creating diarization systems from configuration."""

from src.config import SystemConfig
from src.systems.base import StreamingDiarizationSystem


def create_system(config: SystemConfig) -> StreamingDiarizationSystem:
    """Create system from configuration.
    
    Args:
        config: System configuration object
        
    Returns:
        StreamingDiarizationSystem instance
        
    Raises:
        ValueError: If system name is unknown
    """
    system_name = config.name.lower()
    
    if system_name == 'diart_default':
        from src.systems.diart.system import DiartSystem
        return DiartSystem(
            name='diart_default',
            duration=config.duration or 5.0,
            step=config.step or 0.5
            # Uses DIART default models
        )
    elif system_name == 'diart_custom':
        from src.systems.diart.system import DiartSystem
        return DiartSystem(
            name='diart_custom',
            duration=config.duration or 5.0,
            step=config.step or 0.5,
            segmentation_model='pyannote/segmentation-3.0',
            embedding_model='pyannote/wespeaker-voxceleb-resnet34-LM'
        )
    elif system_name == 'streaming_sortformer':
        from src.systems.sortformer.system import SortformerSystem
        return SortformerSystem(
            chunk_len=config.chunk_len or 10,
            subsampling_factor=config.subsampling_factor or 10,
            chunk_right_context=config.chunk_right_context or 0,
            chunk_left_context=config.chunk_left_context or 10,
            spkcache_len=config.spkcache_len or 188,
            fifo_len=config.fifo_len or 188,
            spkcache_update_period=config.spkcache_update_period or 144,
            log=config.log or False,
            chunk_size=config.chunk_size,  # Can be None
        )
    else:
        raise ValueError(f"Unknown system: {system_name}")

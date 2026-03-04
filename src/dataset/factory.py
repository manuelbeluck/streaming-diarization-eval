"""Factory function for creating dataset providers from configuration."""

from src.config import DatasetConfig
from src.dataset.base import DatasetProvider


def create_dataset(config: DatasetConfig) -> DatasetProvider:
    """Create dataset provider from configuration.
    
    Args:
        config: Dataset configuration object
        
    Returns:
        DatasetProvider instance
        
    Raises:
        ValueError: If dataset type is unknown
    """
    dataset_type = config.name.lower()
    
    if dataset_type == 'test':
        from src.dataset.testdataset import TestDataset
        return TestDataset(
            data_dir=config.path,
            max_duration=config.max_duration
        )
    elif dataset_type == 'callhome':
        from src.dataset.callhome import CallHomeDataset
        return CallHomeDataset(
            language=config.language,
            data_dir=config.path,
            recordings=config.recordings,
            max_duration=config.max_duration
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

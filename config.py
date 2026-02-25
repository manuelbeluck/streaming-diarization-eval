"""Configuration classes and loading utilities."""

import yaml
from dataclasses import dataclass, field


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    name: str
    path: str = "data"
    language: str = "eng"
    recordings: list[int] | None = None
    max_duration: float | None = None


@dataclass
class SystemConfig:
    """System configuration."""
    name: str
    # DIART parameters
    duration: float | None = None  # Processing window duration (DIART)
    step: float | None = None  # Step size between windows (DIART)
    # Sortformer parameters
    chunk_size: float | None = None  # Input chunk size for feeding audio (Sortformer)
    chunk_len: int | None = None  # Number of frames in processing chunk (Sortformer)
    subsampling_factor: int | None = None  # Subsampling factor (Sortformer)
    chunk_right_context: int | None = None  # Right context frames (Sortformer)
    chunk_left_context: int | None = None  # Left context frames (Sortformer)
    spkcache_len: int | None = None  # Speaker cache length (Sortformer)
    fifo_len: int | None = None  # FIFO buffer length (Sortformer)
    spkcache_update_period: int | None = None  # Speaker cache update period (Sortformer)
    log: bool | None = None  # Enable logging (Sortformer)


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    collar: float = 0.25


@dataclass
class Config:
    """Main configuration."""
    dataset: DatasetConfig
    systems: list[SystemConfig]
    evaluation: EvaluationConfig = field(default_factory=lambda: EvaluationConfig())
    output_dir: str = "./results"


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict: dict = yaml.safe_load(f)
    
    # Parse dataset config
    dataset_dict: dict = config_dict['dataset']
    dataset_config = DatasetConfig(
        name=dataset_dict['name'],
        path=dataset_dict.get('path', 'data'),
        language=dataset_dict.get('language', 'eng'),
        recordings=dataset_dict.get('recordings'),
        max_duration=dataset_dict.get('max_duration')
    )
    
    # Parse system configs
    system_configs = [
        SystemConfig(
            name=sys['name'],
            duration=sys.get('duration'),
            step=sys.get('step'),
            chunk_size=sys.get('chunk_size'),
            chunk_len=sys.get('chunk_len'),
            subsampling_factor=sys.get('subsampling_factor')
        )
        for sys in config_dict['systems']
    ]
    
    # Parse evaluation config
    eval_dict: dict = config_dict.get('evaluation', {})
    eval_config = EvaluationConfig(
        collar=eval_dict.get('collar', 0.25)
    )
    
    return Config(
        dataset=dataset_config,
        systems=system_configs,
        evaluation=eval_config,
        output_dir=config_dict.get('output_dir', './results')
    )

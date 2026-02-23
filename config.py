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
    chunk_size: float = 0.5


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
            chunk_size=sys.get('chunk_size', 0.5)
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

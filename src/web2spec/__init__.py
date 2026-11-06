"""web2spec: local, evidence-first web research for API models."""

__version__ = "0.4.0"

from web2spec.config import DataConfig, EvalConfig, TrainConfig, load_data_config
from web2spec.schema import Sample, SampleError, ValidationReport, validate_sample

__all__ = [
    "DataConfig",
    "EvalConfig",
    "Sample",
    "SampleError",
    "TrainConfig",
    "ValidationReport",
    "load_data_config",
    "validate_sample",
    "__version__",
]

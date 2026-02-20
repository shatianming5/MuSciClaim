"""MuSciClaim configuration package.

Re-exports all public symbols so existing ``from musciclaim.config import X``
statements continue to work after the module-to-package conversion.
"""

from musciclaim.config.dataclasses import (
    AdapterKind,
    DatasetConfig,
    DecodingConfig,
    InferenceConfig,
    IOConfig,
    MatrixConfig,
    ModelModality,
    ModelSpec,
    PoliciesConfig,
    PreprocessingConfig,
    ReportingConfig,
    RunConfig,
)
from musciclaim.config.loader import load_models_config, load_run_config

__all__ = [
    "AdapterKind",
    "DatasetConfig",
    "DecodingConfig",
    "InferenceConfig",
    "IOConfig",
    "MatrixConfig",
    "ModelModality",
    "ModelSpec",
    "PoliciesConfig",
    "PreprocessingConfig",
    "ReportingConfig",
    "RunConfig",
    "load_models_config",
    "load_run_config",
]

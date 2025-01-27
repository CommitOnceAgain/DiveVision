from typing import Any
import torch
from torch.nn import Module
from abc import ABC, abstractmethod

# Define a global registry for models
_model_registry: dict[str, "AbstractModel"] = {}


class AbstractModel(ABC, Module):
    """Abstract class for all models."""

    model_name: str

    def __init_subclass__(cls):
        # Register subclasses in the global registry when they are defined
        if cls not in _model_registry:
            _model_registry[cls.model_name] = cls()

    @classmethod
    def get_model(cls, model_name: str) -> "AbstractModel":
        """Get a model from the registry by name."""
        return _model_registry[model_name]

    @abstractmethod
    def preprocessing(self, input: Any) -> torch.Tensor:
        """Preprocess the input data."""
        raise NotImplementedError

    @abstractmethod
    def postprocessing(self, output: torch.Tensor) -> Any:
        """Postprocess the output data."""
        raise NotImplementedError

from typing import Any, Callable
import torch
from torch.nn import Module
from abc import ABC, abstractmethod


class AbstractModel(ABC, Module):
    """Abstract class for all models."""

    @abstractmethod
    def preprocessing(self, input: Any) -> torch.Tensor:
        """Preprocess the input data."""
        raise NotImplementedError

    @abstractmethod
    def postprocessing(self, output: torch.Tensor) -> Any:
        """Postprocess the output data."""
        raise NotImplementedError

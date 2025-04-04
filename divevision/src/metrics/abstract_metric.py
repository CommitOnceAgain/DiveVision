from abc import ABC, abstractmethod
from typing import Final

import torch

from . import _metric_registry


class AbstractMetric(ABC):
    """Abstract class for metrics"""

    name: str

    def __init_subclass__(cls):
        # Register subclasses in the global registry when they are defined
        if cls not in _metric_registry:
            _metric_registry[cls.name] = cls()

    @abstractmethod
    def compute(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the metric between input and target"""
        raise NotImplementedError

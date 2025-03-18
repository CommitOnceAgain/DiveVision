import logging
from pathlib import Path
from typing import Any
import torch
from torch.nn import Module
from abc import ABC, abstractmethod

from . import _model_registry


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

    def predict(self, input: Any) -> Any:
        return self.postprocessing(self.forward(self.preprocessing(input)))

    @abstractmethod
    def preprocessing(self, input: Any) -> torch.Tensor:
        """Preprocess the input data."""
        raise NotImplementedError

    @abstractmethod
    def postprocessing(self, output: torch.Tensor) -> Any:
        """Postprocess the output data."""
        raise NotImplementedError

    def load_model(self, device: torch.device) -> None:
        self.model.to(device)
        checkpoint_path = Path(self.model_ckpt).resolve()
        if not checkpoint_path.exists():
            logging.warning(f"Could not find model weights at: {checkpoint_path}")
        else:
            self.model.load_state_dict(
                torch.load(
                    checkpoint_path,
                    weights_only=False,
                    map_location=device,
                )
            )

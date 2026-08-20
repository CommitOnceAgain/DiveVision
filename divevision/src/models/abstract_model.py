import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from torch.nn import Module

from . import _enhancement_model_registry


class AbstractModel(ABC, Module):
    """Abstract class for all Enhancement Models."""

    name: str

    def __init_subclass__(cls):
        # Register subclasses in the global registry when they are defined
        if cls not in _enhancement_model_registry:
            _enhancement_model_registry[cls.name] = cls()

    @classmethod
    def get_model(cls, model_name: str) -> "AbstractModel":
        """Get an Enhancement Model from the registry by name."""
        return _enhancement_model_registry[model_name]

    def predict(self, degraded_image: Any) -> Any:
        return self.postprocessing(self.forward(self.preprocessing(degraded_image)))

    @abstractmethod
    def preprocessing(self, degraded_image: Any) -> torch.Tensor:
        """Preprocess the Degraded Image."""
        raise NotImplementedError

    @abstractmethod
    def postprocessing(self, model_output: torch.Tensor) -> Any:
        """Postprocess the model output into an Enhanced Image."""
        raise NotImplementedError

    def load_checkpoint(self, device: torch.device) -> None:
        self.model_implementation.to(device)
        checkpoint_path = Path(self.checkpoint_path).resolve()
        if not checkpoint_path.exists():
            logging.warning(f"Could not find checkpoint at: {checkpoint_path}")
        else:
            self.model_implementation.load_state_dict(
                torch.load(
                    checkpoint_path,
                    weights_only=False,
                    map_location=device,
                )
            )

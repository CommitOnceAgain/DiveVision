import logging
from pathlib import Path

import PIL
import numpy as np
from omegaconf import OmegaConf
from divevision.models.CEVAE.build.from_config import instantiate_from_config
from divevision.src.models.abstract_model import AbstractModel
import torch
from torchvision.transforms.functional import resize, to_tensor
from PIL import Image


class CVAEModelWrapper(AbstractModel):

    model_name = "CVAE"

    def __init__(
        self,
        config_file: str = "divevision/models/CEVAE/configs/cevae_GAN_lsui.yaml",
        device=torch.device("cpu"),
        **kwargs,
    ):
        super().__init__()

        # Check if config file exists
        assert (
            Path(config_file).resolve().exists()
        ), f"Cannot find configuration file at: {config_file}"

        # Re-use the instanciation technique of the original project, with a .yaml file containing all necessary parameters
        config = OmegaConf.load(config_file)
        self.model = instantiate_from_config(config.model)

        # Retrieve model checkpoint path from config
        self.model_ckpt = config.model.params.ckpt_path

        self.load_model(device)

    def load_model(self, device: torch.device) -> None:
        self.model.to(device)
        checkpoint_path = Path(self.model_ckpt).resolve()
        if not checkpoint_path.exists():
            logging.warning(f"Could not find model weights at: {checkpoint_path}")
        else:
            checkpoint = torch.load(
                checkpoint_path,
                weights_only=False,
                map_location=device,
            )
            self.model.load_state_dict(checkpoint["state_dict"], strict=False)

    def preprocessing(self, input: Image.Image) -> torch.Tensor:
        # Preprocessing function is gathered from original ce-vae github implementation
        img = resize(
            input,
            (256, 256),
            interpolation=Image.Resampling.LANCZOS,
        )
        img = to_tensor(img)
        if img.size()[1] < 3:
            img = torch.cat([img, img, img], dim=1)
        return 2.0 * img - 1.0  # Resample values from [0, 1] to [-1, 1]

    def postprocessing(self, output: torch.Tensor) -> Image.Image:
        output = output.clamp(-1, 1)
        output = (output + 1.0) / 2.0
        output = output.permute(1, 2, 0).numpy()
        output = (255 * output).astype(np.uint8)
        output = Image.fromarray(output)
        if not output.mode == "RGB":
            output = output.convert("RGB")
        return output

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Check if the input has a batch dimension (N) and add it if not
        if input.ndim == 3:
            input = torch.unsqueeze(input, dim=0)
        elif input.ndim != 4:
            raise ValueError("Input must be a tensor of shape (N, C, H, W)")

        # Get the model output
        return self.model.forward(input)

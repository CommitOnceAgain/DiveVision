import logging
from pathlib import Path
import torch
from torchvision.transforms import v2 as transforms
from divevision.models.peng_et_al.Ushape_Trans import Generator as UshapeModel
from divevision.src.models.abstract_model import AbstractModel
from PIL import Image


class UShapeModelWrapper(AbstractModel):

    model_name = "U-Shape"

    def __init__(
        self,
        model_ckpt: str = "divevision/models/peng_et_al/saved_models/G/generator_795.pth",
        device: torch.device = torch.device("cpu"),
        # Legacy parameters
        img_dim=256,
        patch_dim=16,
        embedding_dim=512,
        num_channels=3,
        num_heads=8,
        num_layers=4,
        hidden_dim=256,
        dropout_rate=0.0,
        attn_dropout_rate=0.0,
        in_ch=3,
        out_ch=3,
        conv_patch_representation=True,
        positional_encoding_type="learned",
        use_eql=True,
    ):
        super().__init__()

        self.model = UshapeModel(
            img_dim=img_dim,
            patch_dim=patch_dim,
            embedding_dim=embedding_dim,
            num_channels=num_channels,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            attn_dropout_rate=attn_dropout_rate,
            in_ch=in_ch,
            out_ch=out_ch,
            conv_patch_representation=conv_patch_representation,
            positional_encoding_type=positional_encoding_type,
            use_eql=use_eql,
        )

        self.img_dim = img_dim
        self.model_ckpt = model_ckpt

        # Load the model and its weights on the given 'device'
        self.model.to(device)
        checkpoint_path = Path(self.model_ckpt).resolve()
        if not checkpoint_path.exists():
            logging.warning(f"Could not find model weights at {checkpoint_path}")
        else:
            self.model.load_state_dict(
                torch.load(
                    checkpoint_path,
                    weights_only=False,
                    map_location=device,
                )
            )

    def predict(self, input: Image) -> list[Image]:
        """We redefine the predict function, because the model accepts only 256x256 pixels images. We want to resize to the original image size."""
        # Preprocess the input
        input_tensor = self.preprocessing(input)
        model_output = self.forward(input_tensor)
        # Resize the model output to the original input size
        resized_tensor = transforms.Resize(
            tuple(
                input_tensor.shape[1:]
            ),  # Retrieve the size of the image by removing batch size
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True,
        )(model_output)
        # Postprocess the resized image
        return self.postprocessing(resized_tensor)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Check if the input has a batch dimension (N) and add it if not
        if input.ndim == 3:
            input = torch.unsqueeze(input, dim=0)
        elif input.ndim != 4:
            raise ValueError("Input must be a tensor of shape (N, C, H, W)")

        # Get the model output
        output = self.model.forward(input)
        # Output is actually a tuple of four tensors, we want to retrieve the last one
        return output[-1]

    def preprocessing(self, input: Image) -> torch.Tensor:
        transformations = transforms.Compose(
            [
                # Convert the image to a tensor
                transforms.PILToTensor(),
                # Normalize data int the range [0,255] (better for Resizing)
                transforms.ToDtype(torch.uint8, scale=True),
                # Resize the image to expected_size
                transforms.Resize(
                    (self.img_dim, self.img_dim),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                # Convert the tensor to a float32 type in the range [0,1] (better for training)
                transforms.ToDtype(torch.float32, scale=True),
            ]
        )
        return transformations(input)

    def postprocessing(
        self,
        output: torch.Tensor,
    ) -> Image:
        """Postprocess the tensor output of the model to an image. Input can be batched."""

        def process_a_single_tensor(tensor):
            # Remove the batch dimension
            tensor = torch.squeeze(tensor, dim=0)
            # Clip data between [0,1]
            tensor = tensor.clip(0.0, 1.0)
            # Convert to PIL image
            image = transforms.ToPILImage()(tensor)
            return image

        if output.ndim == 4:
            image = [process_a_single_tensor(t) for t in output]
        elif output.ndim == 3:
            image = process_a_single_tensor(output)
        else:
            raise ValueError("Output tensor must be either a single or batched tensor.")

        return image

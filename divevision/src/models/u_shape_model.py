import torch
from PIL.Image import Image
from torchvision.transforms import v2 as transforms

from divevision.models.UShapeTransformer.Ushape_Trans import Generator as UshapeModel
from divevision.src.models.abstract_model import AbstractModel


class UShapeModelWrapper(AbstractModel):

    name = "U-Shape"

    def __init__(
        self,
        checkpoint_path: str = "divevision/models/UShapeTransformer/saved_models/G/generator_795.pth",
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

        self.model_implementation = UshapeModel(
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
        self.checkpoint_path = checkpoint_path

        self.load_checkpoint(device)

    def predict(self, degraded_image: Image) -> list[Image]:
        """We redefine the predict function, because the Model Implementation only accepts 256x256 pixels images. We want to resize back to the original image size."""
        # Preprocess the Degraded Image
        preprocessed_image = self.preprocessing(degraded_image)
        model_output = self.forward(preprocessed_image)
        # Resize the model output to the original input size
        resized_output = transforms.Resize(
            tuple(
                preprocessed_image.shape[1:]
            ),  # Retrieve the size of the image by removing batch size
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True,
        )(model_output)
        # Postprocess the resized output into an Enhanced Image
        return self.postprocessing(resized_output)

    def forward(self, preprocessed_image: torch.Tensor) -> torch.Tensor:
        # Check if the input has a batch dimension (N) and add it if not
        if preprocessed_image.ndim == 3:
            preprocessed_image = torch.unsqueeze(preprocessed_image, dim=0)
        elif preprocessed_image.ndim != 4:
            raise ValueError("Input must be a tensor of shape (N, C, H, W)")

        # Get the Model Implementation output
        model_output = self.model_implementation.forward(preprocessed_image)
        # Output is actually a tuple of four tensors, we want to retrieve the last one
        return model_output[-1]

    def preprocessing(self, degraded_image: Image | list[Image]) -> torch.Tensor:
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
        if isinstance(degraded_image, list):  # Handle batch preprocessing
            preprocessed_images = [transformations(item) for item in degraded_image]
            return torch.stack(preprocessed_images, dim=0)
        return transformations(degraded_image)

    def postprocessing(
        self,
        model_output: torch.Tensor,
    ) -> list[Image]:
        """Postprocess the tensor output of the Model Implementation into an Enhanced Image. Input can be batched."""

        def process_a_single_tensor(tensor):
            # Remove the batch dimension
            tensor = torch.squeeze(tensor, dim=0)
            # Clip data between [0,1]
            tensor = tensor.clip(0.0, 1.0)
            # Convert to PIL image
            image = transforms.ToPILImage()(tensor)
            return image

        if model_output.ndim == 4:
            return [process_a_single_tensor(t) for t in model_output]
        elif model_output.ndim == 3:
            return [process_a_single_tensor(model_output)]
        else:
            raise ValueError("Output tensor must be either a single or batched tensor.")

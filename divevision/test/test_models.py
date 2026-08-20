from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
from PIL import Image

from divevision.src.models.cvae_model import CVAEModelWrapper
from divevision.src.models.u_shape_model import UShapeModelWrapper


@pytest.mark.parametrize(
    "enhancement_model",
    [
        pytest.param(UShapeModelWrapper(), id="UShapeModel"),
        pytest.param(CVAEModelWrapper(), id="CVAEModel"),
    ],
)
class TestModels:

    def id_img(value):
        if isinstance(value, tuple):
            return repr(value)
        return value

    @pytest.mark.parametrize(
        "image_shape, context",
        [
            ((32, 32, 3), does_not_raise()),
            ((256, 256, 3), does_not_raise()),
            ((512, 512, 3), does_not_raise()),
            ((2048, 2048, 3), does_not_raise()),
            ((2, 32, 32, 3), does_not_raise()),
            ((2, 2048, 2048, 3), does_not_raise()),
            ((512, 256, 3), does_not_raise()),
        ],
        ids=id_img,
    )
    def test_model_pipeline(
        self,
        image_shape,
        context,
        enhancement_model,
    ) -> None:
        with context:
            # Create a numpy array with the expected shape
            array = np.random.rand(*image_shape)
            # Create an image from this array, or a list of images if it's a batch
            if len(array.shape) == 3:
                degraded_image = Image.fromarray(array, mode="RGB")
            elif len(array.shape) == 4:
                degraded_image = [
                    Image.fromarray(subarray, mode="RGB") for subarray in array
                ]
            # Preprocess the Degraded Image
            preprocessed_image = enhancement_model.preprocessing(degraded_image)
            # Forward pass on the preprocessed image
            model_output = enhancement_model(preprocessed_image)
            # Postprocess the model output into an Enhanced Image
            enhancement_model.postprocessing(model_output)

from contextlib import nullcontext as does_not_raise
import numpy as np
import pytest
from PIL import Image
from divevision.src.models.cvae_model import CVAEModelWrapper
from divevision.src.models.u_shape_model import UShapeModelWrapper


@pytest.mark.parametrize(
    "model",
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
        model,
    ) -> None:
        with context:
            # Create a numpy array with the expected shape
            array = np.random.rand(*image_shape)
            # Create an image from this array, or a list of images if it's a batch
            if len(array.shape) == 3:
                input = Image.fromarray(array, mode="RGB")
            elif len(array.shape) == 4:
                input = [Image.fromarray(subarray, mode="RGB") for subarray in array]
            # Preprocess an image
            preprocessed_input = model.preprocessing(input)
            # Forward pass on the preprocessed input
            output = model(preprocessed_input)
            # Postprocessing of the model output
            model.postprocessing(output)

from contextlib import nullcontext as does_not_raise
import pytest
import torch

from divevision.src.models.u_shape_model import UShapeModelWrapper


@pytest.mark.parametrize(
    "model",
    [UShapeModelWrapper()],
)
class TestModels:

    @pytest.mark.parametrize(
        "image_shape, context",
        [
            ((3, 32, 32), does_not_raise()),
            ((3, 256, 256), does_not_raise()),
            ((3, 512, 512), does_not_raise()),
            ((3, 2048, 2048), does_not_raise()),
            ((2, 3, 32, 32), does_not_raise()),
            ((2, 3, 2048, 2048), does_not_raise()),
            ((3, 512, 256), does_not_raise()),
            ((1, 256, 256), pytest.raises(RuntimeError)),
            ((4, 256, 256), pytest.raises(RuntimeError)),
        ],
    )
    def test_model_pipeline(
        self,
        image_shape,
        context,
        model,
    ) -> None:
        with context:
            preprocessed_input = model.preprocessing(torch.randn(image_shape))
            output = model(preprocessed_input)
            model.postprocessing(output)

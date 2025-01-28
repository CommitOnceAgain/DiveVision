# Define a global registry for models
_model_registry: dict[str, "AbstractModel"] = {}

# Must import all models here to register them
from .abstract_model import AbstractModel
from .u_shape_model import UShapeModelWrapper

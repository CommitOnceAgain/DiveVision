# Define a global registry for Enhancement Models
_enhancement_model_registry: dict[str, "AbstractModel"] = {}

# Must import all Enhancement Models here to register them
from .abstract_model import AbstractModel
from .u_shape_model import UShapeModelWrapper

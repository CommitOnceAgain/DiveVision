# Define a registry for Evaluation Metrics
_metric_registry: dict[str, "AbstractMetric"] = {}

# Must import all Evaluation Metrics here to register them
from .abstract_metric import AbstractMetric
from .psnr import PSNRMetric
from .ssim import SSIMMetric

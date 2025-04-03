# Define a registry for metrics
_metric_registry: dict[str, "AbstractMetric"] = {}

# Must import all models here to register them
from .abstract_metric import AbstractMetric
from .psnr import PSNRMetric
from .ssim import SSIMMetric

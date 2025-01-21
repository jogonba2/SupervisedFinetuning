# flake8: noqa
from typing import Callable

from .alvaro import contextual_alvaro_calibration, domain_alvaro_calibration
from .batch import adjustable_batch_calibration, batch_calibration
from .contextual import contextual_calibration
from .domain import domain_calibration

# Register here new calibrations
calibration_registry: dict[str, Callable] = {
    "contextual": contextual_calibration,
    "domain": domain_calibration,
    "batch": batch_calibration,
    "adjustable_batch": adjustable_batch_calibration,
    "contextual_alvaro": contextual_alvaro_calibration,
    "domain_alvaro": domain_alvaro_calibration,
}

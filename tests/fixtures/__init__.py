"""
ClashVision synthetic fixture generators and recorded ground-truth datasets.
"""

from tests.fixtures.synthetic_generator import (
    SyntheticGenerator,
    SyntheticFrameConfig,
    SyntheticCropConfig,
    generate_state_frame,
    generate_resource_crop,
    generate_timer_crop,
)
from tests.fixtures.layout_generator import (
    BaseLayoutType,
    BaseLayoutModel,
    LayoutGenerator,
    generate_canonical_layout,
)
from tests.fixtures.recorded_samples import (
    RECORDED_SAMPLES,
    OCR_CORPUS,
    RecordedSample,
    get_sample_by_state,
)

__all__ = [
    "SyntheticGenerator",
    "SyntheticFrameConfig",
    "SyntheticCropConfig",
    "generate_state_frame",
    "generate_resource_crop",
    "generate_timer_crop",
    "BaseLayoutType",
    "BaseLayoutModel",
    "LayoutGenerator",
    "generate_canonical_layout",
    "RECORDED_SAMPLES",
    "OCR_CORPUS",
    "RecordedSample",
    "get_sample_by_state",
]

"""Register the VisionHOPE components required by MMDetection."""

from mmdet.registry import MODELS

from visionhope.tasks.backbone import VisionHOPEBackbone
from .roi_extractor import ContiguousRoIExtractor


MODELS.register_module(module=VisionHOPEBackbone)

__all__ = ["VisionHOPEBackbone", "ContiguousRoIExtractor"]

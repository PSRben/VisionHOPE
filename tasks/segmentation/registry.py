"""Register the VisionHOPE backbone required by MMSegmentation."""

from mmseg.registry import MODELS

from visionhope.tasks.backbone import VisionHOPEBackbone


MODELS.register_module(module=VisionHOPEBackbone)

__all__ = ["VisionHOPEBackbone"]

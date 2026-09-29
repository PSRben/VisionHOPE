"""VisionHOPE Tiny, Small, and Base model configurations."""

from copy import deepcopy
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.registry import register_model
from .hierarchical import HierarchicalVisionHOPE


MODEL_CONFIGS = {
    "visionhope_tiny": dict(embed_dims=(64, 128, 256, 512),
                           mixer_dims=(32, 64, 128, 256), depths=(3, 4, 18, 4)),
    "visionhope_small": dict(embed_dims=(64, 160, 320, 512),
                            mixer_dims=(32, 80, 160, 256), depths=(4, 8, 25, 8)),
    "visionhope_base": dict(embed_dims=(96, 192, 448, 640),
                           mixer_dims=(48, 96, 224, 320), depths=(4, 8, 25, 8)),
}


def create_model(name, pretrained=False, checkpoint_path=None, **kwargs):
    """Create a VisionHOPE model and optionally load a checkpoint."""
    if name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model {name!r}; choose from {tuple(MODEL_CONFIGS)}")
    if pretrained and not checkpoint_path:
        raise ValueError("Provide checkpoint_path for pretrained weights")
    defaults = {"drop_rate": 0.0, "drop_connect_rate": None, "drop_block_rate": None,
                "global_pool": "avg", "bn_momentum": None, "bn_eps": None,
                "scriptable": False, "pretrained_cfg": None, "pretrained_cfg_overlay": None}
    for key, default in defaults.items():
        if key in kwargs:
            value = kwargs.pop(key)
            if value is not None and value != default:
                raise ValueError(f"Unsupported {key}={value!r}")
    config = deepcopy(MODEL_CONFIGS[name])
    config.update(kwargs)
    model = HierarchicalVisionHOPE(**config)
    model.default_cfg = dict(url="", num_classes=1000, input_size=(3, 224, 224),
                             pool_size=None, crop_pct=1.0, interpolation="bicubic",
                             mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD,
                             classifier="head")
    if checkpoint_path:
        from visionhope.tools.checkpoint import load_checkpoint
        load_checkpoint(model, checkpoint_path)
    return model


@register_model
def visionhope_tiny(pretrained=False, **kwargs):
    return create_model("visionhope_tiny", pretrained=pretrained, **kwargs)


@register_model
def visionhope_small(pretrained=False, **kwargs):
    return create_model("visionhope_small", pretrained=pretrained, **kwargs)


@register_model
def visionhope_base(pretrained=False, **kwargs):
    return create_model("visionhope_base", pretrained=pretrained, **kwargs)

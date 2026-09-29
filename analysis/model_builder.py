"""Construct classifiers, Mask R-CNN, and UPerNet without dataset dependencies."""

from importlib import import_module

import torch

from visionhope.tools.checkpoint import load_checkpoint


FORWARD_SCOPES = {
    "classification": "Complete ImageNet classifier",
    "detection": "Mask R-CNN tensor forward: backbone, FPN, RPN proposals, bbox and mask heads; "
                 "excludes final prediction postprocessing",
    "segmentation": "UPerNet tensor forward: backbone and decode head; excludes the training "
                    "auxiliary head and final prediction resizing/postprocessing",
}


def build_model(args, device):
    if args.task == "classification":
        from visionhope.models import create_model

        model = create_model(args.model, img_size=args.shape, drop_path_rate=0.0,
                             share_direction_skip=args.share_direction_skip)
    else:
        from mmengine.config import Config
        from mmengine.model import revert_sync_batchnorm
        from mmengine.registry import init_default_scope

        config_module = import_module(f"visionhope.tasks.{args.task}.config")
        import_module(f"visionhope.tasks.{args.task}.registry")
        cfg = Config(config_module.build_config(args.model.removeprefix("visionhope_")))
        cfg.model.backbone.share_direction_skip = args.share_direction_skip
        init_default_scope(cfg.default_scope)
        registry = import_module("mmdet.registry" if args.task == "detection" else "mmseg.registry")
        model = registry.MODELS.build(cfg.model)
        if args.checkpoint is None:
            model.init_weights()
        model = revert_sync_batchnorm(model)
    if args.checkpoint is not None:
        load_checkpoint(model, args.checkpoint, strict=True)

    params = sum(parameter.numel() for parameter in model.parameters())
    auxiliary_params = 0
    if args.task == "segmentation" and model.auxiliary_head is not None:
        auxiliary_params = sum(parameter.numel() for parameter in model.auxiliary_head.parameters())
        model.auxiliary_head = None
    metadata = {
        "task": args.task,
        "model": args.model,
        "share_direction_skip": args.share_direction_skip,
        "checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
        "params": params,
        "params_millions": params / 1e6,
        "inference_params": params - auxiliary_params,
        "auxiliary_params": auxiliary_params,
        "forward_scope": FORWARD_SCOPES[args.task],
    }
    return model.eval().to(device), metadata


def make_inputs(args, device, *, channels_last=False):
    inputs = torch.randn(args.batch_size, 3, *args.shape, device=device)
    if channels_last:
        inputs = inputs.contiguous(memory_format=torch.channels_last)
    return inputs


def make_forward(model, inputs, task):
    if task == "classification":
        return lambda: model(inputs)
    if task == "detection":
        from mmdet.structures import DetDataSample

        shape = tuple(inputs.shape[-2:])
        samples = [DetDataSample(metainfo=dict(
            img_id=index, img_shape=shape, ori_shape=shape, pad_shape=shape,
            batch_input_shape=shape, scale_factor=(1.0, 1.0),
        )) for index in range(inputs.shape[0])]
        return lambda: model(inputs, data_samples=samples, mode="tensor")
    return lambda: model(inputs, mode="tensor")

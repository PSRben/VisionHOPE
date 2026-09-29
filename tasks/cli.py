"""Public task dispatcher used by the train and inference shell entry points."""

import argparse
import sys

from .classification.config import TRAINING_CHOICES, add_arguments, read_config, validate_args
from .specs import MODEL_NAMES


def parser(classification_defaults=None):
    root = argparse.ArgumentParser(description="VisionHOPE training and inference tasks")
    actions = root.add_subparsers(dest="action", required=True)
    training = actions.add_parser("train").add_subparsers(dest="task", required=True)
    classification = training.add_parser(
        "classification", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="ImageNet training; model recipes are in scripts/train.",
        epilog="See tasks/README.md for training and evaluation examples.",
    )
    add_arguments(classification, classification_defaults)
    for task in ("detection", "segmentation"):
        dense = training.add_parser(
            task, epilog="See tasks/README.md for base defaults and model recipes.")
        dense.add_argument("--arch", choices=("tiny", "small", "base"), default="tiny",
                           help="Hierarchical backbone size; default: tiny")
        dense.add_argument("--data-root", required=True,
                           help="COCO root or ADEChallengeData2016 root for the selected task")
        dense.add_argument("--output", required=True,
                           help="Directory for checkpoints, configuration and logs")
        dense.add_argument("--pretrained",
                           help="ImageNet checkpoint for backbone initialization")
        dense.add_argument("--resume",
                           help="training_state.pth containing model, optimizer and scheduler state")
        dense.add_argument("--schedule", choices=("1x", "3x") if task == "detection" else ("1x",),
                           default="1x",
                           help="COCO: 1x=12 epochs, 3x=36 epochs; "
                                "ADE20K: 1x=160k iterations")
        dense.add_argument("--batch-size", type=int,
                           help="Training images per GPU; base defaults: COCO 8, ADE20K 2")
        dense.add_argument("--lr", type=float,
                           help="Absolute LR; otherwise scale 0.0004 at batch 64 (COCO), or "
                                "0.00006 at batch 16 (ADE20K)")
        dense.add_argument("--save-top-k", type=int, default=5,
                           help="Best validation checkpoints to retain, ranked by bbox mAP or "
                                "mIoU; default: 5")
        dense.add_argument("--workers", type=int,
                           help="Loader workers per process for train/val/test; 0 disables "
                                "worker processes; omitted: 4/2/2")
        dense.add_argument("--precision", choices=TRAINING_CHOICES["precision"],
                           help="Training precision; omitted: FP16 for COCO, FP32 for ADE20K; "
                                "validation during training uses FP32")
        dense.add_argument("--drop-path-rate", type=float,
                           help="Backbone stochastic-depth maximum in [0, 1); base default: 0.2")
        dense.add_argument("--collect-device", choices=("cpu", "gpu"),
                           help="Device for collecting evaluation results; default: cpu")
        dense.add_argument("--grad-accum-steps", type=int, default=1,
                           help="Microbatches per optimizer update attempt; default: 1")
        dense.add_argument("--activation-checkpointing", action=argparse.BooleanOptionalAction,
                           default=None,
                           help="Recompute backbone block activations during backward; base "
                                "default: disabled")
        dense.add_argument("--share-direction-skip", action=argparse.BooleanOptionalAction,
                           default=None,
                           help="Share channel-wise skip scales across scan directions; base "
                                "default: enabled")
        dense.add_argument("--max-steps", type=int, default=0,
                           help="Maximum optimizer update attempts; 0 uses the "
                                "full schedule; incompatible with resume")
        dense.add_argument("--seed", type=int, default=42, help="Training random seed; default: 42")
    inference = actions.add_parser("inference").add_subparsers(dest="task", required=True)
    for task in ("classification", "detection", "segmentation"):
        single = inference.add_parser(
            task, epilog="See tasks/README.md; scripts/test supplies each model's recipe.")
        single.add_argument("--model", choices=MODEL_NAMES, default="visionhope_tiny", metavar="MODEL",
                            help="Architecture matching the checkpoint; choices: %(choices)s. "
                                 "Default: visionhope_tiny")
        single.add_argument("--checkpoint", required=True,
                            help="Checkpoint containing state_dict; dense tasks require the full Mask "
                                 "R-CNN or UPerNet model")
        single.add_argument("--data-root", required=True, help="Dataset root for the selected task")
        single.add_argument("--output", required=True,
                            help="Result directory; saves <id>/<mode>.json without overwriting")
        single.add_argument("--id", default=task,
                            help="Experiment identifier used in output paths; defaults to the "
                                 "task name")
        single.add_argument("--mode", choices=("ordinary", "fused"), default="fused",
                            help="Inference path; fused enables fast inference (default: fused)")
        single.add_argument("--schedule",
                            choices=("1x",) if task == "segmentation" else ("1x", "3x"),
                            default="1x",
                            help="COCO checkpoint schedule; ignored for classification; ADE20K "
                                 "keeps 1x")
        batch_help = "Images per evaluation batch; default: " + ("32" if task == "classification" else "1")
        if task == "segmentation":
            batch_help += "; images in a batch must have the same size after preprocessing"
        single.add_argument("--batch-size", type=int, help=batch_help)
        single.add_argument("--workers", type=int,
                            help="Loader workers; 0 loads in the main process; defaults: "
                                 "classification 4, dense tasks 2")
        single.add_argument("--limit", type=int, default=0,
                            help="Maximum images to evaluate; classification finishes "
                                 "the last batch; 0 evaluates all images")
        single.add_argument("--share-direction-skip", action=argparse.BooleanOptionalAction,
                            default=True,
                            help="Skip-scale sharing must match the checkpoint")
        single.add_argument("--precision", choices=TRAINING_CHOICES["precision"],
                            help="Evaluation precision; default: fp32")
        if task == "classification":
            single.add_argument("--num-classes", type=int,
                                help="Number of output classes; must match the checkpoint; "
                                     "default: 1000")
            single.add_argument("--crop-pct", type=float,
                                help="Validation resize/center-crop ratio in (0, 1]; base "
                                     "default: 1.0")
            single.add_argument("--input-size", type=int, nargs=3, metavar=("C", "H", "W"),
                                help="Input dimensions; C must be 3; base default: 3 224 224")
            single.add_argument("--interpolation", choices=("bicubic", "bilinear", "nearest"),
                                help="Validation resize interpolation; base default: bicubic")
            single.add_argument("--channels-last", action=argparse.BooleanOptionalAction,
                                default=None,
                                help="Use channels-last memory layout; base default: disabled")
            single.add_argument("--cudnn-benchmark", action=argparse.BooleanOptionalAction,
                                default=None,
                                help="Enable cuDNN convolution algorithm benchmarking; "
                                     "default: enabled")
        else:
            single.add_argument("--test-scale", type=int, nargs=2, metavar=("W", "H"),
                                help="Test resize scale, preserving aspect ratio; default: "
                                     + ("1333 800" if task == "detection" else "2048 512"))
            single.add_argument("--collect-device", choices=("cpu", "gpu"),
                                help="Device for collecting evaluation results; default: cpu")
    evaluation = inference.add_parser(
        "evaluate", help="Evaluate one entry from a resource manifest")
    merge = inference.add_parser(
        "merge", help="Evaluate concatenated COCO/ADE20K prediction shards")
    for command in (evaluation, merge):
        command.add_argument("--manifest", required=True,
                             help="JSON evaluation manifest")
        command.add_argument("--id", required=True, help="Identifier of one entry in the manifest")
        command.add_argument("--mode", choices=("ordinary", "fused"), required=True,
                             help="Evaluation mode; must match the prediction shards when merging")
        command.add_argument("--output", required=True,
                             help="Result root; merge reads prediction shards from the same root")
        command.add_argument("--num-shards", type=int, default=1,
                             help="Total number of disjoint COCO/ADE20K prediction shards; "
                                  "default: 1")
    evaluation.add_argument("--shard-index", type=int,
                            help="Zero-based shard index; omit for full evaluation; dense tasks "
                                 "only")
    evaluation.add_argument("--batch-size", type=int,
                            help="Override the manifest's evaluation batch size")
    evaluation.add_argument("--workers", type=int,
                            help="Override loader workers; omitted values use task defaults")
    evaluation.add_argument("--limit", type=int, default=0,
                            help="Maximum images to evaluate; 0 evaluates the full "
                                 "entry; incompatible with sharding")
    table = inference.add_parser(
        "table", help="Run the full manifest and produce CSV/Markdown tables")
    table.add_argument("--manifest", required=True,
                       help="JSON evaluation manifest; entries may select different Python "
                            "environments")
    table.add_argument("--output", required=True,
                       help="Root for per-entry results and summary tables")
    table.add_argument("--ids", nargs="+",
                       help="Manifest IDs to evaluate; omit to select every entry")
    table.add_argument("--modes", nargs="+", choices=("ordinary", "fused"),
                       default=["ordinary", "fused"],
                       help="Evaluation modes; defaults to both ordinary and fused")
    table.add_argument("--devices", nargs="+", default=["0"],
                       help="CUDA device indices or UUIDs; one evaluation process per listed "
                            "device; default: 0")
    table.add_argument("--workers", type=int,
                       help="Loader workers per evaluation process; omitted values use task "
                            "defaults")
    table.add_argument("--summarize-only", action="store_true",
                       help="Build summary tables from complete saved evaluation results")
    inference.add_parser(
        "efficiency", help="VisionHOPE inference throughput and memory", add_help=False)
    inference.add_parser(
        "complexity", help="VisionHOPE parameter count and inference FLOPs", add_help=False)
    return root


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:2] == ["inference", "efficiency"]:
        from visionhope.analysis.benchmark import main as benchmark
        return benchmark(argv[2:])
    if argv[:2] == ["inference", "complexity"]:
        from visionhope.analysis.complexity import main as complexity
        return complexity(argv[2:])
    defaults = read_config(argv[2:]) if argv[:2] == ["train", "classification"] else None
    args = parser(defaults).parse_args(argv)
    action, task = args.action, args.task
    del args.action, args.task
    if action == "train" and task == "classification":
        del args.config
        validate_args(args)
        from visionhope.train.classification import train
        return train(args)
    values = vars(args).copy()
    if action == "train":
        from visionhope.train.dense import train
        return train(task, **values)
    if task == "table":
        from visionhope.inference.table import run_table
        values["output_dir"] = values.pop("output")
        return run_table(**values)
    from visionhope.inference.evaluate import evaluate
    if task in {"evaluate", "merge"}:
        from visionhope.inference.table import load_manifest
        entries = load_manifest(values.pop("manifest"))
        entry_id = values.pop("id")
        matches = [entry for entry in entries if entry["id"] == entry_id]
        if not matches:
            raise ValueError(f"Unknown manifest ID {entry_id!r}")
        values["output_dir"] = values.pop("output")
        if task == "merge":
            from visionhope.inference.merge import merge
            return merge(matches[0], **values)
        return evaluate(matches[0], **values)
    entry = {"id": values.pop("id"), "task": task, "model": values.pop("model"),
             "checkpoint": values.pop("checkpoint"),
             "dataset": values.pop("data_root"), "schedule": values.pop("schedule")}
    entry["share_direction_skip"] = values.pop("share_direction_skip")
    protocol_fields = ("precision", "num_classes", "crop_pct", "input_size", "interpolation",
                       "channels_last", "cudnn_benchmark") \
        if task == "classification" else ("precision", "test_scale", "collect_device")
    for key in protocol_fields:
        value = values.pop(key)
        if value is not None:
            entry[key] = value
    values["output_dir"] = values.pop("output")
    return evaluate(entry, **values)


if __name__ == "__main__":
    main()

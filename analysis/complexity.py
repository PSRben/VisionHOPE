"""Measure VisionHOPE parameters and inference FLOPs at a fixed input shape."""

import argparse

import torch

from .common import (add_model_arguments, check_output, configure_runtime, runtime_info,
                     validate_model_arguments, write_report)
from .model_builder import build_model, make_forward, make_inputs


def profile(args):
    from .operation_count import OperationCount

    device = configure_runtime(args)
    model, metadata = build_model(args, device)
    records = []
    with torch.no_grad():
        for index in range(args.samples):
            inputs = make_inputs(args, device)
            forward = make_forward(model, inputs, args.task)
            rois, handles = [], []
            for name, module in model.named_modules():
                if name.endswith(("bbox_roi_extractor", "mask_roi_extractor")):
                    def record_rois(layer, values, name=name):
                        rois.append({"name": name, "count": int(values[1].shape[0])})
                    handles.append(module.register_forward_pre_hook(record_rois))
            try:
                with OperationCount(model) as counter:
                    output = forward()
                check_output(output)
            finally:
                for handle in handles:
                    handle.remove()
            records.append(dict(sample=index, rois=rois, **counter.report()))
            del output, forward, inputs
    mean_macs = sum(row["macs"] for row in records) / len(records)
    return {
        **metadata,
        "input_shape": [args.batch_size, 3, *args.shape],
        "precision": "fp32",
        "seed": args.seed,
        "samples": args.samples,
        "macs_per_batch": mean_macs,
        "macs_per_image": mean_macs / args.batch_size,
        "gflops_per_image": mean_macs / args.batch_size / 1e9,
        "convention": "One multiply-add counts as one operation; inference SRNL arithmetic",
        "records": records,
        "runtime": runtime_info(device),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--batch-size", type=int, default=1, help="Input batch size (default: 1)")
    parser.add_argument("--samples", type=int, default=1,
                        help="Number of synthetic batches to average (default: 1)")
    args = parser.parse_args(argv)
    validate_model_arguments(parser, args)
    if args.batch_size < 1 or args.samples < 1:
        parser.error("batch-size and samples must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    write_report(profile(args), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

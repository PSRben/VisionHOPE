"""Measure fast VisionHOPE inference throughput and peak CUDA memory."""

import argparse
import gc
import statistics

import torch

from .common import (add_model_arguments, check_output, configure_runtime, output_tensors,
                     precision_context, runtime_info, validate_model_arguments, write_report)
from .model_builder import build_model, make_forward, make_inputs


def timed_loop(forward, precision, iterations, graph=None):
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    if graph is None:
        with precision_context(precision):
            for _ in range(iterations):
                forward()
    else:
        for _ in range(iterations):
            graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def memory_start():
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    return torch.cuda.memory_allocated(), torch.cuda.memory_reserved()


def memory_report(baseline):
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return {
        "baseline_allocated_bytes": baseline[0],
        "baseline_reserved_bytes": baseline[1],
        "peak_allocated_bytes": peak,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "incremental_peak_allocated_bytes": peak - baseline[0],
        "peak_allocated_gb": peak / 1e9,
    }


def check_fast_path(model, forward, precision):
    """Verify query reuse and actual fused execution outside measured forwards."""
    from visionhope.inference.audit import ModelAudit
    audit = ModelAudit(model)
    try:
        with torch.inference_mode(), precision_context(precision), torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as profile:
            output = forward()
        check_output(output)
        evidence = audit.finish()
    finally:
        audit.close()
    events = profile.events()
    spatial = sum("srnl_line_token_parallel<" in event.name for event in events)
    sequence = sum("srnl_sequence_token_parallel<" in event.name for event in events)
    expected = sum(record["call_count"] for key in ("query_audit", "srnl_audit")
                   for record in evidence[key].values())
    if spatial + sequence != expected:
        raise RuntimeError(f"Expected {expected} fused SRNL launches, observed {spatial + sequence}")
    return {"fused_blocks": audit.block_count, "fused_operators": len(evidence["query_audit"]),
            "fused_srnl_calls": expected, "query_once": True if evidence["query_audit"] else None,
            "fp32_grn": True if evidence["grn_audit"] else None,
            "spatial_kernels": spatial, "sequence_kernels": sequence}


def validate_graph(actual, forward, precision):
    """Compare changed-input replay with an independent eager forward."""
    with precision_context(precision):
        expected = forward()
    check_output(expected)
    expected = [tensor.detach().float().cpu() for tensor in output_tensors(expected)]
    if len(actual) != len(expected):
        raise AssertionError("Graph and eager output structures differ")
    atol, rtol = (3e-5, 3e-4) if precision == "fp32" else (1e-3, 1e-2)
    differences = []
    for measured, reference in zip(actual, expected):
        torch.testing.assert_close(measured, reference, atol=atol, rtol=rtol)
        differences.append(float((measured - reference).abs().max()) if measured.numel() else 0.0)
    return {"changed_input": True, "max_abs": max(differences, default=0.0)}


def benchmark(args):
    from visionhope.inference.prepare import prepare_for_inference

    device = configure_runtime(args, tf32=args.tf32, cudnn_benchmark=args.cudnn_benchmark)
    model, metadata = build_model(args, device)
    if args.channels_last:
        model.to(memory_format=torch.channels_last)
    prepared = prepare_for_inference(model)
    inputs = make_inputs(args, device, channels_last=args.channels_last)
    forward = make_forward(model, inputs, args.task)

    with torch.inference_mode():
        with precision_context(args.precision):
            for _ in range(args.warmup):
                output = forward()
        torch.cuda.synchronize()
        check_output(output)
        del output
        gc.collect()
        torch.cuda.empty_cache()

        baseline = memory_start()
        with precision_context(args.precision):
            output = forward()
        eager_memory = memory_report(baseline)
        eager_memory["output_bytes"] = sum(
            tensor.numel() * tensor.element_size() for tensor in output_tensors(output))
        del output

        graph, graph_output, graph_memory = None, None, None
        if args.execution == "cuda_graph":
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph), precision_context(args.precision):
                graph_output = forward()
            for _ in range(max(3, args.warmup)):
                graph.replay()
            baseline = memory_start()
            graph.replay()
            graph_memory = memory_report(baseline)

        pilot_ms = timed_loop(forward, args.precision, min(3, args.min_iters), graph)
        iterations = round(args.target_seconds * 1000 / max(pilot_ms, 1e-6))
        iterations = max(args.min_iters, min(args.max_iters, iterations))
        trials = [timed_loop(forward, args.precision, iterations, graph)
                  for _ in range(args.trials)]
        graph_validation = None
        if graph is not None:
            inputs.neg_().add_(0.125)
            graph.replay()
            torch.cuda.synchronize()
            actual = [tensor.detach().float().cpu() for tensor in output_tensors(graph_output)]
            del graph_output, graph
            gc.collect()
            torch.cuda.empty_cache()
            graph_validation = validate_graph(actual, forward, args.precision)
        fast_path = check_fast_path(model, forward, args.precision)

    median_ms = statistics.median(trials)
    return {
        **metadata,
        "input_shape": list(inputs.shape),
        "precision": args.precision,
        "mode": "fused",
        "execution": args.execution,
        "channels_last": args.channels_last,
        "seed": args.seed,
        "warmup": args.warmup,
        "iterations_per_trial": iterations,
        "trial_ms_per_batch": trials,
        "median_ms_per_batch": median_ms,
        "throughput_img_s": args.batch_size * 1000 / median_ms,
        "memory": {"eager": eager_memory, "cuda_graph": graph_memory},
        "inference_preparation": prepared,
        "fast_path": fast_path,
        "graph_validation": graph_validation,
        "runtime": runtime_info(device),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--batch-size", type=int,
                        help="Images per batch; default: 8 for classification, 1 for dense tasks")
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"),
                        help="Inference precision; default: BF16 for classification, FP32 for dense tasks")
    parser.add_argument("--execution", choices=("eager", "cuda_graph"),
                        help="Execution mode; default: CUDA Graph for classification, eager for dense tasks")
    parser.add_argument("--channels-last", action=argparse.BooleanOptionalAction, default=True,
                        help="Use channels-last model and input storage")
    parser.add_argument("--cudnn-benchmark", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable cuDNN autotuning")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="Allow TF32 for FP32 matrix products and convolutions")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup forwards before measurement")
    parser.add_argument("--target-seconds", type=float, default=1.5, help="Target duration per timing trial")
    parser.add_argument("--min-iters", type=int, default=3, help="Minimum forwards per trial")
    parser.add_argument("--max-iters", type=int, default=100, help="Maximum forwards per trial")
    parser.add_argument("--trials", type=int, default=3, help="Timing trials; report median latency")
    args = parser.parse_args(argv)
    validate_model_arguments(parser, args)
    classification = args.task == "classification"
    if args.batch_size is None:
        args.batch_size = 8 if classification else 1
    args.precision = args.precision or ("bf16" if classification else "fp32")
    args.execution = args.execution or ("cuda_graph" if classification else "eager")
    if args.task == "detection" and args.execution == "cuda_graph":
        parser.error("Mask R-CNN has dynamic RPN proposals; use --execution eager with fused SRNL")
    if (args.batch_size < 1 or args.warmup < 1 or args.target_seconds <= 0
            or args.min_iters < 1 or args.max_iters < args.min_iters or args.trials < 1):
        parser.error("invalid batch size, warmup, or timing settings")
    return args


def main(argv=None):
    args = parse_args(argv)
    write_report(benchmark(args), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

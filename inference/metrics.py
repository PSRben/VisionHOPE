"""Record unrounded COCO and ADE20K evaluator outputs."""

from contextlib import contextmanager


@contextmanager
def capture_raw_metrics(task):
    records = []
    if task == "detection":
        from pycocotools.cocoeval import COCOeval
        original = COCOeval.summarize

        def summarize(evaluator):
            original(evaluator)
            records.append({"type": evaluator.params.iouType, "stats": evaluator.stats.tolist(),
                            "images": len(evaluator.params.imgIds)})

        COCOeval.summarize = summarize
        try:
            yield records
        finally:
            COCOeval.summarize = original
    elif task == "segmentation":
        from mmseg.evaluation.metrics import IoUMetric
        original = IoUMetric.total_area_to_metrics

        def total_area(*args, **kwargs):
            values = original(*args, **kwargs)
            records.append({key: value.tolist() for key, value in values.items()})
            return values

        IoUMetric.total_area_to_metrics = staticmethod(total_area)
        try:
            yield records
        finally:
            IoUMetric.total_area_to_metrics = staticmethod(original)
    else:
        raise ValueError(f"Unsupported evaluator {task}")

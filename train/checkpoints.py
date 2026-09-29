"""Keep selected clean weights and a separate complete training state."""

import math
from pathlib import Path
import shutil
import tempfile
import uuid

from visionhope.tools.checkpoint import save_weights


def atomic_copy(source, destination):
    destination = Path(destination)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
            temporary = Path(stream.name)
        shutil.copyfile(source, temporary)
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def classification_score(validation, ema_validation, source="auto"):
    """Select the best available Top-1 and its actual saved weight dictionary."""
    if source not in {"auto", "raw", "ema"}:
        raise ValueError("checkpoint_metric must be auto, raw, or ema")
    candidates = []
    for name, metrics in (("raw", validation), ("ema", ema_validation)):
        if source not in {"auto", name} or metrics is None:
            continue
        score = float(metrics["top1"])
        if math.isfinite(score):
            candidates.append((score, "state_dict_ema" if name == "ema" else "state_dict"))
    return max(candidates, key=lambda item: item[0]) if candidates else (None, "state_dict")


class CheckpointManager:
    """Rank clean weights by validation and keep one complete resume state.

    The writer receives a temporary path and metadata to add to its complete
    training state. Weight files dropped from the ranking are removed only
    after saving succeeds.
    """

    def __init__(self, directory, keep=5, metric="top1:auto"):
        if type(keep) is not int or keep < 1:
            raise ValueError("save_top_k must be a positive integer")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.keep = keep
        self.metric = metric
        self.entries = []

    def restore(self, state, source_directory):
        if state["metric"] != self.metric:
            raise ValueError("Resume must use the same checkpoint ranking metric")
        restored = []
        for record in state["entries"]:
            name = record["file"]
            if Path(name).name != name or not name.startswith("checkpoint_"):
                raise ValueError("Invalid ranked checkpoint filename")
            if not math.isfinite(record["score"]):
                raise ValueError("Invalid ranked checkpoint score")
            source = Path(source_directory) / name
            # Skip entries whose checkpoint files no longer exist.
            if not source.is_file():
                continue
            restored.append(dict(record))
        restored.sort(key=lambda row: (-row["score"], row["step"]))
        entries = restored[:self.keep]
        for record in entries:
            source = Path(source_directory) / record["file"]
            destination = self.directory / record["file"]
            if source.resolve() != destination.resolve():
                atomic_copy(source, destination)
        self.entries = entries
        if self.entries:
            atomic_copy(self.directory / self.entries[0]["file"], self.directory / "best.pth")
        if Path(source_directory).resolve() == self.directory.resolve():
            for record in restored[self.keep:]:
                (self.directory / record["file"]).unlink(missing_ok=True)

    def save(self, writer, *, weights, step, score=None, field="state_dict"):
        if type(step) is not int or step < 0:
            raise ValueError("Checkpoint step must be a nonnegative integer")
        if field not in {"state_dict", "state_dict_ema"}:
            raise ValueError("Unsupported checkpoint weight field")
        score = None if score is None or not math.isfinite(float(score)) else float(score)
        previous = list(self.entries)
        entries = list(previous)
        candidate = None
        if score is not None:
            candidate = {"file": f"checkpoint_{step:06d}_{uuid.uuid4().hex[:8]}.pth",
                         "step": step, "score": score, "field": field}
            entries.append(candidate)
        entries = sorted(entries, key=lambda row: (-row["score"], row["step"]))[:self.keep]
        ranked = candidate is not None and candidate in entries
        metadata = {
            "checkpoint_queue": {"metric": self.metric, "entries": entries},
            "checkpoint_selection": {"metric": self.metric, "score": score,
                                     "field": field, "step": step},
        }
        temporary = weight_file = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.directory, suffix=".pth", delete=False) as stream:
                temporary = Path(stream.name)
            writer(temporary, metadata)
            with tempfile.NamedTemporaryFile(dir=self.directory, suffix=".pth", delete=False) as stream:
                weight_file = Path(stream.name)
            save_weights(weight_file, weights)
            if ranked:
                atomic_copy(weight_file, self.directory / candidate["file"])
            weight_file.replace(self.directory / "last.pth")
            if entries:
                atomic_copy(self.directory / entries[0]["file"], self.directory / "best.pth")
            temporary.replace(self.directory / "training_state.pth")
            self.entries = entries
            retained = {row["file"] for row in entries}
            for row in previous:
                if row["file"] not in retained:
                    (self.directory / row["file"]).unlink(missing_ok=True)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            if weight_file is not None:
                weight_file.unlink(missing_ok=True)

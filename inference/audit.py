"""Check query projection calls and GRN precision during inference."""

import torch

from visionhope.models.block import VisionHOPEBlock
from visionhope.models.operator import VisionHOPEOperator
from visionhope.models.srnl import SRNL
from visionhope.utils.grn import GlobalResponseNorm


class ModelAudit:
    def __init__(self, model):
        self.handles = []
        self.query = {}
        self.grn = {}
        self.srnl = {}
        self.shapes = {}
        self.operator_calls = {}
        self.expected_queries = set()
        self.expected_grn = set()
        self.expected_srnl = set()
        modules = list(model.named_modules())
        owners = {id(module.operator): (name, module) for name, module in modules
                  if isinstance(module, VisionHOPEBlock)}
        nested_srnl = {id(module.srnl) for _, module in modules
                       if isinstance(module, VisionHOPEOperator)}
        self.block_count = len(owners)
        for name, module in modules:
            if isinstance(module, VisionHOPEOperator):
                owner_name, owner = owners.get(id(module), (name, module))
                self.expected_queries.add(owner_name)
                self.handles.append(owner.register_forward_pre_hook(
                    self._input_hook(owner_name), with_kwargs=True))
                self.handles.append(module.W_q.register_forward_pre_hook(self._query_hook(owner_name)))
            if isinstance(module, SRNL) and id(module) not in nested_srnl:
                self.expected_srnl.add(name)
                self.handles.append(module.register_forward_pre_hook(
                    self._srnl_hook(name), with_kwargs=True))
            if isinstance(module, GlobalResponseNorm):
                self.expected_grn.add(name)
                self.handles.append(module.register_forward_hook(self._grn_hook(name)))

    def _input_hook(self, name):
        def hook(module, inputs, kwargs):
            x = inputs[0] if inputs else kwargs["x"]
            self.shapes[name] = list(x.shape)
            self.operator_calls[name] = self.operator_calls.get(name, 0) + 1
        return hook

    def _query_hook(self, name):
        def hook(module, inputs):
            shape = self.shapes[name]
            projected = inputs[0].numel() // inputs[0].shape[-1]
            spatial = shape[0] * shape[-2] * shape[-1]
            if projected != spatial:
                raise RuntimeError(f"Query-once audit failed for {name}: {projected} != {spatial}")
            previous = self.query.get(name, {"call_count": 0, "projected_tokens": 0, "spatial_tokens": 0})
            self.query[name] = {"input_shape": shape, "query_input": list(inputs[0].shape),
                                "call_count": previous["call_count"] + 1,
                                "projected_tokens": previous["projected_tokens"] + projected,
                                "spatial_tokens": previous["spatial_tokens"] + spatial}
        return hook

    def _srnl_hook(self, name):
        def hook(module, inputs, kwargs):
            x = inputs[0] if inputs else kwargs["x"]
            previous = self.srnl.get(name, {"call_count": 0})
            self.srnl[name] = {"input_shape": list(x.shape),
                               "call_count": previous["call_count"] + 1}
        return hook

    def _grn_hook(self, name):
        def hook(module, inputs, output):
            if output.dtype != torch.float32:
                raise RuntimeError(f"GRN output must be FP32: {name} returned {output.dtype}")
            self.grn[name] = {"input_dtype": str(inputs[0].dtype), "output_dtype": str(output.dtype),
                              "shape": list(inputs[0].shape)}
        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def finish(self):
        self.close()
        if not self.expected_queries and not self.expected_srnl:
            raise RuntimeError("No VisionHOPE operator or SRNL instance was found")
        if set(self.query) != self.expected_queries:
            raise RuntimeError("Not every VisionHOPE query projection was observed during evaluation")
        if set(self.srnl) != self.expected_srnl:
            raise RuntimeError("Not every standalone SRNL instance was observed during evaluation")
        if set(self.grn) != self.expected_grn:
            raise RuntimeError("Not every GRN module was observed during evaluation")
        if any(record["call_count"] != self.operator_calls[name]
               or record["projected_tokens"] != record["spatial_tokens"]
               for name, record in self.query.items()):
            raise RuntimeError("A VisionHOPE operator did not project its input exactly once per call")
        return {"query_audit": self.query, "grn_audit": self.grn, "srnl_audit": self.srnl}

"""Model initialization and classifier utilities."""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from timm.models.layers import trunc_normal_


def initialize_weights(module):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


class ClassificationModel(nn.Module):
    """Classifier access and activation checkpointing for VisionHOPE models."""

    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = bool(enable)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=None):
        if global_pool not in (None, "avg"):
            raise ValueError("VisionHOPE uses global average pooling")
        self.num_classes = num_classes
        self.head = nn.Linear(self.num_features, num_classes)
        initialize_weights(self.head)

    def _run_blocks(self, blocks, x):
        if self.grad_checkpointing and self.training and torch.is_grad_enabled():
            for block in blocks:
                x = checkpoint(block, x, use_reentrant=False)
            return x
        return blocks(x)

    def train(self, mode=True):
        if mode and getattr(self, "_inference_prepared", False):
            raise RuntimeError("This model has folded inference weights; train an unfused model")
        return super().train(mode)

    def forward(self, x):
        return self.head(self.forward_features(x))

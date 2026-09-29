"""RoIAlign with the dtype and memory layout required by the MMCV operator."""

import torch
from mmdet.models.roi_heads.roi_extractors import SingleRoIExtractor
from mmdet.registry import MODELS


@MODELS.register_module()
class ContiguousRoIExtractor(SingleRoIExtractor):
    """Perform RoIAlign on FP32 contiguous NCHW features under any AMP policy."""

    @staticmethod
    def _prepare(feature):
        # Convert features to contiguous FP32 NCHW format.
        return (feature if feature.dtype == torch.float32 else feature.float()).contiguous()

    def forward(self, feats, rois, roi_scale_factor=None):
        with torch.autocast("cuda", enabled=False):
            rois = rois.float()
            output_size = self.roi_layers[0].output_size
            features = feats[0].new_zeros(
                rois.size(0), self.out_channels, *output_size, dtype=torch.float32
            )
            if len(feats) == 1:
                if len(rois) == 0:
                    return features
                return self.roi_layers[0](self._prepare(feats[0]), rois)
            target_levels = self.map_roi_levels(rois, len(feats))
            if roi_scale_factor is not None:
                rois = self.roi_rescale(rois, roi_scale_factor)
            for level, input_feature in enumerate(feats):
                indices = (target_levels == level).nonzero(as_tuple=False).squeeze(1)
                feature = self._prepare(input_feature)
                if indices.numel() > 0:
                    features[indices] = self.roi_layers[level](feature, rois[indices])
                else:
                    features += (
                        sum(p.view(-1)[0] for p in self.parameters()) * 0.0
                        + feature.sum() * 0.0
                    )
            return features

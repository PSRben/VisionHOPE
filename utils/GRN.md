# Global Response Normalization

`GlobalResponseNorm` accepts NCHW features and returns FP32 output of the same shape:

```python
import torch
from visionhope.utils import GlobalResponseNorm

grn = GlobalResponseNorm(64).cuda()
y = grn(torch.randn(2, 64, 14, 20, device="cuda"))
```

See the main README for [VisionHOPE components](../README.md#srnl-and-visionhope-operator) and [fast inference](../README.md#fast-inference).

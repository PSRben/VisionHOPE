
torch::Tensor visionhope_postprocess_scale_forward(torch::Tensor Y, torch::Tensor X, torch::Tensor DSkip, torch::Tensor Scale, int H, int W, bool ChannelsLast, bool OutBf16);
std::vector<torch::Tensor> visionhope_postprocess_scale_backward(torch::Tensor DOut, torch::Tensor Y, torch::Tensor X, torch::Tensor DSkip, torch::Tensor Scale, int H, int W, bool ChannelsLast, bool UseLTile, int TileL);

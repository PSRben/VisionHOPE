
std::vector<torch::Tensor> visionhope_qnorm_forward(torch::Tensor QRaw, int NumHeads, int D, bool UseTile);
torch::Tensor visionhope_qnorm_backward(torch::Tensor DQOut, torch::Tensor QOut, torch::Tensor InvNorm, int NumHeads, int D, bool UseTile);

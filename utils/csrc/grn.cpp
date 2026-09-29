// Shared GRN forward/backward bindings. See ../GRN.md.
#include <torch/extension.h>

void grn_forward_cuda(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta,
                      torch::Tensor y, torch::Tensor gx, torch::Tensor inv_den,
                      double eps);
void grn_backward_cuda(torch::Tensor dout, torch::Tensor x, torch::Tensor gamma,
                       torch::Tensor gx, torch::Tensor inv_den, torch::Tensor s,
                       torch::Tensor dgamma_bc, torch::Tensor dbeta_bc,
                       torch::Tensor sum_sg, torch::Tensor dx,
                       torch::Tensor dgamma, torch::Tensor dbeta);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("grn_forward_cuda", &grn_forward_cuda, "SRNL GRN forward CUDA");
    m.def("grn_backward_cuda", &grn_backward_cuda, "SRNL GRN backward CUDA");
}

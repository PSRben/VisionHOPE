// One CUDA backend for training and inference; FP32 statistics and output.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

template <typename T>
__device__ __forceinline__ float to_float(T v) {
    return static_cast<float>(v);
}

template <typename T>
__device__ __forceinline__ T from_float(float v) {
    return static_cast<T>(v);
}

template <>
__device__ __forceinline__ at::Half from_float<at::Half>(float v) {
    return at::Half(v);
}

template <>
__device__ __forceinline__ at::BFloat16 from_float<at::BFloat16>(float v) {
    return at::BFloat16(v);
}

__device__ __forceinline__ float block_reduce_sum(float v) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    smem[tid] = v;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    return smem[0];
}

template <typename scalar_t>
__global__ void grn_stats_tiled_kernel(
    const scalar_t* __restrict__ x,
    float* __restrict__ gx,
    int B, int C, int S)
{
    constexpr int TILE_C = 16;
    constexpr int TILE_S = 16;
    int tile_c = blockIdx.x;
    int b = blockIdx.y;
    int lane = threadIdx.x;
    int c_lane = lane & (TILE_C - 1);
    int s_lane = lane >> 4;
    int c = tile_c * TILE_C + c_lane;
    __shared__ float smem[TILE_C * TILE_S];

    float sumsq = 0.0f;
    if (c < C) {
        int base = b * S * C + c;
        for (int s = s_lane; s < S; s += TILE_S) {
            float xv = to_float(x[base + s * C]);
            sumsq += xv * xv;
        }
    }
    smem[s_lane * TILE_C + c_lane] = sumsq;
    __syncthreads();

    #pragma unroll
    for (int offset = TILE_S >> 1; offset > 0; offset >>= 1) {
        if (s_lane < offset) {
            smem[s_lane * TILE_C + c_lane] += smem[(s_lane + offset) * TILE_C + c_lane];
        }
        __syncthreads();
    }
    if (s_lane == 0 && c < C) {
        gx[b * C + c] = sqrtf(smem[c_lane]);
    }
}

__global__ void grn_inv_den_kernel(
    const float* __restrict__ gx,
    float* __restrict__ inv_den,
    int B, int C, float eps)
{
    int b = blockIdx.x;
    float sum = 0.0f;
    int base = b * C;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        sum += gx[base + c];
    }
    float total = block_reduce_sum(sum);
    if (threadIdx.x == 0) {
        float mean = total / static_cast<float>(C);
        inv_den[b] = 1.0f / (mean + eps);
    }
}

template <typename scalar_t>
__global__ void grn_forward_kernel(
    const scalar_t* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ gx,
    const float* __restrict__ inv_den,
    float* __restrict__ y,
    int B, int C, int S, int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; idx < total; idx += stride) {
        int c = idx % C;
        int sc = idx / C;
        int b = sc / S;
        int bc = b * C + c;
        float xv = to_float(x[idx]);
        float nx = gx[bc] * inv_den[b];
        y[idx] = xv * (1.0f + gamma[c] * nx) + beta[c];
    }
}

template <typename scalar_t, typename dout_t>
__global__ void grn_backward_stats_tiled_kernel(
    const dout_t* __restrict__ dout,
    const scalar_t* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ gx,
    const float* __restrict__ inv_den,
    float* __restrict__ s_out,
    float* __restrict__ dgamma_bc,
    float* __restrict__ dbeta_bc,
    int B, int C, int S)
{
    constexpr int TILE_C = 16;
    constexpr int TILE_S = 16;
    int tile_c = blockIdx.x;
    int b = blockIdx.y;
    int lane = threadIdx.x;
    int c_lane = lane & (TILE_C - 1);
    int s_lane = lane >> 4;
    int c = tile_c * TILE_C + c_lane;
    int bc = b * C + c;
    __shared__ float s_smem[TILE_C * TILE_S];
    __shared__ float dgamma_smem[TILE_C * TILE_S];
    __shared__ float dbeta_smem[TILE_C * TILE_S];

    float s_val = 0.0f;
    float dgamma_val = 0.0f;
    float dbeta_val = 0.0f;
    if (c < C) {
        int base = b * S * C + c;
        float g = gamma[c];
        float nx = gx[bc] * inv_den[b];
        for (int sidx = s_lane; sidx < S; sidx += TILE_S) {
            int off = base + sidx * C;
            float dy = to_float(dout[off]);
            float xv = to_float(x[off]);
            float dyx = dy * xv;
            s_val += dyx * g;
            dgamma_val += dyx * nx;
            dbeta_val += dy;
        }
    }

    int smem_idx = s_lane * TILE_C + c_lane;
    s_smem[smem_idx] = s_val;
    dgamma_smem[smem_idx] = dgamma_val;
    dbeta_smem[smem_idx] = dbeta_val;
    __syncthreads();

    #pragma unroll
    for (int offset = TILE_S >> 1; offset > 0; offset >>= 1) {
        if (s_lane < offset) {
            int dst = s_lane * TILE_C + c_lane;
            int src = (s_lane + offset) * TILE_C + c_lane;
            s_smem[dst] += s_smem[src];
            dgamma_smem[dst] += dgamma_smem[src];
            dbeta_smem[dst] += dbeta_smem[src];
        }
        __syncthreads();
    }
    if (s_lane == 0 && c < C) {
        s_out[bc] = s_smem[c_lane];
        dgamma_bc[bc] = dgamma_smem[c_lane];
        dbeta_bc[bc] = dbeta_smem[c_lane];
    }
}

__global__ void grn_sum_sg_kernel(
    const float* __restrict__ s_in,
    const float* __restrict__ gx,
    float* __restrict__ sum_sg,
    int B, int C)
{
    int b = blockIdx.x;
    int base = b * C;
    float sum = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        sum += s_in[base + c] * gx[base + c];
    }
    float total = block_reduce_sum(sum);
    if (threadIdx.x == 0) {
        sum_sg[b] = total;
    }
}

__global__ void grn_param_reduce_kernel(
    const float* __restrict__ dgamma_bc,
    const float* __restrict__ dbeta_bc,
    float* __restrict__ dgamma,
    float* __restrict__ dbeta,
    int B, int C)
{
    int c = blockIdx.x;
    float dgamma_sum = 0.0f;
    float dbeta_sum = 0.0f;
    for (int b = threadIdx.x; b < B; b += blockDim.x) {
        int idx = b * C + c;
        dgamma_sum += dgamma_bc[idx];
        dbeta_sum += dbeta_bc[idx];
    }
    float dgamma_total = block_reduce_sum(dgamma_sum);
    // Synchronize readers before reusing smem[0].
    __syncthreads();
    float dbeta_total = block_reduce_sum(dbeta_sum);
    if (threadIdx.x == 0) {
        dgamma[c] = dgamma_total;
        dbeta[c] = dbeta_total;
    }
}

template <typename scalar_t, typename dout_t>
__global__ void grn_dx_kernel(
    const dout_t* __restrict__ dout,
    const scalar_t* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ gx,
    const float* __restrict__ inv_den,
    const float* __restrict__ s_in,
    const float* __restrict__ sum_sg,
    scalar_t* __restrict__ dx,
    int B, int C, int S, int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; idx < total; idx += stride) {
        int c = idx % C;
        int sc = idx / C;
        int b = sc / S;
        int bc = b * C + c;
        float dy = to_float(dout[idx]);
        float xv = to_float(x[idx]);
        float gxc = gx[bc];
        float inv = inv_den[b];
        float nx = gxc * inv;
        float d_gx = s_in[bc] * inv - (sum_sg[b] * inv * inv) / static_cast<float>(C);
        float inv_gx = (gxc > 0.0f) ? (1.0f / gxc) : 0.0f;
        float dxv = dy * (1.0f + gamma[c] * nx) + d_gx * xv * inv_gx;
        dx[idx] = from_float<scalar_t>(dxv);
    }
}

int pick_threads(int n) {
    if (n >= 256) return 256;
    if (n >= 128) return 128;
    if (n >= 64) return 64;
    return 32;
}

void check_input(const torch::Tensor& x, const torch::Tensor& gamma) {
    TORCH_CHECK(x.is_cuda() && x.dim() == 4 && x.is_contiguous(at::MemoryFormat::ChannelsLast),
                "GRN CUDA expects channels-last NCHW input");
    TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
                "GRN CUDA supports FP32, FP16 and BF16 input");
    TORCH_CHECK(x.size(0) > 0 && x.size(0) <= 65535 && x.size(1) > 0 && x.size(2) > 0 && x.size(3) > 0,
                "GRN CUDA expects nonempty input and batch <= 65535");
    TORCH_CHECK(x.numel() <= 2147483647LL - 4096 * 256,
                "GRN CUDA input exceeds the current 32-bit index limit");
    TORCH_CHECK(gamma.device() == x.device() && gamma.scalar_type() == at::kFloat &&
                gamma.is_contiguous() && gamma.numel() == x.size(1),
                "GRN CUDA requires contiguous FP32 channel parameters on the input device");
}

} // namespace

void grn_forward_cuda(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta,
                      torch::Tensor y, torch::Tensor gx, torch::Tensor inv_den,
                      double eps) {
    check_input(x, gamma);
    const c10::cuda::CUDAGuard device_guard(x.device());
    TORCH_CHECK(beta.device() == x.device() && beta.scalar_type() == at::kFloat &&
                beta.is_contiguous() && beta.numel() == x.size(1), "Invalid GRN beta");
    TORCH_CHECK(y.scalar_type() == at::kFloat, "GRN output must be FP32");
    int B = static_cast<int>(x.size(0));
    int C = static_cast<int>(x.size(1));
    int S = static_cast<int>(x.size(2) * x.size(3));
    int total = B * C * S;
    auto stream = at::cuda::getCurrentCUDAStream();
    int reduce_threads = pick_threads(C);
    int elem_threads = 256;
    int elem_blocks = std::min((total + elem_threads - 1) / elem_threads, 4096);
    size_t reduce_smem = reduce_threads * sizeof(float);
    dim3 stat_grid((C + 15) / 16, B);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "grn_forward_cuda", [&] {
        grn_stats_tiled_kernel<scalar_t><<<stat_grid, 256, 0, stream>>>(
            x.data_ptr<scalar_t>(), gx.data_ptr<float>(), B, C, S);
        grn_inv_den_kernel<<<B, reduce_threads, reduce_smem, stream>>>(
            gx.data_ptr<float>(), inv_den.data_ptr<float>(), B, C, static_cast<float>(eps));
        grn_forward_kernel<scalar_t><<<elem_blocks, elem_threads, 0, stream>>>(
            x.data_ptr<scalar_t>(), gamma.data_ptr<float>(), beta.data_ptr<float>(),
            gx.data_ptr<float>(), inv_den.data_ptr<float>(), y.data_ptr<float>(),
            B, C, S, total);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void grn_backward_cuda(torch::Tensor dout, torch::Tensor x, torch::Tensor gamma,
                       torch::Tensor gx, torch::Tensor inv_den, torch::Tensor s,
                       torch::Tensor dgamma_bc, torch::Tensor dbeta_bc,
                       torch::Tensor sum_sg, torch::Tensor dx,
                       torch::Tensor dgamma, torch::Tensor dbeta) {
    check_input(x, gamma);
    const c10::cuda::CUDAGuard device_guard(x.device());
    TORCH_CHECK(dout.scalar_type() == at::kFloat, "GRN output gradient must be FP32");
    int B = static_cast<int>(x.size(0));
    int C = static_cast<int>(x.size(1));
    int S = static_cast<int>(x.size(2) * x.size(3));
    int total = B * C * S;
    auto stream = at::cuda::getCurrentCUDAStream();
    int reduce_threads = pick_threads(C);
    int batch_threads = pick_threads(B);
    int elem_threads = 256;
    int elem_blocks = std::min((total + elem_threads - 1) / elem_threads, 4096);
    size_t reduce_smem = reduce_threads * sizeof(float);
    size_t batch_smem = batch_threads * sizeof(float);
    dim3 stat_grid((C + 15) / 16, B);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "grn_backward_cuda", [&] {
        grn_backward_stats_tiled_kernel<scalar_t, float><<<stat_grid, 256, 0, stream>>>(
            dout.data_ptr<float>(), x.data_ptr<scalar_t>(), gamma.data_ptr<float>(),
            gx.data_ptr<float>(), inv_den.data_ptr<float>(), s.data_ptr<float>(),
            dgamma_bc.data_ptr<float>(), dbeta_bc.data_ptr<float>(), B, C, S);
        grn_sum_sg_kernel<<<B, reduce_threads, reduce_smem, stream>>>(
            s.data_ptr<float>(), gx.data_ptr<float>(), sum_sg.data_ptr<float>(), B, C);
        grn_param_reduce_kernel<<<C, batch_threads, batch_smem, stream>>>(
            dgamma_bc.data_ptr<float>(), dbeta_bc.data_ptr<float>(),
            dgamma.data_ptr<float>(), dbeta.data_ptr<float>(), B, C);
        grn_dx_kernel<scalar_t, float><<<elem_blocks, elem_threads, 0, stream>>>(
            dout.data_ptr<float>(), x.data_ptr<scalar_t>(), gamma.data_ptr<float>(),
            gx.data_ptr<float>(), inv_den.data_ptr<float>(), s.data_ptr<float>(),
            sum_sg.data_ptr<float>(), dx.data_ptr<scalar_t>(), B, C, S, total);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

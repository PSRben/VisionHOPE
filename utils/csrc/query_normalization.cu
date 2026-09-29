
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>

#define QNORM_EPS 1e-12f

__forceinline__ __device__ float halfwarp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffffu, val, offset, 16);
    }
    return __shfl_sync(0xffffffffu, val, 0, 16);
}

__global__ void visionhope_qnorm_forward_kernel(
    const __nv_bfloat16* __restrict__ QRaw,
    float* __restrict__ QOut,
    float* __restrict__ InvNorm,
    int B, int Directions, int L, int NumHeads, int D)
{
    int tid = threadIdx.x;
    int64_t vec = blockIdx.x;
    int head = vec % NumHeads;
    int l = (vec / NumHeads) % L;
    int dir = (vec / (NumHeads * L)) % Directions;
    int b = vec / ((int64_t)Directions * L * NumHeads);
    int C = NumHeads * D;
    __shared__ float s_reduce[64];
    __shared__ float s_inv;

    float v = 0.0f;
    if (tid < D) {
        int64_t in_idx = (((int64_t)b * Directions + dir) * L + l) * C + head * D + tid;
        v = __bfloat162float(QRaw[in_idx]);
    }

    float ss = v * v;
    if (D <= 32) {
        unsigned int mask = 0xffffffffu;
        for (int offset = 16; offset > 0; offset >>= 1) {
            ss += __shfl_down_sync(mask, ss, offset);
        }
        ss = __shfl_sync(mask, ss, 0);
    } else {
        s_reduce[tid] = ss;
        __syncthreads();
        if (tid < 32) {
            ss = s_reduce[tid] + s_reduce[tid + 32];
            for (int offset = 16; offset > 0; offset >>= 1) {
                ss += __shfl_down_sync(0xffffffffu, ss, offset);
            }
            ss = __shfl_sync(0xffffffffu, ss, 0);
            if (tid == 0) {
                s_reduce[0] = ss;
            }
        }
        __syncthreads();
        ss = s_reduce[0];
    }
    float norm = sqrtf(ss);
    float denom = fmaxf(norm, QNORM_EPS);
    float inv = 1.0f / denom;
    if (tid == 0) {
        InvNorm[vec] = inv;
        s_inv = inv;
    }
    __syncthreads();
    inv = (D <= 32) ? inv : s_inv;
    if (tid < D) {
        int64_t out_idx = ((((int64_t)b * Directions + dir) * NumHeads + head) * L + l) * D + tid;
        QOut[out_idx] = v * inv;
    }
}

__global__ void visionhope_qnorm_backward_kernel(
    const float* __restrict__ DQOut,
    const float* __restrict__ QOut,
    const float* __restrict__ InvNorm,
    __nv_bfloat16* __restrict__ DQRaw,
    int B, int Directions, int L, int NumHeads, int D)
{
    int tid = threadIdx.x;
    int64_t vec = blockIdx.x;
    int head = vec % NumHeads;
    int l = (vec / NumHeads) % L;
    int dir = (vec / (NumHeads * L)) % Directions;
    int b = vec / ((int64_t)Directions * L * NumHeads);
    int C = NumHeads * D;
    __shared__ float s_reduce[64];

    float g = 0.0f;
    float q = 0.0f;
    if (tid < D) {
        int64_t out_idx = ((((int64_t)b * Directions + dir) * NumHeads + head) * L + l) * D + tid;
        g = DQOut[out_idx];
        q = QOut[out_idx];
    }

    float dot_part = g * q;
    if (D <= 32) {
        unsigned int mask = 0xffffffffu;
        for (int offset = 16; offset > 0; offset >>= 1) {
            dot_part += __shfl_down_sync(mask, dot_part, offset);
        }
        dot_part = __shfl_sync(mask, dot_part, 0);
    } else {
        s_reduce[tid] = dot_part;
        __syncthreads();
        if (tid < 32) {
            dot_part = s_reduce[tid] + s_reduce[tid + 32];
            for (int offset = 16; offset > 0; offset >>= 1) {
                dot_part += __shfl_down_sync(0xffffffffu, dot_part, offset);
            }
            dot_part = __shfl_sync(0xffffffffu, dot_part, 0);
            if (tid == 0) {
                s_reduce[0] = dot_part;
            }
        }
        __syncthreads();
        dot_part = s_reduce[0];
    }
    float dot = dot_part;
    float inv = InvNorm[vec];
    float grad = (inv > 9.999e11f) ? (g * inv) : ((g - q * dot) * inv);

    if (tid < D) {
        int64_t in_idx = (((int64_t)b * Directions + dir) * L + l) * C + head * D + tid;
        DQRaw[in_idx] = __float2bfloat16(grad);
    }
}

template <int D>
__global__ void visionhope_qnorm_forward_tile_kernel(
    const __nv_bfloat16* __restrict__ QRaw,
    float* __restrict__ QOut,
    float* __restrict__ InvNorm,
    int B, int Directions, int L, int NumHeads, int64_t NumVec)
{
    constexpr int Width = D < 32 ? D : 32;
    const int Groups = blockDim.x / Width;
    int group = threadIdx.x / Width;
    int lane = threadIdx.x % Width;
    int64_t vec = (int64_t)blockIdx.x * Groups + group;
    if (vec >= NumVec) return;

    int head = vec % NumHeads;
    int l = (vec / NumHeads) % L;
    int dir = (vec / ((int64_t)NumHeads * L)) % Directions;
    int b = vec / ((int64_t)Directions * L * NumHeads);
    int C = NumHeads * D;

    int64_t in_idx = (((int64_t)b * Directions + dir) * L + l) * C + head * D + lane;
    float v = __bfloat162float(QRaw[in_idx]);
    float v_hi = 0.0f;
    if constexpr (D > 32) v_hi = __bfloat162float(QRaw[in_idx + 32]);
    float sumsq = v * v;
    // Round each product before addition.
    if constexpr (D > 32) sumsq = __fmul_rn(v, v) + __fmul_rn(v_hi, v_hi);
    unsigned mask = Width == 32 ? 0xffffffffu : ((1u << Width) - 1u) << ((group * Width) & 31);
    #pragma unroll
    for (int offset = Width / 2; offset > 0; offset /= 2)
        sumsq += __shfl_down_sync(mask, sumsq, offset, Width);
    sumsq = __shfl_sync(mask, sumsq, 0, Width);
    float norm = sqrtf(sumsq);
    float denom = fmaxf(norm, QNORM_EPS);
    float inv = 1.0f / denom;
    if (lane == 0) {
        InvNorm[vec] = inv;
    }

    int64_t out_idx = ((((int64_t)b * Directions + dir) * NumHeads + head) * L + l) * D + lane;
    QOut[out_idx] = v * inv;
    if constexpr (D > 32) QOut[out_idx + 32] = v_hi * inv;
}

template <int D>
__global__ void visionhope_qnorm_backward_tile_kernel(
    const float* __restrict__ DQOut,
    const float* __restrict__ QOut,
    const float* __restrict__ InvNorm,
    __nv_bfloat16* __restrict__ DQRaw,
    int B, int Directions, int L, int NumHeads, int64_t NumVec)
{
    constexpr int Width = D < 32 ? D : 32;
    const int Groups = blockDim.x / Width;
    int group = threadIdx.x / Width;
    int lane = threadIdx.x % Width;
    int64_t vec = (int64_t)blockIdx.x * Groups + group;
    if (vec >= NumVec) return;

    int head = vec % NumHeads;
    int l = (vec / NumHeads) % L;
    int dir = (vec / ((int64_t)NumHeads * L)) % Directions;
    int b = vec / ((int64_t)Directions * L * NumHeads);
    int C = NumHeads * D;

    int64_t out_idx = ((((int64_t)b * Directions + dir) * NumHeads + head) * L + l) * D + lane;
    float g = DQOut[out_idx];
    float q = QOut[out_idx];
    float g_hi = 0.0f, q_hi = 0.0f;
    if constexpr (D > 32) { g_hi = DQOut[out_idx + 32]; q_hi = QOut[out_idx + 32]; }
    float dot = g * q;
    if constexpr (D > 32) dot = __fmul_rn(g, q) + __fmul_rn(g_hi, q_hi);
    unsigned mask = Width == 32 ? 0xffffffffu : ((1u << Width) - 1u) << ((group * Width) & 31);
    #pragma unroll
    for (int offset = Width / 2; offset > 0; offset /= 2)
        dot += __shfl_down_sync(mask, dot, offset, Width);
    dot = __shfl_sync(mask, dot, 0, Width);
    float inv = InvNorm[vec];
    float grad = (inv > 9.999e11f) ? (g * inv) : ((g - q * dot) * inv);

    int64_t in_idx = (((int64_t)b * Directions + dir) * L + l) * C + head * D + lane;
    DQRaw[in_idx] = __float2bfloat16(grad);
    if constexpr (D > 32) {
        float grad_hi = (inv > 9.999e11f) ? (g_hi * inv) : ((g_hi - q_hi * dot) * inv);
        DQRaw[in_idx + 32] = __float2bfloat16(grad_hi);
    }
}

std::vector<torch::Tensor> visionhope_qnorm_forward(torch::Tensor QRaw, int NumHeads, int D, bool UseTile) {
    if (QRaw.scalar_type() != at::ScalarType::BFloat16) {
        throw std::runtime_error("visionhope_qnorm_forward expects bf16 QRaw.");
    }
    if (QRaw.dim() != 4 || QRaw.size(1) != 4 || QRaw.size(3) != NumHeads * D) {
        throw std::runtime_error("Invalid QRaw shape for visionhope_qnorm_forward.");
    }
    int B = QRaw.size(0);
    int Directions = QRaw.size(1);
    int L = QRaw.size(2);
    auto QOut = torch::empty({B, Directions, NumHeads, L, D}, QRaw.options().dtype(torch::kFloat32));
    auto InvNorm = torch::empty({(int64_t)B * Directions * L * NumHeads}, QRaw.options().dtype(torch::kFloat32));

    int64_t blocks64 = (int64_t)B * Directions * L * NumHeads;
    auto stream = at::cuda::getCurrentCUDAStream();
    if (UseTile) {
        int threads = 128;
        int groups = threads / std::min(D, 32);
        int blocks = (int)((blocks64 + groups - 1) / groups);
#define LAUNCH_QNORM_TILE(D_VAL) \
        visionhope_qnorm_forward_tile_kernel<D_VAL><<<blocks, threads, 0, stream>>>( \
            reinterpret_cast<const __nv_bfloat16*>(QRaw.data_ptr<at::BFloat16>()), \
            QOut.data_ptr<float>(), \
            InvNorm.data_ptr<float>(), \
            B, Directions, L, NumHeads, blocks64); \

        if (D == 4) { LAUNCH_QNORM_TILE(4) }
        else if (D == 8) { LAUNCH_QNORM_TILE(8) }
        else if (D == 16) { LAUNCH_QNORM_TILE(16) }
        else if (D == 32) { LAUNCH_QNORM_TILE(32) }
        else if (D == 64) { LAUNCH_QNORM_TILE(64) }
        else { throw std::runtime_error("Unsupported qnorm head dimension"); }
#undef LAUNCH_QNORM_TILE
    } else {
        int threads = (D > 32) ? 64 : 32;
        visionhope_qnorm_forward_kernel<<<(int)blocks64, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(QRaw.data_ptr<at::BFloat16>()),
            QOut.data_ptr<float>(),
            InvNorm.data_ptr<float>(),
            B, Directions, L, NumHeads, D);
    }
    return {QOut, InvNorm};
}

torch::Tensor visionhope_qnorm_backward(torch::Tensor DQOut, torch::Tensor QOut, torch::Tensor InvNorm, int NumHeads, int D, bool UseTile) {
    if (DQOut.scalar_type() != at::ScalarType::Float || QOut.scalar_type() != at::ScalarType::Float) {
        throw std::runtime_error("visionhope_qnorm_backward expects fp32 DQOut and QOut.");
    }
    int B = QOut.size(0);
    int Directions = QOut.size(1);
    int L = QOut.size(3);
    int C = NumHeads * D;
    auto DQRaw = torch::empty({B, Directions, L, C}, QOut.options().dtype(torch::kBFloat16));

    int64_t blocks64 = (int64_t)B * Directions * L * NumHeads;
    auto stream = at::cuda::getCurrentCUDAStream();
    if (UseTile) {
        int threads = 128;
        int groups = threads / std::min(D, 32);
        int blocks = (int)((blocks64 + groups - 1) / groups);
#define LAUNCH_QNORM_TILE(D_VAL) \
        visionhope_qnorm_backward_tile_kernel<D_VAL><<<blocks, threads, 0, stream>>>( \
            DQOut.data_ptr<float>(), \
            QOut.data_ptr<float>(), \
            InvNorm.data_ptr<float>(), \
            reinterpret_cast<__nv_bfloat16*>(DQRaw.data_ptr<at::BFloat16>()), \
            B, Directions, L, NumHeads, blocks64); \

        if (D == 4) { LAUNCH_QNORM_TILE(4) }
        else if (D == 8) { LAUNCH_QNORM_TILE(8) }
        else if (D == 16) { LAUNCH_QNORM_TILE(16) }
        else if (D == 32) { LAUNCH_QNORM_TILE(32) }
        else if (D == 64) { LAUNCH_QNORM_TILE(64) }
        else { throw std::runtime_error("Unsupported qnorm head dimension"); }
#undef LAUNCH_QNORM_TILE
    } else {
        int threads = (D > 32) ? 64 : 32;
        visionhope_qnorm_backward_kernel<<<(int)blocks64, threads, 0, stream>>>(
            DQOut.data_ptr<float>(),
            QOut.data_ptr<float>(),
            InvNorm.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(DQRaw.data_ptr<at::BFloat16>()),
            B, Directions, L, NumHeads, D);
    }
    return DQRaw;
}

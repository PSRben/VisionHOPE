#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <climits>
#include <math.h>

#define ETA_SCALE __ETA_SCALE__
#define ALPHA_MIN __ALPHA_MIN__
#define ALPHA_BIAS __ALPHA_BIAS__
#define PRE_ACT_CLAMP 10.0f
#define STAB_MU __STAB_MU__
#define STAB_DENOM_EPS 1e-6f
#define STAB_NORM_EPS 1e-12f
#define Q_NORM_EPS 1e-12f
#define ETA2ALPHA_CLAMP __ETA2ALPHA_CLAMP__
#define ETA2ALPHA_FACTOR __ETA2ALPHA_FACTOR__
#define ETA2ALPHA_DENOM_EPS 1e-20f
#define DEFAULT_MAX_DYNAMIC_SMEM 49152

__host__ __forceinline__ void set_inference_shared_memory(
    const void* kernel, int smem_bytes) {
    if (smem_bytes > DEFAULT_MAX_DYNAMIC_SMEM) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    }
}

__forceinline__ __device__ float soft_project_eta(float eta, float alpha, float diff_norm) {
    float eta_cap = STAB_MU * (1.0f - alpha) / (diff_norm + STAB_DENOM_EPS);
    float z = eta / eta_cap;
    return eta_cap * (-expm1f(-z));
}

__forceinline__ __device__ float clamp_eta_to_alpha(float eta_eff, float alpha, float key_norm_sq) {
#if ETA2ALPHA_CLAMP
    float denom = fmaxf(key_norm_sq, ETA2ALPHA_DENOM_EPS);
    float eta_limit = nextafterf(ETA2ALPHA_FACTOR * alpha / denom, 0.0f);
    return fminf(eta_eff, eta_limit);
#else
    return eta_eff;
#endif
}

__global__ void merge_line_dirs_kernel(
    const float* __restrict__ DirOut,
    float* __restrict__ Out,
    int Total,
    int SpatialC)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Total) return;
    int b = idx / SpatialC;
    int rem = idx - b * SpatialC;
    int base = (b * 4 * SpatialC) + rem;
    Out[idx] = DirOut[base] + DirOut[base + SpatialC] + DirOut[base + 2 * SpatialC] + DirOut[base + 3 * SpatialC];
}

__SRNL_INFERENCE_SOURCE__

// Packed sequence input and raw queries retain their public SRNL layouts.
template <int D>
struct SRNLSequenceIO {
    const __nv_bfloat16* X;
    const __nv_bfloat16* Q_raw;
    __nv_bfloat16* Out;
    int b, dir, head, Directions, NumHeads, Length;

    __device__ __forceinline__ void load(int t, float& x, float& q) const {
        x = 0.f;
        q = 0.f;
        if (t < Length) {
            const int i = threadIdx.x;
            int xi = (((b * Directions + dir) * NumHeads + head) * Length + t) * D + i;
            int qi = (((b * Directions + dir) * Length + t) * NumHeads + head) * D + i;
            x = __bfloat162float(X[xi]);
            q = __bfloat162float(Q_raw[qi]);
        }
    }

    __device__ __forceinline__ void store(int t, float y, float) const {
        if (t < Length) {
            int ix = (((b * Directions + dir) * NumHeads + head) * Length + t) * D + threadIdx.x;
            Out[ix] = __float2bfloat16(y);
        }
    }
};

struct SRNLChunkSizes { int values[4]; };

template <int D>
__global__ void srnl_sequence_token_parallel(
    const float* M_m_0, const float* M_v_0, const float* M_k_0,
    const float* M_eta_0, const float* M_alpha_0,
    const __nv_bfloat16* X, const __nv_bfloat16* Q_raw, __nv_bfloat16* Out,
    int Directions, int NumHeads, int Length, SRNLChunkSizes chunks, int TileTokens)
{
    const int head = blockIdx.x % NumHeads;
    const int dir = (blockIdx.x / NumHeads) % Directions;
    const int b = blockIdx.x / (NumHeads * Directions);
    const int chunk_size = chunks.values[dir];
    const int state_base = (dir * NumHeads + head) * D * D;
    const int vec_base = (dir * NumHeads + head) * D;
    SRNLSequenceIO<D> io{X, Q_raw, Out, b, dir, head, Directions, NumHeads, Length};
    srnl_inference_chunks<D>(M_m_0, M_v_0, M_k_0, M_eta_0, M_alpha_0,
        state_base, vec_base, chunk_size, (Length - 1) / chunk_size + 1, TileTokens, io);
}

template <int D>
void launch_srnl_sequence(
    const torch::Tensor& M_m_0, const torch::Tensor& M_v_0, const torch::Tensor& M_k_0,
    const torch::Tensor& M_eta_0, const torch::Tensor& M_alpha_0,
    const torch::Tensor& X, const torch::Tensor& Q_raw, torch::Tensor& Out,
    SRNLChunkSizes chunks, int max_chunk)
{
    SRNLInferenceSchedule plan = srnl_inference_schedule(D, max_chunk);
    set_inference_shared_memory((const void*)srnl_sequence_token_parallel<D>, plan.shared_bytes);
    int B = X.size(0), R = X.size(1), heads = X.size(2), length = X.size(3);
    srnl_sequence_token_parallel<D><<<B * R * heads, dim3(D, plan.groups), plan.shared_bytes,
                                    at::cuda::getCurrentCUDAStream()>>>(
        M_m_0.data_ptr<float>(), M_v_0.data_ptr<float>(), M_k_0.data_ptr<float>(),
        M_eta_0.data_ptr<float>(), M_alpha_0.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(Q_raw.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(Out.data_ptr<at::BFloat16>()),
        R, heads, length, chunks, plan.tile_tokens);
}

torch::Tensor srnl_sequence_forward(
    torch::Tensor M_m_0, torch::Tensor M_v_0, torch::Tensor M_k_0,
    torch::Tensor M_eta_0, torch::Tensor M_alpha_0,
    torch::Tensor X, torch::Tensor Q_raw, std::vector<int64_t> chunk_sizes)
{
    TORCH_CHECK(X.is_cuda() && X.dim() == 5 && X.is_contiguous(),
                "SRNL inference expects contiguous CUDA [B, directions, heads, N, d]");
    c10::cuda::CUDAGuard device_guard(X.device());
    TORCH_CHECK(X.scalar_type() == at::ScalarType::BFloat16 &&
                Q_raw.scalar_type() == X.scalar_type() && Q_raw.device() == X.device() &&
                Q_raw.is_contiguous(), "SRNL inference requires matching input/query dtype and device");
    int64_t B = X.size(0), R = X.size(1), heads = X.size(2), length = X.size(3), D = X.size(4);
    TORCH_CHECK(B > 0 && heads > 0 && length > 0 && (R == 1 || R == 2 || R == 4),
                "SRNL inference requires nonempty inputs and 1, 2, or 4 directions");
    TORCH_CHECK(D == 4 || D == 8 || D == 16 || D == 32 || D == 64, "Unsupported SRNL head dimension");
    TORCH_CHECK(Q_raw.sizes() == at::IntArrayRef({B, R, length, heads * D}), "Invalid raw query shape");
    TORCH_CHECK(X.numel() <= INT_MAX && R * heads * D * D <= INT_MAX,
                "SRNL inference input exceeds the supported indexing range");
    for (const auto& state : {M_m_0, M_v_0, M_k_0}) {
        TORCH_CHECK(state.device() == X.device() && state.scalar_type() == at::ScalarType::Float &&
                    state.is_contiguous() && state.sizes() == at::IntArrayRef({R, heads, D, D}),
                    "SRNL matrix memories must be contiguous FP32 [directions, heads, d, d]");
    }
    for (const auto& state : {M_eta_0, M_alpha_0}) {
        TORCH_CHECK(state.device() == X.device() && state.scalar_type() == at::ScalarType::Float &&
                    state.is_contiguous() && state.sizes() == at::IntArrayRef({R, heads, D}),
                    "SRNL vector memories must be contiguous FP32 [directions, heads, d]");
    }
    TORCH_CHECK(chunk_sizes.size() == R, "Provide one chunk length per direction");
    SRNLChunkSizes chunks{};
    int max_chunk = 1;
    for (int r = 0; r < R; ++r) {
        int64_t c = chunk_sizes[r];
        TORCH_CHECK(c > 0 && c <= INT_MAX && ((length - 1) / c + 1) * c <= INT_MAX,
                    "Invalid SRNL chunk length or padded sequence length");
        chunks.values[r] = int(c);
        max_chunk = std::max(max_chunk, int(c));
    }
    auto Out = torch::empty_like(X);
    switch (D) {
        case 4: launch_srnl_sequence<4>(M_m_0, M_v_0, M_k_0, M_eta_0, M_alpha_0, X, Q_raw, Out, chunks, max_chunk); break;
        case 8: launch_srnl_sequence<8>(M_m_0, M_v_0, M_k_0, M_eta_0, M_alpha_0, X, Q_raw, Out, chunks, max_chunk); break;
        case 16: launch_srnl_sequence<16>(M_m_0, M_v_0, M_k_0, M_eta_0, M_alpha_0, X, Q_raw, Out, chunks, max_chunk); break;
        case 32: launch_srnl_sequence<32>(M_m_0, M_v_0, M_k_0, M_eta_0, M_alpha_0, X, Q_raw, Out, chunks, max_chunk); break;
        case 64: launch_srnl_sequence<64>(M_m_0, M_v_0, M_k_0, M_eta_0, M_alpha_0, X, Q_raw, Out, chunks, max_chunk); break;
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return Out;
}

torch::Tensor srnl_line_forward(
    torch::Tensor M_m_0,
    torch::Tensor M_v_0,
    torch::Tensor M_k_0,
    torch::Tensor M_eta_0,
    torch::Tensor M_alpha_0,
    torch::Tensor X,
    torch::Tensor Q_raw,
    torch::Tensor D_skip,
    torch::Tensor Dir_weight,
    int H,
    int W,
    int D_int)
{
    if (X.scalar_type() != at::ScalarType::BFloat16 || Q_raw.scalar_type() != at::ScalarType::BFloat16) {
        throw std::runtime_error("line fused HOPE forward expects bf16 X and bf16 Q_raw.");
    }
    if (M_m_0.scalar_type() != at::ScalarType::Float || D_skip.scalar_type() != at::ScalarType::Float || Dir_weight.scalar_type() != at::ScalarType::Float) {
        throw std::runtime_error("line fused HOPE forward expects fp32 state, skip, and direction weights.");
    }
    int B = X.size(0);
    int NumHeads = X.size(2);
    int L = X.size(3);
    int C = NumHeads * D_int;
    if (L != H * W) {
        throw std::runtime_error("line fused HOPE got inconsistent H/W and sequence length.");
    }
    auto Out = torch::zeros({B, H, W, C}, X.options().dtype(torch::kFloat32));
    auto DirOut = torch::empty({0}, X.options().dtype(torch::kFloat32));
    TORCH_CHECK(D_skip.numel() == C || D_skip.numel() == 4 * C, "d_skip must contain C or 4*C values");
    int SkipStride = D_skip.numel() == C ? 0 : C;
    if (D_int == 4) { DISPATCH_LINE_PARALLEL(4, false, false) }
    else if (D_int == 8) { DISPATCH_LINE_PARALLEL(8, false, false) }
    else if (D_int == 16) { DISPATCH_LINE_PARALLEL(16, false, false) }
    else if (D_int == 32) { DISPATCH_LINE_PARALLEL(32, false, false) }
    else if (D_int == 64) { DISPATCH_LINE_PARALLEL(64, false, false) }
    else { throw std::runtime_error("line fused HOPE supports head_dim in {4, 8, 16, 32, 64}."); }
    return Out;
}

torch::Tensor srnl_line_forward_noatomic(
    torch::Tensor M_m_0,
    torch::Tensor M_v_0,
    torch::Tensor M_k_0,
    torch::Tensor M_eta_0,
    torch::Tensor M_alpha_0,
    torch::Tensor X,
    torch::Tensor Q_raw,
    torch::Tensor D_skip,
    torch::Tensor Dir_weight,
    int H,
    int W,
    int D_int)
{
    if (X.scalar_type() != at::ScalarType::BFloat16 || Q_raw.scalar_type() != at::ScalarType::BFloat16) {
        throw std::runtime_error("line fused no-atomic HOPE forward expects bf16 X and bf16 Q_raw.");
    }
    if (M_m_0.scalar_type() != at::ScalarType::Float || D_skip.scalar_type() != at::ScalarType::Float || Dir_weight.scalar_type() != at::ScalarType::Float) {
        throw std::runtime_error("line fused no-atomic HOPE forward expects fp32 state, skip, and direction weights.");
    }
    int B = X.size(0);
    int NumHeads = X.size(2);
    int L = X.size(3);
    int C = NumHeads * D_int;
    if (L != H * W) {
        throw std::runtime_error("line fused no-atomic HOPE got inconsistent H/W and sequence length.");
    }
    auto Out = torch::empty({B, H, W, C}, X.options().dtype(torch::kFloat32));
    auto DirOut = torch::empty({B, 4, H, W, C}, X.options().dtype(torch::kFloat32));
    TORCH_CHECK(D_skip.numel() == C || D_skip.numel() == 4 * C, "d_skip must contain C or 4*C values");
    int SkipStride = D_skip.numel() == C ? 0 : C;
    if (D_int == 4) { DISPATCH_LINE_PARALLEL(4, true, false) }
    else if (D_int == 8) { DISPATCH_LINE_PARALLEL(8, true, false) }
    else if (D_int == 16) { DISPATCH_LINE_PARALLEL(16, true, false) }
    else if (D_int == 32) { DISPATCH_LINE_PARALLEL(32, true, false) }
    else if (D_int == 64) { DISPATCH_LINE_PARALLEL(64, true, false) }
    else { throw std::runtime_error("line fused HOPE supports head_dim in {4, 8, 16, 32, 64}."); }

    int total = B * H * W * C;
    int spatial_c = H * W * C;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    merge_line_dirs_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(DirOut.data_ptr<float>(), Out.data_ptr<float>(), total, spatial_c);
    return Out;
}

torch::Tensor srnl_line_forward_seq(
    torch::Tensor M_m_0,
    torch::Tensor M_v_0,
    torch::Tensor M_k_0,
    torch::Tensor M_eta_0,
    torch::Tensor M_alpha_0,
    torch::Tensor X,
    torch::Tensor Q_raw,
    torch::Tensor D_skip,
    torch::Tensor Dir_weight,
    int H,
    int W,
    int D_int)
{
    if (X.scalar_type() != at::ScalarType::BFloat16 || Q_raw.scalar_type() != at::ScalarType::BFloat16) {
        throw std::runtime_error("line fused seq-layout HOPE forward expects bf16 X and bf16 Q_raw.");
    }
    if (M_m_0.scalar_type() != at::ScalarType::Float || D_skip.scalar_type() != at::ScalarType::Float || Dir_weight.scalar_type() != at::ScalarType::Float) {
        throw std::runtime_error("line fused seq-layout HOPE forward expects fp32 state, skip, and direction weights.");
    }
    int B = X.size(0);
    int L = X.size(2);
    int C = X.size(3);
    int NumHeads = C / D_int;
    if (L != H * W || C != NumHeads * D_int) {
        throw std::runtime_error("line fused seq-layout HOPE got inconsistent H/W, C, and head_dim.");
    }
    auto Out = torch::zeros({B, H, W, C}, X.options().dtype(torch::kFloat32));
    auto DirOut = torch::empty({0}, X.options().dtype(torch::kFloat32));
    TORCH_CHECK(D_skip.numel() == C || D_skip.numel() == 4 * C, "d_skip must contain C or 4*C values");
    int SkipStride = D_skip.numel() == C ? 0 : C;
    if (D_int == 4) { DISPATCH_LINE_PARALLEL(4, false, true) }
    else if (D_int == 8) { DISPATCH_LINE_PARALLEL(8, false, true) }
    else if (D_int == 16) { DISPATCH_LINE_PARALLEL(16, false, true) }
    else if (D_int == 32) { DISPATCH_LINE_PARALLEL(32, false, true) }
    else if (D_int == 64) { DISPATCH_LINE_PARALLEL(64, false, true) }
    else { throw std::runtime_error("line fused HOPE supports head_dim in {4, 8, 16, 32, 64}."); }
    return Out;
}

torch::Tensor srnl_line_forward_seq_noatomic(
    torch::Tensor M_m_0,
    torch::Tensor M_v_0,
    torch::Tensor M_k_0,
    torch::Tensor M_eta_0,
    torch::Tensor M_alpha_0,
    torch::Tensor X,
    torch::Tensor Q_raw,
    torch::Tensor D_skip,
    torch::Tensor Dir_weight,
    int H,
    int W,
    int D_int)
{
    if (X.scalar_type() != at::ScalarType::BFloat16 || Q_raw.scalar_type() != at::ScalarType::BFloat16) {
        throw std::runtime_error("line fused seq-layout no-atomic HOPE forward expects bf16 X and bf16 Q_raw.");
    }
    if (M_m_0.scalar_type() != at::ScalarType::Float || D_skip.scalar_type() != at::ScalarType::Float || Dir_weight.scalar_type() != at::ScalarType::Float) {
        throw std::runtime_error("line fused seq-layout no-atomic HOPE forward expects fp32 state, skip, and direction weights.");
    }
    int B = X.size(0);
    int L = X.size(2);
    int C = X.size(3);
    int NumHeads = C / D_int;
    if (L != H * W || C != NumHeads * D_int) {
        throw std::runtime_error("line fused seq-layout no-atomic HOPE got inconsistent H/W, C, and head_dim.");
    }
    auto Out = torch::empty({B, H, W, C}, X.options().dtype(torch::kFloat32));
    auto DirOut = torch::empty({B, 4, H, W, C}, X.options().dtype(torch::kFloat32));
    TORCH_CHECK(D_skip.numel() == C || D_skip.numel() == 4 * C, "d_skip must contain C or 4*C values");
    int SkipStride = D_skip.numel() == C ? 0 : C;
    if (D_int == 4) { DISPATCH_LINE_PARALLEL(4, true, true) }
    else if (D_int == 8) { DISPATCH_LINE_PARALLEL(8, true, true) }
    else if (D_int == 16) { DISPATCH_LINE_PARALLEL(16, true, true) }
    else if (D_int == 32) { DISPATCH_LINE_PARALLEL(32, true, true) }
    else if (D_int == 64) { DISPATCH_LINE_PARALLEL(64, true, true) }
    else { throw std::runtime_error("line fused HOPE supports head_dim in {4, 8, 16, 32, 64}."); }

    int total = B * H * W * C;
    int spatial_c = H * W * C;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    merge_line_dirs_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(DirOut.data_ptr<float>(), Out.data_ptr<float>(), total, spatial_c);
    return Out;
}

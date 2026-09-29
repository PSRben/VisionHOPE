
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>

#define MAX_CHUNK 64
#define RECOMP_TILE 8
#define MAX_SUBCHUNKS ((MAX_CHUNK + RECOMP_TILE - 1) / RECOMP_TILE)
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
${thread_groups}

__host__ __forceinline__ int srnl_forward_smem_bytes(int D) {
    int floats = 0;
    floats += 3 * D * (D + 1);
    floats += D;                 // s_x
    floats += D;                 // s_q
    floats += MAX_CHUNK * D;     // s_k_chunk
    floats += MAX_CHUNK * D;     // s_v_chunk
    floats += MAX_CHUNK;         // s_eta_chunk_val
    floats += MAX_CHUNK;         // s_alpha_chunk_val
    floats += D * (D + 1);       // s_g_final
    floats += D;                 // s_eta_init_vec
    floats += D;                 // s_alpha_init_vec
    return floats * (int)sizeof(float);
}

__host__ __forceinline__ int srnl_backward_smem_bytes(int D, int groups, int cmax) {
    int floats_per_group = 0;
    floats_per_group += 3 * D * (D + 1);   // s_m_init_storage
    floats_per_group += D * (D + 1);       // smem_red_storage
    floats_per_group += 3 * D;             // s_x/s_q/s_dy
    floats_per_group += D * (D + 1);       // s_g_mat_storage
    floats_per_group += 2 * D;             // s_vec_grad_storage
    floats_per_group += D;                 // s_eta_init_vec_storage
    floats_per_group += D;                 // s_alpha_init_vec_storage
    floats_per_group += D;                 // s_r_vec_storage
    floats_per_group += D;                 // s_bar_r_vec_storage
    floats_per_group += cmax * D;          // s_k_chunk_storage
    floats_per_group += cmax * D;          // s_v_chunk_storage
    floats_per_group += 5 * cmax;          // scalar chunk buffers
    return groups * floats_per_group * (int)sizeof(float);
}

__host__ __forceinline__ void set_forward_dynamic_smem_attr_if_needed(
    const void* kernel, int smem_bytes) {
    if (smem_bytes > DEFAULT_MAX_DYNAMIC_SMEM) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    }
}

__host__ __forceinline__ void set_backward_dynamic_smem_attr_if_needed(
    const void* kernel, int smem_bytes) {
    if (smem_bytes > DEFAULT_MAX_DYNAMIC_SMEM) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    }
}

template <int D>
__forceinline__ __device__ int warpGroupBaseLane(int group) {
    if constexpr (D >= 32) {
        return 0;
    } else {
        return (group % (32 / D)) * D;
    }
}

template <int D>
__forceinline__ __device__ unsigned int warpGroupMask(int group) {
    if constexpr (D >= 32) {
        return 0xffffffffu;
    } else {
        unsigned int base_mask = (1u << D) - 1u;
        return base_mask << warpGroupBaseLane<D>(group);
    }
}

template <int D>
__inline__ __device__ float warpReduceSum(float val) {
    if constexpr (D == 64) {
        __shared__ float s_reduce64_partial[2];
        int lane = threadIdx.x & 31;
        int warp = threadIdx.x >> 5;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffffu, val, offset);
        }
        if (lane == 0) {
            s_reduce64_partial[warp] = val;
        }
        __syncthreads();
        float result = s_reduce64_partial[0] + s_reduce64_partial[1];
        __syncthreads();
        return result;
    } else {
        unsigned int mask = (D == 32) ? 0xffffffff : ((1u << D) - 1);
        #pragma unroll
        for (int offset = D / 2; offset > 0; offset /= 2)
            val += __shfl_down_sync(mask, val, offset);
        return __shfl_sync(mask, val, 0);
    }
}

template <int D>
__inline__ __device__ float warpReduceSum(float val, int group) {
    if constexpr (D == 64) {
        int d_i = threadIdx.x - group * D;
        int lane = d_i & 31;
        int half = d_i >> 5;
        __shared__ float s_reduce64_group_partial[4];
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffffu, val, offset);
        }
        if (lane == 0) {
            s_reduce64_group_partial[group * 2 + half] = val;
        }
        __syncthreads();
        float result = s_reduce64_group_partial[group * 2] + s_reduce64_group_partial[group * 2 + 1];
        __syncthreads();
        return result;
    } else {
        unsigned int mask = warpGroupMask<D>(group);
        #pragma unroll
        for (int offset = D / 2; offset > 0; offset /= 2)
            val += __shfl_down_sync(mask, val, offset);
        return __shfl_sync(mask, val, warpGroupBaseLane<D>(group));
    }
}

template <int D>
__forceinline__ __device__ float warpBroadcast0(float val) {
    if constexpr (D == 64) {
        __shared__ float s_bcast64;
        if (threadIdx.x == 0) {
            s_bcast64 = val;
        }
        __syncthreads();
        float result = s_bcast64;
        __syncthreads();
        return result;
    } else {
        unsigned int mask = (D == 32) ? 0xffffffff : ((1u << D) - 1);
        return __shfl_sync(mask, val, 0);
    }
}

template <int D>
__forceinline__ __device__ float warpBroadcast0(float val, int group) {
    if constexpr (D == 64) {
        __shared__ float s_bcast64[2];
        int d_i = threadIdx.x - group * D;
        if (d_i == 0) {
            s_bcast64[group] = val;
        }
        __syncthreads();
        float result = s_bcast64[group];
        __syncthreads();
        return result;
    } else {
        unsigned int mask = warpGroupMask<D>(group);
        return __shfl_sync(mask, val, warpGroupBaseLane<D>(group));
    }
}

template <int D>
__forceinline__ __device__ void warpSync() {
    if constexpr (D == 64) {
        __syncthreads();
    } else {
        unsigned int mask = (D == 32) ? 0xffffffff : ((1u << D) - 1);
        __syncwarp(mask);
    }
}

template <int D>
__forceinline__ __device__ void warpSync(int group) {
    if constexpr (D == 64) {
        __syncthreads();
    } else {
        unsigned int mask = warpGroupMask<D>(group);
        __syncwarp(mask);
    }
}

__inline__ __device__ float warpReduceSum16Group(float val, int group) {
    unsigned int mask = 0xffffu << ((group & 1) * 16);
    #pragma unroll
    for (int offset = 8; offset > 0; offset /= 2)
        val += __shfl_down_sync(mask, val, offset, 16);
    return __shfl_sync(mask, val, 0, 16);
}

__forceinline__ __device__ float warpBroadcast016Group(float val, int group) {
    unsigned int mask = 0xffffu << ((group & 1) * 16);
    return __shfl_sync(mask, val, 0, 16);
}

__forceinline__ __device__ void warpSync16Group(int group) {
    unsigned int mask = 0xffffu << ((group & 1) * 16);
    __syncwarp(mask);
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

template <int D>
__forceinline__ __device__ float soft_project_eta_warp(float eta, float alpha, float diff_norm) {
    float eta_eff = 0.0f;
    if (threadIdx.x == 0) {
        eta_eff = soft_project_eta(eta, alpha, diff_norm);
    }
    return warpBroadcast0<D>(eta_eff);
}

template <int D>
__forceinline__ __device__ float soft_project_eta_warp(float eta, float alpha, float diff_norm, int group, int d_i) {
    float eta_eff = 0.0f;
    if (d_i == 0) {
        eta_eff = soft_project_eta(eta, alpha, diff_norm);
    }
    return warpBroadcast0<D>(eta_eff, group);
}

__forceinline__ __device__ float soft_injection_dcap(float x, float exp_neg) {
    if (x < 1e-3f) {
        float psi = 0.5f + x * (-1.0f / 3.0f + x * (1.0f / 8.0f +
                    x * (-1.0f / 30.0f + x / 144.0f)));
        return x * x * psi;
    }
    return -expm1f(-x) - x * exp_neg;
}

template <int D>
__forceinline__ __device__ void soft_project_eta_backward_warp(
    float eta, float alpha, float diff_norm,
    float* diff_denom_out, float* eta_cap_out, float* eta_eff_out,
    float* d_etaeff_deta_out, float* d_etaeff_dcap_out)
{
    float diff_denom = 0.0f;
    float eta_cap = 0.0f;
    float eta_eff = 0.0f;
    float d_etaeff_deta = 0.0f;
    float d_etaeff_dcap = 0.0f;
    if (threadIdx.x == 0) {
        diff_denom = diff_norm + STAB_DENOM_EPS;
        eta_cap = STAB_MU * (1.0f - alpha) / diff_denom;
        float proj_ratio = eta / eta_cap;
        float exp_neg = expf(-proj_ratio);
        eta_eff = eta_cap * (-expm1f(-proj_ratio));
        d_etaeff_deta = exp_neg;
        d_etaeff_dcap = soft_injection_dcap(proj_ratio, exp_neg);
    }
    *diff_denom_out = warpBroadcast0<D>(diff_denom);
    *eta_cap_out = warpBroadcast0<D>(eta_cap);
    *eta_eff_out = warpBroadcast0<D>(eta_eff);
    *d_etaeff_deta_out = warpBroadcast0<D>(d_etaeff_deta);
    *d_etaeff_dcap_out = warpBroadcast0<D>(d_etaeff_dcap);
}

template <int D>
__forceinline__ __device__ void soft_project_eta_backward_warp(
    float eta, float alpha, float diff_norm,
    float* diff_denom_out, float* eta_cap_out, float* eta_eff_out,
    float* d_etaeff_deta_out, float* d_etaeff_dcap_out,
    int group, int d_i)
{
    float diff_denom = 0.0f;
    float eta_cap = 0.0f;
    float eta_eff = 0.0f;
    float d_etaeff_deta = 0.0f;
    float d_etaeff_dcap = 0.0f;
    if (d_i == 0) {
        diff_denom = diff_norm + STAB_DENOM_EPS;
        eta_cap = STAB_MU * (1.0f - alpha) / diff_denom;
        float proj_ratio = eta / eta_cap;
        float exp_neg = expf(-proj_ratio);
        eta_eff = eta_cap * (-expm1f(-proj_ratio));
        d_etaeff_deta = exp_neg;
        d_etaeff_dcap = soft_injection_dcap(proj_ratio, exp_neg);
    }
    *diff_denom_out = warpBroadcast0<D>(diff_denom, group);
    *eta_cap_out = warpBroadcast0<D>(eta_cap, group);
    *eta_eff_out = warpBroadcast0<D>(eta_eff, group);
    *d_etaeff_deta_out = warpBroadcast0<D>(d_etaeff_deta, group);
    *d_etaeff_dcap_out = warpBroadcast0<D>(d_etaeff_dcap, group);
}

template <int D>
__forceinline__ __device__ void projected_eta_backward_warp(
    float eta, float alpha, float diff_norm, float key_norm_sq,
    float* diff_denom_out, float* eta_cap_out, float* eta_eff_out,
    float* d_etaeff_deta_out, float* d_etaeff_dcap_out,
    float* d_etaeff_dalpha_direct_out, float* d_etaeff_dkeynormsq_out,
    int group, int d_i)
{
    soft_project_eta_backward_warp<D>(
        eta, alpha, diff_norm,
        diff_denom_out, eta_cap_out, eta_eff_out,
        d_etaeff_deta_out, d_etaeff_dcap_out,
        group, d_i);
    float d_etaeff_dalpha_direct = 0.0f;
    float d_etaeff_dkeynormsq = 0.0f;
#if ETA2ALPHA_CLAMP
    if (d_i == 0) {
        float key_denom = fmaxf(key_norm_sq, ETA2ALPHA_DENOM_EPS);
        float eta_limit = nextafterf(ETA2ALPHA_FACTOR * alpha / key_denom, 0.0f);
        if (*eta_eff_out > eta_limit) {
            *eta_eff_out = eta_limit;
            *d_etaeff_deta_out = 0.0f;
            *d_etaeff_dcap_out = 0.0f;
            d_etaeff_dalpha_direct = ETA2ALPHA_FACTOR / key_denom;
            if (key_norm_sq > ETA2ALPHA_DENOM_EPS) {
                d_etaeff_dkeynormsq = -ETA2ALPHA_FACTOR * alpha / (key_denom * key_denom);
            }
        }
    }
    *eta_eff_out = warpBroadcast0<D>(*eta_eff_out, group);
    *d_etaeff_deta_out = warpBroadcast0<D>(*d_etaeff_deta_out, group);
    *d_etaeff_dcap_out = warpBroadcast0<D>(*d_etaeff_dcap_out, group);
#endif
    *d_etaeff_dalpha_direct_out = warpBroadcast0<D>(d_etaeff_dalpha_direct, group);
    *d_etaeff_dkeynormsq_out = warpBroadcast0<D>(d_etaeff_dkeynormsq, group);
}

__forceinline__ __device__ float soft_project_eta_warp16_group(float eta, float alpha, float diff_norm, int group, int d_i) {
    float eta_eff = 0.0f;
    if (d_i == 0) {
        eta_eff = soft_project_eta(eta, alpha, diff_norm);
    }
    return warpBroadcast016Group(eta_eff, group);
}


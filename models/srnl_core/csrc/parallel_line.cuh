// Spatial adapter for the shared inference recurrence.
template <int D, bool WriteDirOut, bool SeqInput>
struct SRNLLineIO {
    const __nv_bfloat16* X;
    const __nv_bfloat16* Q_raw;
    float* Output;
    float skip, weight;
    int dir, H, W, C;

    __device__ __forceinline__ void load(int t, float& x, float& q) const {
        const int ix = t * (SeqInput ? C : D);
        x = __bfloat162float(X[ix]);
        q = __bfloat162float(Q_raw[t * C]);
    }

    __device__ __forceinline__ void store(int t, float y, float x) const {
        const int orig = (dir & 1) ? H * W - 1 - t : t;
        const int position = dir < 2 ? orig : (orig % H) * W + orig / H;
        float contrib = y + skip * x;
        float weighted = weight * contrib;
        if constexpr (WriteDirOut) Output[position * C] = weighted;
        else atomicAdd(Output + position * C, weighted);
    }
};

template <int D, bool WriteDirOut, bool SeqInput>
__global__ void srnl_line_token_parallel(
    const float* __restrict__ M_m_0,
    const float* __restrict__ M_v_0,
    const float* __restrict__ M_k_0,
    const float* __restrict__ M_eta_0,
    const float* __restrict__ M_alpha_0,
    const __nv_bfloat16* __restrict__ X,
    const __nv_bfloat16* __restrict__ Q_raw,
    const float* __restrict__ D_skip,
    const float* __restrict__ Dir_weight,
    float* __restrict__ Out,
    float* __restrict__ DirOut,
    int B, int NumHeads, int H, int W, int C, int SkipStride, int TileTokens)
{
    const int head = blockIdx.x % NumHeads;
    const int dir = (blockIdx.x / NumHeads) % 4;
    const int b = blockIdx.x / (NumHeads * 4);
    const int state_base = (dir * NumHeads + head) * D * D;
    const int vec_base = (dir * NumHeads + head) * D;
    const int channel = head * D + threadIdx.x;
    const int L = H * W;
    const int input_base = SeqInput ? (b * 4 + dir) * L * C + channel
        : ((b * 4 + dir) * NumHeads + head) * L * D + threadIdx.x;
    const int query_base = (b * 4 + dir) * L * C + channel;
    float* output = WriteDirOut ? DirOut + (b * 4 + dir) * L * C + channel
        : Out + b * L * C + channel;
    SRNLLineIO<D, WriteDirOut, SeqInput> io{
        X + input_base, Q_raw + query_base, output,
        D_skip[dir * SkipStride + channel], Dir_weight[dir * C + channel],
        dir, H, W, C};
    srnl_inference_chunks<D>(M_m_0, M_v_0, M_k_0, M_eta_0, M_alpha_0,
        state_base, vec_base, dir < 2 ? W : H, dir < 2 ? H : W, TileTokens, io);
}

#define DISPATCH_LINE_PARALLEL(D_VAL, WRITE_DIR_OUT, SEQ_INPUT) \
    do { \
        SRNLInferenceSchedule plan = srnl_inference_schedule(D_VAL, std::max(H, W)); \
        set_inference_shared_memory((const void*)srnl_line_token_parallel<D_VAL, WRITE_DIR_OUT, SEQ_INPUT>, plan.shared_bytes); \
        srnl_line_token_parallel<D_VAL, WRITE_DIR_OUT, SEQ_INPUT><<<B * 4 * NumHeads, dim3(D_VAL, plan.groups), plan.shared_bytes, at::cuda::getCurrentCUDAStream()>>>( \
            M_m_0.data_ptr<float>(), M_v_0.data_ptr<float>(), M_k_0.data_ptr<float>(), \
            M_eta_0.data_ptr<float>(), M_alpha_0.data_ptr<float>(), \
            reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>()), \
            reinterpret_cast<const __nv_bfloat16*>(Q_raw.data_ptr<at::BFloat16>()), \
            D_skip.data_ptr<float>(), Dir_weight.data_ptr<float>(), Out.data_ptr<float>(), \
            WRITE_DIR_OUT ? DirOut.data_ptr<float>() : nullptr, \
            B, NumHeads, H, W, C, SkipStride, plan.tile_tokens); \
    } while (0);

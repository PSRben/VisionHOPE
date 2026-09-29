// Shared inference recurrence. Tiles stage token quantities; only logical
// chunk boundaries refresh the five memory states.
template <int D, class IO>
__device__ __forceinline__ void srnl_inference_chunks(
    const float* M_m_0, const float* M_v_0, const float* M_k_0,
    const float* M_eta_0, const float* M_alpha_0,
    int state_base, int vec_base, int chunk_size, int num_chunks,
    int TileTokens, const IO& io)
{
    const int i = threadIdx.x, group = threadIdx.y, groups = blockDim.y;
    extern __shared__ float storage[];
    float (*state)[D][D + 1] = reinterpret_cast<float (*)[D][D + 1]>(storage);
    float (*gain)[D + 1] = reinterpret_cast<float (*)[D + 1]>(storage + 3 * D * (D + 1));
    float* eta_vec = storage + 4 * D * (D + 1);
    float* alpha_vec = eta_vec + D;
    float (*keys)[D] = reinterpret_cast<float (*)[D]>(alpha_vec + D);
    float (*values)[D] = keys + TileTokens;
    float* eta_chunk = reinterpret_cast<float*>(values + TileTokens);
    float* alpha_chunk = eta_chunk + TileTokens;
    float (*xs)[D] = reinterpret_cast<float (*)[D]>(alpha_chunk + TileTokens);
    float (*qs)[D] = xs + groups;
    for (int row = group; row < D; row += groups) {
        state[0][row][i] = M_m_0[state_base + row * D + i];
        state[1][row][i] = M_v_0[state_base + row * D + i];
        state[2][row][i] = M_k_0[state_base + row * D + i];
    }
    if (group == 0) {
        eta_vec[i] = M_eta_0[vec_base + i];
        alpha_vec[i] = M_alpha_0[vec_base + i];
    }
    __syncthreads();
    for (int c = 0; c < num_chunks; ++c) {
        float g[D];
        if (group == 0) {
            #pragma unroll
            for (int j = 0; j < D; ++j) g[j] = i == j ? 1.f : 0.f;
        }
        for (int base = 0; base < chunk_size; base += TileTokens) {
            const int count = min(TileTokens, chunk_size - base);
            for (int local = group; local < count; local += groups) {
                int t = c * chunk_size + base + local;
                float x, qraw;
                io.load(t, x, qraw);
                float qnorm = srnl_group_sum<D>(qraw * qraw, i, group);
                float q = qraw / fmaxf(sqrtf(qnorm), Q_NORM_EPS);
                if constexpr (D > 32) {
                    xs[group][i] = x; qs[group][i] = q;
                    srnl_group_sync<D>(group);
                }
                float kr = 0.f, v = 0.f, y = 0.f;
                #pragma unroll
                for (int j = 0; j < D; ++j) {
                    float xj, qj;
                    if constexpr (D <= 32) {
                        xj = __shfl_sync(srnl_group_mask<D>(group), x, j, D);
                        qj = __shfl_sync(srnl_group_mask<D>(group), q, j, D);
                    } else {
                        xj = xs[group][j]; qj = qs[group][j];
                    }
                    kr += state[2][i][j] * xj;
                    v += state[1][i][j] * xj;
                    y += state[0][i][j] * qj;
                }
                io.store(t, y, x);
                float kn = srnl_group_sum<D>(kr * kr, i, group);
                float k = kr / sqrtf(kn + 1e-6f);
                keys[local][i] = k; values[local][i] = v;
                float ep = srnl_group_sum<D>(x * eta_vec[i], i, group);
                float ap = srnl_group_sum<D>(x * alpha_vec[i], i, group);
                ep = fminf(PRE_ACT_CLAMP, fmaxf(-PRE_ACT_CLAMP, ep));
                ap = fminf(PRE_ACT_CLAMP, fmaxf(-PRE_ACT_CLAMP, ap + ALPHA_BIAS));
                float eta = 0.f, alpha = 0.f;
                if (i == 0) {
                    float sp = ep > 20.f ? ep : logf(1.f + expf(ep));
                    eta = ETA_SCALE * sp;
                    alpha = ALPHA_MIN + (1.f - ALPHA_MIN) * (1.f / (1.f + expf(-ap)));
                }
                eta = srnl_group_first<D>(eta, i, group);
                alpha = srnl_group_first<D>(alpha, i, group);
                float diff = k - v;
                float dn = sqrtf(srnl_group_sum<D>(diff * diff, i, group) + STAB_NORM_EPS);
                float eff = 0.f;
                if (i == 0) eff = soft_project_eta(eta, alpha, dn);
                eff = srnl_group_first<D>(eff, i, group);
#if ETA2ALPHA_CLAMP
                float ke = srnl_group_sum<D>(k * k, i, group);
                eff = clamp_eta_to_alpha(eff, alpha, ke);
#endif
                if (i == 0) { eta_chunk[local] = eff; alpha_chunk[local] = alpha; }
            }
            __syncthreads();
            if (group == 0) {
                for (int local = 0; local < count; ++local) {
                    float k = keys[local][i], v = values[local][i];
                    float r = k - v;
                    #pragma unroll
                    for (int j = 0; j < D; ++j) r += g[j] * keys[local][j];
                    float alpha = alpha_chunk[local], eta = eta_chunk[local];
                    #pragma unroll
                    for (int j = 0; j < D; ++j) g[j] = alpha * g[j] - eta * r * keys[local][j];
                }
                if (base + count == chunk_size) {
                    #pragma unroll
                    for (int j = 0; j < D; ++j) gain[i][j] = g[j];
                }
            }
            __syncthreads();
            if (count < TileTokens) break;
        }
        float me = 0.f, ma = 0.f;
        if (group == 0) {
            #pragma unroll
            for (int l = 0; l < D; ++l) {
                me += eta_vec[l] * gain[l][i];
                ma += alpha_vec[l] * gain[l][i];
            }
        }
        // Compute each row tile before overwriting its memory rows.
        for (int base = 0; base < D; base += groups) {
            int row = base + group;
            float mm = 0.f, mv = 0.f, mk = 0.f;
            if (row < D) {
                #pragma unroll
                for (int l = 0; l < D; ++l) {
                    float gl = gain[l][i];
                    mm += state[0][row][l] * gl;
                    mv += state[1][row][l] * gl;
                    mk += state[2][row][l] * gl;
                }
            }
            __syncthreads();
            if (row < D) { state[0][row][i] = mm; state[1][row][i] = mv; state[2][row][i] = mk; }
        }
        if (group == 0) { eta_vec[i] = me; alpha_vec[i] = ma; }
        __syncthreads();
    }
}

struct SRNLInferenceSchedule { int groups; int tile_tokens; int shared_bytes; };
__host__ inline SRNLInferenceSchedule srnl_inference_schedule(int D, int work) {
    auto* props = at::cuda::getCurrentDeviceProperties();
    int groups = srnl_parallel_groups(D, work, std::min(256, props->maxThreadsPerBlock));
    int base = (4 * D * (D + 1) + 2 * D + 2 * groups * D) * sizeof(float);
    int per_token = (2 * D + 2) * sizeof(float);
    int limit = std::max(props->sharedMemPerBlock, props->sharedMemPerBlockOptin) - 1024;
    int budget = std::min(limit, std::max(base + groups * per_token,
                                         int(props->sharedMemPerMultiprocessor / 4)));
    TORCH_CHECK(budget >= base + per_token,
                "Insufficient device shared memory for HOPE head_dim=", D);
    int capacity = (budget - base) / per_token;
    int tile = 1;
    while (tile < work && 2 * tile <= capacity) tile *= 2;
    return {groups, tile, base + tile * per_token};
}

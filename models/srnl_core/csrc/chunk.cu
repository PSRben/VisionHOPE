template <int D, bool QRAW>
__global__ void srnl_forward_kernel(
    const float* __restrict__ M_m_0, const float* __restrict__ M_v_0,
    const float* __restrict__ M_k_0, const float* __restrict__ M_eta_0,
    const float* __restrict__ M_alpha_0, const __nv_bfloat16* __restrict__ X,
    const float* __restrict__ Q, const __nv_bfloat16* __restrict__ QRaw,
    __nv_bfloat16* __restrict__ Y,
    float* __restrict__ G_prefix_chunk,
    float* __restrict__ K_saved, float* __restrict__ V_saved,
    float* __restrict__ Eta_saved, float* __restrict__ Alpha_saved,
    float* __restrict__ EtaGrad_saved, float* __restrict__ AlphaGrad_saved,
    float* __restrict__ InvK_saved,
    float* __restrict__ QInv_saved,
    float* __restrict__ G_token_saved, float* __restrict__ G_final_chunk,
    int N, int L, int ChunkSize, int NumChunks, int StateCount, int SaveG, int NumHeads, int SaveQInv)
{
    int pid = blockIdx.x;
    int d_i = threadIdx.x;
    if (pid >= N) return;

    extern __shared__ float dyn_smem_forward[];
    float* dyn_ptr = dyn_smem_forward;
    float (*smem_trans)[D][D + 1] = reinterpret_cast<float (*)[D][D + 1]>(dyn_ptr);
    dyn_ptr += 3 * D * (D + 1);
    float* s_x = dyn_ptr;
    dyn_ptr += D;
    float* s_q = dyn_ptr;
    dyn_ptr += D;
    float (*s_k_chunk)[D] = reinterpret_cast<float (*)[D]>(dyn_ptr);
    dyn_ptr += MAX_CHUNK * D;
    float (*s_v_chunk)[D] = reinterpret_cast<float (*)[D]>(dyn_ptr);
    dyn_ptr += MAX_CHUNK * D;
    float* s_eta_chunk_val = dyn_ptr;
    dyn_ptr += MAX_CHUNK;
    float* s_alpha_chunk_val = dyn_ptr;
    dyn_ptr += MAX_CHUNK;
    float (*s_g_final)[D + 1] = reinterpret_cast<float (*)[D + 1]>(dyn_ptr);
    dyn_ptr += D * (D + 1);
    float* s_eta_init_vec = dyn_ptr;
    dyn_ptr += D;
    float* s_alpha_init_vec = dyn_ptr;

    int state_pid = (StateCount == N) ? pid : (pid % StateCount);
    int state_stride = state_pid * D * D + d_i * D;
    int vec_stride = state_pid * D + d_i;

    float m_mem[D], m_v[D], m_k[D];
    for(int j=0; j<D; ++j){
        m_mem[j] = M_m_0[state_stride + j];
        m_v[j]   = M_v_0[state_stride + j];
        m_k[j]   = M_k_0[state_stride + j];
    }
    float s_eta = M_eta_0[vec_stride];
    float s_alpha = M_alpha_0[vec_stride];
    float prefix_g[D];
    #pragma unroll
    for(int j=0; j<D; ++j) {
        prefix_g[j] = (d_i == j) ? 1.0f : 0.0f;
    }

    int x_stride = pid * L * D;
    int q_b = 0;
    int q_dir = 0;
    int q_head = 0;
    int q_channels = 0;
    if constexpr (QRAW) {
        q_head = pid % NumHeads;
        int tmp = pid / NumHeads;
        q_dir = tmp % 4;
        q_b = tmp / 4;
        q_channels = NumHeads * D;
    }

    for (int c = 0; c < NumChunks; ++c) {
        if (c > 0) {
            float* out_prefix = G_prefix_chunk + pid * (NumChunks - 1) * D * D + (c - 1) * D * D;
            #pragma unroll
            for(int j=0; j<D; ++j) {
                out_prefix[d_i * D + j] = prefix_g[j];
            }
        }

        for(int j=0; j<D; ++j){
            smem_trans[0][d_i][j] = m_mem[j];
            smem_trans[1][d_i][j] = m_v[j];
            smem_trans[2][d_i][j] = m_k[j];
        }
        warpSync<D>();

        float m_eta_init = s_eta;
        float m_alpha_init = s_alpha;
        s_eta_init_vec[d_i] = m_eta_init;
        s_alpha_init_vec[d_i] = m_alpha_init;
        warpSync<D>();

        for (int t_inner = 0; t_inner < ChunkSize; ++t_inner) {
            int t = c * ChunkSize + t_inner;
            s_x[d_i] = __bfloat162float(X[x_stride + t * D + d_i]);
            if constexpr (QRAW) {
                int64_t q_idx = (((int64_t)q_b * 4 + q_dir) * L + t) * q_channels + q_head * D + d_i;
                float q_raw_i = __bfloat162float(QRaw[q_idx]);
                float q_norm_sq = warpReduceSum<D>(q_raw_i * q_raw_i);
                float q_inv = 1.0f / fmaxf(sqrtf(q_norm_sq), Q_NORM_EPS);
                if (SaveQInv && d_i == 0) {
                    QInv_saved[pid * L + t] = q_inv;
                }
                s_q[d_i] = q_raw_i * q_inv;
            } else {
                s_q[d_i] = Q[x_stride + t * D + d_i];
            }
            warpSync<D>();

            float k_raw_i = 0, v_i = 0, y_i = 0;
            #pragma unroll
            for(int j=0; j<D; ++j){
                k_raw_i += smem_trans[2][d_i][j] * s_x[j];
                v_i     += smem_trans[1][d_i][j] * s_x[j];
                y_i     += smem_trans[0][d_i][j] * s_q[j];
            }
            float k_norm_sq = warpReduceSum<D>(k_raw_i * k_raw_i);
            float inv_k_norm = rsqrtf(k_norm_sq + 1e-6f);
            s_k_chunk[t_inner][d_i] = k_raw_i * inv_k_norm;
            s_v_chunk[t_inner][d_i] = v_i;
            K_saved[x_stride + t * D + d_i] = s_k_chunk[t_inner][d_i];
            V_saved[x_stride + t * D + d_i] = v_i;
            Y[x_stride + t * D + d_i] = __float2bfloat16(y_i);

            float eta_pre_raw = warpReduceSum<D>(s_x[d_i] * m_eta_init);
            float alpha_pre_raw = warpReduceSum<D>(s_x[d_i] * m_alpha_init);
            float eta_pre = fminf(PRE_ACT_CLAMP, fmaxf(-PRE_ACT_CLAMP, eta_pre_raw));
            float alpha_pre_biased = alpha_pre_raw + ALPHA_BIAS;
            float alpha_pre = fminf(PRE_ACT_CLAMP, fmaxf(-PRE_ACT_CLAMP, alpha_pre_biased));

            if (d_i == 0) {
                int scalar_idx = pid * L + t;
                float eta_softplus = (eta_pre > 20.0f) ? eta_pre : logf(1.0f + expf(eta_pre));
                s_eta_chunk_val[t_inner] = ETA_SCALE * eta_softplus;
                float alpha_sigmoid = 1.0f / (1.0f + expf(-alpha_pre));
                s_alpha_chunk_val[t_inner] = ALPHA_MIN + (1.0f - ALPHA_MIN) * alpha_sigmoid;
                float eta_clamp_grad = (fabsf(eta_pre_raw) <= PRE_ACT_CLAMP) ? 1.0f : 0.0f;
                float alpha_clamp_grad = (fabsf(alpha_pre_biased) <= PRE_ACT_CLAMP) ? 1.0f : 0.0f;
                Eta_saved[scalar_idx] = s_eta_chunk_val[t_inner];
                Alpha_saved[scalar_idx] = s_alpha_chunk_val[t_inner];
                EtaGrad_saved[scalar_idx] =
                    ETA_SCALE * ((eta_pre > 20.0f) ? 1.0f : 1.0f / (1.0f + expf(-eta_pre))) * eta_clamp_grad;
                AlphaGrad_saved[scalar_idx] =
                    (1.0f - ALPHA_MIN) * alpha_sigmoid * (1.0f - alpha_sigmoid) * alpha_clamp_grad;
                InvK_saved[scalar_idx] = inv_k_norm;
            }
            warpSync<D>();
        }

        float g[D];
        #pragma unroll
        for(int j=0; j<D; ++j) {
            g[j] = (d_i == j) ? 1.0f : 0.0f;
        }

        for (int t_inner = 0; t_inner < ChunkSize; ++t_inner) {
            if (SaveG == 1 || (SaveG == 2 && (t_inner % RECOMP_TILE) == 0)) {
                int save_idx = (SaveG == 1) ? (c * ChunkSize + t_inner) : (c * MAX_SUBCHUNKS + (t_inner / RECOMP_TILE));
                float* out_g_token = G_token_saved + pid * ((SaveG == 1) ? L : (NumChunks * MAX_SUBCHUNKS)) * D * D + save_idx * D * D;
                #pragma unroll
                for(int j=0; j<D; ++j) {
                    out_g_token[d_i * D + j] = g[j];
                }
            }
            float sk_i = s_k_chunk[t_inner][d_i];
            float sv_i = s_v_chunk[t_inner][d_i];
            float eta = s_eta_chunk_val[t_inner];
            float alpha = s_alpha_chunk_val[t_inner];

            float diff_i = sk_i - sv_i;
            float r_i = diff_i;
            #pragma unroll
            for(int j=0; j<D; ++j) {
                float sk_j = s_k_chunk[t_inner][j];
                r_i += g[j] * sk_j;
            }
            float diff_norm = sqrtf(warpReduceSum<D>(diff_i * diff_i) + STAB_NORM_EPS);
            float eta_eff = soft_project_eta_warp<D>(eta, alpha, diff_norm);
#if ETA2ALPHA_CLAMP
            float key_norm_sq_eff = warpReduceSum<D>(sk_i * sk_i);
            eta_eff = clamp_eta_to_alpha(eta_eff, alpha, key_norm_sq_eff);
#endif

            #pragma unroll
            for(int j=0; j<D; ++j) {
                float sk_j = s_k_chunk[t_inner][j];
                g[j] = alpha * g[j] - eta_eff * r_i * sk_j;
            }
        }

        #pragma unroll
        for(int j=0; j<D; ++j) {
            s_g_final[d_i][j] = g[j];
        }
        if (SaveG) {
            float* out_g_final = G_final_chunk + pid * NumChunks * D * D + c * D * D;
            #pragma unroll
            for(int j=0; j<D; ++j) {
                out_g_final[d_i * D + j] = g[j];
            }
        }
        warpSync<D>();

        float next_prefix[D];
        #pragma unroll
        for(int j=0; j<D; ++j) {
            float acc = 0.0f;
            #pragma unroll
            for(int l=0; l<D; ++l) {
                acc += prefix_g[l] * s_g_final[l][j];
            }
            next_prefix[j] = acc;
        }

        #pragma unroll
        for(int j=0; j<D; ++j) {
            float next_mem = 0.0f;
            float next_v = 0.0f;
            float next_k = 0.0f;
            #pragma unroll
            for(int l=0; l<D; ++l) {
                float g_lj = s_g_final[l][j];
                next_mem += smem_trans[0][d_i][l] * g_lj;
                next_v   += smem_trans[1][d_i][l] * g_lj;
                next_k   += smem_trans[2][d_i][l] * g_lj;
            }
            m_mem[j] = next_mem;
            m_v[j] = next_v;
            m_k[j] = next_k;
        }

        float next_eta = 0.0f;
        float next_alpha = 0.0f;
        #pragma unroll
        for(int l=0; l<D; ++l) {
            float g_li = s_g_final[l][d_i];
            next_eta += s_eta_init_vec[l] * g_li;
            next_alpha += s_alpha_init_vec[l] * g_li;
        }
        s_eta = next_eta;
        s_alpha = next_alpha;
        #pragma unroll
        for(int j=0; j<D; ++j) {
            prefix_g[j] = next_prefix[j];
        }
    }
}

template <int D, bool QRAW>
__global__ void srnl_forward_kernel_tpar(
    const float* __restrict__ M_m_0, const float* __restrict__ M_v_0,
    const float* __restrict__ M_k_0, const float* __restrict__ M_eta_0,
    const float* __restrict__ M_alpha_0, const __nv_bfloat16* __restrict__ X,
    const float* __restrict__ Q, const __nv_bfloat16* __restrict__ QRaw,
    __nv_bfloat16* __restrict__ Y,
    float* __restrict__ G_prefix_chunk,
    float* __restrict__ K_saved, float* __restrict__ V_saved,
    float* __restrict__ Eta_saved, float* __restrict__ Alpha_saved,
    float* __restrict__ EtaGrad_saved, float* __restrict__ AlphaGrad_saved,
    float* __restrict__ InvK_saved,
    float* __restrict__ QInv_saved,
    float* __restrict__ G_token_saved, float* __restrict__ G_final_chunk,
    int N, int L, int ChunkSize, int NumChunks, int StateCount, int SaveG, int NumHeads, int SaveQInv)
{
    const int TGROUPS = blockDim.x / D;
    int group = threadIdx.x / D;
    int d_i = threadIdx.x & (D - 1);
    int pid = blockIdx.x;
    if (pid >= N) return;

    extern __shared__ float storage[];
    float (*smem_trans)[D][D+1] = reinterpret_cast<float (*)[D][D+1]>(storage);
    float* ptr = storage + 3 * D * (D+1);
    float (*s_x_group)[D] = reinterpret_cast<float (*)[D]>(ptr); ptr += TGROUPS * D;
    float (*s_q_group)[D] = reinterpret_cast<float (*)[D]>(ptr); ptr += TGROUPS * D;
    float (*s_k_chunk)[D] = reinterpret_cast<float (*)[D]>(ptr); ptr += MAX_CHUNK * D;
    float (*s_v_chunk)[D] = reinterpret_cast<float (*)[D]>(ptr); ptr += MAX_CHUNK * D;
    float* s_eta_chunk_val = ptr; ptr += MAX_CHUNK;
    float* s_alpha_chunk_val = ptr; ptr += MAX_CHUNK;
    float (*s_g_final)[D+1] = reinterpret_cast<float (*)[D+1]>(ptr); ptr += D * (D+1);
    float* s_eta_init_vec = ptr; ptr += D;
    float* s_alpha_init_vec = ptr;

    int state_pid = (StateCount == N) ? pid : (pid % StateCount);
    int state_stride = state_pid * D * D + d_i * D;
    int vec_stride = state_pid * D + d_i;
    int x_stride = pid * L * D;
    int q_b = 0;
    int q_dir = 0;
    int q_head = 0;
    int q_channels = 0;
    if constexpr (QRAW) {
        q_head = pid % NumHeads;
        int tmp = pid / NumHeads;
        q_dir = tmp % 4;
        q_b = tmp / 4;
        q_channels = NumHeads * D;
    }

    float m_mem[D], m_v[D], m_k[D];
    float s_eta = 0.0f;
    float s_alpha = 0.0f;
    float prefix_g[D];
    #pragma unroll
    for(int j=0; j<D; ++j) {
        prefix_g[j] = (d_i == j) ? 1.0f : 0.0f;
    }

    if (group == 0) {
        #pragma unroll
        for(int j=0; j<D; ++j){
            m_mem[j] = M_m_0[state_stride + j];
            m_v[j]   = M_v_0[state_stride + j];
            m_k[j]   = M_k_0[state_stride + j];
        }
        s_eta = M_eta_0[vec_stride];
        s_alpha = M_alpha_0[vec_stride];
    }

    for (int c = 0; c < NumChunks; ++c) {
        if (group == 0) {
            if (c > 0) {
                float* out_prefix = G_prefix_chunk + pid * (NumChunks - 1) * D * D + (c - 1) * D * D;
                #pragma unroll
                for(int j=0; j<D; ++j) {
                    out_prefix[d_i * D + j] = prefix_g[j];
                }
            }

            #pragma unroll
            for(int j=0; j<D; ++j){
                smem_trans[0][d_i][j] = m_mem[j];
                smem_trans[1][d_i][j] = m_v[j];
                smem_trans[2][d_i][j] = m_k[j];
            }
            s_eta_init_vec[d_i] = s_eta;
            s_alpha_init_vec[d_i] = s_alpha;
        }
        __syncthreads();

        for (int t_base = 0; t_base < ChunkSize; t_base += TGROUPS) {
            int t_inner = t_base + group;
            bool valid_t = t_inner < ChunkSize;
            if (valid_t) {
                int t = c * ChunkSize + t_inner;
                s_x_group[group][d_i] = __bfloat162float(X[x_stride + t * D + d_i]);
                if constexpr (QRAW) {
                    int64_t q_idx = (((int64_t)q_b * 4 + q_dir) * L + t) * q_channels + q_head * D + d_i;
                    float q_raw_i = __bfloat162float(QRaw[q_idx]);
                    float q_norm_sq = srnl_group_sum<D>(q_raw_i * q_raw_i, d_i, group);
                    float q_inv = 1.0f / fmaxf(sqrtf(q_norm_sq), Q_NORM_EPS);
                    if (SaveQInv && d_i == 0) {
                        QInv_saved[pid * L + c * ChunkSize + t_inner] = q_inv;
                    }
                    s_q_group[group][d_i] = q_raw_i * q_inv;
                } else {
                    s_q_group[group][d_i] = Q[x_stride + t * D + d_i];
                }
            }
            srnl_group_sync<D>(group);

            if (valid_t) {
                float k_raw_i = 0.0f, v_i = 0.0f, y_i = 0.0f;
                #pragma unroll
                for(int j=0; j<D; ++j){
                    float x_j = s_x_group[group][j];
                    k_raw_i += smem_trans[2][d_i][j] * x_j;
                    v_i     += smem_trans[1][d_i][j] * x_j;
                    y_i     += smem_trans[0][d_i][j] * s_q_group[group][j];
                }
                float k_norm_sq = srnl_group_sum<D>(k_raw_i * k_raw_i, d_i, group);
                float inv_k_norm = rsqrtf(k_norm_sq + 1e-6f);
                s_k_chunk[t_inner][d_i] = k_raw_i * inv_k_norm;
                s_v_chunk[t_inner][d_i] = v_i;
                K_saved[x_stride + (c * ChunkSize + t_inner) * D + d_i] = s_k_chunk[t_inner][d_i];
                V_saved[x_stride + (c * ChunkSize + t_inner) * D + d_i] = v_i;
                Y[x_stride + (c * ChunkSize + t_inner) * D + d_i] = __float2bfloat16(y_i);

                float eta_pre_raw = srnl_group_sum<D>(s_x_group[group][d_i] * s_eta_init_vec[d_i], d_i, group);
                float alpha_pre_raw = srnl_group_sum<D>(s_x_group[group][d_i] * s_alpha_init_vec[d_i], d_i, group);
                float eta_pre = fminf(PRE_ACT_CLAMP, fmaxf(-PRE_ACT_CLAMP, eta_pre_raw));
                float alpha_pre_biased = alpha_pre_raw + ALPHA_BIAS;
                float alpha_pre = fminf(PRE_ACT_CLAMP, fmaxf(-PRE_ACT_CLAMP, alpha_pre_biased));

                if (d_i == 0) {
                    int scalar_idx = pid * L + c * ChunkSize + t_inner;
                    float eta_softplus = (eta_pre > 20.0f) ? eta_pre : logf(1.0f + expf(eta_pre));
                    s_eta_chunk_val[t_inner] = ETA_SCALE * eta_softplus;
                    float alpha_sigmoid = 1.0f / (1.0f + expf(-alpha_pre));
                    s_alpha_chunk_val[t_inner] = ALPHA_MIN + (1.0f - ALPHA_MIN) * alpha_sigmoid;
                    float eta_clamp_grad = (fabsf(eta_pre_raw) <= PRE_ACT_CLAMP) ? 1.0f : 0.0f;
                    float alpha_clamp_grad = (fabsf(alpha_pre_biased) <= PRE_ACT_CLAMP) ? 1.0f : 0.0f;
                    Eta_saved[scalar_idx] = s_eta_chunk_val[t_inner];
                    Alpha_saved[scalar_idx] = s_alpha_chunk_val[t_inner];
                    EtaGrad_saved[scalar_idx] =
                        ETA_SCALE * ((eta_pre > 20.0f) ? 1.0f : 1.0f / (1.0f + expf(-eta_pre))) * eta_clamp_grad;
                    AlphaGrad_saved[scalar_idx] =
                        (1.0f - ALPHA_MIN) * alpha_sigmoid * (1.0f - alpha_sigmoid) * alpha_clamp_grad;
                    InvK_saved[scalar_idx] = inv_k_norm;
                }
            }
        }
        __syncthreads();

        if (group == 0) {
            float g[D];
            #pragma unroll
            for(int j=0; j<D; ++j) {
                g[j] = (d_i == j) ? 1.0f : 0.0f;
            }

            for (int t_inner = 0; t_inner < ChunkSize; ++t_inner) {
                if (SaveG == 1 || (SaveG == 2 && (t_inner % RECOMP_TILE) == 0)) {
                    int save_idx = (SaveG == 1) ? (c * ChunkSize + t_inner) : (c * MAX_SUBCHUNKS + (t_inner / RECOMP_TILE));
                    float* out_g_token = G_token_saved + pid * ((SaveG == 1) ? L : (NumChunks * MAX_SUBCHUNKS)) * D * D + save_idx * D * D;
                    #pragma unroll
                    for(int j=0; j<D; ++j) {
                        out_g_token[d_i * D + j] = g[j];
                    }
                }
                float sk_i = s_k_chunk[t_inner][d_i];
                float sv_i = s_v_chunk[t_inner][d_i];
                float eta = s_eta_chunk_val[t_inner];
                float alpha = s_alpha_chunk_val[t_inner];

                float diff_i = sk_i - sv_i;
                float r_i = diff_i;
                #pragma unroll
                for(int j=0; j<D; ++j) {
                    float sk_j = s_k_chunk[t_inner][j];
                    r_i += g[j] * sk_j;
                }
                float diff_norm = sqrtf(srnl_group_sum<D>(diff_i * diff_i, d_i, 0) + STAB_NORM_EPS);
                float eta_eff = d_i == 0 ? soft_project_eta(eta, alpha, diff_norm) : 0.0f;
                eta_eff = srnl_group_first<D>(eta_eff, d_i, 0);
#if ETA2ALPHA_CLAMP
                float key_norm_sq_eff = srnl_group_sum<D>(sk_i * sk_i, d_i, 0);
                eta_eff = clamp_eta_to_alpha(eta_eff, alpha, key_norm_sq_eff);
#endif

                #pragma unroll
                for(int j=0; j<D; ++j) {
                    float sk_j = s_k_chunk[t_inner][j];
                    g[j] = alpha * g[j] - eta_eff * r_i * sk_j;
                }
            }

            #pragma unroll
            for(int j=0; j<D; ++j) {
                s_g_final[d_i][j] = g[j];
            }
            if (SaveG) {
                float* out_g_final = G_final_chunk + pid * NumChunks * D * D + c * D * D;
                #pragma unroll
                for(int j=0; j<D; ++j) {
                    out_g_final[d_i * D + j] = g[j];
                }
            }
            srnl_group_sync<D>(0);

            float next_prefix[D];
            #pragma unroll
            for(int j=0; j<D; ++j) {
                float acc = 0.0f;
                #pragma unroll
                for(int l=0; l<D; ++l) {
                    acc += prefix_g[l] * s_g_final[l][j];
                }
                next_prefix[j] = acc;
            }

            #pragma unroll
            for(int j=0; j<D; ++j) {
                float next_mem = 0.0f;
                float next_v = 0.0f;
                float next_k = 0.0f;
                #pragma unroll
                for(int l=0; l<D; ++l) {
                    float g_lj = s_g_final[l][j];
                    next_mem += smem_trans[0][d_i][l] * g_lj;
                    next_v   += smem_trans[1][d_i][l] * g_lj;
                    next_k   += smem_trans[2][d_i][l] * g_lj;
                }
                m_mem[j] = next_mem;
                m_v[j] = next_v;
                m_k[j] = next_k;
            }

            float next_eta = 0.0f;
            float next_alpha = 0.0f;
            #pragma unroll
            for(int l=0; l<D; ++l) {
                float g_li = s_g_final[l][d_i];
                next_eta += s_eta_init_vec[l] * g_li;
                next_alpha += s_alpha_init_vec[l] * g_li;
            }
            s_eta = next_eta;
            s_alpha = next_alpha;
            #pragma unroll
            for(int j=0; j<D; ++j) {
                prefix_g[j] = next_prefix[j];
            }
        }
        __syncthreads();
    }
}

template <int D, bool QRAW, int GROUPS, int CMAX>
__global__ void srnl_backward_kernel(
    const __nv_bfloat16* __restrict__ dY, const __nv_bfloat16* __restrict__ X,
    const float* __restrict__ Q, const __nv_bfloat16* __restrict__ QRaw,
    const float* __restrict__ M_m_0, const float* __restrict__ M_v_0,
    const float* __restrict__ M_k_0, const float* __restrict__ M_eta_0,
    const float* __restrict__ M_alpha_0, const float* __restrict__ G_prefix_chunk,
    const float* __restrict__ K_saved, const float* __restrict__ V_saved,
    const float* __restrict__ Eta_saved, const float* __restrict__ Alpha_saved,
    const float* __restrict__ EtaGrad_saved, const float* __restrict__ AlphaGrad_saved,
    const float* __restrict__ InvK_saved,
    const float* __restrict__ QInv_saved,
    const float* __restrict__ G_token_saved, const float* __restrict__ G_final_chunk,
    __nv_bfloat16* __restrict__ dX, float* __restrict__ dQ, __nv_bfloat16* __restrict__ dQRaw,
    float* __restrict__ dM_m_0, float* __restrict__ dM_v_0, float* __restrict__ dM_k_0,
    float* __restrict__ dM_eta_0, float* __restrict__ dM_alpha_0,
    int N, int L, int ChunkSize, int NumChunks, int StateCount, int SaveG, int NumHeads, int SaveQInv)
{
    int group = threadIdx.x / D;
    int d_i = threadIdx.x - group * D;
    int groups_per_block = blockDim.x / D;
    int pid = blockIdx.x * groups_per_block + group;
    if (pid >= N) return;

    float g_mem[D] = {0}, g_m_v[D] = {0}, g_m_k[D] = {0};
    float g_m_eta = 0, g_m_alpha = 0;

    extern __shared__ float dyn_smem_backward[];
    float* dyn_ptr = dyn_smem_backward;
    float (*s_m_init_storage)[3][D][D + 1] =
        reinterpret_cast<float (*)[3][D][D + 1]>(dyn_ptr);
    dyn_ptr += GROUPS * 3 * D * (D + 1);
    float (*smem_red_storage)[D][D + 1] =
        reinterpret_cast<float (*)[D][D + 1]>(dyn_ptr);
    dyn_ptr += GROUPS * D * (D + 1);
    float (*s_x_storage)[D] = reinterpret_cast<float (*)[D]>(dyn_ptr);
    dyn_ptr += GROUPS * D;
    float (*s_q_storage)[D] = reinterpret_cast<float (*)[D]>(dyn_ptr);
    dyn_ptr += GROUPS * D;
    float (*s_dy_storage)[D] = reinterpret_cast<float (*)[D]>(dyn_ptr);
    dyn_ptr += GROUPS * D;
    float (*s_g_mat_storage)[D][D + 1] =
        reinterpret_cast<float (*)[D][D + 1]>(dyn_ptr);
    dyn_ptr += GROUPS * D * (D + 1);
    float (*s_vec_grad_storage)[2][D] = reinterpret_cast<float (*)[2][D]>(dyn_ptr);
    dyn_ptr += GROUPS * 2 * D;
    float (*s_eta_init_vec_storage)[D] = reinterpret_cast<float (*)[D]>(dyn_ptr);
    dyn_ptr += GROUPS * D;
    float (*s_alpha_init_vec_storage)[D] = reinterpret_cast<float (*)[D]>(dyn_ptr);
    dyn_ptr += GROUPS * D;
    float (*s_r_vec_storage)[D] = reinterpret_cast<float (*)[D]>(dyn_ptr);
    dyn_ptr += GROUPS * D;
    float (*s_bar_r_vec_storage)[D] = reinterpret_cast<float (*)[D]>(dyn_ptr);
    dyn_ptr += GROUPS * D;

    float (*s_k_chunk_storage)[CMAX][D] =
        reinterpret_cast<float (*)[CMAX][D]>(dyn_ptr);
    dyn_ptr += GROUPS * CMAX * D;
    float (*s_v_chunk_storage)[CMAX][D] =
        reinterpret_cast<float (*)[CMAX][D]>(dyn_ptr);
    dyn_ptr += GROUPS * CMAX * D;
    float (*s_eta_chunk_val_storage)[CMAX] = reinterpret_cast<float (*)[CMAX]>(dyn_ptr);
    dyn_ptr += GROUPS * CMAX;
    float (*s_alpha_chunk_val_storage)[CMAX] = reinterpret_cast<float (*)[CMAX]>(dyn_ptr);
    dyn_ptr += GROUPS * CMAX;
    float (*s_eta_grad_chunk_storage)[CMAX] = reinterpret_cast<float (*)[CMAX]>(dyn_ptr);
    dyn_ptr += GROUPS * CMAX;
    float (*s_alpha_grad_chunk_storage)[CMAX] = reinterpret_cast<float (*)[CMAX]>(dyn_ptr);
    dyn_ptr += GROUPS * CMAX;
    float (*s_k_norm_chunk_storage)[CMAX] = reinterpret_cast<float (*)[CMAX]>(dyn_ptr);

    float (*s_m_init)[D][D + 1] = s_m_init_storage[group];
    float (*smem_red)[D + 1] = smem_red_storage[group];
    float* s_x = s_x_storage[group];
    float* s_q = s_q_storage[group];
    float* s_dy = s_dy_storage[group];
    float (*s_g_mat)[D + 1] = s_g_mat_storage[group];
    float (*s_vec_grad)[D] = s_vec_grad_storage[group];
    float* s_eta_init_vec = s_eta_init_vec_storage[group];
    float* s_alpha_init_vec = s_alpha_init_vec_storage[group];
    float* s_r_vec = s_r_vec_storage[group];
    float* s_bar_r_vec = s_bar_r_vec_storage[group];

    float (*s_k_chunk)[D] = s_k_chunk_storage[group];
    float (*s_v_chunk)[D] = s_v_chunk_storage[group];
    float* s_eta_chunk_val = s_eta_chunk_val_storage[group];
    float* s_alpha_chunk_val = s_alpha_chunk_val_storage[group];
    float* s_eta_grad_chunk = s_eta_grad_chunk_storage[group];
    float* s_alpha_grad_chunk = s_alpha_grad_chunk_storage[group];
    float* s_k_norm_chunk = s_k_norm_chunk_storage[group];

    constexpr int LOCAL_MAX_SUBCHUNKS = (CMAX + RECOMP_TILE - 1) / RECOMP_TILE;
    float g_checkpoint[LOCAL_MAX_SUBCHUNKS][D];
    float g_traj[RECOMP_TILE][D];

    int x_stride = pid * L * D;
    int state_pid = (StateCount == N) ? pid : (pid % StateCount);
    int state_stride = state_pid * D * D + d_i * D;
    int q_b = 0;
    int q_dir = 0;
    int q_head = 0;
    int q_channels = 0;
    if constexpr (QRAW) {
        q_head = pid % NumHeads;
        int tmp = pid / NumHeads;
        q_dir = tmp % 4;
        q_b = tmp / 4;
        q_channels = NumHeads * D;
    }

    for (int c = NumChunks - 1; c >= 0; --c) {
        if (c == 0) {
            #pragma unroll
            for(int j=0; j<D; ++j) {
                s_g_mat[d_i][j] = (d_i == j) ? 1.0f : 0.0f;
                s_m_init[0][d_i][j] = M_m_0[state_stride + j];
                s_m_init[1][d_i][j] = M_v_0[state_stride + j];
                s_m_init[2][d_i][j] = M_k_0[state_stride + j];
            }
        } else {
            const float* in_prefix = G_prefix_chunk + pid * (NumChunks - 1) * D * D + (c - 1) * D * D;
            #pragma unroll
            for(int j=0; j<D; ++j) {
                s_g_mat[d_i][j] = in_prefix[d_i * D + j];
            }
            warpSync<D>(group);

            #pragma unroll
            for(int j=0; j<D; ++j) {
                float mem_acc = 0.0f;
                float v_acc = 0.0f;
                float k_acc = 0.0f;
                #pragma unroll
                for(int l=0; l<D; ++l) {
                    float p_lj = s_g_mat[l][j];
                    mem_acc += M_m_0[state_stride + l] * p_lj;
                    v_acc   += M_v_0[state_stride + l]   * p_lj;
                    k_acc   += M_k_0[state_stride + l]   * p_lj;
                }
                s_m_init[0][d_i][j] = mem_acc;
                s_m_init[1][d_i][j] = v_acc;
                s_m_init[2][d_i][j] = k_acc;
            }
        }
        warpSync<D>(group);

        float m_eta_init = 0.0f;
        float m_alpha_init = 0.0f;
        #pragma unroll
        for(int l=0; l<D; ++l) {
            float p_li = s_g_mat[l][d_i];
            m_eta_init += M_eta_0[state_pid * D + l] * p_li;
            m_alpha_init += M_alpha_0[state_pid * D + l] * p_li;
        }
        s_eta_init_vec[d_i] = m_eta_init;
        s_alpha_init_vec[d_i] = m_alpha_init;
        warpSync<D>(group);

        for (int t_inner = 0; t_inner < ChunkSize; ++t_inner) {
            int t = c * ChunkSize + t_inner;
            s_k_chunk[t_inner][d_i] = K_saved[x_stride + t * D + d_i];
            s_v_chunk[t_inner][d_i] = V_saved[x_stride + t * D + d_i];
            if (d_i == 0) {
                int scalar_idx = pid * L + t;
                s_k_norm_chunk[t_inner] = InvK_saved[scalar_idx];
                s_eta_chunk_val[t_inner] = Eta_saved[scalar_idx];
                s_alpha_chunk_val[t_inner] = Alpha_saved[scalar_idx];
                s_eta_grad_chunk[t_inner] = EtaGrad_saved[scalar_idx];
                s_alpha_grad_chunk[t_inner] = AlphaGrad_saved[scalar_idx];
            }
            warpSync<D>(group);
        }

        int num_subchunks = (ChunkSize + RECOMP_TILE - 1) / RECOMP_TILE;
        float g[D];
        if (SaveG) {
            const float* in_g_final = G_final_chunk + pid * NumChunks * D * D + c * D * D;
            #pragma unroll
            for(int j=0; j<D; ++j) {
                s_g_mat[d_i][j] = in_g_final[d_i * D + j];
            }
        } else {
            #pragma unroll
            for(int j=0; j<D; ++j) {
                g[j] = (d_i == j) ? 1.0f : 0.0f;
            }

            for (int sub = 0; sub < num_subchunks; ++sub) {
                for(int j=0; j<D; ++j) {
                    g_checkpoint[sub][j] = g[j];
                }

                int sub_start = sub * RECOMP_TILE;
                int sub_end = (sub_start + RECOMP_TILE < ChunkSize) ? (sub_start + RECOMP_TILE) : ChunkSize;

                for (int t_inner = sub_start; t_inner < sub_end; ++t_inner) {
                    float sk_i = s_k_chunk[t_inner][d_i];
                    float sv_i = s_v_chunk[t_inner][d_i];
                    float eta = s_eta_chunk_val[t_inner];
                    float alpha = s_alpha_chunk_val[t_inner];

                    float diff_i = sk_i - sv_i;
                    float r_i = diff_i;
                    #pragma unroll
                    for(int j=0; j<D; ++j) {
                        float sk_j = s_k_chunk[t_inner][j];
                        r_i += g[j] * sk_j;
                    }
                    float diff_norm = sqrtf(warpReduceSum<D>(diff_i * diff_i, group) + STAB_NORM_EPS);
                    float eta_eff = soft_project_eta_warp<D>(eta, alpha, diff_norm, group, d_i);
#if ETA2ALPHA_CLAMP
                    float key_norm_sq_eff = warpReduceSum<D>(sk_i * sk_i, group);
                    eta_eff = clamp_eta_to_alpha(eta_eff, alpha, key_norm_sq_eff);
#endif

                    #pragma unroll
                    for(int j=0; j<D; ++j) {
                        float sk_j = s_k_chunk[t_inner][j];
                        g[j] = alpha * g[j] - eta_eff * r_i * sk_j;
                    }
                }
            }

            #pragma unroll
            for(int j=0; j<D; ++j) {
                s_g_mat[d_i][j] = g[j];
            }
        }
        s_vec_grad[0][d_i] = g_m_eta;
        s_vec_grad[1][d_i] = g_m_alpha;
        warpSync<D>(group);

        float dM_mem_init[D] = {0}, dM_v_init[D] = {0}, dM_k_init[D] = {0};
        float dM_eta_init = 0, dM_alpha_init = 0;

        #pragma unroll
        for(int j=0; j<D; ++j) {
            float mem_acc = 0.0f;
            float v_acc = 0.0f;
            float k_acc = 0.0f;
            #pragma unroll
            for(int l=0; l<D; ++l) {
                float g_jl = s_g_mat[j][l];
                mem_acc += g_mem[l] * g_jl;
                v_acc += g_m_v[l] * g_jl;
                k_acc += g_m_k[l] * g_jl;
            }
            dM_mem_init[j] = mem_acc;
            dM_v_init[j] = v_acc;
            dM_k_init[j] = k_acc;
        }

        #pragma unroll
        for(int l=0; l<D; ++l) {
            float g_il = s_g_mat[d_i][l];
            dM_eta_init += s_vec_grad[0][l] * g_il;
            dM_alpha_init += s_vec_grad[1][l] * g_il;
        }

        float bg[D] = {0};
        #pragma unroll
        for(int j=0; j<D; ++j) smem_red[d_i][j] = g_mem[j];
        warpSync<D>(group);
        #pragma unroll
        for(int j=0; j<D; ++j) {
            float acc = 0.0f;
            #pragma unroll
            for(int r=0; r<D; ++r) acc += s_m_init[0][r][d_i] * smem_red[r][j];
            bg[j] += acc;
        }
        warpSync<D>(group);
        #pragma unroll
        for(int j=0; j<D; ++j) smem_red[d_i][j] = g_m_v[j];
        warpSync<D>(group);
        #pragma unroll
        for(int j=0; j<D; ++j) {
            float acc = 0.0f;
            #pragma unroll
            for(int r=0; r<D; ++r) acc += s_m_init[1][r][d_i] * smem_red[r][j];
            bg[j] += acc;
        }
        warpSync<D>(group);
        #pragma unroll
        for(int j=0; j<D; ++j) smem_red[d_i][j] = g_m_k[j];
        warpSync<D>(group);
        #pragma unroll
        for(int j=0; j<D; ++j) {
            float acc = 0.0f;
            #pragma unroll
            for(int r=0; r<D; ++r) acc += s_m_init[2][r][d_i] * smem_red[r][j];
            bg[j] += acc + m_eta_init * s_vec_grad[0][j] + m_alpha_init * s_vec_grad[1][j];
        }
        warpSync<D>(group);

        for (int sub_rev = 0; sub_rev < num_subchunks; ++sub_rev) {
            int sub = num_subchunks - 1 - sub_rev;
            int sub_start = sub * RECOMP_TILE;
            int sub_end = (sub_start + RECOMP_TILE < ChunkSize) ? (sub_start + RECOMP_TILE) : ChunkSize;
            int sub_len = sub_end - sub_start;

            if (!SaveG || SaveG == 2) {
                if (SaveG == 2) {
                    const float* in_g_checkpoint = G_token_saved + pid * (NumChunks * MAX_SUBCHUNKS) * D * D + (c * MAX_SUBCHUNKS + sub) * D * D;
                    #pragma unroll
                    for(int j=0; j<D; ++j) {
                        g[j] = in_g_checkpoint[d_i * D + j];
                    }
                } else {
                    for(int j=0; j<D; ++j) {
                        g[j] = g_checkpoint[sub][j];
                    }
                }

                for (int local_idx = 0; local_idx < sub_len; ++local_idx) {
                    int t_inner = sub_start + local_idx;

                    #pragma unroll
                    for(int j=0; j<D; ++j) {
                        g_traj[local_idx][j] = g[j];
                    }

                    float sk_i = s_k_chunk[t_inner][d_i];
                    float sv_i = s_v_chunk[t_inner][d_i];
                    float eta = s_eta_chunk_val[t_inner];
                    float alpha = s_alpha_chunk_val[t_inner];

                    float diff_i = sk_i - sv_i;
                    float r_i = diff_i;
                    #pragma unroll
                    for(int j=0; j<D; ++j) {
                        float sk_j = s_k_chunk[t_inner][j];
                        r_i += g[j] * sk_j;
                    }
                    float diff_norm = sqrtf(warpReduceSum<D>(diff_i * diff_i, group) + STAB_NORM_EPS);
                    float eta_eff = soft_project_eta_warp<D>(eta, alpha, diff_norm, group, d_i);
#if ETA2ALPHA_CLAMP
                    float key_norm_sq_eff = warpReduceSum<D>(sk_i * sk_i, group);
                    eta_eff = clamp_eta_to_alpha(eta_eff, alpha, key_norm_sq_eff);
#endif

                    for(int j=0; j<D; ++j) {
                        float sk_j = s_k_chunk[t_inner][j];
                        g[j] = alpha * g[j] - eta_eff * r_i * sk_j;
                    }
                }
            }

            for (int local_rev = 0; local_rev < sub_len; ++local_rev) {
                int local_idx = sub_len - 1 - local_rev;
                int t_inner = sub_start + local_idx;
                int t = c * ChunkSize + t_inner;

                if (SaveG == 1) {
                    const float* in_g_token = G_token_saved + pid * L * D * D + t * D * D;
                    #pragma unroll
                    for(int j=0; j<D; ++j) {
                        g[j] = in_g_token[d_i * D + j];
                    }
                } else {
                    for(int j=0; j<D; ++j) {
                        g[j] = g_traj[local_idx][j];
                    }
                }

                int64_t q_idx = 0;
                float q_inv = 0.0f;
                float q_i = 0.0f;
                s_x[d_i] = __bfloat162float(X[x_stride + t * D + d_i]);
                if constexpr (QRAW) {
                    q_idx = (((int64_t)q_b * 4 + q_dir) * L + t) * q_channels + q_head * D + d_i;
                    float q_raw_i = __bfloat162float(QRaw[q_idx]);
                    if (SaveQInv) {
                        q_inv = QInv_saved[pid * L + t];
                    } else {
                        float q_norm_sq = warpReduceSum<D>(q_raw_i * q_raw_i, group);
                        q_inv = 1.0f / fmaxf(sqrtf(q_norm_sq), Q_NORM_EPS);
                    }
                    q_i = q_raw_i * q_inv;
                    s_q[d_i] = q_i;
                } else {
                    s_q[d_i] = Q[x_stride + t * D + d_i];
                }
                s_dy[d_i] = __bfloat162float(dY[x_stride + t * D + d_i]);
                warpSync<D>(group);

                float sk_i = s_k_chunk[t_inner][d_i];
                float sv_i = s_v_chunk[t_inner][d_i];
                float eta = s_eta_chunk_val[t_inner];
                float alpha = s_alpha_chunk_val[t_inner];

                float diff_i = sk_i - sv_i;
                float r_i = diff_i;
                for(int j=0; j<D; ++j) {
                    float sk_j = s_k_chunk[t_inner][j];
                    r_i += g[j] * sk_j;
                }
                float diff_norm = sqrtf(warpReduceSum<D>(diff_i * diff_i, group) + STAB_NORM_EPS);
                float diff_denom = 0.0f;
                float eta_cap = 0.0f;
                float eta_eff = 0.0f;
                float d_etaeff_deta = 0.0f;
                float d_etaeff_dcap = 0.0f;
#if ETA2ALPHA_CLAMP
                float key_norm_sq_eff = warpReduceSum<D>(sk_i * sk_i, group);
                float d_etaeff_dalpha_direct = 0.0f;
                float d_etaeff_dkeynormsq = 0.0f;
                projected_eta_backward_warp<D>(
                    eta, alpha, diff_norm,
                    key_norm_sq_eff,
                    &diff_denom, &eta_cap, &eta_eff,
                    &d_etaeff_deta, &d_etaeff_dcap,
                    &d_etaeff_dalpha_direct, &d_etaeff_dkeynormsq,
                    group, d_i);
#else
                soft_project_eta_backward_warp<D>(
                    eta, alpha, diff_norm,
                    &diff_denom, &eta_cap, &eta_eff,
                    &d_etaeff_deta, &d_etaeff_dcap,
                    group, d_i);
#endif

                #pragma unroll
                for(int j=0; j<D; ++j) smem_red[d_i][j] = s_m_init[0][d_i][j] * s_dy[d_i];
                warpSync<D>(group);
                float dq_i = 0;
                #pragma unroll
                for(int i=0; i<D; ++i) dq_i += smem_red[i][d_i];
                warpSync<D>(group);

                #pragma unroll
                for(int j=0; j<D; ++j) dM_mem_init[j] += s_dy[d_i] * s_q[j];

                float h_k_i = 0.0f;
                float alpha_row = 0.0f;
                #pragma unroll
                for(int j=0; j<D; ++j) {
                    float sk_j = s_k_chunk[t_inner][j];
                    h_k_i += bg[j] * sk_j;
                    alpha_row += bg[j] * g[j];
                    smem_red[d_i][j] = bg[j];
                    s_g_mat[d_i][j] = g[j];
                }
                s_r_vec[d_i] = r_i;
                warpSync<D>(group);

                float h_t_r_i = 0.0f;
                float g_t_bar_r_i = 0.0f;
                float d_alpha = warpReduceSum<D>(alpha_row, group);
                float d_eta_eff = -warpReduceSum<D>(r_i * h_k_i, group);
                float bar_r_i = -eta_eff * h_k_i;
                s_bar_r_vec[d_i] = bar_r_i;
                warpSync<D>(group);

                #pragma unroll
                for(int r=0; r<D; ++r) {
                    h_t_r_i += smem_red[r][d_i] * s_r_vec[r];
                    g_t_bar_r_i += s_g_mat[r][d_i] * s_bar_r_vec[r];
                }

                float d_eta = d_eta_eff * d_etaeff_deta;
                float d_eta_cap = d_eta_eff * d_etaeff_dcap;
                d_alpha += d_eta_cap * (-STAB_MU / diff_denom);
#if ETA2ALPHA_CLAMP
                d_alpha += d_eta_eff * d_etaeff_dalpha_direct;
#endif
                float d_diff_norm = d_eta_cap * (-eta_cap / diff_denom);
#if ETA2ALPHA_CLAMP
                float d_key_norm_sq = d_eta_eff * d_etaeff_dkeynormsq;
#endif

                float dk_i = -eta_eff * h_t_r_i + g_t_bar_r_i + bar_r_i;
                float dv_i = -bar_r_i;

                float inv_diff_norm = 1.0f / diff_norm;
                float diff_norm_grad_i = d_diff_norm * diff_i * inv_diff_norm;
                dk_i += diff_norm_grad_i;
                dv_i -= diff_norm_grad_i;
#if ETA2ALPHA_CLAMP
                dk_i += 2.0f * d_key_norm_sq * sk_i;
#endif

                #pragma unroll
                for(int j=0; j<D; ++j) {
                    float sk_j = s_k_chunk[t_inner][j];
                    bg[j] = alpha * bg[j] + bar_r_i * sk_j;
                }

                float dot = warpReduceSum<D>(dk_i * sk_i, group);
                float dk_raw_i = (dk_i - dot * sk_i) * s_k_norm_chunk[t_inner];

                float d_alpha_pre = d_alpha * s_alpha_grad_chunk[t_inner];
                float d_eta_pre = d_eta * s_eta_grad_chunk[t_inner];

                #pragma unroll
                for(int j=0; j<D; ++j) {
                    dM_v_init[j] += dv_i * s_x[j];
                    dM_k_init[j] += dk_raw_i * s_x[j];
                    smem_red[d_i][j] = s_m_init[1][d_i][j] * dv_i + s_m_init[2][d_i][j] * dk_raw_i;
                }
                dM_eta_init += s_x[d_i] * d_eta_pre;
                dM_alpha_init += s_x[d_i] * d_alpha_pre;
                warpSync<D>(group);

                float dx_i = d_eta_pre * m_eta_init + d_alpha_pre * m_alpha_init;
                #pragma unroll
                for(int i=0; i<D; ++i) dx_i += smem_red[i][d_i];

                dX[x_stride + t * D + d_i] = __float2bfloat16(dx_i);
                if constexpr (QRAW) {
                    float q_dot = warpReduceSum<D>(dq_i * q_i, group);
                    float dq_raw_i = (q_inv > 9.999e11f) ? (dq_i * q_inv) : ((dq_i - q_i * q_dot) * q_inv);
                    dQRaw[q_idx] = __float2bfloat16(dq_raw_i);
                } else {
                    dQ[x_stride + t * D + d_i] = dq_i;
                }
            }
        }

        for(int j=0; j<D; ++j){
            g_mem[j] = dM_mem_init[j];
            g_m_v[j] = dM_v_init[j];
            g_m_k[j] = dM_k_init[j];
        }
        g_m_eta = dM_eta_init;
        g_m_alpha = dM_alpha_init;
        warpSync<D>(group);
    }

    int out_stride = state_pid * D * D;
    if (StateCount == N) {
        for(int j=0; j<D; ++j) {
            dM_m_0[out_stride + d_i * D + j] = g_mem[j];
            dM_v_0[out_stride + d_i * D + j]   = g_m_v[j];
            dM_k_0[out_stride + d_i * D + j]   = g_m_k[j];
        }
        dM_eta_0[state_pid * D + d_i] = g_m_eta;
        dM_alpha_0[state_pid * D + d_i] = g_m_alpha;
    } else {
        for(int j=0; j<D; ++j) {
            atomicAdd(&dM_m_0[out_stride + d_i * D + j], g_mem[j]);
            atomicAdd(&dM_v_0[out_stride + d_i * D + j], g_m_v[j]);
            atomicAdd(&dM_k_0[out_stride + d_i * D + j], g_m_k[j]);
        }
        atomicAdd(&dM_eta_0[state_pid * D + d_i], g_m_eta);
        atomicAdd(&dM_alpha_0[state_pid * D + d_i], g_m_alpha);
    }
}

#define DISPATCH_SRNL_FWD(D_VAL) \
    do { \
    int groups = srnl_parallel_groups(D_VAL, chunk_size, 128); int smem_bytes = srnl_forward_smem_bytes(D_VAL) + 2 * (groups - 1) * D_VAL * sizeof(float); \
    set_forward_dynamic_smem_attr_if_needed((const void*)srnl_forward_kernel_tpar<D_VAL, false>, smem_bytes); \
    srnl_forward_kernel_tpar<D_VAL, false><<<N, D_VAL * groups, smem_bytes, stream>>>( \
        M_m_0.data_ptr<float>(), M_v_0.data_ptr<float>(), M_k_0.data_ptr<float>(), M_eta_0.data_ptr<float>(), M_alpha_0.data_ptr<float>(), \
        reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>()), Q.data_ptr<float>(), reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>()), reinterpret_cast<__nv_bfloat16*>(Y.data_ptr<at::BFloat16>()), G_prefix_chunk.data_ptr<float>(), \
        K_saved.data_ptr<float>(), V_saved.data_ptr<float>(), Eta_saved.data_ptr<float>(), Alpha_saved.data_ptr<float>(), EtaGrad_saved.data_ptr<float>(), AlphaGrad_saved.data_ptr<float>(), InvK_saved.data_ptr<float>(), InvK_saved.data_ptr<float>(), \
        G_token_saved.data_ptr<float>(), G_final_chunk.data_ptr<float>(), \
        N, L, chunk_size, num_chunks, state_count, save_g, 0, 0); \
    } while (0);

#define DISPATCH_SRNL_FWD_QRAW(D_VAL) \
    do { \
    int groups = srnl_parallel_groups(D_VAL, chunk_size, 128); int smem_bytes = srnl_forward_smem_bytes(D_VAL) + 2 * (groups - 1) * D_VAL * sizeof(float); \
    set_forward_dynamic_smem_attr_if_needed((const void*)srnl_forward_kernel_tpar<D_VAL, true>, smem_bytes); \
    srnl_forward_kernel_tpar<D_VAL, true><<<N, D_VAL * groups, smem_bytes, stream>>>( \
        M_m_0.data_ptr<float>(), M_v_0.data_ptr<float>(), M_k_0.data_ptr<float>(), M_eta_0.data_ptr<float>(), M_alpha_0.data_ptr<float>(), \
        reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>()), nullptr, reinterpret_cast<const __nv_bfloat16*>(QRaw.data_ptr<at::BFloat16>()), reinterpret_cast<__nv_bfloat16*>(Y.data_ptr<at::BFloat16>()), G_prefix_chunk.data_ptr<float>(), \
        K_saved.data_ptr<float>(), V_saved.data_ptr<float>(), Eta_saved.data_ptr<float>(), Alpha_saved.data_ptr<float>(), EtaGrad_saved.data_ptr<float>(), AlphaGrad_saved.data_ptr<float>(), InvK_saved.data_ptr<float>(), QInv_saved.data_ptr<float>(), \
        G_token_saved.data_ptr<float>(), G_final_chunk.data_ptr<float>(), \
        N, L, chunk_size, num_chunks, state_count, save_g, num_heads, save_qinv); \
    } while (0);

torch::Tensor srnl_forward(
    torch::Tensor M_m_0, torch::Tensor M_v_0, torch::Tensor M_k_0,
    torch::Tensor M_eta_0, torch::Tensor M_alpha_0, torch::Tensor X, torch::Tensor Q,
    torch::Tensor G_prefix_chunk,
    torch::Tensor K_saved, torch::Tensor V_saved,
    torch::Tensor Eta_saved, torch::Tensor Alpha_saved,
    torch::Tensor EtaGrad_saved, torch::Tensor AlphaGrad_saved,
    torch::Tensor InvK_saved,
    torch::Tensor G_token_saved, torch::Tensor G_final_chunk,
    int chunk_size, int D_int, int state_count, int save_g) {

    int N = X.size(0); int L = X.size(1); int num_chunks = L / chunk_size;
    if (X.scalar_type() != at::ScalarType::BFloat16 || Q.scalar_type() != at::ScalarType::Float) {
        throw std::runtime_error("HOPE gsave_opt forward expects bf16 X and fp32 Q.");
    }
    auto Y = torch::empty_like(X);
    auto stream = at::cuda::getCurrentCUDAStream();

${forward_dispatch}

    return Y;
}

torch::Tensor srnl_forward_qraw(
    torch::Tensor M_m_0, torch::Tensor M_v_0, torch::Tensor M_k_0,
    torch::Tensor M_eta_0, torch::Tensor M_alpha_0, torch::Tensor X, torch::Tensor QRaw,
    torch::Tensor G_prefix_chunk,
    torch::Tensor K_saved, torch::Tensor V_saved,
    torch::Tensor Eta_saved, torch::Tensor Alpha_saved,
    torch::Tensor EtaGrad_saved, torch::Tensor AlphaGrad_saved,
    torch::Tensor InvK_saved,
    torch::Tensor G_token_saved, torch::Tensor G_final_chunk,
    torch::Tensor QInv_saved,
    int chunk_size, int D_int, int state_count, int save_g, int num_heads, int save_qinv) {

    int N = X.size(0); int L = X.size(1); int num_chunks = L / chunk_size;
    if (X.scalar_type() != at::ScalarType::BFloat16 || QRaw.scalar_type() != at::ScalarType::BFloat16) {
        throw std::runtime_error("HOPE qraw forward expects bf16 X and bf16 QRaw.");
    }
    if (QRaw.dim() != 4 || QRaw.size(1) != 4 || QRaw.size(2) != L || QRaw.size(3) != num_heads * D_int) {
        throw std::runtime_error("HOPE qraw forward got invalid QRaw shape.");
    }
    auto Y = torch::empty_like(X);
    auto stream = at::cuda::getCurrentCUDAStream();

${forward_raw_dispatch}

    return Y;
}

#define DISPATCH_SRNL_BWD(D_VAL) \
    do { \
    int smem_bytes = srnl_backward_smem_bytes(D_VAL, 1, MAX_CHUNK); \
    set_backward_dynamic_smem_attr_if_needed((const void*)srnl_backward_kernel<D_VAL, false, 1, MAX_CHUNK>, smem_bytes); \
    srnl_backward_kernel<D_VAL, false, 1, MAX_CHUNK><<<grid, D_VAL, smem_bytes, stream>>>( \
        reinterpret_cast<const __nv_bfloat16*>(dY.data_ptr<at::BFloat16>()), reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>()), Q.data_ptr<float>(), reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>()), \
        M_m_0.data_ptr<float>(), M_v_0.data_ptr<float>(), M_k_0.data_ptr<float>(), M_eta_0.data_ptr<float>(), M_alpha_0.data_ptr<float>(), G_prefix_chunk.data_ptr<float>(), \
        K_saved.data_ptr<float>(), V_saved.data_ptr<float>(), Eta_saved.data_ptr<float>(), Alpha_saved.data_ptr<float>(), EtaGrad_saved.data_ptr<float>(), AlphaGrad_saved.data_ptr<float>(), InvK_saved.data_ptr<float>(), InvK_saved.data_ptr<float>(), \
        G_token_saved.data_ptr<float>(), G_final_chunk.data_ptr<float>(), \
        reinterpret_cast<__nv_bfloat16*>(dX.data_ptr<at::BFloat16>()), dQ.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(dX.data_ptr<at::BFloat16>()), \
        dM_m_0.data_ptr<float>(), dM_v_0.data_ptr<float>(), dM_k_0.data_ptr<float>(), dM_eta_0.data_ptr<float>(), dM_alpha_0.data_ptr<float>(), \
        N, L, chunk_size, num_chunks, state_count, save_g, 0, 0); \
    } while (0);

#define DISPATCH_SRNL_BWD_QRAW(D_VAL) \
    do { \
    int smem_bytes = srnl_backward_smem_bytes(D_VAL, 1, MAX_CHUNK); \
    set_backward_dynamic_smem_attr_if_needed((const void*)srnl_backward_kernel<D_VAL, true, 1, MAX_CHUNK>, smem_bytes); \
    srnl_backward_kernel<D_VAL, true, 1, MAX_CHUNK><<<grid, D_VAL, smem_bytes, stream>>>( \
        reinterpret_cast<const __nv_bfloat16*>(dY.data_ptr<at::BFloat16>()), reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>()), nullptr, reinterpret_cast<const __nv_bfloat16*>(QRaw.data_ptr<at::BFloat16>()), \
        M_m_0.data_ptr<float>(), M_v_0.data_ptr<float>(), M_k_0.data_ptr<float>(), M_eta_0.data_ptr<float>(), M_alpha_0.data_ptr<float>(), G_prefix_chunk.data_ptr<float>(), \
        K_saved.data_ptr<float>(), V_saved.data_ptr<float>(), Eta_saved.data_ptr<float>(), Alpha_saved.data_ptr<float>(), EtaGrad_saved.data_ptr<float>(), AlphaGrad_saved.data_ptr<float>(), InvK_saved.data_ptr<float>(), QInv_saved.data_ptr<float>(), \
        G_token_saved.data_ptr<float>(), G_final_chunk.data_ptr<float>(), \
        reinterpret_cast<__nv_bfloat16*>(dX.data_ptr<at::BFloat16>()), nullptr, reinterpret_cast<__nv_bfloat16*>(dQRaw.data_ptr<at::BFloat16>()), \
        dM_m_0.data_ptr<float>(), dM_v_0.data_ptr<float>(), dM_k_0.data_ptr<float>(), dM_eta_0.data_ptr<float>(), dM_alpha_0.data_ptr<float>(), \
        N, L, chunk_size, num_chunks, state_count, save_g, num_heads, save_qinv); \
    } while (0);

#define DISPATCH_SRNL_BWD_QRAW_GROUPED(D_VAL, GROUPS_VAL) \
    do { \
    int smem_bytes = srnl_backward_smem_bytes(D_VAL, GROUPS_VAL, MAX_CHUNK); \
    set_backward_dynamic_smem_attr_if_needed((const void*)srnl_backward_kernel<D_VAL, true, GROUPS_VAL, MAX_CHUNK>, smem_bytes); \
    srnl_backward_kernel<D_VAL, true, GROUPS_VAL, MAX_CHUNK><<<(N + GROUPS_VAL - 1) / GROUPS_VAL, D_VAL * GROUPS_VAL, smem_bytes, stream>>>( \
        reinterpret_cast<const __nv_bfloat16*>(dY.data_ptr<at::BFloat16>()), reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>()), nullptr, reinterpret_cast<const __nv_bfloat16*>(QRaw.data_ptr<at::BFloat16>()), \
        M_m_0.data_ptr<float>(), M_v_0.data_ptr<float>(), M_k_0.data_ptr<float>(), M_eta_0.data_ptr<float>(), M_alpha_0.data_ptr<float>(), G_prefix_chunk.data_ptr<float>(), \
        K_saved.data_ptr<float>(), V_saved.data_ptr<float>(), Eta_saved.data_ptr<float>(), Alpha_saved.data_ptr<float>(), EtaGrad_saved.data_ptr<float>(), AlphaGrad_saved.data_ptr<float>(), InvK_saved.data_ptr<float>(), QInv_saved.data_ptr<float>(), \
        G_token_saved.data_ptr<float>(), G_final_chunk.data_ptr<float>(), \
        reinterpret_cast<__nv_bfloat16*>(dX.data_ptr<at::BFloat16>()), nullptr, reinterpret_cast<__nv_bfloat16*>(dQRaw.data_ptr<at::BFloat16>()), \
        dM_m_0.data_ptr<float>(), dM_v_0.data_ptr<float>(), dM_k_0.data_ptr<float>(), dM_eta_0.data_ptr<float>(), dM_alpha_0.data_ptr<float>(), \
        N, L, chunk_size, num_chunks, state_count, save_g, num_heads, save_qinv); \
    } while (0);

#define DISPATCH_SRNL_BWD_QRAW_GROUPED_CMAX(D_VAL, GROUPS_VAL, CMAX_VAL) \
    do { \
    int smem_bytes = srnl_backward_smem_bytes(D_VAL, GROUPS_VAL, CMAX_VAL); \
    set_backward_dynamic_smem_attr_if_needed((const void*)srnl_backward_kernel<D_VAL, true, GROUPS_VAL, CMAX_VAL>, smem_bytes); \
    srnl_backward_kernel<D_VAL, true, GROUPS_VAL, CMAX_VAL><<<(N + GROUPS_VAL - 1) / GROUPS_VAL, D_VAL * GROUPS_VAL, smem_bytes, stream>>>( \
        reinterpret_cast<const __nv_bfloat16*>(dY.data_ptr<at::BFloat16>()), reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>()), nullptr, reinterpret_cast<const __nv_bfloat16*>(QRaw.data_ptr<at::BFloat16>()), \
        M_m_0.data_ptr<float>(), M_v_0.data_ptr<float>(), M_k_0.data_ptr<float>(), M_eta_0.data_ptr<float>(), M_alpha_0.data_ptr<float>(), G_prefix_chunk.data_ptr<float>(), \
        K_saved.data_ptr<float>(), V_saved.data_ptr<float>(), Eta_saved.data_ptr<float>(), Alpha_saved.data_ptr<float>(), EtaGrad_saved.data_ptr<float>(), AlphaGrad_saved.data_ptr<float>(), InvK_saved.data_ptr<float>(), QInv_saved.data_ptr<float>(), \
        G_token_saved.data_ptr<float>(), G_final_chunk.data_ptr<float>(), \
        reinterpret_cast<__nv_bfloat16*>(dX.data_ptr<at::BFloat16>()), nullptr, reinterpret_cast<__nv_bfloat16*>(dQRaw.data_ptr<at::BFloat16>()), \
        dM_m_0.data_ptr<float>(), dM_v_0.data_ptr<float>(), dM_k_0.data_ptr<float>(), dM_eta_0.data_ptr<float>(), dM_alpha_0.data_ptr<float>(), \
        N, L, chunk_size, num_chunks, state_count, save_g, num_heads, save_qinv); \
    } while (0);

#define DISPATCH_SRNL_BWD_QRAW_CMAX1(D_VAL) \
    if (chunk_size <= 8) { DISPATCH_SRNL_BWD_QRAW_GROUPED_CMAX(D_VAL, 1, 8) } \
    else if (chunk_size <= 16) { DISPATCH_SRNL_BWD_QRAW_GROUPED_CMAX(D_VAL, 1, 16) } \
    else if (chunk_size <= 32) { DISPATCH_SRNL_BWD_QRAW_GROUPED_CMAX(D_VAL, 1, 32) } \
    else if (chunk_size <= 64) { DISPATCH_SRNL_BWD_QRAW_GROUPED_CMAX(D_VAL, 1, 64) } \
    else { DISPATCH_SRNL_BWD_QRAW(D_VAL) }

#define DISPATCH_SRNL_BWD_QRAW_D64_CMAX() \
    if (chunk_size <= 32) { DISPATCH_SRNL_BWD_QRAW_GROUPED_CMAX(64, 1, 32) } \
    else if (chunk_size <= 64) { DISPATCH_SRNL_BWD_QRAW_GROUPED_CMAX(64, 1, 64) } \
    else { DISPATCH_SRNL_BWD_QRAW(64) }

#define DISPATCH_SRNL_BWD_QRAW_CMAX_GROUPED(D_VAL, GROUPS_VAL) \
    if (chunk_size <= 8) { DISPATCH_SRNL_BWD_QRAW_GROUPED_CMAX(D_VAL, GROUPS_VAL, 8) } \
    else if (chunk_size <= 16) { DISPATCH_SRNL_BWD_QRAW_GROUPED_CMAX(D_VAL, GROUPS_VAL, 16) } \
    else if (chunk_size <= 32) { DISPATCH_SRNL_BWD_QRAW_GROUPED_CMAX(D_VAL, GROUPS_VAL, 32) } \
    else if (chunk_size <= 64) { DISPATCH_SRNL_BWD_QRAW_GROUPED_CMAX(D_VAL, GROUPS_VAL, 64) } \
    else { DISPATCH_SRNL_BWD_QRAW_GROUPED(D_VAL, GROUPS_VAL) }

std::vector<torch::Tensor> srnl_backward(
    torch::Tensor dY, torch::Tensor X, torch::Tensor Q,
    torch::Tensor M_m_0, torch::Tensor M_v_0, torch::Tensor M_k_0,
    torch::Tensor M_eta_0, torch::Tensor M_alpha_0, torch::Tensor G_prefix_chunk,
    torch::Tensor K_saved, torch::Tensor V_saved,
    torch::Tensor Eta_saved, torch::Tensor Alpha_saved,
    torch::Tensor EtaGrad_saved, torch::Tensor AlphaGrad_saved,
    torch::Tensor InvK_saved,
    torch::Tensor G_token_saved, torch::Tensor G_final_chunk,
    int chunk_size, int D_int, int state_count, int save_g) {

    int N = X.size(0); int L = X.size(1); int num_chunks = L / chunk_size;
    if (dY.scalar_type() != at::ScalarType::BFloat16 || X.scalar_type() != at::ScalarType::BFloat16 || Q.scalar_type() != at::ScalarType::Float) {
        throw std::runtime_error("HOPE gsave_opt backward expects bf16 dY/X and fp32 Q.");
    }
    auto options = Q.options();

    auto dX = torch::empty_like(X);
    auto dQ = torch::empty_like(Q);
    auto dM_m_0 = torch::zeros({state_count, D_int, D_int}, options);
    auto dM_v_0 = torch::zeros({state_count, D_int, D_int}, options);
    auto dM_k_0 = torch::zeros({state_count, D_int, D_int}, options);
    auto dM_eta_0 = torch::zeros({state_count, D_int}, options);
    auto dM_alpha_0 = torch::zeros({state_count, D_int}, options);

    dim3 grid(N);
    auto stream = at::cuda::getCurrentCUDAStream();

${backward_dispatch}

    return {dM_m_0, dM_v_0, dM_k_0, dM_eta_0, dM_alpha_0, dX, dQ};
}

std::vector<torch::Tensor> srnl_backward_qraw(
    torch::Tensor dY, torch::Tensor X, torch::Tensor QRaw,
    torch::Tensor M_m_0, torch::Tensor M_v_0, torch::Tensor M_k_0,
    torch::Tensor M_eta_0, torch::Tensor M_alpha_0, torch::Tensor G_prefix_chunk,
    torch::Tensor K_saved, torch::Tensor V_saved,
    torch::Tensor Eta_saved, torch::Tensor Alpha_saved,
    torch::Tensor EtaGrad_saved, torch::Tensor AlphaGrad_saved,
    torch::Tensor InvK_saved,
    torch::Tensor G_token_saved, torch::Tensor G_final_chunk,
    torch::Tensor QInv_saved,
    int chunk_size, int D_int, int state_count, int save_g, int num_heads, int save_qinv, int bwd_groups, int use_cmax) {

    int N = X.size(0); int L = X.size(1); int num_chunks = L / chunk_size;
    if (dY.scalar_type() != at::ScalarType::BFloat16 || X.scalar_type() != at::ScalarType::BFloat16 || QRaw.scalar_type() != at::ScalarType::BFloat16) {
        throw std::runtime_error("HOPE qraw backward expects bf16 dY/X/QRaw.");
    }
    if (QRaw.dim() != 4 || QRaw.size(1) != 4 || QRaw.size(2) != L || QRaw.size(3) != num_heads * D_int) {
        throw std::runtime_error("HOPE qraw backward got invalid QRaw shape.");
    }
    auto options = M_m_0.options();

    auto dX = torch::empty_like(X);
    auto dQRaw = torch::empty_like(QRaw);
    auto dM_m_0 = torch::zeros({state_count, D_int, D_int}, options);
    auto dM_v_0 = torch::zeros({state_count, D_int, D_int}, options);
    auto dM_k_0 = torch::zeros({state_count, D_int, D_int}, options);
    auto dM_eta_0 = torch::zeros({state_count, D_int}, options);
    auto dM_alpha_0 = torch::zeros({state_count, D_int}, options);

    dim3 grid(N);
    auto stream = at::cuda::getCurrentCUDAStream();

    const auto* device_props = at::cuda::getCurrentDeviceProperties();
    int cmax_capacity = 8;
    while (cmax_capacity < chunk_size) cmax_capacity *= 2;
    if (!use_cmax) cmax_capacity = MAX_CHUNK;
    int per_group_bytes = srnl_backward_smem_bytes(D_int, 1, cmax_capacity);
    int group_budget = std::max(per_group_bytes,
        int(device_props->sharedMemPerMultiprocessor / 4));
    group_budget = std::min(group_budget,
        int(std::max(device_props->sharedMemPerBlock, device_props->sharedMemPerBlockOptin)) - 1024);
    bwd_groups = std::min(bwd_groups, std::max(1, 32 / D_int));
    while (bwd_groups > 1 && per_group_bytes * bwd_groups > group_budget) bwd_groups /= 2;

${backward_raw_dispatch}

    return {dM_m_0, dM_v_0, dM_k_0, dM_eta_0, dM_alpha_0, dX, dQRaw};
}

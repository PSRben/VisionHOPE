

template <int D>
__global__ void srnl_forward_${layout}_qraw_kernel(
    const float* __restrict__ M_m_0, const float* __restrict__ M_v_0,
    const float* __restrict__ M_k_0, const float* __restrict__ M_eta_0,
    const float* __restrict__ M_alpha_0, const __nv_bfloat16* __restrict__ X,
    const __nv_bfloat16* __restrict__ QRaw, __nv_bfloat16* __restrict__ Y,
    float* __restrict__ G_prefix_chunk,
    float* __restrict__ K_saved, float* __restrict__ V_saved,
    float* __restrict__ Eta_saved, float* __restrict__ Alpha_saved,
    float* __restrict__ EtaGrad_saved, float* __restrict__ AlphaGrad_saved,
    float* __restrict__ InvK_saved, float* __restrict__ QInv_saved,
    float* __restrict__ G_checkpoint_saved, float* __restrict__ G_final_chunk,
    ${kernel_parameters}
{
    int group = threadIdx.x / D;
    int d_i = threadIdx.x - group * D;
    int groups_per_block = blockDim.x / D;
    int pid = blockIdx.x;
    if (pid >= N) return;

    // All groups share the same logical state. Token computations run across
    // groups, while state updates follow token order.
    extern __shared__ float storage[];
    float (*s_x_storage)[D] = reinterpret_cast<float (*)[D]>(storage);
    float (*s_q_storage)[D] = s_x_storage + groups_per_block;
    float* s_k = reinterpret_cast<float*>(s_q_storage + groups_per_block);
    float* s_v = s_k + D;
    float (*s_g_final)[D + 1] = reinterpret_cast<float (*)[D + 1]>(s_v + D);
    float* s_eta_init_vec = reinterpret_cast<float*>(s_g_final + D);
    float* s_alpha_init_vec = s_eta_init_vec + D;
    float* s_x = s_x_storage[group];
    float* s_q = s_q_storage[group];

    int state_pid = (StateCount == N) ? pid : (pid % StateCount);
    int state_stride = state_pid * D * D + d_i * D;
    int vec_stride = state_pid * D + d_i;
    int x_stride = pid * L * D;

    int q_head = pid % NumHeads;
    int tmp = pid / NumHeads;
    int q_dir = tmp % DirectionCount;
    int q_b = tmp / DirectionCount;
    ${state_layout}    float m_mem[D], m_v[D], m_k[D];
    #pragma unroll
    for (int j = 0; j < D; ++j) {
        m_mem[j] = M_m_0[state_stride + j];
        m_v[j] = M_v_0[state_stride + j];
        m_k[j] = M_k_0[state_stride + j];
    }
    float s_eta = M_eta_0[vec_stride];
    float s_alpha = M_alpha_0[vec_stride];
    float prefix_g[D];
    #pragma unroll
    for (int j = 0; j < D; ++j) {
        prefix_g[j] = (d_i == j) ? 1.0f : 0.0f;
    }

    for (int c = 0; c < NumChunks; ++c) {
        if (c > 0 && group == 0) {
            float* out_prefix = G_prefix_chunk + ${prefix_index};
            #pragma unroll
            for (int j = 0; j < D; ++j) {
                out_prefix[d_i * D + j] = prefix_g[j];
            }
        }

        if (group == 0) {
            s_eta_init_vec[d_i] = s_eta;
            s_alpha_init_vec[d_i] = s_alpha;
        }
        __syncthreads();

        for (int t_inner = group; t_inner < ChunkSize; t_inner += groups_per_block) {
            int t = c * ChunkSize + t_inner;
            s_x[d_i] = __bfloat162float(X[x_stride + t * D + d_i]);
            int64_t q_idx = (((int64_t)q_b * DirectionCount + q_dir) * L + t) * q_channels + q_head * D + d_i;
            float q_raw_i = __bfloat162float(QRaw[q_idx]);
            float q_norm_sq = srnl_group_sum<D>(q_raw_i * q_raw_i, d_i, group);
            float q_inv = 1.0f / fmaxf(sqrtf(q_norm_sq), Q_NORM_EPS);
            if (SaveQInv && d_i == 0) {
                QInv_saved[pid * L + t] = q_inv;
            }
            s_q[d_i] = q_raw_i * q_inv;
            srnl_group_sync<D>(group);

            float k_raw_i = 0.0f, v_i = 0.0f, y_i = 0.0f;
            #pragma unroll
            for (int j = 0; j < D; ++j) {
                float x_j = s_x[j];
                k_raw_i += m_k[j] * x_j;
                v_i += m_v[j] * x_j;
                y_i += m_mem[j] * s_q[j];
            }
            float k_norm_sq = srnl_group_sum<D>(k_raw_i * k_raw_i, d_i, group);
            float inv_k_norm = rsqrtf(k_norm_sq + 1e-6f);
            float k_i = k_raw_i * inv_k_norm;
            K_saved[x_stride + t * D + d_i] = k_i;
            V_saved[x_stride + t * D + d_i] = v_i;
            Y[x_stride + t * D + d_i] = __float2bfloat16(y_i);

            float eta_pre_raw = srnl_group_sum<D>(s_x[d_i] * s_eta, d_i, group);
            float alpha_pre_raw = srnl_group_sum<D>(s_x[d_i] * s_alpha, d_i, group);
            float eta_pre = fminf(PRE_ACT_CLAMP, fmaxf(-PRE_ACT_CLAMP, eta_pre_raw));
            float alpha_pre_biased = alpha_pre_raw + ALPHA_BIAS;
            float alpha_pre = fminf(PRE_ACT_CLAMP, fmaxf(-PRE_ACT_CLAMP, alpha_pre_biased));
            if (d_i == 0) {
                int scalar_idx = pid * L + t;
                float eta_softplus = (eta_pre > 20.0f) ? eta_pre : logf(1.0f + expf(eta_pre));
                float alpha_sigmoid = 1.0f / (1.0f + expf(-alpha_pre));
                Eta_saved[scalar_idx] = ETA_SCALE * eta_softplus;
                Alpha_saved[scalar_idx] = ALPHA_MIN + (1.0f - ALPHA_MIN) * alpha_sigmoid;
                float eta_clamp_grad = (fabsf(eta_pre_raw) <= PRE_ACT_CLAMP) ? 1.0f : 0.0f;
                float alpha_clamp_grad = (fabsf(alpha_pre_biased) <= PRE_ACT_CLAMP) ? 1.0f : 0.0f;
                EtaGrad_saved[scalar_idx] =
                    ETA_SCALE * ((eta_pre > 20.0f) ? 1.0f : 1.0f / (1.0f + expf(-eta_pre))) * eta_clamp_grad;
                AlphaGrad_saved[scalar_idx] =
                    (1.0f - ALPHA_MIN) * alpha_sigmoid * (1.0f - alpha_sigmoid) * alpha_clamp_grad;
                InvK_saved[scalar_idx] = inv_k_norm;
            }
            srnl_group_sync<D>(group);
        }

        __syncthreads();
        if (group == 0) {
            float g[D];
            #pragma unroll
            for (int j = 0; j < D; ++j) {
                g[j] = (d_i == j) ? 1.0f : 0.0f;
            }
    
            for (int t_inner = 0; t_inner < ChunkSize; ++t_inner) {
                if ((t_inner % RECOMP_TILE) == 0) {
                    int save_idx = c * MaxSubchunks + (t_inner / RECOMP_TILE);
                    float* out_g = G_checkpoint_saved + ${checkpoint_write_index};
                    #pragma unroll
                    for (int j = 0; j < D; ++j) {
                        out_g[d_i * D + j] = g[j];
                    }
                }
    
                int t = c * ChunkSize + t_inner;
                float sk_i = K_saved[x_stride + t * D + d_i];
                float sv_i = V_saved[x_stride + t * D + d_i];
                s_k[d_i] = sk_i;
                s_v[d_i] = sv_i;
                float eta = Eta_saved[pid * L + t];
                float alpha = Alpha_saved[pid * L + t];
                srnl_group_sync<D>(group);
    
                float diff_i = sk_i - sv_i;
                float r_i = diff_i;
                #pragma unroll
                for (int j = 0; j < D; ++j) {
                    r_i += g[j] * s_k[j];
                }
                float diff_norm = sqrtf(srnl_group_sum<D>(diff_i * diff_i, d_i, group) + STAB_NORM_EPS);
                float eta_eff = d_i == 0 ? soft_project_eta(eta, alpha, diff_norm) : 0.0f;
                eta_eff = srnl_group_first<D>(eta_eff, d_i, group);
    #if ETA2ALPHA_CLAMP
                float key_norm_sq_eff = srnl_group_sum<D>(sk_i * sk_i, d_i, group);
                eta_eff = clamp_eta_to_alpha(eta_eff, alpha, key_norm_sq_eff);
    #endif
                #pragma unroll
                for (int j = 0; j < D; ++j) {
                    g[j] = alpha * g[j] - eta_eff * r_i * s_k[j];
                }
                srnl_group_sync<D>(group);
            }
    
            #pragma unroll
            for (int j = 0; j < D; ++j) {
                s_g_final[d_i][j] = g[j];
            }
            float* out_g_final = G_final_chunk + ${final_index};
            #pragma unroll
            for (int j = 0; j < D; ++j) {
                out_g_final[d_i * D + j] = g[j];
            }
        }
        __syncthreads();

        float next_prefix[D];
        #pragma unroll
        for (int j = 0; j < D; ++j) {
            float acc = 0.0f;
            #pragma unroll
            for (int l = 0; l < D; ++l) {
                acc += prefix_g[l] * s_g_final[l][j];
            }
            next_prefix[j] = acc;
        }

        float next_mem_row[D], next_v_row[D], next_k_row[D];
        #pragma unroll
        for (int j = 0; j < D; ++j) {
            float next_mem = 0.0f;
            float next_v = 0.0f;
            float next_k = 0.0f;
            #pragma unroll
            for (int l = 0; l < D; ++l) {
                float g_lj = s_g_final[l][j];
                next_mem += m_mem[l] * g_lj;
                next_v += m_v[l] * g_lj;
                next_k += m_k[l] * g_lj;
            }
            next_mem_row[j] = next_mem;
            next_v_row[j] = next_v;
            next_k_row[j] = next_k;
        }

        // Compute every column of M * G before overwriting the input row.
        #pragma unroll
        for (int j = 0; j < D; ++j) {
            m_mem[j] = next_mem_row[j];
            m_v[j] = next_v_row[j];
            m_k[j] = next_k_row[j];
        }

        float next_eta = 0.0f;
        float next_alpha = 0.0f;
        #pragma unroll
        for (int l = 0; l < D; ++l) {
            float g_li = s_g_final[l][d_i];
            next_eta += s_eta_init_vec[l] * g_li;
            next_alpha += s_alpha_init_vec[l] * g_li;
        }
        s_eta = next_eta;
        s_alpha = next_alpha;
        #pragma unroll
        for (int j = 0; j < D; ++j) {
            prefix_g[j] = next_prefix[j];
        }
        __syncthreads();
    }
}

__host__ __forceinline__ int srnl_${layout}_backward_smem_bytes(int D, int groups) {
    int floats_per_group = 0;
    floats_per_group += 3 * D * (D + 1);  // s_m_init
    floats_per_group += D * (D + 1);      // smem_red
    floats_per_group += 3 * D;            // s_x/s_q/s_dy
    floats_per_group += 2 * D;            // s_k/s_v
    floats_per_group += D * (D + 1);      // s_g_mat
    floats_per_group += 2 * D;            // s_vec_grad
    floats_per_group += 2 * D;            // s_eta/s_alpha init
    floats_per_group += 2 * D;            // s_r/s_bar_r
    return groups * floats_per_group * (int)sizeof(float);
}

template <int D, int GROUPS>
__global__ void srnl_backward_${layout}_qraw_kernel(
    const __nv_bfloat16* __restrict__ dY, const __nv_bfloat16* __restrict__ X,
    const __nv_bfloat16* __restrict__ QRaw,
    const float* __restrict__ M_m_0, const float* __restrict__ M_v_0,
    const float* __restrict__ M_k_0, const float* __restrict__ M_eta_0,
    const float* __restrict__ M_alpha_0, const float* __restrict__ G_prefix_chunk,
    const float* __restrict__ K_saved, const float* __restrict__ V_saved,
    const float* __restrict__ Eta_saved, const float* __restrict__ Alpha_saved,
    const float* __restrict__ EtaGrad_saved, const float* __restrict__ AlphaGrad_saved,
    const float* __restrict__ InvK_saved, const float* __restrict__ QInv_saved,
    const float* __restrict__ G_checkpoint_saved, const float* __restrict__ G_final_chunk,
    __nv_bfloat16* __restrict__ dX, __nv_bfloat16* __restrict__ dQRaw,
    float* __restrict__ dM_m_0, float* __restrict__ dM_v_0, float* __restrict__ dM_k_0,
    float* __restrict__ dM_eta_0, float* __restrict__ dM_alpha_0,
    ${kernel_parameters}
{
    int group = threadIdx.x / D;
    int d_i = threadIdx.x - group * D;
    int groups_per_block = blockDim.x / D;
    int pid = blockIdx.x * groups_per_block + group;
    if (pid >= N) return;

    float g_mem[D] = {0}, g_m_v[D] = {0}, g_m_k[D] = {0};
    float g_m_eta = 0.0f, g_m_alpha = 0.0f;

    extern __shared__ float dyn_storage[];
    float* dyn_ptr = dyn_storage;
    float* s_m_init_base = dyn_ptr;
    dyn_ptr += GROUPS * 3 * D * (D + 1);
    float* smem_red_base = dyn_ptr;
    dyn_ptr += GROUPS * D * (D + 1);
    float* s_x_base = dyn_ptr;
    dyn_ptr += GROUPS * D;
    float* s_q_base = dyn_ptr;
    dyn_ptr += GROUPS * D;
    float* s_dy_base = dyn_ptr;
    dyn_ptr += GROUPS * D;
    float* s_k_base = dyn_ptr;
    dyn_ptr += GROUPS * D;
    float* s_v_base = dyn_ptr;
    dyn_ptr += GROUPS * D;
    float* s_g_mat_base = dyn_ptr;
    dyn_ptr += GROUPS * D * (D + 1);
    float* s_vec_grad_base = dyn_ptr;
    dyn_ptr += GROUPS * 2 * D;
    float* s_eta_init_base = dyn_ptr;
    dyn_ptr += GROUPS * D;
    float* s_alpha_init_base = dyn_ptr;
    dyn_ptr += GROUPS * D;
    float* s_r_vec_base = dyn_ptr;
    dyn_ptr += GROUPS * D;
    float* s_bar_r_vec_base = dyn_ptr;

    float (*s_m_init)[D][D + 1] = reinterpret_cast<float (*)[D][D + 1]>(s_m_init_base + group * 3 * D * (D + 1));
    float (*smem_red)[D + 1] = reinterpret_cast<float (*)[D + 1]>(smem_red_base + group * D * (D + 1));
    float* s_x = s_x_base + group * D;
    float* s_q = s_q_base + group * D;
    float* s_dy = s_dy_base + group * D;
    float* s_k = s_k_base + group * D;
    float* s_v = s_v_base + group * D;
    float (*s_g_mat)[D + 1] = reinterpret_cast<float (*)[D + 1]>(s_g_mat_base + group * D * (D + 1));
    float (*s_vec_grad)[D] = reinterpret_cast<float (*)[D]>(s_vec_grad_base + group * 2 * D);
    float* s_eta_init_vec = s_eta_init_base + group * D;
    float* s_alpha_init_vec = s_alpha_init_base + group * D;
    float* s_r_vec = s_r_vec_base + group * D;
    float* s_bar_r_vec = s_bar_r_vec_base + group * D;

    float g_traj[RECOMP_TILE][D];
    float g[D];

    int x_stride = pid * L * D;
    int state_pid = (StateCount == N) ? pid : (pid % StateCount);
    int state_stride = state_pid * D * D + d_i * D;
    int q_head = pid % NumHeads;
    int tmp = pid / NumHeads;
    int q_dir = tmp % DirectionCount;
    int q_b = tmp / DirectionCount;
    ${state_layout}    for (int c = NumChunks - 1; c >= 0; --c) {
        if (c == 0) {
            #pragma unroll
            for (int j = 0; j < D; ++j) {
                s_g_mat[d_i][j] = (d_i == j) ? 1.0f : 0.0f;
                s_m_init[0][d_i][j] = M_m_0[state_stride + j];
                s_m_init[1][d_i][j] = M_v_0[state_stride + j];
                s_m_init[2][d_i][j] = M_k_0[state_stride + j];
            }
        } else {
            const float* in_prefix = G_prefix_chunk + ${prefix_index};
            #pragma unroll
            for (int j = 0; j < D; ++j) {
                s_g_mat[d_i][j] = in_prefix[d_i * D + j];
            }
            warpSync<D>(group);
            #pragma unroll
            for (int j = 0; j < D; ++j) {
                float mem_acc = 0.0f, v_acc = 0.0f, k_acc = 0.0f;
                #pragma unroll
                for (int l = 0; l < D; ++l) {
                    float p_lj = s_g_mat[l][j];
                    mem_acc += M_m_0[state_stride + l] * p_lj;
                    v_acc += M_v_0[state_stride + l] * p_lj;
                    k_acc += M_k_0[state_stride + l] * p_lj;
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
        for (int l = 0; l < D; ++l) {
            float p_li = s_g_mat[l][d_i];
            m_eta_init += M_eta_0[state_pid * D + l] * p_li;
            m_alpha_init += M_alpha_0[state_pid * D + l] * p_li;
        }
        s_eta_init_vec[d_i] = m_eta_init;
        s_alpha_init_vec[d_i] = m_alpha_init;
        warpSync<D>(group);

        const float* in_g_final = G_final_chunk + ${final_index};
        #pragma unroll
        for (int j = 0; j < D; ++j) {
            s_g_mat[d_i][j] = in_g_final[d_i * D + j];
        }
        s_vec_grad[0][d_i] = g_m_eta;
        s_vec_grad[1][d_i] = g_m_alpha;
        warpSync<D>(group);

        float dM_mem_init[D] = {0}, dM_v_init[D] = {0}, dM_k_init[D] = {0};
        float dM_eta_init = 0.0f, dM_alpha_init = 0.0f;
        #pragma unroll
        for (int j = 0; j < D; ++j) {
            float mem_acc = 0.0f, v_acc = 0.0f, k_acc = 0.0f;
            #pragma unroll
            for (int l = 0; l < D; ++l) {
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
        for (int l = 0; l < D; ++l) {
            float g_il = s_g_mat[d_i][l];
            dM_eta_init += s_vec_grad[0][l] * g_il;
            dM_alpha_init += s_vec_grad[1][l] * g_il;
        }

        float bg[D] = {0};
        #pragma unroll
        for (int j = 0; j < D; ++j) smem_red[d_i][j] = g_mem[j];
        warpSync<D>(group);
        #pragma unroll
        for (int j = 0; j < D; ++j) {
            float acc = 0.0f;
            #pragma unroll
            for (int r = 0; r < D; ++r) acc += s_m_init[0][r][d_i] * smem_red[r][j];
            bg[j] += acc;
        }
        warpSync<D>(group);
        #pragma unroll
        for (int j = 0; j < D; ++j) smem_red[d_i][j] = g_m_v[j];
        warpSync<D>(group);
        #pragma unroll
        for (int j = 0; j < D; ++j) {
            float acc = 0.0f;
            #pragma unroll
            for (int r = 0; r < D; ++r) acc += s_m_init[1][r][d_i] * smem_red[r][j];
            bg[j] += acc;
        }
        warpSync<D>(group);
        #pragma unroll
        for (int j = 0; j < D; ++j) smem_red[d_i][j] = g_m_k[j];
        warpSync<D>(group);
        #pragma unroll
        for (int j = 0; j < D; ++j) {
            float acc = 0.0f;
            #pragma unroll
            for (int r = 0; r < D; ++r) acc += s_m_init[2][r][d_i] * smem_red[r][j];
            bg[j] += acc + m_eta_init * s_vec_grad[0][j] + m_alpha_init * s_vec_grad[1][j];
        }
        warpSync<D>(group);

        int num_subchunks = (ChunkSize + RECOMP_TILE - 1) / RECOMP_TILE;
        for (int sub_rev = 0; sub_rev < num_subchunks; ++sub_rev) {
            int sub = num_subchunks - 1 - sub_rev;
            int sub_start = sub * RECOMP_TILE;
            int sub_end = (sub_start + RECOMP_TILE < ChunkSize) ? (sub_start + RECOMP_TILE) : ChunkSize;
            int sub_len = sub_end - sub_start;

            const float* in_g_checkpoint =
                G_checkpoint_saved + ${checkpoint_read_index};
            #pragma unroll
            for (int j = 0; j < D; ++j) {
                g[j] = in_g_checkpoint[d_i * D + j];
            }

            for (int local_idx = 0; local_idx < sub_len; ++local_idx) {
                int t_inner = sub_start + local_idx;
                int t = c * ChunkSize + t_inner;
                #pragma unroll
                for (int j = 0; j < D; ++j) {
                    g_traj[local_idx][j] = g[j];
                }

                float sk_i = K_saved[x_stride + t * D + d_i];
                float sv_i = V_saved[x_stride + t * D + d_i];
                s_k[d_i] = sk_i;
                s_v[d_i] = sv_i;
                float eta = Eta_saved[pid * L + t];
                float alpha = Alpha_saved[pid * L + t];
                warpSync<D>(group);

                float diff_i = sk_i - sv_i;
                float r_i = diff_i;
                #pragma unroll
                for (int j = 0; j < D; ++j) {
                    r_i += g[j] * s_k[j];
                }
                float diff_norm = sqrtf(warpReduceSum<D>(diff_i * diff_i, group) + STAB_NORM_EPS);
                float eta_eff = soft_project_eta_warp<D>(eta, alpha, diff_norm, group, d_i);
#if ETA2ALPHA_CLAMP
                float key_norm_sq_eff = warpReduceSum<D>(sk_i * sk_i, group);
                eta_eff = clamp_eta_to_alpha(eta_eff, alpha, key_norm_sq_eff);
#endif
                #pragma unroll
                for (int j = 0; j < D; ++j) {
                    g[j] = alpha * g[j] - eta_eff * r_i * s_k[j];
                }
                warpSync<D>(group);
            }

            for (int local_rev = 0; local_rev < sub_len; ++local_rev) {
                int local_idx = sub_len - 1 - local_rev;
                int t_inner = sub_start + local_idx;
                int t = c * ChunkSize + t_inner;
                #pragma unroll
                for (int j = 0; j < D; ++j) {
                    g[j] = g_traj[local_idx][j];
                }

                int64_t q_idx = (((int64_t)q_b * DirectionCount + q_dir) * L + t) * q_channels + q_head * D + d_i;
                float q_raw_i = __bfloat162float(QRaw[q_idx]);
                float q_inv;
                if (SaveQInv) {
                    q_inv = QInv_saved[pid * L + t];
                } else {
                    float q_norm_sq = warpReduceSum<D>(q_raw_i * q_raw_i, group);
                    q_inv = 1.0f / fmaxf(sqrtf(q_norm_sq), Q_NORM_EPS);
                }
                float q_i = q_raw_i * q_inv;
                s_q[d_i] = q_i;
                s_x[d_i] = __bfloat162float(X[x_stride + t * D + d_i]);
                s_dy[d_i] = __bfloat162float(dY[x_stride + t * D + d_i]);
                float sk_i = K_saved[x_stride + t * D + d_i];
                float sv_i = V_saved[x_stride + t * D + d_i];
                s_k[d_i] = sk_i;
                s_v[d_i] = sv_i;
                float eta = Eta_saved[pid * L + t];
                float alpha = Alpha_saved[pid * L + t];
                warpSync<D>(group);

                float diff_i = sk_i - sv_i;
                float r_i = diff_i;
                #pragma unroll
                for (int j = 0; j < D; ++j) {
                    r_i += g[j] * s_k[j];
                }
                float diff_norm = sqrtf(warpReduceSum<D>(diff_i * diff_i, group) + STAB_NORM_EPS);
                float diff_denom = 0.0f, eta_cap = 0.0f, eta_eff = 0.0f;
                float d_etaeff_deta = 0.0f, d_etaeff_dcap = 0.0f;
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
                for (int j = 0; j < D; ++j) smem_red[d_i][j] = s_m_init[0][d_i][j] * s_dy[d_i];
                warpSync<D>(group);
                float dq_i = 0.0f;
                #pragma unroll
                for (int i = 0; i < D; ++i) dq_i += smem_red[i][d_i];
                warpSync<D>(group);

                #pragma unroll
                for (int j = 0; j < D; ++j) dM_mem_init[j] += s_dy[d_i] * s_q[j];

                float h_k_i = 0.0f;
                float alpha_row = 0.0f;
                #pragma unroll
                for (int j = 0; j < D; ++j) {
                    h_k_i += bg[j] * s_k[j];
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
                for (int r = 0; r < D; ++r) {
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
                for (int j = 0; j < D; ++j) {
                    bg[j] = alpha * bg[j] + bar_r_i * s_k[j];
                }

                float dot = warpReduceSum<D>(dk_i * sk_i, group);
                float dk_raw_i = (dk_i - dot * sk_i) * InvK_saved[pid * L + t];
                float d_alpha_pre = d_alpha * AlphaGrad_saved[pid * L + t];
                float d_eta_pre = d_eta * EtaGrad_saved[pid * L + t];

                #pragma unroll
                for (int j = 0; j < D; ++j) {
                    dM_v_init[j] += dv_i * s_x[j];
                    dM_k_init[j] += dk_raw_i * s_x[j];
                    smem_red[d_i][j] = s_m_init[1][d_i][j] * dv_i + s_m_init[2][d_i][j] * dk_raw_i;
                }
                dM_eta_init += s_x[d_i] * d_eta_pre;
                dM_alpha_init += s_x[d_i] * d_alpha_pre;
                warpSync<D>(group);

                float dx_i = d_eta_pre * m_eta_init + d_alpha_pre * m_alpha_init;
                #pragma unroll
                for (int i = 0; i < D; ++i) dx_i += smem_red[i][d_i];
                dX[x_stride + t * D + d_i] = __float2bfloat16(dx_i);

                float q_dot = warpReduceSum<D>(dq_i * q_i, group);
                float dq_raw_i = (q_inv > 9.999e11f) ? (dq_i * q_inv) : ((dq_i - q_i * q_dot) * q_inv);
                dQRaw[q_idx] = __float2bfloat16(dq_raw_i);
            }
        }

        #pragma unroll
        for (int j = 0; j < D; ++j) {
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
        #pragma unroll
        for (int j = 0; j < D; ++j) {
            dM_m_0[out_stride + d_i * D + j] = g_mem[j];
            dM_v_0[out_stride + d_i * D + j] = g_m_v[j];
            dM_k_0[out_stride + d_i * D + j] = g_m_k[j];
        }
        dM_eta_0[state_pid * D + d_i] = g_m_eta;
        dM_alpha_0[state_pid * D + d_i] = g_m_alpha;
    } else {
        #pragma unroll
        for (int j = 0; j < D; ++j) {
            atomicAdd(&dM_m_0[out_stride + d_i * D + j], g_mem[j]);
            atomicAdd(&dM_v_0[out_stride + d_i * D + j], g_m_v[j]);
            atomicAdd(&dM_k_0[out_stride + d_i * D + j], g_m_k[j]);
        }
        atomicAdd(&dM_eta_0[state_pid * D + d_i], g_m_eta);
        atomicAdd(&dM_alpha_0[state_pid * D + d_i], g_m_alpha);
    }
}

#define DISPATCH_${macro_layout}_FWD(D_VAL) \
    do { \
    int token_groups = srnl_parallel_groups(D_VAL, ${parallel_length}, std::min(128, at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock)); \
    int smem_bytes = (2 * token_groups * D_VAL + 4 * D_VAL + D_VAL * (D_VAL + 1)) * sizeof(float); \
    srnl_forward_${layout}_qraw_kernel<D_VAL><<<N, D_VAL * token_groups, smem_bytes, stream>>>( \
        M_m_0.data_ptr<float>(), M_v_0.data_ptr<float>(), M_k_0.data_ptr<float>(), \
        M_eta_0.data_ptr<float>(), M_alpha_0.data_ptr<float>(), \
        reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>()), \
        reinterpret_cast<const __nv_bfloat16*>(QRaw.data_ptr<at::BFloat16>()), \
        reinterpret_cast<__nv_bfloat16*>(Y.data_ptr<at::BFloat16>()), \
        G_prefix_chunk.data_ptr<float>(), K_saved.data_ptr<float>(), V_saved.data_ptr<float>(), \
        Eta_saved.data_ptr<float>(), Alpha_saved.data_ptr<float>(), EtaGrad_saved.data_ptr<float>(), \
        AlphaGrad_saved.data_ptr<float>(), InvK_saved.data_ptr<float>(), QInv_saved.data_ptr<float>(), \
        G_checkpoint_saved.data_ptr<float>(), G_final_chunk.data_ptr<float>(), \
        ${call_parameters} \
    } while (0);

torch::Tensor srnl_forward_qraw_${layout}(
    torch::Tensor M_m_0, torch::Tensor M_v_0, torch::Tensor M_k_0,
    torch::Tensor M_eta_0, torch::Tensor M_alpha_0, torch::Tensor X, torch::Tensor QRaw,
    torch::Tensor G_prefix_chunk,
    torch::Tensor K_saved, torch::Tensor V_saved,
    torch::Tensor Eta_saved, torch::Tensor Alpha_saved,
    torch::Tensor EtaGrad_saved, torch::Tensor AlphaGrad_saved,
    torch::Tensor InvK_saved, torch::Tensor QInv_saved,
    torch::Tensor G_checkpoint_saved, torch::Tensor G_final_chunk,
    ${api_parameters}

    int N = X.size(0);
    int L = X.size(1);
    ${chunk_counts}
    if (X.scalar_type() != at::ScalarType::BFloat16 || QRaw.scalar_type() != at::ScalarType::BFloat16) {
        throw std::runtime_error("large-line qraw forward expects bf16 X and bf16 QRaw.");
    }
    if (QRaw.dim() != 4 || ${direction_check} || QRaw.size(2) != L || QRaw.size(3) != num_heads * D_int) {
        throw std::runtime_error("large-line qraw forward got invalid QRaw shape.");
    }
    auto Y = torch::empty_like(X);
    auto stream = at::cuda::getCurrentCUDAStream();
${forward_dispatch}
    return Y;
}

#define DISPATCH_${macro_layout}_BWD(D_VAL, GROUPS_VAL) \
    do { \
    int smem_bytes = srnl_${layout}_backward_smem_bytes(D_VAL, GROUPS_VAL); \
    set_backward_dynamic_smem_attr_if_needed((const void*)srnl_backward_${layout}_qraw_kernel<D_VAL, GROUPS_VAL>, smem_bytes); \
    srnl_backward_${layout}_qraw_kernel<D_VAL, GROUPS_VAL><<<(N + GROUPS_VAL - 1) / GROUPS_VAL, D_VAL * GROUPS_VAL, smem_bytes, stream>>>( \
        reinterpret_cast<const __nv_bfloat16*>(dY.data_ptr<at::BFloat16>()), \
        reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>()), \
        reinterpret_cast<const __nv_bfloat16*>(QRaw.data_ptr<at::BFloat16>()), \
        M_m_0.data_ptr<float>(), M_v_0.data_ptr<float>(), M_k_0.data_ptr<float>(), \
        M_eta_0.data_ptr<float>(), M_alpha_0.data_ptr<float>(), G_prefix_chunk.data_ptr<float>(), \
        K_saved.data_ptr<float>(), V_saved.data_ptr<float>(), Eta_saved.data_ptr<float>(), \
        Alpha_saved.data_ptr<float>(), EtaGrad_saved.data_ptr<float>(), AlphaGrad_saved.data_ptr<float>(), \
        InvK_saved.data_ptr<float>(), QInv_saved.data_ptr<float>(), G_checkpoint_saved.data_ptr<float>(), \
        G_final_chunk.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(dX.data_ptr<at::BFloat16>()), \
        reinterpret_cast<__nv_bfloat16*>(dQRaw.data_ptr<at::BFloat16>()), \
        dM_m_0.data_ptr<float>(), dM_v_0.data_ptr<float>(), dM_k_0.data_ptr<float>(), \
        dM_eta_0.data_ptr<float>(), dM_alpha_0.data_ptr<float>(), \
        ${call_parameters} \
    } while (0);

std::vector<torch::Tensor> srnl_backward_qraw_${layout}(
    torch::Tensor dY, torch::Tensor X, torch::Tensor QRaw,
    torch::Tensor M_m_0, torch::Tensor M_v_0, torch::Tensor M_k_0,
    torch::Tensor M_eta_0, torch::Tensor M_alpha_0, torch::Tensor G_prefix_chunk,
    torch::Tensor K_saved, torch::Tensor V_saved,
    torch::Tensor Eta_saved, torch::Tensor Alpha_saved,
    torch::Tensor EtaGrad_saved, torch::Tensor AlphaGrad_saved,
    torch::Tensor InvK_saved, torch::Tensor QInv_saved,
    torch::Tensor G_checkpoint_saved, torch::Tensor G_final_chunk,
    ${api_parameters}

    int N = X.size(0);
    int L = X.size(1);
    ${chunk_counts}
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(X.device());
    auto dM_m_0 = torch::zeros({state_count, D_int, D_int}, options);
    auto dM_v_0 = torch::zeros({state_count, D_int, D_int}, options);
    auto dM_k_0 = torch::zeros({state_count, D_int, D_int}, options);
    auto dM_eta_0 = torch::zeros({state_count, D_int}, options);
    auto dM_alpha_0 = torch::zeros({state_count, D_int}, options);
    auto dX = torch::empty_like(X);
    auto dQRaw = torch::empty_like(QRaw);
    auto stream = at::cuda::getCurrentCUDAStream();
${backward_dispatch}
    return {dM_m_0, dM_v_0, dM_k_0, dM_eta_0, dM_alpha_0, dX, dQRaw};
}



torch::Tensor srnl_sequence_forward(
    torch::Tensor M_m_0, torch::Tensor M_v_0, torch::Tensor M_k_0,
    torch::Tensor M_eta_0, torch::Tensor M_alpha_0,
    torch::Tensor X, torch::Tensor Q_raw, std::vector<int64_t> chunk_sizes);

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
    int D_int);

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
    int D_int);

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
    int D_int);

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
    int D_int);

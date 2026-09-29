    if (D_int == 4) {
        if (bwd_groups >= 8) {
            if (use_cmax) { DISPATCH_SRNL_BWD_QRAW_CMAX_GROUPED(4, 8) }
            else { DISPATCH_SRNL_BWD_QRAW_GROUPED(4, 8) }
        } else if (bwd_groups >= 4) {
            if (use_cmax) { DISPATCH_SRNL_BWD_QRAW_CMAX_GROUPED(4, 4) }
            else { DISPATCH_SRNL_BWD_QRAW_GROUPED(4, 4) }
        } else if (bwd_groups >= 2) {
            if (use_cmax) { DISPATCH_SRNL_BWD_QRAW_CMAX_GROUPED(4, 2) }
            else { DISPATCH_SRNL_BWD_QRAW_GROUPED(4, 2) }
        } else {
            if (use_cmax) { DISPATCH_SRNL_BWD_QRAW_CMAX1(4) }
            else { DISPATCH_SRNL_BWD_QRAW(4) }
        }
    }
    else if (D_int == 8)  {
        if (bwd_groups >= 4) {
            if (use_cmax) { DISPATCH_SRNL_BWD_QRAW_CMAX_GROUPED(8, 4) }
            else { DISPATCH_SRNL_BWD_QRAW_GROUPED(8, 4) }
        } else if (bwd_groups >= 2) {
            if (use_cmax) { DISPATCH_SRNL_BWD_QRAW_CMAX_GROUPED(8, 2) }
            else { DISPATCH_SRNL_BWD_QRAW_GROUPED(8, 2) }
        } else {
            if (use_cmax) { DISPATCH_SRNL_BWD_QRAW_CMAX1(8) }
            else { DISPATCH_SRNL_BWD_QRAW(8) }
        }
    }
    else if (D_int == 16) {
        if (!use_cmax) {
            if (bwd_groups >= 3) { DISPATCH_SRNL_BWD_QRAW_GROUPED(16, 3) }
            else if (bwd_groups >= 2) { DISPATCH_SRNL_BWD_QRAW_GROUPED(16, 2) }
            else { DISPATCH_SRNL_BWD_QRAW(16) }
        } else if (bwd_groups >= 3) {
            if (chunk_size <= 8) { DISPATCH_SRNL_BWD_QRAW_GROUPED_CMAX(16, 3, 8) }
            else if (chunk_size <= 16) { DISPATCH_SRNL_BWD_QRAW_GROUPED_CMAX(16, 3, 16) }
            else if (chunk_size <= 32) { DISPATCH_SRNL_BWD_QRAW_GROUPED_CMAX(16, 3, 32) }
            else if (chunk_size <= 64) { DISPATCH_SRNL_BWD_QRAW_GROUPED_CMAX(16, 3, 64) }
            else { DISPATCH_SRNL_BWD_QRAW_GROUPED(16, 3) }
        } else if (bwd_groups >= 2) {
            if (chunk_size <= 8) { DISPATCH_SRNL_BWD_QRAW_GROUPED_CMAX(16, 2, 8) }
            else if (chunk_size <= 16) { DISPATCH_SRNL_BWD_QRAW_GROUPED_CMAX(16, 2, 16) }
            else if (chunk_size <= 32) { DISPATCH_SRNL_BWD_QRAW_GROUPED_CMAX(16, 2, 32) }
            else if (chunk_size <= 64) { DISPATCH_SRNL_BWD_QRAW_GROUPED_CMAX(16, 2, 64) }
            else { DISPATCH_SRNL_BWD_QRAW_GROUPED(16, 2) }
        } else {
            if (chunk_size <= 8) { DISPATCH_SRNL_BWD_QRAW_GROUPED_CMAX(16, 1, 8) }
            else if (chunk_size <= 16) { DISPATCH_SRNL_BWD_QRAW_GROUPED_CMAX(16, 1, 16) }
            else if (chunk_size <= 32) { DISPATCH_SRNL_BWD_QRAW_GROUPED_CMAX(16, 1, 32) }
            else if (chunk_size <= 64) { DISPATCH_SRNL_BWD_QRAW_GROUPED_CMAX(16, 1, 64) }
            else { DISPATCH_SRNL_BWD_QRAW(16) }
        }
    }
    else if (D_int == 32) {
        if (bwd_groups >= 2 && use_cmax && chunk_size <= 8) {
            DISPATCH_SRNL_BWD_QRAW_GROUPED_CMAX(32, 2, 8)
        } else {
            if (use_cmax) { DISPATCH_SRNL_BWD_QRAW_CMAX1(32) }
            else { DISPATCH_SRNL_BWD_QRAW(32) }
        }
    }
    else { throw std::runtime_error("HOPE qraw kernel solely supports head_dim in {4, 8, 16, 32}."); }

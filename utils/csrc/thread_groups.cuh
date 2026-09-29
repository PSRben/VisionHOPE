
__host__ __forceinline__ int srnl_parallel_groups(int dimension, int work, int thread_budget) {
    int groups = 1;
    while (groups < work && 2 * groups * dimension <= thread_budget) groups *= 2;
    return groups;
}

template <int D>
__device__ __forceinline__ unsigned srnl_group_mask(int group) {
    if constexpr (D >= 32) return 0xffffffffu;
    else return ((1u << D) - 1u) << ((group * D) & 31);
}

template <int D>
__device__ __forceinline__ void srnl_group_sync(int group) {
    if constexpr (D <= 32) {
        __syncwarp(srnl_group_mask<D>(group));
    } else {
        // Use one barrier per 64-thread group; barrier 0 is reserved for the CTA.
        // Callers limit D=64 to four groups per block.
        asm volatile("bar.sync %0, %1;" : : "r"(group + 1), "r"(D) : "memory");
    }
}

template <int D>
__device__ __forceinline__ float srnl_group_sum(float value, int lane, int group) {
    if constexpr (D <= 32) {
        unsigned mask = srnl_group_mask<D>(group);
        #pragma unroll
        for (int offset = D / 2; offset > 0; offset /= 2)
            value += __shfl_down_sync(mask, value, offset, D);
        return __shfl_sync(mask, value, 0, D);
    } else {
        __shared__ float partial[4][2];
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
            value += __shfl_down_sync(0xffffffffu, value, offset);
        if ((lane & 31) == 0) partial[group][lane / 32] = value;
        srnl_group_sync<D>(group);
        float result = partial[group][0] + partial[group][1];
        srnl_group_sync<D>(group);
        return result;
    }
}

template <int D>
__device__ __forceinline__ float srnl_group_first(float value, int lane, int group) {
    if constexpr (D <= 32) {
        return __shfl_sync(srnl_group_mask<D>(group), value, 0, D);
    } else {
        __shared__ float first[4];
        if (lane == 0) first[group] = value;
        srnl_group_sync<D>(group);
        float result = first[group];
        srnl_group_sync<D>(group);
        return result;
    }
}

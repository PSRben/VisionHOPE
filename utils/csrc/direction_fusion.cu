
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

__forceinline__ __device__ float io_load(const __nv_bfloat16* ptr, int64_t idx) {
    return __bfloat162float(ptr[idx]);
}

template <bool SharedSkip>
__global__ void visionhope_postprocess_scale_forward_kernel(
    const __nv_bfloat16* __restrict__ Y,
    const __nv_bfloat16* __restrict__ X,
    const float* __restrict__ DSkip,
    const float* __restrict__ Scale,
    float* __restrict__ Out,
    int B, int NumHeads, int H, int W, int D, bool ChannelsLast)
{
    int C = NumHeads * D;
    int L = H * W;
    int64_t total = (int64_t)B * C * H * W;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int w = idx % W;
    int h_img = (idx / W) % H;
    int c = (idx / (W * H)) % C;
    int b = idx / ((int64_t)C * H * W);
    int head = c / D;
    int d_i = c - head * D;
    int ds_idx = head * D + d_i;

    int l_hw = h_img * W + w;
    int l_hw_rev = L - 1 - l_hw;
    int l_wh = w * H + h_img;
    int l_wh_rev = L - 1 - l_wh;

    int64_t base_b = ((int64_t)b * 4 * NumHeads + head) * L * D + d_i;
    int64_t stride_dir = (int64_t)NumHeads * L * D;
    int64_t stride_l = D;

    int64_t idx0 = base_b + (int64_t)0 * stride_dir + (int64_t)l_hw * stride_l;
    int64_t idx1 = base_b + (int64_t)1 * stride_dir + (int64_t)l_hw_rev * stride_l;
    int64_t idx2 = base_b + (int64_t)2 * stride_dir + (int64_t)l_wh * stride_l;
    int64_t idx3 = base_b + (int64_t)3 * stride_dir + (int64_t)l_wh_rev * stride_l;

    float ds = DSkip[ds_idx];
    float ds1 = SharedSkip ? ds : DSkip[C + ds_idx];
    float ds2 = SharedSkip ? ds : DSkip[2 * C + ds_idx];
    float ds3 = SharedSkip ? ds : DSkip[3 * C + ds_idx];
    float v0 = io_load(Y, idx0) + ds * io_load(X, idx0);
    float v1 = io_load(Y, idx1) + ds1 * io_load(X, idx1);
    float v2 = io_load(Y, idx2) + ds2 * io_load(X, idx2);
    float v3 = io_load(Y, idx3) + ds3 * io_load(X, idx3);

    float out = Scale[c] * v0;
    out += Scale[C + c] * v1;
    out += Scale[2 * C + c] * v2;
    out += Scale[3 * C + c] * v3;
    int64_t out_idx = ChannelsLast
        ? (((int64_t)b * H + h_img) * W + w) * C + c
        : idx;
    Out[out_idx] = out;
}

template <bool SharedSkip>
__global__ void visionhope_postprocess_scale_forward_bf16_kernel(
    const __nv_bfloat16* __restrict__ Y,
    const __nv_bfloat16* __restrict__ X,
    const float* __restrict__ DSkip,
    const float* __restrict__ Scale,
    __nv_bfloat16* __restrict__ Out,
    int B, int NumHeads, int H, int W, int D, bool ChannelsLast)
{
    int C = NumHeads * D;
    int L = H * W;
    int64_t total = (int64_t)B * C * H * W;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int w = idx % W;
    int h_img = (idx / W) % H;
    int c = (idx / (W * H)) % C;
    int b = idx / ((int64_t)C * H * W);
    int head = c / D;
    int d_i = c - head * D;
    int ds_idx = head * D + d_i;

    int l_hw = h_img * W + w;
    int l_hw_rev = L - 1 - l_hw;
    int l_wh = w * H + h_img;
    int l_wh_rev = L - 1 - l_wh;

    int64_t base_b = ((int64_t)b * 4 * NumHeads + head) * L * D + d_i;
    int64_t stride_dir = (int64_t)NumHeads * L * D;
    int64_t stride_l = D;

    int64_t idx0 = base_b + (int64_t)0 * stride_dir + (int64_t)l_hw * stride_l;
    int64_t idx1 = base_b + (int64_t)1 * stride_dir + (int64_t)l_hw_rev * stride_l;
    int64_t idx2 = base_b + (int64_t)2 * stride_dir + (int64_t)l_wh * stride_l;
    int64_t idx3 = base_b + (int64_t)3 * stride_dir + (int64_t)l_wh_rev * stride_l;

    float ds = DSkip[ds_idx];
    float ds1 = SharedSkip ? ds : DSkip[C + ds_idx];
    float ds2 = SharedSkip ? ds : DSkip[2 * C + ds_idx];
    float ds3 = SharedSkip ? ds : DSkip[3 * C + ds_idx];
    float v0 = io_load(Y, idx0) + ds * io_load(X, idx0);
    float v1 = io_load(Y, idx1) + ds1 * io_load(X, idx1);
    float v2 = io_load(Y, idx2) + ds2 * io_load(X, idx2);
    float v3 = io_load(Y, idx3) + ds3 * io_load(X, idx3);

    float out = Scale[c] * v0;
    out += Scale[C + c] * v1;
    out += Scale[2 * C + c] * v2;
    out += Scale[3 * C + c] * v3;
    int64_t out_idx = ChannelsLast
        ? (((int64_t)b * H + h_img) * W + w) * C + c
        : idx;
    Out[out_idx] = __float2bfloat16(out);
}

template <bool SharedSkip>
__global__ void visionhope_postprocess_scale_backward_channel_kernel(
    const float* __restrict__ DOut,
    const __nv_bfloat16* __restrict__ Y,
    const __nv_bfloat16* __restrict__ X,
    const float* __restrict__ DSkip,
    const float* __restrict__ Scale,
    __nv_bfloat16* __restrict__ DY,
    __nv_bfloat16* __restrict__ DX,
    float* __restrict__ DDSkip,
    float* __restrict__ DScale,
    int B, int NumHeads, int H, int W, int D,
    int64_t DOutS0, int64_t DOutS1, int64_t DOutS2, int64_t DOutS3)
{
    int b = blockIdx.x;
    int c = blockIdx.y;
    int tid = threadIdx.x;
    int C = NumHeads * D;
    int L = H * W;
    int head = c / D;
    int d_i = c - head * D;
    int ds_idx = head * D + d_i;

    int64_t base_b = ((int64_t)b * 4 * NumHeads + head) * L * D + d_i;
    int64_t stride_dir = (int64_t)NumHeads * L * D;
    int64_t stride_l = D;

    float ds = DSkip[ds_idx];
    float ds1 = SharedSkip ? ds : DSkip[C + ds_idx];
    float ds2 = SharedSkip ? ds : DSkip[2 * C + ds_idx];
    float ds3 = SharedSkip ? ds : DSkip[3 * C + ds_idx];
    float sc0 = Scale[c];
    float sc1 = Scale[C + c];
    float sc2 = Scale[2 * C + c];
    float sc3 = Scale[3 * C + c];

    constexpr int SkipDirs = SharedSkip ? 1 : 4;
    float acc_dskip[SkipDirs] = {};
    float acc_s0 = 0.0f;
    float acc_s1 = 0.0f;
    float acc_s2 = 0.0f;
    float acc_s3 = 0.0f;

    for (int l_out = tid; l_out < L; l_out += blockDim.x) {
        int h_img = l_out / W;
        int w = l_out - h_img * W;
        int l_hw = l_out;
        int l_hw_rev = L - 1 - l_hw;
        int l_wh = w * H + h_img;
        int l_wh_rev = L - 1 - l_wh;

        int64_t idx0 = base_b + (int64_t)0 * stride_dir + (int64_t)l_hw * stride_l;
        int64_t idx1 = base_b + (int64_t)1 * stride_dir + (int64_t)l_hw_rev * stride_l;
        int64_t idx2 = base_b + (int64_t)2 * stride_dir + (int64_t)l_wh * stride_l;
        int64_t idx3 = base_b + (int64_t)3 * stride_dir + (int64_t)l_wh_rev * stride_l;
        int64_t out_idx = (int64_t)b * DOutS0 + (int64_t)c * DOutS1 + (int64_t)h_img * DOutS2 + (int64_t)w * DOutS3;

        float go = DOut[out_idx];
        float x0 = io_load(X, idx0);
        float x1 = io_load(X, idx1);
        float x2 = io_load(X, idx2);
        float x3 = io_load(X, idx3);
        float y0 = io_load(Y, idx0);
        float y1 = io_load(Y, idx1);
        float y2 = io_load(Y, idx2);
        float y3 = io_load(Y, idx3);

        DY[idx0] = __float2bfloat16(go * sc0);
        DY[idx1] = __float2bfloat16(go * sc1);
        DY[idx2] = __float2bfloat16(go * sc2);
        DY[idx3] = __float2bfloat16(go * sc3);
        DX[idx0] = __float2bfloat16(go * sc0 * ds);
        DX[idx1] = __float2bfloat16(go * sc1 * ds1);
        DX[idx2] = __float2bfloat16(go * sc2 * ds2);
        DX[idx3] = __float2bfloat16(go * sc3 * ds3);

        if constexpr (SharedSkip) {
            acc_dskip[0] += go * (sc0 * x0 + sc1 * x1 + sc2 * x2 + sc3 * x3);
        } else {
            acc_dskip[0] += go * sc0 * x0;
            acc_dskip[1] += go * sc1 * x1;
            acc_dskip[2] += go * sc2 * x2;
            acc_dskip[3] += go * sc3 * x3;
        }
        acc_s0 += go * (y0 + ds * x0);
        acc_s1 += go * (y1 + ds1 * x1);
        acc_s2 += go * (y2 + ds2 * x2);
        acc_s3 += go * (y3 + ds3 * x3);
    }

    __shared__ float sh_dskip[SkipDirs][256];
    __shared__ float sh_s0[256];
    __shared__ float sh_s1[256];
    __shared__ float sh_s2[256];
    __shared__ float sh_s3[256];
    for (int direction = 0; direction < SkipDirs; ++direction)
        sh_dskip[direction][tid] = acc_dskip[direction];
    sh_s0[tid] = acc_s0;
    sh_s1[tid] = acc_s1;
    sh_s2[tid] = acc_s2;
    sh_s3[tid] = acc_s3;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            for (int direction = 0; direction < SkipDirs; ++direction)
                sh_dskip[direction][tid] += sh_dskip[direction][tid + offset];
            sh_s0[tid] += sh_s0[tid + offset];
            sh_s1[tid] += sh_s1[tid + offset];
            sh_s2[tid] += sh_s2[tid + offset];
            sh_s3[tid] += sh_s3[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        for (int direction = 0; direction < SkipDirs; ++direction)
            atomicAdd(&DDSkip[direction * C + ds_idx], sh_dskip[direction][0]);
        atomicAdd(&DScale[c], sh_s0[0]);
        atomicAdd(&DScale[C + c], sh_s1[0]);
        atomicAdd(&DScale[2 * C + c], sh_s2[0]);
        atomicAdd(&DScale[3 * C + c], sh_s3[0]);
    }
}

template <bool SharedSkip>
__global__ void visionhope_postprocess_scale_backward_cl_tile_kernel(
    const float* __restrict__ DOut,
    const __nv_bfloat16* __restrict__ Y,
    const __nv_bfloat16* __restrict__ X,
    const float* __restrict__ DSkip,
    const float* __restrict__ Scale,
    __nv_bfloat16* __restrict__ DY,
    __nv_bfloat16* __restrict__ DX,
    float* __restrict__ DDSkip,
    float* __restrict__ DScale,
    int B, int NumHeads, int H, int W, int D,
    int64_t DOutS0, int64_t DOutS1, int64_t DOutS2, int64_t DOutS3)
{
    int b = blockIdx.x;
    int tid = threadIdx.x;
    int c_lane = tid & 15;
    int l_lane = tid >> 4;
    int C = NumHeads * D;
    int L = H * W;
    int c = blockIdx.y * 16 + c_lane;
    int head = c / D;
    int head_lane = c % D;
    int ds_idx = c;

    int64_t base_b = ((int64_t)b * 4 * NumHeads + head) * L * D + head_lane;
    int64_t stride_dir = (int64_t)NumHeads * L * D;
    int64_t stride_l = D;

    float ds = (c < C ? DSkip[ds_idx] : 0.0f);
    float ds1 = SharedSkip ? ds : (c < C ? DSkip[C + ds_idx] : 0.0f);
    float ds2 = SharedSkip ? ds : (c < C ? DSkip[2 * C + ds_idx] : 0.0f);
    float ds3 = SharedSkip ? ds : (c < C ? DSkip[3 * C + ds_idx] : 0.0f);
    float sc0 = c < C ? Scale[c] : 0.0f;
    float sc1 = c < C ? Scale[C + c] : 0.0f;
    float sc2 = c < C ? Scale[2 * C + c] : 0.0f;
    float sc3 = c < C ? Scale[3 * C + c] : 0.0f;

    constexpr int SkipDirs = SharedSkip ? 1 : 4;
    float acc_dskip[SkipDirs] = {};
    float acc_s0 = 0.0f;
    float acc_s1 = 0.0f;
    float acc_s2 = 0.0f;
    float acc_s3 = 0.0f;

    for (int l_out = l_lane; c < C && l_out < L; l_out += 16) {
        int h_img = l_out / W;
        int w = l_out - h_img * W;
        int l_hw = l_out;
        int l_hw_rev = L - 1 - l_hw;
        int l_wh = w * H + h_img;
        int l_wh_rev = L - 1 - l_wh;

        int64_t idx0 = base_b + (int64_t)0 * stride_dir + (int64_t)l_hw * stride_l;
        int64_t idx1 = base_b + (int64_t)1 * stride_dir + (int64_t)l_hw_rev * stride_l;
        int64_t idx2 = base_b + (int64_t)2 * stride_dir + (int64_t)l_wh * stride_l;
        int64_t idx3 = base_b + (int64_t)3 * stride_dir + (int64_t)l_wh_rev * stride_l;
        int64_t out_idx = (int64_t)b * DOutS0 + (int64_t)c * DOutS1 + (int64_t)h_img * DOutS2 + (int64_t)w * DOutS3;

        float go = DOut[out_idx];
        float x0 = io_load(X, idx0);
        float x1 = io_load(X, idx1);
        float x2 = io_load(X, idx2);
        float x3 = io_load(X, idx3);
        float y0 = io_load(Y, idx0);
        float y1 = io_load(Y, idx1);
        float y2 = io_load(Y, idx2);
        float y3 = io_load(Y, idx3);

        DY[idx0] = __float2bfloat16(go * sc0);
        DY[idx1] = __float2bfloat16(go * sc1);
        DY[idx2] = __float2bfloat16(go * sc2);
        DY[idx3] = __float2bfloat16(go * sc3);
        DX[idx0] = __float2bfloat16(go * sc0 * ds);
        DX[idx1] = __float2bfloat16(go * sc1 * ds1);
        DX[idx2] = __float2bfloat16(go * sc2 * ds2);
        DX[idx3] = __float2bfloat16(go * sc3 * ds3);

        if constexpr (SharedSkip) {
            acc_dskip[0] += go * (sc0 * x0 + sc1 * x1 + sc2 * x2 + sc3 * x3);
        } else {
            acc_dskip[0] += go * sc0 * x0;
            acc_dskip[1] += go * sc1 * x1;
            acc_dskip[2] += go * sc2 * x2;
            acc_dskip[3] += go * sc3 * x3;
        }
        acc_s0 += go * (y0 + ds * x0);
        acc_s1 += go * (y1 + ds1 * x1);
        acc_s2 += go * (y2 + ds2 * x2);
        acc_s3 += go * (y3 + ds3 * x3);
    }

    __shared__ float sh_dskip[SkipDirs][16][16];
    __shared__ float sh_s0[16][16];
    __shared__ float sh_s1[16][16];
    __shared__ float sh_s2[16][16];
    __shared__ float sh_s3[16][16];
    for (int direction = 0; direction < SkipDirs; ++direction)
        sh_dskip[direction][l_lane][c_lane] = acc_dskip[direction];
    sh_s0[l_lane][c_lane] = acc_s0;
    sh_s1[l_lane][c_lane] = acc_s1;
    sh_s2[l_lane][c_lane] = acc_s2;
    sh_s3[l_lane][c_lane] = acc_s3;
    __syncthreads();

    for (int offset = 8; offset > 0; offset >>= 1) {
        if (l_lane < offset) {
            for (int direction = 0; direction < SkipDirs; ++direction)
                sh_dskip[direction][l_lane][c_lane] += sh_dskip[direction][l_lane + offset][c_lane];
            sh_s0[l_lane][c_lane] += sh_s0[l_lane + offset][c_lane];
            sh_s1[l_lane][c_lane] += sh_s1[l_lane + offset][c_lane];
            sh_s2[l_lane][c_lane] += sh_s2[l_lane + offset][c_lane];
            sh_s3[l_lane][c_lane] += sh_s3[l_lane + offset][c_lane];
        }
        __syncthreads();
    }

    if (l_lane == 0 && c < C) {
        for (int direction = 0; direction < SkipDirs; ++direction)
            atomicAdd(&DDSkip[direction * C + ds_idx], sh_dskip[direction][0][c_lane]);
        atomicAdd(&DScale[c], sh_s0[0][c_lane]);
        atomicAdd(&DScale[C + c], sh_s1[0][c_lane]);
        atomicAdd(&DScale[2 * C + c], sh_s2[0][c_lane]);
        atomicAdd(&DScale[3 * C + c], sh_s3[0][c_lane]);
    }
}

template <bool SharedSkip>
__global__ void visionhope_postprocess_scale_backward_cl_ltile_kernel(
    const float* __restrict__ DOut,
    const __nv_bfloat16* __restrict__ Y,
    const __nv_bfloat16* __restrict__ X,
    const float* __restrict__ DSkip,
    const float* __restrict__ Scale,
    __nv_bfloat16* __restrict__ DY,
    __nv_bfloat16* __restrict__ DX,
    float* __restrict__ DDSkip,
    float* __restrict__ DScale,
    int B, int NumHeads, int H, int W, int D, int TileL,
    int64_t DOutS0, int64_t DOutS1, int64_t DOutS2, int64_t DOutS3)
{
    int b = blockIdx.x;
    int tile = blockIdx.z;
    int tid = threadIdx.x;
    int c_lane = tid & 15;
    int l_lane = tid >> 4;
    int C = NumHeads * D;
    int L = H * W;
    int c = blockIdx.y * 16 + c_lane;
    int head = c / D;
    int head_lane = c % D;
    int ds_idx = c;
    int tile_start = tile * TileL;
    int tile_end = min(tile_start + TileL, L);

    int64_t base_b = ((int64_t)b * 4 * NumHeads + head) * L * D + head_lane;
    int64_t stride_dir = (int64_t)NumHeads * L * D;
    int64_t stride_l = D;

    float ds = (c < C ? DSkip[ds_idx] : 0.0f);
    float ds1 = SharedSkip ? ds : (c < C ? DSkip[C + ds_idx] : 0.0f);
    float ds2 = SharedSkip ? ds : (c < C ? DSkip[2 * C + ds_idx] : 0.0f);
    float ds3 = SharedSkip ? ds : (c < C ? DSkip[3 * C + ds_idx] : 0.0f);
    float sc0 = c < C ? Scale[c] : 0.0f;
    float sc1 = c < C ? Scale[C + c] : 0.0f;
    float sc2 = c < C ? Scale[2 * C + c] : 0.0f;
    float sc3 = c < C ? Scale[3 * C + c] : 0.0f;

    constexpr int SkipDirs = SharedSkip ? 1 : 4;
    float acc_dskip[SkipDirs] = {};
    float acc_s0 = 0.0f;
    float acc_s1 = 0.0f;
    float acc_s2 = 0.0f;
    float acc_s3 = 0.0f;

    for (int l_out = tile_start + l_lane; c < C && l_out < tile_end; l_out += 16) {
        int h_img = l_out / W;
        int w = l_out - h_img * W;
        int l_hw = l_out;
        int l_hw_rev = L - 1 - l_hw;
        int l_wh = w * H + h_img;
        int l_wh_rev = L - 1 - l_wh;

        int64_t idx0 = base_b + (int64_t)0 * stride_dir + (int64_t)l_hw * stride_l;
        int64_t idx1 = base_b + (int64_t)1 * stride_dir + (int64_t)l_hw_rev * stride_l;
        int64_t idx2 = base_b + (int64_t)2 * stride_dir + (int64_t)l_wh * stride_l;
        int64_t idx3 = base_b + (int64_t)3 * stride_dir + (int64_t)l_wh_rev * stride_l;
        int64_t out_idx = (int64_t)b * DOutS0 + (int64_t)c * DOutS1 + (int64_t)h_img * DOutS2 + (int64_t)w * DOutS3;

        float go = DOut[out_idx];
        float x0 = io_load(X, idx0);
        float x1 = io_load(X, idx1);
        float x2 = io_load(X, idx2);
        float x3 = io_load(X, idx3);
        float y0 = io_load(Y, idx0);
        float y1 = io_load(Y, idx1);
        float y2 = io_load(Y, idx2);
        float y3 = io_load(Y, idx3);

        DY[idx0] = __float2bfloat16(go * sc0);
        DY[idx1] = __float2bfloat16(go * sc1);
        DY[idx2] = __float2bfloat16(go * sc2);
        DY[idx3] = __float2bfloat16(go * sc3);
        DX[idx0] = __float2bfloat16(go * sc0 * ds);
        DX[idx1] = __float2bfloat16(go * sc1 * ds1);
        DX[idx2] = __float2bfloat16(go * sc2 * ds2);
        DX[idx3] = __float2bfloat16(go * sc3 * ds3);

        if constexpr (SharedSkip) {
            acc_dskip[0] += go * (sc0 * x0 + sc1 * x1 + sc2 * x2 + sc3 * x3);
        } else {
            acc_dskip[0] += go * sc0 * x0;
            acc_dskip[1] += go * sc1 * x1;
            acc_dskip[2] += go * sc2 * x2;
            acc_dskip[3] += go * sc3 * x3;
        }
        acc_s0 += go * (y0 + ds * x0);
        acc_s1 += go * (y1 + ds1 * x1);
        acc_s2 += go * (y2 + ds2 * x2);
        acc_s3 += go * (y3 + ds3 * x3);
    }

    __shared__ float sh_dskip[SkipDirs][16][16];
    __shared__ float sh_s0[16][16];
    __shared__ float sh_s1[16][16];
    __shared__ float sh_s2[16][16];
    __shared__ float sh_s3[16][16];
    for (int direction = 0; direction < SkipDirs; ++direction)
        sh_dskip[direction][l_lane][c_lane] = acc_dskip[direction];
    sh_s0[l_lane][c_lane] = acc_s0;
    sh_s1[l_lane][c_lane] = acc_s1;
    sh_s2[l_lane][c_lane] = acc_s2;
    sh_s3[l_lane][c_lane] = acc_s3;
    __syncthreads();

    for (int offset = 8; offset > 0; offset >>= 1) {
        if (l_lane < offset) {
            for (int direction = 0; direction < SkipDirs; ++direction)
                sh_dskip[direction][l_lane][c_lane] += sh_dskip[direction][l_lane + offset][c_lane];
            sh_s0[l_lane][c_lane] += sh_s0[l_lane + offset][c_lane];
            sh_s1[l_lane][c_lane] += sh_s1[l_lane + offset][c_lane];
            sh_s2[l_lane][c_lane] += sh_s2[l_lane + offset][c_lane];
            sh_s3[l_lane][c_lane] += sh_s3[l_lane + offset][c_lane];
        }
        __syncthreads();
    }

    if (l_lane == 0 && c < C) {
        for (int direction = 0; direction < SkipDirs; ++direction)
            atomicAdd(&DDSkip[direction * C + ds_idx], sh_dskip[direction][0][c_lane]);
        atomicAdd(&DScale[c], sh_s0[0][c_lane]);
        atomicAdd(&DScale[C + c], sh_s1[0][c_lane]);
        atomicAdd(&DScale[2 * C + c], sh_s2[0][c_lane]);
        atomicAdd(&DScale[3 * C + c], sh_s3[0][c_lane]);
    }
}

torch::Tensor visionhope_postprocess_scale_forward(
    torch::Tensor Y, torch::Tensor X, torch::Tensor DSkip, torch::Tensor Scale,
    int H, int W, bool ChannelsLast, bool OutBf16)
{
    if (Y.scalar_type() != at::ScalarType::BFloat16 || X.scalar_type() != at::ScalarType::BFloat16) {
        throw std::runtime_error("visionhope_postprocess_scale_forward expects bf16 Y and X.");
    }
    if (DSkip.scalar_type() != at::ScalarType::Float || Scale.scalar_type() != at::ScalarType::Float) {
        throw std::runtime_error("visionhope_postprocess_scale_forward expects fp32 d_skip and scale.");
    }
    int B = Y.size(0);
    int NumHeads = Y.size(2);
    int L = Y.size(3);
    int D = Y.size(4);
    TORCH_CHECK(DSkip.numel() == NumHeads * D || DSkip.numel() == 4 * NumHeads * D, "d_skip must contain C or 4*C values");
    bool shared_skip = DSkip.numel() == NumHeads * D;
    if (Y.size(1) != 4 || X.sizes() != Y.sizes() || H * W != L || Scale.size(0) != 4 || Scale.size(1) != NumHeads * D) {
        throw std::runtime_error("Invalid shape for visionhope_postprocess_scale_forward.");
    }

    auto out_options = Y.options().dtype(OutBf16 ? torch::kBFloat16 : torch::kFloat32);
    torch::Tensor Out;
    if (ChannelsLast) {
        int C = NumHeads * D;
        Out = torch::empty_strided({B, C, H, W}, {(int64_t)H * W * C, 1, (int64_t)W * C, C}, out_options);
    } else {
        Out = torch::empty({B, NumHeads * D, H, W}, out_options);
    }
    int64_t total = (int64_t)B * NumHeads * D * H * W;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    auto stream = at::cuda::getCurrentCUDAStream();
    if (OutBf16) {
        if (shared_skip) {
            visionhope_postprocess_scale_forward_bf16_kernel<true><<<blocks, threads, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(Y.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>()),
                DSkip.data_ptr<float>(),
                Scale.data_ptr<float>(),
                reinterpret_cast<__nv_bfloat16*>(Out.data_ptr<at::BFloat16>()),
                B, NumHeads, H, W, D, ChannelsLast);
        } else {
            visionhope_postprocess_scale_forward_bf16_kernel<false><<<blocks, threads, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(Y.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>()),
                DSkip.data_ptr<float>(),
                Scale.data_ptr<float>(),
                reinterpret_cast<__nv_bfloat16*>(Out.data_ptr<at::BFloat16>()),
                B, NumHeads, H, W, D, ChannelsLast);
        }
    } else {
        if (shared_skip) {
            visionhope_postprocess_scale_forward_kernel<true><<<blocks, threads, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(Y.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>()),
                DSkip.data_ptr<float>(),
                Scale.data_ptr<float>(),
                Out.data_ptr<float>(),
                B, NumHeads, H, W, D, ChannelsLast);
        } else {
            visionhope_postprocess_scale_forward_kernel<false><<<blocks, threads, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(Y.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>()),
                DSkip.data_ptr<float>(),
                Scale.data_ptr<float>(),
                Out.data_ptr<float>(),
                B, NumHeads, H, W, D, ChannelsLast);
        }
    }
    return Out;
}

std::vector<torch::Tensor> visionhope_postprocess_scale_backward(
    torch::Tensor DOut, torch::Tensor Y, torch::Tensor X, torch::Tensor DSkip, torch::Tensor Scale,
    int H, int W, bool ChannelsLast, bool UseLTile, int TileL)
{
    if (DOut.scalar_type() != at::ScalarType::Float || Y.scalar_type() != at::ScalarType::BFloat16 || X.scalar_type() != at::ScalarType::BFloat16) {
        throw std::runtime_error("visionhope_postprocess_scale_backward expects fp32 DOut and bf16 Y/X.");
    }
    int B = Y.size(0);
    int NumHeads = Y.size(2);
    int L = Y.size(3);
    int D = Y.size(4);
    TORCH_CHECK(DSkip.numel() == NumHeads * D || DSkip.numel() == 4 * NumHeads * D, "d_skip must contain C or 4*C values");
    bool shared_skip = DSkip.numel() == NumHeads * D;
    if (Y.size(1) != 4 || X.sizes() != Y.sizes() || H * W != L || DOut.size(1) != NumHeads * D) {
        throw std::runtime_error("Invalid shape for visionhope_postprocess_scale_backward.");
    }

    auto DY = torch::empty_like(Y);
    auto DX = torch::empty_like(X);
    auto DDSkip = torch::zeros_like(DSkip);
    auto DScale = torch::zeros_like(Scale);
    auto stream = at::cuda::getCurrentCUDAStream();
    int threads = 256;
    if (ChannelsLast && UseLTile) {
        int tile_l = max(16, TileL);
        int num_tiles = (L + tile_l - 1) / tile_l;
        dim3 grid(B, (NumHeads * D + 15) / 16, num_tiles);
        if (shared_skip) {
            visionhope_postprocess_scale_backward_cl_ltile_kernel<true><<<grid, threads, 0, stream>>>(
                DOut.data_ptr<float>(),
                reinterpret_cast<const __nv_bfloat16*>(Y.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>()),
                DSkip.data_ptr<float>(),
                Scale.data_ptr<float>(),
                reinterpret_cast<__nv_bfloat16*>(DY.data_ptr<at::BFloat16>()),
                reinterpret_cast<__nv_bfloat16*>(DX.data_ptr<at::BFloat16>()),
                DDSkip.data_ptr<float>(),
                DScale.data_ptr<float>(),
                B, NumHeads, H, W, D, tile_l,
                DOut.stride(0), DOut.stride(1), DOut.stride(2), DOut.stride(3));
        } else {
            visionhope_postprocess_scale_backward_cl_ltile_kernel<false><<<grid, threads, 0, stream>>>(
                DOut.data_ptr<float>(),
                reinterpret_cast<const __nv_bfloat16*>(Y.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>()),
                DSkip.data_ptr<float>(),
                Scale.data_ptr<float>(),
                reinterpret_cast<__nv_bfloat16*>(DY.data_ptr<at::BFloat16>()),
                reinterpret_cast<__nv_bfloat16*>(DX.data_ptr<at::BFloat16>()),
                DDSkip.data_ptr<float>(),
                DScale.data_ptr<float>(),
                B, NumHeads, H, W, D, tile_l,
                DOut.stride(0), DOut.stride(1), DOut.stride(2), DOut.stride(3));
        }
    } else if (ChannelsLast) {
        dim3 grid(B, (NumHeads * D + 15) / 16);
        if (shared_skip) {
            visionhope_postprocess_scale_backward_cl_tile_kernel<true><<<grid, threads, 0, stream>>>(
                DOut.data_ptr<float>(),
                reinterpret_cast<const __nv_bfloat16*>(Y.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>()),
                DSkip.data_ptr<float>(),
                Scale.data_ptr<float>(),
                reinterpret_cast<__nv_bfloat16*>(DY.data_ptr<at::BFloat16>()),
                reinterpret_cast<__nv_bfloat16*>(DX.data_ptr<at::BFloat16>()),
                DDSkip.data_ptr<float>(),
                DScale.data_ptr<float>(),
                B, NumHeads, H, W, D,
                DOut.stride(0), DOut.stride(1), DOut.stride(2), DOut.stride(3));
        } else {
            visionhope_postprocess_scale_backward_cl_tile_kernel<false><<<grid, threads, 0, stream>>>(
                DOut.data_ptr<float>(),
                reinterpret_cast<const __nv_bfloat16*>(Y.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>()),
                DSkip.data_ptr<float>(),
                Scale.data_ptr<float>(),
                reinterpret_cast<__nv_bfloat16*>(DY.data_ptr<at::BFloat16>()),
                reinterpret_cast<__nv_bfloat16*>(DX.data_ptr<at::BFloat16>()),
                DDSkip.data_ptr<float>(),
                DScale.data_ptr<float>(),
                B, NumHeads, H, W, D,
                DOut.stride(0), DOut.stride(1), DOut.stride(2), DOut.stride(3));
        }
    } else {
        dim3 grid(B, NumHeads * D);
        if (shared_skip) {
            visionhope_postprocess_scale_backward_channel_kernel<true><<<grid, threads, 0, stream>>>(
                DOut.data_ptr<float>(),
                reinterpret_cast<const __nv_bfloat16*>(Y.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>()),
                DSkip.data_ptr<float>(),
                Scale.data_ptr<float>(),
                reinterpret_cast<__nv_bfloat16*>(DY.data_ptr<at::BFloat16>()),
                reinterpret_cast<__nv_bfloat16*>(DX.data_ptr<at::BFloat16>()),
                DDSkip.data_ptr<float>(),
                DScale.data_ptr<float>(),
                B, NumHeads, H, W, D,
                DOut.stride(0), DOut.stride(1), DOut.stride(2), DOut.stride(3));
        } else {
            visionhope_postprocess_scale_backward_channel_kernel<false><<<grid, threads, 0, stream>>>(
                DOut.data_ptr<float>(),
                reinterpret_cast<const __nv_bfloat16*>(Y.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>()),
                DSkip.data_ptr<float>(),
                Scale.data_ptr<float>(),
                reinterpret_cast<__nv_bfloat16*>(DY.data_ptr<at::BFloat16>()),
                reinterpret_cast<__nv_bfloat16*>(DX.data_ptr<at::BFloat16>()),
                DDSkip.data_ptr<float>(),
                DScale.data_ptr<float>(),
                B, NumHeads, H, W, D,
                DOut.stride(0), DOut.stride(1), DOut.stride(2), DOut.stride(3));
        }
    }
    return {DY, DX, DDSkip, DScale};
}

#include "../include/reductions.cuh"
#include "../include/kernel_common.cuh"
#include <cfloat>

// =========================================================================
// Sum Reduction
// =========================================================================
__global__ void reduce_sum_block_kernel(const float* __restrict__ in,
                                        float* __restrict__ block_sums,
                                        size_t n) {
    extern __shared__ float s_data[];

    size_t tid = threadIdx.x;
    size_t idx = static_cast<size_t>(blockIdx.x) * (blockDim.x * 2) + tid;
    size_t stride = static_cast<size_t>(blockDim.x) * 2 * gridDim.x;

    float sum = 0.0f;
    while (idx < n) {
        sum += in[idx];
        if (idx + blockDim.x < n) {
            sum += in[idx + blockDim.x];
        }
        idx += stride;
    }

    // Warp-level reduction
    sum = warp_reduce_sum(sum);

    int lane = tid % 32;
    int wid = tid / 32;

    if (lane == 0) {
        s_data[wid] = sum;
    }
    __syncthreads();

    // Final warp reduction across warps
    float block_sum = (tid < (blockDim.x / 32)) ? s_data[lane] : 0.0f;
    if (wid == 0) {
        block_sum = warp_reduce_sum(block_sum);
        if (tid == 0) {
            block_sums[blockIdx.x] = block_sum;
        }
    }
}

__global__ void reduce_sum_final_kernel(const float* __restrict__ block_sums,
                                        float* __restrict__ out,
                                        int num_blocks) {
    extern __shared__ float s_data[];

    int tid = threadIdx.x;
    float sum = (tid < num_blocks) ? block_sums[tid] : 0.0f;
    sum = warp_reduce_sum(sum);

    int lane = tid % 32;
    int wid = tid / 32;

    if (lane == 0) {
        s_data[wid] = sum;
    }
    __syncthreads();

    if (wid == 0) {
        float total = (tid < (blockDim.x / 32)) ? s_data[lane] : 0.0f;
        total = warp_reduce_sum(total);
        if (tid == 0) {
            out[0] = total;
        }
    }
}

void launch_reduce_sum(const float* in, float* out, float* d_workspace, size_t n, cudaStream_t stream) {
    if (n == 0) return;

    if (n == 1) {
        cudaMemcpyAsync(out, in, sizeof(float), cudaMemcpyDeviceToDevice, stream);
        return;
    }

    int threads = 256;
    int blocks = static_cast<int>((n + threads * 2 - 1) / (threads * 2));
    if (blocks > 1024) blocks = 1024;

    size_t shared_mem = (threads / 32) * sizeof(float);

    if (blocks == 1) {
        reduce_sum_block_kernel<<<1, threads, shared_mem, stream>>>(in, out, n);
    } else {
        reduce_sum_block_kernel<<<blocks, threads, shared_mem, stream>>>(in, d_workspace, n);
        reduce_sum_final_kernel<<<1, 1024, 32 * sizeof(float), stream>>>(d_workspace, out, blocks);
    }
}

// =========================================================================
// Max Reduction
// =========================================================================
__global__ void reduce_max_block_kernel(const float* __restrict__ in,
                                        float* __restrict__ block_maxs,
                                        size_t n) {
    extern __shared__ float s_data[];

    size_t tid = threadIdx.x;
    size_t idx = static_cast<size_t>(blockIdx.x) * (blockDim.x * 2) + tid;
    size_t stride = static_cast<size_t>(blockDim.x) * 2 * gridDim.x;

    float max_val = -FLT_MAX;
    while (idx < n) {
        max_val = fmaxf(max_val, in[idx]);
        if (idx + blockDim.x < n) {
            max_val = fmaxf(max_val, in[idx + blockDim.x]);
        }
        idx += stride;
    }

    max_val = warp_reduce_max(max_val);

    int lane = tid % 32;
    int wid = tid / 32;

    if (lane == 0) {
        s_data[wid] = max_val;
    }
    __syncthreads();

    float block_max = (tid < (blockDim.x / 32)) ? s_data[lane] : -FLT_MAX;
    if (wid == 0) {
        block_max = warp_reduce_max(block_max);
        if (tid == 0) {
            block_maxs[blockIdx.x] = block_max;
        }
    }
}

__global__ void reduce_max_final_kernel(const float* __restrict__ block_maxs,
                                        float* __restrict__ out,
                                        int num_blocks) {
    extern __shared__ float s_data[];

    int tid = threadIdx.x;
    float val = (tid < num_blocks) ? block_maxs[tid] : -FLT_MAX;
    val = warp_reduce_max(val);

    int lane = tid % 32;
    int wid = tid / 32;

    if (lane == 0) {
        s_data[wid] = val;
    }
    __syncthreads();

    if (wid == 0) {
        float total = (tid < (blockDim.x / 32)) ? s_data[lane] : -FLT_MAX;
        total = warp_reduce_max(total);
        if (tid == 0) {
            out[0] = total;
        }
    }
}

void launch_reduce_max(const float* in, float* out, float* d_workspace, size_t n, cudaStream_t stream) {
    if (n == 0) return;

    if (n == 1) {
        cudaMemcpyAsync(out, in, sizeof(float), cudaMemcpyDeviceToDevice, stream);
        return;
    }

    int threads = 256;
    int blocks = static_cast<int>((n + threads * 2 - 1) / (threads * 2));
    if (blocks > 1024) blocks = 1024;

    size_t shared_mem = (threads / 32) * sizeof(float);

    if (blocks == 1) {
        reduce_max_block_kernel<<<1, threads, shared_mem, stream>>>(in, out, n);
    } else {
        reduce_max_block_kernel<<<blocks, threads, shared_mem, stream>>>(in, d_workspace, n);
        reduce_max_final_kernel<<<1, 1024, 32 * sizeof(float), stream>>>(d_workspace, out, blocks);
    }
}

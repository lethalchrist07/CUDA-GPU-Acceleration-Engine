#include "../include/matmul_tiled.cuh"
#include "../include/kernel_common.cuh"

__global__ void tiled_matmul_kernel(const float* __restrict__ a,
                                    const float* __restrict__ b,
                                    float* __restrict__ c,
                                    int M,
                                    int K,
                                    int N) {
    __shared__ float s_A[TILE_DIM][TILE_DIM];
    __shared__ float s_B[TILE_DIM][TILE_DIM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_DIM + ty;
    int col = blockIdx.x * TILE_DIM + tx;

    float acc = 0.0f;
    int num_tiles = (K + TILE_DIM - 1) / TILE_DIM;

    for (int t = 0; t < num_tiles; ++t) {
        int a_col = t * TILE_DIM + tx;
        int b_row = t * TILE_DIM + ty;

        // Boundary checks when loading tile from global memory into shared memory
        s_A[ty][tx] = (row < M && a_col < K) ? a[row * K + a_col] : 0.0f;
        s_B[ty][tx] = (b_row < K && col < N) ? b[b_row * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            acc += s_A[ty][k] * s_B[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        c[row * N + col] = acc;
    }
}

void launch_tiled_matmul(const float* a,
                         const float* b,
                         float* c,
                         int M,
                         int K,
                         int N,
                         cudaStream_t stream) {
    if (M == 0 || K == 0 || N == 0) return;

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid(
        (N + TILE_DIM - 1) / TILE_DIM,
        (M + TILE_DIM - 1) / TILE_DIM
    );

    tiled_matmul_kernel<<<grid, block, 0, stream>>>(a, b, c, M, K, N);
}

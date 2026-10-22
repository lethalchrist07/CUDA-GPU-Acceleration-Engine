#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <type_traits>

#include "../include/matmul_wmma.cuh"
#include "../include/kernel_common.cuh"

using namespace nvcuda;

namespace {

// Tiling constants
constexpr int BM = 64;  // Block tile M
constexpr int BN = 64;  // Block tile N
constexpr int BK = 16;  // Block tile K (matches WMMA K dimension)
constexpr int WM = 32;  // Warp tile M
constexpr int WN = 32;  // Warp tile N

constexpr int BK_STRIDE = BK + 8;  // 24 (48 bytes: divisible by 16 bytes for WMMA load)
constexpr int BN_STRIDE = BN + 8;  // 72 (144 bytes: divisible by 16 bytes for WMMA load)
constexpr int SC_STRIDE = BN + 4;  // 68 (272 bytes: divisible by 16 bytes for WMMA store)

union __align__(16) SharedStorage {
    struct {
        half s_A[BM][BK_STRIDE]; // 64 * 24 * 2 = 3072 bytes
        half s_B[BK][BN_STRIDE]; // 16 * 72 * 2 = 2304 bytes
    } compute;
    float s_C[BM][SC_STRIDE];    // 64 * 68 * 4 = 17408 bytes
};

template<typename OutType>
__global__ void wmma_matmul_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    OutType* __restrict__ c,
    int M,
    int K,
    int N
) {
    __shared__ SharedStorage storage;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int warp_row = warp_id / 2;  // 0 or 1
    const int warp_col = warp_id % 2;  // 0 or 1

    // Global load indices for A: 128 threads load 64 * 16 = 1024 elements (8 elements/thread)
    const int a_row = tid / 2;               // 0 to 63
    const int a_col_base = (tid % 2) * 8;    // 0 or 8
    const int global_a_row = blockIdx.y * BM + a_row;

    // Global load indices for B: 128 threads load 16 * 64 = 1024 elements (8 elements/thread)
    const int b_row = tid / 8;               // 0 to 15
    const int b_col_base = (tid % 8) * 8;    // 0, 8, 16, 24, 32, 40, 48, 56
    const int global_b_col_base = blockIdx.x * BN + b_col_base;

    // Accumulators for this warp: 32x32 warp tile = 2x2 fragments of 16x16
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[2][2];
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }

    const int num_k_tiles = (K + BK - 1) / BK;

    for (int t = 0; t < num_k_tiles; ++t) {
        // 1. Load A tile from global to shared memory with boundary checks
        const int global_a_col_start = t * BK + a_col_base;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int col = global_a_col_start + i;
            storage.compute.s_A[a_row][a_col_base + i] =
                (global_a_row < M && col < K) ? a[global_a_row * K + col] : __float2half(0.0f);
        }

        // 2. Load B tile from global to shared memory with boundary checks
        const int global_b_row = t * BK + b_row;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int col = global_b_col_base + i;
            storage.compute.s_B[b_row][b_col_base + i] =
                (global_b_row < K && col < N) ? b[global_b_row * N + col] : __float2half(0.0f);
        }

        __syncthreads();

        // 3. Load WMMA fragments from shared memory
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag_0;
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag_1;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag_0;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag_1;

        wmma::load_matrix_sync(a_frag_0, &storage.compute.s_A[warp_row * WM + 0][0], BK_STRIDE);
        wmma::load_matrix_sync(a_frag_1, &storage.compute.s_A[warp_row * WM + 16][0], BK_STRIDE);
        wmma::load_matrix_sync(b_frag_0, &storage.compute.s_B[0][warp_col * WN + 0], BN_STRIDE);
        wmma::load_matrix_sync(b_frag_1, &storage.compute.s_B[0][warp_col * WN + 16], BN_STRIDE);

        // 4. Warp-level WMMA multiply-accumulate
        wmma::mma_sync(c_frag[0][0], a_frag_0, b_frag_0, c_frag[0][0]);
        wmma::mma_sync(c_frag[0][1], a_frag_0, b_frag_1, c_frag[0][1]);
        wmma::mma_sync(c_frag[1][0], a_frag_1, b_frag_0, c_frag[1][0]);
        wmma::mma_sync(c_frag[1][1], a_frag_1, b_frag_1, c_frag[1][1]);

        __syncthreads();
    }

    // 5. Store accumulated float fragments into shared memory tile s_C
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            int tile_row = warp_row * WM + i * 16;
            int tile_col = warp_col * WN + j * 16;
            wmma::store_matrix_sync(&storage.s_C[tile_row][tile_col], c_frag[i][j], SC_STRIDE, wmma::mem_row_major);
        }
    }

    __syncthreads();

    // 6. Coalesced write-back from shared memory to global memory with boundary checks
    #pragma unroll
    for (int iter = 0; iter < 32; ++iter) {
        int elem_idx = tid + iter * 128;
        int local_r = elem_idx / 64;
        int local_c = elem_idx % 64;
        int global_r = blockIdx.y * BM + local_r;
        int global_c = blockIdx.x * BN + local_c;

        if (global_r < M && global_c < N) {
            float val = storage.s_C[local_r][local_c];
            if constexpr (std::is_same_v<OutType, half>) {
                c[global_r * N + global_c] = __float2half(val);
            } else {
                c[global_r * N + global_c] = val;
            }
        }
    }
}

} // namespace

void launch_wmma_matmul_fp16_out(
    const half* a,
    const half* b,
    half* c,
    int M,
    int K,
    int N,
    cudaStream_t stream
) {
    if (M == 0 || N == 0) return;
    if (K == 0) {
        cudaMemsetAsync(c, 0, sizeof(half) * M * N, stream);
        return;
    }

    dim3 block(128, 1, 1);
    dim3 grid(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM,
        1
    );

    wmma_matmul_kernel<half><<<grid, block, 0, stream>>>(a, b, c, M, K, N);
}

void launch_wmma_matmul_fp32_out(
    const half* a,
    const half* b,
    float* c,
    int M,
    int K,
    int N,
    cudaStream_t stream
) {
    if (M == 0 || N == 0) return;
    if (K == 0) {
        cudaMemsetAsync(c, 0, sizeof(float) * M * N, stream);
        return;
    }

    dim3 block(128, 1, 1);
    dim3 grid(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM,
        1
    );

    wmma_matmul_kernel<float><<<grid, block, 0, stream>>>(a, b, c, M, K, N);
}

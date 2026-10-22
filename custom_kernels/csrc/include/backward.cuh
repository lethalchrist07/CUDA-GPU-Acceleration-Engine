#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Vector add backward: dA = dC (if need_a), dB = dC (if need_b)
void launch_vector_add_backward(
    const float* grad_out,
    float* grad_a,
    float* grad_b,
    size_t n,
    bool need_a,
    bool need_b,
    cudaStream_t stream
);

void launch_vector_add_backward_fp16(
    const half* grad_out,
    half* grad_a,
    half* grad_b,
    size_t n,
    bool need_a,
    bool need_b,
    cudaStream_t stream
);

void launch_vector_add_backward_bf16(
    const __nv_bfloat16* grad_out,
    __nv_bfloat16* grad_a,
    __nv_bfloat16* grad_b,
    size_t n,
    bool need_a,
    bool need_b,
    cudaStream_t stream
);

// Vector scale backward: dX = alpha * grad_out
void launch_vector_scale_backward(
    const float* grad_out,
    float* grad_x,
    float alpha,
    size_t n,
    cudaStream_t stream
);

void launch_vector_scale_backward_fp16(
    const half* grad_out,
    half* grad_x,
    float alpha,
    size_t n,
    cudaStream_t stream
);

void launch_vector_scale_backward_bf16(
    const __nv_bfloat16* grad_out,
    __nv_bfloat16* grad_x,
    float alpha,
    size_t n,
    cudaStream_t stream
);

// ReLU backward: dX = (x > 0) ? grad_out : 0
void launch_relu_backward(
    const float* grad_out,
    const float* x,
    float* grad_x,
    size_t n,
    cudaStream_t stream
);

void launch_relu_backward_fp16(
    const half* grad_out,
    const half* x,
    half* grad_x,
    size_t n,
    cudaStream_t stream
);

void launch_relu_backward_bf16(
    const __nv_bfloat16* grad_out,
    const __nv_bfloat16* x,
    __nv_bfloat16* grad_x,
    size_t n,
    cudaStream_t stream
);

// Sigmoid backward: dX = grad_out * y * (1 - y)
void launch_sigmoid_backward(
    const float* grad_out,
    const float* forward_out,
    float* grad_x,
    size_t n,
    cudaStream_t stream
);

void launch_sigmoid_backward_fp16(
    const half* grad_out,
    const half* forward_out,
    half* grad_x,
    size_t n,
    cudaStream_t stream
);

void launch_sigmoid_backward_bf16(
    const __nv_bfloat16* grad_out,
    const __nv_bfloat16* forward_out,
    __nv_bfloat16* grad_x,
    size_t n,
    cudaStream_t stream
);

// GELU backward: dX = grad_out * gelu_tanh_derivative(x)
void launch_gelu_backward(
    const float* grad_out,
    const float* x,
    float* grad_x,
    size_t n,
    cudaStream_t stream
);

void launch_gelu_backward_fp16(
    const half* grad_out,
    const half* x,
    half* grad_x,
    size_t n,
    cudaStream_t stream
);

void launch_gelu_backward_bf16(
    const __nv_bfloat16* grad_out,
    const __nv_bfloat16* x,
    __nv_bfloat16* grad_x,
    size_t n,
    cudaStream_t stream
);

// Fused add + ReLU backward: dA, dB = (forward_out > 0) ? grad_out : 0
void launch_fused_add_relu_backward(
    const float* grad_out,
    const float* forward_out,
    float* grad_a,
    float* grad_b,
    size_t n,
    bool need_a,
    bool need_b,
    cudaStream_t stream
);

void launch_fused_add_relu_backward_fp16(
    const half* grad_out,
    const half* forward_out,
    half* grad_a,
    half* grad_b,
    size_t n,
    bool need_a,
    bool need_b,
    cudaStream_t stream
);

void launch_fused_add_relu_backward_bf16(
    const __nv_bfloat16* grad_out,
    const __nv_bfloat16* forward_out,
    __nv_bfloat16* grad_a,
    __nv_bfloat16* grad_b,
    size_t n,
    bool need_a,
    bool need_b,
    cudaStream_t stream
);

// Fused bias + activation backward:
// 1. grad_x = grad_out * act'(x + bias)
// 2. grad_bias = sum_m grad_x[m, :]
void launch_fused_bias_activation_backward(
    const float* grad_out,
    const float* x,
    const float* bias,
    float* grad_x,
    float* grad_bias,
    int M,
    int N,
    int act_type, // 0: None, 1: ReLU, 2: GELU
    bool need_x,
    bool need_bias,
    cudaStream_t stream
);

void launch_fused_bias_activation_backward_fp16(
    const half* grad_out,
    const half* x,
    const half* bias,
    half* grad_x,
    half* grad_bias,
    int M,
    int N,
    int act_type, // 0: None, 1: ReLU, 2: GELU
    bool need_x,
    bool need_bias,
    cudaStream_t stream
);

void launch_fused_bias_activation_backward_bf16(
    const __nv_bfloat16* grad_out,
    const __nv_bfloat16* x,
    const __nv_bfloat16* bias,
    __nv_bfloat16* grad_x,
    __nv_bfloat16* grad_bias,
    int M,
    int N,
    int act_type, // 0: None, 1: ReLU, 2: GELU
    bool need_x,
    bool need_bias,
    cudaStream_t stream
);

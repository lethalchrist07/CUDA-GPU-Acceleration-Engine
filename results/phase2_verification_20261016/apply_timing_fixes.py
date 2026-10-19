from pathlib import Path
import re

root = Path(__file__).resolve().parents[2]
for name in ('vector_add.cu', 'matmul.cu'):
    path = root / 'src' / name
    text = path.read_text()
    text = text.replace('std::chrono::high_resolution_clock', 'std::chrono::steady_clock')
    # Check warmup, launch and cleanup errors as well as timed CUDA calls.
    text = re.sub(r'^(\s*)(cuda(?:Memcpy|DeviceSynchronize|EventDestroy|Free)\([^\n]+\));$',
                  r'\1CUDA_CHECK(\2);', text, flags=re.M)
    text = re.sub(r'^(\s*)(\w+<<<[^\n]+;)$',
                  r'\1\2\n\1CUDA_CHECK(cudaGetLastError());', text, flags=re.M)
    if name == 'vector_add.cu':
        text = text.replace('    const size_t bytes =', '    if (n == 0 || warmup_runs < 0 || timed_runs <= 0) return false;\n    const size_t bytes =', 1)
        text = text.replace('        // H2D\n', '        CUDA_CHECK(cudaDeviceSynchronize());\n        const auto wall_start = std::chrono::steady_clock::now();\n        // H2D\n')
        text = text.replace('        CUDA_CHECK(cudaEventSynchronize(evD2HStop));', '        CUDA_CHECK(cudaEventSynchronize(evD2HStop));\n        const auto wall_stop = std::chrono::steady_clock::now();')
        text = text.replace('e2e_times.push_back(h2d_ms + k_ms + d2h_ms);', 'e2e_times.push_back(std::chrono::duration<double, std::milli>(wall_stop - wall_start).count());')
        start = text.index('    float max_err =')
        end = text.index('    // Cleanup', start)
        text = text[:start] + '''    verifyAgainstCPU(h_C_gpu.data(), h_C_cpu.data(), n, outResult);
    outResult.cpu_timed_runs = timed_runs;
    outResult.cpu_samples_ms = cpu_times;
    outResult.h2d_samples_ms = h2d_times;
    outResult.kernel_samples_ms = kernel_times;
    outResult.d2h_samples_ms = d2h_times;
    outResult.e2e_samples_ms = e2e_times;

''' + text[end:]
    else:
        text = text.replace('    const size_t elements =', '    if (n <= 0 || warmup_runs < 0 || timed_runs <= 0) return false;\n    const size_t elements =', 1)
        text = text.replace('    for (int r = 0; r < timed_runs; ++r) {\n        CUDA_CHECK(cudaEventRecord(evH2DStart));', '    for (int r = 0; r < timed_runs; ++r) {\n        CUDA_CHECK(cudaDeviceSynchronize());\n        const auto wall_start = std::chrono::steady_clock::now();\n        CUDA_CHECK(cudaEventRecord(evH2DStart));')
        text = text.replace('        CUDA_CHECK(cudaEventSynchronize(evD2HStop));', '        CUDA_CHECK(cudaEventSynchronize(evD2HStop));\n        const auto wall_stop = std::chrono::steady_clock::now();')
        text = text.replace('push_back(h2d_ms + k_ms + d2h_ms);', 'push_back(std::chrono::duration<double, std::milli>(wall_stop - wall_start).count());')
        for variant, result, end_marker in [('basic', 'outBasicResult', '    // ==========================================\n    // 6.'), ('tiled', 'outTiledResult', '    // Cleanup')]:
            start = text.index(f'    float max_err_{variant} =')
            end = text.index(end_marker, start)
            text = text[:start] + f'''    verifyAgainstCPU(h_C_{variant}.data(), h_C_cpu.data(), elements, {result});
    {result}.cpu_timed_runs = cpu_runs;
    {result}.cpu_samples_ms = cpu_times;
    {result}.h2d_samples_ms = {variant}_h2d;
    {result}.kernel_samples_ms = {variant}_kernel;
    {result}.d2h_samples_ms = {variant}_d2h;
    {result}.e2e_samples_ms = {variant}_e2e;

''' + text[end:]
        text = text.replace('(2.0 * static_cast<double>(n) / MATMUL_TILE_DIM + 1.0)', '(2.0 * ((n + MATMUL_TILE_DIM - 1) / MATMUL_TILE_DIM) + 1.0)')
    path.write_text(text)

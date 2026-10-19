"""Independent arithmetic audit of preserved and newly executed result artifacts."""
import csv
import json
import math
import statistics
from pathlib import Path

AUDIT = Path(__file__).resolve().parent
FIELDS = {
    'Category': 'category', 'Variant': 'variant', 'WorkloadSize': 'workload_size',
    'Dimension': 'dimension', 'MemoryMB': 'memory_mb', 'CPUTimeMs': 'cpu_time_ms',
    'GPUH2DMs': 'gpu_h2d_ms', 'GPUKernelMs': 'gpu_kernel_ms', 'GPUD2HMs': 'gpu_d2h_ms',
    'GPUEndToEndMs': 'gpu_end_to_end_ms', 'KernelSpeedup': 'kernel_speedup',
    'EndToEndSpeedup': 'end_to_end_speedup', 'GFLOPS': 'gflops',
    'MemoryBandwidthGBs': 'memory_bandwidth_gb_s', 'MaxAbsError': 'max_abs_error',
    'RMSE': 'rmse', 'Tolerance': 'tolerance', 'MismatchedElements': 'mismatched_elements',
    'Verified': 'verified', 'WarmupRuns': 'warmup_runs', 'TimedRuns': 'timed_runs',
    'CPUTimedRuns': 'cpu_timed_runs',
}

def read_json(path):
    return json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))

def compare_rows(csv_rows, json_rows, mapping):
    assert len(csv_rows) == len(json_rows)
    for csv_row, json_row in zip(csv_rows, json_rows):
        for csv_key, text in csv_row.items():
            value = json_row[mapping.get(csv_key, csv_key)]
            if isinstance(value, bool):
                assert text.lower() in (('true', 'pass') if value else ('false', 'fail'))
            elif isinstance(value, (int, float)):
                assert float(text) == value, (csv_key, text, value)
            else:
                assert text == value, (csv_key, text, value)

def benchmark_audit(directory, legacy):
    document = read_json(directory / 'benchmark_results.json')
    rows = document['benchmarks']
    with (directory / 'benchmark_results.csv').open(newline='') as file:
        compare_rows(list(csv.DictReader(file)), rows, FIELDS)
    assert len(rows) == 12
    expected = [('Vector Addition', n, 'Standard CUDA Kernel') for n in (100000, 1000000, 10000000, 25000000)]
    expected += [('Matrix Multiplication', n, variant) for n in (256, 512, 1024, 2048)
                 for variant in ('Basic Naive CUDA', 'Tiled Shared Memory CUDA')]
    assert [(r['category'], r['dimension'], r['variant']) for r in rows] == expected
    assert document['summary']['passed'] == 12 and document['summary']['failed'] == 0
    assert document['summary']['status'] == 'PASS'
    timer_rounding = 0.000050001 if legacy else 0.000000000501
    metric_rounding = 0.005001 if legacy else 0.000000000501
    for row in rows:
        assert row['verified'] and row['mismatched_elements'] == 0
        assert 0 <= row['rmse'] <= row['max_abs_error'] <= row['tolerance']
        n = row['dimension']
        matrix = row['category'] == 'Matrix Multiplication'
        elements = n*n if matrix else n
        assert abs(row['memory_mb'] - 12*elements/1024**2) <= (0.005001 if legacy else 5.01e-10)
        for key in ('cpu_time_ms', 'gpu_kernel_ms', 'gpu_end_to_end_ms', 'gpu_h2d_ms', 'gpu_d2h_ms'):
            assert math.isfinite(row[key]) and row[key] > 0
        # Check ratios against intervals allowing each exported input's rounding.
        for metric, denominator in [('kernel_speedup', 'gpu_kernel_ms'), ('end_to_end_speedup', 'gpu_end_to_end_ms')]:
            cpu, gpu = row['cpu_time_ms'], row[denominator]
            low = (cpu-timer_rounding)/(gpu+timer_rounding)-metric_rounding
            high = (cpu+timer_rounding)/(gpu-timer_rounding)+metric_rounding
            assert low <= row[metric] <= high, (directory, n, metric)
        operations = 2*n**3 if matrix else n
        requested_bytes = 12*n
        if matrix:
            loads = 2*n+1 if row['variant'] == 'Basic Naive CUDA' else 2*math.ceil(n/16)+1
            requested_bytes = loads*n*n*4
        for metric, amount in [('gflops', operations), ('memory_bandwidth_gb_s', requested_bytes)]:
            low = amount/((row['gpu_kernel_ms']+timer_rounding)*1e6)-metric_rounding
            high = amount/((row['gpu_kernel_ms']-timer_rounding)*1e6)+metric_rounding
            assert low <= row[metric] <= high, (directory, n, metric)
        if not legacy:
            assert document['schema_version'] == 2
            assert document['summary']['missing_benchmarks'] == 0
            assert document['system_info']['cuda_runtime_version'] == '13.4'
            for sample_name, field in [('cpu', 'cpu_time_ms'), ('h2d', 'gpu_h2d_ms'), ('kernel', 'gpu_kernel_ms'),
                                       ('d2h', 'gpu_d2h_ms'), ('end_to_end', 'gpu_end_to_end_ms')]:
                samples = row['samples_ms'][sample_name]
                expected_count = row['cpu_timed_runs'] if sample_name == 'cpu' else row['timed_runs']
                assert len(samples) == expected_count
                assert all(math.isfinite(x) and x > 0 for x in samples)
                assert abs(statistics.median(samples) - row[field]) < 1.01e-9
    print(f'PASS: {directory.relative_to(AUDIT)}: 12 rows, CSV/JSON parity, correctness fields, units and formulas'
          + ('' if legacy else ', raw-sample medians and sample counts'))
    return rows

benchmark_audit(AUDIT / 'prior_results', True)
benchmark_audit(AUDIT / 'original_run' / 'results', True)
benchmark_audit(AUDIT / 'verified_run', False)
fresh = benchmark_audit(AUDIT / 'final_run', False)
for basename, key in [('training_experiments', 'experiments'), ('inference_experiments', 'inferences')]:
    rows = read_json(AUDIT / 'prior_results' / f'{basename}.json')[key]
    with (AUDIT / 'prior_results' / f'{basename}.csv').open(newline='') as file:
        compare_rows(list(csv.DictReader(file)), rows, {})
    for row in rows:
        if key == 'experiments':
            assert abs(row['total_train_time_ms']/row['epochs'] - row['avg_epoch_time_ms']) < 0.006
            assert abs(rows[0]['total_train_time_ms']/row['total_train_time_ms'] - row['speedup_vs_cpu']) < 0.006
        else:
            assert abs(row['output_tokens']*1000/row['generation_duration_ms'] - row['tokens_per_second']) < 0.006
    print(f'PASS: {basename}: {len(rows)} rows, CSV/JSON parity and derived arithmetic; historical executions NOT authenticated')

print('\nMeasured Release run (ms; speedups relative to single-threaded CPU reference):')
print('workload,variant,cpu_ms,kernel_ms,e2e_ms,kernel_speedup,e2e_speedup,max_abs_error')
for row in fresh:
    print(','.join(str(row[k]) for k in ('workload_size', 'variant', 'cpu_time_ms', 'gpu_kernel_ms',
                                       'gpu_end_to_end_ms', 'kernel_speedup', 'end_to_end_speedup', 'max_abs_error')))

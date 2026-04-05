........................................................................ [  7%]
........................................................................ [ 14%]
........................................................................ [ 21%]
........................................................................ [ 28%]
........................................................................ [ 35%]
........................................................................ [ 43%]
.......................................xxx.............................. [ 50%]
........................................................................ [ 57%]
................sss..................................................... [ 64%]
........................................................................ [ 71%]
........................................................................ [ 78%]
........................................................................ [ 86%]
........................................................................ [ 93%]
....................................................................     [100%]Benchmark report saved to profile_run.log

=============================== warnings summary ===============================
../usr/local/lib/python3.11/dist-packages/torch/jit/_script.py:362: 14 warnings
  /usr/local/lib/python3.11/dist-packages/torch/jit/_script.py:362: DeprecationWarning: `torch.jit.script_method` is deprecated. Please switch to `torch.compile` or `torch.export`.
    warnings.warn(

<frozen importlib._bootstrap>:241
  <frozen importlib._bootstrap>:241: DeprecationWarning: builtin type SwigPyPacked has no __module__ attribute

<frozen importlib._bootstrap>:241
  <frozen importlib._bootstrap>:241: DeprecationWarning: builtin type SwigPyObject has no __module__ attribute

benchmarks/ops/bench_activation.py::test_r2_small_tensor_unary[4096-dtype0]
  /usr/local/lib/python3.11/dist-packages/torch/profiler/profiler.py:217: UserWarning: Warning: Profiler clears events at the end of each cycle.Only events from the current cycle will be reported.To keep events across cycles, set acc_events=True.
    _warn_once(

benchmarks/ops/bench_grouped_gemm_block_m.py: 20 warnings
  /usr/local/lib/python3.11/dist-packages/tilelang/profiler/bench.py:182: UserWarning: Profiler won't be using warmup, this can skew profiler results
    schedule = torch.profiler.schedule(wait=1, warmup=0, active=1, repeat=1)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
998 passed, 3 skipped, 3 xfailed, 37 warnings in 1597.60s (0:26:37)

===== profile_run.log summary =====
# TileOPs Benchmark Report
Generated: 2026-04-02 19:35:23

## Environment

- **Torch version**: 2.10.0+cu128
- **CUDA version (torch)**: 12.8
- **GPU model**: NVIDIA H200
- **Driver version**: 575.57.08

## ReluOp

### tileops

| n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 4096 | torch.float16 | 0.00 | 0.00 | 0.01 |
| 4194304 | torch.float16 | 0.01 | 0.67 | 2.67 |
| 4194304 | torch.bfloat16 | 0.01 | 0.67 | 2.67 |
| 4194304 | torch.float32 | 0.01 | 0.39 | 3.13 |

### torch

| n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 4096 | torch.float16 | 0.00 | 0.00 | 0.01 |
| 4194304 | torch.float16 | 0.01 | 0.51 | 2.04 |
| 4194304 | torch.bfloat16 | 0.01 | 0.51 | 2.05 |
| 4194304 | torch.float32 | 0.01 | 0.29 | 2.28 |

## r3_jit_cost

### relu_jit

| n_total | cold_ms | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 1000 | 19.59 | 0.00 | 0.00 | 0.00 |
| 2000 | 18.08 | 0.00 | 0.00 | 0.00 |
| 4000 | 17.89 | 0.00 | 0.00 | 0.01 |
| 8000 | 17.96 | 0.00 | 0.00 | 0.02 |
| 16000 | 17.81 | 0.00 | 0.01 | 0.04 |
| 32000 | 17.82 | 0.00 | 0.02 | 0.07 |
| 64000 | 18.0 | 0.00 | 0.03 | 0.14 |
| 128000 | 18.05 | 0.00 | 0.07 | 0.26 |
| 256000 | 18.86 | 0.00 | 0.12 | 0.49 |
| 512000 | 17.86 | 0.00 | 0.22 | 0.86 |

## r4_strategy_unary

### relu_direct

| n_total | size_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | torch.float16 | direct | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.bfloat16 | direct | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.float32 | direct | 0.00 | 0.00 | 0.02 |
| 4194304 | 4M | torch.float16 | direct | 0.01 | 0.29 | 1.17 |
| 4194304 | 4M | torch.bfloat16 | direct | 0.01 | 0.29 | 1.17 |
| 4194304 | 4M | torch.float32 | direct | 0.02 | 0.27 | 2.13 |
| 16777216 | 16M | torch.float16 | direct | 0.07 | 0.23 | 0.92 |
| 16777216 | 16M | torch.bfloat16 | direct | 0.07 | 0.23 | 0.92 |
| 16777216 | 16M | torch.float32 | direct | 0.08 | 0.20 | 1.63 |

### relu_explicit_parallel

| n_total | size_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | torch.float16 | explicit_parallel | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.bfloat16 | explicit_parallel | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.float32 | explicit_parallel | 0.00 | 0.00 | 0.02 |
| 4194304 | 4M | torch.float16 | explicit_parallel | 0.01 | 0.62 | 2.47 |
| 4194304 | 4M | torch.bfloat16 | explicit_parallel | 0.01 | 0.62 | 2.46 |
| 4194304 | 4M | torch.float32 | explicit_parallel | 0.01 | 0.38 | 3.06 |
| 16777216 | 16M | torch.float16 | explicit_parallel | 0.02 | 0.80 | 3.21 |
| 16777216 | 16M | torch.bfloat16 | explicit_parallel | 0.02 | 0.80 | 3.21 |
| 16777216 | 16M | torch.float32 | explicit_parallel | 0.04 | 0.39 | 3.14 |

### relu_register_copy

| n_total | size_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | torch.float16 | register_copy | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.bfloat16 | register_copy | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.float32 | register_copy | 0.00 | 0.00 | 0.02 |
| 4194304 | 4M | torch.float16 | register_copy | 0.01 | 0.67 | 2.67 |
| 4194304 | 4M | torch.bfloat16 | register_copy | 0.01 | 0.67 | 2.66 |
| 4194304 | 4M | torch.float32 | register_copy | 0.01 | 0.39 | 3.14 |
| 16777216 | 16M | torch.float16 | register_copy | 0.02 | 0.90 | 3.60 |
| 16777216 | 16M | torch.bfloat16 | register_copy | 0.02 | 0.90 | 3.61 |
| 16777216 | 16M | torch.float32 | register_copy | 0.04 | 0.40 | 3.20 |

## r4_strategy_gelu

### gelu_direct

| n_total | size_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | torch.float16 | direct | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.bfloat16 | direct | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.float32 | direct | 0.00 | 0.00 | 0.02 |
| 4194304 | 4M | torch.float16 | direct | 0.02 | 0.27 | 1.07 |
| 4194304 | 4M | torch.bfloat16 | direct | 0.02 | 0.27 | 1.07 |
| 4194304 | 4M | torch.float32 | direct | 0.02 | 0.25 | 1.99 |
| 16777216 | 16M | torch.float16 | direct | 0.08 | 0.20 | 0.81 |
| 16777216 | 16M | torch.bfloat16 | direct | 0.08 | 0.20 | 0.81 |
| 16777216 | 16M | torch.float32 | direct | 0.09 | 0.19 | 1.50 |

### gelu_explicit_parallel

| n_total | size_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | torch.float16 | explicit_parallel | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.bfloat16 | explicit_parallel | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.float32 | explicit_parallel | 0.00 | 0.00 | 0.02 |
| 4194304 | 4M | torch.float16 | explicit_parallel | 0.01 | 0.45 | 1.79 |
| 4194304 | 4M | torch.bfloat16 | explicit_parallel | 0.01 | 0.44 | 1.77 |
| 4194304 | 4M | torch.float32 | explicit_parallel | 0.01 | 0.38 | 3.07 |
| 16777216 | 16M | torch.float16 | explicit_parallel | 0.04 | 0.46 | 1.83 |
| 16777216 | 16M | torch.bfloat16 | explicit_parallel | 0.04 | 0.44 | 1.78 |
| 16777216 | 16M | torch.float32 | explicit_parallel | 0.05 | 0.37 | 2.97 |

### gelu_register_copy

| n_total | size_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | torch.float16 | register_copy | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.bfloat16 | register_copy | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.float32 | register_copy | 0.00 | 0.00 | 0.02 |
| 4194304 | 4M | torch.float16 | register_copy | 0.01 | 0.48 | 1.93 |
| 4194304 | 4M | torch.bfloat16 | register_copy | 0.01 | 0.47 | 1.89 |
| 4194304 | 4M | torch.float32 | register_copy | 0.01 | 0.39 | 3.12 |
| 16777216 | 16M | torch.float16 | register_copy | 0.03 | 0.55 | 2.20 |
| 16777216 | 16M | torch.bfloat16 | register_copy | 0.03 | 0.52 | 2.09 |
| 16777216 | 16M | torch.float32 | register_copy | 0.04 | 0.39 | 3.10 |

## r5_boundary

### relu_aligned

| n_total | align_label | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 2048000 | aligned | 0.00 | 0.46 | 1.84 |

### relu_unaligned_plus_1

| n_total | align_label | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 2048001 | unaligned_plus_1 | 0.01 | 0.30 | 1.19 |

### relu_unaligned_plus_127

| n_total | align_label | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 2048127 | unaligned_plus_127 | 0.01 | 0.30 | 1.19 |

## r6_threads

### relu_t128

| n_total | size_label | op_name | threads | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | relu | 128 | 0.00 | 0.00 | 0.01 |
| 4194304 | 4M | relu | 128 | 0.03 | 0.14 | 0.58 |
| 16777216 | 16M | relu | 128 | 0.11 | 0.15 | 0.59 |

### relu_t256

| n_total | size_label | op_name | threads | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | relu | 256 | 0.00 | 0.00 | 0.01 |
| 4194304 | 4M | relu | 256 | 0.03 | 0.15 | 0.59 |
| 16777216 | 16M | relu | 256 | 0.12 | 0.14 | 0.56 |

### erf_t128

| n_total | size_label | op_name | threads | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | erf | 128 | 0.00 | 0.00 | 0.01 |
| 4194304 | 4M | erf | 128 | 0.01 | 0.40 | 1.60 |
| 16777216 | 16M | erf | 128 | 0.04 | 0.39 | 1.54 |

### erf_t256

| n_total | size_label | op_name | threads | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | erf | 256 | 0.00 | 0.00 | 0.01 |
| 4194304 | 4M | erf | 256 | 0.01 | 0.39 | 1.57 |
| 16777216 | 16M | erf | 256 | 0.04 | 0.38 | 1.54 |

### mish_t128

| n_total | size_label | op_name | threads | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | mish | 128 | 0.00 | 0.00 | 0.01 |
| 4194304 | 4M | mish | 128 | 0.02 | 0.28 | 1.10 |
| 16777216 | 16M | mish | 128 | 0.07 | 0.26 | 1.03 |

### mish_t256

| n_total | size_label | op_name | threads | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | mish | 256 | 0.00 | 0.00 | 0.01 |
| 4194304 | 4M | mish | 256 | 0.02 | 0.27 | 1.09 |
| 16777216 | 16M | mish | 256 | 0.07 | 0.26 | 1.02 |

## r7_dtype_npt

### relu_fp32_npt4

| dtype_label | num_per_thread | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| fp32 | 4 | 0.00 | 0.21 | 1.66 |

### relu_fp32_npt8

| dtype_label | num_per_thread | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| fp32 | 8 | 0.00 | 0.21 | 1.66 |

### relu_fp16_npt4

| dtype_label | num_per_thread | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| fp16 | 4 | 0.00 | 0.27 | 1.09 |

### relu_fp16_npt8

| dtype_label | num_per_thread | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| fp16 | 8 | 0.00 | 0.21 | 0.82 |

## AdaLayerNormOp

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 1152 | torch.float16 | 0.93 | 0.01 | 0.01 |
| 1024 | 1152 | torch.bfloat16 | 1.11 | 0.01 | 0.01 |
| 2048 | 4096 | torch.float16 | 0.03 | 1.49 | 2.39 |
| 2048 | 4096 | torch.bfloat16 | 0.03 | 1.48 | 2.36 |
| 1 | 4096 | torch.bfloat16 | 0.01 | 0.00 | 0.01 |

### torch-ref

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 1152 | torch.float16 | 0.01 | 0.49 | 0.78 |
| 1024 | 1152 | torch.bfloat16 | 0.01 | 0.49 | 0.78 |
| 2048 | 4096 | torch.float16 | 0.05 | 0.81 | 1.30 |
| 2048 | 4096 | torch.bfloat16 | 0.05 | 0.80 | 1.29 |
| 1 | 4096 | torch.bfloat16 | 0.01 | 0.00 | 0.00 |

## AdaLayerNormZeroOp

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 1152 | torch.float16 | 1.12 | 0.01 | 0.01 |
| 1024 | 1152 | torch.bfloat16 | 1.32 | 0.01 | 0.01 |
| 2048 | 4096 | torch.float16 | 0.03 | 1.46 | 2.43 |
| 2048 | 4096 | torch.bfloat16 | 0.03 | 1.44 | 2.40 |
| 1 | 4096 | torch.bfloat16 | 0.01 | 0.00 | 0.01 |

### torch-ref

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 1152 | torch.float16 | 0.02 | 0.47 | 0.78 |
| 1024 | 1152 | torch.bfloat16 | 0.02 | 0.47 | 0.78 |
| 2048 | 4096 | torch.float16 | 0.07 | 0.75 | 1.24 |
| 2048 | 4096 | torch.bfloat16 | 0.07 | 0.74 | 1.24 |
| 1 | 4096 | torch.bfloat16 | 0.01 | 0.00 | 0.00 |

## ArgmaxOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | argmax | 6.09 | 0.00 | 0.00 |
| 1024 | 4096 | torch.bfloat16 | argmax | 6.10 | 0.00 | 0.00 |
| 4096 | 4096 | torch.float16 | argmax | 21.17 | 0.00 | 0.00 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | argmax | 0.03 | 0.13 | 0.25 |
| 1024 | 4096 | torch.bfloat16 | argmax | 0.03 | 0.12 | 0.25 |
| 4096 | 4096 | torch.float16 | argmax | 0.04 | 0.38 | 0.76 |

## ArgminOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | argmin | 6.88 | 0.00 | 0.00 |
| 1024 | 4096 | torch.bfloat16 | argmin | 6.90 | 0.00 | 0.00 |
| 4096 | 4096 | torch.float16 | argmin | 23.99 | 0.00 | 0.00 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | argmin | 0.03 | 0.13 | 0.25 |
| 1024 | 4096 | torch.bfloat16 | argmin | 0.03 | 0.13 | 0.25 |
| 4096 | 4096 | torch.float16 | argmin | 0.04 | 0.39 | 0.77 |

## avg_pool1d

### tileops

| n | c_in | l_in | kernel_size | stride | padding | ceil_mode | count_include_pad | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 128 | 4096 | 3 | 2 | 1 | False | True | torch.float16 | 0.02 | 0.18 | 0.37 |
| 2 | 256 | 32000 | 5 | 4 | 2 | False | True | torch.float16 | 0.03 | 0.80 | 1.59 |
| 2 | 128 | 2048 | 4 | 2 | 1 | True | False | torch.bfloat16 | 0.01 | 0.11 | 0.17 |

### torch-ref

| n | c_in | l_in | kernel_size | stride | padding | ceil_mode | count_include_pad | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 128 | 4096 | 3 | 2 | 1 | False | True | torch.float16 | 0.02 | 0.18 | 0.35 |
| 2 | 256 | 32000 | 5 | 4 | 2 | False | True | torch.float16 | 0.08 | 0.25 | 0.49 |
| 2 | 128 | 2048 | 4 | 2 | 1 | True | False | torch.bfloat16 | 0.01 | 0.18 | 0.28 |

## avg_pool2d

### tileops

| n | c_in | h_in | w_in | ceil_mode | count_include_pad | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 64 | 112 | 112 | False | True | torch.float16 | 0.03 | 0.13 | 0.15 |
| 2 | 128 | 56 | 56 | False | True | torch.float16 | 0.05 | 0.10 | 0.04 |
| 3 | 96 | 55 | 57 | True | False | torch.bfloat16 | 0.02 | 0.20 | 0.13 |

### torch-ref

| n | c_in | h_in | w_in | ceil_mode | count_include_pad | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 64 | 112 | 112 | False | True | torch.float16 | 0.01 | 0.34 | 0.37 |
| 2 | 128 | 56 | 56 | False | True | torch.float16 | 0.01 | 0.44 | 0.18 |
| 3 | 96 | 55 | 57 | True | False | torch.bfloat16 | 0.01 | 0.41 | 0.26 |

## avg_pool3d

### tileops

| n | c_in | d_in | h_in | w_in | ceil_mode | count_include_pad | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 16 | 56 | 56 | False | True | torch.float16 | 0.00 | 0.43 | 0.96 |
| 2 | 64 | 8 | 28 | 28 | True | False | torch.float16 | 0.01 | 0.18 | 0.13 |
| 2 | 24 | 10 | 20 | 22 | False | True | torch.bfloat16 | 0.00 | 0.08 | 0.11 |

### torch-ref

| n | c_in | d_in | h_in | w_in | ceil_mode | count_include_pad | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 16 | 56 | 56 | False | True | torch.float16 | 0.01 | 0.14 | 0.31 |
| 2 | 64 | 8 | 28 | 28 | True | False | torch.float16 | 0.01 | 0.20 | 0.14 |
| 2 | 24 | 10 | 20 | 22 | False | True | torch.bfloat16 | 0.01 | 0.06 | 0.09 |

## BatchNormFwdOp

### tileops

| N | C | spatial | dtype | training | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 64 | () | torch.float16 | True | 0.01 | 0.00 | 0.00 |
| 8 | 64 | (32, 32) | torch.float16 | True | 0.01 | 0.44 | 0.18 |
| 4 | 128 | (32, 32) | torch.float16 | True | 0.01 | 0.45 | 0.18 |
| 4 | 256 | (28, 28) | torch.float16 | True | 0.02 | 0.52 | 0.21 |
| 4 | 128 | (1024, 1024) | torch.float16 | True | 7.77 | 0.69 | 0.28 |

### torch-cudnn

| N | C | spatial | dtype | training | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 64 | () | torch.float16 | True | 0.02 | 0.00 | 0.00 |
| 8 | 64 | (32, 32) | torch.float16 | True | 0.01 | 0.38 | 0.15 |
| 4 | 128 | (32, 32) | torch.float16 | True | 0.01 | 0.40 | 0.16 |
| 4 | 256 | (28, 28) | torch.float16 | True | 0.01 | 0.58 | 0.23 |
| 4 | 128 | (1024, 1024) | torch.float16 | True | 8.18 | 0.66 | 0.26 |

## BatchNormBwdOp

### tileops

| N | C | spatial | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 64 | () | torch.float16 | 0.01 | 0.00 | 0.00 |
| 8 | 64 | (32, 32) | torch.float16 | 0.02 | 0.26 | 0.20 |
| 4 | 128 | (32, 32) | torch.float16 | 0.02 | 0.27 | 0.20 |
| 4 | 256 | (28, 28) | torch.float16 | 0.02 | 0.32 | 0.24 |
| 4 | 128 | (1024, 1024) | torch.float16 | 12.34 | 0.35 | 0.26 |

### torch-autograd

| N | C | spatial | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 64 | () | torch.float16 | 0.02 | 0.00 | 0.00 |
| 8 | 64 | (32, 32) | torch.float16 | 0.03 | 0.16 | 0.12 |
| 4 | 128 | (32, 32) | torch.float16 | 0.02 | 0.19 | 0.14 |
| 4 | 256 | (28, 28) | torch.float16 | 0.02 | 0.27 | 0.21 |
| 4 | 128 | (1024, 1024) | torch.float16 | 23.94 | 0.18 | 0.13 |

## r1_vectorization

### add_same_shape_1d

| pattern_name | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| same_shape_1d | 1000000 | 0.00 | 0.25 | 1.51 |

### torch-same_shape_1d

| pattern_name | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| same_shape_1d | 1000000 | 0.00 | 0.21 | 1.23 |

### add_bias_add_2d

| pattern_name | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| bias_add_2d | 1000000 | 0.00 | 0.29 | 1.14 |

### torch-bias_add_2d

| pattern_name | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| bias_add_2d | 1000000 | 0.01 | 0.14 | 0.54 |

### add_interleaved_3d

| pattern_name | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| interleaved_3d | 1048576 | 0.00 | 0.45 | 0.91 |

### torch-interleaved_3d

| pattern_name | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| interleaved_3d | 1048576 | 0.01 | 0.16 | 0.31 |

## r2_small_tensor_binary

### add_same_shape

| pattern_name | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| same_shape | 4096 | 0.00 | 0.00 | 0.01 |

### torch-same_shape

| pattern_name | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| same_shape | 4096 | 0.00 | 0.00 | 0.01 |

### add_broadcast_3d

| pattern_name | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| broadcast_3d | 4096 | 0.00 | 0.00 | 0.01 |

### torch-broadcast_3d

| pattern_name | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| broadcast_3d | 4096 | 0.00 | 0.00 | 0.00 |

## r4_strategy_binary

### add_direct_same_shape

| size_label | pattern_name | strategy | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4K | same_shape | direct | 4096 | 0.00 | 0.00 | 0.01 |
| 4K | same_shape | direct | 4096 | 0.00 | 0.00 | 0.01 |
| 4K | same_shape | direct | 4096 | 0.00 | 0.00 | 0.03 |
| 1M | same_shape | direct | 1048576 | 0.01 | 0.19 | 1.15 |
| 1M | same_shape | direct | 1048576 | 0.01 | 0.19 | 1.15 |
| 1M | same_shape | direct | 1048576 | 0.01 | 0.17 | 2.03 |
| 16M | same_shape | direct | 16777216 | 0.08 | 0.21 | 1.26 |
| 16M | same_shape | direct | 16777216 | 0.08 | 0.21 | 1.26 |
| 16M | same_shape | direct | 16777216 | 0.10 | 0.17 | 2.05 |

### add_direct_bias_add_2d

| size_label | pattern_name | strategy | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4K | bias_add_2d | direct | 4096 | 0.00 | 0.00 | 0.01 |
| 4K | bias_add_2d | direct | 4096 | 0.00 | 0.00 | 0.01 |
| 4K | bias_add_2d | direct | 4096 | 0.00 | 0.00 | 0.02 |
| 1M | bias_add_2d | direct | 1048576 | 0.01 | 0.21 | 0.82 |
| 1M | bias_add_2d | direct | 1048576 | 0.01 | 0.21 | 0.83 |
| 1M | bias_add_2d | direct | 1048576 | 0.01 | 0.19 | 1.55 |
| 16M | bias_add_2d | direct | 16777216 | 0.07 | 0.23 | 0.94 |
| 16M | bias_add_2d | direct | 16777216 | 0.07 | 0.23 | 0.94 |
| 16M | bias_add_2d | direct | 16777216 | 0.08 | 0.20 | 1.62 |

### add_direct_interleaved_3d

| size_label | pattern_name | strategy | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4K | interleaved_3d | direct | 4096 | 0.00 | 0.00 | 0.01 |
| 4K | interleaved_3d | direct | 4096 | 0.00 | 0.00 | 0.01 |
| 4K | interleaved_3d | direct | 4096 | 0.00 | 0.00 | 0.01 |
| 1M | interleaved_3d | direct | 1048576 | 0.00 | 0.24 | 0.49 |
| 1M | interleaved_3d | direct | 1048576 | 0.00 | 0.24 | 0.49 |
| 1M | interleaved_3d | direct | 1048576 | 0.00 | 0.23 | 0.94 |
| 16M | interleaved_3d | direct | 16777216 | 0.05 | 0.32 | 0.64 |
| 16M | interleaved_3d | direct | 16777216 | 0.05 | 0.32 | 0.64 |
| 16M | interleaved_3d | direct | 16777216 | 0.05 | 0.31 | 1.24 |

### add_explicit_parallel_same_shape

| size_label | pattern_name | strategy | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4K | same_shape | explicit_parallel | 4096 | 0.00 | 0.00 | 0.01 |
| 4K | same_shape | explicit_parallel | 4096 | 0.00 | 0.00 | 0.01 |
| 4K | same_shape | explicit_parallel | 4096 | 0.00 | 0.00 | 0.03 |
| 1M | same_shape | explicit_parallel | 1048576 | 0.00 | 0.27 | 1.60 |
| 1M | same_shape | explicit_parallel | 1048576 | 0.00 | 0.27 | 1.60 |
| 1M | same_shape | explicit_parallel | 1048576 | 0.01 | 0.19 | 2.30 |
| 16M | same_shape | explicit_parallel | 16777216 | 0.03 | 0.56 | 3.36 |
| 16M | same_shape | explicit_parallel | 16777216 | 0.03 | 0.57 | 3.43 |
| 16M | same_shape | explicit_parallel | 16777216 | 0.07 | 0.25 | 2.94 |

### add_explicit_parallel_bias_add_2d

| size_label | pattern_name | strategy | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4K | bias_add_2d | explicit_parallel | 4096 | 0.00 | 0.00 | 0.01 |
| 4K | bias_add_2d | explicit_parallel | 4096 | 0.00 | 0.00 | 0.01 |
| 4K | bias_add_2d | explicit_parallel | 4096 | 0.00 | 0.00 | 0.02 |
| 1M | bias_add_2d | explicit_parallel | 1048576 | 0.00 | 0.33 | 1.32 |
| 1M | bias_add_2d | explicit_parallel | 1048576 | 0.00 | 0.33 | 1.34 |
| 1M | bias_add_2d | explicit_parallel | 1048576 | 0.00 | 0.25 | 2.00 |
| 16M | bias_add_2d | explicit_parallel | 16777216 | 0.02 | 0.81 | 3.25 |
| 16M | bias_add_2d | explicit_parallel | 16777216 | 0.02 | 0.82 | 3.26 |
| 16M | bias_add_2d | explicit_parallel | 16777216 | 0.04 | 0.43 | 3.44 |

### add_explicit_parallel_interleaved_3d

| size_label | pattern_name | strategy | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4K | interleaved_3d | explicit_parallel | 4096 | 0.00 | 0.00 | 0.01 |
| 4K | interleaved_3d | explicit_parallel | 4096 | 0.00 | 0.00 | 0.01 |
| 4K | interleaved_3d | explicit_parallel | 4096 | 0.00 | 0.00 | 0.01 |
| 1M | interleaved_3d | explicit_parallel | 1048576 | 0.00 | 0.46 | 0.92 |
| 1M | interleaved_3d | explicit_parallel | 1048576 | 0.00 | 0.46 | 0.93 |
| 1M | interleaved_3d | explicit_parallel | 1048576 | 0.00 | 0.42 | 1.67 |
| 16M | interleaved_3d | explicit_parallel | 16777216 | 0.01 | 1.20 | 2.40 |
| 16M | interleaved_3d | explicit_parallel | 16777216 | 0.01 | 1.20 | 2.41 |
| 16M | interleaved_3d | explicit_parallel | 16777216 | 0.02 | 0.97 | 3.88 |

## r4_where

### tileops-where

| n_total | size_label | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 4096 | 4K | 0.00 | 0.00 | 0.02 |
| 1048576 | 1M | 0.00 | 0.25 | 1.72 |
| 16777216 | 16M | 0.03 | 0.48 | 3.36 |

### torch

| n_total | size_label | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 4096 | 4K | 0.00 | 0.00 | 0.01 |
| 1048576 | 1M | 0.01 | 0.21 | 1.45 |
| 16777216 | 16M | 0.05 | 0.33 | 2.32 |

## AddOp

### tileops

| n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 4194304 | torch.float16 | 0.01 | 0.48 | 2.87 |
| 4194304 | torch.bfloat16 | 0.01 | 0.48 | 2.87 |
| 4194304 | torch.float32 | 0.02 | 0.27 | 3.30 |

### torch

| n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 4194304 | torch.float16 | 0.01 | 0.35 | 2.07 |
| 4194304 | torch.bfloat16 | 0.01 | 0.34 | 2.07 |
| 4194304 | torch.float32 | 0.02 | 0.19 | 2.28 |

## sub

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| sub | torch.float16 | torch.float16 | 0.01 | 0.48 | 2.88 |
| sub | torch.float16 | torch.float16 | 0.02 | 0.58 | 3.48 |
| sub | torch.float16 | torch.float16 | 0.04 | 0.57 | 3.41 |

### torch

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| sub | torch.float16 | torch.float16 | 0.01 | 0.34 | 2.06 |
| sub | torch.float16 | torch.float16 | 0.03 | 0.38 | 2.25 |
| sub | torch.float16 | torch.float16 | 0.05 | 0.38 | 2.30 |

## mul

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| mul | torch.float16 | torch.float16 | 0.01 | 0.48 | 2.85 |
| mul | torch.float16 | torch.float16 | 0.02 | 0.58 | 3.45 |
| mul | torch.float16 | torch.float16 | 0.04 | 0.57 | 3.42 |

### torch

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| mul | torch.float16 | torch.float16 | 0.01 | 0.34 | 2.04 |
| mul | torch.float16 | torch.float16 | 0.03 | 0.37 | 2.24 |
| mul | torch.float16 | torch.float16 | 0.05 | 0.38 | 2.30 |

## div

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| div | torch.float16 | torch.float16 | 0.01 | 0.47 | 2.80 |
| div | torch.float16 | torch.float16 | 0.02 | 0.58 | 3.45 |
| div | torch.float16 | torch.float16 | 0.04 | 0.56 | 3.35 |

### torch

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| div | torch.float16 | torch.float16 | 0.01 | 0.33 | 1.96 |
| div | torch.float16 | torch.float16 | 0.03 | 0.37 | 2.22 |
| div | torch.float16 | torch.float16 | 0.06 | 0.38 | 2.27 |

## remainder

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| remainder | torch.float16 | torch.float16 | 0.01 | 0.46 | 2.79 |
| remainder | torch.float16 | torch.float16 | 0.02 | 0.57 | 3.42 |

### torch

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| remainder | torch.float16 | torch.float16 | 0.01 | 0.29 | 1.76 |
| remainder | torch.float16 | torch.float16 | 0.03 | 0.33 | 1.95 |

## pow

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| pow | torch.float16 | torch.float16 | 0.02 | 0.25 | 1.48 |
| pow | torch.float16 | torch.float16 | 0.05 | 0.23 | 1.40 |

### torch

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| pow | torch.float16 | torch.float16 | 0.03 | 0.16 | 0.97 |
| pow | torch.float16 | torch.float16 | 0.06 | 0.16 | 0.99 |

## floor_divide

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| floor_divide | torch.float16 | torch.float16 | 0.01 | 0.46 | 2.77 |
| floor_divide | torch.float16 | torch.float16 | 0.02 | 0.57 | 3.42 |

### torch

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| floor_divide | torch.float16 | torch.float16 | 0.03 | 0.12 | 0.72 |
| floor_divide | torch.float16 | torch.float16 | 0.09 | 0.12 | 0.72 |

## lerp

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| lerp | torch.float16 | torch.float16 | 0.01 | 0.47 | 2.84 |
| lerp | torch.float16 | torch.float16 | 0.02 | 0.58 | 3.46 |

### torch

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| lerp | torch.float16 | torch.float16 | 0.01 | 0.35 | 2.11 |
| lerp | torch.float16 | torch.float16 | 0.03 | 0.38 | 2.28 |

## maximum

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| maximum | torch.float16 | torch.float16 | 0.01 | 0.47 | 2.80 |
| maximum | torch.float16 | torch.float16 | 0.02 | 0.57 | 3.43 |
| maximum | torch.float16 | torch.float16 | 0.04 | 0.57 | 3.40 |

### torch

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| maximum | torch.float16 | torch.float16 | 0.01 | 0.34 | 2.01 |
| maximum | torch.float16 | torch.float16 | 0.03 | 0.37 | 2.22 |
| maximum | torch.float16 | torch.float16 | 0.05 | 0.38 | 2.29 |

## minimum

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| minimum | torch.float16 | torch.float16 | 0.01 | 0.47 | 2.82 |
| minimum | torch.float16 | torch.float16 | 0.02 | 0.56 | 3.37 |
| minimum | torch.float16 | torch.float16 | 0.04 | 0.57 | 3.40 |

### torch

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| minimum | torch.float16 | torch.float16 | 0.01 | 0.33 | 1.97 |
| minimum | torch.float16 | torch.float16 | 0.03 | 0.37 | 2.19 |
| minimum | torch.float16 | torch.float16 | 0.05 | 0.38 | 2.29 |

## cmp_eq

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| eq | torch.float16 | 0.02 | 0.22 | 1.12 |
| eq | torch.float16 | 0.04 | 0.26 | 1.28 |

### torch

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| eq | torch.float16 | 0.01 | 0.37 | 1.87 |
| eq | torch.float16 | 0.03 | 0.42 | 2.09 |

## cmp_ne

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| ne | torch.float16 | 0.02 | 0.22 | 1.11 |

### torch

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| ne | torch.float16 | 0.01 | 0.37 | 1.85 |

## cmp_gt

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| gt | torch.float16 | 0.02 | 0.22 | 1.10 |

### torch

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| gt | torch.float16 | 0.01 | 0.37 | 1.86 |

## cmp_lt

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| lt | torch.float16 | 0.02 | 0.22 | 1.11 |

### torch

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| lt | torch.float16 | 0.01 | 0.36 | 1.82 |

## cmp_ge

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| ge | torch.float16 | 0.02 | 0.22 | 1.11 |

### torch

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| ge | torch.float16 | 0.01 | 0.37 | 1.83 |

## cmp_le

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| le | torch.float16 | 0.02 | 0.22 | 1.11 |

### torch

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| le | torch.float16 | 0.01 | 0.36 | 1.82 |

## logical_and

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| logical_and | torch.float16 | 0.02 | 0.22 | 1.11 |
| logical_and | torch.float16 | 0.04 | 0.26 | 1.30 |

### torch

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| logical_and | torch.float16 | 0.01 | 0.59 | 2.93 |
| logical_and | torch.float16 | 0.01 | 0.70 | 3.52 |

## logical_or

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| logical_or | torch.float16 | 0.02 | 0.22 | 1.11 |
| logical_or | torch.float16 | 0.04 | 0.26 | 1.29 |

### torch

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| logical_or | torch.float16 | 0.01 | 0.59 | 2.95 |
| logical_or | torch.float16 | 0.01 | 0.71 | 3.53 |

## bitwise_and

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| bitwise_and | torch.int32 | 0.02 | 0.27 | 3.26 |
| bitwise_and | torch.int32 | 0.04 | 0.28 | 3.36 |

### torch

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| bitwise_and | torch.int32 | 0.02 | 0.19 | 2.25 |
| bitwise_and | torch.int32 | 0.05 | 0.19 | 2.31 |

## bitwise_or

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| bitwise_or | torch.int32 | 0.02 | 0.27 | 3.27 |

### torch

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| bitwise_or | torch.int32 | 0.02 | 0.19 | 2.25 |

## bitwise_xor

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| bitwise_xor | torch.int32 | 0.02 | 0.27 | 3.26 |

### torch

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| bitwise_xor | torch.int32 | 0.02 | 0.19 | 2.26 |

## gelu_and_mul

### tileops

| op_name | M | N | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| gelu_and_mul | 1024 | 4096 | torch.float16 | 0.01 | 0.79 | 2.37 |
| gelu_and_mul | 1024 | 10240 | torch.float16 | 0.03 | 0.79 | 2.38 |
| gelu_and_mul | 1024 | 20480 | torch.float16 | 0.06 | 0.69 | 2.06 |

### torch-ref

| op_name | M | N | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| gelu_and_mul | 1024 | 4096 | torch.float16 | 0.04 | 0.20 | 0.61 |
| gelu_and_mul | 1024 | 10240 | torch.float16 | 0.11 | 0.19 | 0.57 |
| gelu_and_mul | 1024 | 20480 | torch.float16 | 0.23 | 0.18 | 0.54 |

## gelu_tanh_and_mul

### tileops

| op_name | M | N | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| gelu_tanh_and_mul | 1024 | 4096 | torch.float16 | 0.01 | 0.93 | 2.79 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.float16 | 0.02 | 1.02 | 3.06 |
| gelu_tanh_and_mul | 1024 | 20480 | torch.float16 | 0.05 | 0.88 | 2.64 |

### torch-ref

| op_name | M | N | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| gelu_tanh_and_mul | 1024 | 4096 | torch.float16 | 0.04 | 0.21 | 0.64 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.float16 | 0.11 | 0.20 | 0.59 |
| gelu_tanh_and_mul | 1024 | 20480 | torch.float16 | 0.22 | 0.19 | 0.57 |

## silu_and_mul_strategy

### tileops-direct

| op_name | M | N | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| silu_and_mul | 1024 | 4096 | torch.float16 | direct | 0.02 | 0.50 | 1.49 |
| silu_and_mul | 1024 | 4096 | torch.bfloat16 | direct | 0.02 | 0.50 | 1.50 |
| silu_and_mul | 1024 | 4096 | torch.float32 | direct | 0.02 | 0.42 | 2.50 |
| silu_and_mul | 1024 | 10240 | torch.float16 | direct | 0.05 | 0.39 | 1.18 |
| silu_and_mul | 1024 | 10240 | torch.bfloat16 | direct | 0.05 | 0.39 | 1.18 |
| silu_and_mul | 1024 | 10240 | torch.float32 | direct | 0.06 | 0.34 | 2.02 |
| silu_and_mul | 4096 | 4096 | torch.float16 | direct | 0.09 | 0.36 | 1.09 |
| silu_and_mul | 4096 | 4096 | torch.bfloat16 | direct | 0.09 | 0.36 | 1.09 |
| silu_and_mul | 4096 | 4096 | torch.float32 | direct | 0.11 | 0.31 | 1.89 |

### tileops-explicit_parallel

| op_name | M | N | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| silu_and_mul | 1024 | 4096 | torch.float16 | explicit_parallel | 0.01 | 0.93 | 2.78 |
| silu_and_mul | 1024 | 4096 | torch.bfloat16 | explicit_parallel | 0.01 | 0.92 | 2.76 |
| silu_and_mul | 1024 | 4096 | torch.float32 | explicit_parallel | 0.02 | 0.55 | 3.31 |
| silu_and_mul | 1024 | 10240 | torch.float16 | explicit_parallel | 0.02 | 1.04 | 3.13 |
| silu_and_mul | 1024 | 10240 | torch.bfloat16 | explicit_parallel | 0.02 | 1.01 | 3.02 |
| silu_and_mul | 1024 | 10240 | torch.float32 | explicit_parallel | 0.04 | 0.50 | 2.99 |
| silu_and_mul | 4096 | 4096 | torch.float16 | explicit_parallel | 0.04 | 0.93 | 2.78 |
| silu_and_mul | 4096 | 4096 | torch.bfloat16 | explicit_parallel | 0.04 | 0.89 | 2.68 |
| silu_and_mul | 4096 | 4096 | torch.float32 | explicit_parallel | 0.07 | 0.46 | 2.74 |

## gelu_and_mul_strategy

### tileops-direct

| op_name | M | N | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gelu_and_mul | 1024 | 4096 | torch.float16 | direct | 0.02 | 0.49 | 1.48 |
| gelu_and_mul | 1024 | 4096 | torch.bfloat16 | direct | 0.02 | 0.49 | 1.47 |
| gelu_and_mul | 1024 | 4096 | torch.float32 | direct | 0.02 | 0.42 | 2.51 |
| gelu_and_mul | 1024 | 10240 | torch.float16 | direct | 0.05 | 0.39 | 1.17 |
| gelu_and_mul | 1024 | 10240 | torch.bfloat16 | direct | 0.05 | 0.39 | 1.17 |
| gelu_and_mul | 1024 | 10240 | torch.float32 | direct | 0.06 | 0.34 | 2.03 |
| gelu_and_mul | 4096 | 4096 | torch.float16 | direct | 0.09 | 0.36 | 1.08 |
| gelu_and_mul | 4096 | 4096 | torch.bfloat16 | direct | 0.09 | 0.36 | 1.08 |
| gelu_and_mul | 4096 | 4096 | torch.float32 | direct | 0.11 | 0.32 | 1.89 |

### tileops-explicit_parallel

| op_name | M | N | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gelu_and_mul | 1024 | 4096 | torch.float16 | explicit_parallel | 0.01 | 0.78 | 2.33 |
| gelu_and_mul | 1024 | 4096 | torch.bfloat16 | explicit_parallel | 0.01 | 0.78 | 2.33 |
| gelu_and_mul | 1024 | 4096 | torch.float32 | explicit_parallel | 0.02 | 0.55 | 3.28 |
| gelu_and_mul | 1024 | 10240 | torch.float16 | explicit_parallel | 0.03 | 0.79 | 2.37 |
| gelu_and_mul | 1024 | 10240 | torch.bfloat16 | explicit_parallel | 0.03 | 0.77 | 2.32 |
| gelu_and_mul | 1024 | 10240 | torch.float32 | explicit_parallel | 0.04 | 0.48 | 2.85 |
| gelu_and_mul | 4096 | 4096 | torch.float16 | explicit_parallel | 0.04 | 0.76 | 2.27 |
| gelu_and_mul | 4096 | 4096 | torch.bfloat16 | explicit_parallel | 0.04 | 0.75 | 2.26 |
| gelu_and_mul | 4096 | 4096 | torch.float32 | explicit_parallel | 0.07 | 0.46 | 2.73 |

## gelu_tanh_and_mul_strategy

### tileops-direct

| op_name | M | N | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gelu_tanh_and_mul | 1024 | 4096 | torch.float16 | direct | 0.02 | 0.50 | 1.51 |
| gelu_tanh_and_mul | 1024 | 4096 | torch.bfloat16 | direct | 0.02 | 0.50 | 1.50 |
| gelu_tanh_and_mul | 1024 | 4096 | torch.float32 | direct | 0.02 | 0.42 | 2.53 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.float16 | direct | 0.05 | 0.40 | 1.20 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.bfloat16 | direct | 0.05 | 0.40 | 1.19 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.float32 | direct | 0.06 | 0.34 | 2.05 |
| gelu_tanh_and_mul | 4096 | 4096 | torch.float16 | direct | 0.09 | 0.37 | 1.10 |
| gelu_tanh_and_mul | 4096 | 4096 | torch.bfloat16 | direct | 0.09 | 0.37 | 1.10 |
| gelu_tanh_and_mul | 4096 | 4096 | torch.float32 | direct | 0.10 | 0.32 | 1.92 |

### tileops-explicit_parallel

| op_name | M | N | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gelu_tanh_and_mul | 1024 | 4096 | torch.float16 | explicit_parallel | 0.01 | 0.92 | 2.77 |
| gelu_tanh_and_mul | 1024 | 4096 | torch.bfloat16 | explicit_parallel | 0.01 | 0.92 | 2.77 |
| gelu_tanh_and_mul | 1024 | 4096 | torch.float32 | explicit_parallel | 0.02 | 0.55 | 3.33 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.float16 | explicit_parallel | 0.02 | 1.02 | 3.06 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.bfloat16 | explicit_parallel | 0.02 | 1.01 | 3.03 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.float32 | explicit_parallel | 0.04 | 0.50 | 2.99 |
| gelu_tanh_and_mul | 4096 | 4096 | torch.float16 | explicit_parallel | 0.04 | 0.90 | 2.71 |
| gelu_tanh_and_mul | 4096 | 4096 | torch.bfloat16 | explicit_parallel | 0.04 | 0.90 | 2.71 |
| gelu_tanh_and_mul | 4096 | 4096 | torch.float32 | explicit_parallel | 0.07 | 0.46 | 2.75 |

## sub_bcast

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| sub | torch.float16 | 0.01 | 0.63 | 2.54 |
| sub | torch.float16 | 0.01 | 0.77 | 3.07 |
| sub | torch.float16 | 0.03 | 0.83 | 3.32 |

### torch

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| sub | torch.float16 | 0.02 | 0.20 | 0.80 |
| sub | torch.float16 | 0.05 | 0.21 | 0.84 |
| sub | torch.float16 | 0.10 | 0.21 | 0.83 |

## mul_bcast

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| mul | torch.float16 | 0.01 | 0.63 | 2.53 |
| mul | torch.float16 | 0.01 | 0.77 | 3.07 |
| mul | torch.float16 | 0.03 | 0.83 | 3.32 |

### torch

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| mul | torch.float16 | 0.02 | 0.20 | 0.80 |
| mul | torch.float16 | 0.05 | 0.21 | 0.83 |
| mul | torch.float16 | 0.10 | 0.21 | 0.84 |

## div_bcast

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| div | torch.float16 | 0.01 | 0.60 | 2.41 |
| div | torch.float16 | 0.01 | 0.73 | 2.93 |
| div | torch.float16 | 0.03 | 0.79 | 3.15 |

### torch

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| div | torch.float16 | 0.02 | 0.19 | 0.74 |
| div | torch.float16 | 0.05 | 0.19 | 0.77 |
| div | torch.float16 | 0.11 | 0.19 | 0.77 |

## binary_strategy

### add_direct

| n_total | shape_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4194304 | 1024x4096 | torch.float16 | direct | 0.02 | 0.26 | 1.57 |
| 4194304 | 1024x4096 | torch.bfloat16 | direct | 0.02 | 0.26 | 1.57 |
| 4194304 | 1024x4096 | torch.float32 | direct | 0.02 | 0.22 | 2.63 |
| 10485760 | 1024x10240 | torch.float16 | direct | 0.04 | 0.23 | 1.40 |
| 10485760 | 1024x10240 | torch.bfloat16 | direct | 0.04 | 0.23 | 1.40 |
| 10485760 | 1024x10240 | torch.float32 | direct | 0.06 | 0.19 | 2.27 |
| 20971520 | 1024x20480 | torch.float16 | direct | 0.11 | 0.20 | 1.19 |
| 20971520 | 1024x20480 | torch.bfloat16 | direct | 0.11 | 0.20 | 1.19 |
| 20971520 | 1024x20480 | torch.float32 | direct | 0.12 | 0.17 | 2.02 |

### add_explicit_parallel

| n_total | shape_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4194304 | 1024x4096 | torch.float16 | explicit_parallel | 0.01 | 0.46 | 2.75 |
| 4194304 | 1024x4096 | torch.bfloat16 | explicit_parallel | 0.01 | 0.46 | 2.76 |
| 4194304 | 1024x4096 | torch.float32 | explicit_parallel | 0.02 | 0.28 | 3.30 |
| 10485760 | 1024x10240 | torch.float16 | explicit_parallel | 0.02 | 0.56 | 3.35 |
| 10485760 | 1024x10240 | torch.bfloat16 | explicit_parallel | 0.02 | 0.56 | 3.34 |
| 10485760 | 1024x10240 | torch.float32 | explicit_parallel | 0.04 | 0.28 | 3.37 |
| 20971520 | 1024x20480 | torch.float16 | explicit_parallel | 0.04 | 0.53 | 3.16 |
| 20971520 | 1024x20480 | torch.bfloat16 | explicit_parallel | 0.04 | 0.53 | 3.16 |
| 20971520 | 1024x20480 | torch.float32 | explicit_parallel | 0.09 | 0.24 | 2.83 |

## conv1d

### tileops

| n | c_in | l_in | c_out | kernel_size | stride | padding | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 256 | 32000 | 512 | 1 | 1 | 0 | torch.float16 | 0.34 | 97.84 | 0.57 |
| 4 | 128 | 4096 | 256 | 3 | 1 | 1 | torch.float16 | 0.02 | 155.47 | 0.62 |
| 4 | 64 | 16000 | 128 | 5 | 2 | 2 | torch.float16 | 0.02 | 135.70 | 0.85 |
| 4 | 128 | 8192 | 256 | 7 | 1 | 3 | torch.float16 | 0.05 | 311.24 | 0.53 |
| 2 | 128 | 4096 | 256 | 3 | 2 | 1 | torch.bfloat16 | 0.01 | 76.65 | 0.42 |

### torch

| n | c_in | l_in | c_out | kernel_size | stride | padding | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 256 | 32000 | 512 | 1 | 1 | 0 | torch.float16 | 0.52 | 64.79 | 0.38 |
| 4 | 128 | 4096 | 256 | 3 | 1 | 1 | torch.float16 | 0.05 | 62.43 | 0.25 |
| 4 | 64 | 16000 | 128 | 5 | 2 | 2 | torch.float16 | 0.06 | 43.54 | 0.27 |
| 4 | 128 | 8192 | 256 | 7 | 1 | 3 | torch.float16 | 0.12 | 127.26 | 0.22 |
| 2 | 128 | 4096 | 256 | 3 | 2 | 1 | torch.bfloat16 | 0.02 | 32.65 | 0.18 |

## conv2d

### tileops

| n | c_in | h | w | c_out | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 64 | 56 | 56 | 64 | torch.float16 | 0.01 | 54.12 | 0.20 |
| 1 | 3 | 112 | 112 | 64 | torch.float16 | 0.01 | 1.42 | 0.06 |
| 1 | 128 | 56 | 56 | 256 | torch.float16 | 0.01 | 37.30 | 0.14 |
| 1 | 256 | 112 | 112 | 512 | torch.float16 | 0.08 | 382.83 | 0.28 |
| 1 | 64 | 56 | 56 | 128 | torch.float16 | 0.01 | 93.74 | 0.12 |
| 1 | 128 | 56 | 56 | 256 | torch.float16 | 0.02 | 55.28 | 0.12 |
| 1 | 128 | 28 | 28 | 128 | torch.bfloat16 | 0.01 | 5.32 | 0.05 |
| 2 | 64 | 56 | 56 | 256 | torch.float16 | 0.00 | 47.69 | 0.94 |
| 2 | 128 | 28 | 28 | 512 | torch.float16 | 0.00 | 52.13 | 0.54 |
| 2 | 512 | 28 | 28 | 128 | torch.float16 | 0.00 | 42.81 | 0.45 |
| 1 | 256 | 14 | 14 | 1024 | torch.float16 | 0.00 | 24.70 | 0.25 |
| 1 | 512 | 7 | 7 | 2048 | torch.float16 | 0.01 | 18.95 | 0.43 |
| 2 | 64 | 56 | 56 | 256 | torch.bfloat16 | 0.00 | 47.60 | 0.94 |

### torch-nchw

| n | c_in | h | w | c_out | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 64 | 56 | 56 | 64 | torch.float16 | 0.02 | 25.37 | 0.09 |
| 1 | 3 | 112 | 112 | 64 | torch.float16 | 0.01 | 1.15 | 0.05 |
| 1 | 128 | 56 | 56 | 256 | torch.float16 | 0.02 | 22.56 | 0.09 |
| 1 | 256 | 112 | 112 | 512 | torch.float16 | 0.15 | 193.49 | 0.14 |
| 1 | 64 | 56 | 56 | 128 | torch.float16 | 0.02 | 56.18 | 0.07 |
| 1 | 128 | 56 | 56 | 256 | torch.float16 | 0.03 | 44.48 | 0.10 |
| 1 | 128 | 28 | 28 | 128 | torch.bfloat16 | 0.02 | 3.12 | 0.03 |
| 2 | 64 | 56 | 56 | 256 | torch.float16 | 0.01 | 18.95 | 0.37 |
| 2 | 128 | 28 | 28 | 512 | torch.float16 | 0.01 | 24.61 | 0.26 |
| 2 | 512 | 28 | 28 | 128 | torch.float16 | 0.01 | 27.12 | 0.28 |
| 1 | 256 | 14 | 14 | 1024 | torch.float16 | 0.01 | 10.94 | 0.11 |
| 1 | 512 | 7 | 7 | 2048 | torch.float16 | 0.01 | 8.67 | 0.20 |
| 2 | 64 | 56 | 56 | 256 | torch.bfloat16 | 0.01 | 18.88 | 0.37 |

### torch-nhwc

| n | c_in | h | w | c_out | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 64 | 56 | 56 | 64 | torch.float16 | 0.01 | 36.40 | 0.13 |
| 1 | 3 | 112 | 112 | 64 | torch.float16 | 0.01 | 0.82 | 0.04 |
| 1 | 128 | 56 | 56 | 256 | torch.float16 | 0.01 | 31.44 | 0.12 |
| 1 | 256 | 112 | 112 | 512 | torch.float16 | 0.11 | 260.60 | 0.19 |
| 1 | 64 | 56 | 56 | 128 | torch.float16 | 0.02 | 72.37 | 0.09 |
| 1 | 128 | 56 | 56 | 256 | torch.float16 | 0.02 | 53.01 | 0.12 |
| 1 | 128 | 28 | 28 | 128 | torch.bfloat16 | 0.01 | 4.05 | 0.04 |
| 2 | 64 | 56 | 56 | 256 | torch.float16 | 0.01 | 20.09 | 0.40 |
| 2 | 128 | 28 | 28 | 512 | torch.float16 | 0.01 | 25.24 | 0.26 |
| 2 | 512 | 28 | 28 | 128 | torch.float16 | 0.01 | 27.39 | 0.28 |
| 1 | 256 | 14 | 14 | 1024 | torch.float16 | 0.01 | 14.34 | 0.14 |
| 1 | 512 | 7 | 7 | 2048 | torch.float16 | 0.01 | 13.62 | 0.31 |
| 2 | 64 | 56 | 56 | 256 | torch.bfloat16 | 0.01 | 20.18 | 0.40 |

## conv3d

### tileops

| n | c_in | d_in | h_in | w_in | c_out | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3 | 16 | 112 | 112 | 64 | torch.float16 | 0.08 | 24.82 | 0.32 |
| 1 | 64 | 8 | 56 | 56 | 128 | torch.float16 | 0.02 | 66.94 | 0.22 |
| 1 | 32 | 32 | 64 | 64 | 64 | torch.bfloat16 | 0.09 | 157.83 | 0.28 |

### torch-ncdhw

| n | c_in | d_in | h_in | w_in | c_out | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3 | 16 | 112 | 112 | 64 | torch.float16 | 0.23 | 9.24 | 0.12 |
| 1 | 64 | 8 | 56 | 56 | 128 | torch.float16 | 0.03 | 46.38 | 0.15 |
| 1 | 32 | 32 | 64 | 64 | 64 | torch.bfloat16 | 0.21 | 68.81 | 0.12 |

### torch-ndhwc

| n | c_in | d_in | h_in | w_in | c_out | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3 | 16 | 112 | 112 | 64 | torch.float16 | 0.16 | 13.31 | 0.17 |
| 1 | 64 | 8 | 56 | 56 | 128 | torch.float16 | 0.02 | 61.13 | 0.20 |
| 1 | 32 | 32 | 64 | 64 | 64 | torch.bfloat16 | 0.17 | 87.29 | 0.15 |

## CumsumOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | cumsum | 0.08 | 0.05 | 0.21 |
| 1024 | 4096 | torch.bfloat16 | cumsum | 0.08 | 0.05 | 0.21 |
| 4096 | 4096 | torch.float16 | cumsum | 0.19 | 0.09 | 0.35 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | cumsum | 0.16 | 0.03 | 0.10 |
| 1024 | 4096 | torch.bfloat16 | cumsum | 0.16 | 0.03 | 0.10 |
| 4096 | 4096 | torch.float16 | cumsum | 0.47 | 0.04 | 0.14 |

## CumprodOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | cumprod | 0.08 | 0.05 | 0.21 |
| 1024 | 4096 | torch.bfloat16 | cumprod | 0.08 | 0.05 | 0.22 |
| 4096 | 4096 | torch.float16 | cumprod | 0.19 | 0.09 | 0.35 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | cumprod | 0.17 | 0.03 | 0.10 |
| 1024 | 4096 | torch.bfloat16 | cumprod | 0.17 | 0.03 | 0.10 |
| 4096 | 4096 | torch.float16 | cumprod | 0.47 | 0.04 | 0.14 |

## DaCumsumFwdOp

### tileops

| batch | num_chunks | chunk_len | n_heads | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 64 | 4 | 0.00 | 0.00 | 0.00 |
| 2 | 4 | 64 | 8 | 0.00 | 0.00 | 0.01 |
| 1 | 2 | 128 | 4 | 0.00 | 0.00 | 0.00 |
| 2 | 4 | 128 | 16 | 0.01 | 0.00 | 0.02 |

## da_cumsum_fwd

### baseline

| batch | num_chunks | chunk_len | n_heads | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 64 | 4 | 0.01 | 0.00 | 0.00 |
| 2 | 4 | 64 | 8 | 0.02 | 0.00 | 0.00 |
| 1 | 2 | 128 | 4 | 0.02 | 0.00 | 0.00 |
| 2 | 4 | 128 | 16 | 0.04 | 0.00 | 0.00 |

## DeepSeekSparseAttentionDecodeWithKVCacheOp

### tileops

| batch | heads | seq_len_q | seq_len_kv | dim | dim_tail | topk | stride_kv | heads_kv | q_start_index_s | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 128 | 1024 | 2048 | 512 | 64 | 2048 | 1 | 1 | 1024 | torch.float16 | 2.26 | 258.94 | 0.13 |
| 1 | 128 | 512 | 4096 | 512 | 64 | 1024 | 1 | 1 | 512 | torch.float16 | 0.57 | 254.65 | 0.26 |

### torch-ref

| batch | heads | seq_len_q | seq_len_kv | dim | dim_tail | topk | stride_kv | heads_kv | q_start_index_s | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 128 | 1024 | 2048 | 512 | 64 | 2048 | 1 | 1 | 1024 | torch.float16 | 31.23 | 18.70 | 0.01 |
| 1 | 128 | 512 | 4096 | 512 | 64 | 1024 | 1 | 1 | 512 | torch.float16 | 31.64 | 4.62 | 0.00 |

## MultiHeadLatentAttentionDecodeWithKVCacheOp

### tileops

| batch | heads | heads_kv | seq_len_kv | dim | dim_pe | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 128 | 1 | 8192 | 512 | 64 | torch.float16 | 0.26 | 281.68 | 1.18 |
| 16 | 128 | 1 | 4096 | 512 | 64 | torch.float16 | 0.05 | 369.06 | 1.57 |

### torch-ref

| batch | heads | heads_kv | seq_len_kv | dim | dim_pe | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 128 | 1 | 8192 | 512 | 64 | torch.float16 | 0.89 | 81.83 | 0.34 |
| 16 | 128 | 1 | 4096 | 512 | 64 | torch.float16 | 0.15 | 123.95 | 0.53 |

## DeltaNetFwdOp

### tileops

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 0.10 | 2.78 | 0.17 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 0.21 | 1.29 | 0.08 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 0.21 | 1.29 | 0.08 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 0.05 | 2.55 | 0.16 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 0.32 | 1.70 | 0.11 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 0.73 | 1.46 | 0.09 |
| 2 | 32768 | 4 | 64 | 64 | 64 | torch.float16 | 1.55 | 1.39 | 0.09 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.05 | 2.54 | 0.16 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.10 | 2.72 | 0.17 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.32 | 1.68 | 0.11 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.74 | 1.46 | 0.09 |
| 2 | 32768 | 4 | 64 | 64 | 64 | torch.bfloat16 | 1.55 | 1.38 | 0.09 |

### fla

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.08 | 3.19 | 0.20 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 0.125 | 0.08 | 3.18 | 0.20 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 0.125 | 0.08 | 3.17 | 0.20 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.05 | 2.44 | 0.15 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.16 | 3.30 | 0.21 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.33 | 3.22 | 0.20 |
| 2 | 32768 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.92 | 2.33 | 0.15 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.05 | 2.45 | 0.15 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.08 | 3.18 | 0.20 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.16 | 3.28 | 0.21 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.34 | 3.11 | 0.20 |
| 2 | 32768 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.93 | 2.30 | 0.14 |

## DeltaNetBwdOp

### tileops

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | H | S | DK | DV | BC | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.33 | 1.61 | 0.09 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.39 | 1.36 | 0.07 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.40 | 1.35 | 0.07 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.12 | 2.28 | 0.13 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.75 | 1.43 | 0.08 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 16384 | 64 | 64 | 64 | 1.59 | 1.35 | 0.07 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.12 | 2.33 | 0.13 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.33 | 1.63 | 0.09 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.75 | 1.43 | 0.08 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 16384 | 64 | 64 | 64 | 1.60 | 1.35 | 0.07 |

### fla

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | H | S | DK | DV | BC | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.125 | 0.18 | 3.03 | 0.17 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.125 | 0.18 | 3.02 | 0.17 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.125 | 0.18 | 3.02 | 0.17 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.125 | 0.09 | 2.83 | 0.16 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.125 | 0.36 | 3.01 | 0.17 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 16384 | 64 | 64 | 64 | 0.125 | 0.92 | 2.33 | 0.13 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.125 | 0.09 | 2.84 | 0.16 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.125 | 0.18 | 3.02 | 0.17 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.125 | 0.36 | 2.96 | 0.16 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 16384 | 64 | 64 | 64 | 0.125 | 0.93 | 2.30 | 0.13 |

## DeltaNetOp

### tileops

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | H | S | DK | DV | BC | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.32 | 2.48 | 0.14 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.47 | 1.71 | 0.10 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.47 | 1.71 | 0.10 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.17 | 2.42 | 0.14 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.94 | 1.71 | 0.10 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 16384 | 64 | 64 | 64 | 2.19 | 1.47 | 0.08 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.17 | 2.43 | 0.14 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.32 | 2.51 | 0.14 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.94 | 1.71 | 0.10 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 16384 | 64 | 64 | 64 | 2.20 | 1.46 | 0.08 |

### fla

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | H | S | DK | DV | BC | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.125 | 0.26 | 3.11 | 0.18 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.125 | 0.26 | 3.12 | 0.18 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.125 | 0.26 | 3.11 | 0.18 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.125 | 0.15 | 2.75 | 0.16 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.125 | 0.52 | 3.12 | 0.18 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 16384 | 64 | 64 | 64 | 0.125 | 1.17 | 2.74 | 0.16 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.125 | 0.15 | 2.75 | 0.16 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.125 | 0.26 | 3.11 | 0.18 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.125 | 0.52 | 3.10 | 0.18 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 16384 | 64 | 64 | 64 | 0.125 | 1.19 | 2.71 | 0.16 |

## DeltaNetDecodeOp

### tileops

| batch | heads | dim_k | dim_v | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 128 | 128 | torch.float32 | 0.01 | 0.32 | 0.43 |
| 1 | 32 | 128 | 128 | torch.bfloat16 | 0.01 | 0.34 | 0.23 |
| 8 | 32 | 128 | 128 | torch.float32 | 0.03 | 0.96 | 1.29 |
| 8 | 32 | 128 | 128 | torch.bfloat16 | 0.01 | 2.18 | 1.47 |
| 16 | 32 | 128 | 128 | torch.float32 | 0.06 | 0.84 | 1.13 |
| 16 | 32 | 128 | 128 | torch.bfloat16 | 0.02 | 2.69 | 1.82 |
| 32 | 32 | 128 | 128 | torch.float32 | 0.13 | 0.78 | 1.05 |
| 32 | 32 | 128 | 128 | torch.bfloat16 | 0.04 | 2.46 | 1.66 |
| 1 | 32 | 64 | 64 | torch.float32 | 0.01 | 0.15 | 0.21 |
| 8 | 32 | 64 | 64 | torch.float32 | 0.01 | 0.63 | 0.86 |
| 32 | 32 | 64 | 64 | torch.float32 | 0.05 | 0.49 | 0.68 |
| 64 | 32 | 128 | 128 | torch.float32 | 0.27 | 0.74 | 0.99 |
| 64 | 32 | 128 | 128 | torch.bfloat16 | 0.09 | 2.30 | 1.56 |

### torch

| batch | heads | dim_k | dim_v | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 128 | 128 | torch.float32 | 0.02 | 0.14 | 0.19 |
| 1 | 32 | 128 | 128 | torch.bfloat16 | 0.04 | 0.08 | 0.05 |
| 8 | 32 | 128 | 128 | torch.float32 | 0.06 | 0.44 | 0.59 |
| 8 | 32 | 128 | 128 | torch.bfloat16 | 0.09 | 0.29 | 0.20 |
| 16 | 32 | 128 | 128 | torch.float32 | 0.08 | 0.59 | 0.80 |
| 16 | 32 | 128 | 128 | torch.bfloat16 | 0.14 | 0.37 | 0.25 |
| 32 | 32 | 128 | 128 | torch.float32 | 0.17 | 0.60 | 0.80 |
| 32 | 32 | 128 | 128 | torch.bfloat16 | 0.28 | 0.36 | 0.24 |
| 1 | 32 | 64 | 64 | torch.float32 | 0.02 | 0.04 | 0.06 |
| 8 | 32 | 64 | 64 | torch.float32 | 0.03 | 0.24 | 0.32 |
| 32 | 32 | 64 | 64 | torch.float32 | 0.05 | 0.46 | 0.63 |
| 64 | 32 | 128 | 128 | torch.float32 | 0.41 | 0.49 | 0.66 |
| 64 | 32 | 128 | 128 | torch.bfloat16 | 0.67 | 0.30 | 0.20 |

## DropoutOp

### tileops

| dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| torch.float16 | 4194304 | 0.01 | 0.64 | 2.58 |
| torch.bfloat16 | 4194304 | 0.01 | 0.65 | 2.59 |
| torch.float32 | 4194304 | 0.01 | 0.39 | 3.16 |
| torch.float16 | 10485760 | 0.01 | 0.83 | 3.33 |
| torch.bfloat16 | 10485760 | 0.01 | 0.83 | 3.34 |
| torch.float32 | 10485760 | 0.02 | 0.47 | 3.74 |
| torch.float16 | 20971520 | 0.02 | 0.94 | 3.74 |
| torch.bfloat16 | 20971520 | 0.02 | 0.94 | 3.74 |
| torch.float32 | 20971520 | 0.05 | 0.40 | 3.23 |

### torch

| dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| torch.float16 | 4194304 | 0.01 | 0.35 | 1.41 |
| torch.bfloat16 | 4194304 | 0.01 | 0.35 | 1.39 |
| torch.float32 | 4194304 | 0.02 | 0.24 | 1.89 |
| torch.float16 | 10485760 | 0.03 | 0.38 | 1.51 |
| torch.bfloat16 | 10485760 | 0.03 | 0.37 | 1.49 |
| torch.float32 | 10485760 | 0.05 | 0.22 | 1.74 |
| torch.float16 | 20971520 | 0.06 | 0.36 | 1.42 |
| torch.bfloat16 | 20971520 | 0.06 | 0.35 | 1.39 |
| torch.float32 | 20971520 | 0.10 | 0.20 | 1.64 |

## relu_fp8

### tileops

| op_name | n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| relu_fp8 | 8388608 | torch.float8_e4m3fn | 0.01 | 0.58 | 1.16 |
| relu_fp8 | 8388608 | torch.float8_e5m2 | 0.02 | 0.46 | 0.92 |
| relu_fp8 | 67108864 | torch.float8_e4m3fn | 0.16 | 0.43 | 0.86 |
| relu_fp8 | 67108864 | torch.float8_e5m2 | 0.20 | 0.34 | 0.67 |
| relu_fp8 | 134217728 | torch.float8_e4m3fn | 0.34 | 0.40 | 0.80 |
| relu_fp8 | 134217728 | torch.float8_e5m2 | 0.44 | 0.31 | 0.62 |

### torch

| op_name | n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| relu_fp8 | 8388608 | torch.float8_e4m3fn | 0.06 | 0.14 | 0.27 |
| relu_fp8 | 8388608 | torch.float8_e5m2 | 0.06 | 0.14 | 0.28 |
| relu_fp8 | 67108864 | torch.float8_e4m3fn | 0.65 | 0.10 | 0.21 |
| relu_fp8 | 67108864 | torch.float8_e5m2 | 0.63 | 0.11 | 0.21 |
| relu_fp8 | 134217728 | torch.float8_e4m3fn | 1.33 | 0.10 | 0.20 |
| relu_fp8 | 134217728 | torch.float8_e5m2 | 1.28 | 0.10 | 0.21 |

## exp_fp8

### tileops

| op_name | n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| exp_fp8 | 8388608 | torch.float8_e4m3fn | 0.01 | 0.99 | 1.97 |
| exp_fp8 | 8388608 | torch.float8_e5m2 | 0.02 | 0.46 | 0.91 |
| exp_fp8 | 67108864 | torch.float8_e4m3fn | 0.07 | 0.94 | 1.87 |
| exp_fp8 | 67108864 | torch.float8_e5m2 | 0.20 | 0.33 | 0.66 |
| exp_fp8 | 134217728 | torch.float8_e4m3fn | 0.16 | 0.82 | 1.65 |
| exp_fp8 | 134217728 | torch.float8_e5m2 | 0.44 | 0.31 | 0.61 |

### torch

| op_name | n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| exp_fp8 | 8388608 | torch.float8_e4m3fn | 0.06 | 0.13 | 0.27 |
| exp_fp8 | 8388608 | torch.float8_e5m2 | 0.06 | 0.14 | 0.28 |
| exp_fp8 | 67108864 | torch.float8_e4m3fn | 0.64 | 0.10 | 0.21 |
| exp_fp8 | 67108864 | torch.float8_e5m2 | 0.62 | 0.11 | 0.21 |
| exp_fp8 | 134217728 | torch.float8_e4m3fn | 1.31 | 0.10 | 0.21 |
| exp_fp8 | 134217728 | torch.float8_e5m2 | 1.27 | 0.11 | 0.21 |

## add_fp8

### tileops

| op_name | n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| add_fp8 | 8388608 | torch.float8_e4m3fn | 0.01 | 0.86 | 2.57 |
| add_fp8 | 8388608 | torch.float8_e5m2 | 0.02 | 0.42 | 1.25 |
| add_fp8 | 67108864 | torch.float8_e4m3fn | 0.08 | 0.81 | 2.43 |
| add_fp8 | 67108864 | torch.float8_e5m2 | 0.22 | 0.31 | 0.93 |
| add_fp8 | 134217728 | torch.float8_e4m3fn | 0.19 | 0.72 | 2.15 |
| add_fp8 | 134217728 | torch.float8_e5m2 | 0.47 | 0.28 | 0.85 |

### torch

| op_name | n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| add_fp8 | 8388608 | torch.float8_e4m3fn | 0.11 | 0.07 | 0.22 |
| add_fp8 | 8388608 | torch.float8_e5m2 | 0.11 | 0.08 | 0.23 |
| add_fp8 | 67108864 | torch.float8_e4m3fn | 1.11 | 0.06 | 0.18 |
| add_fp8 | 67108864 | torch.float8_e5m2 | 1.06 | 0.06 | 0.19 |
| add_fp8 | 134217728 | torch.float8_e4m3fn | 2.23 | 0.06 | 0.18 |
| add_fp8 | 134217728 | torch.float8_e5m2 | 2.15 | 0.06 | 0.19 |

## silu_and_mul_fp8

### tileops

| op_name | M | N | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| silu_and_mul_fp8 | 2048 | 11008 | torch.float8_e4m3fn | 0.06 | 1.84 | 1.10 |
| silu_and_mul_fp8 | 2048 | 11008 | torch.float8_e5m2 | 0.09 | 1.26 | 0.76 |
| silu_and_mul_fp8 | 16384 | 11008 | torch.float8_e4m3fn | 0.62 | 1.47 | 0.88 |
| silu_and_mul_fp8 | 16384 | 11008 | torch.float8_e5m2 | 0.89 | 1.01 | 0.61 |
| silu_and_mul_fp8 | 16384 | 28672 | torch.float8_e4m3fn | 1.68 | 1.40 | 0.84 |
| silu_and_mul_fp8 | 16384 | 28672 | torch.float8_e5m2 | 2.39 | 0.98 | 0.59 |

### torch-ref

| op_name | M | N | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| silu_and_mul_fp8 | 2048 | 11008 | torch.float8_e4m3fn | 0.54 | 0.21 | 0.12 |
| silu_and_mul_fp8 | 2048 | 11008 | torch.float8_e5m2 | 0.53 | 0.21 | 0.13 |
| silu_and_mul_fp8 | 16384 | 11008 | torch.float8_e4m3fn | 4.52 | 0.20 | 0.12 |
| silu_and_mul_fp8 | 16384 | 11008 | torch.float8_e5m2 | 4.39 | 0.21 | 0.12 |
| silu_and_mul_fp8 | 16384 | 28672 | torch.float8_e4m3fn | 11.82 | 0.20 | 0.12 |
| silu_and_mul_fp8 | 16384 | 28672 | torch.float8_e5m2 | 11.47 | 0.20 | 0.12 |

## EngramGateConvBwdOp

### tileops

| M | seq_len | d | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 256 | torch.float16 | 0.01 | 0.07 | 0.03 |
| 2 | 64 | 512 | torch.float16 | 0.01 | 0.33 | 0.11 |
| 1 | 128 | 256 | torch.bfloat16 | 0.01 | 0.22 | 0.08 |
| 2 | 16 | 256 | torch.bfloat16 | 0.01 | 0.07 | 0.02 |

### torch

| M | seq_len | d | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 256 | torch.float16 | 0.20 | 0.00 | 0.00 |
| 2 | 64 | 512 | torch.float16 | 0.23 | 0.02 | 0.01 |
| 1 | 128 | 256 | torch.bfloat16 | 0.21 | 0.01 | 0.00 |
| 2 | 16 | 256 | torch.bfloat16 | 0.20 | 0.00 | 0.00 |

## EngramDecodeOp

### tileops

| batch | d_mem | d | max_conv_len | conv_kernel_size | dilation | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 512 | 256 | 12 | 4 | 3 | torch.float16 | 0.03 | 0.02 | 0.02 |
| 4 | 1024 | 512 | 20 | 4 | 5 | torch.float16 | 0.10 | 0.09 | 0.02 |
| 8 | 512 | 256 | 18 | 4 | 3 | torch.bfloat16 | 0.03 | 0.14 | 0.02 |

### torch-ref

| batch | d_mem | d | max_conv_len | conv_kernel_size | dilation | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 512 | 256 | 12 | 4 | 3 | torch.float16 | 0.10 | 0.01 | 0.01 |
| 4 | 1024 | 512 | 20 | 4 | 5 | torch.float16 | 0.13 | 0.06 | 0.02 |
| 8 | 512 | 256 | 18 | 4 | 3 | torch.bfloat16 | 0.12 | 0.03 | 0.01 |

## EngramGateConvFwdOp

### tileops

| M | seq_len | d | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 256 | torch.float16 | 0.00 | 0.05 | 0.02 |
| 2 | 64 | 512 | torch.float16 | 0.01 | 0.30 | 0.13 |
| 1 | 128 | 256 | torch.bfloat16 | 0.00 | 0.16 | 0.07 |
| 2 | 16 | 256 | torch.bfloat16 | 0.00 | 0.05 | 0.02 |

### torch-ref

| M | seq_len | d | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 256 | torch.float16 | 0.08 | 0.00 | 0.00 |
| 2 | 64 | 512 | torch.float16 | 0.09 | 0.02 | 0.01 |
| 1 | 128 | 256 | torch.bfloat16 | 0.09 | 0.01 | 0.00 |
| 2 | 16 | 256 | torch.bfloat16 | 0.08 | 0.00 | 0.00 |

## FFTC2COp

### tileops

| n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 4096 | torch.complex64 | 0.01 | 0.02 | 0.01 |
| 16384 | torch.complex64 | 0.01 | 0.09 | 0.02 |
| 65536 | torch.complex64 | 0.02 | 0.30 | 0.06 |
| 262144 | torch.complex64 | 0.02 | 0.96 | 0.17 |
| 1048576 | torch.complex64 | 0.06 | 1.69 | 0.27 |
| 4096 | torch.complex64 | 0.02 | 0.89 | 0.24 |
| 4096 | torch.complex64 | 0.04 | 1.50 | 0.40 |
| 1024 | torch.complex64 | 0.04 | 1.30 | 0.42 |
| 4096 | torch.complex128 | 0.02 | 0.01 | 0.01 |
| 65536 | torch.complex128 | 0.03 | 0.16 | 0.06 |
| 4096 | torch.complex128 | 0.03 | 0.47 | 0.25 |

### torch-cufft

| n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 4096 | torch.complex64 | 0.00 | 0.05 | 0.01 |
| 16384 | torch.complex64 | 0.01 | 0.08 | 0.02 |
| 65536 | torch.complex64 | 0.01 | 0.73 | 0.15 |
| 262144 | torch.complex64 | 0.01 | 2.11 | 0.38 |
| 1048576 | torch.complex64 | 0.02 | 4.33 | 0.69 |
| 4096 | torch.complex64 | 0.01 | 2.84 | 0.76 |
| 4096 | torch.complex64 | 0.01 | 6.46 | 1.72 |
| 1024 | torch.complex64 | 0.01 | 6.55 | 2.10 |
| 4096 | torch.complex128 | 0.01 | 0.03 | 0.02 |
| 65536 | torch.complex128 | 0.01 | 0.54 | 0.22 |
| 4096 | torch.complex128 | 0.01 | 1.77 | 0.95 |

## Fp8LightingIndexerOp

### tileops

| batch | seq_len | heads | index_dim | seq_len_kv | kv_group | clean_logits | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 4096 | 32 | 64 | 8192 | 1 | True | 0.20 | N/A | 0.70 |
| 1 | 2048 | 16 | 64 | 4096 | 1 | True | 0.12 | N/A | 0.30 |

### torch-ref

| batch | seq_len | heads | index_dim | seq_len_kv | kv_group | clean_logits | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 4096 | 32 | 64 | 8192 | 1 | True | 23.44 | N/A | 0.01 |
| 1 | 2048 | 16 | 64 | 4096 | 1 | True | 2.90 | N/A | 0.01 |

## Fp8QuantOp

### tileops

| batch | seq_len_kv | kv_group | index_dim | in_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8192 | 1 | 64 | torch.float16 | 0.00 | 1.10 | 0.36 |
| 1 | 8192 | 1 | 64 | torch.bfloat16 | 0.00 | 1.09 | 0.36 |
| 1 | 4096 | 1 | 128 | torch.float32 | 0.00 | 0.81 | 0.54 |
| 1 | 16384 | 1 | 32 | torch.float32 | 0.00 | 1.02 | 0.67 |

### torch-ref

| batch | seq_len_kv | kv_group | index_dim | in_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8192 | 1 | 64 | torch.float16 | 0.02 | 0.18 | 0.06 |
| 1 | 8192 | 1 | 64 | torch.bfloat16 | 0.02 | 0.18 | 0.06 |
| 1 | 4096 | 1 | 128 | torch.float32 | 0.02 | 0.18 | 0.12 |
| 1 | 16384 | 1 | 32 | torch.float32 | 0.02 | 0.15 | 0.10 |

## FusedAddLayerNormOp

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 2048 | 4096 | torch.float16 | 0.04 | 1.43 | 1.90 |
| 2048 | 4096 | torch.bfloat16 | 0.05 | 0.95 | 1.27 |
| 1 | 4096 | torch.bfloat16 | 0.00 | 0.01 | 0.01 |
| 2048 | 8192 | torch.float16 | 0.10 | 1.00 | 1.33 |
| 2048 | 8192 | torch.bfloat16 | 0.10 | 1.05 | 1.40 |
| 1 | 8192 | torch.bfloat16 | 0.01 | 0.01 | 0.02 |

### torch-ref

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 2048 | 4096 | torch.float16 | 0.16 | 0.31 | 0.41 |
| 2048 | 4096 | torch.bfloat16 | 0.16 | 0.31 | 0.41 |
| 1 | 4096 | torch.bfloat16 | 0.02 | 0.00 | 0.00 |
| 2048 | 8192 | torch.float16 | 0.37 | 0.28 | 0.37 |
| 2048 | 8192 | torch.bfloat16 | 0.37 | 0.27 | 0.36 |
| 1 | 8192 | torch.bfloat16 | 0.03 | 0.00 | 0.00 |

## FusedAddRmsNormOp

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 2048 | 4096 | torch.float16 | 0.03 | 1.29 | 2.07 |
| 2048 | 4096 | torch.bfloat16 | 0.05 | 0.90 | 1.44 |
| 1 | 4096 | torch.bfloat16 | 0.00 | 0.01 | 0.01 |
| 2048 | 8192 | torch.float16 | 0.08 | 0.99 | 1.58 |
| 2048 | 8192 | torch.bfloat16 | 0.09 | 0.97 | 1.56 |
| 1 | 8192 | torch.bfloat16 | 0.00 | 0.01 | 0.02 |

### torch-ref

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 2048 | 4096 | torch.float16 | 0.37 | 0.11 | 0.18 |
| 2048 | 4096 | torch.bfloat16 | 0.38 | 0.11 | 0.18 |
| 1 | 4096 | torch.bfloat16 | 0.03 | 0.00 | 0.00 |
| 2048 | 8192 | torch.float16 | 0.83 | 0.10 | 0.16 |
| 2048 | 8192 | torch.bfloat16 | 0.83 | 0.10 | 0.16 |
| 1 | 8192 | torch.bfloat16 | 0.03 | 0.00 | 0.00 |

## GatedDeltaNetFwdOp

### tileops

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 0.24 | 1.12 | 0.07 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 0.25 | 1.09 | 0.07 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 0.08 | 1.65 | 0.10 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 0.14 | 1.93 | 0.12 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 0.42 | 1.28 | 0.08 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 0.94 | 1.14 | 0.07 |
| 2 | 32768 | 4 | 64 | 64 | 64 | torch.float16 | 1.97 | 1.09 | 0.07 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.08 | 1.65 | 0.10 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.14 | 1.93 | 0.12 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.41 | 1.29 | 0.08 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.93 | 1.16 | 0.07 |
| 2 | 32768 | 4 | 64 | 64 | 64 | torch.bfloat16 | 1.96 | 1.10 | 0.07 |

### fla

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 0.125 | 0.10 | 2.73 | 0.17 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 0.125 | 0.10 | 2.71 | 0.17 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.06 | 2.15 | 0.14 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.10 | 2.73 | 0.17 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.19 | 2.84 | 0.18 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.38 | 2.79 | 0.18 |
| 2 | 32768 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 1.04 | 2.06 | 0.13 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.06 | 2.14 | 0.13 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.10 | 2.71 | 0.17 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.19 | 2.83 | 0.18 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.39 | 2.78 | 0.18 |
| 2 | 32768 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 1.06 | 2.03 | 0.13 |

## GatedDeltaNetBwdOp

### tileops

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | H | S | DK | DV | BC | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.58 | 0.93 | 0.05 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.56 | 0.95 | 0.05 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.19 | 1.43 | 0.08 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.40 | 1.33 | 0.07 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.98 | 1.09 | 0.06 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 16384 | 64 | 64 | 64 | 2.20 | 0.98 | 0.05 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.19 | 1.41 | 0.08 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.41 | 1.31 | 0.07 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.99 | 1.09 | 0.06 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 16384 | 64 | 64 | 64 | 2.22 | 0.97 | 0.05 |

### fla

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | H | S | DK | DV | BC | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.125 | 0.25 | 2.17 | 0.12 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.125 | 0.25 | 2.16 | 0.12 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.125 | 0.13 | 2.00 | 0.11 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.125 | 0.25 | 2.17 | 0.12 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.125 | 0.47 | 2.26 | 0.12 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 16384 | 64 | 64 | 64 | 0.125 | 1.18 | 1.82 | 0.10 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.125 | 0.14 | 1.99 | 0.11 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.125 | 0.25 | 2.16 | 0.12 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.125 | 0.48 | 2.25 | 0.12 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 16384 | 64 | 64 | 64 | 0.125 | 1.18 | 1.81 | 0.10 |

## GatedDeltaNetOp

### tileops

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | H | S | DK | DV | BC | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.66 | 1.22 | 0.07 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.66 | 1.22 | 0.07 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.27 | 1.52 | 0.09 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.49 | 1.64 | 0.09 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 8192 | 64 | 64 | 64 | 1.24 | 1.30 | 0.07 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 16384 | 64 | 64 | 64 | 2.97 | 1.08 | 0.06 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.27 | 1.50 | 0.09 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.50 | 1.62 | 0.09 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 8192 | 64 | 64 | 64 | 1.26 | 1.28 | 0.07 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 16384 | 64 | 64 | 64 | 2.98 | 1.08 | 0.06 |

### fla

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | H | S | DK | DV | BC | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.125 | 0.34 | 2.35 | 0.14 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.125 | 0.34 | 2.34 | 0.14 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.125 | 0.19 | 2.08 | 0.12 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.125 | 0.34 | 2.35 | 0.14 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.125 | 0.66 | 2.44 | 0.14 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 16384 | 64 | 64 | 64 | 0.125 | 1.45 | 2.23 | 0.13 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.125 | 0.19 | 2.07 | 0.12 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.125 | 0.34 | 2.34 | 0.14 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.125 | 0.66 | 2.42 | 0.14 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 16384 | 64 | 64 | 64 | 0.125 | 1.47 | 2.19 | 0.13 |

## GatedDeltaNetDecodeOp

### tileops

| batch | heads | dim_k | dim_v | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 128 | 128 | torch.float32 | 0.01 | 0.32 | 0.43 |
| 1 | 32 | 128 | 128 | torch.bfloat16 | 0.01 | 0.32 | 0.21 |
| 8 | 32 | 128 | 128 | torch.float32 | 0.03 | 0.95 | 1.28 |
| 8 | 32 | 128 | 128 | torch.bfloat16 | 0.01 | 2.09 | 1.41 |
| 16 | 32 | 128 | 128 | torch.float32 | 0.06 | 0.84 | 1.14 |
| 16 | 32 | 128 | 128 | torch.bfloat16 | 0.02 | 2.64 | 1.78 |
| 32 | 32 | 128 | 128 | torch.float32 | 0.13 | 0.77 | 1.04 |
| 32 | 32 | 128 | 128 | torch.bfloat16 | 0.04 | 2.34 | 1.58 |
| 1 | 32 | 64 | 64 | torch.float32 | 0.01 | 0.15 | 0.20 |
| 8 | 32 | 64 | 64 | torch.float32 | 0.01 | 0.62 | 0.85 |
| 32 | 32 | 64 | 64 | torch.float32 | 0.05 | 0.49 | 0.66 |
| 64 | 32 | 128 | 128 | torch.float32 | 0.28 | 0.73 | 0.99 |
| 64 | 32 | 128 | 128 | torch.bfloat16 | 0.09 | 2.25 | 1.52 |

### fla

| batch | heads | dim_k | dim_v | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 128 | 128 | torch.float32 | 0.00 | 0.73 | 0.99 |
| 1 | 32 | 128 | 128 | torch.bfloat16 | 0.00 | 0.68 | 0.46 |
| 8 | 32 | 128 | 128 | torch.float32 | 0.02 | 1.65 | 2.23 |
| 8 | 32 | 128 | 128 | torch.bfloat16 | 0.01 | 1.76 | 1.19 |
| 16 | 32 | 128 | 128 | torch.float32 | 0.03 | 1.90 | 2.57 |
| 16 | 32 | 128 | 128 | torch.bfloat16 | 0.02 | 2.03 | 1.37 |
| 32 | 32 | 128 | 128 | torch.float32 | 0.05 | 2.04 | 2.75 |
| 32 | 32 | 128 | 128 | torch.bfloat16 | 0.05 | 2.22 | 1.50 |
| 1 | 32 | 64 | 64 | torch.float32 | 0.00 | 0.25 | 0.35 |
| 8 | 32 | 64 | 64 | torch.float32 | 0.01 | 1.16 | 1.59 |
| 32 | 32 | 64 | 64 | torch.float32 | 0.02 | 1.68 | 2.30 |
| 64 | 32 | 128 | 128 | torch.float32 | 0.12 | 1.68 | 2.27 |
| 64 | 32 | 128 | 128 | torch.bfloat16 | 0.11 | 1.92 | 1.30 |

## GemmOp

### tileops

| m | n | k | dtype | trans_a | trans_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1024 | 1024 | 1024 | torch.float16 | False | False | 0.01 | 249.29 | 0.73 |
| 1 | 7168 | 16384 | torch.float16 | False | True | 0.11 | 2.15 | 2.15 |
| 1 | 18432 | 7168 | torch.bfloat16 | False | True | 0.12 | 2.25 | 2.25 |
| 7168 | 1 | 16384 | torch.float16 | False | False | 0.11 | 2.15 | 2.15 |
| 18432 | 1 | 7168 | torch.bfloat16 | False | False | 0.12 | 2.25 | 2.25 |

### torch-cublas

| m | n | k | dtype | trans_a | trans_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1024 | 1024 | 1024 | torch.float16 | False | False | 0.01 | 330.23 | 0.97 |
| 1 | 7168 | 16384 | torch.float16 | False | True | 0.11 | 2.06 | 2.06 |
| 1 | 18432 | 7168 | torch.bfloat16 | False | True | 0.13 | 1.98 | 1.98 |
| 7168 | 1 | 16384 | torch.float16 | False | False | 0.12 | 2.02 | 2.02 |
| 18432 | 1 | 7168 | torch.bfloat16 | False | False | 0.14 | 1.96 | 1.96 |

## GLAFwdOp

### tileops

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.10 | 1.28 | 0.08 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.22 | 1.24 | 0.08 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.48 | 1.13 | 0.07 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.98 | 1.10 | 0.07 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.11 | 1.22 | 0.08 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.21 | 1.25 | 0.08 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.47 | 1.14 | 0.07 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.99 | 1.09 | 0.07 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.33 | 1.21 | 0.07 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.75 | 1.07 | 0.06 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 1.60 | 1.01 | 0.06 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 3.26 | 0.99 | 0.06 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.33 | 1.21 | 0.07 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.75 | 1.07 | 0.06 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 1.63 | 0.99 | 0.06 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 3.38 | 0.95 | 0.05 |

### fla

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.06 | 2.10 | 0.13 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.12 | 2.27 | 0.14 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.23 | 2.37 | 0.15 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.63 | 1.70 | 0.11 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.06 | 2.07 | 0.13 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.12 | 2.24 | 0.14 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.23 | 2.34 | 0.15 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.65 | 1.66 | 0.10 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.22 | 1.81 | 0.10 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.42 | 1.93 | 0.11 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.86 | 1.87 | 0.11 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 2.35 | 1.37 | 0.08 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.22 | 1.81 | 0.10 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.42 | 1.93 | 0.11 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.87 | 1.85 | 0.11 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 2.35 | 1.37 | 0.08 |

## GLABwdOp

### tileops

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | T | H | K | V | BC | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 2048 | 4 | 64 | 64 | 64 | 0.125 | 0.24 | 1.10 | 0.06 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4096 | 4 | 64 | 64 | 64 | 0.125 | 0.54 | 0.99 | 0.05 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 8192 | 4 | 64 | 64 | 64 | 0.125 | 1.12 | 0.96 | 0.05 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 16384 | 4 | 64 | 64 | 64 | 0.125 | 2.28 | 0.94 | 0.05 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 2048 | 4 | 64 | 64 | 64 | 0.125 | 0.24 | 1.10 | 0.06 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4096 | 4 | 64 | 64 | 64 | 0.125 | 0.55 | 0.98 | 0.05 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 8192 | 4 | 64 | 64 | 64 | 0.125 | 1.16 | 0.93 | 0.05 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 16384 | 4 | 64 | 64 | 64 | 0.125 | 2.39 | 0.90 | 0.05 |

### fla

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | T | H | K | V | BC | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 2048 | 4 | 64 | 64 | 64 | 0.125 | 0.16 | 1.73 | 0.09 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4096 | 4 | 64 | 64 | 64 | 0.125 | 0.28 | 1.89 | 0.10 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 8192 | 4 | 64 | 64 | 64 | 0.125 | 0.66 | 1.62 | 0.09 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 16384 | 4 | 64 | 64 | 64 | 0.125 | 1.67 | 1.28 | 0.07 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 2048 | 4 | 64 | 64 | 64 | 0.125 | 0.15 | 1.74 | 0.09 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4096 | 4 | 64 | 64 | 64 | 0.125 | 0.28 | 1.89 | 0.10 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 8192 | 4 | 64 | 64 | 64 | 0.125 | 0.67 | 1.60 | 0.09 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 16384 | 4 | 64 | 64 | 64 | 0.125 | 1.67 | 1.28 | 0.07 |

## GLADecodeOp

### tileops

| batch | heads | dim_k | dim_v | dtype | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 64 | 64 | torch.float32 | 0.125 | 0.01 | 0.07 | 0.15 |
| 1 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.02 | 0.11 | 0.22 |
| 1 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.01 | 0.17 | 0.17 |
| 1 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.01 | 0.17 | 0.17 |
| 8 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.05 | 0.34 | 0.68 |
| 8 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.02 | 1.08 | 1.10 |
| 8 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.02 | 1.07 | 1.08 |
| 16 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.11 | 0.31 | 0.63 |
| 16 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.02 | 1.55 | 1.58 |
| 16 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.02 | 1.50 | 1.53 |
| 32 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.23 | 0.29 | 0.59 |
| 32 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.05 | 1.34 | 1.36 |
| 32 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.05 | 1.33 | 1.35 |
| 64 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.48 | 0.28 | 0.57 |
| 64 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.10 | 1.31 | 1.33 |
| 64 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.10 | 1.28 | 1.30 |

### fla

| batch | heads | dim_k | dim_v | dtype | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 64 | 64 | torch.float32 | 0.125 | 0.01 | 0.08 | 0.16 |
| 1 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.01 | 0.31 | 0.62 |
| 1 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.01 | 0.26 | 0.27 |
| 1 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.01 | 0.25 | 0.25 |
| 8 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.02 | 1.05 | 2.13 |
| 8 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.02 | 1.00 | 1.01 |
| 8 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.02 | 1.00 | 1.01 |
| 16 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.03 | 1.24 | 2.53 |
| 16 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.03 | 1.30 | 1.32 |
| 16 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.03 | 1.29 | 1.31 |
| 32 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.05 | 1.44 | 2.92 |
| 32 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.04 | 1.54 | 1.56 |
| 32 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.04 | 1.53 | 1.56 |
| 64 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.09 | 1.58 | 3.20 |
| 64 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.08 | 1.72 | 1.75 |
| 64 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.08 | 1.71 | 1.74 |

## GroupQueryAttentionFwdOp

### tileops

| batch | seq_len | heads | heads_kv | dim | causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 32 | 8 | 128 | True | torch.float16 | 0.05 | 184.80 | 0.45 |
| 1 | 4096 | 32 | 8 | 128 | True | torch.float16 | 0.81 | 170.07 | 0.10 |
| 1 | 8192 | 32 | 8 | 128 | True | torch.float16 | 3.26 | 168.85 | 0.05 |
| 1 | 32768 | 32 | 8 | 128 | True | torch.float16 | 32.23 | 272.93 | 0.02 |
| 1 | 131072 | 32 | 8 | 128 | True | torch.float16 | 552.62 | 254.67 | 0.00 |
| 1 | 4096 | 64 | 8 | 128 | True | torch.float16 | 0.96 | 286.98 | 0.16 |
| 1 | 4096 | 128 | 8 | 128 | True | torch.float16 | 1.92 | 285.73 | 0.15 |
| 2 | 4096 | 32 | 8 | 128 | True | torch.bfloat16 | 0.93 | 295.60 | 0.18 |
| 1 | 8192 | 32 | 8 | 128 | True | torch.bfloat16 | 1.71 | 321.88 | 0.10 |
| 1 | 4096 | 64 | 8 | 128 | True | torch.bfloat16 | 0.90 | 305.13 | 0.17 |
| 1 | 4096 | 128 | 8 | 128 | True | torch.bfloat16 | 1.78 | 308.19 | 0.16 |
| 2 | 2048 | 32 | 8 | 128 | True | torch.bfloat16 | 0.26 | 260.35 | 0.32 |

### fa3

| batch | seq_len | heads | heads_kv | dim | causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 32 | 8 | 128 | True | torch.float16 | 0.06 | 150.66 | 0.37 |
| 1 | 4096 | 32 | 8 | 128 | True | torch.float16 | 0.86 | 159.52 | 0.10 |
| 1 | 8192 | 32 | 8 | 128 | True | torch.float16 | 3.32 | 165.69 | 0.05 |
| 1 | 32768 | 32 | 8 | 128 | True | torch.float16 | 48.73 | 180.52 | 0.01 |
| 1 | 131072 | 32 | 8 | 128 | True | torch.float16 | 824.49 | 170.70 | 0.00 |
| 1 | 4096 | 64 | 8 | 128 | True | torch.float16 | 1.46 | 188.42 | 0.10 |
| 1 | 4096 | 128 | 8 | 128 | True | torch.float16 | 2.84 | 193.62 | 0.10 |
| 2 | 4096 | 32 | 8 | 128 | True | torch.bfloat16 | 1.44 | 190.59 | 0.12 |
| 1 | 8192 | 32 | 8 | 128 | True | torch.bfloat16 | 2.70 | 203.71 | 0.06 |
| 1 | 4096 | 64 | 8 | 128 | True | torch.bfloat16 | 1.38 | 199.26 | 0.11 |
| 1 | 4096 | 128 | 8 | 128 | True | torch.bfloat16 | 2.63 | 208.66 | 0.11 |
| 2 | 2048 | 32 | 8 | 128 | True | torch.bfloat16 | 0.41 | 165.66 | 0.20 |

### flashinfer

| batch | seq_len | heads | heads_kv | dim | causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 32 | 8 | 128 | True | torch.float16 | 0.04 | 236.73 | 0.58 |
| 1 | 4096 | 32 | 8 | 128 | True | torch.float16 | 0.45 | 304.96 | 0.19 |
| 1 | 8192 | 32 | 8 | 128 | True | torch.float16 | 1.85 | 297.23 | 0.09 |
| 1 | 32768 | 32 | 8 | 128 | True | torch.float16 | 28.40 | 309.76 | 0.02 |
| 1 | 131072 | 32 | 8 | 128 | True | torch.float16 | 465.50 | 302.33 | 0.01 |
| 1 | 4096 | 64 | 8 | 128 | True | torch.float16 | 0.85 | 324.83 | 0.18 |
| 1 | 4096 | 128 | 8 | 128 | True | torch.float16 | 1.74 | 316.11 | 0.16 |
| 2 | 4096 | 32 | 8 | 128 | True | torch.bfloat16 | 0.78 | 351.24 | 0.21 |
| 1 | 8192 | 32 | 8 | 128 | True | torch.bfloat16 | 1.54 | 357.17 | 0.11 |
| 1 | 4096 | 64 | 8 | 128 | True | torch.bfloat16 | 0.80 | 344.12 | 0.19 |
| 1 | 4096 | 128 | 8 | 128 | True | torch.bfloat16 | 1.83 | 300.38 | 0.16 |
| 2 | 2048 | 32 | 8 | 128 | True | torch.bfloat16 | 0.23 | 304.46 | 0.37 |

## GroupQueryAttentionBwdOp

### tileops

| batch | seq_len | heads | heads_kv | dim | causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 32 | 8 | 128 | True | torch.bfloat16 | 3.94 | 174.41 | 0.07 |
| 1 | 8192 | 32 | 8 | 128 | True | torch.bfloat16 | 7.48 | 183.68 | 0.04 |
| 1 | 4096 | 64 | 8 | 128 | True | torch.bfloat16 | 6.68 | 102.94 | 0.04 |
| 1 | 4096 | 128 | 8 | 128 | True | torch.bfloat16 | 8.85 | 155.23 | 0.05 |
| 2 | 2048 | 32 | 8 | 128 | True | torch.bfloat16 | 1.11 | 155.29 | 0.12 |

### fa3

| batch | seq_len | heads | heads_kv | dim | causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 32 | 8 | 128 | True | torch.bfloat16 | 6.30 | 109.04 | 0.04 |
| 1 | 8192 | 32 | 8 | 128 | True | torch.bfloat16 | 11.94 | 115.14 | 0.02 |
| 1 | 4096 | 64 | 8 | 128 | True | torch.bfloat16 | 6.32 | 108.79 | 0.04 |
| 1 | 4096 | 128 | 8 | 128 | True | torch.bfloat16 | 12.33 | 111.42 | 0.04 |
| 2 | 2048 | 32 | 8 | 128 | True | torch.bfloat16 | 1.73 | 99.49 | 0.08 |

## GroupQueryAttentionDecodeWithKVCacheOp

### tileops

| batch | heads | heads_kv | seq_len_kv | dim | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 8 | 4096 | 128 | torch.float16 | 0.01 | 5.04 | 1.26 |
| 1 | 32 | 8 | 32768 | 128 | torch.float16 | 0.06 | 9.65 | 2.41 |
| 1 | 64 | 8 | 4096 | 128 | torch.float16 | 0.02 | 6.76 | 0.85 |
| 1 | 128 | 8 | 8192 | 128 | torch.float16 | 0.02 | 28.34 | 1.77 |

### fa3

| batch | heads | heads_kv | seq_len_kv | dim | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 8 | 4096 | 128 | torch.float16 | 0.02 | 3.24 | 0.81 |
| 1 | 32 | 8 | 32768 | 128 | torch.float16 | 0.05 | 9.83 | 2.46 |
| 1 | 64 | 8 | 4096 | 128 | torch.float16 | 0.02 | 6.53 | 0.82 |
| 1 | 128 | 8 | 8192 | 128 | torch.float16 | 0.03 | 20.37 | 1.28 |

### flashinfer

| batch | heads | heads_kv | seq_len_kv | dim | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 8 | 4096 | 128 | torch.float16 | 0.02 | 3.31 | 0.83 |
| 1 | 32 | 8 | 32768 | 128 | torch.float16 | 0.08 | 6.54 | 1.64 |
| 1 | 64 | 8 | 4096 | 128 | torch.float16 | 0.03 | 5.02 | 0.63 |

## GroupQueryAttentionDecodePagedWithKVCacheOp

### tileops

| batch | heads | heads_kv | seqlen_kv | dim | page_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 32 | 8 | 4096 | 128 | 64 | torch.float16 | 0.08 | 28.41 | 0.23 |
| 8 | 32 | 8 | 32768 | 128 | 64 | torch.float16 | 0.18 | 23.71 | 0.74 |
| 64 | 32 | 8 | 2048 | 128 | 64 | torch.float16 | 0.09 | 23.26 | 0.10 |
| 8 | 64 | 8 | 4096 | 128 | 64 | torch.float16 | 0.04 | 26.39 | 0.42 |
| 32 | 32 | 8 | 4096 | 128 | 256 | torch.float16 | 0.09 | 23.32 | 0.19 |
| 8 | 64 | 8 | 4096 | 128 | 256 | torch.float16 | 0.04 | 29.98 | 0.48 |
| 4 | 128 | 8 | 4096 | 128 | 256 | torch.float16 | 0.03 | 35.42 | 0.56 |

### flashinfer

| batch | heads | heads_kv | seqlen_kv | dim | page_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 32 | 8 | 4096 | 128 | 64 | torch.float16 | 0.18 | 12.11 | 0.10 |
| 8 | 32 | 8 | 32768 | 128 | 64 | torch.float16 | 0.44 | 9.81 | 0.31 |
| 64 | 32 | 8 | 2048 | 128 | 64 | torch.float16 | 0.22 | 9.81 | 0.04 |
| 8 | 64 | 8 | 4096 | 128 | 64 | torch.float16 | 0.05 | 21.14 | 0.34 |
| 32 | 32 | 8 | 4096 | 128 | 256 | torch.float16 | 0.18 | 12.13 | 0.10 |
| 8 | 64 | 8 | 4096 | 128 | 256 | torch.float16 | 0.07 | 15.53 | 0.25 |

### fa3

| batch | heads | heads_kv | seqlen_kv | dim | page_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 32 | 8 | 4096 | 128 | 256 | torch.float16 | 0.17 | 12.47 | 0.10 |
| 8 | 64 | 8 | 4096 | 128 | 256 | torch.float16 | 0.04 | 25.13 | 0.40 |
| 4 | 128 | 8 | 4096 | 128 | 256 | torch.float16 | 0.03 | 38.44 | 0.61 |

## GqaSlidingWindowFwdOp

### tileops

| batch | seq | heads | heads_kv | dim | is_causal | wl | wr | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 512 | 8 | 2 | 64 | True | -1 | -1 | torch.float16 | 0.01 | 74.36 | 0.36 |
| 2 | 512 | 8 | 2 | 64 | True | 128 | -1 | torch.float16 | 0.01 | 44.17 | 0.49 |
| 2 | 768 | 8 | 2 | 64 | False | 256 | -1 | torch.float16 | 0.01 | 152.06 | 0.32 |
| 2 | 512 | 8 | 2 | 64 | False | 64 | 64 | torch.bfloat16 | 0.01 | 46.69 | 0.48 |
| 1 | 2048 | 8 | 2 | 64 | True | 512 | -1 | torch.float16 | 0.01 | 150.29 | 0.42 |

### fa3

| batch | seq | heads | heads_kv | dim | is_causal | wl | wr | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 512 | 8 | 2 | 64 | True | -1 | -1 | torch.float16 | 0.01 | 38.33 | 0.19 |
| 2 | 512 | 8 | 2 | 64 | True | 128 | -1 | torch.float16 | 0.01 | 16.07 | 0.18 |
| 2 | 768 | 8 | 2 | 64 | False | 256 | -1 | torch.float16 | 0.03 | 67.99 | 0.14 |
| 2 | 512 | 8 | 2 | 64 | False | 64 | 64 | torch.bfloat16 | 0.01 | 17.16 | 0.18 |
| 1 | 2048 | 8 | 2 | 64 | True | 512 | -1 | torch.float16 | 0.02 | 79.72 | 0.22 |

### flashinfer

| batch | seq | heads | heads_kv | dim | is_causal | wl | wr | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 512 | 8 | 2 | 64 | True | -1 | -1 | torch.float16 | 0.01 | 44.14 | 0.22 |
| 2 | 512 | 8 | 2 | 64 | True | 128 | -1 | torch.float16 | 0.01 | 19.40 | 0.21 |
| 2 | 768 | 8 | 2 | 64 | False | 256 | -1 | torch.float16 | 0.02 | 113.26 | 0.24 |
| 1 | 2048 | 8 | 2 | 64 | True | 512 | -1 | torch.float16 | 0.02 | 90.50 | 0.25 |

## GqaSlidingWindowVarlenFwdOp

### tileops

| batch | heads | heads_kv | dim | is_causal | wl | wr | dtype | max_seqlen_q | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 8 | 128 | True | -1 | -1 | torch.float16 | 3000 | 0.22 | 338.27 | 0.28 |
| 2 | 32 | 8 | 128 | True | 2000 | -1 | torch.float16 | 3073 | 0.35 | 251.79 | 0.27 |
| 4 | 32 | 8 | 128 | False | 1500 | 1500 | torch.float16 | 3500 | 0.94 | 366.35 | 0.22 |
| 2 | 32 | 8 | 128 | True | -1 | -1 | torch.float16 | 1024 | 0.40 | 398.58 | 0.19 |
| 2 | 32 | 8 | 128 | True | -1 | -1 | torch.float16 | 8193 | 2.03 | 371.22 | 0.13 |

### fa3

| batch | heads | heads_kv | dim | is_causal | wl | wr | dtype | max_seqlen_q | max_seqlen_k | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 8 | 128 | True | -1 | -1 | torch.float16 | 3000 | 3000 | 0.44 | 167.02 | 0.14 |
| 2 | 32 | 8 | 128 | True | 2000 | -1 | torch.float16 | 3073 | 3073 | 0.63 | 138.64 | 0.15 |
| 4 | 32 | 8 | 128 | False | 1500 | 1500 | torch.float16 | 3500 | 3500 | 1.91 | 180.51 | 0.11 |
| 2 | 32 | 8 | 128 | True | -1 | -1 | torch.float16 | 1024 | 8192 | 0.91 | 177.01 | 0.08 |
| 2 | 32 | 8 | 128 | True | -1 | -1 | torch.float16 | 8193 | 8193 | 3.91 | 192.84 | 0.07 |

### flashinfer

| batch | heads | heads_kv | dim | is_causal | wl | wr | dtype | max_seqlen_q | max_seqlen_k | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 8 | 128 | True | -1 | -1 | torch.float16 | 3000 | 3000 | 0.23 | 320.31 | 0.27 |
| 2 | 32 | 8 | 128 | True | 2000 | -1 | torch.float16 | 3073 | 3073 | 0.29 | 300.38 | 0.32 |
| 2 | 32 | 8 | 128 | True | -1 | -1 | torch.float16 | 1024 | 8192 | 0.44 | 364.04 | 0.17 |
| 2 | 32 | 8 | 128 | True | -1 | -1 | torch.float16 | 8193 | 8193 | 2.24 | 336.40 | 0.12 |

## GroupNormOp

### tileops

| n | c | g | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 128 | 32 | torch.float16 | 0.01 | 0.36 | 0.29 |
| 8 | 128 | 32 | torch.bfloat16 | 0.01 | 0.36 | 0.29 |
| 4 | 256 | 32 | torch.float16 | 0.02 | 0.19 | 0.15 |
| 4 | 128 | 16 | torch.float16 | 0.02 | 0.12 | 0.10 |

### torch

| n | c | g | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 128 | 32 | torch.float16 | 0.01 | 0.36 | 0.28 |
| 8 | 128 | 32 | torch.bfloat16 | 0.01 | 0.35 | 0.28 |
| 4 | 256 | 32 | torch.float16 | 0.02 | 0.25 | 0.20 |
| 4 | 128 | 16 | torch.float16 | 0.02 | 0.15 | 0.12 |

## grouped_gemm_nt

### tileops

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_nt | 16384 | 4 | 4864 | 4096 | torch.float16 | False | True | 49.87 | 13.09 | 0.01 |

### torch-ref

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_nt | 16384 | 4 | 4864 | 4096 | torch.float16 | False | True | 1.59 | 410.80 | 0.29 |

## grouped_gemm_nn

### tileops

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_nn | 16384 | 4 | 4864 | 4096 | torch.float16 | False | False | 53.66 | 12.17 | 0.01 |

### torch-ref

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_nn | 16384 | 4 | 4864 | 4096 | torch.float16 | False | False | 1.08 | 603.09 | 0.42 |

## grouped_gemm_tn

### tileops

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_tn | 16384 | 4 | 4864 | 4096 | torch.float16 | True | False | 11.79 | 55.38 | 0.04 |

### torch-ref

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_tn | 16384 | 4 | 4864 | 4096 | torch.float16 | True | False | 1.06 | 616.95 | 0.43 |

## grouped_gemm_tt

### tileops

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_tt | 16384 | 4 | 4864 | 4096 | torch.float16 | True | True | 14.79 | 44.15 | 0.03 |

### torch-ref

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_tt | 16384 | 4 | 4864 | 4096 | torch.float16 | True | True | 1.06 | 615.07 | 0.43 |

## GroupedGemmOp

### tileops

| batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16384 | 4 | 4864 | 4096 | torch.float16 | True | False | 130.22 | 15.04 | N/A |

### torch-ref

| batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16384 | 4 | 4864 | 4096 | torch.float16 | True | False | 4.82 | 406.10 | N/A |

## leaky_relu

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| leaky_relu | torch.float16 | 4194304 | 0.01 | 0.66 | 2.64 |
| leaky_relu | torch.bfloat16 | 4194304 | 0.01 | 0.66 | 2.63 |
| leaky_relu | torch.float32 | 4194304 | 0.01 | 0.40 | 3.17 |
| leaky_relu | torch.float16 | 10485760 | 0.01 | 0.82 | 3.29 |
| leaky_relu | torch.bfloat16 | 10485760 | 0.01 | 0.82 | 3.29 |
| leaky_relu | torch.float32 | 10485760 | 0.02 | 0.47 | 3.73 |
| leaky_relu | torch.float16 | 20971520 | 0.02 | 0.93 | 3.73 |
| leaky_relu | torch.bfloat16 | 20971520 | 0.02 | 0.93 | 3.73 |
| leaky_relu | torch.float32 | 20971520 | 0.06 | 0.38 | 3.02 |

### torch

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| leaky_relu | torch.float16 | 4194304 | 0.01 | 0.46 | 1.84 |
| leaky_relu | torch.bfloat16 | 4194304 | 0.01 | 0.52 | 2.06 |
| leaky_relu | torch.float32 | 4194304 | 0.01 | 0.29 | 2.31 |
| leaky_relu | torch.float16 | 10485760 | 0.02 | 0.58 | 2.31 |
| leaky_relu | torch.bfloat16 | 10485760 | 0.02 | 0.58 | 2.30 |
| leaky_relu | torch.float32 | 10485760 | 0.04 | 0.30 | 2.39 |
| leaky_relu | torch.float16 | 20971520 | 0.03 | 0.60 | 2.40 |
| leaky_relu | torch.bfloat16 | 20971520 | 0.03 | 0.60 | 2.40 |
| leaky_relu | torch.float32 | 20971520 | 0.07 | 0.30 | 2.36 |

## elu

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| elu | torch.float16 | 4194304 | 0.01 | 0.61 | 2.44 |
| elu | torch.bfloat16 | 4194304 | 0.01 | 0.61 | 2.44 |
| elu | torch.float32 | 4194304 | 0.01 | 0.39 | 3.11 |
| elu | torch.float16 | 10485760 | 0.01 | 0.77 | 3.07 |
| elu | torch.bfloat16 | 10485760 | 0.01 | 0.77 | 3.07 |
| elu | torch.float32 | 10485760 | 0.02 | 0.46 | 3.66 |
| elu | torch.float16 | 20971520 | 0.03 | 0.81 | 3.25 |
| elu | torch.bfloat16 | 20971520 | 0.03 | 0.81 | 3.26 |
| elu | torch.float32 | 20971520 | 0.06 | 0.37 | 2.97 |

### torch

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| elu | torch.float16 | 4194304 | 0.01 | 0.31 | 1.25 |
| elu | torch.bfloat16 | 4194304 | 0.01 | 0.31 | 1.23 |
| elu | torch.float32 | 4194304 | 0.02 | 0.27 | 2.15 |
| elu | torch.float16 | 10485760 | 0.03 | 0.34 | 1.36 |
| elu | torch.bfloat16 | 10485760 | 0.03 | 0.33 | 1.33 |
| elu | torch.float32 | 10485760 | 0.04 | 0.28 | 2.27 |
| elu | torch.float16 | 20971520 | 0.06 | 0.34 | 1.35 |
| elu | torch.bfloat16 | 20971520 | 0.06 | 0.33 | 1.33 |
| elu | torch.float32 | 20971520 | 0.06 | 0.33 | 2.64 |

## hardtanh

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| hardtanh | torch.float16 | 4194304 | 0.01 | 0.66 | 2.62 |
| hardtanh | torch.bfloat16 | 4194304 | 0.01 | 0.65 | 2.62 |
| hardtanh | torch.float32 | 4194304 | 0.01 | 0.39 | 3.16 |
| hardtanh | torch.float16 | 10485760 | 0.01 | 0.82 | 3.29 |
| hardtanh | torch.bfloat16 | 10485760 | 0.01 | 0.82 | 3.29 |
| hardtanh | torch.float32 | 10485760 | 0.02 | 0.46 | 3.72 |
| hardtanh | torch.float16 | 20971520 | 0.02 | 0.93 | 3.72 |
| hardtanh | torch.bfloat16 | 20971520 | 0.02 | 0.93 | 3.72 |
| hardtanh | torch.float32 | 20971520 | 0.06 | 0.38 | 3.02 |

### torch

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| hardtanh | torch.float16 | 4194304 | 0.01 | 0.54 | 2.17 |
| hardtanh | torch.bfloat16 | 4194304 | 0.01 | 0.55 | 2.22 |
| hardtanh | torch.float32 | 4194304 | 0.01 | 0.31 | 2.52 |
| hardtanh | torch.float16 | 10485760 | 0.02 | 0.58 | 2.34 |
| hardtanh | torch.bfloat16 | 10485760 | 0.02 | 0.62 | 2.47 |
| hardtanh | torch.float32 | 10485760 | 0.03 | 0.31 | 2.48 |
| hardtanh | torch.float16 | 20971520 | 0.04 | 0.57 | 2.29 |
| hardtanh | torch.bfloat16 | 20971520 | 0.03 | 0.62 | 2.47 |
| hardtanh | torch.float32 | 20971520 | 0.06 | 0.33 | 2.67 |

## softplus

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| softplus | torch.float16 | 4194304 | 0.01 | 0.42 | 1.70 |
| softplus | torch.bfloat16 | 4194304 | 0.01 | 0.42 | 1.67 |
| softplus | torch.float32 | 4194304 | 0.01 | 0.37 | 2.98 |
| softplus | torch.float16 | 10485760 | 0.02 | 0.51 | 2.06 |
| softplus | torch.bfloat16 | 10485760 | 0.02 | 0.51 | 2.02 |
| softplus | torch.float32 | 10485760 | 0.03 | 0.41 | 3.29 |
| softplus | torch.float16 | 20971520 | 0.05 | 0.44 | 1.76 |
| softplus | torch.bfloat16 | 20971520 | 0.05 | 0.43 | 1.71 |
| softplus | torch.float32 | 20971520 | 0.06 | 0.33 | 2.66 |

### torch

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| softplus | torch.float16 | 4194304 | 0.02 | 0.26 | 1.02 |
| softplus | torch.bfloat16 | 4194304 | 0.02 | 0.25 | 1.02 |
| softplus | torch.float32 | 4194304 | 0.02 | 0.23 | 1.87 |
| softplus | torch.float16 | 10485760 | 0.04 | 0.27 | 1.08 |
| softplus | torch.bfloat16 | 10485760 | 0.04 | 0.27 | 1.08 |
| softplus | torch.float32 | 10485760 | 0.04 | 0.26 | 2.07 |
| softplus | torch.float16 | 20971520 | 0.08 | 0.27 | 1.07 |
| softplus | torch.bfloat16 | 20971520 | 0.08 | 0.26 | 1.05 |
| softplus | torch.float32 | 20971520 | 0.07 | 0.30 | 2.38 |

## clamp

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| clamp | torch.float16 | 4194304 | 0.01 | 0.66 | 2.63 |
| clamp | torch.bfloat16 | 4194304 | 0.01 | 0.66 | 2.63 |
| clamp | torch.float32 | 4194304 | 0.01 | 0.40 | 3.16 |
| clamp | torch.float16 | 10485760 | 0.01 | 0.82 | 3.28 |
| clamp | torch.bfloat16 | 10485760 | 0.01 | 0.82 | 3.30 |
| clamp | torch.float32 | 10485760 | 0.02 | 0.46 | 3.71 |
| clamp | torch.float16 | 20971520 | 0.02 | 0.93 | 3.71 |
| clamp | torch.bfloat16 | 20971520 | 0.02 | 0.93 | 3.72 |
| clamp | torch.float32 | 20971520 | 0.05 | 0.38 | 3.06 |

### torch

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| clamp | torch.float16 | 4194304 | 0.01 | 0.50 | 1.99 |
| clamp | torch.bfloat16 | 4194304 | 0.01 | 0.51 | 2.04 |
| clamp | torch.float32 | 4194304 | 0.01 | 0.29 | 2.33 |
| clamp | torch.float16 | 10485760 | 0.02 | 0.55 | 2.20 |
| clamp | torch.bfloat16 | 10485760 | 0.02 | 0.58 | 2.33 |
| clamp | torch.float32 | 10485760 | 0.04 | 0.30 | 2.39 |
| clamp | torch.float16 | 20971520 | 0.04 | 0.56 | 2.23 |
| clamp | torch.bfloat16 | 20971520 | 0.04 | 0.60 | 2.39 |
| clamp | torch.float32 | 20971520 | 0.06 | 0.33 | 2.67 |

## nan_to_num

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| nan_to_num | torch.float16 | 4194304 | 0.01 | 0.64 | 2.57 |
| nan_to_num | torch.bfloat16 | 4194304 | 0.01 | 0.64 | 2.57 |
| nan_to_num | torch.float32 | 4194304 | 0.01 | 0.39 | 3.15 |
| nan_to_num | torch.float16 | 10485760 | 0.01 | 0.81 | 3.25 |
| nan_to_num | torch.bfloat16 | 10485760 | 0.01 | 0.81 | 3.25 |
| nan_to_num | torch.float32 | 10485760 | 0.02 | 0.46 | 3.72 |
| nan_to_num | torch.float16 | 20971520 | 0.02 | 0.91 | 3.64 |
| nan_to_num | torch.bfloat16 | 20971520 | 0.02 | 0.91 | 3.65 |
| nan_to_num | torch.float32 | 20971520 | 0.06 | 0.38 | 3.02 |

### torch

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| nan_to_num | torch.float16 | 4194304 | 0.01 | 0.52 | 2.08 |
| nan_to_num | torch.bfloat16 | 4194304 | 0.01 | 0.52 | 2.08 |
| nan_to_num | torch.float32 | 4194304 | 0.01 | 0.30 | 2.38 |
| nan_to_num | torch.float16 | 10485760 | 0.02 | 0.59 | 2.35 |
| nan_to_num | torch.bfloat16 | 10485760 | 0.02 | 0.59 | 2.36 |
| nan_to_num | torch.float32 | 10485760 | 0.03 | 0.30 | 2.41 |
| nan_to_num | torch.float16 | 20971520 | 0.03 | 0.60 | 2.40 |
| nan_to_num | torch.bfloat16 | 20971520 | 0.04 | 0.60 | 2.39 |
| nan_to_num | torch.float32 | 20971520 | 0.07 | 0.30 | 2.39 |

## PreluOp

### tileops

| num_channels | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 128 | torch.float16 | 131072 | 0.00 | 0.07 | 0.27 |
| 128 | torch.bfloat16 | 131072 | 0.00 | 0.07 | 0.27 |
| 128 | torch.float32 | 131072 | 0.00 | 0.06 | 0.49 |
| 4096 | torch.float16 | 4194304 | 0.01 | 0.68 | 2.70 |
| 4096 | torch.bfloat16 | 4194304 | 0.01 | 0.68 | 2.72 |
| 4096 | torch.float32 | 4194304 | 0.01 | 0.40 | 3.24 |
| 10240 | torch.float16 | 10485760 | 0.01 | 0.84 | 3.34 |
| 10240 | torch.bfloat16 | 10485760 | 0.01 | 0.83 | 3.34 |
| 10240 | torch.float32 | 10485760 | 0.02 | 0.47 | 3.74 |
| 20480 | torch.float16 | 20971520 | 0.02 | 0.93 | 3.73 |
| 20480 | torch.bfloat16 | 20971520 | 0.02 | 0.93 | 3.73 |
| 20480 | torch.float32 | 20971520 | 0.06 | 0.38 | 3.04 |

### torch

| num_channels | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 128 | torch.float16 | 131072 | 0.01 | 0.03 | 0.10 |
| 128 | torch.bfloat16 | 131072 | 0.01 | 0.03 | 0.10 |
| 128 | torch.float32 | 131072 | 0.00 | 0.03 | 0.28 |
| 4096 | torch.float16 | 4194304 | 0.02 | 0.21 | 0.84 |
| 4096 | torch.bfloat16 | 4194304 | 0.02 | 0.20 | 0.82 |
| 4096 | torch.float32 | 4194304 | 0.02 | 0.18 | 1.43 |
| 10240 | torch.float16 | 10485760 | 0.05 | 0.21 | 0.85 |
| 10240 | torch.bfloat16 | 10485760 | 0.05 | 0.21 | 0.84 |
| 10240 | torch.float32 | 10485760 | 0.06 | 0.17 | 1.40 |
| 20480 | torch.float16 | 20971520 | 0.10 | 0.21 | 0.83 |
| 20480 | torch.bfloat16 | 20971520 | 0.10 | 0.21 | 0.82 |
| 20480 | torch.float32 | 20971520 | 0.10 | 0.20 | 1.62 |

## WhereOp

### tileops

| dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| torch.float16 | 4194304 | 0.01 | 0.42 | 2.91 |
| torch.bfloat16 | 4194304 | 0.01 | 0.41 | 2.90 |
| torch.float32 | 4194304 | 0.02 | 0.25 | 3.29 |
| torch.float16 | 10485760 | 0.02 | 0.49 | 3.46 |
| torch.bfloat16 | 10485760 | 0.02 | 0.50 | 3.47 |
| torch.float32 | 10485760 | 0.04 | 0.26 | 3.34 |
| torch.float16 | 20971520 | 0.05 | 0.46 | 3.24 |
| torch.bfloat16 | 20971520 | 0.05 | 0.46 | 3.23 |
| torch.float32 | 20971520 | 0.10 | 0.21 | 2.79 |
| torch.float8_e4m3fn | 4194304 | 0.01 | 0.52 | 2.08 |
| torch.float8_e5m2 | 4194304 | 0.01 | 0.52 | 2.09 |
| torch.float8_e4m3fn | 10485760 | 0.02 | 0.59 | 2.36 |
| torch.float8_e5m2 | 10485760 | 0.02 | 0.59 | 2.36 |
| torch.float8_e4m3fn | 20971520 | 0.03 | 0.61 | 2.43 |
| torch.float8_e5m2 | 20971520 | 0.03 | 0.60 | 2.41 |

### torch

| dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| torch.float16 | 4194304 | 0.01 | 0.32 | 2.24 |
| torch.bfloat16 | 4194304 | 0.01 | 0.32 | 2.24 |
| torch.float32 | 4194304 | 0.02 | 0.18 | 2.36 |
| torch.float16 | 10485760 | 0.03 | 0.34 | 2.35 |
| torch.bfloat16 | 10485760 | 0.03 | 0.34 | 2.35 |
| torch.float32 | 10485760 | 0.05 | 0.20 | 2.59 |
| torch.float16 | 20971520 | 0.06 | 0.34 | 2.35 |
| torch.bfloat16 | 20971520 | 0.06 | 0.37 | 2.61 |
| torch.float32 | 20971520 | 0.10 | 0.21 | 2.76 |
| torch.float8_e4m3fn | 4194304 | 0.01 | 0.48 | 1.90 |
| torch.float8_e5m2 | 4194304 | 0.01 | 0.48 | 1.90 |
| torch.float8_e4m3fn | 10485760 | 0.02 | 0.54 | 2.14 |
| torch.float8_e5m2 | 10485760 | 0.02 | 0.54 | 2.15 |
| torch.float8_e4m3fn | 20971520 | 0.04 | 0.56 | 2.24 |
| torch.float8_e5m2 | 20971520 | 0.04 | 0.56 | 2.24 |

## MaskedFillOp

### tileops

| dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| torch.float16 | 4194304 | 0.01 | 0.55 | 2.77 |
| torch.bfloat16 | 4194304 | 0.01 | 0.55 | 2.77 |
| torch.float32 | 4194304 | 0.01 | 0.35 | 3.19 |
| torch.float16 | 10485760 | 0.02 | 0.67 | 3.35 |
| torch.bfloat16 | 10485760 | 0.02 | 0.67 | 3.35 |
| torch.float32 | 10485760 | 0.03 | 0.41 | 3.71 |
| torch.float16 | 20971520 | 0.03 | 0.72 | 3.59 |
| torch.bfloat16 | 20971520 | 0.03 | 0.73 | 3.67 |
| torch.float32 | 20971520 | 0.06 | 0.33 | 3.01 |
| torch.float8_e4m3fn | 4194304 | 0.01 | 0.58 | 1.74 |
| torch.float8_e5m2 | 4194304 | 0.01 | 0.32 | 0.97 |
| torch.float8_e4m3fn | 10485760 | 0.01 | 0.75 | 2.24 |
| torch.float8_e5m2 | 10485760 | 0.03 | 0.40 | 1.20 |
| torch.float8_e4m3fn | 20971520 | 0.03 | 0.81 | 2.44 |
| torch.float8_e5m2 | 20971520 | 0.06 | 0.35 | 1.04 |

### torch

| dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| torch.float16 | 4194304 | 0.01 | 0.31 | 1.55 |
| torch.bfloat16 | 4194304 | 0.01 | 0.31 | 1.54 |
| torch.float32 | 4194304 | 0.02 | 0.17 | 1.54 |
| torch.float16 | 10485760 | 0.03 | 0.32 | 1.60 |
| torch.bfloat16 | 10485760 | 0.03 | 0.32 | 1.60 |
| torch.float32 | 10485760 | 0.07 | 0.14 | 1.28 |
| torch.float16 | 20971520 | 0.08 | 0.27 | 1.33 |
| torch.bfloat16 | 20971520 | 0.08 | 0.27 | 1.33 |
| torch.float32 | 20971520 | 0.14 | 0.15 | 1.32 |

### torch-ref

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| masked_fill | torch.float8_e4m3fn | 4194304 | 0.03 | 0.13 | 0.39 |
| masked_fill | torch.float8_e5m2 | 4194304 | 0.03 | 0.14 | 0.41 |
| masked_fill | torch.float8_e4m3fn | 10485760 | 0.09 | 0.11 | 0.33 |
| masked_fill | torch.float8_e5m2 | 10485760 | 0.09 | 0.11 | 0.34 |
| masked_fill | torch.float8_e4m3fn | 20971520 | 0.23 | 0.09 | 0.27 |
| masked_fill | torch.float8_e5m2 | 20971520 | 0.23 | 0.09 | 0.28 |

## alibi

### tileops

| op_name | seq_len | dim | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| alibi | 512 | 64 | torch.float16 | 0.04 | 0.41 | 0.82 |
| alibi | 512 | 64 | torch.bfloat16 | 0.04 | 0.41 | 0.83 |
| alibi | 512 | 64 | torch.float32 | 0.02 | 0.87 | 3.48 |
| alibi | 2048 | 64 | torch.float16 | 0.43 | 0.63 | 1.26 |
| alibi | 2048 | 64 | torch.bfloat16 | 0.43 | 0.63 | 1.26 |
| alibi | 2048 | 64 | torch.float32 | 0.42 | 0.64 | 2.58 |
| alibi | 4096 | 128 | torch.float16 | 1.60 | 1.35 | 2.69 |
| alibi | 4096 | 128 | torch.bfloat16 | 1.64 | 1.31 | 2.61 |
| alibi | 4096 | 128 | torch.float32 | 3.15 | 0.68 | 2.72 |

### torch-ref

| op_name | seq_len | dim | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| alibi | 512 | 64 | torch.float16 | 0.08 | 0.20 | 0.41 |
| alibi | 512 | 64 | torch.bfloat16 | 0.08 | 0.21 | 0.41 |
| alibi | 512 | 64 | torch.float32 | 0.06 | 0.30 | 1.21 |
| alibi | 2048 | 64 | torch.float16 | 1.95 | 0.14 | 0.27 |
| alibi | 2048 | 64 | torch.bfloat16 | 1.95 | 0.14 | 0.27 |
| alibi | 2048 | 64 | torch.float32 | 1.23 | 0.22 | 0.87 |
| alibi | 4096 | 128 | torch.float16 | 18.52 | 0.12 | 0.23 |
| alibi | 4096 | 128 | torch.bfloat16 | 18.52 | 0.12 | 0.23 |
| alibi | 4096 | 128 | torch.float32 | 12.80 | 0.17 | 0.67 |

## sinusoidal

### tileops

| op_name | seq_len | dim | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| sinusoidal | 512 | 256 | torch.float16 | 0.00 | 0.05 | 0.10 |
| sinusoidal | 512 | 256 | torch.bfloat16 | 0.00 | 0.05 | 0.10 |
| sinusoidal | 512 | 256 | torch.float32 | 0.00 | 0.06 | 0.23 |
| sinusoidal | 2048 | 300 | torch.float16 | 0.00 | 0.15 | 0.31 |
| sinusoidal | 2048 | 300 | torch.bfloat16 | 0.00 | 0.15 | 0.31 |
| sinusoidal | 2048 | 300 | torch.float32 | 0.00 | 0.14 | 0.55 |
| sinusoidal | 4096 | 512 | torch.float16 | 0.01 | 0.32 | 0.63 |
| sinusoidal | 4096 | 512 | torch.bfloat16 | 0.01 | 0.32 | 0.64 |
| sinusoidal | 4096 | 512 | torch.float32 | 0.01 | 0.19 | 0.76 |

### torch-ref

| op_name | seq_len | dim | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| sinusoidal | 512 | 256 | torch.float16 | 0.02 | 0.01 | 0.01 |
| sinusoidal | 512 | 256 | torch.bfloat16 | 0.02 | 0.01 | 0.01 |
| sinusoidal | 512 | 256 | torch.float32 | 0.02 | 0.01 | 0.03 |
| sinusoidal | 2048 | 300 | torch.float16 | 0.02 | 0.03 | 0.05 |
| sinusoidal | 2048 | 300 | torch.bfloat16 | 0.02 | 0.03 | 0.05 |
| sinusoidal | 2048 | 300 | torch.float32 | 0.02 | 0.03 | 0.12 |
| sinusoidal | 4096 | 512 | torch.float16 | 0.03 | 0.07 | 0.13 |
| sinusoidal | 4096 | 512 | torch.bfloat16 | 0.03 | 0.07 | 0.13 |
| sinusoidal | 4096 | 512 | torch.float32 | 0.03 | 0.07 | 0.30 |

## leaky_relu_fp8

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| leaky_relu | torch.float8_e4m3fn | 4194304 | 0.01 | 0.49 | 0.98 |
| leaky_relu | torch.float8_e5m2 | 4194304 | 0.03 | 0.16 | 0.32 |
| leaky_relu | torch.float8_e4m3fn | 10485760 | 0.02 | 0.56 | 1.13 |
| leaky_relu | torch.float8_e5m2 | 10485760 | 0.06 | 0.16 | 0.33 |
| leaky_relu | torch.float8_e4m3fn | 20971520 | 0.04 | 0.50 | 1.00 |
| leaky_relu | torch.float8_e5m2 | 20971520 | 0.15 | 0.14 | 0.28 |

### torch-ref

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| leaky_relu | torch.float8_e4m3fn | 4194304 | 0.03 | 0.15 | 0.30 |
| leaky_relu | torch.float8_e5m2 | 4194304 | 0.03 | 0.16 | 0.31 |
| leaky_relu | torch.float8_e4m3fn | 10485760 | 0.08 | 0.13 | 0.26 |
| leaky_relu | torch.float8_e5m2 | 10485760 | 0.08 | 0.14 | 0.27 |
| leaky_relu | torch.float8_e4m3fn | 20971520 | 0.19 | 0.11 | 0.22 |
| leaky_relu | torch.float8_e5m2 | 20971520 | 0.18 | 0.11 | 0.23 |

## elu_fp8

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| elu | torch.float8_e4m3fn | 4194304 | 0.01 | 0.68 | 1.36 |
| elu | torch.float8_e5m2 | 4194304 | 0.02 | 0.26 | 0.52 |
| elu | torch.float8_e4m3fn | 10485760 | 0.01 | 0.85 | 1.70 |
| elu | torch.float8_e5m2 | 10485760 | 0.03 | 0.31 | 0.62 |
| elu | torch.float8_e4m3fn | 20971520 | 0.02 | 0.95 | 1.90 |
| elu | torch.float8_e5m2 | 20971520 | 0.09 | 0.23 | 0.47 |

### torch-ref

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| elu | torch.float8_e4m3fn | 4194304 | 0.04 | 0.12 | 0.24 |
| elu | torch.float8_e5m2 | 4194304 | 0.03 | 0.12 | 0.25 |
| elu | torch.float8_e4m3fn | 10485760 | 0.10 | 0.11 | 0.21 |
| elu | torch.float8_e5m2 | 10485760 | 0.09 | 0.11 | 0.22 |
| elu | torch.float8_e4m3fn | 20971520 | 0.22 | 0.10 | 0.19 |
| elu | torch.float8_e5m2 | 20971520 | 0.21 | 0.10 | 0.20 |

## clamp_fp8

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| clamp | torch.float8_e4m3fn | 4194304 | 0.00 | 1.00 | 2.00 |
| clamp | torch.float8_e5m2 | 4194304 | 0.01 | 0.40 | 0.80 |
| clamp | torch.float8_e4m3fn | 10485760 | 0.01 | 1.34 | 2.68 |
| clamp | torch.float8_e5m2 | 10485760 | 0.02 | 0.50 | 0.99 |
| clamp | torch.float8_e4m3fn | 20971520 | 0.01 | 1.57 | 3.14 |
| clamp | torch.float8_e5m2 | 20971520 | 0.05 | 0.46 | 0.91 |

### torch-ref

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| clamp | torch.float8_e4m3fn | 4194304 | 0.03 | 0.14 | 0.28 |
| clamp | torch.float8_e5m2 | 4194304 | 0.03 | 0.15 | 0.30 |
| clamp | torch.float8_e4m3fn | 10485760 | 0.08 | 0.13 | 0.26 |
| clamp | torch.float8_e5m2 | 10485760 | 0.08 | 0.13 | 0.27 |
| clamp | torch.float8_e4m3fn | 20971520 | 0.19 | 0.11 | 0.22 |
| clamp | torch.float8_e5m2 | 20971520 | 0.19 | 0.11 | 0.22 |

## InstanceNormOp

### tileops

| n | c | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 8 | 128 | torch.float16 | 0.02 | 0.35 | 0.28 |
| 8 | 128 | torch.bfloat16 | 0.01 | 0.35 | 0.28 |
| 4 | 256 | torch.float16 | 0.02 | 0.22 | 0.18 |
| 4 | 64 | torch.float16 | 0.01 | 0.08 | 0.07 |

### torch

| n | c | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 8 | 128 | torch.float16 | 0.02 | 0.26 | 0.21 |
| 8 | 128 | torch.bfloat16 | 0.02 | 0.26 | 0.21 |
| 4 | 256 | torch.float16 | 0.02 | 0.20 | 0.16 |
| 4 | 64 | torch.float16 | 0.01 | 0.09 | 0.07 |

## LayerNormOp

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 2048 | 4096 | torch.float16 | 0.03 | 1.65 | 1.32 |
| 2048 | 4096 | torch.bfloat16 | 0.04 | 0.96 | 0.77 |
| 1 | 4096 | torch.bfloat16 | 0.00 | 0.01 | 0.01 |
| 2048 | 8192 | torch.float16 | 0.04 | 2.27 | 1.82 |
| 2048 | 8192 | torch.bfloat16 | 0.04 | 2.11 | 1.69 |
| 1 | 8192 | torch.bfloat16 | 0.00 | 0.01 | 0.02 |
| 2048 | 16384 | torch.float16 | 0.12 | 1.39 | 1.11 |
| 2048 | 16384 | torch.bfloat16 | 0.13 | 1.33 | 1.07 |
| 1 | 16384 | torch.bfloat16 | 0.01 | 0.01 | 0.02 |

### torch

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 2048 | 4096 | torch.float16 | 0.02 | 1.71 | 1.37 |
| 2048 | 4096 | torch.bfloat16 | 0.02 | 1.70 | 1.36 |
| 1 | 4096 | torch.bfloat16 | 0.01 | 0.00 | 0.00 |
| 2048 | 8192 | torch.float16 | 0.05 | 1.54 | 1.24 |
| 2048 | 8192 | torch.bfloat16 | 0.05 | 1.54 | 1.23 |
| 1 | 8192 | torch.bfloat16 | 0.02 | 0.00 | 0.00 |
| 2048 | 16384 | torch.float16 | 0.10 | 1.70 | 1.36 |
| 2048 | 16384 | torch.bfloat16 | 0.10 | 1.69 | 1.35 |
| 1 | 16384 | torch.bfloat16 | 0.05 | 0.00 | 0.00 |

## AnyOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | any | 0.01 | 0.42 | 0.85 |
| 1024 | 4096 | torch.bfloat16 | any | 0.01 | 0.42 | 0.84 |
| 1024 | 4096 | torch.float32 | any | 0.01 | 0.30 | 1.20 |
| 1024 | 4096 | torch.int32 | any | 0.03 | 0.16 | 0.63 |
| 4096 | 4096 | torch.float16 | any | 0.03 | 0.65 | 1.29 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | any | 0.03 | 0.12 | 0.24 |
| 1024 | 4096 | torch.bfloat16 | any | 0.03 | 0.12 | 0.24 |
| 1024 | 4096 | torch.float32 | any | 0.04 | 0.12 | 0.47 |
| 1024 | 4096 | torch.int32 | any | 0.04 | 0.12 | 0.47 |
| 4096 | 4096 | torch.float16 | any | 0.11 | 0.15 | 0.31 |

## AllOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | all | 0.01 | 0.43 | 0.85 |
| 1024 | 4096 | torch.bfloat16 | all | 0.01 | 0.42 | 0.84 |
| 1024 | 4096 | torch.float32 | all | 0.01 | 0.30 | 1.20 |
| 1024 | 4096 | torch.int32 | all | 0.03 | 0.16 | 0.62 |
| 4096 | 4096 | torch.float16 | all | 0.03 | 0.64 | 1.29 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | all | 0.03 | 0.12 | 0.25 |
| 1024 | 4096 | torch.bfloat16 | all | 0.03 | 0.12 | 0.25 |
| 1024 | 4096 | torch.float32 | all | 0.03 | 0.12 | 0.48 |
| 1024 | 4096 | torch.int32 | all | 0.04 | 0.12 | 0.48 |
| 4096 | 4096 | torch.float16 | all | 0.11 | 0.15 | 0.31 |

## CountNonzeroOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | count_nonzero | 0.01 | 0.56 | 1.12 |
| 1024 | 4096 | torch.bfloat16 | count_nonzero | 0.01 | 0.56 | 1.13 |
| 1024 | 4096 | torch.float32 | count_nonzero | 0.01 | 0.36 | 1.45 |
| 1024 | 4096 | torch.int32 | count_nonzero | 0.02 | 0.17 | 0.69 |
| 4096 | 4096 | torch.float16 | count_nonzero | 0.02 | 0.71 | 1.43 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | count_nonzero | 0.05 | 0.09 | 0.19 |
| 1024 | 4096 | torch.bfloat16 | count_nonzero | 0.04 | 0.09 | 0.19 |
| 1024 | 4096 | torch.float32 | count_nonzero | 0.05 | 0.08 | 0.34 |
| 1024 | 4096 | torch.int32 | count_nonzero | 0.05 | 0.09 | 0.35 |
| 4096 | 4096 | torch.float16 | count_nonzero | 0.19 | 0.09 | 0.18 |

## MeanPoolingForwardOp

### tileops

| batch_size | seq_len | heads | dim | chunk_size | dtype | accum_dtype | chunks_per_bacth | seq_num | use_offsets | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8192 | 64 | 128 | 64 | torch.float16 | torch.float32 | 128 | 1 | 0 | 0.20 | 0.33 | 0.68 |
| 2 | 2048 | 64 | 128 | 64 | torch.float16 | torch.float32 | 32 | 2 | 0 | 0.11 | 0.32 | 0.64 |
| 1 | 8192 | 64 | 128 | 64 | torch.float16 | torch.float32 | 128 | 4 | 1 | 0.29 | 0.23 | 0.47 |
| 1 | 1000 | 64 | 128 | 32 | torch.float16 | torch.float32 | 34 | 4 | 1 | 0.02 | 0.39 | 0.74 |

### torch-ref

| batch_size | seq_len | heads | dim | chunk_size | dtype | accum_dtype | chunks_per_bacth | seq_num | use_offsets | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8192 | 64 | 128 | 64 | torch.float16 | torch.float32 | 128 | 1 | 0 | 0.77 | 0.09 | 0.18 |
| 2 | 2048 | 64 | 128 | 64 | torch.float16 | torch.float32 | 32 | 2 | 0 | 0.27 | 0.12 | 0.25 |
| 1 | 8192 | 64 | 128 | 64 | torch.float16 | torch.float32 | 128 | 4 | 1 | 0.79 | 0.08 | 0.17 |
| 1 | 1000 | 64 | 128 | 32 | torch.float16 | torch.float32 | 34 | 4 | 1 | 0.27 | 0.03 | 0.06 |

## MultiHeadAttentionFwdOp

### tileops

| batch | seq_len | heads | dim | causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 8 | 64 | False | torch.float16 | 0.01 | 211.74 | 0.41 |
| 16 | 2048 | 16 | 128 | False | torch.float16 | 1.51 | 363.84 | 0.36 |
| 4 | 4096 | 16 | 128 | False | torch.bfloat16 | 1.39 | 396.23 | 0.19 |

### fa3

| batch | seq_len | heads | dim | causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 8 | 64 | False | torch.float16 | 0.02 | 123.09 | 0.24 |
| 16 | 2048 | 16 | 128 | False | torch.float16 | 2.30 | 238.69 | 0.23 |
| 4 | 4096 | 16 | 128 | False | torch.bfloat16 | 2.20 | 250.22 | 0.12 |

### flashinfer

| batch | seq_len | heads | dim | causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 8 | 64 | False | torch.float16 | 0.02 | 120.76 | 0.24 |
| 16 | 2048 | 16 | 128 | False | torch.float16 | 1.64 | 334.67 | 0.33 |
| 4 | 4096 | 16 | 128 | False | torch.bfloat16 | 1.54 | 357.93 | 0.17 |

## MultiHeadAttentionBwdOp

### tileops

| batch | seq_len | heads | dim | causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 8 | 64 | False | torch.float16 | 0.04 | 119.51 | 0.16 |
| 16 | 2048 | 16 | 128 | False | torch.float16 | 7.23 | 190.22 | 0.13 |
| 4 | 4096 | 16 | 128 | False | torch.bfloat16 | 6.53 | 210.40 | 0.07 |

### fa3

| batch | seq_len | heads | dim | causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 8 | 64 | False | torch.float16 | 0.07 | 81.27 | 0.11 |
| 16 | 2048 | 16 | 128 | False | torch.float16 | 11.67 | 117.79 | 0.08 |
| 4 | 4096 | 16 | 128 | False | torch.bfloat16 | 10.73 | 128.09 | 0.04 |

## MultiHeadAttentionDecodeWithKVCacheOp

### tileops

| b | h | s_q | s_kv | d | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 128 | 8192 | 128 | torch.float16 | 0.05 | 314.00 | 2.49 |
| 1 | 32 | 128 | 8192 | 128 | torch.bfloat16 | 0.07 | 248.25 | 1.97 |
| 1 | 32 | 128 | 5 | 128 | torch.float16 | 0.00 | 2.20 | 0.46 |

### fa3

| b | h | s_q | s_kv | d | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 128 | 8192 | 128 | torch.float16 | 0.07 | 235.79 | 1.87 |
| 1 | 32 | 128 | 8192 | 128 | torch.bfloat16 | 0.07 | 241.73 | 1.92 |
| 1 | 32 | 128 | 5 | 128 | torch.float16 | 0.01 | 1.58 | 0.33 |

### flashinfer

| b | h | s_q | s_kv | d | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 128 | 8192 | 128 | torch.float16 | 0.14 | 126.65 | 1.00 |
| 1 | 32 | 128 | 8192 | 128 | torch.bfloat16 | 0.13 | 127.51 | 1.01 |
| 1 | 32 | 128 | 5 | 128 | torch.float16 | 0.01 | 1.26 | 0.26 |

## MultiHeadAttentionDecodePagedWithKVCacheOp

### tileops

| batch | heads | seqlen_q | seqlen_kv | dim | page_size | is_causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 16 | 1 | 512 | 128 | 128 | False | torch.float16 | 0.02 | 0.18 | 0.18 |
| 2 | 8 | 1 | 1024 | 64 | 256 | False | torch.float16 | 0.02 | 0.20 | 0.10 |
| 1 | 8 | 1 | 1024 | 64 | 256 | False | torch.float16 | 0.02 | 0.09 | 0.09 |
| 1 | 8 | 1 | 512 | 64 | 256 | False | torch.float16 | 0.02 | 0.05 | 0.05 |

### flashinfer

| batch | heads | seqlen_q | seqlen_kv | dim | page_size | is_causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 16 | 1 | 512 | 128 | 128 | False | torch.float16 | 0.01 | 0.46 | 0.46 |
| 2 | 8 | 1 | 1024 | 64 | 256 | False | torch.float16 | 0.01 | 0.45 | 0.22 |
| 1 | 8 | 1 | 1024 | 64 | 256 | False | torch.float16 | 0.01 | 0.23 | 0.23 |
| 1 | 8 | 1 | 512 | 64 | 256 | False | torch.float16 | 0.01 | 0.12 | 0.12 |

### fa3

| batch | heads | seqlen_q | seqlen_kv | dim | page_size | is_causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 8 | 1 | 1024 | 64 | 256 | False | torch.float16 | 0.01 | 0.38 | 0.19 |
| 1 | 8 | 1 | 1024 | 64 | 256 | False | torch.float16 | 0.01 | 0.19 | 0.19 |
| 1 | 8 | 1 | 512 | 64 | 256 | False | torch.float16 | 0.01 | 0.10 | 0.10 |

## ManifoldConstrainedHyperConnectionPostOp

### tileops

| batch | n_expand | c_x | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | 1280 | torch.bfloat16 | 0.00 | 30.66 | 0.01 |
| 2 | 4 | 1920 | torch.bfloat16 | 0.00 | 133.24 | 0.01 |
| 4 | 4 | 2560 | torch.bfloat16 | 0.00 | 445.54 | 0.01 |

### torch-ref

| batch | n_expand | c_x | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | 1280 | torch.bfloat16 | 0.01 | 4.01 | 0.00 |
| 2 | 4 | 1920 | torch.bfloat16 | 0.01 | 17.16 | 0.00 |
| 4 | 4 | 2560 | torch.bfloat16 | 0.01 | 57.57 | 0.00 |

## ManifoldConstrainedHyperConnectionPreOp

### tileops

| batch | n_expand | c_x | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | 1280 | torch.bfloat16 | 0.04 | 30.37 | 0.00 |
| 2 | 4 | 1920 | torch.bfloat16 | 0.06 | 102.70 | 0.00 |
| 4 | 4 | 2560 | torch.bfloat16 | 0.07 | 286.42 | 0.00 |

### torch-ref

| batch | n_expand | c_x | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | 1280 | torch.bfloat16 | 0.27 | 4.64 | 0.00 |
| 2 | 4 | 1920 | torch.bfloat16 | 0.30 | 18.62 | 0.00 |
| 4 | 4 | 2560 | torch.bfloat16 | 0.35 | 58.23 | 0.00 |

## FusedMoe

### tileops

| num_tokens | num_experts | top_k | hidden_size | ffn_size | scoring_func | renormalize | with_correction_bias | routed_scaling_factor | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 384 | 8 | 7168 | 2048 | sigmoid | True | True | 2.827 | torch.bfloat16 | 0.25 | 2.82 | 135.31 |
| 32 | 384 | 8 | 7168 | 2048 | sigmoid | True | True | 2.827 | torch.bfloat16 | 4.72 | 4.77 | 7.16 |
| 512 | 384 | 8 | 7168 | 2048 | sigmoid | True | True | 2.827 | torch.bfloat16 | 9.03 | 39.94 | 3.75 |
| 4096 | 384 | 8 | 7168 | 2048 | sigmoid | True | True | 2.827 | torch.bfloat16 | 22.10 | 130.60 | 1.54 |
| 1 | 256 | 8 | 7168 | 2048 | sigmoid | True | False | 1.0 | torch.bfloat16 | 0.24 | 2.96 | 94.80 |
| 32 | 256 | 8 | 7168 | 2048 | sigmoid | True | False | 1.0 | torch.bfloat16 | 6.45 | 3.50 | 3.50 |
| 512 | 256 | 8 | 7168 | 2048 | sigmoid | True | False | 1.0 | torch.bfloat16 | 10.87 | 33.18 | 2.07 |
| 4096 | 256 | 8 | 7168 | 2048 | sigmoid | True | False | 1.0 | torch.bfloat16 | 21.39 | 134.94 | 1.06 |
| 1 | 128 | 8 | 7168 | 2048 | softmax | False | False | 1.0 | torch.bfloat16 | 0.22 | 3.15 | 50.45 |
| 32 | 128 | 8 | 7168 | 2048 | softmax | False | False | 1.0 | torch.bfloat16 | 4.59 | 4.92 | 2.46 |
| 512 | 128 | 8 | 7168 | 2048 | softmax | False | False | 1.0 | torch.bfloat16 | 5.53 | 65.18 | 2.04 |
| 4096 | 128 | 8 | 7168 | 2048 | softmax | False | False | 1.0 | torch.bfloat16 | 17.39 | 165.92 | 0.65 |
| 1 | 128 | 8 | 3072 | 1536 | softmax | False | False | 1.0 | torch.bfloat16 | 0.11 | 2.10 | 33.57 |
| 32 | 128 | 8 | 3072 | 1536 | softmax | False | False | 1.0 | torch.bfloat16 | 1.41 | 5.14 | 2.57 |
| 512 | 128 | 8 | 3072 | 1536 | softmax | False | False | 1.0 | torch.bfloat16 | 1.70 | 68.08 | 2.13 |
| 4096 | 128 | 8 | 3072 | 1536 | softmax | False | False | 1.0 | torch.bfloat16 | 5.11 | 181.39 | 0.72 |

### tileops-padded

| num_tokens | num_experts | top_k | hidden_size | ffn_size | scoring_func | renormalize | with_correction_bias | routed_scaling_factor | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 384 | 8 | 7168 | 2048 | sigmoid | True | True | 2.827 | torch.bfloat16 | 0.52 | 1.35 | 64.89 |
| 32 | 384 | 8 | 7168 | 2048 | sigmoid | True | True | 2.827 | torch.bfloat16 | 5.36 | 4.20 | 6.31 |
| 512 | 384 | 8 | 7168 | 2048 | sigmoid | True | True | 2.827 | torch.bfloat16 | 9.70 | 37.21 | 3.49 |
| 4096 | 384 | 8 | 7168 | 2048 | sigmoid | True | True | 2.827 | torch.bfloat16 | 23.14 | 124.71 | 1.47 |
| 1 | 256 | 8 | 7168 | 2048 | sigmoid | True | False | 1.0 | torch.bfloat16 | 0.44 | 1.61 | 51.37 |
| 32 | 256 | 8 | 7168 | 2048 | sigmoid | True | False | 1.0 | torch.bfloat16 | 7.08 | 3.19 | 3.19 |
| 512 | 256 | 8 | 7168 | 2048 | sigmoid | True | False | 1.0 | torch.bfloat16 | 11.69 | 30.85 | 1.93 |
| 4096 | 256 | 8 | 7168 | 2048 | sigmoid | True | False | 1.0 | torch.bfloat16 | 21.53 | 134.03 | 1.05 |
| 1 | 128 | 8 | 7168 | 2048 | softmax | False | False | 1.0 | torch.bfloat16 | 0.36 | 1.96 | 31.39 |
| 32 | 128 | 8 | 7168 | 2048 | softmax | False | False | 1.0 | torch.bfloat16 | 4.98 | 4.53 | 2.26 |
| 512 | 128 | 8 | 7168 | 2048 | softmax | False | False | 1.0 | torch.bfloat16 | 5.75 | 62.77 | 1.96 |
| 4096 | 128 | 8 | 7168 | 2048 | softmax | False | False | 1.0 | torch.bfloat16 | 17.64 | 163.57 | 0.65 |
| 1 | 128 | 8 | 3072 | 1536 | softmax | False | False | 1.0 | torch.bfloat16 | 0.14 | 1.63 | 26.08 |
| 32 | 128 | 8 | 3072 | 1536 | softmax | False | False | 1.0 | torch.bfloat16 | 1.68 | 4.32 | 2.16 |
| 512 | 128 | 8 | 3072 | 1536 | softmax | False | False | 1.0 | torch.bfloat16 | 1.91 | 60.60 | 1.90 |
| 4096 | 128 | 8 | 3072 | 1536 | softmax | False | False | 1.0 | torch.bfloat16 | 5.42 | 171.12 | 0.68 |

### vllm

| num_tokens | num_experts | top_k | hidden_size | ffn_size | scoring_func | renormalize | with_correction_bias | routed_scaling_factor | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 384 | 8 | 7168 | 2048 | sigmoid | True | True | 2.827 | torch.bfloat16 | 0.21 | 3.37 | 161.52 |
| 32 | 384 | 8 | 7168 | 2048 | sigmoid | True | True | 2.827 | torch.bfloat16 | 7.06 | 3.19 | 4.79 |
| 512 | 384 | 8 | 7168 | 2048 | sigmoid | True | True | 2.827 | torch.bfloat16 | 15.67 | 23.03 | 2.16 |
| 4096 | 384 | 8 | 7168 | 2048 | sigmoid | True | True | 2.827 | torch.bfloat16 | 20.60 | 140.14 | 1.65 |
| 1 | 256 | 8 | 7168 | 2048 | sigmoid | True | False | 1.0 | torch.bfloat16 | 0.21 | 3.41 | 109.11 |
| 32 | 256 | 8 | 7168 | 2048 | sigmoid | True | False | 1.0 | torch.bfloat16 | 5.90 | 3.82 | 3.83 |
| 512 | 256 | 8 | 7168 | 2048 | sigmoid | True | False | 1.0 | torch.bfloat16 | 10.40 | 34.70 | 2.17 |
| 4096 | 256 | 8 | 7168 | 2048 | sigmoid | True | False | 1.0 | torch.bfloat16 | 18.78 | 153.72 | 1.21 |
| 1 | 128 | 8 | 7168 | 2048 | softmax | False | False | 1.0 | torch.bfloat16 | 0.21 | 3.42 | 54.66 |
| 32 | 128 | 8 | 7168 | 2048 | softmax | False | False | 1.0 | torch.bfloat16 | 4.25 | 5.31 | 2.66 |
| 512 | 128 | 8 | 7168 | 2048 | softmax | False | False | 1.0 | torch.bfloat16 | 5.39 | 66.90 | 2.09 |
| 4096 | 128 | 8 | 7168 | 2048 | softmax | False | False | 1.0 | torch.bfloat16 | 15.49 | 186.37 | 0.74 |
| 1 | 128 | 8 | 3072 | 1536 | softmax | False | False | 1.0 | torch.bfloat16 | 0.08 | 2.82 | 45.19 |
| 32 | 128 | 8 | 3072 | 1536 | softmax | False | False | 1.0 | torch.bfloat16 | 1.26 | 5.75 | 2.88 |
| 512 | 128 | 8 | 3072 | 1536 | softmax | False | False | 1.0 | torch.bfloat16 | 1.64 | 70.75 | 2.21 |
| 4096 | 128 | 8 | 3072 | 1536 | softmax | False | False | 1.0 | torch.bfloat16 | 4.89 | 189.60 | 0.75 |

## fused_topk

### tileops

| num_tokens | num_experts | top_k | scoring_func | renormalize | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 384 | 8 | sigmoid | True | torch.bfloat16 | 0.01 | 0.00 | 0.00 |
| 32 | 384 | 8 | sigmoid | True | torch.bfloat16 | 0.02 | 0.01 | 0.00 |
| 512 | 384 | 8 | sigmoid | True | torch.bfloat16 | 0.02 | 0.20 | 0.02 |
| 4096 | 384 | 8 | sigmoid | True | torch.bfloat16 | 0.02 | 1.14 | 0.14 |
| 1 | 256 | 8 | sigmoid | True | torch.bfloat16 | 0.01 | 0.00 | 0.00 |
| 32 | 256 | 8 | sigmoid | True | torch.bfloat16 | 0.01 | 0.01 | 0.00 |
| 512 | 256 | 8 | sigmoid | True | torch.bfloat16 | 0.02 | 0.16 | 0.02 |
| 4096 | 256 | 8 | sigmoid | True | torch.bfloat16 | 0.02 | 0.97 | 0.12 |
| 1 | 128 | 8 | softmax | False | torch.bfloat16 | 0.00 | 0.00 | 0.00 |
| 32 | 128 | 8 | softmax | False | torch.bfloat16 | 0.01 | 0.01 | 0.00 |
| 512 | 128 | 8 | softmax | False | torch.bfloat16 | 0.01 | 0.16 | 0.02 |
| 4096 | 128 | 8 | softmax | False | torch.bfloat16 | 0.01 | 0.92 | 0.13 |

### vllm

| num_tokens | num_experts | top_k | scoring_func | renormalize | dtype | has_external | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 384 | 8 | sigmoid | True | torch.bfloat16 | True | 0.01 | 0.00 | 0.00 |
| 32 | 384 | 8 | sigmoid | True | torch.bfloat16 | True | 0.01 | 0.02 | 0.00 |
| 512 | 384 | 8 | sigmoid | True | torch.bfloat16 | True | 0.01 | 0.31 | 0.04 |
| 4096 | 384 | 8 | sigmoid | True | torch.bfloat16 | True | 0.02 | 1.16 | 0.14 |
| 1 | 256 | 8 | sigmoid | True | torch.bfloat16 | True | 0.01 | 0.00 | 0.00 |
| 32 | 256 | 8 | sigmoid | True | torch.bfloat16 | True | 0.01 | 0.02 | 0.00 |
| 512 | 256 | 8 | sigmoid | True | torch.bfloat16 | True | 0.01 | 0.22 | 0.03 |
| 4096 | 256 | 8 | sigmoid | True | torch.bfloat16 | True | 0.02 | 0.91 | 0.11 |
| 1 | 128 | 8 | softmax | False | torch.bfloat16 | True | 0.01 | 0.00 | 0.00 |
| 32 | 128 | 8 | softmax | False | torch.bfloat16 | True | 0.01 | 0.01 | 0.00 |
| 512 | 128 | 8 | softmax | False | torch.bfloat16 | True | 0.01 | 0.14 | 0.02 |
| 4096 | 128 | 8 | softmax | False | torch.bfloat16 | True | 0.01 | 0.70 | 0.10 |

## MoePermutePaddedOp

### tileops

| total_tokens | top_k | num_experts | hidden_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8 | 384 | 7168 | torch.bfloat16 | 0.03 | 0.00 | 0.01 |
| 32 | 8 | 384 | 7168 | torch.bfloat16 | 0.03 | 0.00 | 0.15 |
| 512 | 8 | 384 | 7168 | torch.bfloat16 | 0.08 | 0.00 | 0.80 |
| 4096 | 8 | 384 | 7168 | torch.bfloat16 | 0.63 | 0.00 | 0.84 |
| 1 | 8 | 256 | 7168 | torch.bfloat16 | 0.02 | 0.00 | 0.01 |
| 32 | 8 | 256 | 7168 | torch.bfloat16 | 0.02 | 0.00 | 0.23 |
| 512 | 8 | 256 | 7168 | torch.bfloat16 | 0.07 | 0.00 | 0.93 |
| 4096 | 8 | 256 | 7168 | torch.bfloat16 | 0.61 | 0.00 | 0.86 |
| 1 | 8 | 128 | 7168 | torch.bfloat16 | 0.01 | 0.00 | 0.02 |
| 32 | 8 | 128 | 7168 | torch.bfloat16 | 0.01 | 0.00 | 0.44 |
| 512 | 8 | 128 | 7168 | torch.bfloat16 | 0.06 | 0.00 | 1.15 |
| 4096 | 8 | 128 | 7168 | torch.bfloat16 | 0.59 | 0.00 | 0.90 |
| 1 | 8 | 128 | 3072 | torch.bfloat16 | 0.01 | 0.00 | 0.01 |
| 32 | 8 | 128 | 3072 | torch.bfloat16 | 0.01 | 0.00 | 0.21 |
| 512 | 8 | 128 | 3072 | torch.bfloat16 | 0.03 | 0.00 | 1.06 |
| 4096 | 8 | 128 | 3072 | torch.bfloat16 | 0.31 | 0.00 | 0.74 |

### vllm

| total_tokens | top_k | num_experts | hidden_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8 | 384 | 7168 | torch.bfloat16 | 0.01 | 0.00 | 0.01 |
| 32 | 8 | 384 | 7168 | torch.bfloat16 | 0.01 | 0.00 | 0.31 |
| 512 | 8 | 384 | 7168 | torch.bfloat16 | 0.04 | 0.00 | 1.58 |
| 4096 | 8 | 384 | 7168 | torch.bfloat16 | 0.40 | 0.00 | 1.33 |
| 1 | 8 | 256 | 7168 | torch.bfloat16 | 0.01 | 0.00 | 0.01 |
| 32 | 8 | 256 | 7168 | torch.bfloat16 | 0.01 | 0.00 | 0.31 |
| 512 | 8 | 256 | 7168 | torch.bfloat16 | 0.04 | 0.00 | 1.59 |
| 4096 | 8 | 256 | 7168 | torch.bfloat16 | 0.40 | 0.00 | 1.34 |
| 1 | 8 | 128 | 7168 | torch.bfloat16 | 0.01 | 0.00 | 0.01 |
| 32 | 8 | 128 | 7168 | torch.bfloat16 | 0.01 | 0.00 | 0.31 |
| 512 | 8 | 128 | 7168 | torch.bfloat16 | 0.04 | 0.00 | 1.59 |
| 4096 | 8 | 128 | 7168 | torch.bfloat16 | 0.39 | 0.00 | 1.37 |
| 1 | 8 | 128 | 3072 | torch.bfloat16 | 0.01 | 0.00 | 0.01 |
| 32 | 8 | 128 | 3072 | torch.bfloat16 | 0.01 | 0.00 | 0.15 |
| 512 | 8 | 128 | 3072 | torch.bfloat16 | 0.03 | 0.00 | 1.06 |
| 4096 | 8 | 128 | 3072 | torch.bfloat16 | 0.16 | 0.00 | 1.42 |

## MoePermuteAlignOp

### tileops

| total_tokens | top_k | num_experts | block_size | numel | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8 | 384 | 64 | 8 | 0.01 | 0.00 | 0.01 |
| 32 | 8 | 384 | 64 | 256 | 0.01 | 0.00 | 0.01 |
| 512 | 8 | 384 | 64 | 4096 | 0.01 | 0.00 | 0.01 |
| 4096 | 8 | 384 | 64 | 32768 | 0.03 | 0.00 | 0.01 |
| 1 | 8 | 256 | 64 | 8 | 0.01 | 0.00 | 0.01 |
| 32 | 8 | 256 | 64 | 256 | 0.01 | 0.00 | 0.01 |
| 512 | 8 | 256 | 64 | 4096 | 0.01 | 0.00 | 0.01 |
| 4096 | 8 | 256 | 64 | 32768 | 0.03 | 0.00 | 0.01 |
| 1 | 8 | 128 | 64 | 8 | 0.01 | 0.00 | 0.01 |
| 32 | 8 | 128 | 64 | 256 | 0.01 | 0.00 | 0.01 |
| 512 | 8 | 128 | 64 | 4096 | 0.01 | 0.00 | 0.01 |
| 4096 | 8 | 128 | 64 | 32768 | 0.02 | 0.00 | 0.01 |

### triton

| total_tokens | top_k | num_experts | block_size | numel | max_padded | max_num_blocks | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8 | 384 | 64 | 8 | 24263 | 380 | 0.04 | 0.00 | 0.00 |
| 32 | 8 | 384 | 64 | 256 | 24511 | 383 | 0.04 | 0.00 | 0.00 |
| 512 | 8 | 384 | 64 | 4096 | 28351 | 443 | 0.05 | 0.00 | 0.00 |
| 4096 | 8 | 384 | 64 | 32768 | 57023 | 891 | 0.08 | 0.00 | 0.00 |
| 1 | 8 | 256 | 64 | 8 | 16199 | 254 | 0.03 | 0.00 | 0.00 |
| 32 | 8 | 256 | 64 | 256 | 16447 | 257 | 0.03 | 0.00 | 0.00 |
| 512 | 8 | 256 | 64 | 4096 | 20287 | 317 | 0.04 | 0.00 | 0.00 |
| 4096 | 8 | 256 | 64 | 32768 | 48959 | 765 | 0.07 | 0.00 | 0.00 |
| 1 | 8 | 128 | 64 | 8 | 8135 | 128 | 0.01 | 0.00 | 0.00 |
| 32 | 8 | 128 | 64 | 256 | 8383 | 131 | 0.02 | 0.00 | 0.00 |
| 512 | 8 | 128 | 64 | 4096 | 12223 | 191 | 0.02 | 0.00 | 0.00 |
| 4096 | 8 | 128 | 64 | 32768 | 40895 | 639 | 0.07 | 0.00 | 0.00 |

## MoePermuteNopadOp

### tileops

| total_tokens | top_k | num_experts | hidden_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8 | 384 | 7168 | torch.bfloat16 | 0.01 | 0.00 | 0.02 |
| 32 | 8 | 384 | 7168 | torch.bfloat16 | 0.01 | 0.00 | 0.49 |
| 512 | 8 | 384 | 7168 | torch.bfloat16 | 0.03 | 0.00 | 1.90 |
| 4096 | 8 | 384 | 7168 | torch.bfloat16 | 0.55 | 0.00 | 0.97 |
| 1 | 8 | 256 | 7168 | torch.bfloat16 | 0.01 | 0.00 | 0.02 |
| 32 | 8 | 256 | 7168 | torch.bfloat16 | 0.01 | 0.00 | 0.56 |
| 512 | 8 | 256 | 7168 | torch.bfloat16 | 0.03 | 0.00 | 1.99 |
| 4096 | 8 | 256 | 7168 | torch.bfloat16 | 0.54 | 0.00 | 0.98 |
| 1 | 8 | 128 | 7168 | torch.bfloat16 | 0.01 | 0.00 | 0.03 |
| 32 | 8 | 128 | 7168 | torch.bfloat16 | 0.01 | 0.00 | 0.70 |
| 512 | 8 | 128 | 7168 | torch.bfloat16 | 0.03 | 0.00 | 2.10 |
| 4096 | 8 | 128 | 7168 | torch.bfloat16 | 0.52 | 0.00 | 1.02 |
| 1 | 8 | 128 | 3072 | torch.bfloat16 | 0.00 | 0.00 | 0.01 |
| 32 | 8 | 128 | 3072 | torch.bfloat16 | 0.01 | 0.00 | 0.32 |
| 512 | 8 | 128 | 3072 | torch.bfloat16 | 0.02 | 0.00 | 1.58 |
| 4096 | 8 | 128 | 3072 | torch.bfloat16 | 0.22 | 0.00 | 1.04 |

### vllm

| total_tokens | top_k | num_experts | hidden_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8 | 384 | 7168 | torch.bfloat16 | 0.01 | 0.00 | 0.01 |
| 32 | 8 | 384 | 7168 | torch.bfloat16 | 0.01 | 0.00 | 0.31 |
| 512 | 8 | 384 | 7168 | torch.bfloat16 | 0.04 | 0.00 | 1.59 |
| 4096 | 8 | 384 | 7168 | torch.bfloat16 | 0.40 | 0.00 | 1.33 |
| 1 | 8 | 256 | 7168 | torch.bfloat16 | 0.01 | 0.00 | 0.01 |
| 32 | 8 | 256 | 7168 | torch.bfloat16 | 0.01 | 0.00 | 0.31 |
| 512 | 8 | 256 | 7168 | torch.bfloat16 | 0.04 | 0.00 | 1.59 |
| 4096 | 8 | 256 | 7168 | torch.bfloat16 | 0.39 | 0.00 | 1.34 |
| 1 | 8 | 128 | 7168 | torch.bfloat16 | 0.01 | 0.00 | 0.01 |
| 32 | 8 | 128 | 7168 | torch.bfloat16 | 0.01 | 0.00 | 0.31 |
| 512 | 8 | 128 | 7168 | torch.bfloat16 | 0.04 | 0.00 | 1.59 |
| 4096 | 8 | 128 | 7168 | torch.bfloat16 | 0.38 | 0.00 | 1.38 |
| 1 | 8 | 128 | 3072 | torch.bfloat16 | 0.01 | 0.00 | 0.01 |
| 32 | 8 | 128 | 3072 | torch.bfloat16 | 0.01 | 0.00 | 0.15 |
| 512 | 8 | 128 | 3072 | torch.bfloat16 | 0.03 | 0.00 | 1.06 |
| 4096 | 8 | 128 | 3072 | torch.bfloat16 | 0.16 | 0.00 | 1.44 |

## moe_unpermute

### tileops

| total_tokens | top_k | hidden_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 8 | 7168 | torch.bfloat16 | 0.01 | 0.02 | 0.02 |
| 32 | 8 | 7168 | torch.bfloat16 | 0.01 | 0.47 | 0.52 |
| 512 | 8 | 7168 | torch.bfloat16 | 0.04 | 1.62 | 1.82 |
| 4096 | 8 | 7168 | torch.bfloat16 | 0.20 | 2.33 | 2.62 |
| 1 | 8 | 3072 | torch.bfloat16 | 0.01 | 0.01 | 0.01 |
| 32 | 8 | 3072 | torch.bfloat16 | 0.01 | 0.24 | 0.27 |
| 512 | 8 | 3072 | torch.bfloat16 | 0.01 | 1.85 | 2.09 |
| 4096 | 8 | 3072 | torch.bfloat16 | 0.09 | 2.20 | 2.47 |

### vllm

| total_tokens | top_k | hidden_size | dtype | numel | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8 | 7168 | torch.bfloat16 | 8 | 0.02 | 0.00 | 0.01 |
| 32 | 8 | 7168 | torch.bfloat16 | 256 | 0.03 | 0.14 | 0.16 |
| 512 | 8 | 7168 | torch.bfloat16 | 4096 | 0.04 | 1.36 | 1.53 |
| 4096 | 8 | 7168 | torch.bfloat16 | 32768 | 0.20 | 2.31 | 2.60 |
| 1 | 8 | 3072 | torch.bfloat16 | 8 | 0.01 | 0.00 | 0.00 |
| 32 | 8 | 3072 | torch.bfloat16 | 256 | 0.01 | 0.11 | 0.13 |
| 512 | 8 | 3072 | torch.bfloat16 | 4096 | 0.02 | 1.12 | 1.26 |
| 4096 | 8 | 3072 | torch.bfloat16 | 32768 | 0.10 | 2.04 | 2.30 |

## SumOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | sum | 0.01 | 0.56 | 1.13 |
| 1024 | 4096 | torch.bfloat16 | sum | 0.01 | 0.57 | 1.13 |
| 4096 | 4096 | torch.float16 | sum | 0.02 | 0.71 | 1.42 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | sum | 0.03 | 0.15 | 0.29 |
| 1024 | 4096 | torch.bfloat16 | sum | 0.03 | 0.14 | 0.29 |
| 4096 | 4096 | torch.float16 | sum | 0.13 | 0.13 | 0.27 |

## MeanOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | mean | 0.01 | 0.56 | 1.12 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | mean | 0.03 | 0.15 | 0.29 |

## AmaxOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | amax | 0.01 | 0.56 | 1.12 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | amax | 0.03 | 0.14 | 0.29 |

## AminOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | amin | 0.01 | 0.56 | 1.12 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | amin | 0.03 | 0.14 | 0.28 |

## ProdOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | prod | 0.02 | 0.24 | 0.47 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | prod | 0.03 | 0.14 | 0.29 |

## StdOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | std | 0.01 | 1.45 | 0.96 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | std | 0.05 | 0.26 | 0.17 |

## VarOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | var | 0.01 | 1.46 | 0.97 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | var | 0.05 | 0.26 | 0.17 |

## VarMeanOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | var_mean | 0.01 | 1.58 | 1.06 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | var_mean | 0.05 | 0.24 | 0.16 |

## RmsNormOp

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 2048 | 4096 | torch.float16 | 0.02 | 1.46 | 1.46 |
| 2048 | 4096 | torch.bfloat16 | 0.02 | 1.95 | 1.95 |
| 1 | 4096 | torch.bfloat16 | 0.00 | 0.00 | 0.01 |
| 2048 | 8192 | torch.float16 | 0.03 | 2.01 | 2.01 |
| 2048 | 8192 | torch.bfloat16 | 0.03 | 1.99 | 1.99 |
| 1 | 8192 | torch.bfloat16 | 0.00 | 0.01 | 0.01 |
| 2048 | 16384 | torch.float16 | 0.10 | 1.33 | 1.33 |
| 2048 | 16384 | torch.bfloat16 | 0.10 | 1.28 | 1.28 |
| 1 | 16384 | torch.bfloat16 | 0.00 | 0.01 | 0.02 |

### torch-ref

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 2048 | 4096 | torch.float16 | 0.19 | 0.17 | 0.17 |
| 2048 | 4096 | torch.bfloat16 | 0.20 | 0.17 | 0.17 |
| 1 | 4096 | torch.bfloat16 | 0.02 | 0.00 | 0.00 |
| 2048 | 8192 | torch.float16 | 0.42 | 0.16 | 0.16 |
| 2048 | 8192 | torch.bfloat16 | 0.42 | 0.16 | 0.16 |
| 1 | 8192 | torch.bfloat16 | 0.02 | 0.00 | 0.00 |
| 2048 | 16384 | torch.float16 | 0.86 | 0.16 | 0.16 |
| 2048 | 16384 | torch.bfloat16 | 0.86 | 0.16 | 0.16 |
| 1 | 16384 | torch.bfloat16 | 0.02 | 0.00 | 0.00 |

## RopeNeoxOp

### tileops

| seq_len | head_dim | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 2048 | 64 | torch.float16 | 0.00 | 0.25 | 0.38 |
| 2048 | 64 | torch.bfloat16 | 0.00 | 0.25 | 0.37 |
| 2048 | 64 | torch.float32 | 0.00 | 0.22 | 0.67 |
| 2048 | 128 | torch.float16 | 0.00 | 0.44 | 0.67 |
| 2048 | 128 | torch.bfloat16 | 0.00 | 0.44 | 0.66 |
| 2048 | 128 | torch.float32 | 0.00 | 0.35 | 1.06 |
| 4096 | 128 | torch.float16 | 0.00 | 0.70 | 1.05 |
| 4096 | 128 | torch.bfloat16 | 0.00 | 0.71 | 1.07 |
| 4096 | 128 | torch.float32 | 0.00 | 0.55 | 1.64 |

### torch-ref

| seq_len | head_dim | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 2048 | 64 | torch.float16 | 0.01 | 0.04 | 0.06 |
| 2048 | 64 | torch.bfloat16 | 0.01 | 0.04 | 0.06 |
| 2048 | 64 | torch.float32 | 0.01 | 0.04 | 0.13 |
| 2048 | 128 | torch.float16 | 0.01 | 0.08 | 0.11 |
| 2048 | 128 | torch.bfloat16 | 0.01 | 0.08 | 0.11 |
| 2048 | 128 | torch.float32 | 0.01 | 0.08 | 0.23 |
| 4096 | 128 | torch.float16 | 0.02 | 0.13 | 0.20 |
| 4096 | 128 | torch.bfloat16 | 0.02 | 0.13 | 0.20 |
| 4096 | 128 | torch.float32 | 0.02 | 0.13 | 0.38 |

## SoftmaxOp

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.02 | 1.04 | 1.04 |
| 4096 | 4096 | torch.bfloat16 | 0.08 | 0.86 | 0.86 |
| 1024 | 3000 | torch.float16 | 0.03 | 0.49 | 0.49 |
| 1025 | 4096 | torch.float16 | 0.02 | 1.05 | 1.05 |

### torch

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.03 | 0.65 | 0.65 |
| 4096 | 4096 | torch.bfloat16 | 0.09 | 0.75 | 0.75 |
| 1024 | 3000 | torch.float16 | 0.02 | 0.58 | 0.58 |
| 1025 | 4096 | torch.float16 | 0.03 | 0.66 | 0.66 |

## LogSoftmaxOp

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.02 | 1.01 | 0.81 |
| 4096 | 4096 | torch.bfloat16 | 0.12 | 0.73 | 0.58 |
| 1024 | 3000 | torch.float16 | 0.03 | 0.52 | 0.42 |
| 1025 | 4096 | torch.float16 | 0.02 | 1.01 | 0.81 |

### torch

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.02 | 0.94 | 0.76 |
| 4096 | 4096 | torch.bfloat16 | 0.09 | 0.91 | 0.73 |
| 1024 | 3000 | torch.float16 | 0.02 | 0.81 | 0.65 |
| 1025 | 4096 | torch.float16 | 0.02 | 0.95 | 0.76 |

## LogSumExpOp

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.01 | 1.33 | 0.89 |
| 4096 | 4096 | torch.bfloat16 | 0.03 | 1.58 | 1.06 |
| 1024 | 3000 | torch.float16 | 0.02 | 0.56 | 0.37 |
| 1025 | 4096 | torch.float16 | 0.01 | 1.33 | 0.89 |

### torch

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.05 | 0.27 | 0.18 |
| 4096 | 4096 | torch.bfloat16 | 0.13 | 0.39 | 0.26 |
| 1024 | 3000 | torch.float16 | 0.04 | 0.22 | 0.15 |
| 1025 | 4096 | torch.float16 | 0.05 | 0.26 | 0.17 |

## SsdChunkScanFwdOp

### tileops

| batch | num_chunks | chunk_len | n_heads | d_head | d_state | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 64 | 4 | 64 | 32 | torch.float16 | 0.00 | 0.91 | 0.07 |
| 2 | 4 | 64 | 8 | 64 | 64 | torch.float16 | 0.01 | 8.18 | 0.51 |
| 1 | 2 | 128 | 4 | 128 | 32 | torch.bfloat16 | 0.01 | 3.40 | 0.16 |
| 2 | 2 | 64 | 4 | 64 | 32 | torch.bfloat16 | 0.00 | 1.79 | 0.14 |

## ssd_chunk_scan_fwd

### baseline

| batch | num_chunks | chunk_len | n_heads | d_head | d_state | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 64 | 4 | 64 | 32 | torch.float16 | 0.05 | 0.08 | 0.01 |
| 2 | 4 | 64 | 8 | 64 | 64 | torch.float16 | 0.06 | 0.82 | 0.05 |
| 1 | 2 | 128 | 4 | 128 | 32 | torch.bfloat16 | 0.06 | 0.43 | 0.02 |
| 2 | 2 | 64 | 4 | 64 | 32 | torch.bfloat16 | 0.06 | 0.15 | 0.01 |

## SsdChunkStateFwdOp

### tileops

| batch | num_chunks | chunk_len | n_heads | d_head | d_state | n_groups | dtype | has_seq_idx | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 64 | 4 | 64 | 32 | 1 | torch.float16 | False | 0.01 | 0.31 | 0.02 |
| 2 | 4 | 64 | 8 | 64 | 64 | 2 | torch.float16 | False | 0.01 | 4.49 | 0.23 |
| 1 | 2 | 128 | 4 | 128 | 32 | 1 | torch.bfloat16 | False | 0.01 | 0.77 | 0.04 |
| 2 | 2 | 64 | 4 | 64 | 32 | 2 | torch.bfloat16 | False | 0.01 | 0.68 | 0.05 |
| 2 | 4 | 64 | 8 | 64 | 64 | 2 | torch.float16 | True | 0.01 | 3.52 | 0.18 |

## ssd_chunk_state_fwd

### baseline

| batch | num_chunks | chunk_len | n_heads | d_head | d_state | n_groups | dtype | has_seq_idx | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 64 | 4 | 64 | 32 | 1 | torch.float16 | False | 0.03 | 0.06 | 0.00 |
| 2 | 4 | 64 | 8 | 64 | 64 | 2 | torch.float16 | False | 0.11 | 0.29 | 0.02 |
| 1 | 2 | 128 | 4 | 128 | 32 | 1 | torch.bfloat16 | False | 0.05 | 0.18 | 0.01 |
| 2 | 2 | 64 | 4 | 64 | 32 | 2 | torch.bfloat16 | False | 0.04 | 0.11 | 0.01 |
| 2 | 4 | 64 | 8 | 64 | 64 | 2 | torch.float16 | True | 0.12 | 0.28 | 0.01 |

## SsdDecodeOp

### tileops

| batch | n_heads | d_head | d_state | n_groups | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | 64 | 16 | 1 | torch.float16 | 0.00 | 0.01 | 0.01 |
| 2 | 8 | 64 | 32 | 2 | torch.float16 | 0.01 | 0.03 | 0.05 |
| 1 | 4 | 64 | 16 | 1 | torch.bfloat16 | 0.00 | 0.01 | 0.01 |
| 2 | 8 | 128 | 64 | 4 | torch.bfloat16 | 0.01 | 0.08 | 0.11 |

## ssd_decode

### baseline

| batch | n_heads | d_head | d_state | n_groups | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | 64 | 16 | 1 | torch.float16 | 0.03 | 0.00 | 0.00 |
| 2 | 8 | 64 | 32 | 2 | torch.float16 | 0.03 | 0.01 | 0.01 |
| 1 | 4 | 64 | 16 | 1 | torch.bfloat16 | 0.03 | 0.00 | 0.00 |
| 2 | 8 | 128 | 64 | 4 | torch.bfloat16 | 0.04 | 0.02 | 0.03 |

## SsdStatePassingFwdOp

### tileops

| batch | num_chunks | n_heads | d_state | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 4 | 32 | torch.float16 | 0.00 | 0.00 | 0.00 |
| 2 | 4 | 8 | 64 | torch.float16 | 0.00 | 0.00 | 0.02 |
| 1 | 2 | 4 | 32 | torch.bfloat16 | 0.00 | 0.00 | 0.00 |
| 2 | 4 | 8 | 64 | torch.bfloat16 | 0.00 | 0.00 | 0.02 |

## ssd_state_passing_fwd

### baseline

| batch | num_chunks | n_heads | d_state | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 4 | 32 | torch.float16 | 0.02 | 0.00 | 0.00 |
| 2 | 4 | 8 | 64 | torch.float16 | 0.04 | 0.00 | 0.00 |
| 1 | 2 | 4 | 32 | torch.bfloat16 | 0.02 | 0.00 | 0.00 |
| 2 | 4 | 8 | 64 | torch.bfloat16 | 0.04 | 0.00 | 0.00 |

## TopkSelectorOp

### tileops

| batch | seq_len | seq_len_kv | kv_group | topk | in_dtype | out_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32768 | 65536 | 1 | 1024 | torch.float32 | torch.int32 | 16.25 | N/A | 0.54 |
| 1 | 32768 | 65536 | 1 | 2048 | torch.float32 | torch.int32 | 16.83 | N/A | 0.53 |
| 1 | 65535 | 131072 | 1 | 1024 | torch.float32 | torch.int32 | 88.34 | N/A | 0.39 |
| 1 | 65535 | 131072 | 1 | 2048 | torch.float32 | torch.int32 | 89.14 | N/A | 0.39 |

### torch

| batch | seq_len | seq_len_kv | kv_group | topk | in_dtype | out_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32768 | 65536 | 1 | 1024 | torch.float32 | torch.int32 | 53.74 | N/A | 0.16 |
| 1 | 32768 | 65536 | 1 | 2048 | torch.float32 | torch.int32 | 56.26 | N/A | 0.16 |
| 1 | 65535 | 131072 | 1 | 1024 | torch.float32 | torch.int32 | 206.17 | N/A | 0.17 |
| 1 | 65535 | 131072 | 1 | 2048 | torch.float32 | torch.int32 | 211.40 | N/A | 0.17 |

## exp

### tileops

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| exp | 262144 | torch.float16 | torch.float16 | 0.00 | 0.12 | 0.50 |
| exp | 1048576 | torch.float16 | torch.float16 | 0.00 | 0.36 | 1.42 |
| exp | 4000000 | torch.float16 | torch.float16 | 0.01 | 0.65 | 2.59 |
| exp | 262144 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.12 | 0.49 |
| exp | 1048576 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.36 | 1.43 |
| exp | 4000000 | torch.bfloat16 | torch.bfloat16 | 0.01 | 0.65 | 2.60 |

### torch

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| exp | 262144 | torch.float16 | torch.float16 | 0.00 | 0.10 | 0.39 |
| exp | 1048576 | torch.float16 | torch.float16 | 0.00 | 0.27 | 1.07 |
| exp | 4000000 | torch.float16 | torch.float16 | 0.01 | 0.49 | 1.94 |
| exp | 262144 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.10 | 0.38 |
| exp | 1048576 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.28 | 1.11 |
| exp | 4000000 | torch.bfloat16 | torch.bfloat16 | 0.01 | 0.49 | 1.96 |

## gelu

### tileops

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| gelu | 262144 | torch.float16 | torch.float16 | 0.00 | 0.12 | 0.47 |
| gelu | 1048576 | torch.float16 | torch.float16 | 0.00 | 0.30 | 1.22 |
| gelu | 4000000 | torch.float16 | torch.float16 | 0.01 | 0.48 | 1.91 |
| gelu | 262144 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.12 | 0.47 |
| gelu | 1048576 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.30 | 1.20 |
| gelu | 4000000 | torch.bfloat16 | torch.bfloat16 | 0.01 | 0.46 | 1.85 |

### torch

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| gelu | 262144 | torch.float16 | torch.float16 | 0.00 | 0.09 | 0.37 |
| gelu | 1048576 | torch.float16 | torch.float16 | 0.00 | 0.24 | 0.96 |
| gelu | 4000000 | torch.float16 | torch.float16 | 0.01 | 0.38 | 1.53 |
| gelu | 262144 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.09 | 0.36 |
| gelu | 1048576 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.24 | 0.95 |
| gelu | 4000000 | torch.bfloat16 | torch.bfloat16 | 0.01 | 0.37 | 1.48 |

## logical_not

### tileops

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| logical_not | 262144 | torch.float16 | torch.bool | 0.00 | 0.12 | 0.35 |
| logical_not | 1048576 | torch.float16 | torch.bool | 0.00 | 0.22 | 0.66 |
| logical_not | 4000000 | torch.float16 | torch.bool | 0.01 | 0.30 | 0.91 |

### torch

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| logical_not | 262144 | torch.float16 | torch.bool | 0.00 | 0.11 | 0.34 |
| logical_not | 1048576 | torch.float16 | torch.bool | 0.00 | 0.31 | 0.94 |
| logical_not | 4000000 | torch.float16 | torch.bool | 0.01 | 0.61 | 1.84 |

## bitwise_not

### tileops

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| bitwise_not | 262144 | torch.int32 | torch.int32 | 0.00 | 0.10 | 0.84 |
| bitwise_not | 1048576 | torch.int32 | torch.int32 | 0.01 | 0.20 | 1.59 |
| bitwise_not | 4000000 | torch.int32 | torch.int32 | 0.02 | 0.26 | 2.11 |

### torch

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| bitwise_not | 262144 | torch.int32 | torch.int32 | 0.00 | 0.09 | 0.71 |
| bitwise_not | 1048576 | torch.int32 | torch.int32 | 0.01 | 0.21 | 1.65 |
| bitwise_not | 4000000 | torch.int32 | torch.int32 | 0.01 | 0.28 | 2.26 |

## isnan

### tileops

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| isnan | 262144 | torch.float16 | torch.bool | 0.00 | 0.11 | 0.34 |
| isnan | 1048576 | torch.float16 | torch.bool | 0.00 | 0.22 | 0.66 |
| isnan | 4000000 | torch.float16 | torch.bool | 0.01 | 0.30 | 0.91 |

### torch

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| isnan | 262144 | torch.float16 | torch.bool | 0.00 | 0.10 | 0.31 |
| isnan | 1048576 | torch.float16 | torch.bool | 0.00 | 0.29 | 0.87 |
| isnan | 4000000 | torch.float16 | torch.bool | 0.01 | 0.58 | 1.73 |

## unary_strategy

### relu_direct

| n_total | shape_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4194304 | 1024x4096 | torch.float16 | direct | 0.01 | 0.29 | 1.17 |
| 4194304 | 1024x4096 | torch.bfloat16 | direct | 0.01 | 0.29 | 1.17 |
| 4194304 | 1024x4096 | torch.float32 | direct | 0.02 | 0.27 | 2.13 |
| 10485760 | 1024x10240 | torch.float16 | direct | 0.04 | 0.27 | 1.07 |
| 10485760 | 1024x10240 | torch.bfloat16 | direct | 0.04 | 0.27 | 1.07 |
| 10485760 | 1024x10240 | torch.float32 | direct | 0.04 | 0.25 | 2.04 |
| 20971520 | 1024x20480 | torch.float16 | direct | 0.08 | 0.26 | 1.05 |
| 20971520 | 1024x20480 | torch.bfloat16 | direct | 0.08 | 0.28 | 1.11 |
| 20971520 | 1024x20480 | torch.float32 | direct | 0.09 | 0.23 | 1.85 |

### relu_explicit_parallel

| n_total | shape_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4194304 | 1024x4096 | torch.float16 | explicit_parallel | 0.01 | 0.62 | 2.47 |
| 4194304 | 1024x4096 | torch.bfloat16 | explicit_parallel | 0.01 | 0.62 | 2.47 |
| 4194304 | 1024x4096 | torch.float32 | explicit_parallel | 0.01 | 0.39 | 3.09 |
| 10485760 | 1024x10240 | torch.float16 | explicit_parallel | 0.01 | 0.75 | 3.01 |
| 10485760 | 1024x10240 | torch.bfloat16 | explicit_parallel | 0.01 | 0.75 | 3.01 |
| 10485760 | 1024x10240 | torch.float32 | explicit_parallel | 0.02 | 0.45 | 3.64 |
| 20971520 | 1024x20480 | torch.float16 | explicit_parallel | 0.03 | 0.79 | 3.15 |
| 20971520 | 1024x20480 | torch.bfloat16 | explicit_parallel | 0.03 | 0.79 | 3.14 |
| 20971520 | 1024x20480 | torch.float32 | explicit_parallel | 0.06 | 0.37 | 2.96 |

### relu_register_copy

| n_total | shape_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4194304 | 1024x4096 | torch.float16 | register_copy | 0.01 | 0.66 | 2.65 |
| 4194304 | 1024x4096 | torch.bfloat16 | register_copy | 0.01 | 0.67 | 2.66 |
| 4194304 | 1024x4096 | torch.float32 | register_copy | 0.01 | 0.40 | 3.17 |
| 10485760 | 1024x10240 | torch.float16 | register_copy | 0.01 | 0.83 | 3.31 |
| 10485760 | 1024x10240 | torch.bfloat16 | register_copy | 0.01 | 0.82 | 3.29 |
| 10485760 | 1024x10240 | torch.float32 | register_copy | 0.02 | 0.46 | 3.71 |
| 20971520 | 1024x20480 | torch.float16 | register_copy | 0.02 | 0.93 | 3.72 |
| 20971520 | 1024x20480 | torch.bfloat16 | register_copy | 0.02 | 0.93 | 3.72 |
| 20971520 | 1024x20480 | torch.float32 | register_copy | 0.05 | 0.40 | 3.20 |

## L1NormOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | l1 | 0.01 | 0.56 | 1.11 |
| 1024 | 4096 | torch.bfloat16 | l1 | 0.01 | 0.56 | 1.12 |
| 1024 | 4096 | torch.float32 | l1 | 0.01 | 0.36 | 1.45 |
| 4096 | 4096 | torch.float16 | l1 | 0.02 | 0.71 | 1.43 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | l1 | 0.02 | 0.22 | 0.44 |
| 1024 | 4096 | torch.bfloat16 | l1 | 0.02 | 0.18 | 0.36 |
| 1024 | 4096 | torch.float32 | l1 | 0.03 | 0.15 | 0.61 |
| 4096 | 4096 | torch.float16 | l1 | 0.03 | 0.52 | 1.04 |

## L2NormOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | l2 | 0.01 | 0.55 | 1.10 |
| 1024 | 4096 | torch.bfloat16 | l2 | 0.01 | 0.56 | 1.11 |
| 1024 | 4096 | torch.float32 | l2 | 0.01 | 0.35 | 1.41 |
| 4096 | 4096 | torch.float16 | l2 | 0.02 | 0.70 | 1.41 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | l2 | 0.02 | 0.18 | 0.35 |
| 1024 | 4096 | torch.bfloat16 | l2 | 0.02 | 0.20 | 0.40 |
| 1024 | 4096 | torch.float32 | l2 | 0.03 | 0.15 | 0.61 |
| 4096 | 4096 | torch.float16 | l2 | 0.03 | 0.52 | 1.04 |

## InfNormOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | inf | 0.03 | 0.15 | 0.30 |
| 1024 | 4096 | torch.bfloat16 | inf | 0.03 | 0.15 | 0.30 |
| 1024 | 4096 | torch.float32 | inf | 0.03 | 0.12 | 0.50 |
| 4096 | 4096 | torch.float16 | inf | 0.06 | 0.29 | 0.59 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | inf | 0.02 | 0.21 | 0.43 |
| 1024 | 4096 | torch.bfloat16 | inf | 0.02 | 0.21 | 0.42 |
| 1024 | 4096 | torch.float32 | inf | 0.02 | 0.17 | 0.69 |
| 4096 | 4096 | torch.float16 | inf | 0.03 | 0.65 | 1.31 |

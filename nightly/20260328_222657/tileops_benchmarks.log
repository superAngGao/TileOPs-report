........................................................................ [  7%]
........................................................................ [ 15%]
........................................................................ [ 23%]
........................................................................ [ 31%]
........................................................................ [ 38%]
........................................................................ [ 46%]
........................................................................ [ 54%]
........................................................................ [ 62%]
........................................................................ [ 69%]
........................................................................ [ 77%]
........................................................................ [ 85%]
........................................................................ [ 93%]
..............................................................           [100%]Benchmark report saved to profile_run.log

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

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
926 passed, 17 warnings in 3034.19s (0:50:34)

===== profile_run.log summary =====
# TileOPs Benchmark Report
Generated: 2026-03-28 19:59:05

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
| 4194304 | torch.float16 | 0.01 | 0.55 | 2.20 |
| 4194304 | torch.bfloat16 | 0.01 | 0.55 | 2.20 |
| 4194304 | torch.float32 | 0.01 | 0.33 | 2.66 |

### torch

| n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 4096 | torch.float16 | 0.00 | 0.00 | 0.01 |
| 4194304 | torch.float16 | 0.01 | 0.52 | 2.08 |
| 4194304 | torch.bfloat16 | 0.01 | 0.54 | 2.16 |
| 4194304 | torch.float32 | 0.01 | 0.32 | 2.58 |

## r3_jit_cost

### relu_jit

| n_total | cold_ms | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 1000 | 19.25 | 0.00 | 0.00 | 0.00 |
| 2000 | 19.14 | 0.00 | 0.00 | 0.00 |
| 4000 | 19.01 | 0.00 | 0.00 | 0.01 |
| 8000 | 19.04 | 0.00 | 0.00 | 0.01 |
| 16000 | 18.6 | 0.00 | 0.01 | 0.03 |
| 32000 | 18.89 | 0.00 | 0.01 | 0.06 |
| 64000 | 19.21 | 0.00 | 0.03 | 0.11 |
| 128000 | 19.04 | 0.00 | 0.05 | 0.20 |
| 256000 | 18.68 | 0.00 | 0.10 | 0.38 |
| 512000 | 21.84 | 0.00 | 0.16 | 0.66 |

## r4_strategy_unary

### relu_direct

| n_total | size_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | torch.float16 | direct | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.bfloat16 | direct | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.float32 | direct | 0.00 | 0.00 | 0.02 |
| 4194304 | 4M | torch.float16 | direct | 0.02 | 0.24 | 0.95 |
| 4194304 | 4M | torch.bfloat16 | direct | 0.02 | 0.24 | 0.96 |
| 4194304 | 4M | torch.float32 | direct | 0.02 | 0.22 | 1.75 |
| 16777216 | 16M | torch.float16 | direct | 0.07 | 0.25 | 1.01 |
| 16777216 | 16M | torch.bfloat16 | direct | 0.07 | 0.25 | 1.01 |
| 16777216 | 16M | torch.float32 | direct | 0.07 | 0.23 | 1.81 |

### relu_explicit_parallel

| n_total | size_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | torch.float16 | explicit_parallel | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.bfloat16 | explicit_parallel | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.float32 | explicit_parallel | 0.00 | 0.00 | 0.01 |
| 4194304 | 4M | torch.float16 | explicit_parallel | 0.01 | 0.51 | 2.04 |
| 4194304 | 4M | torch.bfloat16 | explicit_parallel | 0.01 | 0.51 | 2.03 |
| 4194304 | 4M | torch.float32 | explicit_parallel | 0.01 | 0.32 | 2.59 |
| 16777216 | 16M | torch.float16 | explicit_parallel | 0.03 | 0.66 | 2.63 |
| 16777216 | 16M | torch.bfloat16 | explicit_parallel | 0.03 | 0.66 | 2.62 |
| 16777216 | 16M | torch.float32 | explicit_parallel | 0.04 | 0.39 | 3.08 |

### relu_register_copy

| n_total | size_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | torch.float16 | register_copy | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.bfloat16 | register_copy | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.float32 | register_copy | 0.00 | 0.00 | 0.01 |
| 4194304 | 4M | torch.float16 | register_copy | 0.01 | 0.55 | 2.19 |
| 4194304 | 4M | torch.bfloat16 | register_copy | 0.01 | 0.55 | 2.18 |
| 4194304 | 4M | torch.float32 | register_copy | 0.01 | 0.33 | 2.64 |
| 16777216 | 16M | torch.float16 | register_copy | 0.02 | 0.75 | 2.99 |
| 16777216 | 16M | torch.bfloat16 | register_copy | 0.02 | 0.75 | 2.99 |
| 16777216 | 16M | torch.float32 | register_copy | 0.04 | 0.39 | 3.14 |

## r4_strategy_gelu

### gelu_direct

| n_total | size_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | torch.float16 | direct | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.bfloat16 | direct | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.float32 | direct | 0.00 | 0.00 | 0.01 |
| 4194304 | 4M | torch.float16 | direct | 0.02 | 0.21 | 0.85 |
| 4194304 | 4M | torch.bfloat16 | direct | 0.02 | 0.22 | 0.87 |
| 4194304 | 4M | torch.float32 | direct | 0.02 | 0.20 | 1.63 |
| 16777216 | 16M | torch.float16 | direct | 0.07 | 0.23 | 0.91 |
| 16777216 | 16M | torch.bfloat16 | direct | 0.07 | 0.23 | 0.90 |
| 16777216 | 16M | torch.float32 | direct | 0.08 | 0.21 | 1.67 |

### gelu_explicit_parallel

| n_total | size_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | torch.float16 | explicit_parallel | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.bfloat16 | explicit_parallel | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.float32 | explicit_parallel | 0.00 | 0.00 | 0.01 |
| 4194304 | 4M | torch.float16 | explicit_parallel | 0.01 | 0.37 | 1.48 |
| 4194304 | 4M | torch.bfloat16 | explicit_parallel | 0.01 | 0.36 | 1.45 |
| 4194304 | 4M | torch.float32 | explicit_parallel | 0.01 | 0.32 | 2.57 |
| 16777216 | 16M | torch.float16 | explicit_parallel | 0.04 | 0.43 | 1.71 |
| 16777216 | 16M | torch.bfloat16 | explicit_parallel | 0.04 | 0.42 | 1.69 |
| 16777216 | 16M | torch.float32 | explicit_parallel | 0.05 | 0.37 | 2.95 |

### gelu_register_copy

| n_total | size_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | torch.float16 | register_copy | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.bfloat16 | register_copy | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.float32 | register_copy | 0.00 | 0.00 | 0.01 |
| 4194304 | 4M | torch.float16 | register_copy | 0.01 | 0.40 | 1.60 |
| 4194304 | 4M | torch.bfloat16 | register_copy | 0.01 | 0.39 | 1.57 |
| 4194304 | 4M | torch.float32 | register_copy | 0.01 | 0.33 | 2.61 |
| 16777216 | 16M | torch.float16 | register_copy | 0.03 | 0.49 | 1.95 |
| 16777216 | 16M | torch.bfloat16 | register_copy | 0.04 | 0.48 | 1.90 |
| 16777216 | 16M | torch.float32 | register_copy | 0.04 | 0.38 | 3.02 |

## r5_boundary

### relu_aligned

| n_total | align_label | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 2048000 | aligned | 0.01 | 0.37 | 1.49 |

### relu_unaligned_plus_1

| n_total | align_label | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 2048001 | unaligned_plus_1 | 0.01 | 0.24 | 0.97 |

### relu_unaligned_plus_127

| n_total | align_label | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 2048127 | unaligned_plus_127 | 0.01 | 0.24 | 0.98 |

## r6_threads

### relu_t128

| n_total | size_label | op_name | threads | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | relu | 128 | 0.00 | 0.00 | 0.01 |
| 4194304 | 4M | relu | 128 | 0.02 | 0.17 | 0.69 |
| 16777216 | 16M | relu | 128 | 0.09 | 0.19 | 0.77 |

### relu_t256

| n_total | size_label | op_name | threads | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | relu | 256 | 0.00 | 0.00 | 0.01 |
| 4194304 | 4M | relu | 256 | 0.02 | 0.17 | 0.70 |
| 16777216 | 16M | relu | 256 | 0.09 | 0.18 | 0.74 |

### erf_t128

| n_total | size_label | op_name | threads | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | erf | 128 | 0.00 | 0.00 | 0.01 |
| 4194304 | 4M | erf | 128 | 0.01 | 0.41 | 1.65 |
| 16777216 | 16M | erf | 128 | 0.03 | 0.49 | 1.97 |

### erf_t256

| n_total | size_label | op_name | threads | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | erf | 256 | 0.00 | 0.00 | 0.01 |
| 4194304 | 4M | erf | 256 | 0.01 | 0.41 | 1.64 |
| 16777216 | 16M | erf | 256 | 0.03 | 0.49 | 1.95 |

### mish_t128

| n_total | size_label | op_name | threads | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | mish | 128 | 0.00 | 0.00 | 0.01 |
| 4194304 | 4M | mish | 128 | 0.01 | 0.30 | 1.20 |
| 16777216 | 16M | mish | 128 | 0.05 | 0.33 | 1.34 |

### mish_t256

| n_total | size_label | op_name | threads | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | mish | 256 | 0.00 | 0.00 | 0.01 |
| 4194304 | 4M | mish | 256 | 0.01 | 0.29 | 1.17 |
| 16777216 | 16M | mish | 256 | 0.05 | 0.33 | 1.33 |

## r7_dtype_npt

### relu_fp32_npt4

| dtype_label | num_per_thread | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| fp32 | 4 | 0.01 | 0.19 | 1.50 |

### relu_fp32_npt8

| dtype_label | num_per_thread | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| fp32 | 8 | 0.01 | 0.18 | 1.47 |

### relu_fp16_npt4

| dtype_label | num_per_thread | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| fp16 | 4 | 0.00 | 0.24 | 0.97 |

### relu_fp16_npt8

| dtype_label | num_per_thread | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| fp16 | 8 | 0.01 | 0.19 | 0.74 |

## AdaLayerNormOp

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.02 | 1.06 | 1.69 |
| 4096 | 4096 | torch.bfloat16 | 0.07 | 1.21 | 1.94 |
| 1024 | 3000 | torch.float16 | 0.04 | 0.38 | 0.61 |
| 1025 | 4096 | torch.float16 | 0.02 | 1.05 | 1.68 |

### torch-ref

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.03 | 0.77 | 1.24 |
| 4096 | 4096 | torch.bfloat16 | 0.09 | 0.88 | 1.41 |
| 1024 | 3000 | torch.float16 | 0.02 | 0.68 | 1.09 |
| 1025 | 4096 | torch.float16 | 0.03 | 0.76 | 1.21 |

## AdaLayerNormZeroOp

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.02 | 1.02 | 1.70 |
| 4096 | 4096 | torch.bfloat16 | 0.09 | 1.16 | 1.93 |
| 1024 | 3000 | torch.float16 | 0.05 | 0.35 | 0.58 |
| 1025 | 4096 | torch.float16 | 0.02 | 1.02 | 1.70 |

### torch-ref

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.03 | 0.74 | 1.23 |
| 4096 | 4096 | torch.bfloat16 | 0.12 | 0.82 | 1.36 |
| 1024 | 3000 | torch.float16 | 0.03 | 0.65 | 1.09 |
| 1025 | 4096 | torch.float16 | 0.03 | 0.73 | 1.22 |

## ArgmaxOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | argmax | 6.16 | 0.00 | 0.00 |
| 1024 | 4096 | torch.bfloat16 | argmax | 6.16 | 0.00 | 0.00 |
| 4096 | 4096 | torch.float16 | argmax | 21.37 | 0.00 | 0.00 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | argmax | 0.03 | 0.15 | 0.31 |
| 1024 | 4096 | torch.bfloat16 | argmax | 0.03 | 0.15 | 0.31 |
| 4096 | 4096 | torch.float16 | argmax | 0.03 | 0.48 | 0.96 |

## ArgminOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | argmin | 6.95 | 0.00 | 0.00 |
| 1024 | 4096 | torch.bfloat16 | argmin | 6.97 | 0.00 | 0.00 |
| 4096 | 4096 | torch.float16 | argmin | 24.18 | 0.00 | 0.00 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | argmin | 0.03 | 0.15 | 0.31 |
| 1024 | 4096 | torch.bfloat16 | argmin | 0.03 | 0.15 | 0.31 |
| 4096 | 4096 | torch.float16 | argmin | 0.03 | 0.48 | 0.96 |

## BatchNormFwdOp

### tileops

| N | C | spatial | dtype | training | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 64 | () | torch.float16 | True | 0.01 | 0.00 | 0.00 |
| 8 | 64 | (32, 32) | torch.float16 | True | 0.01 | 0.43 | 0.17 |
| 4 | 128 | (32, 32) | torch.float16 | True | 0.01 | 0.44 | 0.18 |
| 4 | 256 | (28, 28) | torch.float16 | True | 0.02 | 0.51 | 0.20 |
| 4 | 128 | (1024, 1024) | torch.float16 | True | 7.88 | 0.68 | 0.27 |
| 4 | 256 | (1024, 1024) | torch.float16 | True | 14.67 | 0.73 | 0.29 |

### torch-cudnn

| N | C | spatial | dtype | training | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 64 | () | torch.float16 | True | 0.02 | 0.00 | 0.00 |
| 8 | 64 | (32, 32) | torch.float16 | True | 0.02 | 0.34 | 0.14 |
| 4 | 128 | (32, 32) | torch.float16 | True | 0.01 | 0.36 | 0.15 |
| 4 | 256 | (28, 28) | torch.float16 | True | 0.02 | 0.52 | 0.21 |
| 4 | 128 | (1024, 1024) | torch.float16 | True | 8.26 | 0.65 | 0.26 |
| 4 | 256 | (1024, 1024) | torch.float16 | True | 14.58 | 0.74 | 0.29 |

## BatchNormBwdOp

### tileops

| N | C | spatial | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 64 | () | torch.float16 | 0.01 | 0.00 | 0.00 |
| 8 | 64 | (32, 32) | torch.float16 | 0.02 | 0.26 | 0.19 |
| 4 | 128 | (32, 32) | torch.float16 | 0.02 | 0.26 | 0.20 |
| 4 | 256 | (28, 28) | torch.float16 | 0.02 | 0.32 | 0.24 |
| 4 | 128 | (1024, 1024) | torch.float16 | 12.46 | 0.34 | 0.26 |
| 4 | 256 | (1024, 1024) | torch.float16 | 22.56 | 0.38 | 0.29 |

### torch-autograd

| N | C | spatial | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 64 | () | torch.float16 | 0.02 | 0.00 | 0.00 |
| 8 | 64 | (32, 32) | torch.float16 | 0.03 | 0.16 | 0.12 |
| 4 | 128 | (32, 32) | torch.float16 | 0.02 | 0.19 | 0.14 |
| 4 | 256 | (28, 28) | torch.float16 | 0.02 | 0.27 | 0.20 |
| 4 | 128 | (1024, 1024) | torch.float16 | 24.15 | 0.18 | 0.13 |
| 4 | 256 | (1024, 1024) | torch.float16 | 36.35 | 0.24 | 0.18 |

## r1_vectorization

### add_same_shape_1d

| pattern_name | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| same_shape_1d | 1000000 | 0.01 | 0.20 | 1.19 |

### torch-same_shape_1d

| pattern_name | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| same_shape_1d | 1000000 | 0.00 | 0.21 | 1.28 |

### add_bias_add_2d

| pattern_name | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| bias_add_2d | 1000000 | 0.00 | 0.23 | 0.91 |

### torch-bias_add_2d

| pattern_name | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| bias_add_2d | 1000000 | 0.01 | 0.14 | 0.57 |

### add_interleaved_3d

| pattern_name | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| interleaved_3d | 1048576 | 0.00 | 0.35 | 0.71 |

### torch-interleaved_3d

| pattern_name | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| interleaved_3d | 1048576 | 0.01 | 0.16 | 0.32 |

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
| broadcast_3d | 4096 | 0.00 | 0.00 | 0.00 |

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
| 4K | same_shape | direct | 4096 | 0.00 | 0.00 | 0.02 |
| 1M | same_shape | direct | 1048576 | 0.01 | 0.16 | 0.95 |
| 1M | same_shape | direct | 1048576 | 0.01 | 0.16 | 0.94 |
| 1M | same_shape | direct | 1048576 | 0.01 | 0.14 | 1.65 |
| 16M | same_shape | direct | 16777216 | 0.07 | 0.22 | 1.35 |
| 16M | same_shape | direct | 16777216 | 0.07 | 0.22 | 1.35 |
| 16M | same_shape | direct | 16777216 | 0.09 | 0.19 | 2.28 |

### add_direct_bias_add_2d

| size_label | pattern_name | strategy | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4K | bias_add_2d | direct | 4096 | 0.00 | 0.00 | 0.01 |
| 4K | bias_add_2d | direct | 4096 | 0.00 | 0.00 | 0.01 |
| 4K | bias_add_2d | direct | 4096 | 0.00 | 0.00 | 0.02 |
| 1M | bias_add_2d | direct | 1048576 | 0.01 | 0.16 | 0.66 |
| 1M | bias_add_2d | direct | 1048576 | 0.01 | 0.17 | 0.67 |
| 1M | bias_add_2d | direct | 1048576 | 0.01 | 0.15 | 1.24 |
| 16M | bias_add_2d | direct | 16777216 | 0.07 | 0.25 | 0.99 |
| 16M | bias_add_2d | direct | 16777216 | 0.07 | 0.25 | 0.99 |
| 16M | bias_add_2d | direct | 16777216 | 0.08 | 0.22 | 1.78 |

### add_direct_interleaved_3d

| size_label | pattern_name | strategy | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4K | interleaved_3d | direct | 4096 | 0.00 | 0.00 | 0.00 |
| 4K | interleaved_3d | direct | 4096 | 0.00 | 0.00 | 0.00 |
| 4K | interleaved_3d | direct | 4096 | 0.00 | 0.00 | 0.01 |
| 1M | interleaved_3d | direct | 1048576 | 0.01 | 0.20 | 0.39 |
| 1M | interleaved_3d | direct | 1048576 | 0.01 | 0.20 | 0.39 |
| 1M | interleaved_3d | direct | 1048576 | 0.01 | 0.19 | 0.75 |
| 16M | interleaved_3d | direct | 16777216 | 0.05 | 0.32 | 0.63 |
| 16M | interleaved_3d | direct | 16777216 | 0.05 | 0.31 | 0.63 |
| 16M | interleaved_3d | direct | 16777216 | 0.05 | 0.31 | 1.24 |

### add_explicit_parallel_same_shape

| size_label | pattern_name | strategy | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4K | same_shape | explicit_parallel | 4096 | 0.00 | 0.00 | 0.01 |
| 4K | same_shape | explicit_parallel | 4096 | 0.00 | 0.00 | 0.01 |
| 4K | same_shape | explicit_parallel | 4096 | 0.00 | 0.00 | 0.02 |
| 1M | same_shape | explicit_parallel | 1048576 | 0.00 | 0.22 | 1.32 |
| 1M | same_shape | explicit_parallel | 1048576 | 0.00 | 0.22 | 1.32 |
| 1M | same_shape | explicit_parallel | 1048576 | 0.01 | 0.16 | 1.93 |
| 16M | same_shape | explicit_parallel | 16777216 | 0.04 | 0.48 | 2.87 |
| 16M | same_shape | explicit_parallel | 16777216 | 0.04 | 0.48 | 2.86 |
| 16M | same_shape | explicit_parallel | 16777216 | 0.06 | 0.26 | 3.11 |

### add_explicit_parallel_bias_add_2d

| size_label | pattern_name | strategy | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4K | bias_add_2d | explicit_parallel | 4096 | 0.00 | 0.00 | 0.01 |
| 4K | bias_add_2d | explicit_parallel | 4096 | 0.00 | 0.00 | 0.01 |
| 4K | bias_add_2d | explicit_parallel | 4096 | 0.00 | 0.00 | 0.01 |
| 1M | bias_add_2d | explicit_parallel | 1048576 | 0.00 | 0.26 | 1.04 |
| 1M | bias_add_2d | explicit_parallel | 1048576 | 0.00 | 0.26 | 1.04 |
| 1M | bias_add_2d | explicit_parallel | 1048576 | 0.01 | 0.20 | 1.61 |
| 16M | bias_add_2d | explicit_parallel | 16777216 | 0.03 | 0.67 | 2.66 |
| 16M | bias_add_2d | explicit_parallel | 16777216 | 0.03 | 0.66 | 2.65 |
| 16M | bias_add_2d | explicit_parallel | 16777216 | 0.04 | 0.39 | 3.14 |

### add_explicit_parallel_interleaved_3d

| size_label | pattern_name | strategy | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4K | interleaved_3d | explicit_parallel | 4096 | 0.00 | 0.00 | 0.00 |
| 4K | interleaved_3d | explicit_parallel | 4096 | 0.00 | 0.00 | 0.00 |
| 4K | interleaved_3d | explicit_parallel | 4096 | 0.00 | 0.00 | 0.01 |
| 1M | interleaved_3d | explicit_parallel | 1048576 | 0.00 | 0.35 | 0.70 |
| 1M | interleaved_3d | explicit_parallel | 1048576 | 0.00 | 0.35 | 0.70 |
| 1M | interleaved_3d | explicit_parallel | 1048576 | 0.00 | 0.33 | 1.31 |
| 16M | interleaved_3d | explicit_parallel | 16777216 | 0.02 | 0.98 | 1.96 |
| 16M | interleaved_3d | explicit_parallel | 16777216 | 0.02 | 0.98 | 1.97 |
| 16M | interleaved_3d | explicit_parallel | 16777216 | 0.02 | 0.79 | 3.18 |

## r4_where

### tileops-where

| n_total | size_label | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 4096 | 4K | 0.00 | 0.00 | 0.01 |
| 1048576 | 1M | 0.01 | 0.20 | 1.43 |
| 16777216 | 16M | 0.04 | 0.43 | 2.98 |

### torch

| n_total | size_label | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 4096 | 4K | 0.00 | 0.00 | 0.01 |
| 1048576 | 1M | 0.01 | 0.21 | 1.44 |
| 16777216 | 16M | 0.04 | 0.43 | 3.02 |

## AddOp

### tileops

| n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 4194304 | torch.float16 | 0.01 | 0.39 | 2.35 |
| 4194304 | torch.bfloat16 | 0.01 | 0.39 | 2.34 |
| 4194304 | torch.float32 | 0.02 | 0.23 | 2.74 |

### torch

| n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 4194304 | torch.float16 | 0.01 | 0.39 | 2.35 |
| 4194304 | torch.bfloat16 | 0.01 | 0.39 | 2.36 |
| 4194304 | torch.float32 | 0.02 | 0.23 | 2.76 |

## sub

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| sub | torch.float16 | torch.float16 | 0.01 | 0.39 | 2.34 |
| sub | torch.float16 | torch.float16 | 0.02 | 0.47 | 2.83 |
| sub | torch.float16 | torch.float16 | 0.04 | 0.51 | 3.08 |

### torch

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| sub | torch.float16 | torch.float16 | 0.01 | 0.39 | 2.35 |
| sub | torch.float16 | torch.float16 | 0.02 | 0.47 | 2.83 |
| sub | torch.float16 | torch.float16 | 0.04 | 0.52 | 3.11 |

## mul

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| mul | torch.float16 | torch.float16 | 0.01 | 0.39 | 2.34 |
| mul | torch.float16 | torch.float16 | 0.02 | 0.47 | 2.83 |
| mul | torch.float16 | torch.float16 | 0.04 | 0.51 | 3.08 |

### torch

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| mul | torch.float16 | torch.float16 | 0.01 | 0.39 | 2.36 |
| mul | torch.float16 | torch.float16 | 0.03 | 0.41 | 2.49 |
| mul | torch.float16 | torch.float16 | 0.04 | 0.52 | 3.10 |

## div

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| div | torch.float16 | torch.float16 | 0.01 | 0.38 | 2.30 |
| div | torch.float16 | torch.float16 | 0.02 | 0.47 | 2.81 |
| div | torch.float16 | torch.float16 | 0.04 | 0.50 | 3.01 |

### torch

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| div | torch.float16 | torch.float16 | 0.01 | 0.38 | 2.27 |
| div | torch.float16 | torch.float16 | 0.02 | 0.47 | 2.81 |
| div | torch.float16 | torch.float16 | 0.04 | 0.50 | 2.99 |

## remainder

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| remainder | torch.float16 | torch.float16 | 0.01 | 0.38 | 2.25 |
| remainder | torch.float16 | torch.float16 | 0.02 | 0.46 | 2.78 |

### torch

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| remainder | torch.float16 | torch.float16 | 0.01 | 0.34 | 2.04 |
| remainder | torch.float16 | torch.float16 | 0.03 | 0.41 | 2.48 |

## pow

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| pow | torch.float16 | torch.float16 | 0.02 | 0.20 | 1.21 |
| pow | torch.float16 | torch.float16 | 0.05 | 0.22 | 1.32 |

### torch

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| pow | torch.float16 | torch.float16 | 0.02 | 0.20 | 1.21 |
| pow | torch.float16 | torch.float16 | 0.05 | 0.22 | 1.30 |

## floor_divide

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| floor_divide | torch.float16 | torch.float16 | 0.01 | 0.38 | 2.28 |
| floor_divide | torch.float16 | torch.float16 | 0.02 | 0.47 | 2.80 |

### torch

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| floor_divide | torch.float16 | torch.float16 | 0.03 | 0.15 | 0.90 |
| floor_divide | torch.float16 | torch.float16 | 0.07 | 0.16 | 0.95 |

## lerp

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| lerp | torch.float16 | torch.float16 | 0.01 | 0.39 | 2.34 |
| lerp | torch.float16 | torch.float16 | 0.02 | 0.47 | 2.81 |

### torch

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| lerp | torch.float16 | torch.float16 | 0.01 | 0.39 | 2.36 |
| lerp | torch.float16 | torch.float16 | 0.02 | 0.47 | 2.85 |

## maximum

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| maximum | torch.float16 | torch.float16 | 0.01 | 0.38 | 2.29 |
| maximum | torch.float16 | torch.float16 | 0.02 | 0.47 | 2.79 |
| maximum | torch.float16 | torch.float16 | 0.04 | 0.51 | 3.05 |

### torch

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| maximum | torch.float16 | torch.float16 | 0.01 | 0.38 | 2.29 |
| maximum | torch.float16 | torch.float16 | 0.02 | 0.47 | 2.80 |
| maximum | torch.float16 | torch.float16 | 0.04 | 0.51 | 3.07 |

## minimum

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| minimum | torch.float16 | torch.float16 | 0.01 | 0.38 | 2.28 |
| minimum | torch.float16 | torch.float16 | 0.02 | 0.46 | 2.79 |
| minimum | torch.float16 | torch.float16 | 0.04 | 0.51 | 3.05 |

### torch

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| minimum | torch.float16 | torch.float16 | 0.01 | 0.38 | 2.30 |
| minimum | torch.float16 | torch.float16 | 0.02 | 0.47 | 2.82 |
| minimum | torch.float16 | torch.float16 | 0.04 | 0.51 | 3.05 |

## cmp_eq

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| eq | torch.float16 | 0.02 | 0.19 | 0.96 |
| eq | torch.float16 | 0.05 | 0.22 | 1.12 |

### torch

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| eq | torch.float16 | 0.01 | 0.43 | 2.17 |
| eq | torch.float16 | 0.02 | 0.54 | 2.68 |

## cmp_ne

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| ne | torch.float16 | 0.02 | 0.19 | 0.96 |

### torch

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| ne | torch.float16 | 0.01 | 0.43 | 2.17 |

## cmp_gt

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| gt | torch.float16 | 0.02 | 0.19 | 0.96 |

### torch

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| gt | torch.float16 | 0.01 | 0.43 | 2.17 |

## cmp_lt

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| lt | torch.float16 | 0.02 | 0.19 | 0.96 |

### torch

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| lt | torch.float16 | 0.01 | 0.43 | 2.15 |

## cmp_ge

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| ge | torch.float16 | 0.02 | 0.19 | 0.96 |

### torch

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| ge | torch.float16 | 0.01 | 0.43 | 2.16 |

## cmp_le

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| le | torch.float16 | 0.02 | 0.19 | 0.96 |

### torch

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| le | torch.float16 | 0.01 | 0.42 | 2.12 |

## logical_and

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| logical_and | torch.float16 | 0.02 | 0.19 | 0.96 |
| logical_and | torch.float16 | 0.05 | 0.22 | 1.12 |

### torch

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| logical_and | torch.float16 | 0.01 | 0.61 | 3.03 |
| logical_and | torch.float16 | 0.01 | 0.80 | 3.99 |

## logical_or

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| logical_or | torch.float16 | 0.02 | 0.19 | 0.96 |
| logical_or | torch.float16 | 0.05 | 0.22 | 1.12 |

### torch

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| logical_or | torch.float16 | 0.01 | 0.61 | 3.05 |
| logical_or | torch.float16 | 0.01 | 0.80 | 4.02 |

## bitwise_and

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| bitwise_and | torch.int32 | 0.02 | 0.22 | 2.66 |
| bitwise_and | torch.int32 | 0.04 | 0.25 | 2.95 |

### torch

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| bitwise_and | torch.int32 | 0.02 | 0.23 | 2.76 |
| bitwise_and | torch.int32 | 0.04 | 0.26 | 3.07 |

## bitwise_or

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| bitwise_or | torch.int32 | 0.02 | 0.22 | 2.66 |

### torch

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| bitwise_or | torch.int32 | 0.02 | 0.23 | 2.75 |

## bitwise_xor

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| bitwise_xor | torch.int32 | 0.02 | 0.22 | 2.62 |

### torch

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| bitwise_xor | torch.int32 | 0.02 | 0.23 | 2.75 |

## gelu_and_mul

### tileops

| op_name | M | N | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| gelu_and_mul | 1024 | 4096 | torch.float16 | 0.01 | 0.65 | 1.94 |
| gelu_and_mul | 1024 | 10240 | torch.float16 | 0.03 | 0.73 | 2.18 |
| gelu_and_mul | 1024 | 20480 | torch.float16 | 0.06 | 0.76 | 2.27 |

### torch-ref

| op_name | M | N | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| gelu_and_mul | 1024 | 4096 | torch.float16 | 0.04 | 0.23 | 0.69 |
| gelu_and_mul | 1024 | 10240 | torch.float16 | 0.08 | 0.25 | 0.74 |
| gelu_and_mul | 1024 | 20480 | torch.float16 | 0.17 | 0.24 | 0.73 |

## gelu_tanh_and_mul

### tileops

| op_name | M | N | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| gelu_tanh_and_mul | 1024 | 4096 | torch.float16 | 0.01 | 0.76 | 2.28 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.float16 | 0.02 | 0.87 | 2.62 |
| gelu_tanh_and_mul | 1024 | 20480 | torch.float16 | 0.05 | 0.93 | 2.78 |

### torch-ref

| op_name | M | N | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| gelu_tanh_and_mul | 1024 | 4096 | torch.float16 | 0.03 | 0.24 | 0.72 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.float16 | 0.08 | 0.26 | 0.77 |
| gelu_tanh_and_mul | 1024 | 20480 | torch.float16 | 0.16 | 0.25 | 0.76 |

## silu_and_mul_strategy

### tileops-direct

| op_name | M | N | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| silu_and_mul | 1024 | 4096 | torch.float16 | direct | 0.02 | 0.41 | 1.23 |
| silu_and_mul | 1024 | 4096 | torch.bfloat16 | direct | 0.02 | 0.41 | 1.23 |
| silu_and_mul | 1024 | 4096 | torch.float32 | direct | 0.02 | 0.35 | 2.10 |
| silu_and_mul | 1024 | 10240 | torch.float16 | direct | 0.05 | 0.42 | 1.27 |
| silu_and_mul | 1024 | 10240 | torch.bfloat16 | direct | 0.05 | 0.43 | 1.28 |
| silu_and_mul | 1024 | 10240 | torch.float32 | direct | 0.06 | 0.37 | 2.23 |
| silu_and_mul | 4096 | 4096 | torch.float16 | direct | 0.08 | 0.42 | 1.27 |
| silu_and_mul | 4096 | 4096 | torch.bfloat16 | direct | 0.08 | 0.43 | 1.28 |
| silu_and_mul | 4096 | 4096 | torch.float32 | direct | 0.09 | 0.37 | 2.22 |

### tileops-explicit_parallel

| op_name | M | N | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| silu_and_mul | 1024 | 4096 | torch.float16 | explicit_parallel | 0.01 | 0.75 | 2.24 |
| silu_and_mul | 1024 | 4096 | torch.bfloat16 | explicit_parallel | 0.01 | 0.75 | 2.26 |
| silu_and_mul | 1024 | 4096 | torch.float32 | explicit_parallel | 0.02 | 0.45 | 2.72 |
| silu_and_mul | 1024 | 10240 | torch.float16 | explicit_parallel | 0.02 | 0.88 | 2.63 |
| silu_and_mul | 1024 | 10240 | torch.bfloat16 | explicit_parallel | 0.02 | 0.86 | 2.57 |
| silu_and_mul | 1024 | 10240 | torch.float32 | explicit_parallel | 0.04 | 0.51 | 3.06 |
| silu_and_mul | 4096 | 4096 | torch.float16 | explicit_parallel | 0.04 | 0.90 | 2.69 |
| silu_and_mul | 4096 | 4096 | torch.bfloat16 | explicit_parallel | 0.04 | 0.90 | 2.69 |
| silu_and_mul | 4096 | 4096 | torch.float32 | explicit_parallel | 0.06 | 0.52 | 3.10 |

## gelu_and_mul_strategy

### tileops-direct

| op_name | M | N | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gelu_and_mul | 1024 | 4096 | torch.float16 | direct | 0.02 | 0.41 | 1.22 |
| gelu_and_mul | 1024 | 4096 | torch.bfloat16 | direct | 0.02 | 0.40 | 1.21 |
| gelu_and_mul | 1024 | 4096 | torch.float32 | direct | 0.02 | 0.35 | 2.12 |
| gelu_and_mul | 1024 | 10240 | torch.float16 | direct | 0.05 | 0.42 | 1.25 |
| gelu_and_mul | 1024 | 10240 | torch.bfloat16 | direct | 0.05 | 0.42 | 1.25 |
| gelu_and_mul | 1024 | 10240 | torch.float32 | direct | 0.06 | 0.37 | 2.25 |
| gelu_and_mul | 4096 | 4096 | torch.float16 | direct | 0.08 | 0.42 | 1.26 |
| gelu_and_mul | 4096 | 4096 | torch.bfloat16 | direct | 0.08 | 0.42 | 1.25 |
| gelu_and_mul | 4096 | 4096 | torch.float32 | direct | 0.09 | 0.37 | 2.23 |

### tileops-explicit_parallel

| op_name | M | N | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gelu_and_mul | 1024 | 4096 | torch.float16 | explicit_parallel | 0.01 | 0.65 | 1.95 |
| gelu_and_mul | 1024 | 4096 | torch.bfloat16 | explicit_parallel | 0.01 | 0.64 | 1.93 |
| gelu_and_mul | 1024 | 4096 | torch.float32 | explicit_parallel | 0.02 | 0.45 | 2.72 |
| gelu_and_mul | 1024 | 10240 | torch.float16 | explicit_parallel | 0.03 | 0.73 | 2.19 |
| gelu_and_mul | 1024 | 10240 | torch.bfloat16 | explicit_parallel | 0.03 | 0.72 | 2.16 |
| gelu_and_mul | 1024 | 10240 | torch.float32 | explicit_parallel | 0.04 | 0.50 | 2.99 |
| gelu_and_mul | 4096 | 4096 | torch.float16 | explicit_parallel | 0.04 | 0.79 | 2.38 |
| gelu_and_mul | 4096 | 4096 | torch.bfloat16 | explicit_parallel | 0.04 | 0.79 | 2.38 |
| gelu_and_mul | 4096 | 4096 | torch.float32 | explicit_parallel | 0.06 | 0.52 | 3.11 |

## gelu_tanh_and_mul_strategy

### tileops-direct

| op_name | M | N | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gelu_tanh_and_mul | 1024 | 4096 | torch.float16 | direct | 0.02 | 0.41 | 1.24 |
| gelu_tanh_and_mul | 1024 | 4096 | torch.bfloat16 | direct | 0.02 | 0.41 | 1.24 |
| gelu_tanh_and_mul | 1024 | 4096 | torch.float32 | direct | 0.02 | 0.36 | 2.14 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.float16 | direct | 0.05 | 0.43 | 1.29 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.bfloat16 | direct | 0.05 | 0.43 | 1.29 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.float32 | direct | 0.06 | 0.38 | 2.26 |
| gelu_tanh_and_mul | 4096 | 4096 | torch.float16 | direct | 0.08 | 0.43 | 1.28 |
| gelu_tanh_and_mul | 4096 | 4096 | torch.bfloat16 | direct | 0.08 | 0.43 | 1.28 |
| gelu_tanh_and_mul | 4096 | 4096 | torch.float32 | direct | 0.09 | 0.38 | 2.27 |

### tileops-explicit_parallel

| op_name | M | N | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gelu_tanh_and_mul | 1024 | 4096 | torch.float16 | explicit_parallel | 0.01 | 0.76 | 2.27 |
| gelu_tanh_and_mul | 1024 | 4096 | torch.bfloat16 | explicit_parallel | 0.01 | 0.76 | 2.27 |
| gelu_tanh_and_mul | 1024 | 4096 | torch.float32 | explicit_parallel | 0.02 | 0.46 | 2.74 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.float16 | explicit_parallel | 0.02 | 0.88 | 2.63 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.bfloat16 | explicit_parallel | 0.02 | 0.87 | 2.60 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.float32 | explicit_parallel | 0.04 | 0.51 | 3.07 |
| gelu_tanh_and_mul | 4096 | 4096 | torch.float16 | explicit_parallel | 0.04 | 0.91 | 2.73 |
| gelu_tanh_and_mul | 4096 | 4096 | torch.bfloat16 | explicit_parallel | 0.04 | 0.91 | 2.73 |
| gelu_tanh_and_mul | 4096 | 4096 | torch.float32 | explicit_parallel | 0.06 | 0.52 | 3.11 |

## sub_bcast

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| sub | torch.float16 | 0.01 | 0.52 | 2.07 |
| sub | torch.float16 | 0.02 | 0.63 | 2.50 |
| sub | torch.float16 | 0.03 | 0.67 | 2.68 |

### torch

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| sub | torch.float16 | 0.02 | 0.24 | 0.96 |
| sub | torch.float16 | 0.04 | 0.27 | 1.09 |
| sub | torch.float16 | 0.07 | 0.28 | 1.12 |

## mul_bcast

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| mul | torch.float16 | 0.01 | 0.52 | 2.07 |
| mul | torch.float16 | 0.02 | 0.63 | 2.50 |
| mul | torch.float16 | 0.03 | 0.67 | 2.69 |

### torch

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| mul | torch.float16 | 0.02 | 0.24 | 0.97 |
| mul | torch.float16 | 0.04 | 0.27 | 1.10 |
| mul | torch.float16 | 0.07 | 0.28 | 1.13 |

## div_bcast

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| div | torch.float16 | 0.01 | 0.49 | 1.96 |
| div | torch.float16 | 0.02 | 0.60 | 2.39 |
| div | torch.float16 | 0.03 | 0.64 | 2.55 |

### torch

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| div | torch.float16 | 0.02 | 0.22 | 0.90 |
| div | torch.float16 | 0.04 | 0.25 | 1.02 |
| div | torch.float16 | 0.08 | 0.26 | 1.03 |

## binary_strategy

### add_direct

| n_total | shape_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4194304 | 1024x4096 | torch.float16 | direct | 0.02 | 0.21 | 1.28 |
| 4194304 | 1024x4096 | torch.bfloat16 | direct | 0.02 | 0.21 | 1.28 |
| 4194304 | 1024x4096 | torch.float32 | direct | 0.02 | 0.18 | 2.13 |
| 10485760 | 1024x10240 | torch.float16 | direct | 0.05 | 0.22 | 1.34 |
| 10485760 | 1024x10240 | torch.bfloat16 | direct | 0.05 | 0.22 | 1.34 |
| 10485760 | 1024x10240 | torch.float32 | direct | 0.05 | 0.19 | 2.30 |
| 20971520 | 1024x20480 | torch.float16 | direct | 0.09 | 0.22 | 1.35 |
| 20971520 | 1024x20480 | torch.bfloat16 | direct | 0.09 | 0.22 | 1.35 |
| 20971520 | 1024x20480 | torch.float32 | direct | 0.11 | 0.19 | 2.26 |

### add_explicit_parallel

| n_total | shape_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4194304 | 1024x4096 | torch.float16 | explicit_parallel | 0.01 | 0.38 | 2.28 |
| 4194304 | 1024x4096 | torch.bfloat16 | explicit_parallel | 0.01 | 0.38 | 2.27 |
| 4194304 | 1024x4096 | torch.float32 | explicit_parallel | 0.02 | 0.23 | 2.73 |
| 10485760 | 1024x10240 | torch.float16 | explicit_parallel | 0.02 | 0.45 | 2.72 |
| 10485760 | 1024x10240 | torch.bfloat16 | explicit_parallel | 0.02 | 0.45 | 2.69 |
| 10485760 | 1024x10240 | torch.float32 | explicit_parallel | 0.04 | 0.26 | 3.08 |
| 20971520 | 1024x20480 | torch.float16 | explicit_parallel | 0.04 | 0.49 | 2.93 |
| 20971520 | 1024x20480 | torch.bfloat16 | explicit_parallel | 0.04 | 0.49 | 2.93 |
| 20971520 | 1024x20480 | torch.float32 | explicit_parallel | 0.08 | 0.26 | 3.11 |

## conv1d

### tileops

| n | c_in | l_in | c_out | kernel_size | stride | padding | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 256 | 32000 | 512 | 1 | 1 | 0 | torch.float16 | 0.30 | 112.21 | 0.66 |
| 4 | 128 | 4096 | 256 | 3 | 1 | 1 | torch.float16 | 0.02 | 129.50 | 0.51 |
| 4 | 64 | 16000 | 128 | 5 | 2 | 2 | torch.float16 | 0.02 | 110.72 | 0.70 |
| 4 | 128 | 8192 | 256 | 7 | 1 | 3 | torch.float16 | 0.06 | 254.23 | 0.43 |
| 2 | 128 | 4096 | 256 | 3 | 2 | 1 | torch.bfloat16 | 0.01 | 68.88 | 0.38 |

### torch

| n | c_in | l_in | c_out | kernel_size | stride | padding | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 256 | 32000 | 512 | 1 | 1 | 0 | torch.float16 | 0.44 | 75.73 | 0.44 |
| 4 | 128 | 4096 | 256 | 3 | 1 | 1 | torch.float16 | 0.05 | 66.76 | 0.26 |
| 4 | 64 | 16000 | 128 | 5 | 2 | 2 | torch.float16 | 0.05 | 51.46 | 0.32 |
| 4 | 128 | 8192 | 256 | 7 | 1 | 3 | torch.float16 | 0.09 | 165.94 | 0.28 |
| 2 | 128 | 4096 | 256 | 3 | 2 | 1 | torch.bfloat16 | 0.03 | 29.85 | 0.16 |

## conv2d

### tileops

| n | c_in | h | w | c_out | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 64 | 56 | 56 | 64 | torch.float16 | 0.01 | 50.79 | 0.18 |
| 1 | 3 | 112 | 112 | 64 | torch.float16 | 0.01 | 1.34 | 0.06 |
| 1 | 128 | 56 | 56 | 256 | torch.float16 | 0.01 | 34.58 | 0.13 |
| 1 | 256 | 112 | 112 | 512 | torch.float16 | 0.08 | 362.94 | 0.27 |
| 1 | 64 | 56 | 56 | 128 | torch.float16 | 0.01 | 87.66 | 0.11 |
| 1 | 128 | 56 | 56 | 256 | torch.float16 | 0.03 | 45.80 | 0.10 |
| 1 | 128 | 28 | 28 | 128 | torch.bfloat16 | 0.01 | 5.01 | 0.05 |
| 2 | 64 | 56 | 56 | 256 | torch.float16 | 0.01 | 36.68 | 0.72 |
| 2 | 128 | 28 | 28 | 512 | torch.float16 | 0.01 | 38.24 | 0.40 |
| 2 | 512 | 28 | 28 | 128 | torch.float16 | 0.01 | 31.40 | 0.33 |
| 1 | 256 | 14 | 14 | 1024 | torch.float16 | 0.01 | 18.59 | 0.19 |
| 1 | 512 | 7 | 7 | 2048 | torch.float16 | 0.01 | 14.51 | 0.33 |
| 2 | 64 | 56 | 56 | 256 | torch.bfloat16 | 0.01 | 36.78 | 0.72 |

### torch-nchw

| n | c_in | h | w | c_out | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 64 | 56 | 56 | 64 | torch.float16 | 0.02 | 23.47 | 0.09 |
| 1 | 3 | 112 | 112 | 64 | torch.float16 | 0.01 | 0.98 | 0.04 |
| 1 | 128 | 56 | 56 | 256 | torch.float16 | 0.02 | 20.88 | 0.08 |
| 1 | 256 | 112 | 112 | 512 | torch.float16 | 0.11 | 264.87 | 0.19 |
| 1 | 64 | 56 | 56 | 128 | torch.float16 | 0.02 | 51.93 | 0.07 |
| 1 | 128 | 56 | 56 | 256 | torch.float16 | 0.03 | 40.46 | 0.09 |
| 1 | 128 | 28 | 28 | 128 | torch.bfloat16 | 0.02 | 2.85 | 0.03 |
| 2 | 64 | 56 | 56 | 256 | torch.float16 | 0.01 | 16.12 | 0.32 |
| 2 | 128 | 28 | 28 | 512 | torch.float16 | 0.01 | 20.64 | 0.21 |
| 2 | 512 | 28 | 28 | 128 | torch.float16 | 0.01 | 22.76 | 0.24 |
| 1 | 256 | 14 | 14 | 1024 | torch.float16 | 0.01 | 9.33 | 0.09 |
| 1 | 512 | 7 | 7 | 2048 | torch.float16 | 0.01 | 7.24 | 0.17 |
| 2 | 64 | 56 | 56 | 256 | torch.bfloat16 | 0.01 | 15.71 | 0.31 |

### torch-nhwc

| n | c_in | h | w | c_out | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 64 | 56 | 56 | 64 | torch.float16 | 0.01 | 31.97 | 0.12 |
| 1 | 3 | 112 | 112 | 64 | torch.float16 | 0.02 | 0.72 | 0.03 |
| 1 | 128 | 56 | 56 | 256 | torch.float16 | 0.02 | 27.66 | 0.11 |
| 1 | 256 | 112 | 112 | 512 | torch.float16 | 0.09 | 321.11 | 0.23 |
| 1 | 64 | 56 | 56 | 128 | torch.float16 | 0.02 | 63.40 | 0.08 |
| 1 | 128 | 56 | 56 | 256 | torch.float16 | 0.03 | 46.89 | 0.10 |
| 1 | 128 | 28 | 28 | 128 | torch.bfloat16 | 0.02 | 3.53 | 0.03 |
| 2 | 64 | 56 | 56 | 256 | torch.float16 | 0.01 | 17.23 | 0.34 |
| 2 | 128 | 28 | 28 | 512 | torch.float16 | 0.01 | 21.27 | 0.22 |
| 2 | 512 | 28 | 28 | 128 | torch.float16 | 0.01 | 22.86 | 0.24 |
| 1 | 256 | 14 | 14 | 1024 | torch.float16 | 0.01 | 11.69 | 0.12 |
| 1 | 512 | 7 | 7 | 2048 | torch.float16 | 0.01 | 11.21 | 0.26 |
| 2 | 64 | 56 | 56 | 256 | torch.bfloat16 | 0.01 | 17.37 | 0.34 |

## conv3d

### tileops

| n | c_in | d_in | h_in | w_in | c_out | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3 | 16 | 112 | 112 | 64 | torch.float16 | 0.09 | 23.67 | 0.31 |
| 1 | 64 | 8 | 56 | 56 | 128 | torch.float16 | 0.03 | 55.29 | 0.18 |
| 1 | 32 | 32 | 64 | 64 | 64 | torch.bfloat16 | 0.09 | 154.31 | 0.27 |

### torch-ncdhw

| n | c_in | d_in | h_in | w_in | c_out | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3 | 16 | 112 | 112 | 64 | torch.float16 | 0.16 | 13.23 | 0.17 |
| 1 | 64 | 8 | 56 | 56 | 128 | torch.float16 | 0.03 | 42.43 | 0.14 |
| 1 | 32 | 32 | 64 | 64 | 64 | torch.bfloat16 | 0.15 | 97.41 | 0.17 |

### torch-ndhwc

| n | c_in | d_in | h_in | w_in | c_out | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3 | 16 | 112 | 112 | 64 | torch.float16 | 0.12 | 17.70 | 0.23 |
| 1 | 64 | 8 | 56 | 56 | 128 | torch.float16 | 0.03 | 53.82 | 0.17 |
| 1 | 32 | 32 | 64 | 64 | 64 | torch.bfloat16 | 0.12 | 117.98 | 0.21 |

## CumsumOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | cumsum | 0.08 | 0.05 | 0.21 |
| 1024 | 4096 | torch.bfloat16 | cumsum | 0.08 | 0.05 | 0.21 |
| 4096 | 4096 | torch.float16 | cumsum | 0.18 | 0.09 | 0.37 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | cumsum | 0.12 | 0.04 | 0.14 |
| 1024 | 4096 | torch.bfloat16 | cumsum | 0.12 | 0.04 | 0.14 |
| 4096 | 4096 | torch.float16 | cumsum | 0.36 | 0.05 | 0.19 |

## CumprodOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | cumprod | 0.08 | 0.05 | 0.21 |
| 1024 | 4096 | torch.bfloat16 | cumprod | 0.08 | 0.05 | 0.21 |
| 4096 | 4096 | torch.float16 | cumprod | 0.18 | 0.09 | 0.37 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | cumprod | 0.12 | 0.04 | 0.14 |
| 1024 | 4096 | torch.bfloat16 | cumprod | 0.12 | 0.04 | 0.14 |
| 4096 | 4096 | torch.float16 | cumprod | 0.36 | 0.05 | 0.19 |

## DeepSeekSparseAttentionDecodeWithKVCacheOp

### tileops

| batch | heads | seq_len_q | seq_len_kv | dim | dim_tail | topk | stride_kv | heads_kv | q_start_index_s | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 128 | 1024 | 2048 | 512 | 64 | 2048 | 1 | 1 | 1024 | torch.float16 | 2.35 | 248.50 | 0.13 |
| 1 | 128 | 512 | 4096 | 512 | 64 | 1024 | 1 | 1 | 512 | torch.float16 | 0.62 | 237.07 | 0.24 |

### torch-ref

| batch | heads | seq_len_q | seq_len_kv | dim | dim_tail | topk | stride_kv | heads_kv | q_start_index_s | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 128 | 1024 | 2048 | 512 | 64 | 2048 | 1 | 1 | 1024 | torch.float16 | 33.00 | 17.70 | 0.01 |
| 1 | 128 | 512 | 4096 | 512 | 64 | 1024 | 1 | 1 | 512 | torch.float16 | 33.44 | 4.37 | 0.00 |

## MultiHeadLatentAttentionDecodeWithKVCacheOp

### tileops

| batch | heads | heads_kv | seq_len_kv | dim | dim_pe | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 128 | 1 | 8192 | 512 | 64 | torch.float16 | 0.23 | 310.72 | 1.31 |
| 16 | 128 | 1 | 4096 | 512 | 64 | torch.float16 | 0.06 | 311.92 | 1.33 |

### torch-ref

| batch | heads | heads_kv | seq_len_kv | dim | dim_pe | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 128 | 1 | 8192 | 512 | 64 | torch.float16 | 0.72 | 100.99 | 0.42 |
| 16 | 128 | 1 | 4096 | 512 | 64 | torch.float16 | 0.16 | 111.61 | 0.48 |

## NSACmpFwdVarlenOp

### tileops

| seq_num | c_seq_len | heads | dim_k | dim_v | group | scale | bc | bs | bk | bv | dtype | accum_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 9 | 8192 | 32 | 128 | 128 | 16 | 0.08838834764831845 | 32 | 32 | 128 | 128 | torch.float16 | torch.float32 | 0.63 | 27.13 | 3.50 |
| 16 | 16384 | 32 | 128 | 128 | 16 | 0.08838834764831845 | 32 | 32 | 128 | 128 | torch.float16 | torch.float32 | 0.69 | 99.20 | 12.59 |

### torch-ref

| seq_num | c_seq_len | heads | dim_k | dim_v | group | scale | bc | bs | bk | bv | dtype | accum_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 9 | 8192 | 32 | 128 | 128 | 16 | 0.08838834764831845 | 32 | 32 | 128 | 128 | torch.float16 | torch.float32 | 306.98 | 0.06 | 0.01 |
| 16 | 16384 | 32 | 128 | 128 | 16 | 0.08838834764831845 | 32 | 32 | 128 | 128 | torch.float16 | torch.float32 | 597.03 | 0.12 | 0.01 |

## NSAFwdVarlenOp

### tileops

| batch | heads | c_seq_len | dim | is_causal | scale | block_size | groups | selected_blocks | dtype | accum_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 16 | 1024 | 64 | True | 0.1 | 32 | 16 | 1 | torch.float16 | torch.float32 | 0.01 | 11.70 | 0.39 |
| 4 | 16 | 8192 | 64 | True | 0.1 | 32 | 16 | 1 | torch.float16 | torch.float32 | 0.05 | 22.06 | 0.73 |
| 2 | 16 | 8192 | 64 | True | 0.1 | 32 | 16 | 4 | torch.float16 | torch.float32 | 0.07 | 58.25 | 0.49 |

### torch-ref

| batch | heads | c_seq_len | dim | is_causal | scale | block_size | groups | selected_blocks | dtype | accum_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 16 | 1024 | 64 | True | 0.1 | 32 | 16 | 1 | torch.float16 | torch.float32 | 65.06 | 0.00 | 0.00 |
| 4 | 16 | 8192 | 64 | True | 0.1 | 32 | 16 | 1 | torch.float16 | torch.float32 | 522.23 | 0.00 | 0.00 |
| 2 | 16 | 8192 | 64 | True | 0.1 | 32 | 16 | 4 | torch.float16 | torch.float32 | 483.89 | 0.01 | 0.00 |

## NSATopkVarlenOp

### tileops

| seq_num | c_seq_len | heads | dim | group | scale | selected_block_num | bc | bs | bk | dtype | accum_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 1024 | 32 | 128 | 16 | 1 | 16 | 32 | 32 | 128 | torch.float16 | torch.float32 | 0.05 | 5.28 | 0.50 |
| 3 | 512 | 32 | 128 | 16 | 1 | 16 | 32 | 32 | 128 | torch.float16 | torch.float32 | 0.03 | 2.12 | 0.27 |
| 9 | 8192 | 32 | 128 | 16 | 1 | 16 | 32 | 32 | 128 | torch.float16 | torch.float32 | 0.75 | 23.05 | 1.53 |

### torch-ref

| seq_num | c_seq_len | heads | dim | group | scale | selected_block_num | bc | bs | bk | dtype | accum_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 1024 | 32 | 128 | 16 | 1 | 16 | 32 | 32 | 128 | torch.float16 | torch.float32 | 243.36 | 0.00 | 0.00 |
| 3 | 512 | 32 | 128 | 16 | 1 | 16 | 32 | 32 | 128 | torch.float16 | torch.float32 | 120.83 | 0.00 | 0.00 |
| 9 | 8192 | 32 | 128 | 16 | 1 | 16 | 32 | 32 | 128 | torch.float16 | torch.float32 | 2523.35 | 0.01 | 0.00 |

## DeltaNetFwdOp

### tileops

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 0.11 | 2.35 | 0.15 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 0.19 | 1.38 | 0.09 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 0.20 | 1.37 | 0.09 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 0.05 | 2.55 | 0.16 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 0.28 | 1.91 | 0.12 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 0.65 | 1.66 | 0.10 |
| 2 | 32768 | 4 | 64 | 64 | 64 | torch.float16 | 1.61 | 1.33 | 0.08 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.05 | 2.53 | 0.16 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.11 | 2.35 | 0.15 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.28 | 1.91 | 0.12 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.65 | 1.66 | 0.10 |
| 2 | 32768 | 4 | 64 | 64 | 64 | torch.bfloat16 | 1.62 | 1.32 | 0.08 |

### fla

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.08 | 3.19 | 0.20 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 0.125 | 0.08 | 3.19 | 0.20 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 0.125 | 0.08 | 3.17 | 0.20 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.05 | 2.45 | 0.15 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.16 | 3.31 | 0.21 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.41 | 2.62 | 0.16 |
| 2 | 32768 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.93 | 2.30 | 0.14 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.05 | 2.45 | 0.15 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.08 | 3.17 | 0.20 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.16 | 3.28 | 0.21 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.42 | 2.58 | 0.16 |
| 2 | 32768 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.95 | 2.27 | 0.14 |

## DeltaNetBwdOp

### tileops

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | H | S | DK | DV | BC | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.32 | 1.66 | 0.09 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.38 | 1.42 | 0.08 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.38 | 1.42 | 0.08 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.15 | 1.83 | 0.10 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.73 | 1.46 | 0.08 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 16384 | 64 | 64 | 64 | 1.67 | 1.29 | 0.07 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.15 | 1.82 | 0.10 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.32 | 1.68 | 0.09 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.74 | 1.45 | 0.08 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 16384 | 64 | 64 | 64 | 1.67 | 1.28 | 0.07 |

### fla

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | H | S | DK | DV | BC | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.125 | 0.18 | 3.02 | 0.17 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.125 | 0.18 | 3.03 | 0.17 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.125 | 0.18 | 3.03 | 0.17 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.125 | 0.09 | 2.86 | 0.16 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.125 | 0.36 | 2.96 | 0.16 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 16384 | 64 | 64 | 64 | 0.125 | 1.33 | 1.61 | 0.09 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.125 | 0.09 | 2.84 | 0.16 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.125 | 0.18 | 3.01 | 0.17 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.125 | 0.36 | 2.96 | 0.16 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 16384 | 64 | 64 | 64 | 0.125 | 1.34 | 1.61 | 0.09 |

## DeltaNetOp

### tileops

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | H | S | DK | DV | BC | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.32 | 2.49 | 0.14 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.54 | 1.49 | 0.09 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.57 | 1.42 | 0.08 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.17 | 2.43 | 0.14 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 8192 | 64 | 64 | 64 | 1.28 | 1.26 | 0.07 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 16384 | 64 | 64 | 64 | 2.45 | 1.31 | 0.08 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.17 | 2.43 | 0.14 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.32 | 2.51 | 0.14 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 8192 | 64 | 64 | 64 | 1.28 | 1.26 | 0.07 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 16384 | 64 | 64 | 64 | 2.46 | 1.31 | 0.08 |

### fla

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | H | S | DK | DV | BC | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.125 | 0.26 | 3.09 | 0.18 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.125 | 0.26 | 3.12 | 0.18 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.125 | 0.26 | 3.11 | 0.18 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.125 | 0.15 | 2.76 | 0.16 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.125 | 0.52 | 3.11 | 0.18 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 16384 | 64 | 64 | 64 | 0.125 | 1.30 | 2.48 | 0.14 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.125 | 0.15 | 2.75 | 0.16 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.125 | 0.26 | 3.09 | 0.18 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.125 | 0.52 | 3.09 | 0.18 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 16384 | 64 | 64 | 64 | 0.125 | 1.39 | 2.31 | 0.13 |

## DeltaNetDecodeOp

### tileops

| batch | heads | dim_k | dim_v | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 128 | 128 | torch.float32 | 0.01 | 0.26 | 0.35 |
| 1 | 32 | 128 | 128 | torch.bfloat16 | 0.01 | 0.28 | 0.19 |
| 8 | 32 | 128 | 128 | torch.float32 | 0.03 | 0.97 | 1.30 |
| 8 | 32 | 128 | 128 | torch.bfloat16 | 0.01 | 1.81 | 1.22 |
| 16 | 32 | 128 | 128 | torch.float32 | 0.05 | 0.98 | 1.33 |
| 16 | 32 | 128 | 128 | torch.bfloat16 | 0.02 | 2.53 | 1.71 |
| 32 | 32 | 128 | 128 | torch.float32 | 0.11 | 0.95 | 1.29 |
| 32 | 32 | 128 | 128 | torch.bfloat16 | 0.04 | 2.65 | 1.79 |
| 1 | 32 | 64 | 64 | torch.float32 | 0.01 | 0.12 | 0.17 |
| 8 | 32 | 64 | 64 | torch.float32 | 0.01 | 0.51 | 0.70 |
| 32 | 32 | 64 | 64 | torch.float32 | 0.05 | 0.55 | 0.76 |
| 64 | 32 | 128 | 128 | torch.float32 | 0.23 | 0.86 | 1.16 |
| 64 | 32 | 128 | 128 | torch.bfloat16 | 0.07 | 2.82 | 1.91 |

### torch

| batch | heads | dim_k | dim_v | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 128 | 128 | torch.float32 | 0.02 | 0.14 | 0.19 |
| 1 | 32 | 128 | 128 | torch.bfloat16 | 0.04 | 0.08 | 0.05 |
| 8 | 32 | 128 | 128 | torch.float32 | 0.06 | 0.44 | 0.59 |
| 8 | 32 | 128 | 128 | torch.bfloat16 | 0.09 | 0.29 | 0.20 |
| 16 | 32 | 128 | 128 | torch.float32 | 0.08 | 0.60 | 0.80 |
| 16 | 32 | 128 | 128 | torch.bfloat16 | 0.14 | 0.37 | 0.25 |
| 32 | 32 | 128 | 128 | torch.float32 | 0.17 | 0.60 | 0.81 |
| 32 | 32 | 128 | 128 | torch.bfloat16 | 0.26 | 0.38 | 0.26 |
| 1 | 32 | 64 | 64 | torch.float32 | 0.02 | 0.04 | 0.06 |
| 8 | 32 | 64 | 64 | torch.float32 | 0.03 | 0.23 | 0.32 |
| 32 | 32 | 64 | 64 | torch.float32 | 0.06 | 0.46 | 0.63 |
| 64 | 32 | 128 | 128 | torch.float32 | 0.33 | 0.61 | 0.82 |
| 64 | 32 | 128 | 128 | torch.bfloat16 | 0.51 | 0.40 | 0.27 |

## DropoutOp

### tileops

| dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| torch.float16 | 4194304 | 0.01 | 0.52 | 2.09 |
| torch.bfloat16 | 4194304 | 0.01 | 0.52 | 2.10 |
| torch.float32 | 4194304 | 0.01 | 0.31 | 2.47 |
| torch.float16 | 10485760 | 0.02 | 0.67 | 2.70 |
| torch.bfloat16 | 10485760 | 0.02 | 0.67 | 2.69 |
| torch.float32 | 10485760 | 0.03 | 0.37 | 2.99 |
| torch.float16 | 20971520 | 0.03 | 0.75 | 3.00 |
| torch.bfloat16 | 20971520 | 0.03 | 0.75 | 3.00 |
| torch.float32 | 20971520 | 0.05 | 0.39 | 3.15 |

### torch

| dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| torch.float16 | 4194304 | 0.01 | 0.32 | 1.27 |
| torch.bfloat16 | 4194304 | 0.01 | 0.32 | 1.27 |
| torch.float32 | 4194304 | 0.02 | 0.22 | 1.76 |
| torch.float16 | 10485760 | 0.03 | 0.41 | 1.63 |
| torch.bfloat16 | 10485760 | 0.03 | 0.40 | 1.60 |
| torch.float32 | 10485760 | 0.04 | 0.26 | 2.05 |
| torch.float16 | 20971520 | 0.05 | 0.43 | 1.71 |
| torch.bfloat16 | 20971520 | 0.05 | 0.42 | 1.68 |
| torch.float32 | 20971520 | 0.08 | 0.26 | 2.05 |

## relu_fp8

### tileops

| op_name | n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| relu_fp8 | 8388608 | torch.float8_e4m3fn | 0.02 | 0.47 | 0.93 |
| relu_fp8 | 8388608 | torch.float8_e5m2 | 0.02 | 0.39 | 0.78 |
| relu_fp8 | 67108864 | torch.float8_e4m3fn | 0.14 | 0.49 | 0.98 |
| relu_fp8 | 67108864 | torch.float8_e5m2 | 0.16 | 0.41 | 0.82 |
| relu_fp8 | 134217728 | torch.float8_e4m3fn | 0.31 | 0.43 | 0.86 |
| relu_fp8 | 134217728 | torch.float8_e5m2 | 0.37 | 0.36 | 0.73 |

### torch

| op_name | n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| relu_fp8 | 8388608 | torch.float8_e4m3fn | 0.05 | 0.16 | 0.32 |
| relu_fp8 | 8388608 | torch.float8_e5m2 | 0.05 | 0.17 | 0.34 |
| relu_fp8 | 67108864 | torch.float8_e4m3fn | 0.52 | 0.13 | 0.26 |
| relu_fp8 | 67108864 | torch.float8_e5m2 | 0.50 | 0.13 | 0.27 |
| relu_fp8 | 134217728 | torch.float8_e4m3fn | 1.32 | 0.10 | 0.20 |
| relu_fp8 | 134217728 | torch.float8_e5m2 | 1.26 | 0.11 | 0.21 |

## exp_fp8

### tileops

| op_name | n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| exp_fp8 | 8388608 | torch.float8_e4m3fn | 0.01 | 0.82 | 1.64 |
| exp_fp8 | 8388608 | torch.float8_e5m2 | 0.02 | 0.39 | 0.78 |
| exp_fp8 | 67108864 | torch.float8_e4m3fn | 0.07 | 1.01 | 2.02 |
| exp_fp8 | 67108864 | torch.float8_e5m2 | 0.17 | 0.40 | 0.80 |
| exp_fp8 | 134217728 | torch.float8_e4m3fn | 0.14 | 0.96 | 1.92 |
| exp_fp8 | 134217728 | torch.float8_e5m2 | 0.37 | 0.36 | 0.73 |

### torch

| op_name | n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| exp_fp8 | 8388608 | torch.float8_e4m3fn | 0.05 | 0.16 | 0.32 |
| exp_fp8 | 8388608 | torch.float8_e5m2 | 0.05 | 0.17 | 0.34 |
| exp_fp8 | 67108864 | torch.float8_e4m3fn | 0.51 | 0.13 | 0.26 |
| exp_fp8 | 67108864 | torch.float8_e5m2 | 0.49 | 0.14 | 0.27 |
| exp_fp8 | 134217728 | torch.float8_e4m3fn | 1.30 | 0.10 | 0.21 |
| exp_fp8 | 134217728 | torch.float8_e5m2 | 1.25 | 0.11 | 0.21 |

## add_fp8

### tileops

| op_name | n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| add_fp8 | 8388608 | torch.float8_e4m3fn | 0.01 | 0.71 | 2.13 |
| add_fp8 | 8388608 | torch.float8_e5m2 | 0.02 | 0.36 | 1.08 |
| add_fp8 | 67108864 | torch.float8_e4m3fn | 0.08 | 0.86 | 2.58 |
| add_fp8 | 67108864 | torch.float8_e5m2 | 0.18 | 0.37 | 1.10 |
| add_fp8 | 134217728 | torch.float8_e4m3fn | 0.17 | 0.81 | 2.44 |
| add_fp8 | 134217728 | torch.float8_e5m2 | 0.41 | 0.33 | 0.98 |

### torch

| op_name | n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| add_fp8 | 8388608 | torch.float8_e4m3fn | 0.09 | 0.09 | 0.28 |
| add_fp8 | 8388608 | torch.float8_e5m2 | 0.09 | 0.10 | 0.29 |
| add_fp8 | 67108864 | torch.float8_e4m3fn | 0.95 | 0.07 | 0.21 |
| add_fp8 | 67108864 | torch.float8_e5m2 | 0.90 | 0.07 | 0.22 |
| add_fp8 | 134217728 | torch.float8_e4m3fn | 2.21 | 0.06 | 0.18 |
| add_fp8 | 134217728 | torch.float8_e5m2 | 2.13 | 0.06 | 0.19 |

## silu_and_mul_fp8

### tileops

| op_name | M | N | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| silu_and_mul_fp8 | 2048 | 11008 | torch.float8_e4m3fn | 0.06 | 2.00 | 1.20 |
| silu_and_mul_fp8 | 2048 | 11008 | torch.float8_e5m2 | 0.08 | 1.43 | 0.86 |
| silu_and_mul_fp8 | 16384 | 11008 | torch.float8_e4m3fn | 0.64 | 1.42 | 0.85 |
| silu_and_mul_fp8 | 16384 | 11008 | torch.float8_e5m2 | 0.89 | 1.02 | 0.61 |
| silu_and_mul_fp8 | 16384 | 28672 | torch.float8_e4m3fn | 1.63 | 1.44 | 0.87 |
| silu_and_mul_fp8 | 16384 | 28672 | torch.float8_e5m2 | 2.31 | 1.02 | 0.61 |

### torch-ref

| op_name | M | N | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| silu_and_mul_fp8 | 2048 | 11008 | torch.float8_e4m3fn | 0.41 | 0.28 | 0.17 |
| silu_and_mul_fp8 | 2048 | 11008 | torch.float8_e5m2 | 0.40 | 0.28 | 0.17 |
| silu_and_mul_fp8 | 16384 | 11008 | torch.float8_e4m3fn | 4.29 | 0.21 | 0.13 |
| silu_and_mul_fp8 | 16384 | 11008 | torch.float8_e5m2 | 4.17 | 0.22 | 0.13 |
| silu_and_mul_fp8 | 16384 | 28672 | torch.float8_e4m3fn | 9.89 | 0.24 | 0.14 |
| silu_and_mul_fp8 | 16384 | 28672 | torch.float8_e5m2 | 9.66 | 0.24 | 0.15 |

## EngramGateConvBwdOp

### tileops

| M | seq_len | d | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 256 | torch.float16 | 0.01 | 0.07 | 0.03 |
| 2 | 64 | 512 | torch.float16 | 0.01 | 0.33 | 0.11 |
| 1 | 128 | 256 | torch.bfloat16 | 0.01 | 0.19 | 0.06 |
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
| 1 | 512 | 256 | 12 | 4 | 3 | torch.float16 | 0.04 | 0.01 | 0.01 |
| 4 | 1024 | 512 | 20 | 4 | 5 | torch.float16 | 0.10 | 0.09 | 0.02 |
| 8 | 512 | 256 | 18 | 4 | 3 | torch.bfloat16 | 0.04 | 0.11 | 0.02 |

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
| 1 | 32 | 256 | torch.float16 | 0.00 | 0.04 | 0.02 |
| 2 | 64 | 512 | torch.float16 | 0.01 | 0.30 | 0.13 |
| 1 | 128 | 256 | torch.bfloat16 | 0.00 | 0.16 | 0.07 |
| 2 | 16 | 256 | torch.bfloat16 | 0.00 | 0.04 | 0.02 |

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
| 64 | torch.complex64 | 0.02 | 0.00 | 0.00 |
| 128 | torch.complex64 | 0.02 | 0.00 | 0.00 |
| 256 | torch.complex64 | 0.03 | 0.00 | 0.00 |
| 512 | torch.complex64 | 0.03 | 0.00 | 0.00 |
| 1024 | torch.complex64 | 0.03 | 0.00 | 0.00 |

### torch-cufft

| n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 64 | torch.complex64 | 0.00 | 0.00 | 0.00 |
| 128 | torch.complex64 | 0.00 | 0.00 | 0.00 |
| 256 | torch.complex64 | 0.00 | 0.00 | 0.00 |
| 512 | torch.complex64 | 0.00 | 0.01 | 0.00 |
| 1024 | torch.complex64 | 0.00 | 0.01 | 0.00 |

### tileops-base

| n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 64 | torch.complex64 | 0.02 | 0.00 | 0.00 |
| 128 | torch.complex64 | 0.02 | 0.00 | 0.00 |
| 256 | torch.complex64 | 0.03 | 0.00 | 0.00 |
| 512 | torch.complex64 | 0.03 | 0.00 | 0.00 |
| 1024 | torch.complex64 | 0.03 | 0.00 | 0.00 |
| 4096 | torch.complex64 | 0.04 | 0.01 | 0.00 |
| 16384 | torch.complex64 | 0.04 | 0.03 | 0.01 |

## FFTC2CLUTOp

### tileops-lut

| n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 64 | torch.complex64 | 0.01 | 0.00 | 0.00 |
| 128 | torch.complex64 | 0.01 | 0.00 | 0.00 |
| 256 | torch.complex64 | 0.01 | 0.00 | 0.00 |
| 512 | torch.complex64 | 0.01 | 0.00 | 0.00 |
| 1024 | torch.complex64 | 0.01 | 0.00 | 0.00 |
| 4096 | torch.complex64 | 0.01 | 0.02 | 0.01 |
| 16384 | torch.complex64 | 0.01 | 0.09 | 0.02 |

### torch-cufft

| n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 64 | torch.complex64 | 0.00 | 0.00 | 0.00 |
| 128 | torch.complex64 | 0.00 | 0.00 | 0.00 |
| 256 | torch.complex64 | 0.00 | 0.00 | 0.00 |
| 512 | torch.complex64 | 0.00 | 0.01 | 0.00 |
| 1024 | torch.complex64 | 0.00 | 0.01 | 0.00 |
| 4096 | torch.complex64 | 0.01 | 0.04 | 0.01 |
| 16384 | torch.complex64 | 0.01 | 0.08 | 0.02 |

## Fp8LightingIndexerOp

### tileops

| batch | seq_len | heads | index_dim | seq_len_kv | kv_group | clean_logits | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 4096 | 32 | 64 | 8192 | 1 | True | 0.21 | N/A | 0.69 |
| 1 | 2048 | 16 | 64 | 4096 | 1 | True | 0.12 | N/A | 0.30 |

### torch-ref

| batch | seq_len | heads | index_dim | seq_len_kv | kv_group | clean_logits | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 4096 | 32 | 64 | 8192 | 1 | True | 21.55 | N/A | 0.01 |
| 1 | 2048 | 16 | 64 | 4096 | 1 | True | 2.39 | N/A | 0.02 |

## Fp8QuantOp

### tileops

| batch | seq_len_kv | kv_group | index_dim | in_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8192 | 1 | 64 | torch.float16 | 0.00 | 0.86 | 0.29 |
| 1 | 8192 | 1 | 64 | torch.bfloat16 | 0.00 | 0.85 | 0.28 |
| 1 | 4096 | 1 | 128 | torch.float32 | 0.00 | 0.64 | 0.43 |
| 1 | 16384 | 1 | 32 | torch.float32 | 0.00 | 0.81 | 0.54 |

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
| 1024 | 4096 | torch.float16 | 0.02 | 1.12 | 1.49 |
| 4096 | 4096 | torch.bfloat16 | 0.09 | 1.09 | 1.45 |
| 1024 | 3000 | torch.float16 | 0.04 | 0.52 | 0.69 |
| 1025 | 4096 | torch.float16 | 0.02 | 1.14 | 1.53 |

### torch-ref

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.06 | 0.40 | 0.54 |
| 4096 | 4096 | torch.bfloat16 | 0.26 | 0.38 | 0.51 |
| 1024 | 3000 | torch.float16 | 0.05 | 0.38 | 0.51 |
| 1025 | 4096 | torch.float16 | 0.06 | 0.40 | 0.54 |

## FusedAddRmsNormOp

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.02 | 1.01 | 1.62 |
| 4096 | 4096 | torch.bfloat16 | 0.08 | 1.04 | 1.66 |
| 2048 | 5120 | torch.float16 | 0.05 | 1.04 | 1.66 |
| 1025 | 4096 | torch.float16 | 0.02 | 1.01 | 1.62 |

### torch-ref

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.13 | 0.16 | 0.25 |
| 4096 | 4096 | torch.bfloat16 | 0.58 | 0.15 | 0.23 |
| 2048 | 5120 | torch.float16 | 0.36 | 0.14 | 0.23 |
| 1025 | 4096 | torch.float16 | 0.13 | 0.16 | 0.25 |

## GatedDeltaNetFwdOp

### tileops

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 0.23 | 1.18 | 0.07 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 0.23 | 1.16 | 0.07 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 0.08 | 1.65 | 0.10 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 0.16 | 1.68 | 0.11 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 0.36 | 1.50 | 0.09 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 0.78 | 1.38 | 0.09 |
| 2 | 32768 | 4 | 64 | 64 | 64 | torch.float16 | 1.87 | 1.15 | 0.07 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.08 | 1.63 | 0.10 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.16 | 1.68 | 0.11 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.35 | 1.51 | 0.10 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.77 | 1.39 | 0.09 |
| 2 | 32768 | 4 | 64 | 64 | 64 | torch.bfloat16 | 1.85 | 1.16 | 0.07 |

### fla

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 0.125 | 0.10 | 2.73 | 0.17 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 0.125 | 0.10 | 2.71 | 0.17 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.06 | 2.15 | 0.14 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.10 | 2.73 | 0.17 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.19 | 2.85 | 0.18 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.46 | 2.31 | 0.15 |
| 2 | 32768 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 1.07 | 2.01 | 0.13 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.06 | 2.13 | 0.13 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.10 | 2.71 | 0.17 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.19 | 2.83 | 0.18 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.46 | 2.34 | 0.15 |
| 2 | 32768 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 1.07 | 2.01 | 0.13 |

## GatedDeltaNetBwdOp

### tileops

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | H | S | DK | DV | BC | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.51 | 1.05 | 0.06 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.51 | 1.06 | 0.06 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.19 | 1.42 | 0.08 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.40 | 1.34 | 0.07 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.80 | 1.33 | 0.07 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 16384 | 64 | 64 | 64 | 1.77 | 1.21 | 0.07 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.19 | 1.41 | 0.08 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.40 | 1.33 | 0.07 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.81 | 1.33 | 0.07 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 16384 | 64 | 64 | 64 | 1.79 | 1.20 | 0.07 |

### fla

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | H | S | DK | DV | BC | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.125 | 0.25 | 2.16 | 0.12 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.125 | 0.25 | 2.17 | 0.12 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.125 | 0.13 | 2.01 | 0.11 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.125 | 0.25 | 2.17 | 0.12 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.125 | 0.47 | 2.27 | 0.13 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 16384 | 64 | 64 | 64 | 0.125 | 1.68 | 1.28 | 0.07 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.125 | 0.14 | 1.99 | 0.11 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.125 | 0.25 | 2.18 | 0.12 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.125 | 0.48 | 2.26 | 0.12 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 16384 | 64 | 64 | 64 | 0.125 | 1.69 | 1.27 | 0.07 |

## GatedDeltaNetOp

### tileops

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | H | S | DK | DV | BC | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.70 | 1.16 | 0.07 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.68 | 1.18 | 0.07 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.27 | 1.51 | 0.09 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.49 | 1.64 | 0.09 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 8192 | 64 | 64 | 64 | 1.36 | 1.18 | 0.07 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 16384 | 64 | 64 | 64 | 3.29 | 0.98 | 0.06 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.27 | 1.50 | 0.09 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.49 | 1.63 | 0.09 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 8192 | 64 | 64 | 64 | 1.37 | 1.18 | 0.07 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 16384 | 64 | 64 | 64 | 3.33 | 0.97 | 0.06 |

### fla

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | H | S | DK | DV | BC | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.125 | 0.34 | 2.35 | 0.14 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.125 | 0.34 | 2.35 | 0.14 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.125 | 0.19 | 2.08 | 0.12 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.125 | 0.34 | 2.35 | 0.14 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.125 | 0.66 | 2.44 | 0.14 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 16384 | 64 | 64 | 64 | 0.125 | 1.69 | 1.91 | 0.11 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.125 | 0.19 | 2.07 | 0.12 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.125 | 0.34 | 2.34 | 0.14 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.125 | 0.66 | 2.43 | 0.14 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 16384 | 64 | 64 | 64 | 0.125 | 1.72 | 1.87 | 0.11 |

## GatedDeltaNetDecodeOp

### tileops

| batch | heads | dim_k | dim_v | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 128 | 128 | torch.float32 | 0.01 | 0.25 | 0.34 |
| 1 | 32 | 128 | 128 | torch.bfloat16 | 0.01 | 0.26 | 0.18 |
| 8 | 32 | 128 | 128 | torch.float32 | 0.03 | 0.95 | 1.29 |
| 8 | 32 | 128 | 128 | torch.bfloat16 | 0.01 | 1.75 | 1.18 |
| 16 | 32 | 128 | 128 | torch.float32 | 0.05 | 0.98 | 1.32 |
| 16 | 32 | 128 | 128 | torch.bfloat16 | 0.02 | 2.52 | 1.70 |
| 32 | 32 | 128 | 128 | torch.float32 | 0.11 | 0.95 | 1.29 |
| 32 | 32 | 128 | 128 | torch.bfloat16 | 0.04 | 2.61 | 1.76 |
| 1 | 32 | 64 | 64 | torch.float32 | 0.01 | 0.11 | 0.16 |
| 8 | 32 | 64 | 64 | torch.float32 | 0.01 | 0.49 | 0.67 |
| 32 | 32 | 64 | 64 | torch.float32 | 0.05 | 0.55 | 0.76 |
| 64 | 32 | 128 | 128 | torch.float32 | 0.23 | 0.86 | 1.16 |
| 64 | 32 | 128 | 128 | torch.bfloat16 | 0.07 | 2.74 | 1.85 |

### torch-ref

| batch | heads | dim_k | dim_v | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 128 | 128 | torch.float32 | 0.03 | 0.09 | 0.13 |
| 1 | 32 | 128 | 128 | torch.bfloat16 | 0.05 | 0.06 | 0.04 |
| 8 | 32 | 128 | 128 | torch.float32 | 0.08 | 0.31 | 0.42 |
| 8 | 32 | 128 | 128 | torch.bfloat16 | 0.11 | 0.23 | 0.15 |
| 16 | 32 | 128 | 128 | torch.float32 | 0.12 | 0.41 | 0.56 |
| 16 | 32 | 128 | 128 | torch.bfloat16 | 0.18 | 0.28 | 0.19 |
| 32 | 32 | 128 | 128 | torch.float32 | 0.24 | 0.43 | 0.58 |
| 32 | 32 | 128 | 128 | torch.bfloat16 | 0.34 | 0.30 | 0.20 |
| 1 | 32 | 64 | 64 | torch.float32 | 0.03 | 0.03 | 0.04 |
| 8 | 32 | 64 | 64 | torch.float32 | 0.04 | 0.16 | 0.21 |
| 32 | 32 | 64 | 64 | torch.float32 | 0.08 | 0.32 | 0.44 |
| 64 | 32 | 128 | 128 | torch.float32 | 0.46 | 0.44 | 0.60 |
| 64 | 32 | 128 | 128 | torch.bfloat16 | 0.64 | 0.32 | 0.21 |

## GemmOp

### tileops

| m | n | k | dtype | trans_a | trans_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1024 | 1024 | 1024 | torch.float16 | False | False | 0.01 | 195.89 | 0.57 |
| 1 | 7168 | 16384 | torch.float16 | False | True | 0.09 | 2.66 | 2.66 |
| 1 | 18432 | 7168 | torch.bfloat16 | False | True | 0.09 | 2.79 | 2.79 |
| 7168 | 1 | 16384 | torch.float16 | False | False | 0.09 | 2.67 | 2.67 |
| 18432 | 1 | 7168 | torch.bfloat16 | False | False | 0.09 | 2.78 | 2.79 |

### torch-cublas

| m | n | k | dtype | trans_a | trans_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1024 | 1024 | 1024 | torch.float16 | False | False | 0.01 | 268.38 | 0.79 |
| 1 | 7168 | 16384 | torch.float16 | False | True | 0.09 | 2.71 | 2.71 |
| 1 | 18432 | 7168 | torch.bfloat16 | False | True | 0.11 | 2.44 | 2.44 |
| 7168 | 1 | 16384 | torch.float16 | False | False | 0.09 | 2.73 | 2.73 |
| 18432 | 1 | 7168 | torch.bfloat16 | False | False | 0.11 | 2.47 | 2.47 |

## GLAFwdOp

### tileops

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.09 | 1.49 | 0.09 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.17 | 1.62 | 0.10 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.36 | 1.48 | 0.09 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.84 | 1.28 | 0.08 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.09 | 1.43 | 0.09 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.17 | 1.60 | 0.10 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.36 | 1.48 | 0.09 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.85 | 1.26 | 0.08 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.31 | 1.30 | 0.07 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.71 | 1.13 | 0.06 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 1.69 | 0.95 | 0.05 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 3.22 | 1.00 | 0.06 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.31 | 1.29 | 0.07 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.71 | 1.13 | 0.06 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 1.72 | 0.94 | 0.05 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 3.34 | 0.96 | 0.06 |

### fla

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.06 | 2.09 | 0.13 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.12 | 2.28 | 0.14 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.23 | 2.36 | 0.15 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.64 | 1.69 | 0.11 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.06 | 2.07 | 0.13 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.12 | 2.23 | 0.14 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.23 | 2.33 | 0.15 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.65 | 1.65 | 0.10 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.22 | 1.81 | 0.10 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.42 | 1.94 | 0.11 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.93 | 1.73 | 0.10 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 3.00 | 1.07 | 0.06 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.22 | 1.82 | 0.10 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.42 | 1.94 | 0.11 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.93 | 1.72 | 0.10 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 3.00 | 1.07 | 0.06 |

## GLABwdOp

### tileops

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | T | H | K | V | BC | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 2048 | 4 | 64 | 64 | 64 | 0.125 | 0.20 | 1.33 | 0.07 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4096 | 4 | 64 | 64 | 64 | 0.125 | 0.47 | 1.15 | 0.06 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 8192 | 4 | 64 | 64 | 64 | 0.125 | 1.11 | 0.97 | 0.05 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 16384 | 4 | 64 | 64 | 64 | 0.125 | 2.26 | 0.95 | 0.05 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 2048 | 4 | 64 | 64 | 64 | 0.125 | 0.20 | 1.34 | 0.07 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4096 | 4 | 64 | 64 | 64 | 0.125 | 0.46 | 1.17 | 0.06 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 8192 | 4 | 64 | 64 | 64 | 0.125 | 1.15 | 0.94 | 0.05 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 16384 | 4 | 64 | 64 | 64 | 0.125 | 2.36 | 0.91 | 0.05 |

### fla

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | T | H | K | V | BC | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 2048 | 4 | 64 | 64 | 64 | 0.125 | 0.15 | 1.74 | 0.09 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4096 | 4 | 64 | 64 | 64 | 0.125 | 0.28 | 1.90 | 0.10 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 8192 | 4 | 64 | 64 | 64 | 0.125 | 0.93 | 1.15 | 0.06 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 16384 | 4 | 64 | 64 | 64 | 0.125 | 1.99 | 1.08 | 0.06 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 2048 | 4 | 64 | 64 | 64 | 0.125 | 0.15 | 1.74 | 0.09 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4096 | 4 | 64 | 64 | 64 | 0.125 | 0.28 | 1.90 | 0.10 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 8192 | 4 | 64 | 64 | 64 | 0.125 | 0.93 | 1.15 | 0.06 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 16384 | 4 | 64 | 64 | 64 | 0.125 | 1.99 | 1.08 | 0.06 |

## GLADecodeOp

### tileops

| batch | heads | dim_k | dim_v | dtype | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 64 | 64 | torch.float32 | 0.125 | 0.01 | 0.06 | 0.12 |
| 1 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.02 | 0.10 | 0.21 |
| 1 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.01 | 0.14 | 0.14 |
| 1 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.01 | 0.14 | 0.14 |
| 8 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.04 | 0.39 | 0.79 |
| 8 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.02 | 0.99 | 1.01 |
| 8 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.02 | 0.98 | 0.99 |
| 16 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.09 | 0.38 | 0.77 |
| 16 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.02 | 1.53 | 1.56 |
| 16 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.02 | 1.52 | 1.54 |
| 32 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.19 | 0.35 | 0.71 |
| 32 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.04 | 1.51 | 1.54 |
| 32 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.04 | 1.51 | 1.53 |
| 64 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.46 | 0.29 | 0.59 |
| 64 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.08 | 1.61 | 1.63 |
| 64 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.08 | 1.59 | 1.61 |

### fla

| batch | heads | dim_k | dim_v | dtype | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 64 | 64 | torch.float32 | 0.125 | 0.01 | 0.08 | 0.17 |
| 1 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.01 | 0.31 | 0.62 |
| 1 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.01 | 0.27 | 0.27 |
| 1 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.01 | 0.27 | 0.27 |
| 8 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.02 | 1.04 | 2.12 |
| 8 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.02 | 0.99 | 1.01 |
| 8 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.02 | 0.99 | 1.00 |
| 16 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.03 | 1.23 | 2.51 |
| 16 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.03 | 1.29 | 1.31 |
| 16 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.03 | 1.29 | 1.31 |
| 32 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.05 | 1.44 | 2.92 |
| 32 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.04 | 1.54 | 1.57 |
| 32 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.04 | 1.54 | 1.56 |
| 64 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.11 | 1.26 | 2.57 |
| 64 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.08 | 1.65 | 1.67 |
| 64 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.08 | 1.71 | 1.74 |

## GroupQueryAttentionFwdOp

### tileops

| batch | seq_len | heads | heads_kv | dim | causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 8 | 4 | 64 | False | torch.float16 | 0.01 | 168.93 | 0.25 |
| 4 | 2048 | 64 | 4 | 128 | False | torch.float16 | 1.86 | 295.89 | 0.15 |
| 4 | 2048 | 64 | 4 | 128 | False | torch.bfloat16 | 1.80 | 306.00 | 0.16 |

### torch-sdpa

| batch | seq_len | heads | heads_kv | dim | causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 8 | 4 | 64 | False | torch.float16 | 0.02 | 101.69 | 0.15 |
| 4 | 2048 | 64 | 4 | 128 | False | torch.float16 | 2.86 | 192.46 | 0.10 |
| 4 | 2048 | 64 | 4 | 128 | False | torch.bfloat16 | 2.84 | 193.74 | 0.10 |

## GroupQueryAttentionBwdOp

### tileops

| batch | seq_len | heads | heads_kv | dim | causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 8 | 4 | 64 | False | torch.float16 | 0.06 | 92.81 | 0.09 |
| 4 | 2048 | 64 | 4 | 128 | False | torch.float16 | 7.94 | 173.12 | 0.05 |
| 4 | 2048 | 64 | 4 | 128 | False | torch.bfloat16 | 7.66 | 179.33 | 0.06 |

### torch-sdpa

| batch | seq_len | heads | heads_kv | dim | causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 8 | 4 | 64 | False | torch.float16 | 0.08 | 69.18 | 0.07 |
| 4 | 2048 | 64 | 4 | 128 | False | torch.float16 | 11.12 | 123.57 | 0.04 |
| 4 | 2048 | 64 | 4 | 128 | False | torch.bfloat16 | 11.08 | 124.06 | 0.04 |

## GroupQueryAttentionDecodeWithKVCacheOp

### tileops

| batch | heads | heads_kv | seq_len_kv | dim | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 8 | 8192 | 128 | torch.float16 | 0.02 | 7.45 | 1.86 |
| 4 | 32 | 4 | 4096 | 128 | torch.bfloat16 | 0.02 | 11.53 | 1.44 |
| 8 | 64 | 16 | 8192 | 128 | torch.float16 | 0.15 | 14.25 | 3.56 |

### torch-sdpa

| batch | heads | heads_kv | seq_len_kv | dim | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 8 | 8192 | 128 | torch.float16 | 0.50 | 0.27 | 0.07 |
| 4 | 32 | 4 | 4096 | 128 | torch.bfloat16 | 1.05 | 0.26 | 0.03 |
| 8 | 64 | 16 | 8192 | 128 | torch.float16 | 13.47 | 0.16 | 0.04 |

## GroupQueryAttentionDecodePagedWithKVCacheOp

### tileops

| batch | heads | heads_kv | seqlen_kv | dim | page_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 16 | 8 | 512 | 128 | 128 | torch.float16 | 0.02 | 0.21 | 0.11 |
| 2 | 8 | 4 | 1024 | 64 | 256 | torch.float16 | 0.02 | 0.21 | 0.05 |
| 1 | 16 | 4 | 2048 | 128 | 512 | torch.float16 | 0.02 | 0.77 | 0.19 |
| 1 | 32 | 16 | 512 | 64 | 128 | torch.float16 | 0.01 | 0.37 | 0.19 |

### torch-ref

| batch | heads | heads_kv | seqlen_kv | dim | page_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 16 | 8 | 512 | 128 | 128 | torch.float16 | 0.07 | 0.06 | 0.03 |
| 2 | 8 | 4 | 1024 | 64 | 256 | torch.float16 | 0.13 | 0.03 | 0.01 |
| 1 | 16 | 4 | 2048 | 128 | 512 | torch.float16 | 0.07 | 0.25 | 0.06 |
| 1 | 32 | 16 | 512 | 64 | 128 | torch.float16 | 0.07 | 0.06 | 0.03 |

## GqaSlidingWindowFwdOp

### tileops

| batch | seq | heads | heads_kv | dim | is_causal | wl | wr | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 512 | 8 | 2 | 64 | True | -1 | -1 | torch.float16 | 0.01 | 62.99 | 0.31 |
| 2 | 512 | 8 | 2 | 64 | True | 128 | -1 | torch.float16 | 0.01 | 38.71 | 0.43 |
| 2 | 768 | 8 | 2 | 64 | False | 256 | -1 | torch.float16 | 0.02 | 121.99 | 0.26 |
| 2 | 512 | 8 | 2 | 64 | False | 64 | 64 | torch.bfloat16 | 0.01 | 40.98 | 0.42 |
| 1 | 2048 | 8 | 2 | 64 | True | 512 | -1 | torch.float16 | 0.02 | 117.60 | 0.33 |

### fa3

| batch | seq | heads | heads_kv | dim | is_causal | wl | wr | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 512 | 8 | 2 | 64 | True | -1 | -1 | torch.float16 | 0.01 | 38.24 | 0.19 |
| 2 | 512 | 8 | 2 | 64 | True | 128 | -1 | torch.float16 | 0.01 | 16.06 | 0.18 |
| 2 | 768 | 8 | 2 | 64 | False | 256 | -1 | torch.float16 | 0.04 | 49.64 | 0.10 |
| 2 | 512 | 8 | 2 | 64 | False | 64 | 64 | torch.bfloat16 | 0.01 | 17.20 | 0.18 |
| 1 | 2048 | 8 | 2 | 64 | True | 512 | -1 | torch.float16 | 0.03 | 57.85 | 0.16 |

## GqaSlidingWindowVarlenFwdOp

### tileops

| batch | heads | heads_kv | dim | is_causal | wl | wr | dtype | max_seqlen_q | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 8 | 128 | True | -1 | -1 | torch.float16 | 3000 | 0.22 | 338.89 | 0.28 |
| 2 | 32 | 8 | 128 | True | 2000 | -1 | torch.float16 | 3073 | 0.35 | 251.65 | 0.27 |
| 4 | 32 | 8 | 128 | False | 1500 | 1500 | torch.float16 | 3500 | 0.94 | 365.83 | 0.22 |
| 2 | 32 | 8 | 128 | True | -1 | -1 | torch.float16 | 1024 | 0.40 | 401.38 | 0.19 |
| 2 | 32 | 8 | 128 | True | -1 | -1 | torch.float16 | 8193 | 2.03 | 372.48 | 0.13 |

### fa3

| batch | heads | heads_kv | dim | is_causal | wl | wr | dtype | max_seqlen_q | max_seqlen_k | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 8 | 128 | True | -1 | -1 | torch.float16 | 3000 | 3000 | 0.53 | 139.60 | 0.12 |
| 2 | 32 | 8 | 128 | True | 2000 | -1 | torch.float16 | 3073 | 3073 | 0.78 | 112.28 | 0.12 |
| 4 | 32 | 8 | 128 | False | 1500 | 1500 | torch.float16 | 3500 | 3500 | 2.27 | 151.68 | 0.09 |
| 2 | 32 | 8 | 128 | True | -1 | -1 | torch.float16 | 1024 | 8192 | 1.13 | 142.14 | 0.07 |
| 2 | 32 | 8 | 128 | True | -1 | -1 | torch.float16 | 8193 | 8193 | 3.81 | 198.32 | 0.07 |

## GroupNormOp

### tileops

| n | c | g | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 128 | 32 | torch.float16 | 0.01 | 0.37 | 0.29 |
| 8 | 128 | 32 | torch.bfloat16 | 0.01 | 0.36 | 0.29 |
| 4 | 256 | 32 | torch.float16 | 0.02 | 0.19 | 0.15 |
| 4 | 128 | 16 | torch.float16 | 0.02 | 0.12 | 0.10 |

### torch

| n | c | g | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 128 | 32 | torch.float16 | 0.02 | 0.32 | 0.26 |
| 8 | 128 | 32 | torch.bfloat16 | 0.02 | 0.32 | 0.26 |
| 4 | 256 | 32 | torch.float16 | 0.02 | 0.23 | 0.18 |
| 4 | 128 | 16 | torch.float16 | 0.02 | 0.13 | 0.10 |

## grouped_gemm_nt

### tileops

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_nt | 16384 | 4 | 4864 | 4096 | torch.float16 | False | True | 28.13 | 23.20 | 0.02 |

### torch-ref

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_nt | 16384 | 4 | 4864 | 4096 | torch.float16 | False | True | 1.58 | 412.53 | 0.29 |

## grouped_gemm_nn

### tileops

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_nn | 16384 | 4 | 4864 | 4096 | torch.float16 | False | False | 1.75 | 372.66 | 0.26 |

### torch-ref

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_nn | 16384 | 4 | 4864 | 4096 | torch.float16 | False | False | 1.09 | 599.16 | 0.42 |

## grouped_gemm_tn

### tileops

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_tn | 16384 | 4 | 4864 | 4096 | torch.float16 | True | False | 6.24 | 104.69 | 0.07 |

### torch-ref

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_tn | 16384 | 4 | 4864 | 4096 | torch.float16 | True | False | 1.05 | 619.26 | 0.43 |

## grouped_gemm_tt

### tileops

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_tt | 16384 | 4 | 4864 | 4096 | torch.float16 | True | True | 5.39 | 121.03 | 0.08 |

### torch-ref

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_tt | 16384 | 4 | 4864 | 4096 | torch.float16 | True | True | 1.05 | 619.42 | 0.43 |

## GroupedGemmOp

### tileops

| batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16384 | 4 | 4864 | 4096 | torch.float16 | True | False | 41.46 | 47.24 | N/A |

### torch-ref

| batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16384 | 4 | 4864 | 4096 | torch.float16 | True | False | 4.80 | 407.98 | N/A |

## leaky_relu

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| leaky_relu | torch.float16 | 4194304 | 0.01 | 0.54 | 2.17 |
| leaky_relu | torch.bfloat16 | 4194304 | 0.01 | 0.54 | 2.17 |
| leaky_relu | torch.float32 | 4194304 | 0.01 | 0.31 | 2.52 |
| leaky_relu | torch.float16 | 10485760 | 0.02 | 0.68 | 2.73 |
| leaky_relu | torch.bfloat16 | 10485760 | 0.02 | 0.68 | 2.72 |
| leaky_relu | torch.float32 | 10485760 | 0.03 | 0.38 | 3.04 |
| leaky_relu | torch.float16 | 20971520 | 0.03 | 0.76 | 3.05 |
| leaky_relu | torch.bfloat16 | 20971520 | 0.03 | 0.74 | 2.97 |
| leaky_relu | torch.float32 | 20971520 | 0.05 | 0.40 | 3.17 |

### torch

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| leaky_relu | torch.float16 | 4194304 | 0.01 | 0.54 | 2.18 |
| leaky_relu | torch.bfloat16 | 4194304 | 0.01 | 0.54 | 2.18 |
| leaky_relu | torch.float32 | 4194304 | 0.01 | 0.33 | 2.64 |
| leaky_relu | torch.float16 | 10485760 | 0.02 | 0.68 | 2.72 |
| leaky_relu | torch.bfloat16 | 10485760 | 0.02 | 0.68 | 2.72 |
| leaky_relu | torch.float32 | 10485760 | 0.03 | 0.38 | 3.02 |
| leaky_relu | torch.float16 | 20971520 | 0.03 | 0.76 | 3.06 |
| leaky_relu | torch.bfloat16 | 20971520 | 0.03 | 0.77 | 3.06 |
| leaky_relu | torch.float32 | 20971520 | 0.05 | 0.39 | 3.12 |

## elu

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| elu | torch.float16 | 4194304 | 0.01 | 0.50 | 2.01 |
| elu | torch.bfloat16 | 4194304 | 0.01 | 0.50 | 2.02 |
| elu | torch.float32 | 4194304 | 0.01 | 0.30 | 2.41 |
| elu | torch.float16 | 10485760 | 0.02 | 0.63 | 2.52 |
| elu | torch.bfloat16 | 10485760 | 0.02 | 0.63 | 2.51 |
| elu | torch.float32 | 10485760 | 0.03 | 0.37 | 2.99 |
| elu | torch.float16 | 20971520 | 0.03 | 0.69 | 2.76 |
| elu | torch.bfloat16 | 20971520 | 0.03 | 0.70 | 2.79 |
| elu | torch.float32 | 20971520 | 0.05 | 0.39 | 3.09 |

### torch

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| elu | torch.float16 | 4194304 | 0.01 | 0.35 | 1.40 |
| elu | torch.bfloat16 | 4194304 | 0.01 | 0.35 | 1.39 |
| elu | torch.float32 | 4194304 | 0.01 | 0.31 | 2.46 |
| elu | torch.float16 | 10485760 | 0.02 | 0.42 | 1.68 |
| elu | torch.bfloat16 | 10485760 | 0.03 | 0.42 | 1.66 |
| elu | torch.float32 | 10485760 | 0.03 | 0.37 | 2.95 |
| elu | torch.float16 | 20971520 | 0.05 | 0.45 | 1.78 |
| elu | torch.bfloat16 | 20971520 | 0.05 | 0.43 | 1.73 |
| elu | torch.float32 | 20971520 | 0.05 | 0.39 | 3.08 |

## hardtanh

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| hardtanh | torch.float16 | 4194304 | 0.01 | 0.54 | 2.17 |
| hardtanh | torch.bfloat16 | 4194304 | 0.01 | 0.54 | 2.16 |
| hardtanh | torch.float32 | 4194304 | 0.01 | 0.32 | 2.52 |
| hardtanh | torch.float16 | 10485760 | 0.02 | 0.68 | 2.73 |
| hardtanh | torch.bfloat16 | 10485760 | 0.02 | 0.65 | 2.59 |
| hardtanh | torch.float32 | 10485760 | 0.03 | 0.38 | 3.04 |
| hardtanh | torch.float16 | 20971520 | 0.03 | 0.75 | 3.02 |
| hardtanh | torch.bfloat16 | 20971520 | 0.03 | 0.76 | 3.04 |
| hardtanh | torch.float32 | 20971520 | 0.05 | 0.39 | 3.15 |

### torch

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| hardtanh | torch.float16 | 4194304 | 0.01 | 0.52 | 2.09 |
| hardtanh | torch.bfloat16 | 4194304 | 0.01 | 0.54 | 2.15 |
| hardtanh | torch.float32 | 4194304 | 0.01 | 0.33 | 2.62 |
| hardtanh | torch.float16 | 10485760 | 0.02 | 0.65 | 2.59 |
| hardtanh | torch.bfloat16 | 10485760 | 0.02 | 0.68 | 2.72 |
| hardtanh | torch.float32 | 10485760 | 0.03 | 0.38 | 3.03 |
| hardtanh | torch.float16 | 20971520 | 0.03 | 0.71 | 2.82 |
| hardtanh | torch.bfloat16 | 20971520 | 0.03 | 0.76 | 3.04 |
| hardtanh | torch.float32 | 20971520 | 0.05 | 0.39 | 3.13 |

## softplus

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| softplus | torch.float16 | 4194304 | 0.01 | 0.35 | 1.39 |
| softplus | torch.bfloat16 | 4194304 | 0.01 | 0.34 | 1.37 |
| softplus | torch.float32 | 4194304 | 0.01 | 0.31 | 2.45 |
| softplus | torch.float16 | 10485760 | 0.03 | 0.41 | 1.66 |
| softplus | torch.bfloat16 | 10485760 | 0.03 | 0.41 | 1.64 |
| softplus | torch.float32 | 10485760 | 0.03 | 0.35 | 2.79 |
| softplus | torch.float16 | 20971520 | 0.05 | 0.43 | 1.74 |
| softplus | torch.bfloat16 | 20971520 | 0.05 | 0.43 | 1.73 |
| softplus | torch.float32 | 20971520 | 0.06 | 0.35 | 2.82 |

### torch

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| softplus | torch.float16 | 4194304 | 0.01 | 0.29 | 1.16 |
| softplus | torch.bfloat16 | 4194304 | 0.01 | 0.29 | 1.18 |
| softplus | torch.float32 | 4194304 | 0.02 | 0.28 | 2.21 |
| softplus | torch.float16 | 10485760 | 0.03 | 0.35 | 1.38 |
| softplus | torch.bfloat16 | 10485760 | 0.03 | 0.34 | 1.36 |
| softplus | torch.float32 | 10485760 | 0.03 | 0.33 | 2.65 |
| softplus | torch.float16 | 20971520 | 0.06 | 0.36 | 1.42 |
| softplus | torch.bfloat16 | 20971520 | 0.06 | 0.35 | 1.40 |
| softplus | torch.float32 | 20971520 | 0.06 | 0.35 | 2.77 |

## clamp

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| clamp | torch.float16 | 4194304 | 0.01 | 0.54 | 2.17 |
| clamp | torch.bfloat16 | 4194304 | 0.01 | 0.54 | 2.16 |
| clamp | torch.float32 | 4194304 | 0.01 | 0.31 | 2.51 |
| clamp | torch.float16 | 10485760 | 0.02 | 0.68 | 2.72 |
| clamp | torch.bfloat16 | 10485760 | 0.02 | 0.68 | 2.72 |
| clamp | torch.float32 | 10485760 | 0.03 | 0.38 | 3.04 |
| clamp | torch.float16 | 20971520 | 0.03 | 0.76 | 3.02 |
| clamp | torch.bfloat16 | 20971520 | 0.03 | 0.76 | 3.03 |
| clamp | torch.float32 | 20971520 | 0.05 | 0.40 | 3.17 |

### torch

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| clamp | torch.float16 | 4194304 | 0.01 | 0.52 | 2.09 |
| clamp | torch.bfloat16 | 4194304 | 0.01 | 0.53 | 2.13 |
| clamp | torch.float32 | 4194304 | 0.01 | 0.33 | 2.63 |
| clamp | torch.float16 | 10485760 | 0.02 | 0.65 | 2.59 |
| clamp | torch.bfloat16 | 10485760 | 0.02 | 0.68 | 2.71 |
| clamp | torch.float32 | 10485760 | 0.03 | 0.38 | 3.04 |
| clamp | torch.float16 | 20971520 | 0.03 | 0.70 | 2.82 |
| clamp | torch.bfloat16 | 20971520 | 0.03 | 0.74 | 2.96 |
| clamp | torch.float32 | 20971520 | 0.05 | 0.39 | 3.12 |

## nan_to_num

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| nan_to_num | torch.float16 | 4194304 | 0.01 | 0.53 | 2.13 |
| nan_to_num | torch.bfloat16 | 4194304 | 0.01 | 0.53 | 2.12 |
| nan_to_num | torch.float32 | 4194304 | 0.01 | 0.31 | 2.52 |
| nan_to_num | torch.float16 | 10485760 | 0.02 | 0.67 | 2.69 |
| nan_to_num | torch.bfloat16 | 10485760 | 0.02 | 0.67 | 2.69 |
| nan_to_num | torch.float32 | 10485760 | 0.03 | 0.38 | 3.04 |
| nan_to_num | torch.float16 | 20971520 | 0.03 | 0.75 | 2.98 |
| nan_to_num | torch.bfloat16 | 20971520 | 0.03 | 0.75 | 2.99 |
| nan_to_num | torch.float32 | 20971520 | 0.05 | 0.39 | 3.15 |

### torch

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| nan_to_num | torch.float16 | 4194304 | 0.01 | 0.53 | 2.11 |
| nan_to_num | torch.bfloat16 | 4194304 | 0.01 | 0.53 | 2.12 |
| nan_to_num | torch.float32 | 4194304 | 0.01 | 0.33 | 2.63 |
| nan_to_num | torch.float16 | 10485760 | 0.02 | 0.67 | 2.69 |
| nan_to_num | torch.bfloat16 | 10485760 | 0.02 | 0.67 | 2.69 |
| nan_to_num | torch.float32 | 10485760 | 0.03 | 0.38 | 3.05 |
| nan_to_num | torch.float16 | 20971520 | 0.03 | 0.74 | 2.97 |
| nan_to_num | torch.bfloat16 | 20971520 | 0.03 | 0.75 | 2.99 |
| nan_to_num | torch.float32 | 20971520 | 0.05 | 0.39 | 3.10 |

## PreluOp

### tileops

| num_channels | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 128 | torch.float16 | 131072 | 0.00 | 0.05 | 0.21 |
| 128 | torch.bfloat16 | 131072 | 0.00 | 0.05 | 0.22 |
| 128 | torch.float32 | 131072 | 0.00 | 0.05 | 0.38 |
| 4096 | torch.float16 | 4194304 | 0.01 | 0.54 | 2.17 |
| 4096 | torch.bfloat16 | 4194304 | 0.01 | 0.55 | 2.19 |
| 4096 | torch.float32 | 4194304 | 0.01 | 0.32 | 2.53 |
| 10240 | torch.float16 | 10485760 | 0.02 | 0.68 | 2.74 |
| 10240 | torch.bfloat16 | 10485760 | 0.02 | 0.68 | 2.71 |
| 10240 | torch.float32 | 10485760 | 0.03 | 0.38 | 3.03 |
| 20480 | torch.float16 | 20971520 | 0.03 | 0.74 | 2.98 |
| 20480 | torch.bfloat16 | 20971520 | 0.03 | 0.76 | 3.04 |
| 20480 | torch.float32 | 20971520 | 0.05 | 0.39 | 3.12 |

### torch

| num_channels | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 128 | torch.float16 | 131072 | 0.01 | 0.02 | 0.10 |
| 128 | torch.bfloat16 | 131072 | 0.01 | 0.02 | 0.10 |
| 128 | torch.float32 | 131072 | 0.00 | 0.03 | 0.24 |
| 4096 | torch.float16 | 4194304 | 0.03 | 0.16 | 0.65 |
| 4096 | torch.bfloat16 | 4194304 | 0.02 | 0.23 | 0.94 |
| 4096 | torch.float32 | 4194304 | 0.02 | 0.21 | 1.66 |
| 10240 | torch.float16 | 10485760 | 0.04 | 0.27 | 1.06 |
| 10240 | torch.bfloat16 | 10485760 | 0.04 | 0.26 | 1.06 |
| 10240 | torch.float32 | 10485760 | 0.05 | 0.23 | 1.81 |
| 20480 | torch.float16 | 20971520 | 0.08 | 0.27 | 1.09 |
| 20480 | torch.bfloat16 | 20971520 | 0.08 | 0.27 | 1.07 |
| 20480 | torch.float32 | 20971520 | 0.09 | 0.23 | 1.80 |

## WhereOp

### tileops

| dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| torch.float16 | 4194304 | 0.01 | 0.33 | 2.32 |
| torch.bfloat16 | 4194304 | 0.01 | 0.33 | 2.33 |
| torch.float32 | 4194304 | 0.02 | 0.21 | 2.68 |
| torch.float16 | 10485760 | 0.03 | 0.40 | 2.77 |
| torch.bfloat16 | 10485760 | 0.03 | 0.40 | 2.78 |
| torch.float32 | 10485760 | 0.04 | 0.23 | 3.03 |
| torch.float16 | 20971520 | 0.05 | 0.43 | 3.02 |
| torch.bfloat16 | 20971520 | 0.05 | 0.43 | 3.02 |
| torch.float32 | 20971520 | 0.09 | 0.24 | 3.06 |
| torch.float8_e4m3fn | 4194304 | 0.01 | 0.42 | 1.67 |
| torch.float8_e5m2 | 4194304 | 0.01 | 0.42 | 1.67 |
| torch.float8_e4m3fn | 10485760 | 0.02 | 0.48 | 1.92 |
| torch.float8_e5m2 | 10485760 | 0.02 | 0.48 | 1.93 |
| torch.float8_e4m3fn | 20971520 | 0.04 | 0.52 | 2.07 |
| torch.float8_e5m2 | 20971520 | 0.04 | 0.51 | 2.03 |

### torch

| dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| torch.float16 | 4194304 | 0.01 | 0.34 | 2.40 |
| torch.bfloat16 | 4194304 | 0.01 | 0.34 | 2.39 |
| torch.float32 | 4194304 | 0.02 | 0.21 | 2.74 |
| torch.float16 | 10485760 | 0.03 | 0.41 | 2.84 |
| torch.bfloat16 | 10485760 | 0.03 | 0.40 | 2.82 |
| torch.float32 | 10485760 | 0.04 | 0.24 | 3.06 |
| torch.float16 | 20971520 | 0.05 | 0.44 | 3.06 |
| torch.bfloat16 | 20971520 | 0.05 | 0.44 | 3.07 |
| torch.float32 | 20971520 | 0.09 | 0.24 | 3.08 |
| torch.float8_e4m3fn | 4194304 | 0.01 | 0.50 | 1.98 |
| torch.float8_e5m2 | 4194304 | 0.01 | 0.49 | 1.98 |
| torch.float8_e4m3fn | 10485760 | 0.02 | 0.62 | 2.48 |
| torch.float8_e5m2 | 10485760 | 0.02 | 0.62 | 2.48 |
| torch.float8_e4m3fn | 20971520 | 0.03 | 0.68 | 2.73 |
| torch.float8_e5m2 | 20971520 | 0.03 | 0.68 | 2.74 |

## MaskedFillOp

### tileops

| dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| torch.float16 | 4194304 | 0.01 | 0.45 | 2.26 |
| torch.bfloat16 | 4194304 | 0.01 | 0.45 | 2.27 |
| torch.float32 | 4194304 | 0.01 | 0.28 | 2.54 |
| torch.float16 | 10485760 | 0.02 | 0.52 | 2.59 |
| torch.bfloat16 | 10485760 | 0.02 | 0.53 | 2.65 |
| torch.float32 | 10485760 | 0.03 | 0.34 | 3.02 |
| torch.float16 | 20971520 | 0.03 | 0.61 | 3.03 |
| torch.bfloat16 | 20971520 | 0.03 | 0.61 | 3.06 |
| torch.float32 | 20971520 | 0.06 | 0.34 | 3.09 |
| torch.float8_e4m3fn | 4194304 | 0.01 | 0.48 | 1.44 |
| torch.float8_e5m2 | 4194304 | 0.02 | 0.28 | 0.84 |
| torch.float8_e4m3fn | 10485760 | 0.02 | 0.61 | 1.83 |
| torch.float8_e5m2 | 10485760 | 0.03 | 0.34 | 1.02 |
| torch.float8_e4m3fn | 20971520 | 0.03 | 0.66 | 1.97 |
| torch.float8_e5m2 | 20971520 | 0.06 | 0.34 | 1.03 |

### torch

| dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| torch.float16 | 4194304 | 0.01 | 0.31 | 1.57 |
| torch.bfloat16 | 4194304 | 0.01 | 0.31 | 1.56 |
| torch.float32 | 4194304 | 0.02 | 0.19 | 1.73 |
| torch.float16 | 10485760 | 0.03 | 0.38 | 1.91 |
| torch.bfloat16 | 10485760 | 0.03 | 0.38 | 1.91 |
| torch.float32 | 10485760 | 0.05 | 0.19 | 1.73 |
| torch.float16 | 20971520 | 0.06 | 0.36 | 1.79 |
| torch.bfloat16 | 20971520 | 0.06 | 0.36 | 1.80 |
| torch.float32 | 20971520 | 0.11 | 0.19 | 1.70 |

### torch-ref

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| masked_fill | torch.float8_e4m3fn | 4194304 | 0.03 | 0.12 | 0.37 |
| masked_fill | torch.float8_e5m2 | 4194304 | 0.03 | 0.13 | 0.38 |
| masked_fill | torch.float8_e4m3fn | 10485760 | 0.07 | 0.14 | 0.42 |
| masked_fill | torch.float8_e5m2 | 10485760 | 0.07 | 0.14 | 0.43 |
| masked_fill | torch.float8_e4m3fn | 20971520 | 0.17 | 0.13 | 0.38 |
| masked_fill | torch.float8_e5m2 | 20971520 | 0.16 | 0.13 | 0.39 |

## alibi

### tileops

| op_name | seq_len | dim | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| alibi | 512 | 64 | torch.float16 | 0.04 | 0.41 | 0.82 |
| alibi | 512 | 64 | torch.bfloat16 | 0.04 | 0.38 | 0.77 |
| alibi | 512 | 64 | torch.float32 | 0.02 | 0.70 | 2.80 |
| alibi | 2048 | 64 | torch.float16 | 0.56 | 0.48 | 0.96 |
| alibi | 2048 | 64 | torch.bfloat16 | 0.56 | 0.48 | 0.96 |
| alibi | 2048 | 64 | torch.float32 | 0.54 | 0.50 | 1.99 |
| alibi | 4096 | 128 | torch.float16 | 1.99 | 1.08 | 2.16 |
| alibi | 4096 | 128 | torch.bfloat16 | 2.02 | 1.06 | 2.12 |
| alibi | 4096 | 128 | torch.float32 | 3.55 | 0.60 | 2.42 |

### torch-ref

| op_name | seq_len | dim | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| alibi | 512 | 64 | torch.float16 | 0.09 | 0.19 | 0.39 |
| alibi | 512 | 64 | torch.bfloat16 | 0.09 | 0.19 | 0.39 |
| alibi | 512 | 64 | torch.float32 | 0.06 | 0.30 | 1.20 |
| alibi | 2048 | 64 | torch.float16 | 1.54 | 0.17 | 0.35 |
| alibi | 2048 | 64 | torch.bfloat16 | 1.55 | 0.17 | 0.35 |
| alibi | 2048 | 64 | torch.float32 | 0.91 | 0.30 | 1.18 |
| alibi | 4096 | 128 | torch.float16 | 17.71 | 0.12 | 0.24 |
| alibi | 4096 | 128 | torch.bfloat16 | 17.71 | 0.12 | 0.24 |
| alibi | 4096 | 128 | torch.float32 | 12.35 | 0.17 | 0.70 |

## sinusoidal

### tileops

| op_name | seq_len | dim | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| sinusoidal | 512 | 256 | torch.float16 | 0.00 | 0.04 | 0.07 |
| sinusoidal | 512 | 256 | torch.bfloat16 | 0.00 | 0.04 | 0.07 |
| sinusoidal | 512 | 256 | torch.float32 | 0.00 | 0.04 | 0.16 |
| sinusoidal | 2048 | 300 | torch.float16 | 0.01 | 0.12 | 0.24 |
| sinusoidal | 2048 | 300 | torch.bfloat16 | 0.01 | 0.12 | 0.24 |
| sinusoidal | 2048 | 300 | torch.float32 | 0.01 | 0.11 | 0.43 |
| sinusoidal | 4096 | 512 | torch.float16 | 0.01 | 0.25 | 0.51 |
| sinusoidal | 4096 | 512 | torch.bfloat16 | 0.01 | 0.26 | 0.51 |
| sinusoidal | 4096 | 512 | torch.float32 | 0.01 | 0.15 | 0.61 |

### torch-ref

| op_name | seq_len | dim | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| sinusoidal | 512 | 256 | torch.float16 | 0.02 | 0.01 | 0.01 |
| sinusoidal | 512 | 256 | torch.bfloat16 | 0.02 | 0.01 | 0.01 |
| sinusoidal | 512 | 256 | torch.float32 | 0.02 | 0.01 | 0.03 |
| sinusoidal | 2048 | 300 | torch.float16 | 0.02 | 0.03 | 0.05 |
| sinusoidal | 2048 | 300 | torch.bfloat16 | 0.02 | 0.03 | 0.05 |
| sinusoidal | 2048 | 300 | torch.float32 | 0.02 | 0.03 | 0.12 |
| sinusoidal | 4096 | 512 | torch.float16 | 0.03 | 0.06 | 0.13 |
| sinusoidal | 4096 | 512 | torch.bfloat16 | 0.03 | 0.06 | 0.13 |
| sinusoidal | 4096 | 512 | torch.float32 | 0.03 | 0.07 | 0.29 |

## leaky_relu_fp8

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| leaky_relu | torch.float8_e4m3fn | 4194304 | 0.01 | 0.40 | 0.80 |
| leaky_relu | torch.float8_e5m2 | 4194304 | 0.03 | 0.13 | 0.27 |
| leaky_relu | torch.float8_e4m3fn | 10485760 | 0.02 | 0.46 | 0.92 |
| leaky_relu | torch.float8_e5m2 | 10485760 | 0.06 | 0.17 | 0.33 |
| leaky_relu | torch.float8_e4m3fn | 20971520 | 0.04 | 0.48 | 0.97 |
| leaky_relu | torch.float8_e5m2 | 20971520 | 0.13 | 0.17 | 0.33 |

### torch-ref

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| leaky_relu | torch.float8_e4m3fn | 4194304 | 0.03 | 0.15 | 0.29 |
| leaky_relu | torch.float8_e5m2 | 4194304 | 0.03 | 0.15 | 0.30 |
| leaky_relu | torch.float8_e4m3fn | 10485760 | 0.06 | 0.17 | 0.33 |
| leaky_relu | torch.float8_e5m2 | 10485760 | 0.06 | 0.17 | 0.34 |
| leaky_relu | torch.float8_e4m3fn | 20971520 | 0.14 | 0.15 | 0.30 |
| leaky_relu | torch.float8_e5m2 | 20971520 | 0.13 | 0.16 | 0.31 |

## elu_fp8

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| elu | torch.float8_e4m3fn | 4194304 | 0.01 | 0.54 | 1.08 |
| elu | torch.float8_e5m2 | 4194304 | 0.02 | 0.22 | 0.45 |
| elu | torch.float8_e4m3fn | 10485760 | 0.02 | 0.69 | 1.39 |
| elu | torch.float8_e5m2 | 10485760 | 0.04 | 0.26 | 0.52 |
| elu | torch.float8_e4m3fn | 20971520 | 0.03 | 0.77 | 1.54 |
| elu | torch.float8_e5m2 | 20971520 | 0.08 | 0.26 | 0.52 |

### torch-ref

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| elu | torch.float8_e4m3fn | 4194304 | 0.03 | 0.12 | 0.25 |
| elu | torch.float8_e5m2 | 4194304 | 0.03 | 0.13 | 0.26 |
| elu | torch.float8_e4m3fn | 10485760 | 0.08 | 0.14 | 0.28 |
| elu | torch.float8_e5m2 | 10485760 | 0.07 | 0.14 | 0.29 |
| elu | torch.float8_e4m3fn | 20971520 | 0.16 | 0.13 | 0.26 |
| elu | torch.float8_e5m2 | 20971520 | 0.15 | 0.14 | 0.27 |

## clamp_fp8

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| clamp | torch.float8_e4m3fn | 4194304 | 0.01 | 0.81 | 1.62 |
| clamp | torch.float8_e5m2 | 4194304 | 0.01 | 0.35 | 0.69 |
| clamp | torch.float8_e4m3fn | 10485760 | 0.01 | 1.10 | 2.20 |
| clamp | torch.float8_e5m2 | 10485760 | 0.02 | 0.43 | 0.85 |
| clamp | torch.float8_e4m3fn | 20971520 | 0.02 | 1.28 | 2.56 |
| clamp | torch.float8_e5m2 | 20971520 | 0.05 | 0.43 | 0.85 |

### torch-ref

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| clamp | torch.float8_e4m3fn | 4194304 | 0.03 | 0.14 | 0.28 |
| clamp | torch.float8_e5m2 | 4194304 | 0.03 | 0.15 | 0.30 |
| clamp | torch.float8_e4m3fn | 10485760 | 0.07 | 0.16 | 0.32 |
| clamp | torch.float8_e5m2 | 10485760 | 0.06 | 0.17 | 0.33 |
| clamp | torch.float8_e4m3fn | 20971520 | 0.14 | 0.15 | 0.30 |
| clamp | torch.float8_e5m2 | 20971520 | 0.14 | 0.15 | 0.31 |

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
| 8 | 128 | torch.float16 | 0.02 | 0.24 | 0.19 |
| 8 | 128 | torch.bfloat16 | 0.02 | 0.23 | 0.19 |
| 4 | 256 | torch.float16 | 0.02 | 0.18 | 0.15 |
| 4 | 64 | torch.float16 | 0.01 | 0.09 | 0.07 |

## LayerNormOp

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.01 | 1.48 | 1.18 |
| 4096 | 4096 | torch.bfloat16 | 0.05 | 1.76 | 1.41 |
| 2048 | 5120 | torch.float16 | 0.03 | 1.58 | 1.27 |
| 1025 | 4096 | torch.float16 | 0.01 | 1.45 | 1.16 |

### torch

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.02 | 1.36 | 1.09 |
| 4096 | 4096 | torch.bfloat16 | 0.04 | 2.03 | 1.62 |
| 2048 | 5120 | torch.float16 | 0.03 | 1.89 | 1.51 |
| 1025 | 4096 | torch.float16 | 0.02 | 1.36 | 1.09 |

## AnyOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | any | 0.01 | 0.36 | 0.72 |
| 1024 | 4096 | torch.bfloat16 | any | 0.01 | 0.36 | 0.72 |
| 1024 | 4096 | torch.float32 | any | 0.02 | 0.24 | 0.96 |
| 1024 | 4096 | torch.int32 | any | 0.03 | 0.14 | 0.56 |
| 4096 | 4096 | torch.float16 | any | 0.03 | 0.52 | 1.05 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | any | 0.03 | 0.14 | 0.28 |
| 1024 | 4096 | torch.bfloat16 | any | 0.03 | 0.14 | 0.27 |
| 1024 | 4096 | torch.float32 | any | 0.03 | 0.13 | 0.53 |
| 1024 | 4096 | torch.int32 | any | 0.03 | 0.13 | 0.53 |
| 4096 | 4096 | torch.float16 | any | 0.08 | 0.20 | 0.41 |

## AllOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | all | 0.01 | 0.36 | 0.73 |
| 1024 | 4096 | torch.bfloat16 | all | 0.01 | 0.36 | 0.72 |
| 1024 | 4096 | torch.float32 | all | 0.02 | 0.24 | 0.96 |
| 1024 | 4096 | torch.int32 | all | 0.03 | 0.14 | 0.55 |
| 4096 | 4096 | torch.float16 | all | 0.03 | 0.53 | 1.05 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | all | 0.03 | 0.14 | 0.28 |
| 1024 | 4096 | torch.bfloat16 | all | 0.03 | 0.14 | 0.28 |
| 1024 | 4096 | torch.float32 | all | 0.03 | 0.14 | 0.54 |
| 1024 | 4096 | torch.int32 | all | 0.03 | 0.14 | 0.54 |
| 4096 | 4096 | torch.float16 | all | 0.08 | 0.21 | 0.41 |

## CountNonzeroOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | count_nonzero | 0.01 | 0.43 | 0.86 |
| 1024 | 4096 | torch.bfloat16 | count_nonzero | 0.01 | 0.43 | 0.86 |
| 1024 | 4096 | torch.float32 | count_nonzero | 0.02 | 0.27 | 1.10 |
| 1024 | 4096 | torch.int32 | count_nonzero | 0.03 | 0.15 | 0.59 |
| 4096 | 4096 | torch.float16 | count_nonzero | 0.03 | 0.55 | 1.10 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | count_nonzero | 0.04 | 0.10 | 0.21 |
| 1024 | 4096 | torch.bfloat16 | count_nonzero | 0.04 | 0.10 | 0.21 |
| 1024 | 4096 | torch.float32 | count_nonzero | 0.04 | 0.10 | 0.38 |
| 1024 | 4096 | torch.int32 | count_nonzero | 0.04 | 0.10 | 0.38 |
| 4096 | 4096 | torch.float16 | count_nonzero | 0.13 | 0.12 | 0.25 |

## MeanPoolingForwardOp

### tileops

| batch_size | seq_len | heads | dim | chunk_size | dtype | accum_dtype | chunks_per_bacth | seq_num | use_offsets | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8192 | 64 | 128 | 64 | torch.float16 | torch.float32 | 128 | 1 | 0 | 0.21 | 0.32 | 0.66 |
| 2 | 2048 | 64 | 128 | 64 | torch.float16 | torch.float32 | 32 | 2 | 0 | 0.10 | 0.34 | 0.68 |
| 1 | 8192 | 64 | 128 | 64 | torch.float16 | torch.float32 | 128 | 4 | 1 | 0.34 | 0.20 | 0.40 |
| 1 | 1000 | 64 | 128 | 32 | torch.float16 | torch.float32 | 34 | 4 | 1 | 0.03 | 0.31 | 0.59 |

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
| 1 | 1024 | 8 | 64 | False | torch.float16 | 0.01 | 166.00 | 0.32 |
| 16 | 2048 | 16 | 128 | False | torch.float16 | 1.91 | 288.24 | 0.28 |
| 4 | 4096 | 16 | 128 | False | torch.bfloat16 | 1.75 | 314.41 | 0.15 |

### torch-sdpa

| batch | seq_len | heads | dim | causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 8 | 64 | False | torch.float16 | 0.02 | 101.84 | 0.20 |
| 16 | 2048 | 16 | 128 | False | torch.float16 | 2.90 | 189.82 | 0.19 |
| 4 | 4096 | 16 | 128 | False | torch.bfloat16 | 2.81 | 195.89 | 0.10 |

## MultiHeadAttentionBwdOp

### tileops

| batch | seq_len | heads | dim | causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 8 | 64 | False | torch.float16 | 0.05 | 106.90 | 0.15 |
| 16 | 2048 | 16 | 128 | False | torch.float16 | 6.92 | 198.67 | 0.14 |
| 4 | 4096 | 16 | 128 | False | torch.bfloat16 | 6.46 | 212.89 | 0.07 |

### torch-sdpa

| batch | seq_len | heads | dim | causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 8 | 64 | False | torch.float16 | 0.07 | 79.31 | 0.11 |
| 16 | 2048 | 16 | 128 | False | torch.float16 | 10.38 | 132.45 | 0.09 |
| 4 | 4096 | 16 | 128 | False | torch.bfloat16 | 10.12 | 135.83 | 0.05 |

## MultiHeadAttentionDecodeWithKVCacheOp

### tileops

| b | h | s_q | s_kv | d | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 128 | 8192 | 128 | torch.float16 | 0.05 | 316.97 | 2.52 |
| 1 | 32 | 128 | 8192 | 128 | torch.bfloat16 | 0.07 | 247.97 | 1.97 |
| 1 | 32 | 128 | 5 | 128 | torch.float16 | 0.01 | 1.68 | 0.35 |

### torch-sdpa

| b | h | s_q | s_kv | d | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 128 | 8192 | 128 | torch.float16 | 0.08 | 206.04 | 1.63 |
| 1 | 32 | 128 | 8192 | 128 | torch.bfloat16 | 0.08 | 208.79 | 1.66 |
| 1 | 32 | 128 | 5 | 128 | torch.float16 | 0.01 | 1.20 | 0.25 |

## MultiHeadAttentionDecodePagedWithKVCacheOp

### tileops

| batch | heads | seqlen_q | seqlen_kv | dim | page_size | is_causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 16 | 1 | 512 | 128 | 128 | False | torch.float16 | 0.02 | 0.17 | 0.17 |
| 2 | 8 | 1 | 1024 | 64 | 256 | False | torch.float16 | 0.02 | 0.19 | 0.10 |
| 1 | 8 | 1 | 1024 | 64 | 256 | False | torch.float16 | 0.02 | 0.09 | 0.09 |
| 1 | 8 | 1 | 512 | 64 | 256 | False | torch.float16 | 0.02 | 0.05 | 0.05 |

### torch-ref

| batch | heads | seqlen_q | seqlen_kv | dim | page_size | is_causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 16 | 1 | 512 | 128 | 128 | False | torch.float16 | 0.04 | 0.10 | 0.10 |
| 2 | 8 | 1 | 1024 | 64 | 256 | False | torch.float16 | 0.08 | 0.05 | 0.03 |
| 1 | 8 | 1 | 1024 | 64 | 256 | False | torch.float16 | 0.04 | 0.05 | 0.05 |
| 1 | 8 | 1 | 512 | 64 | 256 | False | torch.float16 | 0.03 | 0.03 | 0.03 |

## ManifoldConstrainedHyperConnectionPostOp

### tileops

| batch | n_expand | c_x | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | 1280 | torch.bfloat16 | 0.00 | 23.00 | 0.01 |
| 2 | 4 | 1920 | torch.bfloat16 | 0.00 | 100.87 | 0.01 |
| 4 | 4 | 2560 | torch.bfloat16 | 0.00 | 338.00 | 0.01 |

### torch-ref

| batch | n_expand | c_x | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | 1280 | torch.bfloat16 | 0.01 | 3.64 | 0.00 |
| 2 | 4 | 1920 | torch.bfloat16 | 0.01 | 15.78 | 0.00 |
| 4 | 4 | 2560 | torch.bfloat16 | 0.02 | 52.04 | 0.00 |

## ManifoldConstrainedHyperConnectionPreOp

### tileops

| batch | n_expand | c_x | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | 1280 | torch.bfloat16 | 0.04 | 30.28 | 0.00 |
| 2 | 4 | 1920 | torch.bfloat16 | 0.06 | 91.01 | 0.00 |
| 4 | 4 | 2560 | torch.bfloat16 | 0.08 | 246.34 | 0.00 |

### torch-ref

| batch | n_expand | c_x | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | 1280 | torch.bfloat16 | 0.28 | 4.56 | 0.00 |
| 2 | 4 | 1920 | torch.bfloat16 | 0.30 | 18.66 | 0.00 |
| 4 | 4 | 2560 | torch.bfloat16 | 0.34 | 59.35 | 0.00 |

## fused_topk

### tileops

| num_tokens | num_experts | top_k | scoring_func | renormalize | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 128 | 8 | softmax | False | torch.bfloat16 | 0.01 | 0.14 | 0.02 |
| 2048 | 128 | 8 | softmax | False | torch.bfloat16 | 0.01 | 0.51 | 0.07 |
| 4096 | 128 | 8 | softmax | False | torch.bfloat16 | 0.01 | 0.81 | 0.11 |
| 512 | 256 | 8 | softmax | True | torch.bfloat16 | 0.01 | 0.16 | 0.02 |
| 2048 | 256 | 8 | softmax | True | torch.bfloat16 | 0.02 | 0.59 | 0.07 |
| 4096 | 256 | 8 | softmax | True | torch.bfloat16 | 0.02 | 0.94 | 0.12 |
| 512 | 256 | 8 | sigmoid | True | torch.bfloat16 | 0.02 | 0.14 | 0.02 |
| 2048 | 256 | 8 | sigmoid | True | torch.bfloat16 | 0.02 | 0.54 | 0.07 |
| 4096 | 256 | 8 | sigmoid | True | torch.bfloat16 | 0.02 | 0.89 | 0.11 |

### pytorch-ref

| num_tokens | num_experts | top_k | scoring_func | renormalize | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 128 | 8 | softmax | False | torch.bfloat16 | 0.02 | 0.06 | 0.01 |
| 2048 | 128 | 8 | softmax | False | torch.bfloat16 | 0.03 | 0.14 | 0.02 |
| 4096 | 128 | 8 | softmax | False | torch.bfloat16 | 0.05 | 0.19 | 0.03 |
| 512 | 256 | 8 | softmax | True | torch.bfloat16 | 0.03 | 0.07 | 0.01 |
| 2048 | 256 | 8 | softmax | True | torch.bfloat16 | 0.06 | 0.16 | 0.02 |
| 4096 | 256 | 8 | softmax | True | torch.bfloat16 | 0.09 | 0.21 | 0.03 |
| 512 | 256 | 8 | sigmoid | True | torch.bfloat16 | 0.03 | 0.08 | 0.01 |
| 2048 | 256 | 8 | sigmoid | True | torch.bfloat16 | 0.06 | 0.16 | 0.02 |
| 4096 | 256 | 8 | sigmoid | True | torch.bfloat16 | 0.10 | 0.20 | 0.02 |

### vllm

| num_tokens | num_experts | top_k | scoring_func | renormalize | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 128 | 8 | softmax | False | torch.bfloat16 | 0.01 | 0.13 | 0.02 |
| 2048 | 128 | 8 | softmax | False | torch.bfloat16 | 0.01 | 0.41 | 0.06 |
| 4096 | 128 | 8 | softmax | False | torch.bfloat16 | 0.02 | 0.59 | 0.08 |
| 512 | 256 | 8 | softmax | True | torch.bfloat16 | 0.01 | 0.20 | 0.02 |
| 2048 | 256 | 8 | softmax | True | torch.bfloat16 | 0.02 | 0.50 | 0.06 |
| 4096 | 256 | 8 | softmax | True | torch.bfloat16 | 0.02 | 0.80 | 0.10 |
| 512 | 256 | 8 | sigmoid | True | torch.bfloat16 | 0.01 | 0.19 | 0.02 |
| 2048 | 256 | 8 | sigmoid | True | torch.bfloat16 | 0.02 | 0.48 | 0.06 |
| 4096 | 256 | 8 | sigmoid | True | torch.bfloat16 | 0.02 | 0.80 | 0.10 |

## KimiMoENopadOp

### tileops

| num_tokens | num_experts | top_k | hidden_size | ffn_size | routed_scaling_factor | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 384 | 6 | 7168 | 2048 | 2.872 | torch.bfloat16 | 7.45 | 36.33 | 4.54 |
| 2048 | 384 | 6 | 7168 | 2048 | 2.872 | torch.bfloat16 | 11.51 | 94.03 | 2.94 |
| 4096 | 384 | 6 | 7168 | 2048 | 2.872 | torch.bfloat16 | 15.96 | 135.60 | 2.13 |
| 512 | 384 | 6 | 2048 | 1024 | 2.872 | torch.bfloat16 | 0.91 | 42.29 | 5.29 |
| 2048 | 384 | 6 | 2048 | 1024 | 2.872 | torch.bfloat16 | 1.22 | 126.58 | 3.97 |
| 4096 | 384 | 6 | 2048 | 1024 | 2.872 | torch.bfloat16 | 1.92 | 160.83 | 2.53 |

### vllm

| num_tokens | num_experts | top_k | hidden_size | ffn_size | routed_scaling_factor | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 384 | 6 | 7168 | 2048 | 2.872 | torch.bfloat16 | 7.04 | 38.42 | 4.81 |
| 2048 | 384 | 6 | 7168 | 2048 | 2.872 | torch.bfloat16 | 10.87 | 99.55 | 3.12 |
| 4096 | 384 | 6 | 7168 | 2048 | 2.872 | torch.bfloat16 | 13.46 | 160.84 | 2.52 |
| 512 | 384 | 6 | 2048 | 1024 | 2.872 | torch.bfloat16 | 1.01 | 38.11 | 4.77 |
| 2048 | 384 | 6 | 2048 | 1024 | 2.872 | torch.bfloat16 | 1.54 | 100.21 | 3.14 |
| 4096 | 384 | 6 | 2048 | 1024 | 2.872 | torch.bfloat16 | 2.61 | 118.49 | 1.86 |

## KimiMoEPaddedOp

### tileops-padded

| num_tokens | num_experts | top_k | hidden_size | ffn_size | routed_scaling_factor | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 384 | 6 | 7168 | 2048 | 2.872 | torch.bfloat16 | 9.60 | 28.19 | 3.52 |
| 2048 | 384 | 6 | 7168 | 2048 | 2.872 | torch.bfloat16 | 14.09 | 76.79 | 2.40 |
| 4096 | 384 | 6 | 7168 | 2048 | 2.872 | torch.bfloat16 | 17.45 | 124.08 | 1.95 |
| 512 | 384 | 6 | 2048 | 1024 | 2.872 | torch.bfloat16 | 1.41 | 27.37 | 3.42 |
| 2048 | 384 | 6 | 2048 | 1024 | 2.872 | torch.bfloat16 | 1.95 | 79.42 | 2.49 |
| 4096 | 384 | 6 | 2048 | 1024 | 2.872 | torch.bfloat16 | 2.96 | 104.52 | 1.64 |

## MoePermutePaddedOp

### tileops

| total_tokens | top_k | num_experts | hidden_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 2 | 8 | 4096 | torch.bfloat16 | 0.01 | 0.00 | 1.40 |
| 2048 | 2 | 8 | 4096 | torch.bfloat16 | 0.03 | 0.00 | 1.90 |
| 4096 | 2 | 8 | 4096 | torch.bfloat16 | 0.05 | 0.00 | 1.89 |
| 512 | 8 | 256 | 7168 | torch.bfloat16 | 0.06 | 0.00 | 1.14 |
| 2048 | 8 | 256 | 7168 | torch.bfloat16 | 0.25 | 0.00 | 1.04 |
| 4096 | 8 | 256 | 7168 | torch.bfloat16 | 0.64 | 0.00 | 0.83 |
| 512 | 8 | 128 | 2048 | torch.bfloat16 | 0.03 | 0.00 | 0.71 |
| 2048 | 8 | 128 | 2048 | torch.bfloat16 | 0.09 | 0.00 | 0.88 |
| 4096 | 8 | 128 | 2048 | torch.bfloat16 | 0.19 | 0.00 | 0.81 |

### torch-ref

| total_tokens | top_k | num_experts | hidden_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 2 | 8 | 4096 | torch.bfloat16 | 1.35 | 0.00 | 0.01 |
| 2048 | 2 | 8 | 4096 | torch.bfloat16 | 5.53 | 0.00 | 0.01 |
| 4096 | 2 | 8 | 4096 | torch.bfloat16 | 11.43 | 0.00 | 0.01 |
| 512 | 8 | 256 | 7168 | torch.bfloat16 | 5.21 | 0.00 | 0.01 |
| 2048 | 8 | 256 | 7168 | torch.bfloat16 | 22.86 | 0.00 | 0.01 |
| 4096 | 8 | 256 | 7168 | torch.bfloat16 | 47.73 | 0.00 | 0.01 |
| 512 | 8 | 128 | 2048 | torch.bfloat16 | 4.83 | 0.00 | 0.00 |
| 2048 | 8 | 128 | 2048 | torch.bfloat16 | 19.53 | 0.00 | 0.00 |
| 4096 | 8 | 128 | 2048 | torch.bfloat16 | 40.47 | 0.00 | 0.00 |

## MoePermuteAlignOp

### tileops

| total_tokens | top_k | num_experts | block_size | numel | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 2 | 8 | 16 | 1024 | 0.01 | N/A | 0.00 |
| 2048 | 2 | 8 | 16 | 4096 | 0.01 | N/A | 0.00 |
| 4096 | 2 | 8 | 16 | 8192 | 0.01 | N/A | 0.00 |
| 512 | 6 | 64 | 64 | 3072 | 0.01 | N/A | 0.01 |
| 2048 | 6 | 64 | 64 | 12288 | 0.01 | N/A | 0.01 |
| 4096 | 6 | 64 | 128 | 24576 | 0.02 | N/A | 0.01 |
| 8192 | 6 | 64 | 128 | 49152 | 0.04 | N/A | 0.01 |
| 2048 | 6 | 256 | 128 | 12288 | 0.02 | N/A | 0.01 |
| 8192 | 6 | 256 | 128 | 49152 | 0.04 | N/A | 0.01 |

### triton

| total_tokens | top_k | num_experts | block_size | numel | max_num_blocks | max_padded | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 2 | 8 | 16 | 1024 | 73 | 1159 | 0.03 | N/A | 0.00 |
| 2048 | 2 | 8 | 16 | 4096 | 265 | 4231 | 0.11 | N/A | 0.00 |
| 4096 | 2 | 8 | 16 | 8192 | 521 | 8327 | 0.24 | N/A | 0.00 |
| 512 | 6 | 64 | 64 | 3072 | 112 | 7167 | 0.02 | N/A | 0.00 |
| 2048 | 6 | 64 | 64 | 12288 | 256 | 16383 | 0.05 | N/A | 0.00 |
| 4096 | 6 | 64 | 128 | 24576 | 257 | 32831 | 0.09 | N/A | 0.00 |
| 8192 | 6 | 64 | 128 | 49152 | 449 | 57407 | 0.18 | N/A | 0.00 |
| 2048 | 6 | 256 | 128 | 12288 | 351 | 44927 | 0.05 | N/A | 0.00 |
| 8192 | 6 | 256 | 128 | 49152 | 639 | 81791 | 0.08 | N/A | 0.01 |

## Qwen3MoENopadOp

### tileops

| num_tokens | num_experts | top_k | hidden_size | ffn_size | scoring_func | renormalize | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 128 | 8 | 2048 | 1024 | softmax | False | torch.bfloat16 | 0.60 | 85.92 | 2.69 |
| 2048 | 128 | 8 | 2048 | 1024 | softmax | False | torch.bfloat16 | 1.08 | 190.63 | 1.50 |
| 4096 | 128 | 8 | 2048 | 1024 | softmax | False | torch.bfloat16 | 2.08 | 197.81 | 0.79 |
| 512 | 256 | 8 | 2048 | 1024 | softmax | True | torch.bfloat16 | 1.20 | 43.08 | 2.70 |
| 2048 | 256 | 8 | 2048 | 1024 | softmax | True | torch.bfloat16 | 1.55 | 133.24 | 2.09 |
| 4096 | 256 | 8 | 2048 | 1024 | softmax | True | torch.bfloat16 | 2.46 | 167.72 | 1.32 |
| 512 | 256 | 8 | 2048 | 1024 | sigmoid | True | torch.bfloat16 | 1.20 | 42.99 | 2.69 |
| 2048 | 256 | 8 | 2048 | 1024 | sigmoid | True | torch.bfloat16 | 1.55 | 133.18 | 2.09 |
| 4096 | 256 | 8 | 2048 | 1024 | sigmoid | True | torch.bfloat16 | 2.45 | 168.12 | 1.33 |

### torch

| num_tokens | num_experts | top_k | hidden_size | ffn_size | scoring_func | renormalize | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 128 | 8 | 2048 | 1024 | softmax | False | torch.bfloat16 | 12.28 | 4.20 | 0.13 |
| 2048 | 128 | 8 | 2048 | 1024 | softmax | False | torch.bfloat16 | 17.87 | 11.53 | 0.09 |
| 4096 | 128 | 8 | 2048 | 1024 | softmax | False | torch.bfloat16 | 23.90 | 17.25 | 0.07 |
| 512 | 256 | 8 | 2048 | 1024 | softmax | True | torch.bfloat16 | 24.66 | 2.09 | 0.13 |
| 2048 | 256 | 8 | 2048 | 1024 | softmax | True | torch.bfloat16 | 29.44 | 7.00 | 0.11 |
| 4096 | 256 | 8 | 2048 | 1024 | softmax | True | torch.bfloat16 | 36.19 | 11.39 | 0.09 |
| 512 | 256 | 8 | 2048 | 1024 | sigmoid | True | torch.bfloat16 | 24.68 | 2.09 | 0.13 |
| 2048 | 256 | 8 | 2048 | 1024 | sigmoid | True | torch.bfloat16 | 29.44 | 7.00 | 0.11 |
| 4096 | 256 | 8 | 2048 | 1024 | sigmoid | True | torch.bfloat16 | 36.05 | 11.44 | 0.09 |

### vllm

| num_tokens | num_experts | top_k | hidden_size | ffn_size | scoring_func | renormalize | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 128 | 8 | 2048 | 1024 | softmax | False | torch.bfloat16 | 0.89 | 58.16 | 1.82 |
| 2048 | 128 | 8 | 2048 | 1024 | softmax | False | torch.bfloat16 | 2.52 | 81.93 | 0.65 |
| 4096 | 128 | 8 | 2048 | 1024 | softmax | False | torch.bfloat16 | 2.58 | 160.04 | 0.64 |
| 512 | 256 | 8 | 2048 | 1024 | softmax | True | torch.bfloat16 | 1.38 | 37.43 | 2.34 |
| 2048 | 256 | 8 | 2048 | 1024 | softmax | True | torch.bfloat16 | 1.87 | 110.38 | 1.73 |
| 4096 | 256 | 8 | 2048 | 1024 | softmax | True | torch.bfloat16 | 2.81 | 146.59 | 1.16 |

## Qwen3MoEPaddedOp

### tileops-padded

| num_tokens | num_experts | top_k | hidden_size | ffn_size | scoring_func | renormalize | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 128 | 8 | 2048 | 1024 | softmax | False | torch.bfloat16 | 0.81 | 63.86 | 2.00 |
| 2048 | 128 | 8 | 2048 | 1024 | softmax | False | torch.bfloat16 | 1.52 | 135.46 | 1.07 |
| 4096 | 128 | 8 | 2048 | 1024 | softmax | False | torch.bfloat16 | 2.86 | 144.13 | 0.57 |
| 512 | 256 | 8 | 2048 | 1024 | softmax | True | torch.bfloat16 | 1.82 | 28.29 | 1.77 |
| 2048 | 256 | 8 | 2048 | 1024 | softmax | True | torch.bfloat16 | 2.34 | 88.17 | 1.38 |
| 4096 | 256 | 8 | 2048 | 1024 | softmax | True | torch.bfloat16 | 3.67 | 112.49 | 0.89 |
| 512 | 256 | 8 | 2048 | 1024 | sigmoid | True | torch.bfloat16 | 1.82 | 28.26 | 1.77 |
| 2048 | 256 | 8 | 2048 | 1024 | sigmoid | True | torch.bfloat16 | 2.34 | 88.08 | 1.38 |
| 4096 | 256 | 8 | 2048 | 1024 | sigmoid | True | torch.bfloat16 | 3.67 | 112.41 | 0.89 |

## moe_unpermute

### tileops

| total_tokens | top_k | hidden_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 512 | 2 | 4096 | torch.bfloat16 | 0.01 | 1.20 | 1.80 |
| 2048 | 2 | 4096 | torch.bfloat16 | 0.02 | 1.65 | 2.47 |
| 4096 | 2 | 4096 | torch.bfloat16 | 0.04 | 1.84 | 2.77 |
| 512 | 8 | 7168 | torch.bfloat16 | 0.03 | 1.90 | 2.13 |
| 2048 | 8 | 7168 | torch.bfloat16 | 0.10 | 2.36 | 2.65 |
| 4096 | 8 | 7168 | torch.bfloat16 | 0.21 | 2.29 | 2.58 |
| 512 | 8 | 2048 | torch.bfloat16 | 0.01 | 1.75 | 1.97 |
| 2048 | 8 | 2048 | torch.bfloat16 | 0.03 | 2.26 | 2.55 |
| 4096 | 8 | 2048 | torch.bfloat16 | 0.05 | 2.48 | 2.79 |

### torch-ref

| total_tokens | top_k | hidden_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 512 | 2 | 4096 | torch.bfloat16 | 0.06 | 0.14 | 0.21 |
| 2048 | 2 | 4096 | torch.bfloat16 | 0.23 | 0.14 | 0.22 |
| 4096 | 2 | 4096 | torch.bfloat16 | 0.48 | 0.14 | 0.21 |
| 512 | 8 | 7168 | torch.bfloat16 | 0.38 | 0.15 | 0.17 |
| 2048 | 8 | 7168 | torch.bfloat16 | 1.99 | 0.12 | 0.13 |
| 4096 | 8 | 7168 | torch.bfloat16 | 4.63 | 0.10 | 0.11 |
| 512 | 8 | 2048 | torch.bfloat16 | 0.11 | 0.15 | 0.17 |
| 2048 | 8 | 2048 | torch.bfloat16 | 0.45 | 0.15 | 0.17 |
| 4096 | 8 | 2048 | torch.bfloat16 | 0.98 | 0.14 | 0.15 |

## SumOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | sum | 0.01 | 0.43 | 0.86 |
| 1024 | 4096 | torch.bfloat16 | sum | 0.01 | 0.43 | 0.87 |
| 4096 | 4096 | torch.float16 | sum | 0.03 | 0.54 | 1.08 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | sum | 0.03 | 0.14 | 0.28 |
| 1024 | 4096 | torch.bfloat16 | sum | 0.03 | 0.14 | 0.28 |
| 4096 | 4096 | torch.float16 | sum | 0.09 | 0.18 | 0.35 |

## MeanOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | mean | 0.01 | 0.43 | 0.87 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | mean | 0.03 | 0.14 | 0.28 |

## AmaxOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | amax | 0.01 | 0.43 | 0.87 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | amax | 0.03 | 0.14 | 0.28 |

## AminOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | amin | 0.01 | 0.43 | 0.87 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | amin | 0.03 | 0.14 | 0.28 |

## ProdOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | prod | 0.02 | 0.19 | 0.37 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | prod | 0.03 | 0.14 | 0.28 |

## StdOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | std | 0.01 | 1.11 | 0.74 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | std | 0.04 | 0.29 | 0.19 |

## VarOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | var | 0.01 | 1.12 | 0.75 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | var | 0.04 | 0.28 | 0.19 |

## VarMeanOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | var_mean | 0.01 | 1.46 | 0.98 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | var_mean | 0.05 | 0.23 | 0.15 |

## RmsNormOp

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.01 | 1.31 | 1.31 |
| 4096 | 4096 | torch.bfloat16 | 0.04 | 1.62 | 1.62 |
| 2048 | 5120 | torch.float16 | 0.03 | 1.54 | 1.54 |
| 1025 | 4096 | torch.float16 | 0.01 | 1.37 | 1.37 |

### torch-ref

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.08 | 0.22 | 0.22 |
| 4096 | 4096 | torch.bfloat16 | 0.30 | 0.23 | 0.23 |
| 2048 | 5120 | torch.float16 | 0.19 | 0.22 | 0.22 |
| 1025 | 4096 | torch.float16 | 0.08 | 0.22 | 0.22 |

## RopeNeoxOp

### tileops

| seq_len | head_dim | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 2048 | 64 | torch.float16 | 0.00 | 0.20 | 0.30 |
| 2048 | 64 | torch.bfloat16 | 0.00 | 0.20 | 0.30 |
| 2048 | 64 | torch.float32 | 0.00 | 0.18 | 0.54 |
| 2048 | 128 | torch.float16 | 0.00 | 0.35 | 0.52 |
| 2048 | 128 | torch.bfloat16 | 0.00 | 0.35 | 0.52 |
| 2048 | 128 | torch.float32 | 0.00 | 0.28 | 0.85 |
| 4096 | 128 | torch.float16 | 0.00 | 0.56 | 0.85 |
| 4096 | 128 | torch.bfloat16 | 0.00 | 0.56 | 0.84 |
| 4096 | 128 | torch.float32 | 0.00 | 0.43 | 1.29 |

### torch-ref

| seq_len | head_dim | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 2048 | 64 | torch.float16 | 0.01 | 0.04 | 0.06 |
| 2048 | 64 | torch.bfloat16 | 0.01 | 0.04 | 0.06 |
| 2048 | 64 | torch.float32 | 0.01 | 0.04 | 0.12 |
| 2048 | 128 | torch.float16 | 0.02 | 0.07 | 0.10 |
| 2048 | 128 | torch.bfloat16 | 0.02 | 0.07 | 0.10 |
| 2048 | 128 | torch.float32 | 0.02 | 0.07 | 0.21 |
| 4096 | 128 | torch.float16 | 0.02 | 0.12 | 0.18 |
| 4096 | 128 | torch.bfloat16 | 0.02 | 0.12 | 0.18 |
| 4096 | 128 | torch.float32 | 0.02 | 0.12 | 0.35 |

## SoftmaxOp

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.02 | 0.82 | 0.82 |
| 4096 | 4096 | torch.bfloat16 | 0.08 | 0.86 | 0.86 |
| 1024 | 3000 | torch.float16 | 0.03 | 0.47 | 0.47 |
| 1025 | 4096 | torch.float16 | 0.02 | 0.83 | 0.83 |

### torch

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.02 | 0.79 | 0.79 |
| 4096 | 4096 | torch.bfloat16 | 0.08 | 0.83 | 0.83 |
| 1024 | 3000 | torch.float16 | 0.02 | 0.69 | 0.69 |
| 1025 | 4096 | torch.float16 | 0.02 | 0.79 | 0.79 |

## LogSoftmaxOp

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.03 | 0.79 | 0.63 |
| 4096 | 4096 | torch.bfloat16 | 0.11 | 0.77 | 0.62 |
| 1024 | 3000 | torch.float16 | 0.03 | 0.45 | 0.36 |
| 1025 | 4096 | torch.float16 | 0.03 | 0.78 | 0.62 |

### torch

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.02 | 1.13 | 0.90 |
| 4096 | 4096 | torch.bfloat16 | 0.07 | 1.21 | 0.97 |
| 1024 | 3000 | torch.float16 | 0.02 | 0.96 | 0.77 |
| 1025 | 4096 | torch.float16 | 0.02 | 1.13 | 0.91 |

## LogSumExpOp

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.01 | 1.03 | 0.69 |
| 4096 | 4096 | torch.bfloat16 | 0.04 | 1.24 | 0.82 |
| 1024 | 3000 | torch.float16 | 0.02 | 0.56 | 0.37 |
| 1025 | 4096 | torch.float16 | 0.01 | 1.04 | 0.69 |

### torch

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.05 | 0.25 | 0.17 |
| 4096 | 4096 | torch.bfloat16 | 0.11 | 0.47 | 0.31 |
| 1024 | 3000 | torch.float16 | 0.04 | 0.21 | 0.14 |
| 1025 | 4096 | torch.float16 | 0.05 | 0.24 | 0.16 |

## SsdChunkScanFwdOp

### tileops

| batch | num_chunks | chunk_len | n_heads | d_head | d_state | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 64 | 4 | 64 | 32 | torch.float16 | 104.73 | 0.00 | 0.00 |
| 2 | 4 | 64 | 8 | 64 | 64 | torch.float16 | 102.67 | 0.00 | 0.00 |
| 1 | 2 | 128 | 4 | 128 | 32 | torch.bfloat16 | 102.13 | 0.00 | 0.00 |
| 2 | 2 | 64 | 4 | 64 | 32 | torch.bfloat16 | 102.56 | 0.00 | 0.00 |

## ssd_chunk_scan_fwd

### baseline

| batch | num_chunks | chunk_len | n_heads | d_head | d_state | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 64 | 4 | 64 | 32 | torch.float16 | 0.05 | 0.08 | 0.01 |
| 2 | 4 | 64 | 8 | 64 | 64 | torch.float16 | 0.06 | 0.82 | 0.05 |
| 1 | 2 | 128 | 4 | 128 | 32 | torch.bfloat16 | 0.06 | 0.42 | 0.02 |
| 2 | 2 | 64 | 4 | 64 | 32 | torch.bfloat16 | 0.06 | 0.15 | 0.01 |

## SsdChunkStateFwdOp

### tileops

| batch | num_chunks | chunk_len | n_heads | d_head | d_state | n_groups | dtype | has_seq_idx | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 64 | 4 | 64 | 32 | 1 | torch.float16 | False | 87.39 | 0.00 | 0.00 |
| 2 | 4 | 64 | 8 | 64 | 64 | 2 | torch.float16 | False | 86.16 | 0.00 | 0.00 |
| 1 | 2 | 128 | 4 | 128 | 32 | 1 | torch.bfloat16 | False | 87.63 | 0.00 | 0.00 |
| 2 | 2 | 64 | 4 | 64 | 32 | 2 | torch.bfloat16 | False | 85.80 | 0.00 | 0.00 |
| 2 | 4 | 64 | 8 | 64 | 64 | 2 | torch.float16 | True | 87.17 | 0.00 | 0.00 |

## ssd_chunk_state_fwd

### baseline

| batch | num_chunks | chunk_len | n_heads | d_head | d_state | n_groups | dtype | has_seq_idx | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 64 | 4 | 64 | 32 | 1 | torch.float16 | False | 0.03 | 0.06 | 0.00 |
| 2 | 4 | 64 | 8 | 64 | 64 | 2 | torch.float16 | False | 0.11 | 0.29 | 0.02 |
| 1 | 2 | 128 | 4 | 128 | 32 | 1 | torch.bfloat16 | False | 0.05 | 0.18 | 0.01 |
| 2 | 2 | 64 | 4 | 64 | 32 | 2 | torch.bfloat16 | False | 0.04 | 0.11 | 0.01 |
| 2 | 4 | 64 | 8 | 64 | 64 | 2 | torch.float16 | True | 0.12 | 0.28 | 0.01 |

## SsdStatePassingFwdOp

### tileops

| batch | num_chunks | n_heads | d_state | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 4 | 32 | torch.float16 | 51.99 | 0.00 | 0.00 |
| 2 | 4 | 8 | 64 | torch.float16 | 52.99 | 0.00 | 0.00 |
| 1 | 2 | 4 | 32 | torch.bfloat16 | 52.47 | 0.00 | 0.00 |
| 2 | 4 | 8 | 64 | torch.bfloat16 | 52.59 | 0.00 | 0.00 |

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
| 1 | 32768 | 65536 | 1 | 1024 | torch.float32 | torch.int32 | 13.53 | N/A | 0.64 |
| 1 | 32768 | 65536 | 1 | 2048 | torch.float32 | torch.int32 | 14.00 | N/A | 0.63 |
| 1 | 65535 | 131072 | 1 | 1024 | torch.float32 | torch.int32 | 44.62 | N/A | 0.78 |
| 1 | 65535 | 131072 | 1 | 2048 | torch.float32 | torch.int32 | 44.99 | N/A | 0.78 |

### torch

| batch | seq_len | seq_len_kv | kv_group | topk | in_dtype | out_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32768 | 65536 | 1 | 1024 | torch.float32 | torch.int32 | 48.20 | N/A | 0.18 |
| 1 | 32768 | 65536 | 1 | 2048 | torch.float32 | torch.int32 | 48.96 | N/A | 0.18 |
| 1 | 65535 | 131072 | 1 | 1024 | torch.float32 | torch.int32 | 108.79 | N/A | 0.32 |
| 1 | 65535 | 131072 | 1 | 2048 | torch.float32 | torch.int32 | 111.63 | N/A | 0.31 |

## exp

### tileops

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| exp | 262144 | torch.float16 | torch.float16 | 0.00 | 0.10 | 0.38 |
| exp | 1048576 | torch.float16 | torch.float16 | 0.00 | 0.26 | 1.03 |
| exp | 4000000 | torch.float16 | torch.float16 | 0.01 | 0.52 | 2.06 |
| exp | 262144 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.10 | 0.38 |
| exp | 1048576 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.26 | 1.04 |
| exp | 4000000 | torch.bfloat16 | torch.bfloat16 | 0.01 | 0.51 | 2.06 |

### torch

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| exp | 262144 | torch.float16 | torch.float16 | 0.00 | 0.09 | 0.37 |
| exp | 1048576 | torch.float16 | torch.float16 | 0.00 | 0.26 | 1.02 |
| exp | 4000000 | torch.float16 | torch.float16 | 0.01 | 0.53 | 2.11 |
| exp | 262144 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.09 | 0.37 |
| exp | 1048576 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.26 | 1.02 |
| exp | 4000000 | torch.bfloat16 | torch.bfloat16 | 0.01 | 0.53 | 2.10 |

## gelu

### tileops

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| gelu | 262144 | torch.float16 | torch.float16 | 0.00 | 0.09 | 0.37 |
| gelu | 1048576 | torch.float16 | torch.float16 | 0.00 | 0.25 | 0.98 |
| gelu | 4000000 | torch.float16 | torch.float16 | 0.01 | 0.39 | 1.56 |
| gelu | 262144 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.09 | 0.37 |
| gelu | 1048576 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.24 | 0.94 |
| gelu | 4000000 | torch.bfloat16 | torch.bfloat16 | 0.01 | 0.38 | 1.51 |

### torch

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| gelu | 262144 | torch.float16 | torch.float16 | 0.00 | 0.09 | 0.35 |
| gelu | 1048576 | torch.float16 | torch.float16 | 0.00 | 0.24 | 0.97 |
| gelu | 4000000 | torch.float16 | torch.float16 | 0.01 | 0.41 | 1.64 |
| gelu | 262144 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.09 | 0.35 |
| gelu | 1048576 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.24 | 0.96 |
| gelu | 4000000 | torch.bfloat16 | torch.bfloat16 | 0.01 | 0.42 | 1.68 |

## logical_not

### tileops

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| logical_not | 262144 | torch.float16 | torch.bool | 0.00 | 0.09 | 0.27 |
| logical_not | 1048576 | torch.float16 | torch.bool | 0.01 | 0.18 | 0.54 |
| logical_not | 4000000 | torch.float16 | torch.bool | 0.02 | 0.25 | 0.74 |

### torch

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| logical_not | 262144 | torch.float16 | torch.bool | 0.00 | 0.10 | 0.29 |
| logical_not | 1048576 | torch.float16 | torch.bool | 0.00 | 0.29 | 0.88 |
| logical_not | 4000000 | torch.float16 | torch.bool | 0.01 | 0.62 | 1.85 |

## bitwise_not

### tileops

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| bitwise_not | 262144 | torch.int32 | torch.int32 | 0.00 | 0.08 | 0.65 |
| bitwise_not | 1048576 | torch.int32 | torch.int32 | 0.01 | 0.16 | 1.27 |
| bitwise_not | 4000000 | torch.int32 | torch.int32 | 0.02 | 0.22 | 1.73 |

### torch

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| bitwise_not | 262144 | torch.int32 | torch.int32 | 0.00 | 0.09 | 0.71 |
| bitwise_not | 1048576 | torch.int32 | torch.int32 | 0.00 | 0.21 | 1.68 |
| bitwise_not | 4000000 | torch.int32 | torch.int32 | 0.01 | 0.33 | 2.65 |

## isnan

### tileops

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| isnan | 262144 | torch.float16 | torch.bool | 0.00 | 0.09 | 0.27 |
| isnan | 1048576 | torch.float16 | torch.bool | 0.01 | 0.18 | 0.53 |
| isnan | 4000000 | torch.float16 | torch.bool | 0.02 | 0.25 | 0.74 |

### torch

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| isnan | 262144 | torch.float16 | torch.bool | 0.00 | 0.10 | 0.29 |
| isnan | 1048576 | torch.float16 | torch.bool | 0.00 | 0.29 | 0.88 |
| isnan | 4000000 | torch.float16 | torch.bool | 0.01 | 0.61 | 1.84 |

## unary_strategy

### relu_direct

| n_total | shape_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4194304 | 1024x4096 | torch.float16 | direct | 0.02 | 0.24 | 0.96 |
| 4194304 | 1024x4096 | torch.bfloat16 | direct | 0.02 | 0.24 | 0.97 |
| 4194304 | 1024x4096 | torch.float32 | direct | 0.02 | 0.22 | 1.76 |
| 10485760 | 1024x10240 | torch.float16 | direct | 0.04 | 0.25 | 1.01 |
| 10485760 | 1024x10240 | torch.bfloat16 | direct | 0.04 | 0.25 | 1.00 |
| 10485760 | 1024x10240 | torch.float32 | direct | 0.05 | 0.23 | 1.82 |
| 20971520 | 1024x20480 | torch.float16 | direct | 0.08 | 0.25 | 1.00 |
| 20971520 | 1024x20480 | torch.bfloat16 | direct | 0.08 | 0.25 | 1.00 |
| 20971520 | 1024x20480 | torch.float32 | direct | 0.09 | 0.22 | 1.78 |

### relu_explicit_parallel

| n_total | shape_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4194304 | 1024x4096 | torch.float16 | explicit_parallel | 0.01 | 0.51 | 2.04 |
| 4194304 | 1024x4096 | torch.bfloat16 | explicit_parallel | 0.01 | 0.50 | 1.98 |
| 4194304 | 1024x4096 | torch.float32 | explicit_parallel | 0.01 | 0.31 | 2.48 |
| 10485760 | 1024x10240 | torch.float16 | explicit_parallel | 0.02 | 0.61 | 2.45 |
| 10485760 | 1024x10240 | torch.bfloat16 | explicit_parallel | 0.02 | 0.62 | 2.47 |
| 10485760 | 1024x10240 | torch.float32 | explicit_parallel | 0.03 | 0.37 | 2.97 |
| 20971520 | 1024x20480 | torch.float16 | explicit_parallel | 0.03 | 0.66 | 2.65 |
| 20971520 | 1024x20480 | torch.bfloat16 | explicit_parallel | 0.03 | 0.67 | 2.66 |
| 20971520 | 1024x20480 | torch.float32 | explicit_parallel | 0.05 | 0.39 | 3.09 |

### relu_register_copy

| n_total | shape_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4194304 | 1024x4096 | torch.float16 | register_copy | 0.01 | 0.55 | 2.20 |
| 4194304 | 1024x4096 | torch.bfloat16 | register_copy | 0.01 | 0.55 | 2.20 |
| 4194304 | 1024x4096 | torch.float32 | register_copy | 0.01 | 0.32 | 2.54 |
| 10485760 | 1024x10240 | torch.float16 | register_copy | 0.02 | 0.69 | 2.75 |
| 10485760 | 1024x10240 | torch.bfloat16 | register_copy | 0.02 | 0.68 | 2.71 |
| 10485760 | 1024x10240 | torch.float32 | register_copy | 0.03 | 0.38 | 3.05 |
| 20971520 | 1024x20480 | torch.float16 | register_copy | 0.03 | 0.76 | 3.04 |
| 20971520 | 1024x20480 | torch.bfloat16 | register_copy | 0.03 | 0.76 | 3.03 |
| 20971520 | 1024x20480 | torch.float32 | register_copy | 0.05 | 0.40 | 3.17 |

## L1NormOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | l1 | 0.01 | 0.43 | 0.86 |
| 1024 | 4096 | torch.bfloat16 | l1 | 0.01 | 0.44 | 0.87 |
| 1024 | 4096 | torch.float32 | l1 | 0.02 | 0.28 | 1.10 |
| 4096 | 4096 | torch.float16 | l1 | 0.03 | 0.55 | 1.10 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | l1 | 0.02 | 0.20 | 0.40 |
| 1024 | 4096 | torch.bfloat16 | l1 | 0.02 | 0.20 | 0.41 |
| 1024 | 4096 | torch.float32 | l1 | 0.02 | 0.18 | 0.72 |
| 4096 | 4096 | torch.float16 | l1 | 0.03 | 0.62 | 1.25 |

## L2NormOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | l2 | 0.01 | 0.42 | 0.85 |
| 1024 | 4096 | torch.bfloat16 | l2 | 0.01 | 0.43 | 0.86 |
| 1024 | 4096 | torch.float32 | l2 | 0.02 | 0.26 | 1.04 |
| 4096 | 4096 | torch.float16 | l2 | 0.03 | 0.54 | 1.08 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | l2 | 0.02 | 0.20 | 0.41 |
| 1024 | 4096 | torch.bfloat16 | l2 | 0.02 | 0.20 | 0.41 |
| 1024 | 4096 | torch.float32 | l2 | 0.02 | 0.18 | 0.72 |
| 4096 | 4096 | torch.float16 | l2 | 0.03 | 0.63 | 1.25 |

## InfNormOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | inf | 0.03 | 0.15 | 0.29 |
| 1024 | 4096 | torch.bfloat16 | inf | 0.03 | 0.15 | 0.29 |
| 1024 | 4096 | torch.float32 | inf | 0.03 | 0.12 | 0.49 |
| 4096 | 4096 | torch.float16 | inf | 0.06 | 0.29 | 0.58 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | inf | 0.02 | 0.20 | 0.40 |
| 1024 | 4096 | torch.bfloat16 | inf | 0.02 | 0.20 | 0.40 |
| 1024 | 4096 | torch.float32 | inf | 0.02 | 0.18 | 0.70 |
| 4096 | 4096 | torch.float16 | inf | 0.03 | 0.62 | 1.23 |

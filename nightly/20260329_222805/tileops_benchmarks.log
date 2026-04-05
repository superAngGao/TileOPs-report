........................................................................ [  7%]
........................................................................ [ 15%]
........................................................................ [ 23%]
........................................................................ [ 31%]
........................................................................ [ 39%]
........................................................................ [ 47%]
........................................................................ [ 54%]
........................................................................ [ 62%]
........................................................................ [ 70%]
........................................................................ [ 78%]
........................................................................ [ 86%]
........................................................................ [ 94%]
......................................................                   [100%]Benchmark report saved to profile_run.log

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
918 passed, 17 warnings in 862.39s (0:14:22)

===== profile_run.log summary =====
# TileOPs Benchmark Report
Generated: 2026-03-29 19:04:23

## Environment

- **Torch version**: 2.10.0+cu128
- **CUDA version (torch)**: 12.8
- **GPU model**: NVIDIA H200
- **Driver version**: 575.57.08

## ReluOp

### tileops

| n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 4096 | torch.float16 | 0.05 | 0.00 | 0.00 |
| 4194304 | torch.float16 | 0.04 | 0.09 | 0.38 |
| 4194304 | torch.bfloat16 | 0.05 | 0.09 | 0.37 |
| 4194304 | torch.float32 | 0.05 | 0.09 | 0.70 |

### torch

| n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 4096 | torch.float16 | 0.02 | 0.00 | 0.00 |
| 4194304 | torch.float16 | 0.01 | 0.29 | 1.14 |
| 4194304 | torch.bfloat16 | 0.01 | 0.29 | 1.17 |
| 4194304 | torch.float32 | 0.02 | 0.25 | 2.02 |

## r3_jit_cost

### relu_jit

| n_total | cold_ms | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 1000 | 20.37 | 0.05 | 0.00 | 0.00 |
| 2000 | 19.74 | 0.05 | 0.00 | 0.00 |
| 4000 | 18.64 | 0.05 | 0.00 | 0.00 |
| 8000 | 18.73 | 0.05 | 0.00 | 0.00 |
| 16000 | 18.4 | 0.05 | 0.00 | 0.00 |
| 32000 | 18.63 | 0.05 | 0.00 | 0.00 |
| 64000 | 18.88 | 0.05 | 0.00 | 0.01 |
| 128000 | 18.75 | 0.05 | 0.00 | 0.01 |
| 256000 | 18.28 | 0.05 | 0.01 | 0.02 |
| 512000 | 18.86 | 0.05 | 0.01 | 0.04 |

## r4_strategy_unary

### relu_direct

| n_total | size_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | torch.float16 | direct | 0.05 | 0.00 | 0.00 |
| 4096 | 4K | torch.bfloat16 | direct | 0.05 | 0.00 | 0.00 |
| 4096 | 4K | torch.float32 | direct | 0.05 | 0.00 | 0.00 |
| 4194304 | 4M | torch.float16 | direct | 0.05 | 0.08 | 0.34 |
| 4194304 | 4M | torch.bfloat16 | direct | 0.05 | 0.09 | 0.36 |
| 4194304 | 4M | torch.float32 | direct | 0.05 | 0.09 | 0.70 |
| 16777216 | 16M | torch.float16 | direct | 0.06 | 0.29 | 1.17 |
| 16777216 | 16M | torch.bfloat16 | direct | 0.07 | 0.25 | 1.01 |
| 16777216 | 16M | torch.float32 | direct | 0.06 | 0.27 | 2.18 |

### relu_explicit_parallel

| n_total | size_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | torch.float16 | explicit_parallel | 0.05 | 0.00 | 0.00 |
| 4096 | 4K | torch.bfloat16 | explicit_parallel | 0.05 | 0.00 | 0.00 |
| 4096 | 4K | torch.float32 | explicit_parallel | 0.05 | 0.00 | 0.00 |
| 4194304 | 4M | torch.float16 | explicit_parallel | 0.05 | 0.08 | 0.34 |
| 4194304 | 4M | torch.bfloat16 | explicit_parallel | 0.05 | 0.09 | 0.37 |
| 4194304 | 4M | torch.float32 | explicit_parallel | 0.05 | 0.09 | 0.69 |
| 16777216 | 16M | torch.float16 | explicit_parallel | 0.06 | 0.28 | 1.11 |
| 16777216 | 16M | torch.bfloat16 | explicit_parallel | 0.05 | 0.33 | 1.33 |
| 16777216 | 16M | torch.float32 | explicit_parallel | 0.06 | 0.29 | 2.34 |

### relu_register_copy

| n_total | size_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | torch.float16 | register_copy | 0.05 | 0.00 | 0.00 |
| 4096 | 4K | torch.bfloat16 | register_copy | 0.05 | 0.00 | 0.00 |
| 4096 | 4K | torch.float32 | register_copy | 0.05 | 0.00 | 0.00 |
| 4194304 | 4M | torch.float16 | register_copy | 0.05 | 0.09 | 0.35 |
| 4194304 | 4M | torch.bfloat16 | register_copy | 0.05 | 0.09 | 0.36 |
| 4194304 | 4M | torch.float32 | register_copy | 0.05 | 0.08 | 0.67 |
| 16777216 | 16M | torch.float16 | register_copy | 0.05 | 0.31 | 1.23 |
| 16777216 | 16M | torch.bfloat16 | register_copy | 0.05 | 0.33 | 1.34 |
| 16777216 | 16M | torch.float32 | register_copy | 0.06 | 0.28 | 2.25 |

## r4_strategy_gelu

### gelu_direct

| n_total | size_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | torch.float16 | direct | 0.05 | 0.00 | 0.00 |
| 4096 | 4K | torch.bfloat16 | direct | 0.05 | 0.00 | 0.00 |
| 4096 | 4K | torch.float32 | direct | 0.05 | 0.00 | 0.00 |
| 4194304 | 4M | torch.float16 | direct | 0.05 | 0.09 | 0.34 |
| 4194304 | 4M | torch.bfloat16 | direct | 0.05 | 0.08 | 0.34 |
| 4194304 | 4M | torch.float32 | direct | 0.05 | 0.09 | 0.73 |
| 16777216 | 16M | torch.float16 | direct | 0.06 | 0.27 | 1.08 |
| 16777216 | 16M | torch.bfloat16 | direct | 0.06 | 0.27 | 1.08 |
| 16777216 | 16M | torch.float32 | direct | 0.07 | 0.25 | 2.01 |

### gelu_explicit_parallel

| n_total | size_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | torch.float16 | explicit_parallel | 0.05 | 0.00 | 0.00 |
| 4096 | 4K | torch.bfloat16 | explicit_parallel | 0.05 | 0.00 | 0.00 |
| 4096 | 4K | torch.float32 | explicit_parallel | 0.05 | 0.00 | 0.00 |
| 4194304 | 4M | torch.float16 | explicit_parallel | 0.05 | 0.09 | 0.35 |
| 4194304 | 4M | torch.bfloat16 | explicit_parallel | 0.05 | 0.08 | 0.33 |
| 4194304 | 4M | torch.float32 | explicit_parallel | 0.05 | 0.09 | 0.69 |
| 16777216 | 16M | torch.float16 | explicit_parallel | 0.05 | 0.32 | 1.28 |
| 16777216 | 16M | torch.bfloat16 | explicit_parallel | 0.05 | 0.33 | 1.32 |
| 16777216 | 16M | torch.float32 | explicit_parallel | 0.05 | 0.31 | 2.48 |

### gelu_register_copy

| n_total | size_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | torch.float16 | register_copy | 0.05 | 0.00 | 0.00 |
| 4096 | 4K | torch.bfloat16 | register_copy | 0.05 | 0.00 | 0.00 |
| 4096 | 4K | torch.float32 | register_copy | 0.05 | 0.00 | 0.00 |
| 4194304 | 4M | torch.float16 | register_copy | 0.05 | 0.09 | 0.35 |
| 4194304 | 4M | torch.bfloat16 | register_copy | 0.05 | 0.09 | 0.35 |
| 4194304 | 4M | torch.float32 | register_copy | 0.05 | 0.09 | 0.69 |
| 16777216 | 16M | torch.float16 | register_copy | 0.05 | 0.32 | 1.30 |
| 16777216 | 16M | torch.bfloat16 | register_copy | 0.07 | 0.26 | 1.03 |
| 16777216 | 16M | torch.float32 | register_copy | 0.05 | 0.33 | 2.62 |

## r5_boundary

### relu_aligned

| n_total | align_label | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 2048000 | aligned | 0.06 | 0.04 | 0.14 |

### relu_unaligned_plus_1

| n_total | align_label | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 2048001 | unaligned_plus_1 | 0.05 | 0.04 | 0.16 |

### relu_unaligned_plus_127

| n_total | align_label | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 2048127 | unaligned_plus_127 | 0.05 | 0.04 | 0.17 |

## r6_threads

### relu_t128

| n_total | size_label | op_name | threads | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | relu | 128 | 0.02 | 0.00 | 0.00 |
| 4194304 | 4M | relu | 128 | 0.02 | 0.17 | 0.69 |
| 16777216 | 16M | relu | 128 | 0.07 | 0.24 | 0.95 |

### relu_t256

| n_total | size_label | op_name | threads | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | relu | 256 | 0.02 | 0.00 | 0.00 |
| 4194304 | 4M | relu | 256 | 0.02 | 0.17 | 0.70 |
| 16777216 | 16M | relu | 256 | 0.07 | 0.23 | 0.92 |

### erf_t128

| n_total | size_label | op_name | threads | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | erf | 128 | 0.02 | 0.00 | 0.00 |
| 4194304 | 4M | erf | 128 | 0.02 | 0.24 | 0.96 |
| 16777216 | 16M | erf | 128 | 0.03 | 0.52 | 2.07 |

### erf_t256

| n_total | size_label | op_name | threads | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | erf | 256 | 0.02 | 0.00 | 0.00 |
| 4194304 | 4M | erf | 256 | 0.02 | 0.23 | 0.91 |
| 16777216 | 16M | erf | 256 | 0.03 | 0.52 | 2.06 |

### mish_t128

| n_total | size_label | op_name | threads | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | mish | 128 | 0.02 | 0.00 | 0.00 |
| 4194304 | 4M | mish | 128 | 0.02 | 0.21 | 0.85 |
| 16777216 | 16M | mish | 128 | 0.04 | 0.38 | 1.51 |

### mish_t256

| n_total | size_label | op_name | threads | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | mish | 256 | 0.02 | 0.00 | 0.00 |
| 4194304 | 4M | mish | 256 | 0.02 | 0.21 | 0.84 |
| 16777216 | 16M | mish | 256 | 0.04 | 0.37 | 1.50 |

## r7_dtype_npt

### relu_fp32_npt4

| dtype_label | num_per_thread | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| fp32 | 4 | 0.02 | 0.06 | 0.48 |

### relu_fp32_npt8

| dtype_label | num_per_thread | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| fp32 | 8 | 0.02 | 0.06 | 0.48 |

### relu_fp16_npt4

| dtype_label | num_per_thread | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| fp16 | 4 | 0.02 | 0.06 | 0.24 |

### relu_fp16_npt8

| dtype_label | num_per_thread | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| fp16 | 8 | 0.02 | 0.06 | 0.24 |

## AdaLayerNormOp

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.07 | 0.29 | 0.47 |
| 4096 | 4096 | torch.bfloat16 | 0.08 | 1.07 | 1.70 |
| 1024 | 3000 | torch.float16 | 0.15 | 0.10 | 0.16 |
| 1025 | 4096 | torch.float16 | 0.07 | 0.29 | 0.47 |

### torch-ref

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.04 | 0.49 | 0.79 |
| 4096 | 4096 | torch.bfloat16 | 0.09 | 0.97 | 1.55 |
| 1024 | 3000 | torch.float16 | 0.04 | 0.38 | 0.60 |
| 1025 | 4096 | torch.float16 | 0.04 | 0.51 | 0.82 |

## AdaLayerNormZeroOp

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.08 | 0.33 | 0.55 |
| 4096 | 4096 | torch.bfloat16 | 0.08 | 1.25 | 2.08 |
| 1024 | 3000 | torch.float16 | 0.17 | 0.11 | 0.18 |
| 1025 | 4096 | torch.float16 | 0.08 | 0.33 | 0.55 |

### torch-ref

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.05 | 0.51 | 0.85 |
| 4096 | 4096 | torch.bfloat16 | 0.11 | 0.88 | 1.46 |
| 1024 | 3000 | torch.float16 | 0.05 | 0.38 | 0.63 |
| 1025 | 4096 | torch.float16 | 0.05 | 0.50 | 0.83 |

## ArgmaxOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | argmax | 3.09 | 0.00 | 0.00 |
| 1024 | 4096 | torch.bfloat16 | argmax | 3.09 | 0.00 | 0.00 |
| 4096 | 4096 | torch.float16 | argmax | 10.69 | 0.00 | 0.00 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | argmax | 0.03 | 0.15 | 0.31 |
| 1024 | 4096 | torch.bfloat16 | argmax | 0.03 | 0.16 | 0.32 |
| 4096 | 4096 | torch.float16 | argmax | 0.03 | 0.51 | 1.03 |

## ArgminOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | argmin | 3.48 | 0.00 | 0.00 |
| 1024 | 4096 | torch.bfloat16 | argmin | 3.49 | 0.00 | 0.00 |
| 4096 | 4096 | torch.float16 | argmin | 12.11 | 0.00 | 0.00 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | argmin | 0.03 | 0.16 | 0.32 |
| 1024 | 4096 | torch.bfloat16 | argmin | 0.03 | 0.16 | 0.31 |
| 4096 | 4096 | torch.float16 | argmin | 0.03 | 0.51 | 1.03 |

## BatchNormFwdOp

### tileops

| N | C | spatial | dtype | training | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 64 | () | torch.float16 | True | 0.12 | 0.00 | 0.00 |
| 8 | 64 | (32, 32) | torch.float16 | True | 0.12 | 0.04 | 0.02 |
| 4 | 128 | (32, 32) | torch.float16 | True | 0.13 | 0.04 | 0.02 |
| 4 | 256 | (28, 28) | torch.float16 | True | 0.12 | 0.07 | 0.03 |
| 4 | 128 | (1024, 1024) | torch.float16 | True | 3.95 | 1.36 | 0.54 |
| 4 | 256 | (1024, 1024) | torch.float16 | True | 7.34 | 1.46 | 0.58 |

### torch-cudnn

| N | C | spatial | dtype | training | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 64 | () | torch.float16 | True | 0.07 | 0.00 | 0.00 |
| 8 | 64 | (32, 32) | torch.float16 | True | 0.07 | 0.07 | 0.03 |
| 4 | 128 | (32, 32) | torch.float16 | True | 0.07 | 0.07 | 0.03 |
| 4 | 256 | (28, 28) | torch.float16 | True | 0.07 | 0.11 | 0.04 |
| 4 | 128 | (1024, 1024) | torch.float16 | True | 4.14 | 1.30 | 0.52 |
| 4 | 256 | (1024, 1024) | torch.float16 | True | 7.29 | 1.47 | 0.59 |

## BatchNormBwdOp

### tileops

| N | C | spatial | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 64 | () | torch.float16 | 0.13 | 0.00 | 0.00 |
| 8 | 64 | (32, 32) | torch.float16 | 0.13 | 0.03 | 0.02 |
| 4 | 128 | (32, 32) | torch.float16 | 0.13 | 0.03 | 0.02 |
| 4 | 256 | (28, 28) | torch.float16 | 0.13 | 0.05 | 0.04 |
| 4 | 128 | (1024, 1024) | torch.float16 | 6.23 | 0.69 | 0.52 |
| 4 | 256 | (1024, 1024) | torch.float16 | 11.30 | 0.76 | 0.57 |

### torch-autograd

| N | C | spatial | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 64 | () | torch.float16 | 0.22 | 0.00 | 0.00 |
| 8 | 64 | (32, 32) | torch.float16 | 0.23 | 0.02 | 0.01 |
| 4 | 128 | (32, 32) | torch.float16 | 0.22 | 0.02 | 0.01 |
| 4 | 256 | (28, 28) | torch.float16 | 0.22 | 0.03 | 0.02 |
| 4 | 128 | (1024, 1024) | torch.float16 | 12.27 | 0.35 | 0.26 |
| 4 | 256 | (1024, 1024) | torch.float16 | 18.18 | 0.47 | 0.35 |

## r1_vectorization

### add_same_shape_1d

| pattern_name | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| same_shape_1d | 1000000 | 0.05 | 0.02 | 0.12 |

### torch-same_shape_1d

| pattern_name | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| same_shape_1d | 1000000 | 0.01 | 0.08 | 0.45 |

### add_bias_add_2d

| pattern_name | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| bias_add_2d | 1000000 | 0.05 | 0.02 | 0.08 |

### torch-bias_add_2d

| pattern_name | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| bias_add_2d | 1000000 | 0.01 | 0.07 | 0.30 |

### add_interleaved_3d

| pattern_name | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| interleaved_3d | 1048576 | 0.05 | 0.02 | 0.04 |

### torch-interleaved_3d

| pattern_name | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| interleaved_3d | 1048576 | 0.01 | 0.07 | 0.15 |

## r2_small_tensor_binary

### add_same_shape

| pattern_name | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| same_shape | 4096 | 0.05 | 0.00 | 0.00 |

### torch-same_shape

| pattern_name | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| same_shape | 4096 | 0.01 | 0.00 | 0.00 |

### add_broadcast_3d

| pattern_name | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| broadcast_3d | 4096 | 0.05 | 0.00 | 0.00 |

### torch-broadcast_3d

| pattern_name | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| broadcast_3d | 4096 | 0.01 | 0.00 | 0.00 |

## r4_strategy_binary

### add_direct_same_shape

| size_label | pattern_name | strategy | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4K | same_shape | direct | 4096 | 0.05 | 0.00 | 0.00 |
| 4K | same_shape | direct | 4096 | 0.05 | 0.00 | 0.00 |
| 4K | same_shape | direct | 4096 | 0.05 | 0.00 | 0.00 |
| 1M | same_shape | direct | 1048576 | 0.05 | 0.02 | 0.13 |
| 1M | same_shape | direct | 1048576 | 0.05 | 0.02 | 0.13 |
| 1M | same_shape | direct | 1048576 | 0.05 | 0.02 | 0.25 |
| 16M | same_shape | direct | 16777216 | 0.06 | 0.27 | 1.63 |
| 16M | same_shape | direct | 16777216 | 0.06 | 0.27 | 1.63 |
| 16M | same_shape | direct | 16777216 | 0.07 | 0.23 | 2.82 |

### add_direct_bias_add_2d

| size_label | pattern_name | strategy | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4K | bias_add_2d | direct | 4096 | 0.05 | 0.00 | 0.00 |
| 4K | bias_add_2d | direct | 4096 | 0.05 | 0.00 | 0.00 |
| 4K | bias_add_2d | direct | 4096 | 0.05 | 0.00 | 0.00 |
| 1M | bias_add_2d | direct | 1048576 | 0.05 | 0.02 | 0.08 |
| 1M | bias_add_2d | direct | 1048576 | 0.05 | 0.02 | 0.08 |
| 1M | bias_add_2d | direct | 1048576 | 0.05 | 0.02 | 0.17 |
| 16M | bias_add_2d | direct | 16777216 | 0.06 | 0.29 | 1.16 |
| 16M | bias_add_2d | direct | 16777216 | 0.06 | 0.29 | 1.17 |
| 16M | bias_add_2d | direct | 16777216 | 0.06 | 0.27 | 2.13 |

### add_direct_interleaved_3d

| size_label | pattern_name | strategy | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4K | interleaved_3d | direct | 4096 | 0.05 | 0.00 | 0.00 |
| 4K | interleaved_3d | direct | 4096 | 0.05 | 0.00 | 0.00 |
| 4K | interleaved_3d | direct | 4096 | 0.05 | 0.00 | 0.00 |
| 1M | interleaved_3d | direct | 1048576 | 0.05 | 0.02 | 0.04 |
| 1M | interleaved_3d | direct | 1048576 | 0.05 | 0.02 | 0.04 |
| 1M | interleaved_3d | direct | 1048576 | 0.05 | 0.02 | 0.08 |
| 16M | interleaved_3d | direct | 16777216 | 0.06 | 0.29 | 0.59 |
| 16M | interleaved_3d | direct | 16777216 | 0.06 | 0.29 | 0.58 |
| 16M | interleaved_3d | direct | 16777216 | 0.06 | 0.27 | 1.07 |

### add_explicit_parallel_same_shape

| size_label | pattern_name | strategy | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4K | same_shape | explicit_parallel | 4096 | 0.05 | 0.00 | 0.00 |
| 4K | same_shape | explicit_parallel | 4096 | 0.05 | 0.00 | 0.00 |
| 4K | same_shape | explicit_parallel | 4096 | 0.05 | 0.00 | 0.00 |
| 1M | same_shape | explicit_parallel | 1048576 | 0.05 | 0.02 | 0.12 |
| 1M | same_shape | explicit_parallel | 1048576 | 0.05 | 0.02 | 0.13 |
| 1M | same_shape | explicit_parallel | 1048576 | 0.05 | 0.02 | 0.25 |
| 16M | same_shape | explicit_parallel | 16777216 | 0.06 | 0.30 | 1.82 |
| 16M | same_shape | explicit_parallel | 16777216 | 0.06 | 0.30 | 1.80 |
| 16M | same_shape | explicit_parallel | 16777216 | 0.07 | 0.25 | 3.04 |

### add_explicit_parallel_bias_add_2d

| size_label | pattern_name | strategy | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4K | bias_add_2d | explicit_parallel | 4096 | 0.05 | 0.00 | 0.00 |
| 4K | bias_add_2d | explicit_parallel | 4096 | 0.09 | 0.00 | 0.00 |
| 4K | bias_add_2d | explicit_parallel | 4096 | 0.05 | 0.00 | 0.00 |
| 1M | bias_add_2d | explicit_parallel | 1048576 | 0.05 | 0.02 | 0.08 |
| 1M | bias_add_2d | explicit_parallel | 1048576 | 0.05 | 0.02 | 0.08 |
| 1M | bias_add_2d | explicit_parallel | 1048576 | 0.05 | 0.02 | 0.16 |
| 16M | bias_add_2d | explicit_parallel | 16777216 | 0.05 | 0.33 | 1.31 |
| 16M | bias_add_2d | explicit_parallel | 16777216 | 0.05 | 0.31 | 1.26 |
| 16M | bias_add_2d | explicit_parallel | 16777216 | 0.07 | 0.25 | 1.99 |

### add_explicit_parallel_interleaved_3d

| size_label | pattern_name | strategy | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4K | interleaved_3d | explicit_parallel | 4096 | 0.05 | 0.00 | 0.00 |
| 4K | interleaved_3d | explicit_parallel | 4096 | 0.05 | 0.00 | 0.00 |
| 4K | interleaved_3d | explicit_parallel | 4096 | 0.05 | 0.00 | 0.00 |
| 1M | interleaved_3d | explicit_parallel | 1048576 | 0.05 | 0.02 | 0.04 |
| 1M | interleaved_3d | explicit_parallel | 1048576 | 0.05 | 0.02 | 0.04 |
| 1M | interleaved_3d | explicit_parallel | 1048576 | 0.05 | 0.02 | 0.08 |
| 16M | interleaved_3d | explicit_parallel | 16777216 | 0.05 | 0.33 | 0.66 |
| 16M | interleaved_3d | explicit_parallel | 16777216 | 0.05 | 0.31 | 0.62 |
| 16M | interleaved_3d | explicit_parallel | 16777216 | 0.06 | 0.26 | 1.05 |

## r4_where

### tileops-where

| n_total | size_label | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 4096 | 4K | 0.05 | 0.00 | 0.00 |
| 1048576 | 1M | 0.05 | 0.02 | 0.14 |
| 16777216 | 16M | 0.06 | 0.27 | 1.88 |

### torch

| n_total | size_label | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 4096 | 4K | 0.02 | 0.00 | 0.00 |
| 1048576 | 1M | 0.02 | 0.05 | 0.35 |
| 16777216 | 16M | 0.04 | 0.46 | 3.24 |

## AddOp

### tileops

| n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 4194304 | torch.float16 | 0.05 | 0.08 | 0.48 |
| 4194304 | torch.bfloat16 | 0.05 | 0.08 | 0.49 |
| 4194304 | torch.float32 | 0.05 | 0.08 | 0.96 |

### torch

| n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 4194304 | torch.float16 | 0.02 | 0.27 | 1.64 |
| 4194304 | torch.bfloat16 | 0.02 | 0.26 | 1.58 |
| 4194304 | torch.float32 | 0.02 | 0.22 | 2.62 |

## sub

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| sub | torch.float16 | torch.float16 | 0.05 | 0.08 | 0.48 |
| sub | torch.float16 | torch.float16 | 0.05 | 0.21 | 1.23 |
| sub | torch.float16 | torch.float16 | 0.06 | 0.36 | 2.18 |

### torch

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| sub | torch.float16 | torch.float16 | 0.02 | 0.28 | 1.65 |
| sub | torch.float16 | torch.float16 | 0.02 | 0.46 | 2.79 |
| sub | torch.float16 | torch.float16 | 0.04 | 0.56 | 3.37 |

## mul

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| mul | torch.float16 | torch.float16 | 0.05 | 0.08 | 0.49 |
| mul | torch.float16 | torch.float16 | 0.05 | 0.20 | 1.20 |
| mul | torch.float16 | torch.float16 | 0.06 | 0.36 | 2.19 |

### torch

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| mul | torch.float16 | torch.float16 | 0.01 | 0.29 | 1.72 |
| mul | torch.float16 | torch.float16 | 0.02 | 0.46 | 2.78 |
| mul | torch.float16 | torch.float16 | 0.04 | 0.56 | 3.37 |

## div

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| div | torch.float16 | torch.float16 | 0.05 | 0.08 | 0.48 |
| div | torch.float16 | torch.float16 | 0.05 | 0.20 | 1.19 |
| div | torch.float16 | torch.float16 | 0.06 | 0.36 | 2.18 |

### torch

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| div | torch.float16 | torch.float16 | 0.01 | 0.28 | 1.70 |
| div | torch.float16 | torch.float16 | 0.02 | 0.45 | 2.72 |
| div | torch.float16 | torch.float16 | 0.04 | 0.56 | 3.33 |

## remainder

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| remainder | torch.float16 | torch.float16 | 0.05 | 0.08 | 0.47 |
| remainder | torch.float16 | torch.float16 | 0.05 | 0.21 | 1.25 |

### torch

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| remainder | torch.float16 | torch.float16 | 0.02 | 0.25 | 1.52 |
| remainder | torch.float16 | torch.float16 | 0.02 | 0.42 | 2.52 |

## pow

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| pow | torch.float16 | torch.float16 | 0.05 | 0.08 | 0.49 |
| pow | torch.float16 | torch.float16 | 0.06 | 0.19 | 1.12 |

### torch

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| pow | torch.float16 | torch.float16 | 0.02 | 0.20 | 1.19 |
| pow | torch.float16 | torch.float16 | 0.04 | 0.25 | 1.47 |

## floor_divide

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| floor_divide | torch.float16 | torch.float16 | 0.05 | 0.08 | 0.49 |
| floor_divide | torch.float16 | torch.float16 | 0.05 | 0.21 | 1.23 |

### torch

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| floor_divide | torch.float16 | torch.float16 | 0.03 | 0.15 | 0.93 |
| floor_divide | torch.float16 | torch.float16 | 0.06 | 0.19 | 1.14 |

## lerp

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| lerp | torch.float16 | torch.float16 | 0.05 | 0.08 | 0.50 |
| lerp | torch.float16 | torch.float16 | 0.05 | 0.20 | 1.22 |

### torch

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| lerp | torch.float16 | torch.float16 | 0.02 | 0.26 | 1.58 |
| lerp | torch.float16 | torch.float16 | 0.02 | 0.46 | 2.78 |

## maximum

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| maximum | torch.float16 | torch.float16 | 0.05 | 0.08 | 0.49 |
| maximum | torch.float16 | torch.float16 | 0.05 | 0.21 | 1.24 |
| maximum | torch.float16 | torch.float16 | 0.06 | 0.37 | 2.19 |

### torch

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| maximum | torch.float16 | torch.float16 | 0.01 | 0.29 | 1.75 |
| maximum | torch.float16 | torch.float16 | 0.02 | 0.46 | 2.73 |
| maximum | torch.float16 | torch.float16 | 0.04 | 0.56 | 3.35 |

## minimum

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| minimum | torch.float16 | torch.float16 | 0.05 | 0.08 | 0.49 |
| minimum | torch.float16 | torch.float16 | 0.05 | 0.21 | 1.26 |
| minimum | torch.float16 | torch.float16 | 0.06 | 0.37 | 2.23 |

### torch

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| minimum | torch.float16 | torch.float16 | 0.01 | 0.29 | 1.72 |
| minimum | torch.float16 | torch.float16 | 0.02 | 0.46 | 2.77 |
| minimum | torch.float16 | torch.float16 | 0.04 | 0.56 | 3.35 |

## cmp_eq

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| eq | torch.float16 | 0.07 | 0.06 | 0.30 |
| eq | torch.float16 | 0.07 | 0.15 | 0.73 |

### torch

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| eq | torch.float16 | 0.01 | 0.30 | 1.49 |
| eq | torch.float16 | 0.02 | 0.51 | 2.56 |

## cmp_ne

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| ne | torch.float16 | 0.07 | 0.06 | 0.31 |

### torch

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| ne | torch.float16 | 0.01 | 0.31 | 1.53 |

## cmp_gt

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| gt | torch.float16 | 0.07 | 0.06 | 0.31 |

### torch

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| gt | torch.float16 | 0.01 | 0.30 | 1.52 |

## cmp_lt

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| lt | torch.float16 | 0.07 | 0.06 | 0.31 |

### torch

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| lt | torch.float16 | 0.01 | 0.31 | 1.53 |

## cmp_ge

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| ge | torch.float16 | 0.07 | 0.06 | 0.31 |

### torch

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| ge | torch.float16 | 0.01 | 0.29 | 1.43 |

## cmp_le

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| le | torch.float16 | 0.07 | 0.06 | 0.31 |

### torch

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| le | torch.float16 | 0.01 | 0.30 | 1.51 |

## logical_and

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| logical_and | torch.float16 | 0.07 | 0.06 | 0.31 |
| logical_and | torch.float16 | 0.07 | 0.15 | 0.73 |

### torch

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| logical_and | torch.float16 | 0.01 | 0.28 | 1.40 |
| logical_and | torch.float16 | 0.02 | 0.59 | 2.93 |

## logical_or

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| logical_or | torch.float16 | 0.07 | 0.06 | 0.31 |
| logical_or | torch.float16 | 0.07 | 0.15 | 0.73 |

### torch

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| logical_or | torch.float16 | 0.02 | 0.28 | 1.39 |
| logical_or | torch.float16 | 0.02 | 0.60 | 2.99 |

## bitwise_and

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| bitwise_and | torch.int32 | 0.05 | 0.08 | 0.97 |
| bitwise_and | torch.int32 | 0.06 | 0.18 | 2.22 |

### torch

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| bitwise_and | torch.int32 | 0.02 | 0.21 | 2.57 |
| bitwise_and | torch.int32 | 0.04 | 0.28 | 3.37 |

## bitwise_or

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| bitwise_or | torch.int32 | 0.05 | 0.08 | 0.96 |

### torch

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| bitwise_or | torch.int32 | 0.02 | 0.22 | 2.61 |

## bitwise_xor

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| bitwise_xor | torch.int32 | 0.05 | 0.08 | 0.95 |

### torch

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| bitwise_xor | torch.int32 | 0.02 | 0.22 | 2.58 |

## gelu_and_mul

### tileops

| op_name | M | N | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| gelu_and_mul | 1024 | 4096 | torch.float16 | 0.04 | 0.21 | 0.64 |
| gelu_and_mul | 1024 | 10240 | torch.float16 | 0.04 | 0.47 | 1.40 |
| gelu_and_mul | 1024 | 20480 | torch.float16 | 0.05 | 0.86 | 2.59 |

### torch-ref

| op_name | M | N | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| gelu_and_mul | 1024 | 4096 | torch.float16 | 0.04 | 0.23 | 0.69 |
| gelu_and_mul | 1024 | 10240 | torch.float16 | 0.08 | 0.28 | 0.84 |
| gelu_and_mul | 1024 | 20480 | torch.float16 | 0.14 | 0.31 | 0.92 |

## gelu_tanh_and_mul

### tileops

| op_name | M | N | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| gelu_tanh_and_mul | 1024 | 4096 | torch.float16 | 0.04 | 0.21 | 0.62 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.float16 | 0.04 | 0.49 | 1.48 |
| gelu_tanh_and_mul | 1024 | 20480 | torch.float16 | 0.04 | 0.93 | 2.80 |

### torch-ref

| op_name | M | N | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| gelu_tanh_and_mul | 1024 | 4096 | torch.float16 | 0.04 | 0.24 | 0.71 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.float16 | 0.07 | 0.29 | 0.87 |
| gelu_tanh_and_mul | 1024 | 20480 | torch.float16 | 0.13 | 0.32 | 0.96 |

## silu_and_mul_strategy

### tileops-direct

| op_name | M | N | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| silu_and_mul | 1024 | 4096 | torch.float16 | direct | 0.04 | 0.21 | 0.62 |
| silu_and_mul | 1024 | 4096 | torch.bfloat16 | direct | 0.04 | 0.21 | 0.63 |
| silu_and_mul | 1024 | 4096 | torch.float32 | direct | 0.04 | 0.20 | 1.20 |
| silu_and_mul | 1024 | 10240 | torch.float16 | direct | 0.04 | 0.47 | 1.42 |
| silu_and_mul | 1024 | 10240 | torch.bfloat16 | direct | 0.04 | 0.47 | 1.40 |
| silu_and_mul | 1024 | 10240 | torch.float32 | direct | 0.05 | 0.43 | 2.57 |
| silu_and_mul | 4096 | 4096 | torch.float16 | direct | 0.07 | 0.51 | 1.54 |
| silu_and_mul | 4096 | 4096 | torch.bfloat16 | direct | 0.07 | 0.51 | 1.54 |
| silu_and_mul | 4096 | 4096 | torch.float32 | direct | 0.07 | 0.46 | 2.74 |

### tileops-explicit_parallel

| op_name | M | N | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| silu_and_mul | 1024 | 4096 | torch.float16 | explicit_parallel | 0.04 | 0.21 | 0.62 |
| silu_and_mul | 1024 | 4096 | torch.bfloat16 | explicit_parallel | 0.04 | 0.22 | 0.65 |
| silu_and_mul | 1024 | 4096 | torch.float32 | explicit_parallel | 0.04 | 0.21 | 1.26 |
| silu_and_mul | 1024 | 10240 | torch.float16 | explicit_parallel | 0.04 | 0.50 | 1.50 |
| silu_and_mul | 1024 | 10240 | torch.bfloat16 | explicit_parallel | 0.04 | 0.49 | 1.48 |
| silu_and_mul | 1024 | 10240 | torch.float32 | explicit_parallel | 0.04 | 0.48 | 2.86 |
| silu_and_mul | 4096 | 4096 | torch.float16 | explicit_parallel | 0.04 | 0.75 | 2.25 |
| silu_and_mul | 4096 | 4096 | torch.bfloat16 | explicit_parallel | 0.04 | 0.75 | 2.25 |
| silu_and_mul | 4096 | 4096 | torch.float32 | explicit_parallel | 0.05 | 0.61 | 3.67 |

## gelu_and_mul_strategy

### tileops-direct

| op_name | M | N | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gelu_and_mul | 1024 | 4096 | torch.float16 | direct | 0.04 | 0.21 | 0.63 |
| gelu_and_mul | 1024 | 4096 | torch.bfloat16 | direct | 0.04 | 0.21 | 0.62 |
| gelu_and_mul | 1024 | 4096 | torch.float32 | direct | 0.04 | 0.20 | 1.18 |
| gelu_and_mul | 1024 | 10240 | torch.float16 | direct | 0.04 | 0.47 | 1.42 |
| gelu_and_mul | 1024 | 10240 | torch.bfloat16 | direct | 0.04 | 0.47 | 1.40 |
| gelu_and_mul | 1024 | 10240 | torch.float32 | direct | 0.05 | 0.43 | 2.59 |
| gelu_and_mul | 4096 | 4096 | torch.float16 | direct | 0.07 | 0.50 | 1.50 |
| gelu_and_mul | 4096 | 4096 | torch.bfloat16 | direct | 0.07 | 0.50 | 1.49 |
| gelu_and_mul | 4096 | 4096 | torch.float32 | direct | 0.07 | 0.45 | 2.73 |

### tileops-explicit_parallel

| op_name | M | N | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gelu_and_mul | 1024 | 4096 | torch.float16 | explicit_parallel | 0.04 | 0.21 | 0.63 |
| gelu_and_mul | 1024 | 4096 | torch.bfloat16 | explicit_parallel | 0.04 | 0.22 | 0.65 |
| gelu_and_mul | 1024 | 4096 | torch.float32 | explicit_parallel | 0.04 | 0.21 | 1.27 |
| gelu_and_mul | 1024 | 10240 | torch.float16 | explicit_parallel | 0.04 | 0.47 | 1.40 |
| gelu_and_mul | 1024 | 10240 | torch.bfloat16 | explicit_parallel | 0.04 | 0.47 | 1.41 |
| gelu_and_mul | 1024 | 10240 | torch.float32 | explicit_parallel | 0.04 | 0.47 | 2.84 |
| gelu_and_mul | 4096 | 4096 | torch.float16 | explicit_parallel | 0.04 | 0.75 | 2.26 |
| gelu_and_mul | 4096 | 4096 | torch.bfloat16 | explicit_parallel | 0.05 | 0.73 | 2.19 |
| gelu_and_mul | 4096 | 4096 | torch.float32 | explicit_parallel | 0.06 | 0.60 | 3.62 |

## gelu_tanh_and_mul_strategy

### tileops-direct

| op_name | M | N | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gelu_tanh_and_mul | 1024 | 4096 | torch.float16 | direct | 0.04 | 0.21 | 0.62 |
| gelu_tanh_and_mul | 1024 | 4096 | torch.bfloat16 | direct | 0.04 | 0.21 | 0.62 |
| gelu_tanh_and_mul | 1024 | 4096 | torch.float32 | direct | 0.04 | 0.20 | 1.20 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.float16 | direct | 0.04 | 0.47 | 1.42 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.bfloat16 | direct | 0.05 | 0.47 | 1.40 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.float32 | direct | 0.05 | 0.43 | 2.60 |
| gelu_tanh_and_mul | 4096 | 4096 | torch.float16 | direct | 0.07 | 0.51 | 1.52 |
| gelu_tanh_and_mul | 4096 | 4096 | torch.bfloat16 | direct | 0.07 | 0.51 | 1.52 |
| gelu_tanh_and_mul | 4096 | 4096 | torch.float32 | direct | 0.07 | 0.46 | 2.76 |

### tileops-explicit_parallel

| op_name | M | N | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gelu_tanh_and_mul | 1024 | 4096 | torch.float16 | explicit_parallel | 0.04 | 0.21 | 0.63 |
| gelu_tanh_and_mul | 1024 | 4096 | torch.bfloat16 | explicit_parallel | 0.04 | 0.22 | 0.66 |
| gelu_tanh_and_mul | 1024 | 4096 | torch.float32 | explicit_parallel | 0.04 | 0.20 | 1.21 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.float16 | explicit_parallel | 0.04 | 0.50 | 1.49 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.bfloat16 | explicit_parallel | 0.04 | 0.49 | 1.48 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.float32 | explicit_parallel | 0.05 | 0.46 | 2.77 |
| gelu_tanh_and_mul | 4096 | 4096 | torch.float16 | explicit_parallel | 0.05 | 0.72 | 2.17 |
| gelu_tanh_and_mul | 4096 | 4096 | torch.bfloat16 | explicit_parallel | 0.04 | 0.75 | 2.26 |
| gelu_tanh_and_mul | 4096 | 4096 | torch.float32 | explicit_parallel | 0.05 | 0.61 | 3.68 |

## sub_bcast

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| sub | torch.float16 | 0.05 | 0.08 | 0.33 |
| sub | torch.float16 | 0.05 | 0.21 | 0.84 |
| sub | torch.float16 | 0.05 | 0.38 | 1.53 |

### torch

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| sub | torch.float16 | 0.02 | 0.22 | 0.89 |
| sub | torch.float16 | 0.04 | 0.29 | 1.15 |
| sub | torch.float16 | 0.06 | 0.33 | 1.34 |

## mul_bcast

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| mul | torch.float16 | 0.05 | 0.08 | 0.32 |
| mul | torch.float16 | 0.05 | 0.20 | 0.79 |
| mul | torch.float16 | 0.05 | 0.39 | 1.57 |

### torch

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| mul | torch.float16 | 0.02 | 0.23 | 0.91 |
| mul | torch.float16 | 0.04 | 0.29 | 1.16 |
| mul | torch.float16 | 0.06 | 0.33 | 1.34 |

## div_bcast

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| div | torch.float16 | 0.05 | 0.08 | 0.32 |
| div | torch.float16 | 0.05 | 0.20 | 0.80 |
| div | torch.float16 | 0.06 | 0.38 | 1.52 |

### torch

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| div | torch.float16 | 0.02 | 0.21 | 0.85 |
| div | torch.float16 | 0.04 | 0.27 | 1.09 |
| div | torch.float16 | 0.07 | 0.31 | 1.25 |

## binary_strategy

### add_direct

| n_total | shape_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4194304 | 1024x4096 | torch.float16 | direct | 0.05 | 0.08 | 0.49 |
| 4194304 | 1024x4096 | torch.bfloat16 | direct | 0.05 | 0.08 | 0.51 |
| 4194304 | 1024x4096 | torch.float32 | direct | 0.05 | 0.08 | 0.96 |
| 10485760 | 1024x10240 | torch.float16 | direct | 0.06 | 0.18 | 1.08 |
| 10485760 | 1024x10240 | torch.bfloat16 | direct | 0.06 | 0.18 | 1.11 |
| 10485760 | 1024x10240 | torch.float32 | direct | 0.06 | 0.18 | 2.18 |
| 20971520 | 1024x20480 | torch.float16 | direct | 0.08 | 0.27 | 1.65 |
| 20971520 | 1024x20480 | torch.bfloat16 | direct | 0.08 | 0.28 | 1.66 |
| 20971520 | 1024x20480 | torch.float32 | direct | 0.09 | 0.24 | 2.89 |

### add_explicit_parallel

| n_total | shape_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4194304 | 1024x4096 | torch.float16 | explicit_parallel | 0.05 | 0.08 | 0.48 |
| 4194304 | 1024x4096 | torch.bfloat16 | explicit_parallel | 0.05 | 0.08 | 0.51 |
| 4194304 | 1024x4096 | torch.float32 | explicit_parallel | 0.05 | 0.08 | 0.99 |
| 10485760 | 1024x10240 | torch.float16 | explicit_parallel | 0.05 | 0.21 | 1.24 |
| 10485760 | 1024x10240 | torch.bfloat16 | explicit_parallel | 0.05 | 0.21 | 1.24 |
| 10485760 | 1024x10240 | torch.float32 | explicit_parallel | 0.06 | 0.18 | 2.20 |
| 20971520 | 1024x20480 | torch.float16 | explicit_parallel | 0.06 | 0.36 | 2.18 |
| 20971520 | 1024x20480 | torch.bfloat16 | explicit_parallel | 0.06 | 0.37 | 2.24 |
| 20971520 | 1024x20480 | torch.float32 | explicit_parallel | 0.07 | 0.32 | 3.79 |

## conv1d

### tileops

| n | c_in | l_in | c_out | kernel_size | stride | padding | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 256 | 32000 | 512 | 1 | 1 | 0 | torch.float16 | 0.21 | 160.56 | 0.94 |
| 4 | 128 | 4096 | 256 | 3 | 1 | 1 | torch.float16 | 0.08 | 40.83 | 0.16 |
| 4 | 64 | 16000 | 128 | 5 | 2 | 2 | torch.float16 | 0.08 | 33.62 | 0.21 |
| 4 | 128 | 8192 | 256 | 7 | 1 | 3 | torch.float16 | 0.08 | 179.74 | 0.31 |
| 2 | 128 | 4096 | 256 | 3 | 2 | 1 | torch.bfloat16 | 0.08 | 10.59 | 0.06 |

### torch

| n | c_in | l_in | c_out | kernel_size | stride | padding | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 256 | 32000 | 512 | 1 | 1 | 0 | torch.float16 | 0.28 | 117.75 | 0.69 |
| 4 | 128 | 4096 | 256 | 3 | 1 | 1 | torch.float16 | 0.06 | 53.06 | 0.21 |
| 4 | 64 | 16000 | 128 | 5 | 2 | 2 | torch.float16 | 0.06 | 46.03 | 0.29 |
| 4 | 128 | 8192 | 256 | 7 | 1 | 3 | torch.float16 | 0.09 | 163.21 | 0.28 |
| 2 | 128 | 4096 | 256 | 3 | 2 | 1 | torch.bfloat16 | 0.05 | 14.65 | 0.08 |

## conv2d

### tileops

| n | c_in | h | w | c_out | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 64 | 56 | 56 | 64 | torch.float16 | 0.08 | 5.82 | 0.02 |
| 1 | 3 | 112 | 112 | 64 | torch.float16 | 0.08 | 0.14 | 0.01 |
| 1 | 128 | 56 | 56 | 256 | torch.float16 | 0.08 | 5.77 | 0.02 |
| 1 | 256 | 112 | 112 | 512 | torch.float16 | 0.09 | 337.17 | 0.25 |
| 1 | 64 | 56 | 56 | 128 | torch.float16 | 0.08 | 16.15 | 0.02 |
| 1 | 128 | 56 | 56 | 256 | torch.float16 | 0.08 | 16.04 | 0.04 |
| 1 | 128 | 28 | 28 | 128 | torch.bfloat16 | 0.08 | 0.71 | 0.01 |
| 2 | 64 | 56 | 56 | 256 | torch.float16 | 0.06 | 3.43 | 0.07 |
| 2 | 128 | 28 | 28 | 512 | torch.float16 | 0.06 | 3.47 | 0.04 |
| 2 | 512 | 28 | 28 | 128 | torch.float16 | 0.06 | 3.40 | 0.04 |
| 1 | 256 | 14 | 14 | 1024 | torch.float16 | 0.06 | 1.68 | 0.02 |
| 1 | 512 | 7 | 7 | 2048 | torch.float16 | 0.06 | 1.72 | 0.04 |
| 2 | 64 | 56 | 56 | 256 | torch.bfloat16 | 0.06 | 3.41 | 0.07 |

### torch-nchw

| n | c_in | h | w | c_out | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 64 | 56 | 56 | 64 | torch.float16 | 0.05 | 9.17 | 0.03 |
| 1 | 3 | 112 | 112 | 64 | torch.float16 | 0.03 | 0.36 | 0.02 |
| 1 | 128 | 56 | 56 | 256 | torch.float16 | 0.05 | 8.67 | 0.03 |
| 1 | 256 | 112 | 112 | 512 | torch.float16 | 0.11 | 271.39 | 0.20 |
| 1 | 64 | 56 | 56 | 128 | torch.float16 | 0.05 | 24.24 | 0.03 |
| 1 | 128 | 56 | 56 | 256 | torch.float16 | 0.06 | 20.92 | 0.05 |
| 1 | 128 | 28 | 28 | 128 | torch.bfloat16 | 0.05 | 1.08 | 0.01 |
| 2 | 64 | 56 | 56 | 256 | torch.float16 | 0.04 | 5.66 | 0.11 |
| 2 | 128 | 28 | 28 | 512 | torch.float16 | 0.04 | 5.44 | 0.06 |
| 2 | 512 | 28 | 28 | 128 | torch.float16 | 0.04 | 5.50 | 0.06 |
| 1 | 256 | 14 | 14 | 1024 | torch.float16 | 0.03 | 3.06 | 0.03 |
| 1 | 512 | 7 | 7 | 2048 | torch.float16 | 0.03 | 2.96 | 0.07 |
| 2 | 64 | 56 | 56 | 256 | torch.bfloat16 | 0.04 | 5.50 | 0.11 |

### torch-nhwc

| n | c_in | h | w | c_out | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 64 | 56 | 56 | 64 | torch.float16 | 0.04 | 10.52 | 0.04 |
| 1 | 3 | 112 | 112 | 64 | torch.float16 | 0.04 | 0.25 | 0.01 |
| 1 | 128 | 56 | 56 | 256 | torch.float16 | 0.05 | 9.11 | 0.04 |
| 1 | 256 | 112 | 112 | 512 | torch.float16 | 0.09 | 341.63 | 0.25 |
| 1 | 64 | 56 | 56 | 128 | torch.float16 | 0.05 | 25.59 | 0.03 |
| 1 | 128 | 56 | 56 | 256 | torch.float16 | 0.06 | 22.64 | 0.05 |
| 1 | 128 | 28 | 28 | 128 | torch.bfloat16 | 0.05 | 1.14 | 0.01 |
| 2 | 64 | 56 | 56 | 256 | torch.float16 | 0.04 | 5.76 | 0.11 |
| 2 | 128 | 28 | 28 | 512 | torch.float16 | 0.04 | 5.70 | 0.06 |
| 2 | 512 | 28 | 28 | 128 | torch.float16 | 0.04 | 5.65 | 0.06 |
| 1 | 256 | 14 | 14 | 1024 | torch.float16 | 0.04 | 2.87 | 0.03 |
| 1 | 512 | 7 | 7 | 2048 | torch.float16 | 0.04 | 2.92 | 0.07 |
| 2 | 64 | 56 | 56 | 256 | torch.bfloat16 | 0.04 | 5.68 | 0.11 |

## conv3d

### tileops

| n | c_in | d_in | h_in | w_in | c_out | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3 | 16 | 112 | 112 | 64 | torch.float16 | 0.09 | 23.99 | 0.31 |
| 1 | 64 | 8 | 56 | 56 | 128 | torch.float16 | 0.08 | 17.67 | 0.06 |
| 1 | 32 | 32 | 64 | 64 | 64 | torch.bfloat16 | 0.09 | 166.22 | 0.29 |

### torch-ncdhw

| n | c_in | d_in | h_in | w_in | c_out | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3 | 16 | 112 | 112 | 64 | torch.float16 | 0.14 | 14.49 | 0.19 |
| 1 | 64 | 8 | 56 | 56 | 128 | torch.float16 | 0.06 | 23.39 | 0.08 |
| 1 | 32 | 32 | 64 | 64 | 64 | torch.bfloat16 | 0.14 | 100.76 | 0.18 |

### torch-ndhwc

| n | c_in | d_in | h_in | w_in | c_out | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3 | 16 | 112 | 112 | 64 | torch.float16 | 0.11 | 19.68 | 0.25 |
| 1 | 64 | 8 | 56 | 56 | 128 | torch.float16 | 0.06 | 24.24 | 0.08 |
| 1 | 32 | 32 | 64 | 64 | 64 | torch.bfloat16 | 0.12 | 122.68 | 0.21 |

## CumsumOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | cumsum | 0.07 | 0.06 | 0.25 |
| 1024 | 4096 | torch.bfloat16 | cumsum | 0.07 | 0.06 | 0.25 |
| 4096 | 4096 | torch.float16 | cumsum | 0.13 | 0.13 | 0.53 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | cumsum | 0.10 | 0.04 | 0.16 |
| 1024 | 4096 | torch.bfloat16 | cumsum | 0.11 | 0.04 | 0.16 |
| 4096 | 4096 | torch.float16 | cumsum | 0.26 | 0.06 | 0.25 |

## CumprodOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | cumprod | 0.07 | 0.06 | 0.25 |
| 1024 | 4096 | torch.bfloat16 | cumprod | 0.07 | 0.06 | 0.25 |
| 4096 | 4096 | torch.float16 | cumprod | 0.13 | 0.13 | 0.53 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | cumprod | 0.11 | 0.04 | 0.15 |
| 1024 | 4096 | torch.bfloat16 | cumprod | 0.11 | 0.04 | 0.16 |
| 4096 | 4096 | torch.float16 | cumprod | 0.27 | 0.06 | 0.25 |

## DeepSeekSparseAttentionDecodeWithKVCacheOp

### tileops

| batch | heads | seq_len_q | seq_len_kv | dim | dim_tail | topk | stride_kv | heads_kv | q_start_index_s | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 128 | 1024 | 2048 | 512 | 64 | 2048 | 1 | 1 | 1024 | torch.float16 | 1.16 | 503.68 | 0.26 |
| 1 | 128 | 512 | 4096 | 512 | 64 | 1024 | 1 | 1 | 512 | torch.float16 | 0.32 | 460.18 | 0.47 |

### torch-ref

| batch | heads | seq_len_q | seq_len_kv | dim | dim_tail | topk | stride_kv | heads_kv | q_start_index_s | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 128 | 1024 | 2048 | 512 | 64 | 2048 | 1 | 1 | 1024 | torch.float16 | 16.55 | 35.30 | 0.02 |
| 1 | 128 | 512 | 4096 | 512 | 64 | 1024 | 1 | 1 | 512 | torch.float16 | 16.76 | 8.71 | 0.01 |

## MultiHeadLatentAttentionDecodeWithKVCacheOp

### tileops

| batch | heads | heads_kv | seq_len_kv | dim | dim_pe | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 128 | 1 | 8192 | 512 | 64 | torch.float16 | 0.17 | 438.39 | 1.84 |
| 16 | 128 | 1 | 4096 | 512 | 64 | torch.float16 | 0.08 | 229.15 | 0.98 |

### torch-ref

| batch | heads | heads_kv | seq_len_kv | dim | dim_pe | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 128 | 1 | 8192 | 512 | 64 | torch.float16 | 0.54 | 135.79 | 0.57 |
| 16 | 128 | 1 | 4096 | 512 | 64 | torch.float16 | 0.17 | 104.98 | 0.45 |

## DeltaNetFwdOp

### tileops

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 0.14 | 1.93 | 0.12 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 0.17 | 1.60 | 0.10 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 0.29 | 0.92 | 0.06 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 0.14 | 0.94 | 0.06 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 0.23 | 2.36 | 0.15 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 0.44 | 2.45 | 0.15 |
| 2 | 32768 | 4 | 64 | 64 | 64 | torch.float16 | 0.85 | 2.54 | 0.16 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.14 | 0.98 | 0.06 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.14 | 1.89 | 0.12 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.23 | 2.36 | 0.15 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.44 | 2.44 | 0.15 |
| 2 | 32768 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.85 | 2.53 | 0.16 |

### fla

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.34 | 0.80 | 0.05 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 0.125 | 0.34 | 0.78 | 0.05 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 0.125 | 0.71 | 0.38 | 0.02 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.36 | 0.37 | 0.02 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.36 | 1.49 | 0.09 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.36 | 2.96 | 0.19 |
| 2 | 32768 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.64 | 3.34 | 0.21 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.35 | 0.38 | 0.02 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.34 | 0.78 | 0.05 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.35 | 1.55 | 0.10 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.36 | 3.01 | 0.19 |
| 2 | 32768 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.65 | 3.31 | 0.21 |

## DeltaNetBwdOp

### tileops

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | H | S | DK | DV | BC | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.24 | 2.23 | 0.12 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.27 | 1.97 | 0.11 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.27 | 1.96 | 0.11 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.15 | 1.74 | 0.10 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.45 | 2.38 | 0.13 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 16384 | 64 | 64 | 64 | 0.87 | 2.47 | 0.14 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.15 | 1.73 | 0.10 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.24 | 2.25 | 0.12 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.45 | 2.39 | 0.13 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 16384 | 64 | 64 | 64 | 0.88 | 2.45 | 0.14 |

### fla

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | H | S | DK | DV | BC | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.125 | 0.18 | 3.03 | 0.17 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.125 | 0.18 | 3.01 | 0.17 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.125 | 0.18 | 3.00 | 0.17 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.125 | 0.09 | 2.85 | 0.16 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.125 | 0.36 | 3.02 | 0.17 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 16384 | 64 | 64 | 64 | 0.125 | 1.15 | 1.87 | 0.10 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.125 | 0.09 | 2.83 | 0.16 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.125 | 0.18 | 3.02 | 0.17 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.125 | 0.36 | 2.99 | 0.16 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 16384 | 64 | 64 | 64 | 0.125 | 1.02 | 2.11 | 0.12 |

## DeltaNetOp

### tileops

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | H | S | DK | DV | BC | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.32 | 2.48 | 0.14 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.51 | 1.59 | 0.09 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.48 | 1.66 | 0.10 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.17 | 2.42 | 0.14 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 8192 | 64 | 64 | 64 | 1.07 | 1.51 | 0.09 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 16384 | 64 | 64 | 64 | 2.55 | 1.26 | 0.07 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.17 | 2.43 | 0.14 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.32 | 2.50 | 0.14 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 8192 | 64 | 64 | 64 | 1.06 | 1.52 | 0.09 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 16384 | 64 | 64 | 64 | 2.62 | 1.23 | 0.07 |

### fla

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | H | S | DK | DV | BC | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.125 | 0.26 | 3.11 | 0.18 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.125 | 0.26 | 3.12 | 0.18 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.125 | 0.26 | 3.10 | 0.18 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.125 | 0.15 | 2.75 | 0.16 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.125 | 0.52 | 3.12 | 0.18 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 16384 | 64 | 64 | 64 | 0.125 | 1.18 | 2.72 | 0.16 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.125 | 0.15 | 2.75 | 0.16 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.125 | 0.26 | 3.09 | 0.18 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.125 | 0.52 | 3.10 | 0.18 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 16384 | 64 | 64 | 64 | 0.125 | 1.20 | 2.69 | 0.15 |

## DeltaNetDecodeOp

### tileops

| batch | heads | dim_k | dim_v | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 128 | 128 | torch.float32 | 0.03 | 0.12 | 0.16 |
| 1 | 32 | 128 | 128 | torch.bfloat16 | 0.03 | 0.12 | 0.08 |
| 8 | 32 | 128 | 128 | torch.float32 | 0.03 | 0.84 | 1.13 |
| 8 | 32 | 128 | 128 | torch.bfloat16 | 0.03 | 0.91 | 0.62 |
| 16 | 32 | 128 | 128 | torch.float32 | 0.04 | 1.14 | 1.54 |
| 16 | 32 | 128 | 128 | torch.bfloat16 | 0.03 | 1.67 | 1.13 |
| 32 | 32 | 128 | 128 | torch.float32 | 0.08 | 1.24 | 1.67 |
| 32 | 32 | 128 | 128 | torch.bfloat16 | 0.03 | 3.01 | 2.03 |
| 1 | 32 | 64 | 64 | torch.float32 | 0.03 | 0.03 | 0.04 |
| 8 | 32 | 64 | 64 | torch.float32 | 0.03 | 0.24 | 0.33 |
| 32 | 32 | 64 | 64 | torch.float32 | 0.04 | 0.64 | 0.88 |
| 64 | 32 | 128 | 128 | torch.float32 | 0.15 | 1.30 | 1.76 |
| 64 | 32 | 128 | 128 | torch.bfloat16 | 0.06 | 3.42 | 2.31 |

### torch

| batch | heads | dim_k | dim_v | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 128 | 128 | torch.float32 | 0.16 | 0.02 | 0.03 |
| 1 | 32 | 128 | 128 | torch.bfloat16 | 0.24 | 0.01 | 0.01 |
| 8 | 32 | 128 | 128 | torch.float32 | 0.16 | 0.15 | 0.21 |
| 8 | 32 | 128 | 128 | torch.bfloat16 | 0.24 | 0.11 | 0.07 |
| 16 | 32 | 128 | 128 | torch.float32 | 0.17 | 0.30 | 0.40 |
| 16 | 32 | 128 | 128 | torch.bfloat16 | 0.25 | 0.20 | 0.14 |
| 32 | 32 | 128 | 128 | torch.float32 | 0.17 | 0.59 | 0.80 |
| 32 | 32 | 128 | 128 | torch.bfloat16 | 0.27 | 0.38 | 0.25 |
| 1 | 32 | 64 | 64 | torch.float32 | 0.16 | 0.00 | 0.01 |
| 8 | 32 | 64 | 64 | torch.float32 | 0.16 | 0.04 | 0.05 |
| 32 | 32 | 64 | 64 | torch.float32 | 0.17 | 0.15 | 0.21 |
| 64 | 32 | 128 | 128 | torch.float32 | 0.30 | 0.68 | 0.92 |
| 64 | 32 | 128 | 128 | torch.bfloat16 | 0.47 | 0.43 | 0.29 |

## DropoutOp

### tileops

| dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| torch.float16 | 4194304 | 0.05 | 0.08 | 0.32 |
| torch.bfloat16 | 4194304 | 0.05 | 0.08 | 0.32 |
| torch.float32 | 4194304 | 0.05 | 0.08 | 0.63 |
| torch.float16 | 10485760 | 0.05 | 0.20 | 0.81 |
| torch.bfloat16 | 10485760 | 0.05 | 0.20 | 0.80 |
| torch.float32 | 10485760 | 0.06 | 0.19 | 1.51 |
| torch.float16 | 20971520 | 0.05 | 0.39 | 1.57 |
| torch.bfloat16 | 20971520 | 0.05 | 0.39 | 1.57 |
| torch.float32 | 20971520 | 0.06 | 0.35 | 2.80 |

### torch

| dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| torch.float16 | 4194304 | 0.03 | 0.17 | 0.67 |
| torch.bfloat16 | 4194304 | 0.03 | 0.17 | 0.67 |
| torch.float32 | 4194304 | 0.03 | 0.15 | 1.19 |
| torch.float16 | 10485760 | 0.03 | 0.38 | 1.51 |
| torch.bfloat16 | 10485760 | 0.03 | 0.38 | 1.51 |
| torch.float32 | 10485760 | 0.04 | 0.28 | 2.21 |
| torch.float16 | 20971520 | 0.04 | 0.48 | 1.93 |
| torch.bfloat16 | 20971520 | 0.05 | 0.46 | 1.86 |
| torch.float32 | 20971520 | 0.07 | 0.32 | 2.56 |

## relu_fp8

### tileops

| op_name | n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| relu_fp8 | 8388608 | torch.float8_e4m3fn | 0.05 | 0.18 | 0.35 |
| relu_fp8 | 8388608 | torch.float8_e5m2 | 0.06 | 0.13 | 0.26 |
| relu_fp8 | 67108864 | torch.float8_e4m3fn | 0.10 | 0.66 | 1.33 |
| relu_fp8 | 67108864 | torch.float8_e5m2 | 0.13 | 0.51 | 1.01 |
| relu_fp8 | 134217728 | torch.float8_e4m3fn | 0.19 | 0.69 | 1.38 |
| relu_fp8 | 134217728 | torch.float8_e5m2 | 0.25 | 0.53 | 1.07 |

### torch

| op_name | n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| relu_fp8 | 8388608 | torch.float8_e4m3fn | 0.05 | 0.16 | 0.31 |
| relu_fp8 | 8388608 | torch.float8_e5m2 | 0.05 | 0.17 | 0.34 |
| relu_fp8 | 67108864 | torch.float8_e4m3fn | 0.36 | 0.19 | 0.38 |
| relu_fp8 | 67108864 | torch.float8_e5m2 | 0.35 | 0.19 | 0.39 |
| relu_fp8 | 134217728 | torch.float8_e4m3fn | 0.69 | 0.19 | 0.39 |
| relu_fp8 | 134217728 | torch.float8_e5m2 | 0.67 | 0.20 | 0.40 |

## exp_fp8

### tileops

| op_name | n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| exp_fp8 | 8388608 | torch.float8_e4m3fn | 0.05 | 0.18 | 0.36 |
| exp_fp8 | 8388608 | torch.float8_e5m2 | 0.06 | 0.13 | 0.26 |
| exp_fp8 | 67108864 | torch.float8_e4m3fn | 0.06 | 1.20 | 2.41 |
| exp_fp8 | 67108864 | torch.float8_e5m2 | 0.13 | 0.50 | 1.00 |
| exp_fp8 | 134217728 | torch.float8_e4m3fn | 0.10 | 1.29 | 2.58 |
| exp_fp8 | 134217728 | torch.float8_e5m2 | 0.25 | 0.53 | 1.06 |

### torch

| op_name | n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| exp_fp8 | 8388608 | torch.float8_e4m3fn | 0.05 | 0.15 | 0.31 |
| exp_fp8 | 8388608 | torch.float8_e5m2 | 0.05 | 0.16 | 0.32 |
| exp_fp8 | 67108864 | torch.float8_e4m3fn | 0.35 | 0.19 | 0.38 |
| exp_fp8 | 67108864 | torch.float8_e5m2 | 0.34 | 0.20 | 0.40 |
| exp_fp8 | 134217728 | torch.float8_e4m3fn | 0.69 | 0.20 | 0.39 |
| exp_fp8 | 134217728 | torch.float8_e5m2 | 0.66 | 0.20 | 0.40 |

## add_fp8

### tileops

| op_name | n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| add_fp8 | 8388608 | torch.float8_e4m3fn | 0.05 | 0.16 | 0.47 |
| add_fp8 | 8388608 | torch.float8_e5m2 | 0.07 | 0.12 | 0.36 |
| add_fp8 | 67108864 | torch.float8_e4m3fn | 0.06 | 1.05 | 3.15 |
| add_fp8 | 67108864 | torch.float8_e5m2 | 0.14 | 0.47 | 1.41 |
| add_fp8 | 134217728 | torch.float8_e4m3fn | 0.12 | 1.12 | 3.37 |
| add_fp8 | 134217728 | torch.float8_e5m2 | 0.27 | 0.49 | 1.48 |

### torch

| op_name | n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| add_fp8 | 8388608 | torch.float8_e4m3fn | 0.09 | 0.10 | 0.29 |
| add_fp8 | 8388608 | torch.float8_e5m2 | 0.08 | 0.10 | 0.30 |
| add_fp8 | 67108864 | torch.float8_e4m3fn | 0.59 | 0.11 | 0.34 |
| add_fp8 | 67108864 | torch.float8_e5m2 | 0.57 | 0.12 | 0.35 |
| add_fp8 | 134217728 | torch.float8_e4m3fn | 1.15 | 0.12 | 0.35 |
| add_fp8 | 134217728 | torch.float8_e5m2 | 1.11 | 0.12 | 0.36 |

## silu_and_mul_fp8

### tileops

| op_name | M | N | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| silu_and_mul_fp8 | 2048 | 11008 | torch.float8_e4m3fn | 0.05 | 2.30 | 1.38 |
| silu_and_mul_fp8 | 2048 | 11008 | torch.float8_e5m2 | 0.07 | 1.59 | 0.95 |
| silu_and_mul_fp8 | 16384 | 11008 | torch.float8_e4m3fn | 0.33 | 2.72 | 1.63 |
| silu_and_mul_fp8 | 16384 | 11008 | torch.float8_e5m2 | 0.48 | 1.89 | 1.13 |
| silu_and_mul_fp8 | 16384 | 28672 | torch.float8_e4m3fn | 0.87 | 2.69 | 1.62 |
| silu_and_mul_fp8 | 16384 | 28672 | torch.float8_e5m2 | 1.23 | 1.90 | 1.14 |

### torch-ref

| op_name | M | N | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| silu_and_mul_fp8 | 2048 | 11008 | torch.float8_e4m3fn | 0.31 | 0.36 | 0.22 |
| silu_and_mul_fp8 | 2048 | 11008 | torch.float8_e5m2 | 0.30 | 0.38 | 0.23 |
| silu_and_mul_fp8 | 16384 | 11008 | torch.float8_e4m3fn | 2.30 | 0.39 | 0.23 |
| silu_and_mul_fp8 | 16384 | 11008 | torch.float8_e5m2 | 2.24 | 0.40 | 0.24 |
| silu_and_mul_fp8 | 16384 | 28672 | torch.float8_e4m3fn | 5.96 | 0.39 | 0.24 |
| silu_and_mul_fp8 | 16384 | 28672 | torch.float8_e5m2 | 5.79 | 0.41 | 0.24 |

## EngramGateConvBwdOp

### tileops

| M | seq_len | d | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 256 | torch.float16 | 0.11 | 0.00 | 0.00 |
| 2 | 64 | 512 | torch.float16 | 0.11 | 0.04 | 0.01 |
| 1 | 128 | 256 | torch.bfloat16 | 0.11 | 0.02 | 0.01 |
| 2 | 16 | 256 | torch.bfloat16 | 0.10 | 0.00 | 0.00 |

### torch

| M | seq_len | d | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 256 | torch.float16 | 1.67 | 0.00 | 0.00 |
| 2 | 64 | 512 | torch.float16 | 1.75 | 0.00 | 0.00 |
| 1 | 128 | 256 | torch.bfloat16 | 1.58 | 0.00 | 0.00 |
| 2 | 16 | 256 | torch.bfloat16 | 1.54 | 0.00 | 0.00 |

## EngramDecodeOp

### tileops

| batch | d_mem | d | max_conv_len | conv_kernel_size | dilation | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 512 | 256 | 12 | 4 | 3 | torch.float16 | 0.07 | 0.01 | 0.01 |
| 4 | 1024 | 512 | 20 | 4 | 5 | torch.float16 | 0.08 | 0.11 | 0.03 |
| 8 | 512 | 256 | 18 | 4 | 3 | torch.bfloat16 | 0.07 | 0.06 | 0.01 |

### torch-ref

| batch | d_mem | d | max_conv_len | conv_kernel_size | dilation | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 512 | 256 | 12 | 4 | 3 | torch.float16 | 0.68 | 0.00 | 0.00 |
| 4 | 1024 | 512 | 20 | 4 | 5 | torch.float16 | 0.70 | 0.01 | 0.00 |
| 8 | 512 | 256 | 18 | 4 | 3 | torch.bfloat16 | 0.70 | 0.01 | 0.00 |

## EngramGateConvFwdOp

### tileops

| M | seq_len | d | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 256 | torch.float16 | 0.09 | 0.00 | 0.00 |
| 2 | 64 | 512 | torch.float16 | 0.10 | 0.02 | 0.01 |
| 1 | 128 | 256 | torch.bfloat16 | 0.09 | 0.01 | 0.00 |
| 2 | 16 | 256 | torch.bfloat16 | 0.10 | 0.00 | 0.00 |

### torch-ref

| M | seq_len | d | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 256 | torch.float16 | 0.47 | 0.00 | 0.00 |
| 2 | 64 | 512 | torch.float16 | 0.57 | 0.00 | 0.00 |
| 1 | 128 | 256 | torch.bfloat16 | 0.55 | 0.00 | 0.00 |
| 2 | 16 | 256 | torch.bfloat16 | 0.64 | 0.00 | 0.00 |

## FFTC2COp

### tileops

| n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 64 | torch.complex64 | 0.15 | 0.00 | 0.00 |
| 128 | torch.complex64 | 0.16 | 0.00 | 0.00 |
| 256 | torch.complex64 | 0.18 | 0.00 | 0.00 |
| 512 | torch.complex64 | 0.15 | 0.00 | 0.00 |
| 1024 | torch.complex64 | 0.15 | 0.00 | 0.00 |

### torch-cufft

| n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 64 | torch.complex64 | 0.02 | 0.00 | 0.00 |
| 128 | torch.complex64 | 0.02 | 0.00 | 0.00 |
| 256 | torch.complex64 | 0.02 | 0.00 | 0.00 |
| 512 | torch.complex64 | 0.02 | 0.00 | 0.00 |
| 1024 | torch.complex64 | 0.02 | 0.00 | 0.00 |

### tileops-base

| n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 64 | torch.complex64 | 0.13 | 0.00 | 0.00 |
| 128 | torch.complex64 | 0.13 | 0.00 | 0.00 |
| 256 | torch.complex64 | 0.14 | 0.00 | 0.00 |
| 512 | torch.complex64 | 0.14 | 0.00 | 0.00 |
| 1024 | torch.complex64 | 0.15 | 0.00 | 0.00 |
| 4096 | torch.complex64 | 0.15 | 0.00 | 0.00 |
| 16384 | torch.complex64 | 0.16 | 0.01 | 0.00 |

## FFTC2CLUTOp

### tileops-lut

| n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 64 | torch.complex64 | 0.11 | 0.00 | 0.00 |
| 128 | torch.complex64 | 0.11 | 0.00 | 0.00 |
| 256 | torch.complex64 | 0.11 | 0.00 | 0.00 |
| 512 | torch.complex64 | 0.11 | 0.00 | 0.00 |
| 1024 | torch.complex64 | 0.11 | 0.00 | 0.00 |
| 4096 | torch.complex64 | 0.12 | 0.00 | 0.00 |
| 16384 | torch.complex64 | 0.12 | 0.01 | 0.00 |

### torch-cufft

| n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 64 | torch.complex64 | 0.02 | 0.00 | 0.00 |
| 128 | torch.complex64 | 0.02 | 0.00 | 0.00 |
| 256 | torch.complex64 | 0.02 | 0.00 | 0.00 |
| 512 | torch.complex64 | 0.02 | 0.00 | 0.00 |
| 1024 | torch.complex64 | 0.02 | 0.00 | 0.00 |
| 4096 | torch.complex64 | 0.02 | 0.01 | 0.00 |
| 16384 | torch.complex64 | 0.02 | 0.05 | 0.01 |

## Fp8LightingIndexerOp

### tileops

| batch | seq_len | heads | index_dim | seq_len_kv | kv_group | clean_logits | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 4096 | 32 | 64 | 8192 | 1 | True | 0.23 | N/A | 0.61 |
| 1 | 2048 | 16 | 64 | 4096 | 1 | True | 0.27 | N/A | 0.13 |

### torch-ref

| batch | seq_len | heads | index_dim | seq_len_kv | kv_group | clean_logits | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 4096 | 32 | 64 | 8192 | 1 | True | 11.92 | N/A | 0.01 |
| 1 | 2048 | 16 | 64 | 4096 | 1 | True | 1.60 | N/A | 0.02 |

## Fp8QuantOp

### tileops

| batch | seq_len_kv | kv_group | index_dim | in_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8192 | 1 | 64 | torch.float16 | 0.06 | 0.05 | 0.02 |
| 1 | 8192 | 1 | 64 | torch.bfloat16 | 0.06 | 0.05 | 0.02 |
| 1 | 4096 | 1 | 128 | torch.float32 | 0.06 | 0.05 | 0.04 |
| 1 | 16384 | 1 | 32 | torch.float32 | 0.06 | 0.05 | 0.04 |

### torch-ref

| batch | seq_len_kv | kv_group | index_dim | in_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8192 | 1 | 64 | torch.float16 | 0.09 | 0.04 | 0.01 |
| 1 | 8192 | 1 | 64 | torch.bfloat16 | 0.10 | 0.03 | 0.01 |
| 1 | 4096 | 1 | 128 | torch.float32 | 0.11 | 0.03 | 0.02 |
| 1 | 16384 | 1 | 32 | torch.float32 | 0.11 | 0.03 | 0.02 |

## FusedAddLayerNormOp

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.08 | 0.32 | 0.42 |
| 4096 | 4096 | torch.bfloat16 | 0.09 | 1.16 | 1.55 |
| 1024 | 3000 | torch.float16 | 0.19 | 0.10 | 0.13 |
| 1025 | 4096 | torch.float16 | 0.08 | 0.31 | 0.41 |

### torch-ref

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.07 | 0.37 | 0.50 |
| 4096 | 4096 | torch.bfloat16 | 0.22 | 0.45 | 0.60 |
| 1024 | 3000 | torch.float16 | 0.06 | 0.28 | 0.38 |
| 1025 | 4096 | torch.float16 | 0.07 | 0.37 | 0.49 |

## FusedAddRmsNormOp

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.08 | 0.27 | 0.42 |
| 4096 | 4096 | torch.bfloat16 | 0.09 | 0.97 | 1.55 |
| 2048 | 5120 | torch.float16 | 0.09 | 0.62 | 0.98 |
| 1025 | 4096 | torch.float16 | 0.08 | 0.27 | 0.43 |

### torch-ref

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.16 | 0.13 | 0.22 |
| 4096 | 4096 | torch.bfloat16 | 0.51 | 0.16 | 0.26 |
| 2048 | 5120 | torch.float16 | 0.35 | 0.15 | 0.24 |
| 1025 | 4096 | torch.float16 | 0.15 | 0.14 | 0.22 |

## GatedDeltaNetFwdOp

### tileops

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 0.21 | 1.29 | 0.08 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 0.21 | 1.25 | 0.08 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 0.19 | 0.71 | 0.04 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 0.19 | 1.40 | 0.09 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 0.31 | 1.76 | 0.11 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 0.57 | 1.88 | 0.12 |
| 2 | 32768 | 4 | 64 | 64 | 64 | torch.float16 | 1.09 | 1.98 | 0.12 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.20 | 0.69 | 0.04 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.21 | 1.30 | 0.08 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.30 | 1.77 | 0.11 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.56 | 1.91 | 0.12 |
| 2 | 32768 | 4 | 64 | 64 | 64 | torch.bfloat16 | 1.08 | 1.99 | 0.13 |

### fla

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 0.125 | 0.39 | 0.68 | 0.04 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 0.125 | 0.39 | 0.69 | 0.04 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.39 | 0.34 | 0.02 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.39 | 0.69 | 0.04 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.39 | 1.36 | 0.09 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.40 | 2.69 | 0.17 |
| 2 | 32768 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.73 | 2.93 | 0.18 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.39 | 0.35 | 0.02 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.40 | 0.68 | 0.04 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.39 | 1.36 | 0.09 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.39 | 2.76 | 0.17 |
| 2 | 32768 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.73 | 2.93 | 0.18 |

## GatedDeltaNetBwdOp

### tileops

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | H | S | DK | DV | BC | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.47 | 1.13 | 0.06 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.47 | 1.14 | 0.06 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.35 | 0.76 | 0.04 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.38 | 1.40 | 0.08 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.69 | 1.57 | 0.09 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 16384 | 64 | 64 | 64 | 1.29 | 1.66 | 0.09 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.38 | 0.70 | 0.04 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.38 | 1.40 | 0.08 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.69 | 1.56 | 0.09 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 16384 | 64 | 64 | 64 | 1.30 | 1.65 | 0.09 |

### fla

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | H | S | DK | DV | BC | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.125 | 0.25 | 2.18 | 0.12 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.125 | 0.25 | 2.15 | 0.12 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.125 | 0.13 | 1.99 | 0.11 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.125 | 0.25 | 2.17 | 0.12 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.125 | 0.47 | 2.26 | 0.12 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 16384 | 64 | 64 | 64 | 0.125 | 1.39 | 1.55 | 0.09 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.125 | 0.14 | 1.99 | 0.11 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.125 | 0.25 | 2.16 | 0.12 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.125 | 0.47 | 2.26 | 0.12 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 16384 | 64 | 64 | 64 | 0.125 | 1.48 | 1.45 | 0.08 |

## GatedDeltaNetOp

### tileops

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | H | S | DK | DV | BC | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 32 | 2.07 | 0.39 | 0.02 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 32 | 2.15 | 0.37 | 0.02 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.28 | 1.42 | 0.08 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.49 | 1.63 | 0.09 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 8192 | 64 | 64 | 64 | 1.12 | 1.44 | 0.08 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 16384 | 64 | 64 | 64 | 3.18 | 1.01 | 0.06 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.27 | 1.50 | 0.09 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.49 | 1.63 | 0.09 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 8192 | 64 | 64 | 64 | 1.14 | 1.41 | 0.08 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 16384 | 64 | 64 | 64 | 3.26 | 0.99 | 0.06 |

### fla

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | H | S | DK | DV | BC | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.125 | 0.35 | 2.31 | 0.13 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.125 | 0.35 | 2.32 | 0.13 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.125 | 0.19 | 2.08 | 0.12 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.125 | 0.34 | 2.35 | 0.14 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.125 | 0.66 | 2.44 | 0.14 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 16384 | 64 | 64 | 64 | 0.125 | 1.27 | 2.53 | 0.15 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.125 | 0.20 | 2.06 | 0.12 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.125 | 0.34 | 2.34 | 0.14 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.125 | 0.66 | 2.42 | 0.14 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 16384 | 64 | 64 | 64 | 0.125 | 1.35 | 2.39 | 0.14 |

## GatedDeltaNetDecodeOp

### tileops

| batch | heads | dim_k | dim_v | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 128 | 128 | torch.float32 | 0.04 | 0.08 | 0.11 |
| 1 | 32 | 128 | 128 | torch.bfloat16 | 0.04 | 0.08 | 0.05 |
| 8 | 32 | 128 | 128 | torch.float32 | 0.04 | 0.57 | 0.77 |
| 8 | 32 | 128 | 128 | torch.bfloat16 | 0.04 | 0.63 | 0.42 |
| 16 | 32 | 128 | 128 | torch.float32 | 0.05 | 1.00 | 1.35 |
| 16 | 32 | 128 | 128 | torch.bfloat16 | 0.04 | 1.21 | 0.82 |
| 32 | 32 | 128 | 128 | torch.float32 | 0.08 | 1.19 | 1.61 |
| 32 | 32 | 128 | 128 | torch.bfloat16 | 0.05 | 2.02 | 1.37 |
| 1 | 32 | 64 | 64 | torch.float32 | 0.04 | 0.02 | 0.03 |
| 8 | 32 | 64 | 64 | torch.float32 | 0.04 | 0.16 | 0.22 |
| 32 | 32 | 64 | 64 | torch.float32 | 0.05 | 0.52 | 0.71 |
| 64 | 32 | 128 | 128 | torch.float32 | 0.16 | 1.26 | 1.70 |
| 64 | 32 | 128 | 128 | torch.bfloat16 | 0.06 | 3.24 | 2.19 |

### torch-ref

| batch | heads | dim_k | dim_v | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 128 | 128 | torch.float32 | 0.33 | 0.01 | 0.01 |
| 1 | 32 | 128 | 128 | torch.bfloat16 | 0.46 | 0.01 | 0.00 |
| 8 | 32 | 128 | 128 | torch.float32 | 0.32 | 0.08 | 0.11 |
| 8 | 32 | 128 | 128 | torch.bfloat16 | 0.52 | 0.05 | 0.03 |
| 16 | 32 | 128 | 128 | torch.float32 | 0.33 | 0.15 | 0.20 |
| 16 | 32 | 128 | 128 | torch.bfloat16 | 0.47 | 0.11 | 0.07 |
| 32 | 32 | 128 | 128 | torch.float32 | 0.34 | 0.29 | 0.40 |
| 32 | 32 | 128 | 128 | torch.bfloat16 | 0.48 | 0.21 | 0.14 |
| 1 | 32 | 64 | 64 | torch.float32 | 0.34 | 0.00 | 0.00 |
| 8 | 32 | 64 | 64 | torch.float32 | 0.34 | 0.02 | 0.03 |
| 32 | 32 | 64 | 64 | torch.float32 | 0.34 | 0.07 | 0.10 |
| 64 | 32 | 128 | 128 | torch.float32 | 0.43 | 0.47 | 0.64 |
| 64 | 32 | 128 | 128 | torch.bfloat16 | 0.60 | 0.34 | 0.23 |

## GemmOp

### tileops

| m | n | k | dtype | trans_a | trans_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1024 | 1024 | 1024 | torch.float16 | False | False | 0.05 | 39.19 | 0.11 |
| 1 | 7168 | 16384 | torch.float16 | False | True | 0.07 | 3.19 | 3.19 |
| 1 | 18432 | 7168 | torch.bfloat16 | False | True | 0.08 | 3.42 | 3.42 |
| 7168 | 1 | 16384 | torch.float16 | False | False | 0.07 | 3.19 | 3.19 |
| 18432 | 1 | 7168 | torch.bfloat16 | False | False | 0.08 | 3.39 | 3.39 |

### torch-cublas

| m | n | k | dtype | trans_a | trans_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1024 | 1024 | 1024 | torch.float16 | False | False | 0.02 | 94.08 | 0.28 |
| 1 | 7168 | 16384 | torch.float16 | False | True | 0.08 | 2.97 | 2.97 |
| 1 | 18432 | 7168 | torch.bfloat16 | False | True | 0.09 | 3.09 | 3.09 |
| 7168 | 1 | 16384 | torch.float16 | False | False | 0.08 | 3.00 | 3.01 |
| 18432 | 1 | 7168 | torch.bfloat16 | False | False | 0.09 | 3.09 | 3.09 |

## GLAFwdOp

### tileops

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.09 | 1.48 | 0.09 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.15 | 1.79 | 0.11 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.28 | 1.92 | 0.12 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.53 | 2.01 | 0.13 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.09 | 1.46 | 0.09 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.15 | 1.82 | 0.11 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.28 | 1.93 | 0.12 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.54 | 1.99 | 0.12 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.31 | 1.29 | 0.07 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.71 | 1.13 | 0.06 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 1.82 | 0.88 | 0.05 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 4.02 | 0.80 | 0.05 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.31 | 1.28 | 0.07 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.71 | 1.13 | 0.06 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 1.86 | 0.87 | 0.05 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 4.16 | 0.77 | 0.04 |

### fla

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.31 | 0.43 | 0.03 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.31 | 0.86 | 0.05 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.31 | 1.71 | 0.11 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.48 | 2.25 | 0.14 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.31 | 0.44 | 0.03 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.30 | 0.88 | 0.06 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.31 | 1.72 | 0.11 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.48 | 2.22 | 0.14 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.22 | 1.81 | 0.10 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.42 | 1.93 | 0.11 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.86 | 1.88 | 0.11 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 2.71 | 1.19 | 0.07 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.22 | 1.81 | 0.10 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.42 | 1.93 | 0.11 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.85 | 1.90 | 0.11 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 2.64 | 1.22 | 0.07 |

## GLABwdOp

### tileops

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | T | H | K | V | BC | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 2048 | 4 | 64 | 64 | 64 | 0.125 | 0.17 | 1.62 | 0.09 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4096 | 4 | 64 | 64 | 64 | 0.125 | 0.32 | 1.69 | 0.09 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 8192 | 4 | 64 | 64 | 64 | 0.125 | 0.61 | 1.77 | 0.10 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 16384 | 4 | 64 | 64 | 64 | 0.125 | 1.19 | 1.81 | 0.10 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 2048 | 4 | 64 | 64 | 64 | 0.125 | 0.17 | 1.61 | 0.09 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4096 | 4 | 64 | 64 | 64 | 0.125 | 0.32 | 1.68 | 0.09 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 8192 | 4 | 64 | 64 | 64 | 0.125 | 0.62 | 1.72 | 0.09 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 16384 | 4 | 64 | 64 | 64 | 0.125 | 1.25 | 1.72 | 0.09 |

### fla

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | T | H | K | V | BC | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 2048 | 4 | 64 | 64 | 64 | 0.125 | 0.16 | 1.73 | 0.09 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4096 | 4 | 64 | 64 | 64 | 0.125 | 0.28 | 1.89 | 0.10 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 8192 | 4 | 64 | 64 | 64 | 0.125 | 0.79 | 1.36 | 0.07 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 16384 | 4 | 64 | 64 | 64 | 0.125 | 2.25 | 0.95 | 0.05 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 2048 | 4 | 64 | 64 | 64 | 0.125 | 0.15 | 1.74 | 0.09 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4096 | 4 | 64 | 64 | 64 | 0.125 | 0.28 | 1.89 | 0.10 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 8192 | 4 | 64 | 64 | 64 | 0.125 | 0.75 | 1.43 | 0.08 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 16384 | 4 | 64 | 64 | 64 | 0.125 | 2.23 | 0.96 | 0.05 |

## GLADecodeOp

### tileops

| batch | heads | dim_k | dim_v | dtype | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 64 | 64 | torch.float32 | 0.125 | 0.03 | 0.02 | 0.04 |
| 1 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.03 | 0.07 | 0.14 |
| 1 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.03 | 0.08 | 0.08 |
| 1 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.03 | 0.07 | 0.08 |
| 8 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.04 | 0.43 | 0.87 |
| 8 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.03 | 0.57 | 0.58 |
| 8 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.03 | 0.57 | 0.58 |
| 16 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.07 | 0.48 | 0.97 |
| 16 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.03 | 1.08 | 1.09 |
| 16 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.03 | 1.07 | 1.09 |
| 32 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.13 | 0.50 | 1.02 |
| 32 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.04 | 1.72 | 1.75 |
| 32 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.04 | 1.68 | 1.70 |
| 64 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.26 | 0.52 | 1.06 |
| 64 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.07 | 2.00 | 2.04 |
| 64 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.07 | 1.97 | 2.00 |

### fla

| batch | heads | dim_k | dim_v | dtype | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 64 | 64 | torch.float32 | 0.125 | 0.01 | 0.08 | 0.16 |
| 1 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.01 | 0.31 | 0.62 |
| 1 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.01 | 0.25 | 0.25 |
| 1 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.01 | 0.27 | 0.27 |
| 8 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.02 | 1.05 | 2.14 |
| 8 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.02 | 1.00 | 1.02 |
| 8 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.02 | 1.00 | 1.01 |
| 16 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.03 | 1.23 | 2.50 |
| 16 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.03 | 1.29 | 1.31 |
| 16 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.03 | 1.29 | 1.31 |
| 32 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.05 | 1.44 | 2.92 |
| 32 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.04 | 1.54 | 1.56 |
| 32 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.04 | 1.53 | 1.56 |
| 64 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.10 | 1.32 | 2.68 |
| 64 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.08 | 1.71 | 1.74 |
| 64 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.08 | 1.60 | 1.63 |

## GroupQueryAttentionFwdOp

### tileops

| batch | seq_len | heads | heads_kv | dim | causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 8 | 4 | 64 | False | torch.float16 | 0.06 | 33.67 | 0.05 |
| 4 | 2048 | 64 | 4 | 128 | False | torch.float16 | 0.93 | 590.77 | 0.31 |
| 4 | 2048 | 64 | 4 | 128 | False | torch.bfloat16 | 0.95 | 576.58 | 0.30 |

### torch-sdpa

| batch | seq_len | heads | heads_kv | dim | causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 8 | 4 | 64 | False | torch.float16 | 0.05 | 44.33 | 0.06 |
| 4 | 2048 | 64 | 4 | 128 | False | torch.float16 | 1.63 | 336.67 | 0.17 |
| 4 | 2048 | 64 | 4 | 128 | False | torch.bfloat16 | 1.61 | 341.59 | 0.18 |

## GroupQueryAttentionBwdOp

### tileops

| batch | seq_len | heads | heads_kv | dim | causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 8 | 4 | 64 | False | torch.float16 | 0.12 | 45.71 | 0.04 |
| 4 | 2048 | 64 | 4 | 128 | False | torch.float16 | 4.33 | 317.46 | 0.10 |
| 4 | 2048 | 64 | 4 | 128 | False | torch.bfloat16 | 4.16 | 330.15 | 0.10 |

### torch-sdpa

| batch | seq_len | heads | heads_kv | dim | causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 8 | 4 | 64 | False | torch.float16 | 0.29 | 18.43 | 0.02 |
| 4 | 2048 | 64 | 4 | 128 | False | torch.float16 | 6.43 | 213.60 | 0.07 |
| 4 | 2048 | 64 | 4 | 128 | False | torch.bfloat16 | 6.42 | 214.01 | 0.07 |

## GroupQueryAttentionDecodeWithKVCacheOp

### tileops

| batch | heads | heads_kv | seq_len_kv | dim | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 8 | 8192 | 128 | torch.float16 | 0.11 | 1.21 | 0.30 |
| 4 | 32 | 4 | 4096 | 128 | torch.bfloat16 | 0.11 | 2.46 | 0.31 |
| 8 | 64 | 16 | 8192 | 128 | torch.float16 | 0.22 | 9.85 | 2.46 |

### torch-sdpa

| batch | heads | heads_kv | seq_len_kv | dim | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 8 | 8192 | 128 | torch.float16 | 0.46 | 0.29 | 0.07 |
| 4 | 32 | 4 | 4096 | 128 | torch.bfloat16 | 0.86 | 0.31 | 0.04 |
| 8 | 64 | 16 | 8192 | 128 | torch.float16 | 7.19 | 0.30 | 0.07 |

## GroupQueryAttentionDecodePagedWithKVCacheOp

### tileops

| batch | heads | heads_kv | seqlen_kv | dim | page_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 16 | 8 | 512 | 128 | 128 | torch.float16 | 0.20 | 0.02 | 0.01 |
| 2 | 8 | 4 | 1024 | 64 | 256 | torch.float16 | 0.18 | 0.02 | 0.01 |
| 1 | 16 | 4 | 2048 | 128 | 512 | torch.float16 | 0.20 | 0.09 | 0.02 |
| 1 | 32 | 16 | 512 | 64 | 128 | torch.float16 | 0.09 | 0.05 | 0.02 |

### torch-ref

| batch | heads | heads_kv | seqlen_kv | dim | page_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 16 | 8 | 512 | 128 | 128 | torch.float16 | 0.51 | 0.01 | 0.00 |
| 2 | 8 | 4 | 1024 | 64 | 256 | torch.float16 | 1.04 | 0.00 | 0.00 |
| 1 | 16 | 4 | 2048 | 128 | 512 | torch.float16 | 0.46 | 0.04 | 0.01 |
| 1 | 32 | 16 | 512 | 64 | 128 | torch.float16 | 0.52 | 0.01 | 0.00 |

## GqaSlidingWindowFwdOp

### tileops

| batch | seq | heads | heads_kv | dim | is_causal | wl | wr | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 512 | 8 | 2 | 64 | True | -1 | -1 | torch.float16 | 0.07 | 7.83 | 0.04 |
| 2 | 512 | 8 | 2 | 64 | True | 128 | -1 | torch.float16 | 0.07 | 3.49 | 0.04 |
| 2 | 768 | 8 | 2 | 64 | False | 256 | -1 | torch.float16 | 0.07 | 27.71 | 0.06 |
| 2 | 512 | 8 | 2 | 64 | False | 64 | 64 | torch.bfloat16 | 0.07 | 3.69 | 0.04 |
| 1 | 2048 | 8 | 2 | 64 | True | 512 | -1 | torch.float16 | 0.07 | 27.95 | 0.08 |

### fa3

| batch | seq | heads | heads_kv | dim | is_causal | wl | wr | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 512 | 8 | 2 | 64 | True | -1 | -1 | torch.float16 | 0.11 | 4.71 | 0.02 |
| 2 | 512 | 8 | 2 | 64 | True | 128 | -1 | torch.float16 | 0.09 | 2.78 | 0.03 |
| 2 | 768 | 8 | 2 | 64 | False | 256 | -1 | torch.float16 | 0.07 | 25.54 | 0.05 |
| 2 | 512 | 8 | 2 | 64 | False | 64 | 64 | torch.bfloat16 | 0.08 | 3.00 | 0.03 |
| 1 | 2048 | 8 | 2 | 64 | True | 512 | -1 | torch.float16 | 0.07 | 25.77 | 0.07 |

## GqaSlidingWindowVarlenFwdOp

### tileops

| batch | heads | heads_kv | dim | is_causal | wl | wr | dtype | max_seqlen_q | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 8 | 128 | True | -1 | -1 | torch.float16 | 3000 | 0.44 | 165.92 | 0.14 |
| 2 | 32 | 8 | 128 | True | 2000 | -1 | torch.float16 | 3073 | 0.57 | 153.19 | 0.17 |
| 4 | 32 | 8 | 128 | False | 1500 | 1500 | torch.float16 | 3500 | 1.16 | 295.50 | 0.18 |
| 2 | 32 | 8 | 128 | True | -1 | -1 | torch.float16 | 1024 | 0.62 | 258.88 | 0.12 |
| 2 | 32 | 8 | 128 | True | -1 | -1 | torch.float16 | 8193 | 2.25 | 336.15 | 0.12 |

### fa3

| batch | heads | heads_kv | dim | is_causal | wl | wr | dtype | max_seqlen_q | max_seqlen_k | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 8 | 128 | True | -1 | -1 | torch.float16 | 3000 | 3000 | 0.29 | 258.42 | 0.22 |
| 2 | 32 | 8 | 128 | True | 2000 | -1 | torch.float16 | 3073 | 3073 | 0.41 | 214.21 | 0.23 |
| 4 | 32 | 8 | 128 | False | 1500 | 1500 | torch.float16 | 3500 | 3500 | 1.26 | 273.94 | 0.16 |
| 2 | 32 | 8 | 128 | True | -1 | -1 | torch.float16 | 1024 | 8192 | 0.60 | 267.10 | 0.13 |
| 2 | 32 | 8 | 128 | True | -1 | -1 | torch.float16 | 8193 | 8193 | 2.24 | 337.21 | 0.12 |

## GroupNormOp

### tileops

| n | c | g | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 128 | 32 | torch.float16 | 0.14 | 0.04 | 0.03 |
| 8 | 128 | 32 | torch.bfloat16 | 0.14 | 0.04 | 0.03 |
| 4 | 256 | 32 | torch.float16 | 0.19 | 0.02 | 0.02 |
| 4 | 128 | 16 | torch.float16 | 0.25 | 0.01 | 0.01 |

### torch

| n | c | g | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 128 | 32 | torch.float16 | 0.05 | 0.11 | 0.09 |
| 8 | 128 | 32 | torch.bfloat16 | 0.05 | 0.11 | 0.09 |
| 4 | 256 | 32 | torch.float16 | 0.05 | 0.08 | 0.07 |
| 4 | 128 | 16 | torch.float16 | 0.05 | 0.05 | 0.04 |

## grouped_gemm_nt

### tileops

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_nt | 16384 | 4 | 4864 | 4096 | torch.float16 | False | True | 28.65 | 22.79 | 0.02 |

### torch-ref

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_nt | 16384 | 4 | 4864 | 4096 | torch.float16 | False | True | 1.83 | 356.56 | 0.25 |

## grouped_gemm_nn

### tileops

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_nn | 16384 | 4 | 4864 | 4096 | torch.float16 | False | False | 13.18 | 49.54 | 0.03 |

### torch-ref

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_nn | 16384 | 4 | 4864 | 4096 | torch.float16 | False | False | 1.30 | 501.29 | 0.35 |

## grouped_gemm_tn

### tileops

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_tn | 16384 | 4 | 4864 | 4096 | torch.float16 | True | False | 9.49 | 68.82 | 0.05 |

### torch-ref

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_tn | 16384 | 4 | 4864 | 4096 | torch.float16 | True | False | 1.30 | 503.85 | 0.35 |

## grouped_gemm_tt

### tileops

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_tt | 16384 | 4 | 4864 | 4096 | torch.float16 | True | True | 10.39 | 62.85 | 0.04 |

### torch-ref

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_tt | 16384 | 4 | 4864 | 4096 | torch.float16 | True | True | 1.42 | 460.11 | 0.32 |

## GroupedGemmOp

### tileops

| batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16384 | 4 | 4864 | 4096 | torch.float16 | True | False | 62.16 | 31.51 | N/A |

### torch-ref

| batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16384 | 4 | 4864 | 4096 | torch.float16 | True | False | 5.84 | 335.12 | N/A |

## leaky_relu

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| leaky_relu | torch.float16 | 4194304 | 0.05 | 0.09 | 0.37 |
| leaky_relu | torch.bfloat16 | 4194304 | 0.05 | 0.09 | 0.37 |
| leaky_relu | torch.float32 | 4194304 | 0.05 | 0.09 | 0.74 |
| leaky_relu | torch.float16 | 10485760 | 0.05 | 0.23 | 0.92 |
| leaky_relu | torch.bfloat16 | 10485760 | 0.05 | 0.23 | 0.93 |
| leaky_relu | torch.float32 | 10485760 | 0.05 | 0.22 | 1.73 |
| leaky_relu | torch.float16 | 20971520 | 0.05 | 0.43 | 1.73 |
| leaky_relu | torch.bfloat16 | 20971520 | 0.05 | 0.44 | 1.77 |
| leaky_relu | torch.float32 | 20971520 | 0.05 | 0.40 | 3.21 |

### torch

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| leaky_relu | torch.float16 | 4194304 | 0.01 | 0.29 | 1.17 |
| leaky_relu | torch.bfloat16 | 4194304 | 0.02 | 0.28 | 1.12 |
| leaky_relu | torch.float32 | 4194304 | 0.02 | 0.26 | 2.09 |
| leaky_relu | torch.float16 | 10485760 | 0.02 | 0.58 | 2.33 |
| leaky_relu | torch.bfloat16 | 10485760 | 0.02 | 0.58 | 2.32 |
| leaky_relu | torch.float32 | 10485760 | 0.03 | 0.39 | 3.15 |
| leaky_relu | torch.float16 | 20971520 | 0.03 | 0.78 | 3.11 |
| leaky_relu | torch.bfloat16 | 20971520 | 0.03 | 0.79 | 3.16 |
| leaky_relu | torch.float32 | 20971520 | 0.05 | 0.45 | 3.59 |

## elu

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| elu | torch.float16 | 4194304 | 0.05 | 0.09 | 0.37 |
| elu | torch.bfloat16 | 4194304 | 0.05 | 0.09 | 0.36 |
| elu | torch.float32 | 4194304 | 0.05 | 0.09 | 0.73 |
| elu | torch.float16 | 10485760 | 0.05 | 0.23 | 0.91 |
| elu | torch.bfloat16 | 10485760 | 0.05 | 0.23 | 0.91 |
| elu | torch.float32 | 10485760 | 0.05 | 0.22 | 1.75 |
| elu | torch.float16 | 20971520 | 0.05 | 0.42 | 1.68 |
| elu | torch.bfloat16 | 20971520 | 0.05 | 0.42 | 1.68 |
| elu | torch.float32 | 20971520 | 0.05 | 0.40 | 3.16 |

### torch

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| elu | torch.float16 | 4194304 | 0.02 | 0.27 | 1.08 |
| elu | torch.bfloat16 | 4194304 | 0.02 | 0.26 | 1.04 |
| elu | torch.float32 | 4194304 | 0.02 | 0.25 | 1.96 |
| elu | torch.float16 | 10485760 | 0.02 | 0.42 | 1.70 |
| elu | torch.bfloat16 | 10485760 | 0.02 | 0.43 | 1.70 |
| elu | torch.float32 | 10485760 | 0.03 | 0.37 | 2.97 |
| elu | torch.float16 | 20971520 | 0.04 | 0.50 | 2.01 |
| elu | torch.bfloat16 | 20971520 | 0.04 | 0.49 | 1.97 |
| elu | torch.float32 | 20971520 | 0.05 | 0.44 | 3.55 |

## hardtanh

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| hardtanh | torch.float16 | 4194304 | 0.05 | 0.09 | 0.37 |
| hardtanh | torch.bfloat16 | 4194304 | 0.05 | 0.09 | 0.37 |
| hardtanh | torch.float32 | 4194304 | 0.05 | 0.09 | 0.72 |
| hardtanh | torch.float16 | 10485760 | 0.05 | 0.23 | 0.91 |
| hardtanh | torch.bfloat16 | 10485760 | 0.05 | 0.22 | 0.89 |
| hardtanh | torch.float32 | 10485760 | 0.05 | 0.22 | 1.75 |
| hardtanh | torch.float16 | 20971520 | 0.05 | 0.44 | 1.76 |
| hardtanh | torch.bfloat16 | 20971520 | 0.05 | 0.44 | 1.76 |
| hardtanh | torch.float32 | 20971520 | 0.05 | 0.40 | 3.24 |

### torch

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| hardtanh | torch.float16 | 4194304 | 0.02 | 0.23 | 0.94 |
| hardtanh | torch.bfloat16 | 4194304 | 0.02 | 0.24 | 0.95 |
| hardtanh | torch.float32 | 4194304 | 0.02 | 0.22 | 1.72 |
| hardtanh | torch.float16 | 10485760 | 0.02 | 0.50 | 1.99 |
| hardtanh | torch.bfloat16 | 10485760 | 0.02 | 0.48 | 1.91 |
| hardtanh | torch.float32 | 10485760 | 0.03 | 0.39 | 3.11 |
| hardtanh | torch.float16 | 20971520 | 0.03 | 0.72 | 2.89 |
| hardtanh | torch.bfloat16 | 20971520 | 0.03 | 0.78 | 3.10 |
| hardtanh | torch.float32 | 20971520 | 0.05 | 0.44 | 3.52 |

## softplus

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| softplus | torch.float16 | 4194304 | 0.05 | 0.09 | 0.37 |
| softplus | torch.bfloat16 | 4194304 | 0.05 | 0.09 | 0.37 |
| softplus | torch.float32 | 4194304 | 0.05 | 0.09 | 0.72 |
| softplus | torch.float16 | 10485760 | 0.05 | 0.22 | 0.89 |
| softplus | torch.bfloat16 | 10485760 | 0.05 | 0.22 | 0.87 |
| softplus | torch.float32 | 10485760 | 0.05 | 0.21 | 1.68 |
| softplus | torch.float16 | 20971520 | 0.05 | 0.39 | 1.56 |
| softplus | torch.bfloat16 | 20971520 | 0.05 | 0.40 | 1.62 |
| softplus | torch.float32 | 20971520 | 0.05 | 0.40 | 3.22 |

### torch

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| softplus | torch.float16 | 4194304 | 0.02 | 0.25 | 0.99 |
| softplus | torch.bfloat16 | 4194304 | 0.02 | 0.25 | 0.98 |
| softplus | torch.float32 | 4194304 | 0.02 | 0.24 | 1.90 |
| softplus | torch.float16 | 10485760 | 0.03 | 0.35 | 1.39 |
| softplus | torch.bfloat16 | 10485760 | 0.03 | 0.36 | 1.43 |
| softplus | torch.float32 | 10485760 | 0.03 | 0.34 | 2.74 |
| softplus | torch.float16 | 20971520 | 0.05 | 0.42 | 1.67 |
| softplus | torch.bfloat16 | 20971520 | 0.05 | 0.40 | 1.59 |
| softplus | torch.float32 | 20971520 | 0.05 | 0.40 | 3.20 |

## clamp

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| clamp | torch.float16 | 4194304 | 0.05 | 0.09 | 0.37 |
| clamp | torch.bfloat16 | 4194304 | 0.05 | 0.09 | 0.37 |
| clamp | torch.float32 | 4194304 | 0.05 | 0.09 | 0.72 |
| clamp | torch.float16 | 10485760 | 0.05 | 0.23 | 0.93 |
| clamp | torch.bfloat16 | 10485760 | 0.04 | 0.24 | 0.94 |
| clamp | torch.float32 | 10485760 | 0.05 | 0.22 | 1.77 |
| clamp | torch.float16 | 20971520 | 0.05 | 0.45 | 1.79 |
| clamp | torch.bfloat16 | 20971520 | 0.05 | 0.44 | 1.77 |
| clamp | torch.float32 | 20971520 | 0.05 | 0.41 | 3.25 |

### torch

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| clamp | torch.float16 | 4194304 | 0.02 | 0.27 | 1.08 |
| clamp | torch.bfloat16 | 4194304 | 0.01 | 0.29 | 1.16 |
| clamp | torch.float32 | 4194304 | 0.02 | 0.25 | 1.97 |
| clamp | torch.float16 | 10485760 | 0.02 | 0.56 | 2.24 |
| clamp | torch.bfloat16 | 10485760 | 0.02 | 0.56 | 2.25 |
| clamp | torch.float32 | 10485760 | 0.03 | 0.39 | 3.09 |
| clamp | torch.float16 | 20971520 | 0.03 | 0.71 | 2.84 |
| clamp | torch.bfloat16 | 20971520 | 0.03 | 0.77 | 3.10 |
| clamp | torch.float32 | 20971520 | 0.05 | 0.44 | 3.50 |

## nan_to_num

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| nan_to_num | torch.float16 | 4194304 | 0.05 | 0.09 | 0.36 |
| nan_to_num | torch.bfloat16 | 4194304 | 0.05 | 0.09 | 0.37 |
| nan_to_num | torch.float32 | 4194304 | 0.05 | 0.09 | 0.74 |
| nan_to_num | torch.float16 | 10485760 | 0.04 | 0.23 | 0.93 |
| nan_to_num | torch.bfloat16 | 10485760 | 0.05 | 0.23 | 0.93 |
| nan_to_num | torch.float32 | 10485760 | 0.05 | 0.22 | 1.77 |
| nan_to_num | torch.float16 | 20971520 | 0.05 | 0.43 | 1.73 |
| nan_to_num | torch.bfloat16 | 20971520 | 0.05 | 0.43 | 1.72 |
| nan_to_num | torch.float32 | 20971520 | 0.05 | 0.40 | 3.22 |

### torch

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| nan_to_num | torch.float16 | 4194304 | 0.02 | 0.27 | 1.07 |
| nan_to_num | torch.bfloat16 | 4194304 | 0.02 | 0.27 | 1.09 |
| nan_to_num | torch.float32 | 4194304 | 0.02 | 0.24 | 1.93 |
| nan_to_num | torch.float16 | 10485760 | 0.02 | 0.53 | 2.13 |
| nan_to_num | torch.bfloat16 | 10485760 | 0.02 | 0.52 | 2.08 |
| nan_to_num | torch.float32 | 10485760 | 0.03 | 0.39 | 3.09 |
| nan_to_num | torch.float16 | 20971520 | 0.03 | 0.78 | 3.11 |
| nan_to_num | torch.bfloat16 | 20971520 | 0.03 | 0.77 | 3.07 |
| nan_to_num | torch.float32 | 20971520 | 0.05 | 0.45 | 3.63 |

## PreluOp

### tileops

| num_channels | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 128 | torch.float16 | 131072 | 0.05 | 0.00 | 0.01 |
| 128 | torch.bfloat16 | 131072 | 0.05 | 0.00 | 0.01 |
| 128 | torch.float32 | 131072 | 0.05 | 0.00 | 0.02 |
| 4096 | torch.float16 | 4194304 | 0.05 | 0.09 | 0.34 |
| 4096 | torch.bfloat16 | 4194304 | 0.05 | 0.09 | 0.35 |
| 4096 | torch.float32 | 4194304 | 0.05 | 0.09 | 0.69 |
| 10240 | torch.float16 | 10485760 | 0.05 | 0.22 | 0.87 |
| 10240 | torch.bfloat16 | 10485760 | 0.05 | 0.22 | 0.86 |
| 10240 | torch.float32 | 10485760 | 0.05 | 0.21 | 1.68 |
| 20480 | torch.float16 | 20971520 | 0.05 | 0.42 | 1.68 |
| 20480 | torch.bfloat16 | 20971520 | 0.05 | 0.43 | 1.70 |
| 20480 | torch.float32 | 20971520 | 0.05 | 0.38 | 3.07 |

### torch

| num_channels | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 128 | torch.float16 | 131072 | 0.02 | 0.01 | 0.03 |
| 128 | torch.bfloat16 | 131072 | 0.02 | 0.01 | 0.03 |
| 128 | torch.float32 | 131072 | 0.02 | 0.01 | 0.06 |
| 4096 | torch.float16 | 4194304 | 0.02 | 0.19 | 0.78 |
| 4096 | torch.bfloat16 | 4194304 | 0.02 | 0.19 | 0.77 |
| 4096 | torch.float32 | 4194304 | 0.02 | 0.19 | 1.54 |
| 10240 | torch.float16 | 10485760 | 0.04 | 0.30 | 1.18 |
| 10240 | torch.bfloat16 | 10485760 | 0.04 | 0.29 | 1.17 |
| 10240 | torch.float32 | 10485760 | 0.04 | 0.26 | 2.05 |
| 20480 | torch.float16 | 20971520 | 0.06 | 0.33 | 1.34 |
| 20480 | torch.bfloat16 | 20971520 | 0.06 | 0.33 | 1.32 |
| 20480 | torch.float32 | 20971520 | 0.07 | 0.28 | 2.28 |

## WhereOp

### tileops

| dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| torch.float16 | 4194304 | 0.05 | 0.08 | 0.56 |
| torch.bfloat16 | 4194304 | 0.05 | 0.08 | 0.57 |
| torch.float32 | 4194304 | 0.05 | 0.08 | 1.05 |
| torch.float16 | 10485760 | 0.05 | 0.20 | 1.38 |
| torch.bfloat16 | 10485760 | 0.05 | 0.19 | 1.36 |
| torch.float32 | 10485760 | 0.06 | 0.18 | 2.38 |
| torch.float16 | 20971520 | 0.06 | 0.36 | 2.54 |
| torch.bfloat16 | 20971520 | 0.06 | 0.36 | 2.55 |
| torch.float32 | 20971520 | 0.07 | 0.29 | 3.80 |
| torch.float8_e4m3fn | 4194304 | 0.05 | 0.08 | 0.31 |
| torch.float8_e5m2 | 4194304 | 0.05 | 0.08 | 0.31 |
| torch.float8_e4m3fn | 10485760 | 0.06 | 0.19 | 0.76 |
| torch.float8_e5m2 | 10485760 | 0.06 | 0.19 | 0.75 |
| torch.float8_e4m3fn | 20971520 | 0.06 | 0.34 | 1.35 |
| torch.float8_e5m2 | 20971520 | 0.06 | 0.34 | 1.35 |

### torch

| dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| torch.float16 | 4194304 | 0.02 | 0.24 | 1.66 |
| torch.bfloat16 | 4194304 | 0.02 | 0.24 | 1.65 |
| torch.float32 | 4194304 | 0.02 | 0.20 | 2.62 |
| torch.float16 | 10485760 | 0.02 | 0.42 | 2.95 |
| torch.bfloat16 | 10485760 | 0.02 | 0.42 | 2.95 |
| torch.float32 | 10485760 | 0.04 | 0.26 | 3.44 |
| torch.float16 | 20971520 | 0.04 | 0.50 | 3.50 |
| torch.bfloat16 | 20971520 | 0.04 | 0.50 | 3.50 |
| torch.float32 | 20971520 | 0.07 | 0.30 | 3.85 |
| torch.float8_e4m3fn | 4194304 | 0.02 | 0.21 | 0.85 |
| torch.float8_e5m2 | 4194304 | 0.02 | 0.21 | 0.83 |
| torch.float8_e4m3fn | 10485760 | 0.02 | 0.43 | 1.71 |
| torch.float8_e5m2 | 10485760 | 0.02 | 0.42 | 1.68 |
| torch.float8_e4m3fn | 20971520 | 0.03 | 0.70 | 2.81 |
| torch.float8_e5m2 | 20971520 | 0.03 | 0.70 | 2.78 |

## MaskedFillOp

### tileops

| dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| torch.float16 | 4194304 | 0.05 | 0.08 | 0.41 |
| torch.bfloat16 | 4194304 | 0.05 | 0.08 | 0.41 |
| torch.float32 | 4194304 | 0.05 | 0.08 | 0.74 |
| torch.float16 | 10485760 | 0.05 | 0.21 | 1.05 |
| torch.bfloat16 | 10485760 | 0.05 | 0.21 | 1.03 |
| torch.float32 | 10485760 | 0.05 | 0.20 | 1.76 |
| torch.float16 | 20971520 | 0.06 | 0.38 | 1.89 |
| torch.bfloat16 | 20971520 | 0.06 | 0.38 | 1.90 |
| torch.float32 | 20971520 | 0.06 | 0.37 | 3.33 |
| torch.float8_e4m3fn | 4194304 | 0.05 | 0.08 | 0.23 |
| torch.float8_e5m2 | 4194304 | 0.07 | 0.06 | 0.18 |
| torch.float8_e4m3fn | 10485760 | 0.05 | 0.20 | 0.60 |
| torch.float8_e5m2 | 10485760 | 0.07 | 0.15 | 0.44 |
| torch.float8_e4m3fn | 20971520 | 0.06 | 0.36 | 1.09 |
| torch.float8_e5m2 | 20971520 | 0.08 | 0.26 | 0.79 |

### torch

| dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| torch.float16 | 4194304 | 0.02 | 0.18 | 0.88 |
| torch.bfloat16 | 4194304 | 0.02 | 0.18 | 0.88 |
| torch.float32 | 4194304 | 0.03 | 0.15 | 1.38 |
| torch.float16 | 10485760 | 0.03 | 0.36 | 1.78 |
| torch.bfloat16 | 10485760 | 0.03 | 0.36 | 1.79 |
| torch.float32 | 10485760 | 0.05 | 0.20 | 1.81 |
| torch.float16 | 20971520 | 0.05 | 0.38 | 1.91 |
| torch.bfloat16 | 20971520 | 0.05 | 0.38 | 1.91 |
| torch.float32 | 20971520 | 0.09 | 0.22 | 2.00 |

### torch-ref

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| masked_fill | torch.float8_e4m3fn | 4194304 | 0.06 | 0.07 | 0.22 |
| masked_fill | torch.float8_e5m2 | 4194304 | 0.06 | 0.07 | 0.22 |
| masked_fill | torch.float8_e4m3fn | 10485760 | 0.07 | 0.14 | 0.42 |
| masked_fill | torch.float8_e5m2 | 10485760 | 0.07 | 0.14 | 0.43 |
| masked_fill | torch.float8_e4m3fn | 20971520 | 0.15 | 0.14 | 0.42 |
| masked_fill | torch.float8_e5m2 | 20971520 | 0.15 | 0.14 | 0.43 |

## alibi

### tileops

| op_name | seq_len | dim | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| alibi | 512 | 64 | torch.float16 | 0.05 | 0.37 | 0.74 |
| alibi | 512 | 64 | torch.bfloat16 | 0.05 | 0.37 | 0.73 |
| alibi | 512 | 64 | torch.float32 | 0.04 | 0.40 | 1.60 |
| alibi | 2048 | 64 | torch.float16 | 0.29 | 0.91 | 1.83 |
| alibi | 2048 | 64 | torch.bfloat16 | 0.29 | 0.91 | 1.83 |
| alibi | 2048 | 64 | torch.float32 | 0.29 | 0.94 | 3.75 |
| alibi | 4096 | 128 | torch.float16 | 1.10 | 1.95 | 3.91 |
| alibi | 4096 | 128 | torch.bfloat16 | 1.11 | 1.93 | 3.86 |
| alibi | 4096 | 128 | torch.float32 | 2.12 | 1.01 | 4.05 |

### torch-ref

| op_name | seq_len | dim | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| alibi | 512 | 64 | torch.float16 | 0.14 | 0.12 | 0.23 |
| alibi | 512 | 64 | torch.bfloat16 | 0.14 | 0.12 | 0.23 |
| alibi | 512 | 64 | torch.float32 | 0.13 | 0.13 | 0.51 |
| alibi | 2048 | 64 | torch.float16 | 1.07 | 0.25 | 0.50 |
| alibi | 2048 | 64 | torch.bfloat16 | 1.06 | 0.25 | 0.50 |
| alibi | 2048 | 64 | torch.float32 | 0.70 | 0.39 | 1.54 |
| alibi | 4096 | 128 | torch.float16 | 9.57 | 0.22 | 0.45 |
| alibi | 4096 | 128 | torch.bfloat16 | 9.56 | 0.22 | 0.45 |
| alibi | 4096 | 128 | torch.float32 | 6.62 | 0.32 | 1.30 |

## sinusoidal

### tileops

| op_name | seq_len | dim | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| sinusoidal | 512 | 256 | torch.float16 | 0.04 | 0.00 | 0.01 |
| sinusoidal | 512 | 256 | torch.bfloat16 | 0.04 | 0.00 | 0.01 |
| sinusoidal | 512 | 256 | torch.float32 | 0.04 | 0.00 | 0.01 |
| sinusoidal | 2048 | 300 | torch.float16 | 0.04 | 0.02 | 0.03 |
| sinusoidal | 2048 | 300 | torch.bfloat16 | 0.04 | 0.02 | 0.03 |
| sinusoidal | 2048 | 300 | torch.float32 | 0.04 | 0.02 | 0.06 |
| sinusoidal | 4096 | 512 | torch.float16 | 0.04 | 0.05 | 0.11 |
| sinusoidal | 4096 | 512 | torch.bfloat16 | 0.04 | 0.05 | 0.10 |
| sinusoidal | 4096 | 512 | torch.float32 | 0.04 | 0.05 | 0.21 |

### torch-ref

| op_name | seq_len | dim | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| sinusoidal | 512 | 256 | torch.float16 | 0.15 | 0.00 | 0.00 |
| sinusoidal | 512 | 256 | torch.bfloat16 | 0.15 | 0.00 | 0.00 |
| sinusoidal | 512 | 256 | torch.float32 | 0.13 | 0.00 | 0.00 |
| sinusoidal | 2048 | 300 | torch.float16 | 0.14 | 0.00 | 0.01 |
| sinusoidal | 2048 | 300 | torch.bfloat16 | 0.14 | 0.00 | 0.01 |
| sinusoidal | 2048 | 300 | torch.float32 | 0.13 | 0.00 | 0.02 |
| sinusoidal | 4096 | 512 | torch.float16 | 0.15 | 0.01 | 0.03 |
| sinusoidal | 4096 | 512 | torch.bfloat16 | 0.15 | 0.01 | 0.03 |
| sinusoidal | 4096 | 512 | torch.float32 | 0.13 | 0.02 | 0.06 |

## leaky_relu_fp8

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| leaky_relu | torch.float8_e4m3fn | 4194304 | 0.05 | 0.09 | 0.18 |
| leaky_relu | torch.float8_e5m2 | 4194304 | 0.06 | 0.07 | 0.13 |
| leaky_relu | torch.float8_e4m3fn | 10485760 | 0.05 | 0.23 | 0.45 |
| leaky_relu | torch.float8_e5m2 | 10485760 | 0.07 | 0.15 | 0.31 |
| leaky_relu | torch.float8_e4m3fn | 20971520 | 0.05 | 0.41 | 0.81 |
| leaky_relu | torch.float8_e5m2 | 20971520 | 0.11 | 0.20 | 0.40 |

### torch-ref

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| leaky_relu | torch.float8_e4m3fn | 4194304 | 0.04 | 0.11 | 0.22 |
| leaky_relu | torch.float8_e5m2 | 4194304 | 0.04 | 0.11 | 0.21 |
| leaky_relu | torch.float8_e4m3fn | 10485760 | 0.06 | 0.17 | 0.33 |
| leaky_relu | torch.float8_e5m2 | 10485760 | 0.06 | 0.17 | 0.34 |
| leaky_relu | torch.float8_e4m3fn | 20971520 | 0.12 | 0.18 | 0.35 |
| leaky_relu | torch.float8_e5m2 | 20971520 | 0.12 | 0.18 | 0.35 |

## elu_fp8

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| elu | torch.float8_e4m3fn | 4194304 | 0.05 | 0.09 | 0.18 |
| elu | torch.float8_e5m2 | 4194304 | 0.06 | 0.07 | 0.13 |
| elu | torch.float8_e4m3fn | 10485760 | 0.05 | 0.23 | 0.46 |
| elu | torch.float8_e5m2 | 10485760 | 0.06 | 0.17 | 0.34 |
| elu | torch.float8_e4m3fn | 20971520 | 0.05 | 0.44 | 0.88 |
| elu | torch.float8_e5m2 | 20971520 | 0.07 | 0.28 | 0.57 |

### torch-ref

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| elu | torch.float8_e4m3fn | 4194304 | 0.04 | 0.10 | 0.21 |
| elu | torch.float8_e5m2 | 4194304 | 0.04 | 0.11 | 0.21 |
| elu | torch.float8_e4m3fn | 10485760 | 0.07 | 0.15 | 0.30 |
| elu | torch.float8_e5m2 | 10485760 | 0.07 | 0.14 | 0.29 |
| elu | torch.float8_e4m3fn | 20971520 | 0.14 | 0.15 | 0.31 |
| elu | torch.float8_e5m2 | 20971520 | 0.13 | 0.16 | 0.31 |

## clamp_fp8

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| clamp | torch.float8_e4m3fn | 4194304 | 0.05 | 0.09 | 0.17 |
| clamp | torch.float8_e5m2 | 4194304 | 0.06 | 0.07 | 0.13 |
| clamp | torch.float8_e4m3fn | 10485760 | 0.05 | 0.23 | 0.46 |
| clamp | torch.float8_e5m2 | 10485760 | 0.06 | 0.17 | 0.34 |
| clamp | torch.float8_e4m3fn | 20971520 | 0.05 | 0.45 | 0.90 |
| clamp | torch.float8_e5m2 | 20971520 | 0.07 | 0.31 | 0.62 |

### torch-ref

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| clamp | torch.float8_e4m3fn | 4194304 | 0.04 | 0.11 | 0.21 |
| clamp | torch.float8_e5m2 | 4194304 | 0.04 | 0.11 | 0.21 |
| clamp | torch.float8_e4m3fn | 10485760 | 0.07 | 0.16 | 0.32 |
| clamp | torch.float8_e5m2 | 10485760 | 0.06 | 0.17 | 0.33 |
| clamp | torch.float8_e4m3fn | 20971520 | 0.12 | 0.17 | 0.35 |
| clamp | torch.float8_e5m2 | 20971520 | 0.12 | 0.17 | 0.35 |

## InstanceNormOp

### tileops

| n | c | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 8 | 128 | torch.float16 | 0.12 | 0.04 | 0.03 |
| 8 | 128 | torch.bfloat16 | 0.12 | 0.04 | 0.03 |
| 4 | 256 | torch.float16 | 0.15 | 0.03 | 0.02 |
| 4 | 64 | torch.float16 | 0.17 | 0.01 | 0.01 |

### torch

| n | c | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 8 | 128 | torch.float16 | 0.08 | 0.06 | 0.05 |
| 8 | 128 | torch.bfloat16 | 0.08 | 0.06 | 0.05 |
| 4 | 256 | torch.float16 | 0.08 | 0.05 | 0.04 |
| 4 | 64 | torch.float16 | 0.08 | 0.01 | 0.01 |

## LayerNormOp

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.07 | 0.32 | 0.25 |
| 4096 | 4096 | torch.bfloat16 | 0.07 | 1.22 | 0.97 |
| 2048 | 5120 | torch.float16 | 0.06 | 0.82 | 0.65 |
| 1025 | 4096 | torch.float16 | 0.06 | 0.33 | 0.27 |

### torch

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.03 | 0.82 | 0.66 |
| 4096 | 4096 | torch.bfloat16 | 0.04 | 2.23 | 1.78 |
| 2048 | 5120 | torch.float16 | 0.03 | 1.94 | 1.55 |
| 1025 | 4096 | torch.float16 | 0.02 | 0.87 | 0.70 |

## AnyOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | any | 0.07 | 0.06 | 0.11 |
| 1024 | 4096 | torch.bfloat16 | any | 0.07 | 0.06 | 0.11 |
| 1024 | 4096 | torch.float32 | any | 0.07 | 0.06 | 0.23 |
| 1024 | 4096 | torch.int32 | any | 0.09 | 0.05 | 0.19 |
| 4096 | 4096 | torch.float16 | any | 0.07 | 0.23 | 0.46 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | any | 0.03 | 0.13 | 0.25 |
| 1024 | 4096 | torch.bfloat16 | any | 0.03 | 0.13 | 0.26 |
| 1024 | 4096 | torch.float32 | any | 0.03 | 0.13 | 0.52 |
| 1024 | 4096 | torch.int32 | any | 0.03 | 0.13 | 0.52 |
| 4096 | 4096 | torch.float16 | any | 0.07 | 0.23 | 0.46 |

## AllOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | all | 0.07 | 0.06 | 0.11 |
| 1024 | 4096 | torch.bfloat16 | all | 0.07 | 0.06 | 0.11 |
| 1024 | 4096 | torch.float32 | all | 0.07 | 0.06 | 0.23 |
| 1024 | 4096 | torch.int32 | all | 0.09 | 0.05 | 0.19 |
| 4096 | 4096 | torch.float16 | all | 0.07 | 0.23 | 0.45 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | all | 0.03 | 0.13 | 0.26 |
| 1024 | 4096 | torch.bfloat16 | all | 0.03 | 0.13 | 0.26 |
| 1024 | 4096 | torch.float32 | all | 0.03 | 0.13 | 0.50 |
| 1024 | 4096 | torch.int32 | all | 0.03 | 0.13 | 0.52 |
| 4096 | 4096 | torch.float16 | all | 0.07 | 0.23 | 0.46 |

## CountNonzeroOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | count_nonzero | 0.06 | 0.07 | 0.14 |
| 1024 | 4096 | torch.bfloat16 | count_nonzero | 0.06 | 0.07 | 0.14 |
| 1024 | 4096 | torch.float32 | count_nonzero | 0.06 | 0.07 | 0.28 |
| 1024 | 4096 | torch.int32 | count_nonzero | 0.08 | 0.05 | 0.22 |
| 4096 | 4096 | torch.float16 | count_nonzero | 0.06 | 0.27 | 0.54 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | count_nonzero | 0.04 | 0.10 | 0.19 |
| 1024 | 4096 | torch.bfloat16 | count_nonzero | 0.04 | 0.10 | 0.19 |
| 1024 | 4096 | torch.float32 | count_nonzero | 0.04 | 0.10 | 0.38 |
| 1024 | 4096 | torch.int32 | count_nonzero | 0.04 | 0.10 | 0.38 |
| 4096 | 4096 | torch.float16 | count_nonzero | 0.12 | 0.14 | 0.29 |

## MeanPoolingForwardOp

### tileops

| batch_size | seq_len | heads | dim | chunk_size | dtype | accum_dtype | chunks_per_bacth | seq_num | use_offsets | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8192 | 64 | 128 | 64 | torch.float16 | torch.float32 | 128 | 1 | 0 | 0.14 | 0.48 | 0.97 |
| 2 | 2048 | 64 | 128 | 64 | torch.float16 | torch.float32 | 32 | 2 | 0 | 0.08 | 0.43 | 0.88 |
| 1 | 8192 | 64 | 128 | 64 | torch.float16 | torch.float32 | 128 | 4 | 1 | 0.20 | 0.33 | 0.67 |
| 1 | 1000 | 64 | 128 | 32 | torch.float16 | torch.float32 | 34 | 4 | 1 | 0.06 | 0.15 | 0.29 |

### torch-ref

| batch_size | seq_len | heads | dim | chunk_size | dtype | accum_dtype | chunks_per_bacth | seq_num | use_offsets | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8192 | 64 | 128 | 64 | torch.float16 | torch.float32 | 128 | 1 | 0 | 3.34 | 0.02 | 0.04 |
| 2 | 2048 | 64 | 128 | 64 | torch.float16 | torch.float32 | 32 | 2 | 0 | 0.85 | 0.04 | 0.08 |
| 1 | 8192 | 64 | 128 | 64 | torch.float16 | torch.float32 | 128 | 4 | 1 | 4.01 | 0.02 | 0.03 |
| 1 | 1000 | 64 | 128 | 32 | torch.float16 | torch.float32 | 34 | 4 | 1 | 1.24 | 0.01 | 0.01 |

## MultiHeadAttentionFwdOp

### tileops

| batch | seq_len | heads | dim | causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 8 | 64 | False | torch.float16 | 0.07 | 32.74 | 0.06 |
| 16 | 2048 | 16 | 128 | False | torch.float16 | 1.03 | 531.36 | 0.52 |
| 4 | 4096 | 16 | 128 | False | torch.bfloat16 | 0.94 | 585.35 | 0.29 |

### torch-sdpa

| batch | seq_len | heads | dim | causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 8 | 64 | False | torch.float16 | 0.06 | 39.04 | 0.08 |
| 16 | 2048 | 16 | 128 | False | torch.float16 | 1.65 | 332.55 | 0.32 |
| 4 | 4096 | 16 | 128 | False | torch.bfloat16 | 1.60 | 343.82 | 0.17 |

## MultiHeadAttentionBwdOp

### tileops

| batch | seq_len | heads | dim | causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 8 | 64 | False | torch.float16 | 0.09 | 60.14 | 0.08 |
| 16 | 2048 | 16 | 128 | False | torch.float16 | 3.98 | 344.93 | 0.24 |
| 4 | 4096 | 16 | 128 | False | torch.bfloat16 | 3.67 | 374.87 | 0.13 |

### torch-sdpa

| batch | seq_len | heads | dim | causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 8 | 64 | False | torch.float16 | 0.21 | 25.94 | 0.04 |
| 16 | 2048 | 16 | 128 | False | torch.float16 | 6.45 | 212.95 | 0.15 |
| 4 | 4096 | 16 | 128 | False | torch.bfloat16 | 6.29 | 218.60 | 0.07 |

## MultiHeadAttentionDecodeWithKVCacheOp

### tileops

| b | h | s_q | s_kv | d | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 128 | 8192 | 128 | torch.float16 | 0.13 | 130.40 | 1.03 |
| 1 | 32 | 128 | 8192 | 128 | torch.bfloat16 | 0.14 | 122.11 | 0.97 |
| 1 | 32 | 128 | 5 | 128 | torch.float16 | 0.06 | 0.17 | 0.03 |

### torch-sdpa

| b | h | s_q | s_kv | d | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 128 | 8192 | 128 | torch.float16 | 0.08 | 206.43 | 1.64 |
| 1 | 32 | 128 | 8192 | 128 | torch.bfloat16 | 0.08 | 225.67 | 1.79 |
| 1 | 32 | 128 | 5 | 128 | torch.float16 | 0.07 | 0.16 | 0.03 |

## MultiHeadAttentionDecodePagedWithKVCacheOp

### tileops

| batch | heads | seqlen_q | seqlen_kv | dim | page_size | is_causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 16 | 1 | 512 | 128 | 128 | False | torch.float16 | 0.21 | 0.02 | 0.02 |
| 2 | 8 | 1 | 1024 | 64 | 256 | False | torch.float16 | 0.20 | 0.02 | 0.01 |
| 1 | 8 | 1 | 1024 | 64 | 256 | False | torch.float16 | 0.21 | 0.01 | 0.01 |
| 1 | 8 | 1 | 512 | 64 | 256 | False | torch.float16 | 0.21 | 0.01 | 0.01 |

### torch-ref

| batch | heads | seqlen_q | seqlen_kv | dim | page_size | is_causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 16 | 1 | 512 | 128 | 128 | False | torch.float16 | 0.35 | 0.01 | 0.01 |
| 2 | 8 | 1 | 1024 | 64 | 256 | False | torch.float16 | 0.67 | 0.01 | 0.00 |
| 1 | 8 | 1 | 1024 | 64 | 256 | False | torch.float16 | 0.34 | 0.01 | 0.01 |
| 1 | 8 | 1 | 512 | 64 | 256 | False | torch.float16 | 0.26 | 0.00 | 0.00 |

## ManifoldConstrainedHyperConnectionPostOp

### tileops

| batch | n_expand | c_x | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | 1280 | torch.bfloat16 | 0.05 | 1.04 | 0.00 |
| 2 | 4 | 1920 | torch.bfloat16 | 0.05 | 4.76 | 0.00 |
| 4 | 4 | 2560 | torch.bfloat16 | 0.05 | 16.56 | 0.00 |

### torch-ref

| batch | n_expand | c_x | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | 1280 | torch.bfloat16 | 0.07 | 0.70 | 0.00 |
| 2 | 4 | 1920 | torch.bfloat16 | 0.08 | 3.06 | 0.00 |
| 4 | 4 | 2560 | torch.bfloat16 | 0.08 | 10.95 | 0.00 |

## ManifoldConstrainedHyperConnectionPreOp

### tileops

| batch | n_expand | c_x | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | 1280 | torch.bfloat16 | 0.12 | 10.58 | 0.00 |
| 2 | 4 | 1920 | torch.bfloat16 | 0.13 | 44.74 | 0.00 |
| 4 | 4 | 2560 | torch.bfloat16 | 0.13 | 159.83 | 0.00 |

### torch-ref

| batch | n_expand | c_x | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | 1280 | torch.bfloat16 | 1.65 | 0.76 | 0.00 |
| 2 | 4 | 1920 | torch.bfloat16 | 1.96 | 2.90 | 0.00 |
| 4 | 4 | 2560 | torch.bfloat16 | 2.37 | 8.49 | 0.00 |

## fused_topk

### tileops

| num_tokens | num_experts | top_k | scoring_func | renormalize | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 128 | 8 | softmax | False | torch.bfloat16 | 0.05 | 0.02 | 0.00 |
| 2048 | 128 | 8 | softmax | False | torch.bfloat16 | 0.04 | 0.11 | 0.02 |
| 4096 | 128 | 8 | softmax | False | torch.bfloat16 | 0.04 | 0.23 | 0.03 |
| 512 | 256 | 8 | softmax | True | torch.bfloat16 | 0.08 | 0.03 | 0.00 |
| 2048 | 256 | 8 | softmax | True | torch.bfloat16 | 0.07 | 0.14 | 0.02 |
| 4096 | 256 | 8 | softmax | True | torch.bfloat16 | 0.06 | 0.30 | 0.04 |
| 512 | 256 | 8 | sigmoid | True | torch.bfloat16 | 0.06 | 0.04 | 0.00 |
| 2048 | 256 | 8 | sigmoid | True | torch.bfloat16 | 0.06 | 0.15 | 0.02 |
| 4096 | 256 | 8 | sigmoid | True | torch.bfloat16 | 0.06 | 0.30 | 0.04 |

### pytorch-ref

| num_tokens | num_experts | top_k | scoring_func | renormalize | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 128 | 8 | softmax | False | torch.bfloat16 | 0.06 | 0.02 | 0.00 |
| 2048 | 128 | 8 | softmax | False | torch.bfloat16 | 0.06 | 0.08 | 0.01 |
| 4096 | 128 | 8 | softmax | False | torch.bfloat16 | 0.06 | 0.15 | 0.02 |
| 512 | 256 | 8 | softmax | True | torch.bfloat16 | 0.10 | 0.02 | 0.00 |
| 2048 | 256 | 8 | softmax | True | torch.bfloat16 | 0.08 | 0.11 | 0.01 |
| 4096 | 256 | 8 | softmax | True | torch.bfloat16 | 0.09 | 0.20 | 0.03 |
| 512 | 256 | 8 | sigmoid | True | torch.bfloat16 | 0.07 | 0.03 | 0.00 |
| 2048 | 256 | 8 | sigmoid | True | torch.bfloat16 | 0.08 | 0.12 | 0.02 |
| 4096 | 256 | 8 | sigmoid | True | torch.bfloat16 | 0.10 | 0.19 | 0.02 |

### vllm

| num_tokens | num_experts | top_k | scoring_func | renormalize | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 128 | 8 | softmax | False | torch.bfloat16 | 0.05 | 0.03 | 0.00 |
| 2048 | 128 | 8 | softmax | False | torch.bfloat16 | 0.05 | 0.10 | 0.01 |
| 4096 | 128 | 8 | softmax | False | torch.bfloat16 | 0.05 | 0.20 | 0.03 |
| 512 | 256 | 8 | softmax | True | torch.bfloat16 | 0.04 | 0.05 | 0.01 |
| 2048 | 256 | 8 | softmax | True | torch.bfloat16 | 0.05 | 0.20 | 0.02 |
| 4096 | 256 | 8 | softmax | True | torch.bfloat16 | 0.05 | 0.42 | 0.05 |
| 512 | 256 | 8 | sigmoid | True | torch.bfloat16 | 0.04 | 0.06 | 0.01 |
| 2048 | 256 | 8 | sigmoid | True | torch.bfloat16 | 0.04 | 0.22 | 0.03 |
| 4096 | 256 | 8 | sigmoid | True | torch.bfloat16 | 0.05 | 0.42 | 0.05 |

## KimiMoENopadOp

### tileops

| num_tokens | num_experts | top_k | hidden_size | ffn_size | routed_scaling_factor | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 384 | 6 | 7168 | 2048 | 2.872 | torch.bfloat16 | 3.90 | 69.33 | 8.67 |
| 2048 | 384 | 6 | 7168 | 2048 | 2.872 | torch.bfloat16 | 6.17 | 175.28 | 5.49 |
| 4096 | 384 | 6 | 7168 | 2048 | 2.872 | torch.bfloat16 | 8.97 | 241.40 | 3.78 |
| 512 | 384 | 6 | 2048 | 1024 | 2.872 | torch.bfloat16 | 0.73 | 52.87 | 6.62 |
| 2048 | 384 | 6 | 2048 | 1024 | 2.872 | torch.bfloat16 | 0.92 | 168.78 | 5.29 |
| 4096 | 384 | 6 | 2048 | 1024 | 2.872 | torch.bfloat16 | 1.30 | 237.42 | 3.74 |

### vllm

| num_tokens | num_experts | top_k | hidden_size | ffn_size | routed_scaling_factor | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 384 | 6 | 7168 | 2048 | 2.872 | torch.bfloat16 | 3.84 | 70.38 | 8.80 |
| 2048 | 384 | 6 | 7168 | 2048 | 2.872 | torch.bfloat16 | 6.34 | 170.62 | 5.34 |
| 4096 | 384 | 6 | 7168 | 2048 | 2.872 | torch.bfloat16 | 9.24 | 234.37 | 3.67 |
| 512 | 384 | 6 | 2048 | 1024 | 2.872 | torch.bfloat16 | 0.67 | 57.39 | 7.18 |
| 2048 | 384 | 6 | 2048 | 1024 | 2.872 | torch.bfloat16 | 0.92 | 168.82 | 5.29 |
| 4096 | 384 | 6 | 2048 | 1024 | 2.872 | torch.bfloat16 | 1.36 | 227.27 | 3.58 |

## KimiMoEPaddedOp

### tileops-padded

| num_tokens | num_experts | top_k | hidden_size | ffn_size | routed_scaling_factor | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 384 | 6 | 7168 | 2048 | 2.872 | torch.bfloat16 | 5.22 | 51.87 | 6.49 |
| 2048 | 384 | 6 | 7168 | 2048 | 2.872 | torch.bfloat16 | 8.27 | 130.91 | 4.10 |
| 4096 | 384 | 6 | 7168 | 2048 | 2.872 | torch.bfloat16 | 9.97 | 217.13 | 3.40 |
| 512 | 384 | 6 | 2048 | 1024 | 2.872 | torch.bfloat16 | 1.05 | 36.98 | 4.63 |
| 2048 | 384 | 6 | 2048 | 1024 | 2.872 | torch.bfloat16 | 1.33 | 116.21 | 3.64 |
| 4096 | 384 | 6 | 2048 | 1024 | 2.872 | torch.bfloat16 | 1.80 | 171.69 | 2.70 |

## MoePermutePaddedOp

### tileops

| total_tokens | top_k | num_experts | hidden_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 2 | 8 | 4096 | torch.bfloat16 | 0.07 | 0.00 | 0.18 |
| 2048 | 2 | 8 | 4096 | torch.bfloat16 | 0.08 | 0.00 | 0.63 |
| 4096 | 2 | 8 | 4096 | torch.bfloat16 | 0.08 | 0.00 | 1.32 |
| 512 | 8 | 256 | 7168 | torch.bfloat16 | 0.14 | 0.00 | 0.46 |
| 2048 | 8 | 256 | 7168 | torch.bfloat16 | 0.32 | 0.00 | 0.83 |
| 4096 | 8 | 256 | 7168 | torch.bfloat16 | 0.57 | 0.00 | 0.93 |
| 512 | 8 | 128 | 2048 | torch.bfloat16 | 0.08 | 0.00 | 0.25 |
| 2048 | 8 | 128 | 2048 | torch.bfloat16 | 0.11 | 0.00 | 0.68 |
| 4096 | 8 | 128 | 2048 | torch.bfloat16 | 0.20 | 0.00 | 0.75 |

### torch-ref

| total_tokens | top_k | num_experts | hidden_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 2 | 8 | 4096 | torch.bfloat16 | 13.46 | 0.00 | 0.00 |
| 2048 | 2 | 8 | 4096 | torch.bfloat16 | 48.78 | 0.00 | 0.00 |
| 4096 | 2 | 8 | 4096 | torch.bfloat16 | 98.96 | 0.00 | 0.00 |
| 512 | 8 | 256 | 7168 | torch.bfloat16 | 48.75 | 0.00 | 0.00 |
| 2048 | 8 | 256 | 7168 | torch.bfloat16 | 198.77 | 0.00 | 0.00 |
| 4096 | 8 | 256 | 7168 | torch.bfloat16 | 407.14 | 0.00 | 0.00 |
| 512 | 8 | 128 | 2048 | torch.bfloat16 | 48.90 | 0.00 | 0.00 |
| 2048 | 8 | 128 | 2048 | torch.bfloat16 | 202.90 | 0.00 | 0.00 |
| 4096 | 8 | 128 | 2048 | torch.bfloat16 | 402.06 | 0.00 | 0.00 |

## MoePermuteAlignOp

### tileops

| total_tokens | top_k | num_experts | block_size | numel | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 2 | 8 | 16 | 1024 | 0.04 | N/A | 0.00 |
| 2048 | 2 | 8 | 16 | 4096 | 0.04 | N/A | 0.00 |
| 4096 | 2 | 8 | 16 | 8192 | 0.05 | N/A | 0.00 |
| 512 | 6 | 64 | 64 | 3072 | 0.05 | N/A | 0.00 |
| 2048 | 6 | 64 | 64 | 12288 | 0.04 | N/A | 0.00 |
| 4096 | 6 | 64 | 128 | 24576 | 0.04 | N/A | 0.01 |
| 8192 | 6 | 64 | 128 | 49152 | 0.05 | N/A | 0.01 |
| 2048 | 6 | 256 | 128 | 12288 | 0.05 | N/A | 0.00 |
| 8192 | 6 | 256 | 128 | 49152 | 0.05 | N/A | 0.01 |

### triton

| total_tokens | top_k | num_experts | block_size | numel | max_num_blocks | max_padded | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 2 | 8 | 16 | 1024 | 73 | 1159 | 0.11 | N/A | 0.00 |
| 2048 | 2 | 8 | 16 | 4096 | 265 | 4231 | 0.12 | N/A | 0.00 |
| 4096 | 2 | 8 | 16 | 8192 | 521 | 8327 | 0.22 | N/A | 0.00 |
| 512 | 6 | 64 | 64 | 3072 | 112 | 7167 | 0.11 | N/A | 0.00 |
| 2048 | 6 | 64 | 64 | 12288 | 256 | 16383 | 0.11 | N/A | 0.00 |
| 4096 | 6 | 64 | 128 | 24576 | 257 | 32831 | 0.12 | N/A | 0.00 |
| 8192 | 6 | 64 | 128 | 49152 | 449 | 57407 | 0.18 | N/A | 0.00 |
| 2048 | 6 | 256 | 128 | 12288 | 351 | 44927 | 0.11 | N/A | 0.00 |
| 8192 | 6 | 256 | 128 | 49152 | 639 | 81791 | 0.12 | N/A | 0.00 |

## Qwen3MoENopadOp

### tileops

| num_tokens | num_experts | top_k | hidden_size | ffn_size | scoring_func | renormalize | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 128 | 8 | 2048 | 1024 | softmax | False | torch.bfloat16 | 0.50 | 103.67 | 3.25 |
| 2048 | 128 | 8 | 2048 | 1024 | softmax | False | torch.bfloat16 | 0.81 | 253.80 | 2.00 |
| 4096 | 128 | 8 | 2048 | 1024 | softmax | False | torch.bfloat16 | 1.29 | 318.90 | 1.27 |
| 512 | 256 | 8 | 2048 | 1024 | softmax | True | torch.bfloat16 | 0.89 | 57.83 | 3.62 |
| 2048 | 256 | 8 | 2048 | 1024 | softmax | True | torch.bfloat16 | 1.08 | 190.23 | 2.99 |
| 4096 | 256 | 8 | 2048 | 1024 | softmax | True | torch.bfloat16 | 1.55 | 266.08 | 2.10 |
| 512 | 256 | 8 | 2048 | 1024 | sigmoid | True | torch.bfloat16 | 0.89 | 57.76 | 3.61 |
| 2048 | 256 | 8 | 2048 | 1024 | sigmoid | True | torch.bfloat16 | 1.08 | 190.41 | 2.99 |
| 4096 | 256 | 8 | 2048 | 1024 | sigmoid | True | torch.bfloat16 | 1.56 | 263.68 | 2.08 |

### torch

| num_tokens | num_experts | top_k | hidden_size | ffn_size | scoring_func | renormalize | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 128 | 8 | 2048 | 1024 | softmax | False | torch.bfloat16 | 35.32 | 1.46 | 0.05 |
| 2048 | 128 | 8 | 2048 | 1024 | softmax | False | torch.bfloat16 | 37.68 | 5.47 | 0.04 |
| 4096 | 128 | 8 | 2048 | 1024 | softmax | False | torch.bfloat16 | 39.64 | 10.40 | 0.04 |
| 512 | 256 | 8 | 2048 | 1024 | softmax | True | torch.bfloat16 | 71.94 | 0.72 | 0.04 |
| 2048 | 256 | 8 | 2048 | 1024 | softmax | True | torch.bfloat16 | 72.14 | 2.86 | 0.04 |
| 4096 | 256 | 8 | 2048 | 1024 | softmax | True | torch.bfloat16 | 76.51 | 5.39 | 0.04 |
| 512 | 256 | 8 | 2048 | 1024 | sigmoid | True | torch.bfloat16 | 73.22 | 0.70 | 0.04 |
| 2048 | 256 | 8 | 2048 | 1024 | sigmoid | True | torch.bfloat16 | 74.56 | 2.77 | 0.04 |
| 4096 | 256 | 8 | 2048 | 1024 | sigmoid | True | torch.bfloat16 | 77.41 | 5.33 | 0.04 |

### vllm

| num_tokens | num_experts | top_k | hidden_size | ffn_size | scoring_func | renormalize | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 128 | 8 | 2048 | 1024 | softmax | False | torch.bfloat16 | 0.60 | 85.68 | 2.68 |
| 2048 | 128 | 8 | 2048 | 1024 | softmax | False | torch.bfloat16 | 1.31 | 157.88 | 1.25 |
| 4096 | 128 | 8 | 2048 | 1024 | softmax | False | torch.bfloat16 | 1.33 | 310.80 | 1.24 |
| 512 | 256 | 8 | 2048 | 1024 | softmax | True | torch.bfloat16 | 0.84 | 61.17 | 3.83 |
| 2048 | 256 | 8 | 2048 | 1024 | softmax | True | torch.bfloat16 | 1.05 | 196.61 | 3.09 |
| 4096 | 256 | 8 | 2048 | 1024 | softmax | True | torch.bfloat16 | 1.46 | 283.08 | 2.23 |

## Qwen3MoEPaddedOp

### tileops-padded

| num_tokens | num_experts | top_k | hidden_size | ffn_size | scoring_func | renormalize | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 128 | 8 | 2048 | 1024 | softmax | False | torch.bfloat16 | 0.63 | 81.66 | 2.56 |
| 2048 | 128 | 8 | 2048 | 1024 | softmax | False | torch.bfloat16 | 1.03 | 200.65 | 1.58 |
| 4096 | 128 | 8 | 2048 | 1024 | softmax | False | torch.bfloat16 | 1.65 | 249.43 | 0.99 |
| 512 | 256 | 8 | 2048 | 1024 | softmax | True | torch.bfloat16 | 1.22 | 42.12 | 2.64 |
| 2048 | 256 | 8 | 2048 | 1024 | softmax | True | torch.bfloat16 | 1.48 | 139.45 | 2.19 |
| 4096 | 256 | 8 | 2048 | 1024 | softmax | True | torch.bfloat16 | 2.04 | 201.86 | 1.59 |
| 512 | 256 | 8 | 2048 | 1024 | sigmoid | True | torch.bfloat16 | 1.23 | 42.00 | 2.63 |
| 2048 | 256 | 8 | 2048 | 1024 | sigmoid | True | torch.bfloat16 | 1.48 | 139.16 | 2.19 |
| 4096 | 256 | 8 | 2048 | 1024 | sigmoid | True | torch.bfloat16 | 2.03 | 203.35 | 1.61 |

## moe_unpermute

### tileops

| total_tokens | top_k | hidden_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 512 | 2 | 4096 | torch.bfloat16 | 0.02 | 0.39 | 0.58 |
| 2048 | 2 | 4096 | torch.bfloat16 | 0.03 | 1.24 | 1.87 |
| 4096 | 2 | 4096 | torch.bfloat16 | 0.03 | 1.96 | 2.95 |
| 512 | 8 | 7168 | torch.bfloat16 | 0.03 | 1.99 | 2.24 |
| 2048 | 8 | 7168 | torch.bfloat16 | 0.08 | 2.96 | 3.33 |
| 4096 | 8 | 7168 | torch.bfloat16 | 0.14 | 3.32 | 3.74 |
| 512 | 8 | 2048 | torch.bfloat16 | 0.02 | 0.73 | 0.83 |
| 2048 | 8 | 2048 | torch.bfloat16 | 0.03 | 2.37 | 2.67 |
| 4096 | 8 | 2048 | torch.bfloat16 | 0.05 | 2.81 | 3.17 |

### torch-ref

| total_tokens | top_k | hidden_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 512 | 2 | 4096 | torch.bfloat16 | 0.12 | 0.07 | 0.10 |
| 2048 | 2 | 4096 | torch.bfloat16 | 0.23 | 0.15 | 0.22 |
| 4096 | 2 | 4096 | torch.bfloat16 | 0.42 | 0.16 | 0.24 |
| 512 | 8 | 7168 | torch.bfloat16 | 0.34 | 0.17 | 0.19 |
| 2048 | 8 | 7168 | torch.bfloat16 | 1.24 | 0.19 | 0.21 |
| 4096 | 8 | 7168 | torch.bfloat16 | 2.44 | 0.19 | 0.22 |
| 512 | 8 | 2048 | torch.bfloat16 | 0.13 | 0.13 | 0.15 |
| 2048 | 8 | 2048 | torch.bfloat16 | 0.39 | 0.17 | 0.20 |
| 4096 | 8 | 2048 | torch.bfloat16 | 0.73 | 0.18 | 0.21 |

## SumOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | sum | 0.06 | 0.07 | 0.14 |
| 1024 | 4096 | torch.bfloat16 | sum | 0.06 | 0.07 | 0.14 |
| 4096 | 4096 | torch.float16 | sum | 0.06 | 0.28 | 0.55 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | sum | 0.04 | 0.10 | 0.20 |
| 1024 | 4096 | torch.bfloat16 | sum | 0.04 | 0.10 | 0.21 |
| 4096 | 4096 | torch.float16 | sum | 0.09 | 0.20 | 0.39 |

## MeanOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | mean | 0.06 | 0.07 | 0.14 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | mean | 0.04 | 0.10 | 0.21 |

## AmaxOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | amax | 0.06 | 0.07 | 0.15 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | amax | 0.04 | 0.11 | 0.21 |

## AminOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | amin | 0.06 | 0.07 | 0.14 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | amin | 0.04 | 0.11 | 0.21 |

## ProdOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | prod | 0.06 | 0.07 | 0.14 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | prod | 0.04 | 0.11 | 0.21 |

## StdOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | std | 0.06 | 0.20 | 0.14 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | std | 0.05 | 0.27 | 0.18 |

## VarOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | var | 0.06 | 0.20 | 0.13 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | var | 0.05 | 0.27 | 0.18 |

## VarMeanOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | var_mean | 0.07 | 0.17 | 0.12 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | var_mean | 0.07 | 0.19 | 0.12 |

## RmsNormOp

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.06 | 0.27 | 0.27 |
| 4096 | 4096 | torch.bfloat16 | 0.07 | 0.95 | 0.95 |
| 2048 | 5120 | torch.float16 | 0.06 | 0.68 | 0.68 |
| 1025 | 4096 | torch.float16 | 0.06 | 0.27 | 0.27 |

### torch-ref

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.11 | 0.15 | 0.15 |
| 4096 | 4096 | torch.bfloat16 | 0.27 | 0.25 | 0.25 |
| 2048 | 5120 | torch.float16 | 0.19 | 0.22 | 0.22 |
| 1025 | 4096 | torch.float16 | 0.10 | 0.16 | 0.16 |

## RopeNeoxOp

### tileops

| seq_len | head_dim | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 2048 | 64 | torch.float16 | 0.05 | 0.01 | 0.02 |
| 2048 | 64 | torch.bfloat16 | 0.05 | 0.01 | 0.02 |
| 2048 | 64 | torch.float32 | 0.05 | 0.01 | 0.03 |
| 2048 | 128 | torch.float16 | 0.05 | 0.02 | 0.03 |
| 2048 | 128 | torch.bfloat16 | 0.05 | 0.02 | 0.03 |
| 2048 | 128 | torch.float32 | 0.05 | 0.02 | 0.07 |
| 4096 | 128 | torch.float16 | 0.05 | 0.04 | 0.07 |
| 4096 | 128 | torch.bfloat16 | 0.05 | 0.04 | 0.07 |
| 4096 | 128 | torch.float32 | 0.05 | 0.04 | 0.13 |

### torch-ref

| seq_len | head_dim | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 2048 | 64 | torch.float16 | 0.06 | 0.01 | 0.01 |
| 2048 | 64 | torch.bfloat16 | 0.06 | 0.01 | 0.01 |
| 2048 | 64 | torch.float32 | 0.06 | 0.01 | 0.03 |
| 2048 | 128 | torch.float16 | 0.06 | 0.02 | 0.03 |
| 2048 | 128 | torch.bfloat16 | 0.06 | 0.02 | 0.02 |
| 2048 | 128 | torch.float32 | 0.06 | 0.02 | 0.05 |
| 4096 | 128 | torch.float16 | 0.06 | 0.03 | 0.05 |
| 4096 | 128 | torch.bfloat16 | 0.06 | 0.03 | 0.05 |
| 4096 | 128 | torch.float32 | 0.06 | 0.03 | 0.10 |

## SoftmaxOp

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.06 | 0.28 | 0.28 |
| 4096 | 4096 | torch.bfloat16 | 0.07 | 1.02 | 1.02 |
| 1024 | 3000 | torch.float16 | 0.10 | 0.13 | 0.13 |
| 1025 | 4096 | torch.float16 | 0.06 | 0.28 | 0.28 |

### torch

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.02 | 0.78 | 0.78 |
| 4096 | 4096 | torch.bfloat16 | 0.06 | 1.04 | 1.04 |
| 1024 | 3000 | torch.float16 | 0.02 | 0.64 | 0.64 |
| 1025 | 4096 | torch.float16 | 0.02 | 0.78 | 0.78 |

## LogSoftmaxOp

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.06 | 0.35 | 0.28 |
| 4096 | 4096 | torch.bfloat16 | 0.08 | 1.00 | 0.80 |
| 1024 | 3000 | torch.float16 | 0.09 | 0.16 | 0.13 |
| 1025 | 4096 | torch.float16 | 0.06 | 0.36 | 0.28 |

### torch

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.02 | 1.07 | 0.85 |
| 4096 | 4096 | torch.bfloat16 | 0.06 | 1.45 | 1.16 |
| 1024 | 3000 | torch.float16 | 0.02 | 0.83 | 0.66 |
| 1025 | 4096 | torch.float16 | 0.02 | 1.06 | 0.85 |

## LogSumExpOp

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.06 | 0.22 | 0.15 |
| 4096 | 4096 | torch.bfloat16 | 0.06 | 0.79 | 0.53 |
| 1024 | 3000 | torch.float16 | 0.09 | 0.11 | 0.07 |
| 1025 | 4096 | torch.float16 | 0.06 | 0.22 | 0.14 |

### torch

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.08 | 0.15 | 0.10 |
| 4096 | 4096 | torch.bfloat16 | 0.12 | 0.43 | 0.29 |
| 1024 | 3000 | torch.float16 | 0.08 | 0.11 | 0.07 |
| 1025 | 4096 | torch.float16 | 0.08 | 0.15 | 0.10 |

## SsdChunkScanFwdOp

### tileops

| batch | num_chunks | chunk_len | n_heads | d_head | d_state | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 64 | 4 | 64 | 32 | torch.float16 | 105.93 | 0.00 | 0.00 |
| 2 | 4 | 64 | 8 | 64 | 64 | torch.float16 | 104.94 | 0.00 | 0.00 |
| 1 | 2 | 128 | 4 | 128 | 32 | torch.bfloat16 | 104.61 | 0.00 | 0.00 |
| 2 | 2 | 64 | 4 | 64 | 32 | torch.bfloat16 | 105.15 | 0.00 | 0.00 |

## ssd_chunk_scan_fwd

### baseline

| batch | num_chunks | chunk_len | n_heads | d_head | d_state | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 64 | 4 | 64 | 32 | torch.float16 | 0.33 | 0.01 | 0.00 |
| 2 | 4 | 64 | 8 | 64 | 64 | torch.float16 | 0.33 | 0.15 | 0.01 |
| 1 | 2 | 128 | 4 | 128 | 32 | torch.bfloat16 | 0.38 | 0.07 | 0.00 |
| 2 | 2 | 64 | 4 | 64 | 32 | torch.bfloat16 | 0.33 | 0.03 | 0.00 |

## SsdChunkStateFwdOp

### tileops

| batch | num_chunks | chunk_len | n_heads | d_head | d_state | n_groups | dtype | has_seq_idx | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 64 | 4 | 64 | 32 | 1 | torch.float16 | False | 92.27 | 0.00 | 0.00 |
| 2 | 4 | 64 | 8 | 64 | 64 | 2 | torch.float16 | False | 89.70 | 0.00 | 0.00 |
| 1 | 2 | 128 | 4 | 128 | 32 | 1 | torch.bfloat16 | False | 90.79 | 0.00 | 0.00 |
| 2 | 2 | 64 | 4 | 64 | 32 | 2 | torch.bfloat16 | False | 89.32 | 0.00 | 0.00 |
| 2 | 4 | 64 | 8 | 64 | 64 | 2 | torch.float16 | True | 89.62 | 0.00 | 0.00 |

## ssd_chunk_state_fwd

### baseline

| batch | num_chunks | chunk_len | n_heads | d_head | d_state | n_groups | dtype | has_seq_idx | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 64 | 4 | 64 | 32 | 1 | torch.float16 | False | 0.19 | 0.01 | 0.00 |
| 2 | 4 | 64 | 8 | 64 | 64 | 2 | torch.float16 | False | 0.21 | 0.16 | 0.01 |
| 1 | 2 | 128 | 4 | 128 | 32 | 1 | torch.bfloat16 | False | 0.19 | 0.04 | 0.00 |
| 2 | 2 | 64 | 4 | 64 | 32 | 2 | torch.bfloat16 | False | 0.21 | 0.02 | 0.00 |
| 2 | 4 | 64 | 8 | 64 | 64 | 2 | torch.float16 | True | 0.28 | 0.12 | 0.01 |

## SsdStatePassingFwdOp

### tileops

| batch | num_chunks | n_heads | d_state | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 4 | 32 | torch.float16 | 51.25 | 0.00 | 0.00 |
| 2 | 4 | 8 | 64 | torch.float16 | 54.69 | 0.00 | 0.00 |
| 1 | 2 | 4 | 32 | torch.bfloat16 | 51.20 | 0.00 | 0.00 |
| 2 | 4 | 8 | 64 | torch.bfloat16 | 55.35 | 0.00 | 0.00 |

## ssd_state_passing_fwd

### baseline

| batch | num_chunks | n_heads | d_state | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 4 | 32 | torch.float16 | 0.15 | 0.00 | 0.00 |
| 2 | 4 | 8 | 64 | torch.float16 | 0.27 | 0.00 | 0.00 |
| 1 | 2 | 4 | 32 | torch.bfloat16 | 0.22 | 0.00 | 0.00 |
| 2 | 4 | 8 | 64 | torch.bfloat16 | 0.33 | 0.00 | 0.00 |

## TopkSelectorOp

### tileops

| batch | seq_len | seq_len_kv | kv_group | topk | in_dtype | out_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32768 | 65536 | 1 | 1024 | torch.float32 | torch.int32 | 13.56 | N/A | 0.64 |
| 1 | 32768 | 65536 | 1 | 2048 | torch.float32 | torch.int32 | 14.02 | N/A | 0.63 |
| 1 | 65535 | 131072 | 1 | 1024 | torch.float32 | torch.int32 | 44.62 | N/A | 0.78 |
| 1 | 65535 | 131072 | 1 | 2048 | torch.float32 | torch.int32 | 45.04 | N/A | 0.77 |

### torch

| batch | seq_len | seq_len_kv | kv_group | topk | in_dtype | out_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32768 | 65536 | 1 | 1024 | torch.float32 | torch.int32 | 28.49 | N/A | 0.31 |
| 1 | 32768 | 65536 | 1 | 2048 | torch.float32 | torch.int32 | 29.83 | N/A | 0.30 |
| 1 | 65535 | 131072 | 1 | 1024 | torch.float32 | torch.int32 | 109.22 | N/A | 0.32 |
| 1 | 65535 | 131072 | 1 | 2048 | torch.float32 | torch.int32 | 112.01 | N/A | 0.31 |

## exp

### tileops

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| exp | 262144 | torch.float16 | torch.float16 | 0.05 | 0.01 | 0.02 |
| exp | 1048576 | torch.float16 | torch.float16 | 0.05 | 0.02 | 0.08 |
| exp | 4000000 | torch.float16 | torch.float16 | 0.05 | 0.08 | 0.34 |
| exp | 262144 | torch.bfloat16 | torch.bfloat16 | 0.05 | 0.01 | 0.02 |
| exp | 1048576 | torch.bfloat16 | torch.bfloat16 | 0.05 | 0.02 | 0.09 |
| exp | 4000000 | torch.bfloat16 | torch.bfloat16 | 0.05 | 0.08 | 0.32 |

### torch

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| exp | 262144 | torch.float16 | torch.float16 | 0.01 | 0.02 | 0.07 |
| exp | 1048576 | torch.float16 | torch.float16 | 0.02 | 0.07 | 0.27 |
| exp | 4000000 | torch.float16 | torch.float16 | 0.02 | 0.25 | 1.02 |
| exp | 262144 | torch.bfloat16 | torch.bfloat16 | 0.02 | 0.02 | 0.06 |
| exp | 1048576 | torch.bfloat16 | torch.bfloat16 | 0.02 | 0.06 | 0.25 |
| exp | 4000000 | torch.bfloat16 | torch.bfloat16 | 0.02 | 0.24 | 0.96 |

## gelu

### tileops

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| gelu | 262144 | torch.float16 | torch.float16 | 0.05 | 0.01 | 0.02 |
| gelu | 1048576 | torch.float16 | torch.float16 | 0.05 | 0.02 | 0.09 |
| gelu | 4000000 | torch.float16 | torch.float16 | 0.05 | 0.08 | 0.33 |
| gelu | 262144 | torch.bfloat16 | torch.bfloat16 | 0.05 | 0.01 | 0.02 |
| gelu | 1048576 | torch.bfloat16 | torch.bfloat16 | 0.05 | 0.02 | 0.09 |
| gelu | 4000000 | torch.bfloat16 | torch.bfloat16 | 0.05 | 0.09 | 0.34 |

### torch

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| gelu | 262144 | torch.float16 | torch.float16 | 0.02 | 0.02 | 0.07 |
| gelu | 1048576 | torch.float16 | torch.float16 | 0.02 | 0.07 | 0.27 |
| gelu | 4000000 | torch.float16 | torch.float16 | 0.02 | 0.21 | 0.84 |
| gelu | 262144 | torch.bfloat16 | torch.bfloat16 | 0.02 | 0.02 | 0.07 |
| gelu | 1048576 | torch.bfloat16 | torch.bfloat16 | 0.02 | 0.06 | 0.26 |
| gelu | 4000000 | torch.bfloat16 | torch.bfloat16 | 0.02 | 0.24 | 0.95 |

## logical_not

### tileops

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| logical_not | 262144 | torch.float16 | torch.bool | 0.05 | 0.01 | 0.02 |
| logical_not | 1048576 | torch.float16 | torch.bool | 0.05 | 0.02 | 0.06 |
| logical_not | 4000000 | torch.float16 | torch.bool | 0.05 | 0.08 | 0.23 |

### torch

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| logical_not | 262144 | torch.float16 | torch.bool | 0.02 | 0.01 | 0.04 |
| logical_not | 1048576 | torch.float16 | torch.bool | 0.02 | 0.06 | 0.18 |
| logical_not | 4000000 | torch.float16 | torch.bool | 0.02 | 0.20 | 0.61 |

## bitwise_not

### tileops

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| bitwise_not | 262144 | torch.int32 | torch.int32 | 0.05 | 0.01 | 0.04 |
| bitwise_not | 1048576 | torch.int32 | torch.int32 | 0.05 | 0.02 | 0.17 |
| bitwise_not | 4000000 | torch.int32 | torch.int32 | 0.05 | 0.08 | 0.68 |

### torch

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| bitwise_not | 262144 | torch.int32 | torch.int32 | 0.02 | 0.01 | 0.11 |
| bitwise_not | 1048576 | torch.int32 | torch.int32 | 0.02 | 0.05 | 0.44 |
| bitwise_not | 4000000 | torch.int32 | torch.int32 | 0.02 | 0.25 | 2.03 |

## isnan

### tileops

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| isnan | 262144 | torch.float16 | torch.bool | 0.05 | 0.01 | 0.02 |
| isnan | 1048576 | torch.float16 | torch.bool | 0.05 | 0.02 | 0.06 |
| isnan | 4000000 | torch.float16 | torch.bool | 0.05 | 0.08 | 0.25 |

### torch

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| isnan | 262144 | torch.float16 | torch.bool | 0.01 | 0.02 | 0.05 |
| isnan | 1048576 | torch.float16 | torch.bool | 0.02 | 0.06 | 0.18 |
| isnan | 4000000 | torch.float16 | torch.bool | 0.02 | 0.24 | 0.72 |

## unary_strategy

### relu_direct

| n_total | shape_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4194304 | 1024x4096 | torch.float16 | direct | 0.05 | 0.09 | 0.35 |
| 4194304 | 1024x4096 | torch.bfloat16 | direct | 0.05 | 0.09 | 0.34 |
| 4194304 | 1024x4096 | torch.float32 | direct | 0.05 | 0.09 | 0.69 |
| 10485760 | 1024x10240 | torch.float16 | direct | 0.06 | 0.18 | 0.73 |
| 10485760 | 1024x10240 | torch.bfloat16 | direct | 0.06 | 0.18 | 0.72 |
| 10485760 | 1024x10240 | torch.float32 | direct | 0.06 | 0.18 | 1.44 |
| 20971520 | 1024x20480 | torch.float16 | direct | 0.07 | 0.30 | 1.21 |
| 20971520 | 1024x20480 | torch.bfloat16 | direct | 0.07 | 0.30 | 1.21 |
| 20971520 | 1024x20480 | torch.float32 | direct | 0.08 | 0.28 | 2.21 |

### relu_explicit_parallel

| n_total | shape_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4194304 | 1024x4096 | torch.float16 | explicit_parallel | 0.05 | 0.08 | 0.34 |
| 4194304 | 1024x4096 | torch.bfloat16 | explicit_parallel | 0.05 | 0.08 | 0.34 |
| 4194304 | 1024x4096 | torch.float32 | explicit_parallel | 0.05 | 0.09 | 0.69 |
| 10485760 | 1024x10240 | torch.float16 | explicit_parallel | 0.05 | 0.21 | 0.83 |
| 10485760 | 1024x10240 | torch.bfloat16 | explicit_parallel | 0.05 | 0.22 | 0.87 |
| 10485760 | 1024x10240 | torch.float32 | explicit_parallel | 0.05 | 0.20 | 1.63 |
| 20971520 | 1024x20480 | torch.float16 | explicit_parallel | 0.05 | 0.39 | 1.56 |
| 20971520 | 1024x20480 | torch.bfloat16 | explicit_parallel | 0.05 | 0.39 | 1.57 |
| 20971520 | 1024x20480 | torch.float32 | explicit_parallel | 0.05 | 0.38 | 3.07 |

### relu_register_copy

| n_total | shape_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4194304 | 1024x4096 | torch.float16 | register_copy | 0.05 | 0.08 | 0.34 |
| 4194304 | 1024x4096 | torch.bfloat16 | register_copy | 0.05 | 0.08 | 0.33 |
| 4194304 | 1024x4096 | torch.float32 | register_copy | 0.05 | 0.09 | 0.70 |
| 10485760 | 1024x10240 | torch.float16 | register_copy | 0.08 | 0.12 | 0.50 |
| 10485760 | 1024x10240 | torch.bfloat16 | register_copy | 0.05 | 0.21 | 0.83 |
| 10485760 | 1024x10240 | torch.float32 | register_copy | 0.05 | 0.20 | 1.61 |
| 20971520 | 1024x20480 | torch.float16 | register_copy | 0.05 | 0.39 | 1.58 |
| 20971520 | 1024x20480 | torch.bfloat16 | register_copy | 0.05 | 0.40 | 1.62 |
| 20971520 | 1024x20480 | torch.float32 | register_copy | 0.05 | 0.41 | 3.26 |

## L1NormOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | l1 | 0.06 | 0.07 | 0.14 |
| 1024 | 4096 | torch.bfloat16 | l1 | 0.06 | 0.07 | 0.14 |
| 1024 | 4096 | torch.float32 | l1 | 0.06 | 0.07 | 0.27 |
| 4096 | 4096 | torch.float16 | l1 | 0.07 | 0.25 | 0.51 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | l1 | 0.02 | 0.19 | 0.38 |
| 1024 | 4096 | torch.bfloat16 | l1 | 0.02 | 0.18 | 0.37 |
| 1024 | 4096 | torch.float32 | l1 | 0.02 | 0.18 | 0.72 |
| 4096 | 4096 | torch.float16 | l1 | 0.03 | 0.51 | 1.03 |

## L2NormOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | l2 | 0.07 | 0.06 | 0.13 |
| 1024 | 4096 | torch.bfloat16 | l2 | 0.06 | 0.07 | 0.14 |
| 1024 | 4096 | torch.float32 | l2 | 0.06 | 0.07 | 0.29 |
| 4096 | 4096 | torch.float16 | l2 | 0.06 | 0.28 | 0.56 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | l2 | 0.03 | 0.14 | 0.28 |
| 1024 | 4096 | torch.bfloat16 | l2 | 0.02 | 0.19 | 0.39 |
| 1024 | 4096 | torch.float32 | l2 | 0.02 | 0.18 | 0.71 |
| 4096 | 4096 | torch.float16 | l2 | 0.03 | 0.64 | 1.28 |

## InfNormOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | inf | 0.13 | 0.03 | 0.07 |
| 1024 | 4096 | torch.bfloat16 | inf | 0.12 | 0.03 | 0.07 |
| 1024 | 4096 | torch.float32 | inf | 0.12 | 0.03 | 0.14 |
| 4096 | 4096 | torch.float16 | inf | 0.12 | 0.14 | 0.27 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | inf | 0.02 | 0.19 | 0.38 |
| 1024 | 4096 | torch.bfloat16 | inf | 0.02 | 0.19 | 0.38 |
| 1024 | 4096 | torch.float32 | inf | 0.02 | 0.18 | 0.71 |
| 4096 | 4096 | torch.float16 | inf | 0.03 | 0.63 | 1.26 |

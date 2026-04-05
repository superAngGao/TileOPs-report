........................................................................ [  7%]
........................................................................ [ 14%]
........................................................................ [ 21%]
........................................................................ [ 28%]
........................................................................ [ 36%]
........................................................................ [ 43%]
..............................xxx....................................... [ 50%]
........................................................................ [ 57%]
.......sss.............................................................. [ 65%]
........................................................................ [ 72%]
........................................................................ [ 79%]
........................................................................ [ 86%]
........................................................................ [ 94%]
...........................................................              [100%]Benchmark report saved to profile_run.log

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
989 passed, 3 skipped, 3 xfailed, 37 warnings in 1594.47s (0:26:34)

===== profile_run.log summary =====
# TileOPs Benchmark Report
Generated: 2026-04-01 19:43:07

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
| 4194304 | torch.bfloat16 | 0.01 | 0.67 | 2.68 |
| 4194304 | torch.float32 | 0.01 | 0.39 | 3.13 |

### torch

| n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 4096 | torch.float16 | 0.00 | 0.00 | 0.01 |
| 4194304 | torch.float16 | 0.01 | 0.49 | 1.96 |
| 4194304 | torch.bfloat16 | 0.01 | 0.50 | 1.99 |
| 4194304 | torch.float32 | 0.01 | 0.28 | 2.27 |

## r3_jit_cost

### relu_jit

| n_total | cold_ms | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 1000 | 18.79 | 0.00 | 0.00 | 0.00 |
| 2000 | 18.09 | 0.00 | 0.00 | 0.00 |
| 4000 | 18.08 | 0.00 | 0.00 | 0.01 |
| 8000 | 17.71 | 0.00 | 0.00 | 0.02 |
| 16000 | 17.86 | 0.00 | 0.01 | 0.04 |
| 32000 | 17.91 | 0.00 | 0.02 | 0.07 |
| 64000 | 17.87 | 0.00 | 0.03 | 0.14 |
| 128000 | 17.96 | 0.00 | 0.07 | 0.27 |
| 256000 | 17.95 | 0.00 | 0.12 | 0.49 |
| 512000 | 17.78 | 0.00 | 0.22 | 0.86 |

## r4_strategy_unary

### relu_direct

| n_total | size_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | torch.float16 | direct | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.bfloat16 | direct | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.float32 | direct | 0.00 | 0.00 | 0.02 |
| 4194304 | 4M | torch.float16 | direct | 0.01 | 0.29 | 1.17 |
| 4194304 | 4M | torch.bfloat16 | direct | 0.01 | 0.29 | 1.17 |
| 4194304 | 4M | torch.float32 | direct | 0.02 | 0.27 | 2.14 |
| 16777216 | 16M | torch.float16 | direct | 0.07 | 0.23 | 0.91 |
| 16777216 | 16M | torch.bfloat16 | direct | 0.07 | 0.23 | 0.91 |
| 16777216 | 16M | torch.float32 | direct | 0.08 | 0.20 | 1.61 |

### relu_explicit_parallel

| n_total | size_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | torch.float16 | explicit_parallel | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.bfloat16 | explicit_parallel | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.float32 | explicit_parallel | 0.00 | 0.00 | 0.02 |
| 4194304 | 4M | torch.float16 | explicit_parallel | 0.01 | 0.62 | 2.47 |
| 4194304 | 4M | torch.bfloat16 | explicit_parallel | 0.01 | 0.62 | 2.47 |
| 4194304 | 4M | torch.float32 | explicit_parallel | 0.01 | 0.38 | 3.06 |
| 16777216 | 16M | torch.float16 | explicit_parallel | 0.02 | 0.80 | 3.20 |
| 16777216 | 16M | torch.bfloat16 | explicit_parallel | 0.02 | 0.80 | 3.21 |
| 16777216 | 16M | torch.float32 | explicit_parallel | 0.04 | 0.39 | 3.09 |

### relu_register_copy

| n_total | size_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | torch.float16 | register_copy | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.bfloat16 | register_copy | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.float32 | register_copy | 0.00 | 0.00 | 0.02 |
| 4194304 | 4M | torch.float16 | register_copy | 0.01 | 0.66 | 2.65 |
| 4194304 | 4M | torch.bfloat16 | register_copy | 0.01 | 0.66 | 2.65 |
| 4194304 | 4M | torch.float32 | register_copy | 0.01 | 0.39 | 3.14 |
| 16777216 | 16M | torch.float16 | register_copy | 0.02 | 0.90 | 3.60 |
| 16777216 | 16M | torch.bfloat16 | register_copy | 0.02 | 0.90 | 3.60 |
| 16777216 | 16M | torch.float32 | register_copy | 0.04 | 0.39 | 3.14 |

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
| 16777216 | 16M | torch.float16 | direct | 0.08 | 0.20 | 0.80 |
| 16777216 | 16M | torch.bfloat16 | direct | 0.08 | 0.20 | 0.80 |
| 16777216 | 16M | torch.float32 | direct | 0.09 | 0.18 | 1.47 |

### gelu_explicit_parallel

| n_total | size_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | torch.float16 | explicit_parallel | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.bfloat16 | explicit_parallel | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.float32 | explicit_parallel | 0.00 | 0.00 | 0.02 |
| 4194304 | 4M | torch.float16 | explicit_parallel | 0.01 | 0.45 | 1.78 |
| 4194304 | 4M | torch.bfloat16 | explicit_parallel | 0.01 | 0.44 | 1.78 |
| 4194304 | 4M | torch.float32 | explicit_parallel | 0.01 | 0.38 | 3.06 |
| 16777216 | 16M | torch.float16 | explicit_parallel | 0.04 | 0.45 | 1.81 |
| 16777216 | 16M | torch.bfloat16 | explicit_parallel | 0.04 | 0.45 | 1.78 |
| 16777216 | 16M | torch.float32 | explicit_parallel | 0.05 | 0.37 | 2.92 |

### gelu_register_copy

| n_total | size_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | torch.float16 | register_copy | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.bfloat16 | register_copy | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.float32 | register_copy | 0.00 | 0.00 | 0.02 |
| 4194304 | 4M | torch.float16 | register_copy | 0.01 | 0.48 | 1.92 |
| 4194304 | 4M | torch.bfloat16 | register_copy | 0.01 | 0.47 | 1.88 |
| 4194304 | 4M | torch.float32 | register_copy | 0.01 | 0.39 | 3.11 |
| 16777216 | 16M | torch.float16 | register_copy | 0.03 | 0.54 | 2.16 |
| 16777216 | 16M | torch.bfloat16 | register_copy | 0.03 | 0.51 | 2.06 |
| 16777216 | 16M | torch.float32 | register_copy | 0.04 | 0.38 | 3.05 |

## r5_boundary

### relu_aligned

| n_total | align_label | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 2048000 | aligned | 0.00 | 0.46 | 1.86 |

### relu_unaligned_plus_1

| n_total | align_label | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 2048001 | unaligned_plus_1 | 0.01 | 0.30 | 1.20 |

### relu_unaligned_plus_127

| n_total | align_label | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 2048127 | unaligned_plus_127 | 0.01 | 0.30 | 1.19 |

## r6_threads

### relu_t128

| n_total | size_label | op_name | threads | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | relu | 128 | 0.00 | 0.00 | 0.01 |
| 4194304 | 4M | relu | 128 | 0.03 | 0.14 | 0.57 |
| 16777216 | 16M | relu | 128 | 0.12 | 0.15 | 0.58 |

### relu_t256

| n_total | size_label | op_name | threads | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | relu | 256 | 0.00 | 0.00 | 0.01 |
| 4194304 | 4M | relu | 256 | 0.03 | 0.14 | 0.58 |
| 16777216 | 16M | relu | 256 | 0.12 | 0.14 | 0.56 |

### erf_t128

| n_total | size_label | op_name | threads | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | erf | 128 | 0.00 | 0.00 | 0.01 |
| 4194304 | 4M | erf | 128 | 0.01 | 0.39 | 1.55 |
| 16777216 | 16M | erf | 128 | 0.04 | 0.39 | 1.55 |

### erf_t256

| n_total | size_label | op_name | threads | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | erf | 256 | 0.00 | 0.00 | 0.01 |
| 4194304 | 4M | erf | 256 | 0.01 | 0.39 | 1.55 |
| 16777216 | 16M | erf | 256 | 0.04 | 0.39 | 1.54 |

### mish_t128

| n_total | size_label | op_name | threads | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | mish | 128 | 0.00 | 0.00 | 0.01 |
| 4194304 | 4M | mish | 128 | 0.02 | 0.27 | 1.09 |
| 16777216 | 16M | mish | 128 | 0.07 | 0.25 | 1.01 |

### mish_t256

| n_total | size_label | op_name | threads | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | mish | 256 | 0.00 | 0.00 | 0.01 |
| 4194304 | 4M | mish | 256 | 0.02 | 0.26 | 1.05 |
| 16777216 | 16M | mish | 256 | 0.07 | 0.25 | 1.00 |

## r7_dtype_npt

### relu_fp32_npt4

| dtype_label | num_per_thread | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| fp32 | 4 | 0.00 | 0.21 | 1.65 |

### relu_fp32_npt8

| dtype_label | num_per_thread | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| fp32 | 8 | 0.00 | 0.20 | 1.63 |

### relu_fp16_npt4

| dtype_label | num_per_thread | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| fp16 | 4 | 0.00 | 0.26 | 1.06 |

### relu_fp16_npt8

| dtype_label | num_per_thread | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| fp16 | 8 | 0.00 | 0.20 | 0.80 |

## AdaLayerNormOp

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 1152 | torch.float16 | 0.93 | 0.01 | 0.01 |
| 1024 | 1152 | torch.bfloat16 | 1.12 | 0.01 | 0.01 |
| 2048 | 4096 | torch.float16 | 0.03 | 1.49 | 2.39 |
| 2048 | 4096 | torch.bfloat16 | 0.03 | 1.48 | 2.36 |
| 1 | 4096 | torch.bfloat16 | 0.01 | 0.00 | 0.01 |

### torch-ref

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 1152 | torch.float16 | 0.01 | 0.49 | 0.78 |
| 1024 | 1152 | torch.bfloat16 | 0.01 | 0.49 | 0.78 |
| 2048 | 4096 | torch.float16 | 0.05 | 0.79 | 1.27 |
| 2048 | 4096 | torch.bfloat16 | 0.05 | 0.79 | 1.26 |
| 1 | 4096 | torch.bfloat16 | 0.01 | 0.00 | 0.00 |

## AdaLayerNormZeroOp

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 1152 | torch.float16 | 1.12 | 0.01 | 0.01 |
| 1024 | 1152 | torch.bfloat16 | 1.32 | 0.01 | 0.01 |
| 2048 | 4096 | torch.float16 | 0.03 | 1.45 | 2.42 |
| 2048 | 4096 | torch.bfloat16 | 0.03 | 1.44 | 2.40 |
| 1 | 4096 | torch.bfloat16 | 0.01 | 0.00 | 0.01 |

### torch-ref

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 1152 | torch.float16 | 0.02 | 0.47 | 0.78 |
| 1024 | 1152 | torch.bfloat16 | 0.02 | 0.47 | 0.78 |
| 2048 | 4096 | torch.float16 | 0.07 | 0.73 | 1.21 |
| 2048 | 4096 | torch.bfloat16 | 0.07 | 0.72 | 1.20 |
| 1 | 4096 | torch.bfloat16 | 0.01 | 0.00 | 0.00 |

## ArgmaxOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | argmax | 6.10 | 0.00 | 0.00 |
| 1024 | 4096 | torch.bfloat16 | argmax | 6.11 | 0.00 | 0.00 |
| 4096 | 4096 | torch.float16 | argmax | 21.17 | 0.00 | 0.00 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | argmax | 0.03 | 0.12 | 0.25 |
| 1024 | 4096 | torch.bfloat16 | argmax | 0.03 | 0.12 | 0.25 |
| 4096 | 4096 | torch.float16 | argmax | 0.04 | 0.38 | 0.75 |

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
| 1024 | 4096 | torch.float16 | argmin | 0.03 | 0.15 | 0.31 |
| 1024 | 4096 | torch.bfloat16 | argmin | 0.03 | 0.12 | 0.25 |
| 4096 | 4096 | torch.float16 | argmin | 0.04 | 0.38 | 0.75 |

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
| 4 | 256 | (28, 28) | torch.float16 | 0.02 | 0.33 | 0.24 |
| 4 | 128 | (1024, 1024) | torch.float16 | 12.34 | 0.35 | 0.26 |

### torch-autograd

| N | C | spatial | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 64 | () | torch.float16 | 0.02 | 0.00 | 0.00 |
| 8 | 64 | (32, 32) | torch.float16 | 0.03 | 0.16 | 0.12 |
| 4 | 128 | (32, 32) | torch.float16 | 0.02 | 0.19 | 0.14 |
| 4 | 256 | (28, 28) | torch.float16 | 0.02 | 0.27 | 0.20 |
| 4 | 128 | (1024, 1024) | torch.float16 | 24.22 | 0.18 | 0.13 |

## r1_vectorization

### add_same_shape_1d

| pattern_name | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| same_shape_1d | 1000000 | 0.00 | 0.25 | 1.50 |

### torch-same_shape_1d

| pattern_name | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| same_shape_1d | 1000000 | 0.01 | 0.20 | 1.19 |

### add_bias_add_2d

| pattern_name | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| bias_add_2d | 1000000 | 0.00 | 0.29 | 1.16 |

### torch-bias_add_2d

| pattern_name | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| bias_add_2d | 1000000 | 0.01 | 0.13 | 0.52 |

### add_interleaved_3d

| pattern_name | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| interleaved_3d | 1048576 | 0.00 | 0.46 | 0.92 |

### torch-interleaved_3d

| pattern_name | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| interleaved_3d | 1048576 | 0.01 | 0.15 | 0.30 |

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
| 1M | same_shape | direct | 1048576 | 0.01 | 0.19 | 1.16 |
| 1M | same_shape | direct | 1048576 | 0.01 | 0.17 | 2.03 |
| 16M | same_shape | direct | 16777216 | 0.08 | 0.21 | 1.24 |
| 16M | same_shape | direct | 16777216 | 0.08 | 0.21 | 1.24 |
| 16M | same_shape | direct | 16777216 | 0.10 | 0.17 | 2.04 |

### add_direct_bias_add_2d

| size_label | pattern_name | strategy | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4K | bias_add_2d | direct | 4096 | 0.00 | 0.00 | 0.01 |
| 4K | bias_add_2d | direct | 4096 | 0.00 | 0.00 | 0.01 |
| 4K | bias_add_2d | direct | 4096 | 0.00 | 0.00 | 0.02 |
| 1M | bias_add_2d | direct | 1048576 | 0.01 | 0.21 | 0.83 |
| 1M | bias_add_2d | direct | 1048576 | 0.01 | 0.21 | 0.82 |
| 1M | bias_add_2d | direct | 1048576 | 0.01 | 0.19 | 1.55 |
| 16M | bias_add_2d | direct | 16777216 | 0.07 | 0.23 | 0.92 |
| 16M | bias_add_2d | direct | 16777216 | 0.07 | 0.23 | 0.92 |
| 16M | bias_add_2d | direct | 16777216 | 0.08 | 0.20 | 1.63 |

### add_direct_interleaved_3d

| size_label | pattern_name | strategy | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4K | interleaved_3d | direct | 4096 | 0.00 | 0.00 | 0.01 |
| 4K | interleaved_3d | direct | 4096 | 0.00 | 0.00 | 0.01 |
| 4K | interleaved_3d | direct | 4096 | 0.00 | 0.00 | 0.01 |
| 1M | interleaved_3d | direct | 1048576 | 0.00 | 0.24 | 0.49 |
| 1M | interleaved_3d | direct | 1048576 | 0.00 | 0.24 | 0.49 |
| 1M | interleaved_3d | direct | 1048576 | 0.00 | 0.23 | 0.94 |
| 16M | interleaved_3d | direct | 16777216 | 0.05 | 0.32 | 0.63 |
| 16M | interleaved_3d | direct | 16777216 | 0.05 | 0.32 | 0.63 |
| 16M | interleaved_3d | direct | 16777216 | 0.05 | 0.31 | 1.22 |

### add_explicit_parallel_same_shape

| size_label | pattern_name | strategy | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4K | same_shape | explicit_parallel | 4096 | 0.00 | 0.00 | 0.01 |
| 4K | same_shape | explicit_parallel | 4096 | 0.00 | 0.00 | 0.01 |
| 4K | same_shape | explicit_parallel | 4096 | 0.00 | 0.00 | 0.03 |
| 1M | same_shape | explicit_parallel | 1048576 | 0.00 | 0.27 | 1.61 |
| 1M | same_shape | explicit_parallel | 1048576 | 0.00 | 0.27 | 1.60 |
| 1M | same_shape | explicit_parallel | 1048576 | 0.01 | 0.19 | 2.31 |
| 16M | same_shape | explicit_parallel | 16777216 | 0.03 | 0.56 | 3.36 |
| 16M | same_shape | explicit_parallel | 16777216 | 0.03 | 0.56 | 3.37 |
| 16M | same_shape | explicit_parallel | 16777216 | 0.07 | 0.25 | 2.94 |

### add_explicit_parallel_bias_add_2d

| size_label | pattern_name | strategy | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4K | bias_add_2d | explicit_parallel | 4096 | 0.00 | 0.00 | 0.01 |
| 4K | bias_add_2d | explicit_parallel | 4096 | 0.00 | 0.00 | 0.01 |
| 4K | bias_add_2d | explicit_parallel | 4096 | 0.00 | 0.00 | 0.02 |
| 1M | bias_add_2d | explicit_parallel | 1048576 | 0.00 | 0.33 | 1.33 |
| 1M | bias_add_2d | explicit_parallel | 1048576 | 0.00 | 0.33 | 1.33 |
| 1M | bias_add_2d | explicit_parallel | 1048576 | 0.00 | 0.25 | 2.00 |
| 16M | bias_add_2d | explicit_parallel | 16777216 | 0.02 | 0.81 | 3.26 |
| 16M | bias_add_2d | explicit_parallel | 16777216 | 0.02 | 0.81 | 3.25 |
| 16M | bias_add_2d | explicit_parallel | 16777216 | 0.04 | 0.42 | 3.38 |

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
| 16M | interleaved_3d | explicit_parallel | 16777216 | 0.01 | 1.20 | 2.40 |
| 16M | interleaved_3d | explicit_parallel | 16777216 | 0.02 | 0.97 | 3.89 |

## r4_where

### tileops-where

| n_total | size_label | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 4096 | 4K | 0.00 | 0.00 | 0.02 |
| 1048576 | 1M | 0.00 | 0.25 | 1.73 |
| 16777216 | 16M | 0.04 | 0.47 | 3.30 |

### torch

| n_total | size_label | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 4096 | 4K | 0.00 | 0.00 | 0.01 |
| 1048576 | 1M | 0.01 | 0.20 | 1.41 |
| 16777216 | 16M | 0.05 | 0.33 | 2.32 |

## AddOp

### tileops

| n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 4194304 | torch.float16 | 0.01 | 0.48 | 2.88 |
| 4194304 | torch.bfloat16 | 0.01 | 0.48 | 2.87 |
| 4194304 | torch.float32 | 0.02 | 0.28 | 3.30 |

### torch

| n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 4194304 | torch.float16 | 0.01 | 0.34 | 2.02 |
| 4194304 | torch.bfloat16 | 0.01 | 0.34 | 2.02 |
| 4194304 | torch.float32 | 0.02 | 0.18 | 2.22 |

## sub

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| sub | torch.float16 | torch.float16 | 0.01 | 0.48 | 2.86 |
| sub | torch.float16 | torch.float16 | 0.02 | 0.58 | 3.47 |
| sub | torch.float16 | torch.float16 | 0.04 | 0.56 | 3.36 |

### torch

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| sub | torch.float16 | torch.float16 | 0.01 | 0.34 | 2.03 |
| sub | torch.float16 | torch.float16 | 0.03 | 0.37 | 2.22 |
| sub | torch.float16 | torch.float16 | 0.06 | 0.38 | 2.28 |

## mul

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| mul | torch.float16 | torch.float16 | 0.01 | 0.48 | 2.86 |
| mul | torch.float16 | torch.float16 | 0.02 | 0.57 | 3.45 |
| mul | torch.float16 | torch.float16 | 0.04 | 0.57 | 3.43 |

### torch

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| mul | torch.float16 | torch.float16 | 0.01 | 0.33 | 2.00 |
| mul | torch.float16 | torch.float16 | 0.03 | 0.36 | 2.19 |
| mul | torch.float16 | torch.float16 | 0.06 | 0.38 | 2.28 |

## div

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| div | torch.float16 | torch.float16 | 0.01 | 0.47 | 2.80 |
| div | torch.float16 | torch.float16 | 0.02 | 0.57 | 3.44 |
| div | torch.float16 | torch.float16 | 0.04 | 0.56 | 3.36 |

### torch

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| div | torch.float16 | torch.float16 | 0.01 | 0.32 | 1.90 |
| div | torch.float16 | torch.float16 | 0.03 | 0.36 | 2.17 |
| div | torch.float16 | torch.float16 | 0.06 | 0.37 | 2.25 |

## remainder

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| remainder | torch.float16 | torch.float16 | 0.01 | 0.46 | 2.77 |
| remainder | torch.float16 | torch.float16 | 0.02 | 0.57 | 3.43 |

### torch

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| remainder | torch.float16 | torch.float16 | 0.01 | 0.28 | 1.68 |
| remainder | torch.float16 | torch.float16 | 0.03 | 0.32 | 1.94 |

## pow

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| pow | torch.float16 | torch.float16 | 0.02 | 0.25 | 1.48 |
| pow | torch.float16 | torch.float16 | 0.04 | 0.23 | 1.40 |

### torch

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| pow | torch.float16 | torch.float16 | 0.03 | 0.16 | 0.96 |
| pow | torch.float16 | torch.float16 | 0.06 | 0.16 | 0.98 |

## floor_divide

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| floor_divide | torch.float16 | torch.float16 | 0.01 | 0.46 | 2.76 |
| floor_divide | torch.float16 | torch.float16 | 0.02 | 0.57 | 3.43 |

### torch

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| floor_divide | torch.float16 | torch.float16 | 0.04 | 0.12 | 0.71 |
| floor_divide | torch.float16 | torch.float16 | 0.09 | 0.12 | 0.72 |

## lerp

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| lerp | torch.float16 | torch.float16 | 0.01 | 0.47 | 2.84 |
| lerp | torch.float16 | torch.float16 | 0.02 | 0.58 | 3.48 |

### torch

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| lerp | torch.float16 | torch.float16 | 0.01 | 0.35 | 2.08 |
| lerp | torch.float16 | torch.float16 | 0.03 | 0.38 | 2.25 |

## maximum

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| maximum | torch.float16 | torch.float16 | 0.01 | 0.47 | 2.82 |
| maximum | torch.float16 | torch.float16 | 0.02 | 0.57 | 3.41 |
| maximum | torch.float16 | torch.float16 | 0.04 | 0.56 | 3.34 |

### torch

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| maximum | torch.float16 | torch.float16 | 0.01 | 0.32 | 1.94 |
| maximum | torch.float16 | torch.float16 | 0.03 | 0.37 | 2.19 |
| maximum | torch.float16 | torch.float16 | 0.06 | 0.38 | 2.26 |

## minimum

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| minimum | torch.float16 | torch.float16 | 0.01 | 0.47 | 2.80 |
| minimum | torch.float16 | torch.float16 | 0.02 | 0.56 | 3.37 |
| minimum | torch.float16 | torch.float16 | 0.04 | 0.55 | 3.28 |

### torch

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| minimum | torch.float16 | torch.float16 | 0.01 | 0.32 | 1.92 |
| minimum | torch.float16 | torch.float16 | 0.03 | 0.40 | 2.37 |
| minimum | torch.float16 | torch.float16 | 0.06 | 0.38 | 2.26 |

## cmp_eq

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| eq | torch.float16 | 0.02 | 0.22 | 1.12 |
| eq | torch.float16 | 0.04 | 0.25 | 1.27 |

### torch

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| eq | torch.float16 | 0.01 | 0.37 | 1.85 |
| eq | torch.float16 | 0.03 | 0.41 | 2.06 |

## cmp_ne

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| ne | torch.float16 | 0.02 | 0.22 | 1.11 |

### torch

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| ne | torch.float16 | 0.01 | 0.36 | 1.82 |

## cmp_gt

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| gt | torch.float16 | 0.02 | 0.22 | 1.10 |

### torch

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| gt | torch.float16 | 0.01 | 0.36 | 1.80 |

## cmp_lt

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| lt | torch.float16 | 0.02 | 0.22 | 1.11 |

### torch

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| lt | torch.float16 | 0.01 | 0.36 | 1.79 |

## cmp_ge

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| ge | torch.float16 | 0.02 | 0.22 | 1.11 |

### torch

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| ge | torch.float16 | 0.01 | 0.36 | 1.81 |

## cmp_le

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| le | torch.float16 | 0.02 | 0.22 | 1.11 |

### torch

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| le | torch.float16 | 0.01 | 0.36 | 1.79 |

## logical_and

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| logical_and | torch.float16 | 0.02 | 0.22 | 1.11 |
| logical_and | torch.float16 | 0.04 | 0.26 | 1.28 |

### torch

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| logical_and | torch.float16 | 0.01 | 0.57 | 2.86 |
| logical_and | torch.float16 | 0.02 | 0.70 | 3.48 |

## logical_or

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| logical_or | torch.float16 | 0.02 | 0.22 | 1.11 |
| logical_or | torch.float16 | 0.04 | 0.26 | 1.28 |

### torch

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| logical_or | torch.float16 | 0.01 | 0.58 | 2.91 |
| logical_or | torch.float16 | 0.02 | 0.70 | 3.48 |

## bitwise_and

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| bitwise_and | torch.int32 | 0.02 | 0.27 | 3.27 |
| bitwise_and | torch.int32 | 0.04 | 0.27 | 3.25 |

### torch

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| bitwise_and | torch.int32 | 0.02 | 0.18 | 2.19 |
| bitwise_and | torch.int32 | 0.06 | 0.19 | 2.29 |

## bitwise_or

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| bitwise_or | torch.int32 | 0.02 | 0.27 | 3.27 |

### torch

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| bitwise_or | torch.int32 | 0.02 | 0.18 | 2.20 |

## bitwise_xor

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| bitwise_xor | torch.int32 | 0.02 | 0.27 | 3.27 |

### torch

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| bitwise_xor | torch.int32 | 0.02 | 0.23 | 2.78 |

## gelu_and_mul

### tileops

| op_name | M | N | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| gelu_and_mul | 1024 | 4096 | torch.float16 | 0.01 | 0.79 | 2.38 |
| gelu_and_mul | 1024 | 10240 | torch.float16 | 0.03 | 0.78 | 2.33 |
| gelu_and_mul | 1024 | 20480 | torch.float16 | 0.06 | 0.68 | 2.04 |

### torch-ref

| op_name | M | N | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| gelu_and_mul | 1024 | 4096 | torch.float16 | 0.04 | 0.20 | 0.60 |
| gelu_and_mul | 1024 | 10240 | torch.float16 | 0.11 | 0.19 | 0.56 |
| gelu_and_mul | 1024 | 20480 | torch.float16 | 0.23 | 0.18 | 0.54 |

## gelu_tanh_and_mul

### tileops

| op_name | M | N | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| gelu_tanh_and_mul | 1024 | 4096 | torch.float16 | 0.01 | 0.91 | 2.73 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.float16 | 0.02 | 1.02 | 3.07 |
| gelu_tanh_and_mul | 1024 | 20480 | torch.float16 | 0.05 | 0.87 | 2.60 |

### torch-ref

| op_name | M | N | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| gelu_tanh_and_mul | 1024 | 4096 | torch.float16 | 0.04 | 0.21 | 0.63 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.float16 | 0.11 | 0.20 | 0.59 |
| gelu_tanh_and_mul | 1024 | 20480 | torch.float16 | 0.22 | 0.19 | 0.57 |

## silu_and_mul_strategy

### tileops-direct

| op_name | M | N | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| silu_and_mul | 1024 | 4096 | torch.float16 | direct | 0.02 | 0.50 | 1.50 |
| silu_and_mul | 1024 | 4096 | torch.bfloat16 | direct | 0.02 | 0.50 | 1.51 |
| silu_and_mul | 1024 | 4096 | torch.float32 | direct | 0.02 | 0.42 | 2.50 |
| silu_and_mul | 1024 | 10240 | torch.float16 | direct | 0.05 | 0.39 | 1.18 |
| silu_and_mul | 1024 | 10240 | torch.bfloat16 | direct | 0.05 | 0.39 | 1.18 |
| silu_and_mul | 1024 | 10240 | torch.float32 | direct | 0.06 | 0.33 | 1.99 |
| silu_and_mul | 4096 | 4096 | torch.float16 | direct | 0.09 | 0.36 | 1.08 |
| silu_and_mul | 4096 | 4096 | torch.bfloat16 | direct | 0.09 | 0.36 | 1.08 |
| silu_and_mul | 4096 | 4096 | torch.float32 | direct | 0.11 | 0.31 | 1.86 |

### tileops-explicit_parallel

| op_name | M | N | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| silu_and_mul | 1024 | 4096 | torch.float16 | explicit_parallel | 0.01 | 0.94 | 2.81 |
| silu_and_mul | 1024 | 4096 | torch.bfloat16 | explicit_parallel | 0.01 | 0.92 | 2.76 |
| silu_and_mul | 1024 | 4096 | torch.float32 | explicit_parallel | 0.02 | 0.56 | 3.35 |
| silu_and_mul | 1024 | 10240 | torch.float16 | explicit_parallel | 0.02 | 1.02 | 3.06 |
| silu_and_mul | 1024 | 10240 | torch.bfloat16 | explicit_parallel | 0.02 | 0.99 | 2.97 |
| silu_and_mul | 1024 | 10240 | torch.float32 | explicit_parallel | 0.04 | 0.49 | 2.96 |
| silu_and_mul | 4096 | 4096 | torch.float16 | explicit_parallel | 0.04 | 0.91 | 2.73 |
| silu_and_mul | 4096 | 4096 | torch.bfloat16 | explicit_parallel | 0.04 | 0.88 | 2.65 |
| silu_and_mul | 4096 | 4096 | torch.float32 | explicit_parallel | 0.07 | 0.46 | 2.75 |

## gelu_and_mul_strategy

### tileops-direct

| op_name | M | N | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gelu_and_mul | 1024 | 4096 | torch.float16 | direct | 0.02 | 0.49 | 1.47 |
| gelu_and_mul | 1024 | 4096 | torch.bfloat16 | direct | 0.02 | 0.49 | 1.46 |
| gelu_and_mul | 1024 | 4096 | torch.float32 | direct | 0.02 | 0.41 | 2.46 |
| gelu_and_mul | 1024 | 10240 | torch.float16 | direct | 0.05 | 0.38 | 1.15 |
| gelu_and_mul | 1024 | 10240 | torch.bfloat16 | direct | 0.05 | 0.38 | 1.15 |
| gelu_and_mul | 1024 | 10240 | torch.float32 | direct | 0.06 | 0.33 | 1.97 |
| gelu_and_mul | 4096 | 4096 | torch.float16 | direct | 0.09 | 0.35 | 1.06 |
| gelu_and_mul | 4096 | 4096 | torch.bfloat16 | direct | 0.09 | 0.35 | 1.06 |
| gelu_and_mul | 4096 | 4096 | torch.float32 | direct | 0.11 | 0.31 | 1.87 |

### tileops-explicit_parallel

| op_name | M | N | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gelu_and_mul | 1024 | 4096 | torch.float16 | explicit_parallel | 0.01 | 0.78 | 2.34 |
| gelu_and_mul | 1024 | 4096 | torch.bfloat16 | explicit_parallel | 0.01 | 0.77 | 2.32 |
| gelu_and_mul | 1024 | 4096 | torch.float32 | explicit_parallel | 0.02 | 0.56 | 3.33 |
| gelu_and_mul | 1024 | 10240 | torch.float16 | explicit_parallel | 0.03 | 0.78 | 2.33 |
| gelu_and_mul | 1024 | 10240 | torch.bfloat16 | explicit_parallel | 0.03 | 0.77 | 2.31 |
| gelu_and_mul | 1024 | 10240 | torch.float32 | explicit_parallel | 0.04 | 0.47 | 2.85 |
| gelu_and_mul | 4096 | 4096 | torch.float16 | explicit_parallel | 0.04 | 0.75 | 2.24 |
| gelu_and_mul | 4096 | 4096 | torch.bfloat16 | explicit_parallel | 0.05 | 0.74 | 2.23 |
| gelu_and_mul | 4096 | 4096 | torch.float32 | explicit_parallel | 0.07 | 0.46 | 2.73 |

## gelu_tanh_and_mul_strategy

### tileops-direct

| op_name | M | N | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gelu_tanh_and_mul | 1024 | 4096 | torch.float16 | direct | 0.02 | 0.50 | 1.51 |
| gelu_tanh_and_mul | 1024 | 4096 | torch.bfloat16 | direct | 0.02 | 0.49 | 1.48 |
| gelu_tanh_and_mul | 1024 | 4096 | torch.float32 | direct | 0.02 | 0.41 | 2.48 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.float16 | direct | 0.05 | 0.39 | 1.17 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.bfloat16 | direct | 0.05 | 0.39 | 1.17 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.float32 | direct | 0.06 | 0.34 | 2.02 |
| gelu_tanh_and_mul | 4096 | 4096 | torch.float16 | direct | 0.09 | 0.36 | 1.09 |
| gelu_tanh_and_mul | 4096 | 4096 | torch.bfloat16 | direct | 0.09 | 0.36 | 1.08 |
| gelu_tanh_and_mul | 4096 | 4096 | torch.float32 | direct | 0.11 | 0.32 | 1.90 |

### tileops-explicit_parallel

| op_name | M | N | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gelu_tanh_and_mul | 1024 | 4096 | torch.float16 | explicit_parallel | 0.01 | 0.94 | 2.81 |
| gelu_tanh_and_mul | 1024 | 4096 | torch.bfloat16 | explicit_parallel | 0.01 | 0.91 | 2.74 |
| gelu_tanh_and_mul | 1024 | 4096 | torch.float32 | explicit_parallel | 0.02 | 0.55 | 3.33 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.float16 | explicit_parallel | 0.02 | 1.00 | 3.00 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.bfloat16 | explicit_parallel | 0.02 | 0.99 | 2.98 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.float32 | explicit_parallel | 0.04 | 0.49 | 2.95 |
| gelu_tanh_and_mul | 4096 | 4096 | torch.float16 | explicit_parallel | 0.04 | 0.89 | 2.68 |
| gelu_tanh_and_mul | 4096 | 4096 | torch.bfloat16 | explicit_parallel | 0.04 | 0.89 | 2.67 |
| gelu_tanh_and_mul | 4096 | 4096 | torch.float32 | explicit_parallel | 0.07 | 0.45 | 2.71 |

## sub_bcast

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| sub | torch.float16 | 0.01 | 0.63 | 2.52 |
| sub | torch.float16 | 0.01 | 0.77 | 3.09 |
| sub | torch.float16 | 0.03 | 0.83 | 3.31 |

### torch

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| sub | torch.float16 | 0.02 | 0.20 | 0.78 |
| sub | torch.float16 | 0.05 | 0.21 | 0.83 |
| sub | torch.float16 | 0.10 | 0.21 | 0.83 |

## mul_bcast

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| mul | torch.float16 | 0.01 | 0.63 | 2.54 |
| mul | torch.float16 | 0.01 | 0.77 | 3.07 |
| mul | torch.float16 | 0.03 | 0.83 | 3.31 |

### torch

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| mul | torch.float16 | 0.02 | 0.20 | 0.78 |
| mul | torch.float16 | 0.05 | 0.21 | 0.83 |
| mul | torch.float16 | 0.10 | 0.21 | 0.84 |

## div_bcast

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| div | torch.float16 | 0.01 | 0.61 | 2.43 |
| div | torch.float16 | 0.01 | 0.73 | 2.92 |
| div | torch.float16 | 0.03 | 0.77 | 3.10 |

### torch

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| div | torch.float16 | 0.02 | 0.18 | 0.72 |
| div | torch.float16 | 0.05 | 0.19 | 0.77 |
| div | torch.float16 | 0.11 | 0.19 | 0.76 |

## binary_strategy

### add_direct

| n_total | shape_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4194304 | 1024x4096 | torch.float16 | direct | 0.02 | 0.26 | 1.58 |
| 4194304 | 1024x4096 | torch.bfloat16 | direct | 0.02 | 0.26 | 1.58 |
| 4194304 | 1024x4096 | torch.float32 | direct | 0.02 | 0.22 | 2.63 |
| 10485760 | 1024x10240 | torch.float16 | direct | 0.04 | 0.28 | 1.69 |
| 10485760 | 1024x10240 | torch.bfloat16 | direct | 0.04 | 0.23 | 1.40 |
| 10485760 | 1024x10240 | torch.float32 | direct | 0.06 | 0.19 | 2.27 |
| 20971520 | 1024x20480 | torch.float16 | direct | 0.10 | 0.20 | 1.20 |
| 20971520 | 1024x20480 | torch.bfloat16 | direct | 0.11 | 0.20 | 1.19 |
| 20971520 | 1024x20480 | torch.float32 | direct | 0.13 | 0.16 | 1.96 |

### add_explicit_parallel

| n_total | shape_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4194304 | 1024x4096 | torch.float16 | explicit_parallel | 0.01 | 0.46 | 2.76 |
| 4194304 | 1024x4096 | torch.bfloat16 | explicit_parallel | 0.01 | 0.46 | 2.75 |
| 4194304 | 1024x4096 | torch.float32 | explicit_parallel | 0.02 | 0.28 | 3.32 |
| 10485760 | 1024x10240 | torch.float16 | explicit_parallel | 0.02 | 0.56 | 3.34 |
| 10485760 | 1024x10240 | torch.bfloat16 | explicit_parallel | 0.02 | 0.56 | 3.34 |
| 10485760 | 1024x10240 | torch.float32 | explicit_parallel | 0.04 | 0.28 | 3.37 |
| 20971520 | 1024x20480 | torch.float16 | explicit_parallel | 0.04 | 0.54 | 3.22 |
| 20971520 | 1024x20480 | torch.bfloat16 | explicit_parallel | 0.03 | 0.61 | 3.67 |
| 20971520 | 1024x20480 | torch.float32 | explicit_parallel | 0.09 | 0.24 | 2.84 |

## conv1d

### tileops

| n | c_in | l_in | c_out | kernel_size | stride | padding | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 256 | 32000 | 512 | 1 | 1 | 0 | torch.float16 | 0.34 | 97.72 | 0.57 |
| 4 | 128 | 4096 | 256 | 3 | 1 | 1 | torch.float16 | 0.02 | 155.28 | 0.62 |
| 4 | 64 | 16000 | 128 | 5 | 2 | 2 | torch.float16 | 0.02 | 135.84 | 0.85 |
| 4 | 128 | 8192 | 256 | 7 | 1 | 3 | torch.float16 | 0.05 | 313.10 | 0.53 |
| 2 | 128 | 4096 | 256 | 3 | 2 | 1 | torch.bfloat16 | 0.01 | 76.45 | 0.42 |

### torch

| n | c_in | l_in | c_out | kernel_size | stride | padding | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 256 | 32000 | 512 | 1 | 1 | 0 | torch.float16 | 0.52 | 64.78 | 0.38 |
| 4 | 128 | 4096 | 256 | 3 | 1 | 1 | torch.float16 | 0.05 | 59.46 | 0.24 |
| 4 | 64 | 16000 | 128 | 5 | 2 | 2 | torch.float16 | 0.06 | 42.18 | 0.26 |
| 4 | 128 | 8192 | 256 | 7 | 1 | 3 | torch.float16 | 0.12 | 124.89 | 0.21 |
| 2 | 128 | 4096 | 256 | 3 | 2 | 1 | torch.bfloat16 | 0.02 | 32.65 | 0.18 |

## conv2d

### tileops

| n | c_in | h | w | c_out | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 64 | 56 | 56 | 64 | torch.float16 | 0.01 | 54.04 | 0.20 |
| 1 | 3 | 112 | 112 | 64 | torch.float16 | 0.01 | 1.42 | 0.06 |
| 1 | 128 | 56 | 56 | 256 | torch.float16 | 0.01 | 37.32 | 0.14 |
| 1 | 256 | 112 | 112 | 512 | torch.float16 | 0.08 | 382.16 | 0.28 |
| 1 | 64 | 56 | 56 | 128 | torch.float16 | 0.01 | 93.37 | 0.12 |
| 1 | 128 | 56 | 56 | 256 | torch.float16 | 0.02 | 55.32 | 0.12 |
| 1 | 128 | 28 | 28 | 128 | torch.bfloat16 | 0.01 | 5.31 | 0.05 |
| 2 | 64 | 56 | 56 | 256 | torch.float16 | 0.00 | 47.94 | 0.94 |
| 2 | 128 | 28 | 28 | 512 | torch.float16 | 0.00 | 52.20 | 0.54 |
| 2 | 512 | 28 | 28 | 128 | torch.float16 | 0.00 | 42.78 | 0.45 |
| 1 | 256 | 14 | 14 | 1024 | torch.float16 | 0.00 | 24.68 | 0.25 |
| 1 | 512 | 7 | 7 | 2048 | torch.float16 | 0.01 | 18.99 | 0.43 |
| 2 | 64 | 56 | 56 | 256 | torch.bfloat16 | 0.00 | 47.58 | 0.94 |

### torch-nchw

| n | c_in | h | w | c_out | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 64 | 56 | 56 | 64 | torch.float16 | 0.02 | 25.44 | 0.09 |
| 1 | 3 | 112 | 112 | 64 | torch.float16 | 0.01 | 1.14 | 0.05 |
| 1 | 128 | 56 | 56 | 256 | torch.float16 | 0.02 | 22.54 | 0.09 |
| 1 | 256 | 112 | 112 | 512 | torch.float16 | 0.16 | 190.86 | 0.14 |
| 1 | 64 | 56 | 56 | 128 | torch.float16 | 0.02 | 56.07 | 0.07 |
| 1 | 128 | 56 | 56 | 256 | torch.float16 | 0.03 | 44.45 | 0.10 |
| 1 | 128 | 28 | 28 | 128 | torch.bfloat16 | 0.02 | 3.12 | 0.03 |
| 2 | 64 | 56 | 56 | 256 | torch.float16 | 0.01 | 19.04 | 0.37 |
| 2 | 128 | 28 | 28 | 512 | torch.float16 | 0.01 | 24.66 | 0.26 |
| 2 | 512 | 28 | 28 | 128 | torch.float16 | 0.01 | 27.14 | 0.28 |
| 1 | 256 | 14 | 14 | 1024 | torch.float16 | 0.01 | 10.89 | 0.11 |
| 1 | 512 | 7 | 7 | 2048 | torch.float16 | 0.01 | 8.69 | 0.20 |
| 2 | 64 | 56 | 56 | 256 | torch.bfloat16 | 0.01 | 18.90 | 0.37 |

### torch-nhwc

| n | c_in | h | w | c_out | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 64 | 56 | 56 | 64 | torch.float16 | 0.01 | 36.45 | 0.13 |
| 1 | 3 | 112 | 112 | 64 | torch.float16 | 0.01 | 0.83 | 0.04 |
| 1 | 128 | 56 | 56 | 256 | torch.float16 | 0.01 | 31.66 | 0.12 |
| 1 | 256 | 112 | 112 | 512 | torch.float16 | 0.12 | 253.54 | 0.19 |
| 1 | 64 | 56 | 56 | 128 | torch.float16 | 0.02 | 72.09 | 0.09 |
| 1 | 128 | 56 | 56 | 256 | torch.float16 | 0.02 | 53.10 | 0.12 |
| 1 | 128 | 28 | 28 | 128 | torch.bfloat16 | 0.01 | 4.11 | 0.04 |
| 2 | 64 | 56 | 56 | 256 | torch.float16 | 0.01 | 20.09 | 0.40 |
| 2 | 128 | 28 | 28 | 512 | torch.float16 | 0.01 | 25.40 | 0.26 |
| 2 | 512 | 28 | 28 | 128 | torch.float16 | 0.01 | 27.28 | 0.28 |
| 1 | 256 | 14 | 14 | 1024 | torch.float16 | 0.01 | 14.24 | 0.14 |
| 1 | 512 | 7 | 7 | 2048 | torch.float16 | 0.01 | 13.59 | 0.31 |
| 2 | 64 | 56 | 56 | 256 | torch.bfloat16 | 0.01 | 20.12 | 0.40 |

## conv3d

### tileops

| n | c_in | d_in | h_in | w_in | c_out | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3 | 16 | 112 | 112 | 64 | torch.float16 | 0.08 | 25.20 | 0.33 |
| 1 | 64 | 8 | 56 | 56 | 128 | torch.float16 | 0.02 | 67.32 | 0.22 |
| 1 | 32 | 32 | 64 | 64 | 64 | torch.bfloat16 | 0.09 | 157.68 | 0.27 |

### torch-ncdhw

| n | c_in | d_in | h_in | w_in | c_out | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3 | 16 | 112 | 112 | 64 | torch.float16 | 0.23 | 9.14 | 0.12 |
| 1 | 64 | 8 | 56 | 56 | 128 | torch.float16 | 0.03 | 46.43 | 0.15 |
| 1 | 32 | 32 | 64 | 64 | 64 | torch.bfloat16 | 0.21 | 67.90 | 0.12 |

### torch-ndhwc

| n | c_in | d_in | h_in | w_in | c_out | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3 | 16 | 112 | 112 | 64 | torch.float16 | 0.15 | 14.08 | 0.18 |
| 1 | 64 | 8 | 56 | 56 | 128 | torch.float16 | 0.02 | 60.89 | 0.20 |
| 1 | 32 | 32 | 64 | 64 | 64 | torch.bfloat16 | 0.17 | 86.40 | 0.15 |

## CumsumOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | cumsum | 0.08 | 0.05 | 0.21 |
| 1024 | 4096 | torch.bfloat16 | cumsum | 0.08 | 0.05 | 0.21 |
| 4096 | 4096 | torch.float16 | cumsum | 0.20 | 0.09 | 0.34 |

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
| 1024 | 4096 | torch.bfloat16 | cumprod | 0.08 | 0.05 | 0.21 |
| 4096 | 4096 | torch.float16 | cumprod | 0.20 | 0.09 | 0.34 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | cumprod | 0.17 | 0.02 | 0.10 |
| 1024 | 4096 | torch.bfloat16 | cumprod | 0.17 | 0.02 | 0.10 |
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
| 1 | 128 | 1024 | 2048 | 512 | 64 | 2048 | 1 | 1 | 1024 | torch.float16 | 2.27 | 257.19 | 0.13 |
| 1 | 128 | 512 | 4096 | 512 | 64 | 1024 | 1 | 1 | 512 | torch.float16 | 0.57 | 255.05 | 0.26 |

### torch-ref

| batch | heads | seq_len_q | seq_len_kv | dim | dim_tail | topk | stride_kv | heads_kv | q_start_index_s | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 128 | 1024 | 2048 | 512 | 64 | 2048 | 1 | 1 | 1024 | torch.float16 | 31.23 | 18.71 | 0.01 |
| 1 | 128 | 512 | 4096 | 512 | 64 | 1024 | 1 | 1 | 512 | torch.float16 | 31.64 | 4.62 | 0.00 |

## MultiHeadLatentAttentionDecodeWithKVCacheOp

### tileops

| batch | heads | heads_kv | seq_len_kv | dim | dim_pe | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 128 | 1 | 8192 | 512 | 64 | torch.float16 | 0.26 | 281.74 | 1.18 |
| 16 | 128 | 1 | 4096 | 512 | 64 | torch.float16 | 0.05 | 361.66 | 1.54 |

### torch-ref

| batch | heads | heads_kv | seq_len_kv | dim | dim_pe | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 128 | 1 | 8192 | 512 | 64 | torch.float16 | 0.89 | 81.66 | 0.34 |
| 16 | 128 | 1 | 4096 | 512 | 64 | torch.float16 | 0.15 | 120.48 | 0.51 |

## DeltaNetFwdOp

### tileops

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 0.10 | 2.78 | 0.17 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 0.21 | 1.30 | 0.08 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 0.21 | 1.29 | 0.08 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 0.05 | 2.55 | 0.16 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 0.32 | 1.70 | 0.11 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 0.73 | 1.46 | 0.09 |
| 2 | 32768 | 4 | 64 | 64 | 64 | torch.float16 | 1.55 | 1.38 | 0.09 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.05 | 2.53 | 0.16 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.10 | 2.67 | 0.17 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.32 | 1.66 | 0.10 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.74 | 1.46 | 0.09 |
| 2 | 32768 | 4 | 64 | 64 | 64 | torch.bfloat16 | 1.56 | 1.38 | 0.09 |

### fla

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.08 | 3.19 | 0.20 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 0.125 | 0.08 | 3.19 | 0.20 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 0.125 | 0.08 | 3.16 | 0.20 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.05 | 2.44 | 0.15 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.16 | 3.29 | 0.21 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.34 | 3.16 | 0.20 |
| 2 | 32768 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.92 | 2.33 | 0.15 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.05 | 2.45 | 0.15 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.08 | 3.18 | 0.20 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.16 | 3.26 | 0.20 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.34 | 3.12 | 0.20 |
| 2 | 32768 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.94 | 2.30 | 0.14 |

## DeltaNetBwdOp

### tileops

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | H | S | DK | DV | BC | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.34 | 1.59 | 0.09 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.40 | 1.34 | 0.07 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.40 | 1.34 | 0.07 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.12 | 2.27 | 0.12 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.75 | 1.43 | 0.08 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 16384 | 64 | 64 | 64 | 1.60 | 1.35 | 0.07 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.12 | 2.32 | 0.13 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.33 | 1.63 | 0.09 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.75 | 1.43 | 0.08 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 16384 | 64 | 64 | 64 | 1.60 | 1.34 | 0.07 |

### fla

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | H | S | DK | DV | BC | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.125 | 0.18 | 3.02 | 0.17 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.125 | 0.18 | 3.03 | 0.17 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.125 | 0.18 | 3.03 | 0.17 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.125 | 0.09 | 2.83 | 0.16 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.125 | 0.36 | 2.99 | 0.16 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 16384 | 64 | 64 | 64 | 0.125 | 0.93 | 2.30 | 0.13 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.125 | 0.09 | 2.84 | 0.16 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.125 | 0.18 | 3.02 | 0.17 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.125 | 0.36 | 2.98 | 0.16 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 16384 | 64 | 64 | 64 | 0.125 | 0.95 | 2.27 | 0.13 |

## DeltaNetOp

### tileops

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | H | S | DK | DV | BC | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.32 | 2.48 | 0.14 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.47 | 1.71 | 0.10 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.47 | 1.71 | 0.10 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.17 | 2.42 | 0.14 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.94 | 1.71 | 0.10 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 16384 | 64 | 64 | 64 | 2.20 | 1.47 | 0.08 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.17 | 2.44 | 0.14 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.32 | 2.51 | 0.14 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.94 | 1.71 | 0.10 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 16384 | 64 | 64 | 64 | 2.20 | 1.46 | 0.08 |

### fla

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | H | S | DK | DV | BC | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.125 | 0.26 | 3.11 | 0.18 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.125 | 0.26 | 3.12 | 0.18 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.125 | 0.26 | 3.10 | 0.18 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.125 | 0.15 | 2.75 | 0.16 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.125 | 0.52 | 3.12 | 0.18 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 16384 | 64 | 64 | 64 | 0.125 | 1.17 | 2.76 | 0.16 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.125 | 0.15 | 2.75 | 0.16 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.125 | 0.26 | 3.10 | 0.18 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.125 | 0.52 | 3.09 | 0.18 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 16384 | 64 | 64 | 64 | 0.125 | 1.19 | 2.70 | 0.16 |

## DeltaNetDecodeOp

### tileops

| batch | heads | dim_k | dim_v | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 128 | 128 | torch.float32 | 0.01 | 0.32 | 0.43 |
| 1 | 32 | 128 | 128 | torch.bfloat16 | 0.01 | 0.34 | 0.23 |
| 8 | 32 | 128 | 128 | torch.float32 | 0.03 | 0.94 | 1.27 |
| 8 | 32 | 128 | 128 | torch.bfloat16 | 0.01 | 2.16 | 1.46 |
| 16 | 32 | 128 | 128 | torch.float32 | 0.06 | 0.84 | 1.13 |
| 16 | 32 | 128 | 128 | torch.bfloat16 | 0.02 | 2.66 | 1.80 |
| 32 | 32 | 128 | 128 | torch.float32 | 0.13 | 0.77 | 1.04 |
| 32 | 32 | 128 | 128 | torch.bfloat16 | 0.04 | 2.44 | 1.65 |
| 1 | 32 | 64 | 64 | torch.float32 | 0.01 | 0.15 | 0.21 |
| 8 | 32 | 64 | 64 | torch.float32 | 0.01 | 0.62 | 0.84 |
| 32 | 32 | 64 | 64 | torch.float32 | 0.05 | 0.49 | 0.67 |
| 64 | 32 | 128 | 128 | torch.float32 | 0.27 | 0.73 | 0.99 |
| 64 | 32 | 128 | 128 | torch.bfloat16 | 0.09 | 2.31 | 1.56 |

### torch

| batch | heads | dim_k | dim_v | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 128 | 128 | torch.float32 | 0.02 | 0.14 | 0.19 |
| 1 | 32 | 128 | 128 | torch.bfloat16 | 0.04 | 0.08 | 0.05 |
| 8 | 32 | 128 | 128 | torch.float32 | 0.06 | 0.44 | 0.59 |
| 8 | 32 | 128 | 128 | torch.bfloat16 | 0.09 | 0.29 | 0.20 |
| 16 | 32 | 128 | 128 | torch.float32 | 0.08 | 0.59 | 0.80 |
| 16 | 32 | 128 | 128 | torch.bfloat16 | 0.14 | 0.37 | 0.25 |
| 32 | 32 | 128 | 128 | torch.float32 | 0.17 | 0.58 | 0.79 |
| 32 | 32 | 128 | 128 | torch.bfloat16 | 0.29 | 0.35 | 0.24 |
| 1 | 32 | 64 | 64 | torch.float32 | 0.02 | 0.04 | 0.06 |
| 8 | 32 | 64 | 64 | torch.float32 | 0.03 | 0.24 | 0.32 |
| 32 | 32 | 64 | 64 | torch.float32 | 0.05 | 0.46 | 0.63 |
| 64 | 32 | 128 | 128 | torch.float32 | 0.42 | 0.48 | 0.65 |
| 64 | 32 | 128 | 128 | torch.bfloat16 | 0.68 | 0.30 | 0.20 |

## DropoutOp

### tileops

| dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| torch.float16 | 4194304 | 0.01 | 0.65 | 2.58 |
| torch.bfloat16 | 4194304 | 0.01 | 0.65 | 2.59 |
| torch.float32 | 4194304 | 0.01 | 0.39 | 3.15 |
| torch.float16 | 10485760 | 0.01 | 0.83 | 3.34 |
| torch.bfloat16 | 10485760 | 0.01 | 0.83 | 3.33 |
| torch.float32 | 10485760 | 0.02 | 0.47 | 3.74 |
| torch.float16 | 20971520 | 0.02 | 0.93 | 3.73 |
| torch.bfloat16 | 20971520 | 0.02 | 0.94 | 3.74 |
| torch.float32 | 20971520 | 0.05 | 0.40 | 3.18 |

### torch

| dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| torch.float16 | 4194304 | 0.01 | 0.35 | 1.41 |
| torch.bfloat16 | 4194304 | 0.01 | 0.35 | 1.39 |
| torch.float32 | 4194304 | 0.02 | 0.23 | 1.86 |
| torch.float16 | 10485760 | 0.03 | 0.38 | 1.51 |
| torch.bfloat16 | 10485760 | 0.03 | 0.37 | 1.47 |
| torch.float32 | 10485760 | 0.05 | 0.22 | 1.74 |
| torch.float16 | 20971520 | 0.06 | 0.35 | 1.40 |
| torch.bfloat16 | 20971520 | 0.05 | 0.40 | 1.61 |
| torch.float32 | 20971520 | 0.10 | 0.20 | 1.63 |

## relu_fp8

### tileops

| op_name | n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| relu_fp8 | 8388608 | torch.float8_e4m3fn | 0.01 | 0.58 | 1.16 |
| relu_fp8 | 8388608 | torch.float8_e5m2 | 0.02 | 0.46 | 0.92 |
| relu_fp8 | 67108864 | torch.float8_e4m3fn | 0.16 | 0.43 | 0.85 |
| relu_fp8 | 67108864 | torch.float8_e5m2 | 0.20 | 0.33 | 0.66 |
| relu_fp8 | 134217728 | torch.float8_e4m3fn | 0.34 | 0.39 | 0.79 |
| relu_fp8 | 134217728 | torch.float8_e5m2 | 0.44 | 0.31 | 0.61 |

### torch

| op_name | n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| relu_fp8 | 8388608 | torch.float8_e4m3fn | 0.06 | 0.13 | 0.27 |
| relu_fp8 | 8388608 | torch.float8_e5m2 | 0.06 | 0.14 | 0.28 |
| relu_fp8 | 67108864 | torch.float8_e4m3fn | 0.65 | 0.10 | 0.21 |
| relu_fp8 | 67108864 | torch.float8_e5m2 | 0.63 | 0.11 | 0.21 |
| relu_fp8 | 134217728 | torch.float8_e4m3fn | 1.33 | 0.10 | 0.20 |
| relu_fp8 | 134217728 | torch.float8_e5m2 | 1.29 | 0.10 | 0.21 |

## exp_fp8

### tileops

| op_name | n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| exp_fp8 | 8388608 | torch.float8_e4m3fn | 0.01 | 0.99 | 1.98 |
| exp_fp8 | 8388608 | torch.float8_e5m2 | 0.02 | 0.46 | 0.91 |
| exp_fp8 | 67108864 | torch.float8_e4m3fn | 0.07 | 0.94 | 1.87 |
| exp_fp8 | 67108864 | torch.float8_e5m2 | 0.20 | 0.33 | 0.66 |
| exp_fp8 | 134217728 | torch.float8_e4m3fn | 0.16 | 0.83 | 1.65 |
| exp_fp8 | 134217728 | torch.float8_e5m2 | 0.44 | 0.31 | 0.61 |

### torch

| op_name | n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| exp_fp8 | 8388608 | torch.float8_e4m3fn | 0.06 | 0.13 | 0.26 |
| exp_fp8 | 8388608 | torch.float8_e5m2 | 0.06 | 0.14 | 0.28 |
| exp_fp8 | 67108864 | torch.float8_e4m3fn | 0.64 | 0.10 | 0.21 |
| exp_fp8 | 67108864 | torch.float8_e5m2 | 0.63 | 0.11 | 0.21 |
| exp_fp8 | 134217728 | torch.float8_e4m3fn | 1.31 | 0.10 | 0.21 |
| exp_fp8 | 134217728 | torch.float8_e5m2 | 1.27 | 0.11 | 0.21 |

## add_fp8

### tileops

| op_name | n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| add_fp8 | 8388608 | torch.float8_e4m3fn | 0.01 | 0.85 | 2.55 |
| add_fp8 | 8388608 | torch.float8_e5m2 | 0.02 | 0.41 | 1.24 |
| add_fp8 | 67108864 | torch.float8_e4m3fn | 0.08 | 0.81 | 2.43 |
| add_fp8 | 67108864 | torch.float8_e5m2 | 0.22 | 0.31 | 0.92 |
| add_fp8 | 134217728 | torch.float8_e4m3fn | 0.19 | 0.71 | 2.12 |
| add_fp8 | 134217728 | torch.float8_e5m2 | 0.47 | 0.28 | 0.85 |

### torch

| op_name | n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| add_fp8 | 8388608 | torch.float8_e4m3fn | 0.12 | 0.07 | 0.22 |
| add_fp8 | 8388608 | torch.float8_e5m2 | 0.11 | 0.08 | 0.23 |
| add_fp8 | 67108864 | torch.float8_e4m3fn | 1.11 | 0.06 | 0.18 |
| add_fp8 | 67108864 | torch.float8_e5m2 | 1.06 | 0.06 | 0.19 |
| add_fp8 | 134217728 | torch.float8_e4m3fn | 2.23 | 0.06 | 0.18 |
| add_fp8 | 134217728 | torch.float8_e5m2 | 2.15 | 0.06 | 0.19 |

## silu_and_mul_fp8

### tileops

| op_name | M | N | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| silu_and_mul_fp8 | 2048 | 11008 | torch.float8_e4m3fn | 0.06 | 1.82 | 1.09 |
| silu_and_mul_fp8 | 2048 | 11008 | torch.float8_e5m2 | 0.09 | 1.26 | 0.76 |
| silu_and_mul_fp8 | 16384 | 11008 | torch.float8_e4m3fn | 0.62 | 1.47 | 0.88 |
| silu_and_mul_fp8 | 16384 | 11008 | torch.float8_e5m2 | 0.89 | 1.01 | 0.61 |
| silu_and_mul_fp8 | 16384 | 28672 | torch.float8_e4m3fn | 1.68 | 1.39 | 0.84 |
| silu_and_mul_fp8 | 16384 | 28672 | torch.float8_e5m2 | 2.39 | 0.98 | 0.59 |

### torch-ref

| op_name | M | N | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| silu_and_mul_fp8 | 2048 | 11008 | torch.float8_e4m3fn | 0.55 | 0.21 | 0.12 |
| silu_and_mul_fp8 | 2048 | 11008 | torch.float8_e5m2 | 0.53 | 0.21 | 0.13 |
| silu_and_mul_fp8 | 16384 | 11008 | torch.float8_e4m3fn | 4.52 | 0.20 | 0.12 |
| silu_and_mul_fp8 | 16384 | 11008 | torch.float8_e5m2 | 4.38 | 0.21 | 0.12 |
| silu_and_mul_fp8 | 16384 | 28672 | torch.float8_e4m3fn | 11.81 | 0.20 | 0.12 |
| silu_and_mul_fp8 | 16384 | 28672 | torch.float8_e5m2 | 11.46 | 0.21 | 0.12 |

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
| 2 | 64 | 512 | torch.float16 | 0.01 | 0.31 | 0.13 |
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
| 4096 | torch.complex64 | 0.01 | 0.02 | 0.01 |
| 16384 | torch.complex64 | 0.01 | 0.09 | 0.02 |
| 65536 | torch.complex64 | 0.02 | 0.30 | 0.06 |
| 262144 | torch.complex64 | 0.02 | 0.96 | 0.17 |
| 1048576 | torch.complex64 | 0.06 | 1.70 | 0.27 |
| 4096 | torch.complex64 | 0.02 | 0.89 | 0.24 |
| 4096 | torch.complex64 | 0.04 | 1.50 | 0.40 |
| 1024 | torch.complex64 | 0.04 | 1.30 | 0.42 |
| 4096 | torch.complex128 | 0.02 | 0.01 | 0.01 |
| 65536 | torch.complex128 | 0.03 | 0.16 | 0.06 |
| 4096 | torch.complex128 | 0.03 | 0.47 | 0.25 |

### torch-cufft

| n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 4096 | torch.complex64 | 0.01 | 0.05 | 0.01 |
| 16384 | torch.complex64 | 0.02 | 0.08 | 0.02 |
| 65536 | torch.complex64 | 0.01 | 0.69 | 0.14 |
| 262144 | torch.complex64 | 0.01 | 2.05 | 0.36 |
| 1048576 | torch.complex64 | 0.02 | 4.22 | 0.68 |
| 4096 | torch.complex64 | 0.01 | 2.79 | 0.74 |
| 4096 | torch.complex64 | 0.01 | 6.26 | 1.67 |
| 1024 | torch.complex64 | 0.01 | 6.27 | 2.01 |
| 4096 | torch.complex128 | 0.01 | 0.03 | 0.02 |
| 65536 | torch.complex128 | 0.01 | 0.52 | 0.21 |
| 4096 | torch.complex128 | 0.01 | 1.74 | 0.93 |

## Fp8LightingIndexerOp

### tileops

| batch | seq_len | heads | index_dim | seq_len_kv | kv_group | clean_logits | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 4096 | 32 | 64 | 8192 | 1 | True | 0.20 | N/A | 0.71 |
| 1 | 2048 | 16 | 64 | 4096 | 1 | True | 0.12 | N/A | 0.30 |

### torch-ref

| batch | seq_len | heads | index_dim | seq_len_kv | kv_group | clean_logits | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 4096 | 32 | 64 | 8192 | 1 | True | 23.44 | N/A | 0.01 |
| 1 | 2048 | 16 | 64 | 4096 | 1 | True | 2.91 | N/A | 0.01 |

## Fp8QuantOp

### tileops

| batch | seq_len_kv | kv_group | index_dim | in_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8192 | 1 | 64 | torch.float16 | 0.00 | 1.10 | 0.36 |
| 1 | 8192 | 1 | 64 | torch.bfloat16 | 0.00 | 1.10 | 0.37 |
| 1 | 4096 | 1 | 128 | torch.float32 | 0.00 | 0.81 | 0.54 |
| 1 | 16384 | 1 | 32 | torch.float32 | 0.00 | 1.03 | 0.68 |

### torch-ref

| batch | seq_len_kv | kv_group | index_dim | in_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8192 | 1 | 64 | torch.float16 | 0.02 | 0.18 | 0.06 |
| 1 | 8192 | 1 | 64 | torch.bfloat16 | 0.02 | 0.18 | 0.06 |
| 1 | 4096 | 1 | 128 | torch.float32 | 0.02 | 0.18 | 0.12 |
| 1 | 16384 | 1 | 32 | torch.float32 | 0.02 | 0.16 | 0.10 |

## FusedAddLayerNormOp

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 2048 | 4096 | torch.float16 | 0.04 | 1.43 | 1.90 |
| 2048 | 4096 | torch.bfloat16 | 0.05 | 0.93 | 1.24 |
| 1 | 4096 | torch.bfloat16 | 0.00 | 0.01 | 0.01 |
| 2048 | 8192 | torch.float16 | 0.10 | 0.98 | 1.31 |
| 2048 | 8192 | torch.bfloat16 | 0.10 | 1.03 | 1.38 |
| 1 | 8192 | torch.bfloat16 | 0.01 | 0.01 | 0.02 |

### torch-ref

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 2048 | 4096 | torch.float16 | 0.16 | 0.31 | 0.41 |
| 2048 | 4096 | torch.bfloat16 | 0.17 | 0.30 | 0.40 |
| 1 | 4096 | torch.bfloat16 | 0.02 | 0.00 | 0.00 |
| 2048 | 8192 | torch.float16 | 0.37 | 0.27 | 0.36 |
| 2048 | 8192 | torch.bfloat16 | 0.37 | 0.27 | 0.36 |
| 1 | 8192 | torch.bfloat16 | 0.03 | 0.00 | 0.00 |

## FusedAddRmsNormOp

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 2048 | 4096 | torch.float16 | 0.03 | 1.29 | 2.06 |
| 2048 | 4096 | torch.bfloat16 | 0.05 | 0.90 | 1.44 |
| 1 | 4096 | torch.bfloat16 | 0.00 | 0.01 | 0.01 |
| 2048 | 8192 | torch.float16 | 0.08 | 0.99 | 1.58 |
| 2048 | 8192 | torch.bfloat16 | 0.09 | 0.99 | 1.58 |
| 1 | 8192 | torch.bfloat16 | 0.00 | 0.01 | 0.02 |

### torch-ref

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 2048 | 4096 | torch.float16 | 0.38 | 0.11 | 0.18 |
| 2048 | 4096 | torch.bfloat16 | 0.38 | 0.11 | 0.18 |
| 1 | 4096 | torch.bfloat16 | 0.03 | 0.00 | 0.00 |
| 2048 | 8192 | torch.float16 | 0.83 | 0.10 | 0.16 |
| 2048 | 8192 | torch.bfloat16 | 0.84 | 0.10 | 0.16 |
| 1 | 8192 | torch.bfloat16 | 0.03 | 0.00 | 0.00 |

## GatedDeltaNetFwdOp

### tileops

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 0.24 | 1.12 | 0.07 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 0.25 | 1.09 | 0.07 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 0.08 | 1.65 | 0.10 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 0.14 | 1.89 | 0.12 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 0.41 | 1.29 | 0.08 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 0.94 | 1.14 | 0.07 |
| 2 | 32768 | 4 | 64 | 64 | 64 | torch.float16 | 1.97 | 1.09 | 0.07 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.08 | 1.65 | 0.10 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.14 | 1.93 | 0.12 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.42 | 1.28 | 0.08 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.93 | 1.15 | 0.07 |
| 2 | 32768 | 4 | 64 | 64 | 64 | torch.bfloat16 | 1.96 | 1.10 | 0.07 |

### fla

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 0.125 | 0.10 | 2.73 | 0.17 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 0.125 | 0.10 | 2.71 | 0.17 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.06 | 2.15 | 0.14 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.10 | 2.73 | 0.17 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.19 | 2.85 | 0.18 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.38 | 2.81 | 0.18 |
| 2 | 32768 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 1.04 | 2.06 | 0.13 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.06 | 2.13 | 0.13 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.10 | 2.71 | 0.17 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.19 | 2.83 | 0.18 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.39 | 2.78 | 0.18 |
| 2 | 32768 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 1.06 | 2.02 | 0.13 |

## GatedDeltaNetBwdOp

### tileops

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | H | S | DK | DV | BC | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.58 | 0.93 | 0.05 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.57 | 0.94 | 0.05 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.19 | 1.43 | 0.08 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.40 | 1.33 | 0.07 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.99 | 1.08 | 0.06 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 16384 | 64 | 64 | 64 | 2.20 | 0.98 | 0.05 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.19 | 1.41 | 0.08 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.41 | 1.30 | 0.07 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 8192 | 64 | 64 | 64 | 1.00 | 1.07 | 0.06 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 16384 | 64 | 64 | 64 | 2.22 | 0.97 | 0.05 |

### fla

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | H | S | DK | DV | BC | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.125 | 0.25 | 2.18 | 0.12 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.125 | 0.25 | 2.17 | 0.12 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.125 | 0.13 | 2.00 | 0.11 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.125 | 0.25 | 2.17 | 0.12 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.125 | 0.47 | 2.26 | 0.12 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 16384 | 64 | 64 | 64 | 0.125 | 1.17 | 1.83 | 0.10 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.125 | 0.14 | 1.99 | 0.11 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.125 | 0.25 | 2.17 | 0.12 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.125 | 0.48 | 2.26 | 0.12 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 16384 | 64 | 64 | 64 | 0.125 | 1.18 | 1.82 | 0.10 |

## GatedDeltaNetOp

### tileops

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | H | S | DK | DV | BC | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.66 | 1.22 | 0.07 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.67 | 1.20 | 0.07 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.27 | 1.51 | 0.09 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.49 | 1.64 | 0.09 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 8192 | 64 | 64 | 64 | 1.25 | 1.28 | 0.07 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 16384 | 64 | 64 | 64 | 2.99 | 1.08 | 0.06 |
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
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 16384 | 64 | 64 | 64 | 0.125 | 1.46 | 2.20 | 0.13 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.125 | 0.19 | 2.07 | 0.12 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.125 | 0.34 | 2.34 | 0.14 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.125 | 0.66 | 2.43 | 0.14 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 16384 | 64 | 64 | 64 | 0.125 | 1.47 | 2.20 | 0.13 |

## GatedDeltaNetDecodeOp

### tileops

| batch | heads | dim_k | dim_v | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 128 | 128 | torch.float32 | 0.01 | 0.32 | 0.43 |
| 1 | 32 | 128 | 128 | torch.bfloat16 | 0.01 | 0.32 | 0.22 |
| 8 | 32 | 128 | 128 | torch.float32 | 0.03 | 0.93 | 1.26 |
| 8 | 32 | 128 | 128 | torch.bfloat16 | 0.01 | 2.04 | 1.38 |
| 16 | 32 | 128 | 128 | torch.float32 | 0.06 | 0.83 | 1.13 |
| 16 | 32 | 128 | 128 | torch.bfloat16 | 0.02 | 2.60 | 1.76 |
| 32 | 32 | 128 | 128 | torch.float32 | 0.13 | 0.77 | 1.04 |
| 32 | 32 | 128 | 128 | torch.bfloat16 | 0.04 | 2.35 | 1.59 |
| 1 | 32 | 64 | 64 | torch.float32 | 0.01 | 0.15 | 0.20 |
| 8 | 32 | 64 | 64 | torch.float32 | 0.01 | 0.61 | 0.83 |
| 32 | 32 | 64 | 64 | torch.float32 | 0.05 | 0.48 | 0.66 |
| 64 | 32 | 128 | 128 | torch.float32 | 0.28 | 0.73 | 0.99 |
| 64 | 32 | 128 | 128 | torch.bfloat16 | 0.09 | 2.25 | 1.52 |

### fla

| batch | heads | dim_k | dim_v | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 128 | 128 | torch.float32 | 0.00 | 0.73 | 0.98 |
| 1 | 32 | 128 | 128 | torch.bfloat16 | 0.00 | 0.68 | 0.46 |
| 8 | 32 | 128 | 128 | torch.float32 | 0.02 | 1.65 | 2.23 |
| 8 | 32 | 128 | 128 | torch.bfloat16 | 0.01 | 1.76 | 1.19 |
| 16 | 32 | 128 | 128 | torch.float32 | 0.03 | 1.91 | 2.57 |
| 16 | 32 | 128 | 128 | torch.bfloat16 | 0.02 | 2.04 | 1.38 |
| 32 | 32 | 128 | 128 | torch.float32 | 0.05 | 2.04 | 2.75 |
| 32 | 32 | 128 | 128 | torch.bfloat16 | 0.05 | 2.22 | 1.50 |
| 1 | 32 | 64 | 64 | torch.float32 | 0.00 | 0.25 | 0.35 |
| 8 | 32 | 64 | 64 | torch.float32 | 0.01 | 1.16 | 1.59 |
| 32 | 32 | 64 | 64 | torch.float32 | 0.02 | 1.68 | 2.30 |
| 64 | 32 | 128 | 128 | torch.float32 | 0.12 | 1.68 | 2.27 |
| 64 | 32 | 128 | 128 | torch.bfloat16 | 0.10 | 1.92 | 1.30 |

## GemmOp

### tileops

| m | n | k | dtype | trans_a | trans_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1024 | 1024 | 1024 | torch.float16 | False | False | 0.01 | 248.28 | 0.73 |
| 1 | 7168 | 16384 | torch.float16 | False | True | 0.11 | 2.15 | 2.15 |
| 1 | 18432 | 7168 | torch.bfloat16 | False | True | 0.12 | 2.25 | 2.25 |
| 7168 | 1 | 16384 | torch.float16 | False | False | 0.11 | 2.15 | 2.15 |
| 18432 | 1 | 7168 | torch.bfloat16 | False | False | 0.12 | 2.25 | 2.25 |

### torch-cublas

| m | n | k | dtype | trans_a | trans_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1024 | 1024 | 1024 | torch.float16 | False | False | 0.01 | 323.10 | 0.95 |
| 1 | 7168 | 16384 | torch.float16 | False | True | 0.12 | 2.02 | 2.02 |
| 1 | 18432 | 7168 | torch.bfloat16 | False | True | 0.14 | 1.96 | 1.96 |
| 7168 | 1 | 16384 | torch.float16 | False | False | 0.12 | 1.99 | 1.99 |
| 18432 | 1 | 7168 | torch.bfloat16 | False | False | 0.14 | 1.93 | 1.93 |

## GLAFwdOp

### tileops

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.11 | 1.26 | 0.08 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.22 | 1.24 | 0.08 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.48 | 1.12 | 0.07 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.98 | 1.09 | 0.07 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.11 | 1.20 | 0.08 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.22 | 1.24 | 0.08 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.47 | 1.13 | 0.07 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.99 | 1.08 | 0.07 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.33 | 1.21 | 0.07 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.76 | 1.07 | 0.06 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 1.60 | 1.01 | 0.06 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 3.26 | 0.99 | 0.06 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.34 | 1.19 | 0.07 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.76 | 1.06 | 0.06 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 1.63 | 0.99 | 0.06 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 3.37 | 0.95 | 0.05 |

### fla

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.06 | 2.10 | 0.13 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.12 | 2.27 | 0.14 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.23 | 2.37 | 0.15 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.62 | 1.73 | 0.11 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.06 | 2.07 | 0.13 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.12 | 2.24 | 0.14 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.23 | 2.34 | 0.15 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.64 | 1.67 | 0.10 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.22 | 1.81 | 0.10 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.42 | 1.93 | 0.11 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.85 | 1.89 | 0.11 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 2.35 | 1.37 | 0.08 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.22 | 1.81 | 0.10 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.42 | 1.93 | 0.11 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.86 | 1.87 | 0.11 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 2.36 | 1.37 | 0.08 |

## GLABwdOp

### tileops

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | T | H | K | V | BC | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 2048 | 4 | 64 | 64 | 64 | 0.125 | 0.25 | 1.09 | 0.06 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4096 | 4 | 64 | 64 | 64 | 0.125 | 0.55 | 0.98 | 0.05 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 8192 | 4 | 64 | 64 | 64 | 0.125 | 1.13 | 0.95 | 0.05 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 16384 | 4 | 64 | 64 | 64 | 0.125 | 2.28 | 0.94 | 0.05 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 2048 | 4 | 64 | 64 | 64 | 0.125 | 0.25 | 1.09 | 0.06 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4096 | 4 | 64 | 64 | 64 | 0.125 | 0.55 | 0.97 | 0.05 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 8192 | 4 | 64 | 64 | 64 | 0.125 | 1.16 | 0.92 | 0.05 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 16384 | 4 | 64 | 64 | 64 | 0.125 | 2.40 | 0.90 | 0.05 |

### fla

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | T | H | K | V | BC | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 2048 | 4 | 64 | 64 | 64 | 0.125 | 0.15 | 1.73 | 0.09 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4096 | 4 | 64 | 64 | 64 | 0.125 | 0.28 | 1.89 | 0.10 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 8192 | 4 | 64 | 64 | 64 | 0.125 | 0.66 | 1.62 | 0.09 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 16384 | 4 | 64 | 64 | 64 | 0.125 | 1.67 | 1.29 | 0.07 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 2048 | 4 | 64 | 64 | 64 | 0.125 | 0.15 | 1.73 | 0.09 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4096 | 4 | 64 | 64 | 64 | 0.125 | 0.28 | 1.90 | 0.10 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 8192 | 4 | 64 | 64 | 64 | 0.125 | 0.67 | 1.60 | 0.09 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 16384 | 4 | 64 | 64 | 64 | 0.125 | 1.67 | 1.29 | 0.07 |

## GLADecodeOp

### tileops

| batch | heads | dim_k | dim_v | dtype | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 64 | 64 | torch.float32 | 0.125 | 0.01 | 0.07 | 0.15 |
| 1 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.02 | 0.11 | 0.21 |
| 1 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.01 | 0.16 | 0.16 |
| 1 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.01 | 0.16 | 0.16 |
| 8 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.05 | 0.34 | 0.68 |
| 8 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.02 | 1.07 | 1.08 |
| 8 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.02 | 1.03 | 1.05 |
| 16 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.11 | 0.30 | 0.62 |
| 16 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.02 | 1.50 | 1.52 |
| 16 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.02 | 1.51 | 1.53 |
| 32 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.23 | 0.29 | 0.59 |
| 32 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.05 | 1.34 | 1.36 |
| 32 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.05 | 1.31 | 1.33 |
| 64 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.48 | 0.28 | 0.57 |
| 64 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.10 | 1.30 | 1.32 |
| 64 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.10 | 1.29 | 1.31 |

### fla

| batch | heads | dim_k | dim_v | dtype | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 64 | 64 | torch.float32 | 0.125 | 0.01 | 0.08 | 0.16 |
| 1 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.01 | 0.31 | 0.62 |
| 1 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.01 | 0.26 | 0.26 |
| 1 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.01 | 0.27 | 0.27 |
| 8 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.02 | 1.05 | 2.13 |
| 8 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.02 | 1.00 | 1.02 |
| 8 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.02 | 1.00 | 1.01 |
| 16 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.03 | 1.24 | 2.53 |
| 16 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.03 | 1.29 | 1.31 |
| 16 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.03 | 1.29 | 1.31 |
| 32 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.05 | 1.44 | 2.92 |
| 32 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.04 | 1.54 | 1.56 |
| 32 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.04 | 1.53 | 1.56 |
| 64 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.09 | 1.58 | 3.21 |
| 64 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.08 | 1.72 | 1.75 |
| 64 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.08 | 1.71 | 1.74 |

## GroupQueryAttentionFwdOp

### tileops

| batch | seq_len | heads | heads_kv | dim | causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 32 | 8 | 128 | True | torch.float16 | 0.05 | 181.32 | 0.44 |
| 1 | 4096 | 32 | 8 | 128 | True | torch.float16 | 0.81 | 170.44 | 0.10 |
| 1 | 8192 | 32 | 8 | 128 | True | torch.float16 | 3.26 | 168.85 | 0.05 |
| 1 | 32768 | 32 | 8 | 128 | True | torch.float16 | 32.17 | 273.45 | 0.02 |
| 1 | 131072 | 32 | 8 | 128 | True | torch.float16 | 553.74 | 254.16 | 0.00 |
| 1 | 4096 | 64 | 8 | 128 | True | torch.float16 | 0.94 | 291.84 | 0.16 |
| 1 | 4096 | 128 | 8 | 128 | True | torch.float16 | 1.91 | 287.19 | 0.15 |
| 2 | 4096 | 32 | 8 | 128 | True | torch.bfloat16 | 0.93 | 294.55 | 0.18 |
| 1 | 8192 | 32 | 8 | 128 | True | torch.bfloat16 | 1.72 | 318.76 | 0.10 |
| 1 | 4096 | 64 | 8 | 128 | True | torch.bfloat16 | 0.92 | 300.14 | 0.16 |
| 1 | 4096 | 128 | 8 | 128 | True | torch.bfloat16 | 1.80 | 305.69 | 0.16 |
| 2 | 2048 | 32 | 8 | 128 | True | torch.bfloat16 | 0.27 | 257.24 | 0.31 |

### fa3

| batch | seq_len | heads | heads_kv | dim | causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 32 | 8 | 128 | True | torch.float16 | 0.06 | 150.13 | 0.37 |
| 1 | 4096 | 32 | 8 | 128 | True | torch.float16 | 0.86 | 160.16 | 0.10 |
| 1 | 8192 | 32 | 8 | 128 | True | torch.float16 | 3.32 | 165.83 | 0.05 |
| 1 | 32768 | 32 | 8 | 128 | True | torch.float16 | 48.73 | 180.49 | 0.01 |
| 1 | 131072 | 32 | 8 | 128 | True | torch.float16 | 825.82 | 170.42 | 0.00 |
| 1 | 4096 | 64 | 8 | 128 | True | torch.float16 | 1.46 | 187.81 | 0.10 |
| 1 | 4096 | 128 | 8 | 128 | True | torch.float16 | 2.82 | 194.72 | 0.10 |
| 2 | 4096 | 32 | 8 | 128 | True | torch.bfloat16 | 1.44 | 190.62 | 0.12 |
| 1 | 8192 | 32 | 8 | 128 | True | torch.bfloat16 | 2.75 | 200.07 | 0.06 |
| 1 | 4096 | 64 | 8 | 128 | True | torch.bfloat16 | 1.40 | 195.94 | 0.11 |
| 1 | 4096 | 128 | 8 | 128 | True | torch.bfloat16 | 2.69 | 204.31 | 0.11 |
| 2 | 2048 | 32 | 8 | 128 | True | torch.bfloat16 | 0.42 | 163.73 | 0.20 |

### flashinfer

| batch | seq_len | heads | heads_kv | dim | causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 32 | 8 | 128 | True | torch.float16 | 0.04 | 231.66 | 0.57 |
| 1 | 4096 | 32 | 8 | 128 | True | torch.float16 | 0.44 | 311.07 | 0.19 |
| 1 | 8192 | 32 | 8 | 128 | True | torch.float16 | 1.87 | 294.36 | 0.09 |
| 1 | 32768 | 32 | 8 | 128 | True | torch.float16 | 28.38 | 309.95 | 0.02 |
| 1 | 131072 | 32 | 8 | 128 | True | torch.float16 | 465.61 | 302.27 | 0.01 |
| 1 | 4096 | 64 | 8 | 128 | True | torch.float16 | 0.84 | 327.93 | 0.18 |
| 1 | 4096 | 128 | 8 | 128 | True | torch.float16 | 1.80 | 306.17 | 0.16 |
| 2 | 4096 | 32 | 8 | 128 | True | torch.bfloat16 | 0.79 | 346.49 | 0.21 |
| 1 | 8192 | 32 | 8 | 128 | True | torch.bfloat16 | 1.62 | 339.96 | 0.10 |
| 1 | 4096 | 64 | 8 | 128 | True | torch.bfloat16 | 0.78 | 351.38 | 0.19 |
| 1 | 4096 | 128 | 8 | 128 | True | torch.bfloat16 | 1.67 | 329.54 | 0.17 |
| 2 | 2048 | 32 | 8 | 128 | True | torch.bfloat16 | 0.23 | 300.84 | 0.37 |

## GroupQueryAttentionBwdOp

### tileops

| batch | seq_len | heads | heads_kv | dim | causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 32 | 8 | 128 | True | torch.bfloat16 | 3.95 | 174.12 | 0.07 |
| 1 | 8192 | 32 | 8 | 128 | True | torch.bfloat16 | 7.45 | 184.60 | 0.04 |
| 1 | 4096 | 64 | 8 | 128 | True | torch.bfloat16 | 6.68 | 102.91 | 0.04 |
| 1 | 4096 | 128 | 8 | 128 | True | torch.bfloat16 | 8.72 | 157.61 | 0.05 |
| 2 | 2048 | 32 | 8 | 128 | True | torch.bfloat16 | 1.11 | 154.91 | 0.12 |

### fa3

| batch | seq_len | heads | heads_kv | dim | causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 32 | 8 | 128 | True | torch.bfloat16 | 6.31 | 108.88 | 0.04 |
| 1 | 8192 | 32 | 8 | 128 | True | torch.bfloat16 | 11.89 | 115.64 | 0.02 |
| 1 | 4096 | 64 | 8 | 128 | True | torch.bfloat16 | 6.26 | 109.80 | 0.04 |
| 1 | 4096 | 128 | 8 | 128 | True | torch.bfloat16 | 12.41 | 110.72 | 0.04 |
| 2 | 2048 | 32 | 8 | 128 | True | torch.bfloat16 | 1.73 | 99.50 | 0.08 |

## GroupQueryAttentionDecodeWithKVCacheOp

### tileops

| batch | heads | heads_kv | seq_len_kv | dim | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 8 | 4096 | 128 | torch.float16 | 0.01 | 5.07 | 1.27 |
| 1 | 32 | 8 | 32768 | 128 | torch.float16 | 0.06 | 9.70 | 2.43 |
| 1 | 64 | 8 | 4096 | 128 | torch.float16 | 0.02 | 6.86 | 0.86 |
| 1 | 128 | 8 | 8192 | 128 | torch.float16 | 0.02 | 28.28 | 1.77 |

### fa3

| batch | heads | heads_kv | seq_len_kv | dim | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 8 | 4096 | 128 | torch.float16 | 0.02 | 3.23 | 0.81 |
| 1 | 32 | 8 | 32768 | 128 | torch.float16 | 0.05 | 9.86 | 2.46 |
| 1 | 64 | 8 | 4096 | 128 | torch.float16 | 0.02 | 6.54 | 0.82 |
| 1 | 128 | 8 | 8192 | 128 | torch.float16 | 0.03 | 20.34 | 1.27 |

### flashinfer

| batch | heads | heads_kv | seq_len_kv | dim | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 8 | 4096 | 128 | torch.float16 | 0.02 | 3.12 | 0.78 |
| 1 | 32 | 8 | 32768 | 128 | torch.float16 | 0.08 | 6.46 | 1.62 |
| 1 | 64 | 8 | 4096 | 128 | torch.float16 | 0.03 | 4.82 | 0.60 |

## GroupQueryAttentionDecodePagedWithKVCacheOp

### tileops

| batch | heads | heads_kv | seqlen_kv | dim | page_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 32 | 8 | 4096 | 128 | 64 | torch.float16 | 0.08 | 28.23 | 0.23 |
| 8 | 32 | 8 | 32768 | 128 | 64 | torch.float16 | 0.18 | 23.68 | 0.74 |
| 64 | 32 | 8 | 2048 | 128 | 64 | torch.float16 | 0.09 | 23.25 | 0.10 |
| 8 | 64 | 8 | 4096 | 128 | 64 | torch.float16 | 0.04 | 26.31 | 0.42 |
| 32 | 32 | 8 | 4096 | 128 | 256 | torch.float16 | 0.09 | 23.28 | 0.19 |
| 8 | 64 | 8 | 4096 | 128 | 256 | torch.float16 | 0.04 | 29.86 | 0.47 |
| 4 | 128 | 8 | 4096 | 128 | 256 | torch.float16 | 0.03 | 35.23 | 0.56 |

### flashinfer

| batch | heads | heads_kv | seqlen_kv | dim | page_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 32 | 8 | 4096 | 128 | 64 | torch.float16 | 0.17 | 12.88 | 0.10 |
| 8 | 32 | 8 | 32768 | 128 | 64 | torch.float16 | 0.44 | 9.78 | 0.31 |
| 64 | 32 | 8 | 2048 | 128 | 64 | torch.float16 | 0.22 | 9.72 | 0.04 |
| 8 | 64 | 8 | 4096 | 128 | 64 | torch.float16 | 0.05 | 20.82 | 0.33 |
| 32 | 32 | 8 | 4096 | 128 | 256 | torch.float16 | 0.18 | 12.12 | 0.10 |
| 8 | 64 | 8 | 4096 | 128 | 256 | torch.float16 | 0.07 | 15.34 | 0.24 |

### fa3

| batch | heads | heads_kv | seqlen_kv | dim | page_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 32 | 8 | 4096 | 128 | 256 | torch.float16 | 0.17 | 12.45 | 0.10 |
| 8 | 64 | 8 | 4096 | 128 | 256 | torch.float16 | 0.04 | 24.68 | 0.39 |
| 4 | 128 | 8 | 4096 | 128 | 256 | torch.float16 | 0.03 | 36.89 | 0.59 |

## GqaSlidingWindowFwdOp

### tileops

| batch | seq | heads | heads_kv | dim | is_causal | wl | wr | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 512 | 8 | 2 | 64 | True | -1 | -1 | torch.float16 | 0.01 | 74.21 | 0.36 |
| 2 | 512 | 8 | 2 | 64 | True | 128 | -1 | torch.float16 | 0.01 | 44.09 | 0.49 |
| 2 | 768 | 8 | 2 | 64 | False | 256 | -1 | torch.float16 | 0.01 | 151.79 | 0.32 |
| 2 | 512 | 8 | 2 | 64 | False | 64 | 64 | torch.bfloat16 | 0.01 | 46.88 | 0.48 |
| 1 | 2048 | 8 | 2 | 64 | True | 512 | -1 | torch.float16 | 0.01 | 149.99 | 0.42 |

### fa3

| batch | seq | heads | heads_kv | dim | is_causal | wl | wr | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 512 | 8 | 2 | 64 | True | -1 | -1 | torch.float16 | 0.01 | 38.29 | 0.19 |
| 2 | 512 | 8 | 2 | 64 | True | 128 | -1 | torch.float16 | 0.01 | 16.03 | 0.18 |
| 2 | 768 | 8 | 2 | 64 | False | 256 | -1 | torch.float16 | 0.03 | 64.97 | 0.14 |
| 2 | 512 | 8 | 2 | 64 | False | 64 | 64 | torch.bfloat16 | 0.01 | 17.14 | 0.18 |
| 1 | 2048 | 8 | 2 | 64 | True | 512 | -1 | torch.float16 | 0.02 | 75.96 | 0.21 |

### flashinfer

| batch | seq | heads | heads_kv | dim | is_causal | wl | wr | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 512 | 8 | 2 | 64 | True | -1 | -1 | torch.float16 | 0.01 | 44.15 | 0.22 |
| 2 | 512 | 8 | 2 | 64 | True | 128 | -1 | torch.float16 | 0.01 | 19.42 | 0.22 |
| 2 | 768 | 8 | 2 | 64 | False | 256 | -1 | torch.float16 | 0.02 | 113.03 | 0.24 |
| 1 | 2048 | 8 | 2 | 64 | True | 512 | -1 | torch.float16 | 0.02 | 89.93 | 0.25 |

## GqaSlidingWindowVarlenFwdOp

### tileops

| batch | heads | heads_kv | dim | is_causal | wl | wr | dtype | max_seqlen_q | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 8 | 128 | True | -1 | -1 | torch.float16 | 3000 | 0.22 | 339.21 | 0.28 |
| 2 | 32 | 8 | 128 | True | 2000 | -1 | torch.float16 | 3073 | 0.35 | 252.08 | 0.27 |
| 4 | 32 | 8 | 128 | False | 1500 | 1500 | torch.float16 | 3500 | 0.94 | 366.75 | 0.22 |
| 2 | 32 | 8 | 128 | True | -1 | -1 | torch.float16 | 1024 | 0.40 | 401.97 | 0.19 |
| 2 | 32 | 8 | 128 | True | -1 | -1 | torch.float16 | 8193 | 2.03 | 372.11 | 0.13 |

### fa3

| batch | heads | heads_kv | dim | is_causal | wl | wr | dtype | max_seqlen_q | max_seqlen_k | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 8 | 128 | True | -1 | -1 | torch.float16 | 3000 | 3000 | 0.44 | 166.37 | 0.14 |
| 2 | 32 | 8 | 128 | True | 2000 | -1 | torch.float16 | 3073 | 3073 | 0.64 | 136.84 | 0.15 |
| 4 | 32 | 8 | 128 | False | 1500 | 1500 | torch.float16 | 3500 | 3500 | 1.92 | 179.16 | 0.11 |
| 2 | 32 | 8 | 128 | True | -1 | -1 | torch.float16 | 1024 | 8192 | 0.91 | 177.13 | 0.08 |
| 2 | 32 | 8 | 128 | True | -1 | -1 | torch.float16 | 8193 | 8193 | 3.53 | 213.96 | 0.08 |

### flashinfer

| batch | heads | heads_kv | dim | is_causal | wl | wr | dtype | max_seqlen_q | max_seqlen_k | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 8 | 128 | True | -1 | -1 | torch.float16 | 3000 | 3000 | 0.23 | 317.15 | 0.26 |
| 2 | 32 | 8 | 128 | True | 2000 | -1 | torch.float16 | 3073 | 3073 | 0.29 | 301.93 | 0.33 |
| 2 | 32 | 8 | 128 | True | -1 | -1 | torch.float16 | 1024 | 8192 | 0.43 | 371.93 | 0.17 |
| 2 | 32 | 8 | 128 | True | -1 | -1 | torch.float16 | 8193 | 8193 | 2.04 | 369.72 | 0.13 |

## GroupNormOp

### tileops

| n | c | g | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 128 | 32 | torch.float16 | 0.01 | 0.35 | 0.28 |
| 8 | 128 | 32 | torch.bfloat16 | 0.01 | 0.36 | 0.29 |
| 4 | 256 | 32 | torch.float16 | 0.02 | 0.19 | 0.15 |
| 4 | 128 | 16 | torch.float16 | 0.02 | 0.12 | 0.10 |

### torch

| n | c | g | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 128 | 32 | torch.float16 | 0.02 | 0.35 | 0.28 |
| 8 | 128 | 32 | torch.bfloat16 | 0.02 | 0.35 | 0.28 |
| 4 | 256 | 32 | torch.float16 | 0.02 | 0.25 | 0.20 |
| 4 | 128 | 16 | torch.float16 | 0.02 | 0.15 | 0.12 |

## grouped_gemm_nt

### tileops

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_nt | 16384 | 4 | 4864 | 4096 | torch.float16 | False | True | 49.47 | 13.20 | 0.01 |

### torch-ref

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_nt | 16384 | 4 | 4864 | 4096 | torch.float16 | False | True | 1.59 | 410.50 | 0.28 |

## grouped_gemm_nn

### tileops

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_nn | 16384 | 4 | 4864 | 4096 | torch.float16 | False | False | 54.13 | 12.06 | 0.01 |

### torch-ref

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_nn | 16384 | 4 | 4864 | 4096 | torch.float16 | False | False | 1.08 | 604.95 | 0.42 |

## grouped_gemm_tn

### tileops

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_tn | 16384 | 4 | 4864 | 4096 | torch.float16 | True | False | 12.01 | 54.34 | 0.04 |

### torch-ref

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_tn | 16384 | 4 | 4864 | 4096 | torch.float16 | True | False | 1.07 | 611.33 | 0.42 |

## grouped_gemm_tt

### tileops

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_tt | 16384 | 4 | 4864 | 4096 | torch.float16 | True | True | 14.97 | 43.62 | 0.03 |

### torch-ref

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_tt | 16384 | 4 | 4864 | 4096 | torch.float16 | True | True | 1.07 | 612.37 | 0.42 |

## GroupedGemmOp

### tileops

| batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16384 | 4 | 4864 | 4096 | torch.float16 | True | False | 130.46 | 15.01 | N/A |

### torch-ref

| batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16384 | 4 | 4864 | 4096 | torch.float16 | True | False | 4.83 | 405.35 | N/A |

## leaky_relu

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| leaky_relu | torch.float16 | 4194304 | 0.01 | 0.67 | 2.68 |
| leaky_relu | torch.bfloat16 | 4194304 | 0.01 | 0.67 | 2.68 |
| leaky_relu | torch.float32 | 4194304 | 0.01 | 0.40 | 3.18 |
| leaky_relu | torch.float16 | 10485760 | 0.01 | 0.84 | 3.36 |
| leaky_relu | torch.bfloat16 | 10485760 | 0.01 | 0.84 | 3.36 |
| leaky_relu | torch.float32 | 10485760 | 0.02 | 0.47 | 3.75 |
| leaky_relu | torch.float16 | 20971520 | 0.02 | 0.92 | 3.68 |
| leaky_relu | torch.bfloat16 | 20971520 | 0.02 | 0.94 | 3.74 |
| leaky_relu | torch.float32 | 20971520 | 0.06 | 0.38 | 3.03 |

### torch

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| leaky_relu | torch.float16 | 4194304 | 0.01 | 0.46 | 1.84 |
| leaky_relu | torch.bfloat16 | 4194304 | 0.01 | 0.50 | 2.00 |
| leaky_relu | torch.float32 | 4194304 | 0.01 | 0.29 | 2.31 |
| leaky_relu | torch.float16 | 10485760 | 0.02 | 0.58 | 2.30 |
| leaky_relu | torch.bfloat16 | 10485760 | 0.02 | 0.58 | 2.31 |
| leaky_relu | torch.float32 | 10485760 | 0.04 | 0.30 | 2.38 |
| leaky_relu | torch.float16 | 20971520 | 0.04 | 0.60 | 2.39 |
| leaky_relu | torch.bfloat16 | 20971520 | 0.04 | 0.60 | 2.39 |
| leaky_relu | torch.float32 | 20971520 | 0.06 | 0.33 | 2.65 |

## elu

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| elu | torch.float16 | 4194304 | 0.01 | 0.61 | 2.44 |
| elu | torch.bfloat16 | 4194304 | 0.01 | 0.62 | 2.46 |
| elu | torch.float32 | 4194304 | 0.01 | 0.39 | 3.15 |
| elu | torch.float16 | 10485760 | 0.01 | 0.77 | 3.06 |
| elu | torch.bfloat16 | 10485760 | 0.01 | 0.77 | 3.07 |
| elu | torch.float32 | 10485760 | 0.02 | 0.46 | 3.68 |
| elu | torch.float16 | 20971520 | 0.03 | 0.80 | 3.19 |
| elu | torch.bfloat16 | 20971520 | 0.03 | 0.80 | 3.20 |
| elu | torch.float32 | 20971520 | 0.06 | 0.37 | 2.93 |

### torch

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| elu | torch.float16 | 4194304 | 0.01 | 0.31 | 1.24 |
| elu | torch.bfloat16 | 4194304 | 0.01 | 0.31 | 1.22 |
| elu | torch.float32 | 4194304 | 0.02 | 0.26 | 2.10 |
| elu | torch.float16 | 10485760 | 0.03 | 0.33 | 1.32 |
| elu | torch.bfloat16 | 10485760 | 0.03 | 0.33 | 1.33 |
| elu | torch.float32 | 10485760 | 0.04 | 0.29 | 2.30 |
| elu | torch.float16 | 20971520 | 0.06 | 0.33 | 1.33 |
| elu | torch.bfloat16 | 20971520 | 0.06 | 0.33 | 1.32 |
| elu | torch.float32 | 20971520 | 0.07 | 0.29 | 2.30 |

## hardtanh

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| hardtanh | torch.float16 | 4194304 | 0.01 | 0.67 | 2.68 |
| hardtanh | torch.bfloat16 | 4194304 | 0.01 | 0.67 | 2.68 |
| hardtanh | torch.float32 | 4194304 | 0.01 | 0.40 | 3.20 |
| hardtanh | torch.float16 | 10485760 | 0.01 | 0.84 | 3.35 |
| hardtanh | torch.bfloat16 | 10485760 | 0.01 | 0.84 | 3.35 |
| hardtanh | torch.float32 | 10485760 | 0.02 | 0.46 | 3.67 |
| hardtanh | torch.float16 | 20971520 | 0.02 | 0.92 | 3.67 |
| hardtanh | torch.bfloat16 | 20971520 | 0.02 | 0.93 | 3.73 |
| hardtanh | torch.float32 | 20971520 | 0.06 | 0.38 | 3.03 |

### torch

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| hardtanh | torch.float16 | 4194304 | 0.01 | 0.53 | 2.11 |
| hardtanh | torch.bfloat16 | 4194304 | 0.01 | 0.56 | 2.24 |
| hardtanh | torch.float32 | 4194304 | 0.01 | 0.31 | 2.49 |
| hardtanh | torch.float16 | 10485760 | 0.02 | 0.58 | 2.32 |
| hardtanh | torch.bfloat16 | 10485760 | 0.02 | 0.62 | 2.47 |
| hardtanh | torch.float32 | 10485760 | 0.03 | 0.31 | 2.49 |
| hardtanh | torch.float16 | 20971520 | 0.04 | 0.57 | 2.29 |
| hardtanh | torch.bfloat16 | 20971520 | 0.03 | 0.62 | 2.49 |
| hardtanh | torch.float32 | 20971520 | 0.06 | 0.33 | 2.64 |

## softplus

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| softplus | torch.float16 | 4194304 | 0.01 | 0.43 | 1.72 |
| softplus | torch.bfloat16 | 4194304 | 0.01 | 0.42 | 1.67 |
| softplus | torch.float32 | 4194304 | 0.01 | 0.37 | 3.00 |
| softplus | torch.float16 | 10485760 | 0.02 | 0.51 | 2.06 |
| softplus | torch.bfloat16 | 10485760 | 0.02 | 0.51 | 2.03 |
| softplus | torch.float32 | 10485760 | 0.03 | 0.40 | 3.23 |
| softplus | torch.float16 | 20971520 | 0.05 | 0.44 | 1.76 |
| softplus | torch.bfloat16 | 20971520 | 0.05 | 0.43 | 1.71 |
| softplus | torch.float32 | 20971520 | 0.06 | 0.33 | 2.67 |

### torch

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| softplus | torch.float16 | 4194304 | 0.02 | 0.25 | 1.00 |
| softplus | torch.bfloat16 | 4194304 | 0.02 | 0.25 | 0.98 |
| softplus | torch.float32 | 4194304 | 0.02 | 0.23 | 1.85 |
| softplus | torch.float16 | 10485760 | 0.04 | 0.27 | 1.07 |
| softplus | torch.bfloat16 | 10485760 | 0.04 | 0.27 | 1.07 |
| softplus | torch.float32 | 10485760 | 0.04 | 0.25 | 2.04 |
| softplus | torch.float16 | 20971520 | 0.08 | 0.27 | 1.07 |
| softplus | torch.bfloat16 | 20971520 | 0.08 | 0.26 | 1.05 |
| softplus | torch.float32 | 20971520 | 0.07 | 0.29 | 2.34 |

## clamp

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| clamp | torch.float16 | 4194304 | 0.01 | 0.67 | 2.69 |
| clamp | torch.bfloat16 | 4194304 | 0.01 | 0.67 | 2.70 |
| clamp | torch.float32 | 4194304 | 0.01 | 0.40 | 3.19 |
| clamp | torch.float16 | 10485760 | 0.01 | 0.84 | 3.35 |
| clamp | torch.bfloat16 | 10485760 | 0.01 | 0.84 | 3.35 |
| clamp | torch.float32 | 10485760 | 0.02 | 0.47 | 3.74 |
| clamp | torch.float16 | 20971520 | 0.02 | 0.92 | 3.67 |
| clamp | torch.bfloat16 | 20971520 | 0.02 | 0.93 | 3.74 |
| clamp | torch.float32 | 20971520 | 0.06 | 0.38 | 3.03 |

### torch

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| clamp | torch.float16 | 4194304 | 0.01 | 0.48 | 1.91 |
| clamp | torch.bfloat16 | 4194304 | 0.01 | 0.50 | 1.99 |
| clamp | torch.float32 | 4194304 | 0.01 | 0.29 | 2.31 |
| clamp | torch.float16 | 10485760 | 0.02 | 0.55 | 2.19 |
| clamp | torch.bfloat16 | 10485760 | 0.02 | 0.57 | 2.29 |
| clamp | torch.float32 | 10485760 | 0.04 | 0.30 | 2.37 |
| clamp | torch.float16 | 20971520 | 0.04 | 0.54 | 2.17 |
| clamp | torch.bfloat16 | 20971520 | 0.04 | 0.59 | 2.36 |
| clamp | torch.float32 | 20971520 | 0.06 | 0.33 | 2.64 |

## nan_to_num

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| nan_to_num | torch.float16 | 4194304 | 0.01 | 0.65 | 2.62 |
| nan_to_num | torch.bfloat16 | 4194304 | 0.01 | 0.66 | 2.63 |
| nan_to_num | torch.float32 | 4194304 | 0.01 | 0.40 | 3.20 |
| nan_to_num | torch.float16 | 10485760 | 0.01 | 0.82 | 3.28 |
| nan_to_num | torch.bfloat16 | 10485760 | 0.01 | 0.82 | 3.28 |
| nan_to_num | torch.float32 | 10485760 | 0.02 | 0.46 | 3.67 |
| nan_to_num | torch.float16 | 20971520 | 0.02 | 0.90 | 3.59 |
| nan_to_num | torch.bfloat16 | 20971520 | 0.02 | 0.89 | 3.58 |
| nan_to_num | torch.float32 | 20971520 | 0.06 | 0.38 | 3.02 |

### torch

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| nan_to_num | torch.float16 | 4194304 | 0.01 | 0.52 | 2.06 |
| nan_to_num | torch.bfloat16 | 4194304 | 0.01 | 0.52 | 2.07 |
| nan_to_num | torch.float32 | 4194304 | 0.01 | 0.30 | 2.37 |
| nan_to_num | torch.float16 | 10485760 | 0.02 | 0.59 | 2.34 |
| nan_to_num | torch.bfloat16 | 10485760 | 0.02 | 0.58 | 2.34 |
| nan_to_num | torch.float32 | 10485760 | 0.03 | 0.30 | 2.41 |
| nan_to_num | torch.float16 | 20971520 | 0.04 | 0.59 | 2.37 |
| nan_to_num | torch.bfloat16 | 20971520 | 0.04 | 0.59 | 2.37 |
| nan_to_num | torch.float32 | 20971520 | 0.06 | 0.33 | 2.64 |

## PreluOp

### tileops

| num_channels | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 128 | torch.float16 | 131072 | 0.00 | 0.07 | 0.27 |
| 128 | torch.bfloat16 | 131072 | 0.00 | 0.07 | 0.27 |
| 128 | torch.float32 | 131072 | 0.00 | 0.06 | 0.50 |
| 4096 | torch.float16 | 4194304 | 0.01 | 0.67 | 2.69 |
| 4096 | torch.bfloat16 | 4194304 | 0.01 | 0.68 | 2.72 |
| 4096 | torch.float32 | 4194304 | 0.01 | 0.40 | 3.22 |
| 10240 | torch.float16 | 10485760 | 0.01 | 0.83 | 3.34 |
| 10240 | torch.bfloat16 | 10485760 | 0.01 | 0.84 | 3.36 |
| 10240 | torch.float32 | 10485760 | 0.02 | 0.47 | 3.73 |
| 20480 | torch.float16 | 20971520 | 0.02 | 0.93 | 3.72 |
| 20480 | torch.bfloat16 | 20971520 | 0.02 | 0.93 | 3.72 |
| 20480 | torch.float32 | 20971520 | 0.06 | 0.38 | 3.04 |

### torch

| num_channels | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 128 | torch.float16 | 131072 | 0.01 | 0.03 | 0.10 |
| 128 | torch.bfloat16 | 131072 | 0.00 | 0.03 | 0.12 |
| 128 | torch.float32 | 131072 | 0.00 | 0.03 | 0.27 |
| 4096 | torch.float16 | 4194304 | 0.02 | 0.20 | 0.82 |
| 4096 | torch.bfloat16 | 4194304 | 0.02 | 0.20 | 0.80 |
| 4096 | torch.float32 | 4194304 | 0.02 | 0.18 | 1.41 |
| 10240 | torch.float16 | 10485760 | 0.05 | 0.21 | 0.83 |
| 10240 | torch.bfloat16 | 10485760 | 0.05 | 0.21 | 0.83 |
| 10240 | torch.float32 | 10485760 | 0.06 | 0.17 | 1.39 |
| 20480 | torch.float16 | 20971520 | 0.10 | 0.21 | 0.82 |
| 20480 | torch.bfloat16 | 20971520 | 0.10 | 0.20 | 0.81 |
| 20480 | torch.float32 | 20971520 | 0.10 | 0.20 | 1.60 |

## WhereOp

### tileops

| dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| torch.float16 | 4194304 | 0.01 | 0.41 | 2.90 |
| torch.bfloat16 | 4194304 | 0.01 | 0.41 | 2.89 |
| torch.float32 | 4194304 | 0.02 | 0.25 | 3.29 |
| torch.float16 | 10485760 | 0.02 | 0.49 | 3.46 |
| torch.bfloat16 | 10485760 | 0.02 | 0.50 | 3.47 |
| torch.float32 | 10485760 | 0.04 | 0.25 | 3.28 |
| torch.float16 | 20971520 | 0.05 | 0.46 | 3.19 |
| torch.bfloat16 | 20971520 | 0.05 | 0.46 | 3.19 |
| torch.float32 | 20971520 | 0.10 | 0.21 | 2.79 |
| torch.float8_e4m3fn | 4194304 | 0.01 | 0.51 | 2.03 |
| torch.float8_e5m2 | 4194304 | 0.01 | 0.51 | 2.03 |
| torch.float8_e4m3fn | 10485760 | 0.02 | 0.59 | 2.34 |
| torch.float8_e5m2 | 10485760 | 0.02 | 0.59 | 2.34 |
| torch.float8_e4m3fn | 20971520 | 0.04 | 0.59 | 2.36 |
| torch.float8_e5m2 | 20971520 | 0.04 | 0.57 | 2.28 |

### torch

| dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| torch.float16 | 4194304 | 0.01 | 0.31 | 2.16 |
| torch.bfloat16 | 4194304 | 0.01 | 0.31 | 2.20 |
| torch.float32 | 4194304 | 0.02 | 0.18 | 2.30 |
| torch.float16 | 10485760 | 0.03 | 0.33 | 2.32 |
| torch.bfloat16 | 10485760 | 0.03 | 0.33 | 2.32 |
| torch.float32 | 10485760 | 0.05 | 0.20 | 2.55 |
| torch.float16 | 20971520 | 0.06 | 0.37 | 2.57 |
| torch.bfloat16 | 20971520 | 0.06 | 0.37 | 2.57 |
| torch.float32 | 20971520 | 0.10 | 0.21 | 2.72 |
| torch.float8_e4m3fn | 4194304 | 0.01 | 0.47 | 1.89 |
| torch.float8_e5m2 | 4194304 | 0.01 | 0.47 | 1.89 |
| torch.float8_e4m3fn | 10485760 | 0.02 | 0.53 | 2.11 |
| torch.float8_e5m2 | 10485760 | 0.02 | 0.60 | 2.38 |
| torch.float8_e4m3fn | 20971520 | 0.04 | 0.55 | 2.22 |
| torch.float8_e5m2 | 20971520 | 0.04 | 0.55 | 2.20 |

## MaskedFillOp

### tileops

| dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| torch.float16 | 4194304 | 0.01 | 0.55 | 2.75 |
| torch.bfloat16 | 4194304 | 0.01 | 0.56 | 2.78 |
| torch.float32 | 4194304 | 0.01 | 0.35 | 3.19 |
| torch.float16 | 10485760 | 0.02 | 0.67 | 3.34 |
| torch.bfloat16 | 10485760 | 0.02 | 0.67 | 3.35 |
| torch.float32 | 10485760 | 0.03 | 0.41 | 3.70 |
| torch.float16 | 20971520 | 0.03 | 0.72 | 3.58 |
| torch.bfloat16 | 20971520 | 0.03 | 0.72 | 3.60 |
| torch.float32 | 20971520 | 0.06 | 0.33 | 3.01 |
| torch.float8_e4m3fn | 4194304 | 0.01 | 0.58 | 1.73 |
| torch.float8_e5m2 | 4194304 | 0.01 | 0.33 | 0.98 |
| torch.float8_e4m3fn | 10485760 | 0.01 | 0.74 | 2.23 |
| torch.float8_e5m2 | 10485760 | 0.03 | 0.40 | 1.20 |
| torch.float8_e4m3fn | 20971520 | 0.03 | 0.81 | 2.44 |
| torch.float8_e5m2 | 20971520 | 0.06 | 0.34 | 1.03 |

### torch

| dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| torch.float16 | 4194304 | 0.01 | 0.30 | 1.50 |
| torch.bfloat16 | 4194304 | 0.01 | 0.30 | 1.50 |
| torch.float32 | 4194304 | 0.02 | 0.17 | 1.51 |
| torch.float16 | 10485760 | 0.03 | 0.31 | 1.57 |
| torch.bfloat16 | 10485760 | 0.03 | 0.32 | 1.58 |
| torch.float32 | 10485760 | 0.07 | 0.14 | 1.27 |
| torch.float16 | 20971520 | 0.08 | 0.26 | 1.32 |
| torch.bfloat16 | 20971520 | 0.08 | 0.26 | 1.32 |
| torch.float32 | 20971520 | 0.14 | 0.15 | 1.31 |

### torch-ref

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| masked_fill | torch.float8_e4m3fn | 4194304 | 0.03 | 0.12 | 0.37 |
| masked_fill | torch.float8_e5m2 | 4194304 | 0.03 | 0.13 | 0.39 |
| masked_fill | torch.float8_e4m3fn | 10485760 | 0.10 | 0.11 | 0.33 |
| masked_fill | torch.float8_e5m2 | 10485760 | 0.09 | 0.11 | 0.34 |
| masked_fill | torch.float8_e4m3fn | 20971520 | 0.24 | 0.09 | 0.27 |
| masked_fill | torch.float8_e5m2 | 20971520 | 0.23 | 0.09 | 0.27 |

## alibi

### tileops

| op_name | seq_len | dim | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| alibi | 512 | 64 | torch.float16 | 0.04 | 0.41 | 0.81 |
| alibi | 512 | 64 | torch.bfloat16 | 0.04 | 0.41 | 0.82 |
| alibi | 512 | 64 | torch.float32 | 0.02 | 0.87 | 3.48 |
| alibi | 2048 | 64 | torch.float16 | 0.43 | 0.62 | 1.24 |
| alibi | 2048 | 64 | torch.bfloat16 | 0.43 | 0.62 | 1.24 |
| alibi | 2048 | 64 | torch.float32 | 0.42 | 0.64 | 2.55 |
| alibi | 4096 | 128 | torch.float16 | 1.61 | 1.33 | 2.66 |
| alibi | 4096 | 128 | torch.bfloat16 | 1.67 | 1.29 | 2.57 |
| alibi | 4096 | 128 | torch.float32 | 3.18 | 0.68 | 2.70 |

### torch-ref

| op_name | seq_len | dim | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| alibi | 512 | 64 | torch.float16 | 0.08 | 0.20 | 0.41 |
| alibi | 512 | 64 | torch.bfloat16 | 0.08 | 0.20 | 0.41 |
| alibi | 512 | 64 | torch.float32 | 0.06 | 0.30 | 1.21 |
| alibi | 2048 | 64 | torch.float16 | 1.97 | 0.14 | 0.27 |
| alibi | 2048 | 64 | torch.bfloat16 | 1.97 | 0.14 | 0.27 |
| alibi | 2048 | 64 | torch.float32 | 1.23 | 0.22 | 0.87 |
| alibi | 4096 | 128 | torch.float16 | 18.55 | 0.12 | 0.23 |
| alibi | 4096 | 128 | torch.bfloat16 | 18.55 | 0.12 | 0.23 |
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
| sinusoidal | 4096 | 512 | torch.bfloat16 | 0.01 | 0.32 | 0.63 |
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
| leaky_relu | torch.float8_e4m3fn | 4194304 | 0.03 | 0.14 | 0.29 |
| leaky_relu | torch.float8_e5m2 | 4194304 | 0.03 | 0.15 | 0.30 |
| leaky_relu | torch.float8_e4m3fn | 10485760 | 0.08 | 0.13 | 0.26 |
| leaky_relu | torch.float8_e5m2 | 10485760 | 0.08 | 0.14 | 0.27 |
| leaky_relu | torch.float8_e4m3fn | 20971520 | 0.19 | 0.11 | 0.22 |
| leaky_relu | torch.float8_e5m2 | 20971520 | 0.18 | 0.11 | 0.23 |

## elu_fp8

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| elu | torch.float8_e4m3fn | 4194304 | 0.01 | 0.68 | 1.35 |
| elu | torch.float8_e5m2 | 4194304 | 0.02 | 0.26 | 0.52 |
| elu | torch.float8_e4m3fn | 10485760 | 0.01 | 0.85 | 1.70 |
| elu | torch.float8_e5m2 | 10485760 | 0.03 | 0.31 | 0.62 |
| elu | torch.float8_e4m3fn | 20971520 | 0.02 | 0.95 | 1.89 |
| elu | torch.float8_e5m2 | 20971520 | 0.09 | 0.23 | 0.47 |

### torch-ref

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| elu | torch.float8_e4m3fn | 4194304 | 0.04 | 0.12 | 0.23 |
| elu | torch.float8_e5m2 | 4194304 | 0.03 | 0.12 | 0.24 |
| elu | torch.float8_e4m3fn | 10485760 | 0.10 | 0.11 | 0.21 |
| elu | torch.float8_e5m2 | 10485760 | 0.10 | 0.11 | 0.22 |
| elu | torch.float8_e4m3fn | 20971520 | 0.22 | 0.10 | 0.19 |
| elu | torch.float8_e5m2 | 20971520 | 0.21 | 0.10 | 0.20 |

## clamp_fp8

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| clamp | torch.float8_e4m3fn | 4194304 | 0.00 | 1.00 | 2.01 |
| clamp | torch.float8_e5m2 | 4194304 | 0.01 | 0.40 | 0.81 |
| clamp | torch.float8_e4m3fn | 10485760 | 0.01 | 1.33 | 2.66 |
| clamp | torch.float8_e5m2 | 10485760 | 0.02 | 0.49 | 0.99 |
| clamp | torch.float8_e4m3fn | 20971520 | 0.01 | 1.57 | 3.14 |
| clamp | torch.float8_e5m2 | 20971520 | 0.05 | 0.46 | 0.91 |

### torch-ref

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| clamp | torch.float8_e4m3fn | 4194304 | 0.03 | 0.14 | 0.28 |
| clamp | torch.float8_e5m2 | 4194304 | 0.03 | 0.15 | 0.29 |
| clamp | torch.float8_e4m3fn | 10485760 | 0.08 | 0.13 | 0.25 |
| clamp | torch.float8_e5m2 | 10485760 | 0.08 | 0.13 | 0.27 |
| clamp | torch.float8_e4m3fn | 20971520 | 0.19 | 0.11 | 0.22 |
| clamp | torch.float8_e5m2 | 20971520 | 0.19 | 0.11 | 0.22 |

## InstanceNormOp

### tileops

| n | c | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 8 | 128 | torch.float16 | 0.02 | 0.34 | 0.27 |
| 8 | 128 | torch.bfloat16 | 0.01 | 0.36 | 0.28 |
| 4 | 256 | torch.float16 | 0.02 | 0.22 | 0.18 |
| 4 | 64 | torch.float16 | 0.01 | 0.08 | 0.07 |

### torch

| n | c | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 8 | 128 | torch.float16 | 0.02 | 0.26 | 0.20 |
| 8 | 128 | torch.bfloat16 | 0.02 | 0.25 | 0.20 |
| 4 | 256 | torch.float16 | 0.02 | 0.20 | 0.16 |
| 4 | 64 | torch.float16 | 0.01 | 0.09 | 0.07 |

## LayerNormOp

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 2048 | 4096 | torch.float16 | 0.03 | 1.65 | 1.32 |
| 2048 | 4096 | torch.bfloat16 | 0.04 | 0.97 | 0.77 |
| 1 | 4096 | torch.bfloat16 | 0.00 | 0.01 | 0.01 |
| 2048 | 8192 | torch.float16 | 0.04 | 2.23 | 1.79 |
| 2048 | 8192 | torch.bfloat16 | 0.04 | 2.04 | 1.63 |
| 1 | 8192 | torch.bfloat16 | 0.00 | 0.01 | 0.02 |
| 2048 | 16384 | torch.float16 | 0.12 | 1.39 | 1.12 |
| 2048 | 16384 | torch.bfloat16 | 0.12 | 1.34 | 1.08 |
| 1 | 16384 | torch.bfloat16 | 0.01 | 0.01 | 0.02 |

### torch

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 2048 | 4096 | torch.float16 | 0.02 | 1.69 | 1.35 |
| 2048 | 4096 | torch.bfloat16 | 0.02 | 1.69 | 1.36 |
| 1 | 4096 | torch.bfloat16 | 0.01 | 0.00 | 0.00 |
| 2048 | 8192 | torch.float16 | 0.06 | 1.52 | 1.22 |
| 2048 | 8192 | torch.bfloat16 | 0.06 | 1.51 | 1.21 |
| 1 | 8192 | torch.bfloat16 | 0.02 | 0.00 | 0.00 |
| 2048 | 16384 | torch.float16 | 0.10 | 1.68 | 1.35 |
| 2048 | 16384 | torch.bfloat16 | 0.10 | 1.67 | 1.34 |
| 1 | 16384 | torch.bfloat16 | 0.05 | 0.00 | 0.00 |

## AnyOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | any | 0.01 | 0.41 | 0.81 |
| 1024 | 4096 | torch.bfloat16 | any | 0.01 | 0.40 | 0.81 |
| 1024 | 4096 | torch.float32 | any | 0.01 | 0.29 | 1.16 |
| 1024 | 4096 | torch.int32 | any | 0.03 | 0.15 | 0.61 |
| 4096 | 4096 | torch.float16 | any | 0.03 | 0.64 | 1.29 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | any | 0.04 | 0.12 | 0.24 |
| 1024 | 4096 | torch.bfloat16 | any | 0.04 | 0.12 | 0.24 |
| 1024 | 4096 | torch.float32 | any | 0.04 | 0.11 | 0.45 |
| 1024 | 4096 | torch.int32 | any | 0.04 | 0.11 | 0.45 |
| 4096 | 4096 | torch.float16 | any | 0.11 | 0.15 | 0.30 |

## AllOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | all | 0.01 | 0.41 | 0.81 |
| 1024 | 4096 | torch.bfloat16 | all | 0.01 | 0.40 | 0.81 |
| 1024 | 4096 | torch.float32 | all | 0.01 | 0.29 | 1.16 |
| 1024 | 4096 | torch.int32 | all | 0.03 | 0.15 | 0.61 |
| 4096 | 4096 | torch.float16 | all | 0.03 | 0.64 | 1.29 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | all | 0.03 | 0.12 | 0.24 |
| 1024 | 4096 | torch.bfloat16 | all | 0.03 | 0.12 | 0.24 |
| 1024 | 4096 | torch.float32 | all | 0.04 | 0.12 | 0.47 |
| 1024 | 4096 | torch.int32 | all | 0.04 | 0.12 | 0.47 |
| 4096 | 4096 | torch.float16 | all | 0.11 | 0.15 | 0.30 |

## CountNonzeroOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | count_nonzero | 0.01 | 0.56 | 1.13 |
| 1024 | 4096 | torch.bfloat16 | count_nonzero | 0.01 | 0.56 | 1.12 |
| 1024 | 4096 | torch.float32 | count_nonzero | 0.01 | 0.36 | 1.44 |
| 1024 | 4096 | torch.int32 | count_nonzero | 0.02 | 0.17 | 0.69 |
| 4096 | 4096 | torch.float16 | count_nonzero | 0.02 | 0.72 | 1.43 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | count_nonzero | 0.05 | 0.09 | 0.18 |
| 1024 | 4096 | torch.bfloat16 | count_nonzero | 0.05 | 0.09 | 0.18 |
| 1024 | 4096 | torch.float32 | count_nonzero | 0.05 | 0.08 | 0.34 |
| 1024 | 4096 | torch.int32 | count_nonzero | 0.05 | 0.08 | 0.34 |
| 4096 | 4096 | torch.float16 | count_nonzero | 0.19 | 0.09 | 0.18 |

## MeanPoolingForwardOp

### tileops

| batch_size | seq_len | heads | dim | chunk_size | dtype | accum_dtype | chunks_per_bacth | seq_num | use_offsets | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8192 | 64 | 128 | 64 | torch.float16 | torch.float32 | 128 | 1 | 0 | 0.23 | 0.29 | 0.60 |
| 2 | 2048 | 64 | 128 | 64 | torch.float16 | torch.float32 | 32 | 2 | 0 | 0.11 | 0.31 | 0.64 |
| 1 | 8192 | 64 | 128 | 64 | torch.float16 | torch.float32 | 128 | 4 | 1 | 0.30 | 0.23 | 0.46 |
| 1 | 1000 | 64 | 128 | 32 | torch.float16 | torch.float32 | 34 | 4 | 1 | 0.02 | 0.39 | 0.74 |

### torch-ref

| batch_size | seq_len | heads | dim | chunk_size | dtype | accum_dtype | chunks_per_bacth | seq_num | use_offsets | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8192 | 64 | 128 | 64 | torch.float16 | torch.float32 | 128 | 1 | 0 | 0.77 | 0.09 | 0.18 |
| 2 | 2048 | 64 | 128 | 64 | torch.float16 | torch.float32 | 32 | 2 | 0 | 0.27 | 0.13 | 0.26 |
| 1 | 8192 | 64 | 128 | 64 | torch.float16 | torch.float32 | 128 | 4 | 1 | 0.80 | 0.08 | 0.17 |
| 1 | 1000 | 64 | 128 | 32 | torch.float16 | torch.float32 | 34 | 4 | 1 | 0.27 | 0.03 | 0.06 |

## MultiHeadAttentionFwdOp

### tileops

| batch | seq_len | heads | dim | causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 8 | 64 | False | torch.float16 | 0.01 | 211.54 | 0.41 |
| 16 | 2048 | 16 | 128 | False | torch.float16 | 1.53 | 358.95 | 0.35 |
| 4 | 4096 | 16 | 128 | False | torch.bfloat16 | 1.40 | 393.81 | 0.19 |

### fa3

| batch | seq_len | heads | dim | causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 8 | 64 | False | torch.float16 | 0.02 | 123.00 | 0.24 |
| 16 | 2048 | 16 | 128 | False | torch.float16 | 2.32 | 236.56 | 0.23 |
| 4 | 4096 | 16 | 128 | False | torch.bfloat16 | 2.22 | 247.63 | 0.12 |

### flashinfer

| batch | seq_len | heads | dim | causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 8 | 64 | False | torch.float16 | 0.02 | 120.81 | 0.24 |
| 16 | 2048 | 16 | 128 | False | torch.float16 | 1.68 | 326.49 | 0.32 |
| 4 | 4096 | 16 | 128 | False | torch.bfloat16 | 1.54 | 357.10 | 0.17 |

## MultiHeadAttentionBwdOp

### tileops

| batch | seq_len | heads | dim | causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 8 | 64 | False | torch.float16 | 0.05 | 119.30 | 0.16 |
| 16 | 2048 | 16 | 128 | False | torch.float16 | 7.17 | 191.66 | 0.13 |
| 4 | 4096 | 16 | 128 | False | torch.bfloat16 | 6.52 | 210.73 | 0.07 |

### fa3

| batch | seq_len | heads | dim | causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 8 | 64 | False | torch.float16 | 0.07 | 81.50 | 0.11 |
| 16 | 2048 | 16 | 128 | False | torch.float16 | 11.68 | 117.64 | 0.08 |
| 4 | 4096 | 16 | 128 | False | torch.bfloat16 | 10.79 | 127.40 | 0.04 |

## MultiHeadAttentionDecodeWithKVCacheOp

### tileops

| b | h | s_q | s_kv | d | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 128 | 8192 | 128 | torch.float16 | 0.05 | 316.26 | 2.51 |
| 1 | 32 | 128 | 8192 | 128 | torch.bfloat16 | 0.07 | 248.57 | 1.97 |
| 1 | 32 | 128 | 5 | 128 | torch.float16 | 0.00 | 2.21 | 0.46 |

### fa3

| b | h | s_q | s_kv | d | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 128 | 8192 | 128 | torch.float16 | 0.07 | 231.24 | 1.83 |
| 1 | 32 | 128 | 8192 | 128 | torch.bfloat16 | 0.07 | 237.28 | 1.88 |
| 1 | 32 | 128 | 5 | 128 | torch.float16 | 0.01 | 1.60 | 0.33 |

### flashinfer

| b | h | s_q | s_kv | d | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 128 | 8192 | 128 | torch.float16 | 0.14 | 124.85 | 0.99 |
| 1 | 32 | 128 | 8192 | 128 | torch.bfloat16 | 0.14 | 125.54 | 1.00 |
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
| 1 | 4 | 1280 | torch.bfloat16 | 0.00 | 30.45 | 0.01 |
| 2 | 4 | 1920 | torch.bfloat16 | 0.00 | 130.60 | 0.01 |
| 4 | 4 | 2560 | torch.bfloat16 | 0.00 | 438.95 | 0.01 |

### torch-ref

| batch | n_expand | c_x | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | 1280 | torch.bfloat16 | 0.01 | 3.96 | 0.00 |
| 2 | 4 | 1920 | torch.bfloat16 | 0.01 | 16.84 | 0.00 |
| 4 | 4 | 2560 | torch.bfloat16 | 0.01 | 57.49 | 0.00 |

## ManifoldConstrainedHyperConnectionPreOp

### tileops

| batch | n_expand | c_x | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | 1280 | torch.bfloat16 | 0.04 | 30.49 | 0.00 |
| 2 | 4 | 1920 | torch.bfloat16 | 0.06 | 102.49 | 0.00 |
| 4 | 4 | 2560 | torch.bfloat16 | 0.07 | 286.73 | 0.00 |

### torch-ref

| batch | n_expand | c_x | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | 1280 | torch.bfloat16 | 0.28 | 4.51 | 0.00 |
| 2 | 4 | 1920 | torch.bfloat16 | 0.30 | 18.76 | 0.00 |
| 4 | 4 | 2560 | torch.bfloat16 | 0.35 | 57.88 | 0.00 |

## FusedMoe

### tileops

| num_tokens | num_experts | top_k | hidden_size | ffn_size | scoring_func | renormalize | with_correction_bias | routed_scaling_factor | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 384 | 8 | 7168 | 2048 | sigmoid | True | True | 2.827 | torch.bfloat16 | 0.25 | 2.84 | 136.09 |
| 32 | 384 | 8 | 7168 | 2048 | sigmoid | True | True | 2.827 | torch.bfloat16 | 4.76 | 4.74 | 7.10 |
| 512 | 384 | 8 | 7168 | 2048 | sigmoid | True | True | 2.827 | torch.bfloat16 | 9.05 | 39.88 | 3.74 |
| 4096 | 384 | 8 | 7168 | 2048 | sigmoid | True | True | 2.827 | torch.bfloat16 | 22.10 | 130.60 | 1.54 |
| 1 | 256 | 8 | 7168 | 2048 | sigmoid | True | False | 1.0 | torch.bfloat16 | 0.24 | 2.98 | 95.26 |
| 32 | 256 | 8 | 7168 | 2048 | sigmoid | True | False | 1.0 | torch.bfloat16 | 6.45 | 3.49 | 3.49 |
| 512 | 256 | 8 | 7168 | 2048 | sigmoid | True | False | 1.0 | torch.bfloat16 | 10.89 | 33.13 | 2.07 |
| 4096 | 256 | 8 | 7168 | 2048 | sigmoid | True | False | 1.0 | torch.bfloat16 | 21.15 | 136.47 | 1.07 |
| 1 | 128 | 8 | 7168 | 2048 | softmax | False | False | 1.0 | torch.bfloat16 | 0.22 | 3.13 | 50.11 |
| 32 | 128 | 8 | 7168 | 2048 | softmax | False | False | 1.0 | torch.bfloat16 | 4.59 | 4.92 | 2.46 |
| 512 | 128 | 8 | 7168 | 2048 | softmax | False | False | 1.0 | torch.bfloat16 | 5.53 | 65.21 | 2.04 |
| 4096 | 128 | 8 | 7168 | 2048 | softmax | False | False | 1.0 | torch.bfloat16 | 17.45 | 165.42 | 0.65 |
| 1 | 128 | 8 | 3072 | 1536 | softmax | False | False | 1.0 | torch.bfloat16 | 0.11 | 2.13 | 34.14 |
| 32 | 128 | 8 | 3072 | 1536 | softmax | False | False | 1.0 | torch.bfloat16 | 1.42 | 5.09 | 2.54 |
| 512 | 128 | 8 | 3072 | 1536 | softmax | False | False | 1.0 | torch.bfloat16 | 1.70 | 68.07 | 2.13 |
| 4096 | 128 | 8 | 3072 | 1536 | softmax | False | False | 1.0 | torch.bfloat16 | 5.13 | 180.96 | 0.72 |

### tileops-padded

| num_tokens | num_experts | top_k | hidden_size | ffn_size | scoring_func | renormalize | with_correction_bias | routed_scaling_factor | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 384 | 8 | 7168 | 2048 | sigmoid | True | True | 2.827 | torch.bfloat16 | 0.52 | 1.34 | 64.55 |
| 32 | 384 | 8 | 7168 | 2048 | sigmoid | True | True | 2.827 | torch.bfloat16 | 5.36 | 4.21 | 6.31 |
| 512 | 384 | 8 | 7168 | 2048 | sigmoid | True | True | 2.827 | torch.bfloat16 | 9.69 | 37.21 | 3.49 |
| 4096 | 384 | 8 | 7168 | 2048 | sigmoid | True | True | 2.827 | torch.bfloat16 | 23.17 | 124.55 | 1.46 |
| 1 | 256 | 8 | 7168 | 2048 | sigmoid | True | False | 1.0 | torch.bfloat16 | 0.44 | 1.60 | 51.18 |
| 32 | 256 | 8 | 7168 | 2048 | sigmoid | True | False | 1.0 | torch.bfloat16 | 7.08 | 3.19 | 3.19 |
| 512 | 256 | 8 | 7168 | 2048 | sigmoid | True | False | 1.0 | torch.bfloat16 | 11.63 | 31.03 | 1.94 |
| 4096 | 256 | 8 | 7168 | 2048 | sigmoid | True | False | 1.0 | torch.bfloat16 | 21.90 | 131.79 | 1.03 |
| 1 | 128 | 8 | 7168 | 2048 | softmax | False | False | 1.0 | torch.bfloat16 | 0.36 | 1.96 | 31.32 |
| 32 | 128 | 8 | 7168 | 2048 | softmax | False | False | 1.0 | torch.bfloat16 | 4.98 | 4.52 | 2.26 |
| 512 | 128 | 8 | 7168 | 2048 | softmax | False | False | 1.0 | torch.bfloat16 | 5.74 | 62.83 | 1.97 |
| 4096 | 128 | 8 | 7168 | 2048 | softmax | False | False | 1.0 | torch.bfloat16 | 17.73 | 162.75 | 0.64 |
| 1 | 128 | 8 | 3072 | 1536 | softmax | False | False | 1.0 | torch.bfloat16 | 0.14 | 1.63 | 26.07 |
| 32 | 128 | 8 | 3072 | 1536 | softmax | False | False | 1.0 | torch.bfloat16 | 1.68 | 4.31 | 2.15 |
| 512 | 128 | 8 | 3072 | 1536 | softmax | False | False | 1.0 | torch.bfloat16 | 1.92 | 60.46 | 1.89 |
| 4096 | 128 | 8 | 3072 | 1536 | softmax | False | False | 1.0 | torch.bfloat16 | 5.47 | 169.50 | 0.67 |

### vllm

| num_tokens | num_experts | top_k | hidden_size | ffn_size | scoring_func | renormalize | with_correction_bias | routed_scaling_factor | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 384 | 8 | 7168 | 2048 | sigmoid | True | True | 2.827 | torch.bfloat16 | 0.21 | 3.38 | 162.29 |
| 32 | 384 | 8 | 7168 | 2048 | sigmoid | True | True | 2.827 | torch.bfloat16 | 7.09 | 3.18 | 4.77 |
| 512 | 384 | 8 | 7168 | 2048 | sigmoid | True | True | 2.827 | torch.bfloat16 | 15.76 | 22.90 | 2.15 |
| 4096 | 384 | 8 | 7168 | 2048 | sigmoid | True | True | 2.827 | torch.bfloat16 | 20.68 | 139.60 | 1.64 |
| 1 | 256 | 8 | 7168 | 2048 | sigmoid | True | False | 1.0 | torch.bfloat16 | 0.21 | 3.42 | 109.40 |
| 32 | 256 | 8 | 7168 | 2048 | sigmoid | True | False | 1.0 | torch.bfloat16 | 5.91 | 3.82 | 3.82 |
| 512 | 256 | 8 | 7168 | 2048 | sigmoid | True | False | 1.0 | torch.bfloat16 | 10.40 | 34.68 | 2.17 |
| 4096 | 256 | 8 | 7168 | 2048 | sigmoid | True | False | 1.0 | torch.bfloat16 | 18.68 | 154.48 | 1.21 |
| 1 | 128 | 8 | 7168 | 2048 | softmax | False | False | 1.0 | torch.bfloat16 | 0.20 | 3.44 | 55.08 |
| 32 | 128 | 8 | 7168 | 2048 | softmax | False | False | 1.0 | torch.bfloat16 | 4.25 | 5.31 | 2.65 |
| 512 | 128 | 8 | 7168 | 2048 | softmax | False | False | 1.0 | torch.bfloat16 | 5.39 | 66.90 | 2.09 |
| 4096 | 128 | 8 | 7168 | 2048 | softmax | False | False | 1.0 | torch.bfloat16 | 15.47 | 186.60 | 0.74 |
| 1 | 128 | 8 | 3072 | 1536 | softmax | False | False | 1.0 | torch.bfloat16 | 0.08 | 2.86 | 45.81 |
| 32 | 128 | 8 | 3072 | 1536 | softmax | False | False | 1.0 | torch.bfloat16 | 1.27 | 5.73 | 2.86 |
| 512 | 128 | 8 | 3072 | 1536 | softmax | False | False | 1.0 | torch.bfloat16 | 1.64 | 70.73 | 2.21 |
| 4096 | 128 | 8 | 3072 | 1536 | softmax | False | False | 1.0 | torch.bfloat16 | 4.92 | 188.44 | 0.75 |

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
| 512 | 256 | 8 | sigmoid | True | torch.bfloat16 | True | 0.01 | 0.21 | 0.03 |
| 4096 | 256 | 8 | sigmoid | True | torch.bfloat16 | True | 0.02 | 0.91 | 0.11 |
| 1 | 128 | 8 | softmax | False | torch.bfloat16 | True | 0.01 | 0.00 | 0.00 |
| 32 | 128 | 8 | softmax | False | torch.bfloat16 | True | 0.01 | 0.01 | 0.00 |
| 512 | 128 | 8 | softmax | False | torch.bfloat16 | True | 0.01 | 0.14 | 0.02 |
| 4096 | 128 | 8 | softmax | False | torch.bfloat16 | True | 0.01 | 0.69 | 0.10 |

## MoePermutePaddedOp

### tileops

| total_tokens | top_k | num_experts | hidden_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8 | 384 | 7168 | torch.bfloat16 | 0.03 | 0.00 | 0.01 |
| 32 | 8 | 384 | 7168 | torch.bfloat16 | 0.03 | 0.00 | 0.15 |
| 512 | 8 | 384 | 7168 | torch.bfloat16 | 0.08 | 0.00 | 0.80 |
| 4096 | 8 | 384 | 7168 | torch.bfloat16 | 0.64 | 0.00 | 0.83 |
| 1 | 8 | 256 | 7168 | torch.bfloat16 | 0.02 | 0.00 | 0.01 |
| 32 | 8 | 256 | 7168 | torch.bfloat16 | 0.02 | 0.00 | 0.22 |
| 512 | 8 | 256 | 7168 | torch.bfloat16 | 0.07 | 0.00 | 0.92 |
| 4096 | 8 | 256 | 7168 | torch.bfloat16 | 0.62 | 0.00 | 0.85 |
| 1 | 8 | 128 | 7168 | torch.bfloat16 | 0.01 | 0.00 | 0.02 |
| 32 | 8 | 128 | 7168 | torch.bfloat16 | 0.01 | 0.00 | 0.44 |
| 512 | 8 | 128 | 7168 | torch.bfloat16 | 0.06 | 0.00 | 1.14 |
| 4096 | 8 | 128 | 7168 | torch.bfloat16 | 0.59 | 0.00 | 0.89 |
| 1 | 8 | 128 | 3072 | torch.bfloat16 | 0.01 | 0.00 | 0.01 |
| 32 | 8 | 128 | 3072 | torch.bfloat16 | 0.01 | 0.00 | 0.20 |
| 512 | 8 | 128 | 3072 | torch.bfloat16 | 0.03 | 0.00 | 1.07 |
| 4096 | 8 | 128 | 3072 | torch.bfloat16 | 0.31 | 0.00 | 0.74 |

### vllm

| total_tokens | top_k | num_experts | hidden_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8 | 384 | 7168 | torch.bfloat16 | 0.01 | 0.00 | 0.01 |
| 32 | 8 | 384 | 7168 | torch.bfloat16 | 0.01 | 0.00 | 0.31 |
| 512 | 8 | 384 | 7168 | torch.bfloat16 | 0.04 | 0.00 | 1.58 |
| 4096 | 8 | 384 | 7168 | torch.bfloat16 | 0.40 | 0.00 | 1.32 |
| 1 | 8 | 256 | 7168 | torch.bfloat16 | 0.01 | 0.00 | 0.01 |
| 32 | 8 | 256 | 7168 | torch.bfloat16 | 0.01 | 0.00 | 0.31 |
| 512 | 8 | 256 | 7168 | torch.bfloat16 | 0.04 | 0.00 | 1.59 |
| 4096 | 8 | 256 | 7168 | torch.bfloat16 | 0.40 | 0.00 | 1.33 |
| 1 | 8 | 128 | 7168 | torch.bfloat16 | 0.01 | 0.00 | 0.01 |
| 32 | 8 | 128 | 7168 | torch.bfloat16 | 0.01 | 0.00 | 0.31 |
| 512 | 8 | 128 | 7168 | torch.bfloat16 | 0.04 | 0.00 | 1.60 |
| 4096 | 8 | 128 | 7168 | torch.bfloat16 | 0.39 | 0.00 | 1.36 |
| 1 | 8 | 128 | 3072 | torch.bfloat16 | 0.01 | 0.00 | 0.01 |
| 32 | 8 | 128 | 3072 | torch.bfloat16 | 0.01 | 0.00 | 0.15 |
| 512 | 8 | 128 | 3072 | torch.bfloat16 | 0.03 | 0.00 | 1.06 |
| 4096 | 8 | 128 | 3072 | torch.bfloat16 | 0.16 | 0.00 | 1.41 |

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
| 4096 | 8 | 256 | 64 | 32768 | 48959 | 765 | 0.07 | 0.00 | 0.01 |
| 1 | 8 | 128 | 64 | 8 | 8135 | 128 | 0.01 | 0.00 | 0.00 |
| 32 | 8 | 128 | 64 | 256 | 8383 | 131 | 0.02 | 0.00 | 0.00 |
| 512 | 8 | 128 | 64 | 4096 | 12223 | 191 | 0.03 | 0.00 | 0.00 |
| 4096 | 8 | 128 | 64 | 32768 | 40895 | 639 | 0.07 | 0.00 | 0.00 |

## MoePermuteNopadOp

### tileops

| total_tokens | top_k | num_experts | hidden_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8 | 384 | 7168 | torch.bfloat16 | 0.01 | 0.00 | 0.02 |
| 32 | 8 | 384 | 7168 | torch.bfloat16 | 0.01 | 0.00 | 0.50 |
| 512 | 8 | 384 | 7168 | torch.bfloat16 | 0.03 | 0.00 | 1.89 |
| 4096 | 8 | 384 | 7168 | torch.bfloat16 | 0.55 | 0.00 | 0.95 |
| 1 | 8 | 256 | 7168 | torch.bfloat16 | 0.01 | 0.00 | 0.02 |
| 32 | 8 | 256 | 7168 | torch.bfloat16 | 0.01 | 0.00 | 0.56 |
| 512 | 8 | 256 | 7168 | torch.bfloat16 | 0.03 | 0.00 | 1.98 |
| 4096 | 8 | 256 | 7168 | torch.bfloat16 | 0.54 | 0.00 | 0.97 |
| 1 | 8 | 128 | 7168 | torch.bfloat16 | 0.00 | 0.00 | 0.03 |
| 32 | 8 | 128 | 7168 | torch.bfloat16 | 0.01 | 0.00 | 0.68 |
| 512 | 8 | 128 | 7168 | torch.bfloat16 | 0.03 | 0.00 | 2.11 |
| 4096 | 8 | 128 | 7168 | torch.bfloat16 | 0.53 | 0.00 | 1.00 |
| 1 | 8 | 128 | 3072 | torch.bfloat16 | 0.01 | 0.00 | 0.01 |
| 32 | 8 | 128 | 3072 | torch.bfloat16 | 0.01 | 0.00 | 0.31 |
| 512 | 8 | 128 | 3072 | torch.bfloat16 | 0.02 | 0.00 | 1.58 |
| 4096 | 8 | 128 | 3072 | torch.bfloat16 | 0.22 | 0.00 | 1.03 |

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
| 4096 | 8 | 256 | 7168 | torch.bfloat16 | 0.40 | 0.00 | 1.33 |
| 1 | 8 | 128 | 7168 | torch.bfloat16 | 0.01 | 0.00 | 0.01 |
| 32 | 8 | 128 | 7168 | torch.bfloat16 | 0.01 | 0.00 | 0.31 |
| 512 | 8 | 128 | 7168 | torch.bfloat16 | 0.04 | 0.00 | 1.59 |
| 4096 | 8 | 128 | 7168 | torch.bfloat16 | 0.39 | 0.00 | 1.36 |
| 1 | 8 | 128 | 3072 | torch.bfloat16 | 0.01 | 0.00 | 0.01 |
| 32 | 8 | 128 | 3072 | torch.bfloat16 | 0.01 | 0.00 | 0.15 |
| 512 | 8 | 128 | 3072 | torch.bfloat16 | 0.03 | 0.00 | 1.06 |
| 4096 | 8 | 128 | 3072 | torch.bfloat16 | 0.16 | 0.00 | 1.41 |

## moe_unpermute

### tileops

| total_tokens | top_k | hidden_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 8 | 7168 | torch.bfloat16 | 0.01 | 0.02 | 0.02 |
| 32 | 8 | 7168 | torch.bfloat16 | 0.01 | 0.46 | 0.52 |
| 512 | 8 | 7168 | torch.bfloat16 | 0.04 | 1.62 | 1.83 |
| 4096 | 8 | 7168 | torch.bfloat16 | 0.20 | 2.29 | 2.58 |
| 1 | 8 | 3072 | torch.bfloat16 | 0.01 | 0.01 | 0.01 |
| 32 | 8 | 3072 | torch.bfloat16 | 0.01 | 0.24 | 0.27 |
| 512 | 8 | 3072 | torch.bfloat16 | 0.01 | 1.81 | 2.04 |
| 4096 | 8 | 3072 | torch.bfloat16 | 0.09 | 2.16 | 2.44 |

### vllm

| total_tokens | top_k | hidden_size | dtype | numel | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8 | 7168 | torch.bfloat16 | 8 | 0.02 | 0.00 | 0.01 |
| 32 | 8 | 7168 | torch.bfloat16 | 256 | 0.03 | 0.14 | 0.15 |
| 512 | 8 | 7168 | torch.bfloat16 | 4096 | 0.04 | 1.34 | 1.50 |
| 4096 | 8 | 7168 | torch.bfloat16 | 32768 | 0.21 | 2.27 | 2.56 |
| 1 | 8 | 3072 | torch.bfloat16 | 8 | 0.01 | 0.00 | 0.00 |
| 32 | 8 | 3072 | torch.bfloat16 | 256 | 0.01 | 0.11 | 0.13 |
| 512 | 8 | 3072 | torch.bfloat16 | 4096 | 0.02 | 1.12 | 1.26 |
| 4096 | 8 | 3072 | torch.bfloat16 | 32768 | 0.12 | 1.68 | 1.89 |

## SumOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | sum | 0.01 | 0.56 | 1.11 |
| 1024 | 4096 | torch.bfloat16 | sum | 0.01 | 0.57 | 1.13 |
| 4096 | 4096 | torch.float16 | sum | 0.02 | 0.72 | 1.44 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | sum | 0.03 | 0.14 | 0.28 |
| 1024 | 4096 | torch.bfloat16 | sum | 0.03 | 0.14 | 0.28 |
| 4096 | 4096 | torch.float16 | sum | 0.13 | 0.13 | 0.27 |

## MeanOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | mean | 0.01 | 0.56 | 1.12 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | mean | 0.03 | 0.14 | 0.28 |

## AmaxOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | amax | 0.01 | 0.56 | 1.13 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | amax | 0.03 | 0.14 | 0.28 |

## AminOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | amin | 0.01 | 0.56 | 1.13 |

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
| 1024 | 4096 | torch.float16 | prod | 0.03 | 0.14 | 0.28 |

## StdOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | std | 0.01 | 1.46 | 0.97 |

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
| 1024 | 4096 | torch.float16 | var | 0.05 | 0.25 | 0.17 |

## VarMeanOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | var_mean | 0.01 | 1.60 | 1.07 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | var_mean | 0.05 | 0.23 | 0.15 |

## RmsNormOp

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 2048 | 4096 | torch.float16 | 0.02 | 1.45 | 1.45 |
| 2048 | 4096 | torch.bfloat16 | 0.02 | 1.94 | 1.94 |
| 1 | 4096 | torch.bfloat16 | 0.00 | 0.00 | 0.01 |
| 2048 | 8192 | torch.float16 | 0.03 | 2.01 | 2.01 |
| 2048 | 8192 | torch.bfloat16 | 0.03 | 1.99 | 1.99 |
| 1 | 8192 | torch.bfloat16 | 0.00 | 0.01 | 0.01 |
| 2048 | 16384 | torch.float16 | 0.10 | 1.30 | 1.30 |
| 2048 | 16384 | torch.bfloat16 | 0.11 | 1.27 | 1.27 |
| 1 | 16384 | torch.bfloat16 | 0.00 | 0.01 | 0.02 |

### torch-ref

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 2048 | 4096 | torch.float16 | 0.20 | 0.17 | 0.17 |
| 2048 | 4096 | torch.bfloat16 | 0.20 | 0.17 | 0.17 |
| 1 | 4096 | torch.bfloat16 | 0.02 | 0.00 | 0.00 |
| 2048 | 8192 | torch.float16 | 0.42 | 0.16 | 0.16 |
| 2048 | 8192 | torch.bfloat16 | 0.42 | 0.16 | 0.16 |
| 1 | 8192 | torch.bfloat16 | 0.02 | 0.00 | 0.00 |
| 2048 | 16384 | torch.float16 | 0.87 | 0.15 | 0.15 |
| 2048 | 16384 | torch.bfloat16 | 0.87 | 0.15 | 0.15 |
| 1 | 16384 | torch.bfloat16 | 0.02 | 0.00 | 0.00 |

## RopeNeoxOp

### tileops

| seq_len | head_dim | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 2048 | 64 | torch.float16 | 0.00 | 0.26 | 0.38 |
| 2048 | 64 | torch.bfloat16 | 0.00 | 0.25 | 0.38 |
| 2048 | 64 | torch.float32 | 0.00 | 0.23 | 0.68 |
| 2048 | 128 | torch.float16 | 0.00 | 0.45 | 0.67 |
| 2048 | 128 | torch.bfloat16 | 0.00 | 0.45 | 0.67 |
| 2048 | 128 | torch.float32 | 0.00 | 0.36 | 1.07 |
| 4096 | 128 | torch.float16 | 0.00 | 0.71 | 1.06 |
| 4096 | 128 | torch.bfloat16 | 0.00 | 0.70 | 1.06 |
| 4096 | 128 | torch.float32 | 0.00 | 0.54 | 1.63 |

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
| 1024 | 4096 | torch.float16 | 0.02 | 1.05 | 1.05 |
| 4096 | 4096 | torch.bfloat16 | 0.08 | 0.86 | 0.86 |
| 1024 | 3000 | torch.float16 | 0.03 | 0.49 | 0.49 |
| 1025 | 4096 | torch.float16 | 0.02 | 1.05 | 1.05 |

### torch

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.03 | 0.64 | 0.64 |
| 4096 | 4096 | torch.bfloat16 | 0.09 | 0.74 | 0.74 |
| 1024 | 3000 | torch.float16 | 0.02 | 0.56 | 0.56 |
| 1025 | 4096 | torch.float16 | 0.03 | 0.65 | 0.65 |

## LogSoftmaxOp

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.02 | 1.01 | 0.81 |
| 4096 | 4096 | torch.bfloat16 | 0.11 | 0.74 | 0.59 |
| 1024 | 3000 | torch.float16 | 0.03 | 0.52 | 0.42 |
| 1025 | 4096 | torch.float16 | 0.02 | 1.01 | 0.81 |

### torch

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.02 | 0.92 | 0.74 |
| 4096 | 4096 | torch.bfloat16 | 0.08 | 1.04 | 0.83 |
| 1024 | 3000 | torch.float16 | 0.02 | 0.80 | 0.64 |
| 1025 | 4096 | torch.float16 | 0.02 | 0.93 | 0.75 |

## LogSumExpOp

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.01 | 1.33 | 0.89 |
| 4096 | 4096 | torch.bfloat16 | 0.03 | 1.57 | 1.04 |
| 1024 | 3000 | torch.float16 | 0.02 | 0.56 | 0.38 |
| 1025 | 4096 | torch.float16 | 0.01 | 1.32 | 0.88 |

### torch

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.05 | 0.27 | 0.18 |
| 4096 | 4096 | torch.bfloat16 | 0.14 | 0.37 | 0.25 |
| 1024 | 3000 | torch.float16 | 0.04 | 0.22 | 0.15 |
| 1025 | 4096 | torch.float16 | 0.05 | 0.26 | 0.17 |

## SsdChunkScanFwdOp

### tileops

| batch | num_chunks | chunk_len | n_heads | d_head | d_state | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 64 | 4 | 64 | 32 | torch.float16 | 0.00 | 0.90 | 0.07 |
| 2 | 4 | 64 | 8 | 64 | 64 | torch.float16 | 0.01 | 8.19 | 0.51 |
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
| 1 | 2 | 64 | 4 | 64 | 32 | 1 | torch.float16 | False | 0.01 | 0.32 | 0.02 |
| 2 | 4 | 64 | 8 | 64 | 64 | 2 | torch.float16 | False | 0.01 | 4.48 | 0.23 |
| 1 | 2 | 128 | 4 | 128 | 32 | 1 | torch.bfloat16 | False | 0.01 | 0.76 | 0.04 |
| 2 | 2 | 64 | 4 | 64 | 32 | 2 | torch.bfloat16 | False | 0.01 | 0.68 | 0.05 |
| 2 | 4 | 64 | 8 | 64 | 64 | 2 | torch.float16 | True | 0.01 | 3.49 | 0.18 |

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
| 1 | 32768 | 65536 | 1 | 2048 | torch.float32 | torch.int32 | 16.82 | N/A | 0.53 |
| 1 | 65535 | 131072 | 1 | 1024 | torch.float32 | torch.int32 | 88.34 | N/A | 0.39 |
| 1 | 65535 | 131072 | 1 | 2048 | torch.float32 | torch.int32 | 89.15 | N/A | 0.39 |

### torch

| batch | seq_len | seq_len_kv | kv_group | topk | in_dtype | out_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32768 | 65536 | 1 | 1024 | torch.float32 | torch.int32 | 53.74 | N/A | 0.16 |
| 1 | 32768 | 65536 | 1 | 2048 | torch.float32 | torch.int32 | 56.26 | N/A | 0.16 |
| 1 | 65535 | 131072 | 1 | 1024 | torch.float32 | torch.int32 | 206.19 | N/A | 0.17 |
| 1 | 65535 | 131072 | 1 | 2048 | torch.float32 | torch.int32 | 211.42 | N/A | 0.17 |

## exp

### tileops

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| exp | 262144 | torch.float16 | torch.float16 | 0.00 | 0.12 | 0.49 |
| exp | 1048576 | torch.float16 | torch.float16 | 0.00 | 0.36 | 1.43 |
| exp | 4000000 | torch.float16 | torch.float16 | 0.01 | 0.65 | 2.60 |
| exp | 262144 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.12 | 0.50 |
| exp | 1048576 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.37 | 1.47 |
| exp | 4000000 | torch.bfloat16 | torch.bfloat16 | 0.01 | 0.64 | 2.57 |

### torch

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| exp | 262144 | torch.float16 | torch.float16 | 0.00 | 0.10 | 0.40 |
| exp | 1048576 | torch.float16 | torch.float16 | 0.00 | 0.29 | 1.14 |
| exp | 4000000 | torch.float16 | torch.float16 | 0.01 | 0.50 | 2.02 |
| exp | 262144 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.11 | 0.45 |
| exp | 1048576 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.35 | 1.39 |
| exp | 4000000 | torch.bfloat16 | torch.bfloat16 | 0.01 | 0.64 | 2.58 |

## gelu

### tileops

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| gelu | 262144 | torch.float16 | torch.float16 | 0.00 | 0.12 | 0.47 |
| gelu | 1048576 | torch.float16 | torch.float16 | 0.00 | 0.31 | 1.23 |
| gelu | 4000000 | torch.float16 | torch.float16 | 0.01 | 0.48 | 1.92 |
| gelu | 262144 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.12 | 0.47 |
| gelu | 1048576 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.30 | 1.21 |
| gelu | 4000000 | torch.bfloat16 | torch.bfloat16 | 0.01 | 0.46 | 1.84 |

### torch

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| gelu | 262144 | torch.float16 | torch.float16 | 0.00 | 0.11 | 0.45 |
| gelu | 1048576 | torch.float16 | torch.float16 | 0.00 | 0.30 | 1.20 |
| gelu | 4000000 | torch.float16 | torch.float16 | 0.01 | 0.52 | 2.06 |
| gelu | 262144 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.11 | 0.44 |
| gelu | 1048576 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.29 | 1.18 |
| gelu | 4000000 | torch.bfloat16 | torch.bfloat16 | 0.01 | 0.38 | 1.53 |

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
| logical_not | 262144 | torch.float16 | torch.bool | 0.00 | 0.12 | 0.36 |
| logical_not | 1048576 | torch.float16 | torch.bool | 0.00 | 0.33 | 1.00 |
| logical_not | 4000000 | torch.float16 | torch.bool | 0.01 | 0.64 | 1.91 |

## bitwise_not

### tileops

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| bitwise_not | 262144 | torch.int32 | torch.int32 | 0.00 | 0.10 | 0.84 |
| bitwise_not | 1048576 | torch.int32 | torch.int32 | 0.01 | 0.20 | 1.59 |
| bitwise_not | 4000000 | torch.int32 | torch.int32 | 0.02 | 0.27 | 2.13 |

### torch

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| bitwise_not | 262144 | torch.int32 | torch.int32 | 0.00 | 0.09 | 0.75 |
| bitwise_not | 1048576 | torch.int32 | torch.int32 | 0.00 | 0.21 | 1.68 |
| bitwise_not | 4000000 | torch.int32 | torch.int32 | 0.01 | 0.29 | 2.33 |

## isnan

### tileops

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| isnan | 262144 | torch.float16 | torch.bool | 0.00 | 0.12 | 0.35 |
| isnan | 1048576 | torch.float16 | torch.bool | 0.00 | 0.22 | 0.65 |
| isnan | 4000000 | torch.float16 | torch.bool | 0.01 | 0.30 | 0.91 |

### torch

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| isnan | 262144 | torch.float16 | torch.bool | 0.00 | 0.11 | 0.32 |
| isnan | 1048576 | torch.float16 | torch.bool | 0.00 | 0.30 | 0.89 |
| isnan | 4000000 | torch.float16 | torch.bool | 0.01 | 0.60 | 1.79 |

## unary_strategy

### relu_direct

| n_total | shape_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4194304 | 1024x4096 | torch.float16 | direct | 0.01 | 0.29 | 1.17 |
| 4194304 | 1024x4096 | torch.bfloat16 | direct | 0.01 | 0.29 | 1.17 |
| 4194304 | 1024x4096 | torch.float32 | direct | 0.02 | 0.27 | 2.14 |
| 10485760 | 1024x10240 | torch.float16 | direct | 0.04 | 0.27 | 1.07 |
| 10485760 | 1024x10240 | torch.bfloat16 | direct | 0.04 | 0.27 | 1.07 |
| 10485760 | 1024x10240 | torch.float32 | direct | 0.04 | 0.23 | 1.87 |
| 20971520 | 1024x20480 | torch.float16 | direct | 0.09 | 0.24 | 0.97 |
| 20971520 | 1024x20480 | torch.bfloat16 | direct | 0.09 | 0.23 | 0.91 |
| 20971520 | 1024x20480 | torch.float32 | direct | 0.10 | 0.21 | 1.70 |

### relu_explicit_parallel

| n_total | shape_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4194304 | 1024x4096 | torch.float16 | explicit_parallel | 0.01 | 0.62 | 2.47 |
| 4194304 | 1024x4096 | torch.bfloat16 | explicit_parallel | 0.01 | 0.62 | 2.47 |
| 4194304 | 1024x4096 | torch.float32 | explicit_parallel | 0.01 | 0.39 | 3.11 |
| 10485760 | 1024x10240 | torch.float16 | explicit_parallel | 0.01 | 0.75 | 3.01 |
| 10485760 | 1024x10240 | torch.bfloat16 | explicit_parallel | 0.01 | 0.76 | 3.03 |
| 10485760 | 1024x10240 | torch.float32 | explicit_parallel | 0.02 | 0.46 | 3.66 |
| 20971520 | 1024x20480 | torch.float16 | explicit_parallel | 0.03 | 0.79 | 3.14 |
| 20971520 | 1024x20480 | torch.bfloat16 | explicit_parallel | 0.03 | 0.79 | 3.14 |
| 20971520 | 1024x20480 | torch.float32 | explicit_parallel | 0.06 | 0.38 | 3.01 |

### relu_register_copy

| n_total | shape_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4194304 | 1024x4096 | torch.float16 | register_copy | 0.01 | 0.66 | 2.66 |
| 4194304 | 1024x4096 | torch.bfloat16 | register_copy | 0.01 | 0.66 | 2.66 |
| 4194304 | 1024x4096 | torch.float32 | register_copy | 0.01 | 0.40 | 3.21 |
| 10485760 | 1024x10240 | torch.float16 | register_copy | 0.01 | 0.84 | 3.36 |
| 10485760 | 1024x10240 | torch.bfloat16 | register_copy | 0.01 | 0.84 | 3.35 |
| 10485760 | 1024x10240 | torch.float32 | register_copy | 0.02 | 0.47 | 3.74 |
| 20971520 | 1024x20480 | torch.float16 | register_copy | 0.02 | 0.94 | 3.75 |
| 20971520 | 1024x20480 | torch.bfloat16 | register_copy | 0.02 | 0.93 | 3.74 |
| 20971520 | 1024x20480 | torch.float32 | register_copy | 0.05 | 0.39 | 3.16 |

## L1NormOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | l1 | 0.01 | 0.56 | 1.12 |
| 1024 | 4096 | torch.bfloat16 | l1 | 0.01 | 0.56 | 1.12 |
| 1024 | 4096 | torch.float32 | l1 | 0.01 | 0.36 | 1.45 |
| 4096 | 4096 | torch.float16 | l1 | 0.02 | 0.72 | 1.43 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | l1 | 0.02 | 0.18 | 0.36 |
| 1024 | 4096 | torch.bfloat16 | l1 | 0.02 | 0.18 | 0.36 |
| 1024 | 4096 | torch.float32 | l1 | 0.03 | 0.16 | 0.62 |
| 4096 | 4096 | torch.float16 | l1 | 0.03 | 0.55 | 1.10 |

## L2NormOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | l2 | 0.01 | 0.55 | 1.10 |
| 1024 | 4096 | torch.bfloat16 | l2 | 0.01 | 0.56 | 1.11 |
| 1024 | 4096 | torch.float32 | l2 | 0.01 | 0.35 | 1.41 |
| 4096 | 4096 | torch.float16 | l2 | 0.02 | 0.71 | 1.41 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | l2 | 0.02 | 0.18 | 0.36 |
| 1024 | 4096 | torch.bfloat16 | l2 | 0.02 | 0.18 | 0.36 |
| 1024 | 4096 | torch.float32 | l2 | 0.03 | 0.16 | 0.63 |
| 4096 | 4096 | torch.float16 | l2 | 0.03 | 0.53 | 1.06 |

## InfNormOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | inf | 0.03 | 0.15 | 0.30 |
| 1024 | 4096 | torch.bfloat16 | inf | 0.03 | 0.15 | 0.29 |
| 1024 | 4096 | torch.float32 | inf | 0.03 | 0.12 | 0.50 |
| 4096 | 4096 | torch.float16 | inf | 0.06 | 0.29 | 0.59 |

### torch

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | inf | 0.02 | 0.20 | 0.40 |
| 1024 | 4096 | torch.bfloat16 | inf | 0.02 | 0.18 | 0.36 |
| 1024 | 4096 | torch.float32 | inf | 0.03 | 0.15 | 0.61 |
| 4096 | 4096 | torch.float16 | inf | 0.03 | 0.52 | 1.05 |

........................................................................ [  8%]
........................................................................ [ 17%]
........................................................................ [ 26%]
........................................................................ [ 35%]
........................................................................ [ 44%]
........................................................................ [ 53%]
........................................................................ [ 61%]
........................................................................ [ 70%]
........................................................................ [ 79%]
........................................................................ [ 88%]
........................................................................ [ 97%]
......................                                                   [100%]Benchmark report saved to profile_run.log

=============================== warnings summary ===============================
benchmarks/ops/bench_activation.py: 93 warnings
benchmarks/ops/bench_ada_layer_norm.py: 8 warnings
benchmarks/ops/bench_argreduce.py: 6 warnings
benchmarks/ops/bench_batch_norm.py: 12 warnings
benchmarks/ops/bench_binary_arith.py: 65 warnings
benchmarks/ops/bench_binary_elementwise.py: 107 warnings
benchmarks/ops/bench_binary_strategy.py: 18 warnings
benchmarks/ops/bench_cumulative.py: 6 warnings
benchmarks/ops/bench_deepseek_dsa_decode.py: 2 warnings
benchmarks/ops/bench_deepseek_mla_decode.py: 2 warnings
benchmarks/ops/bench_deepseek_nsa_cmp_fwd.py: 2 warnings
benchmarks/ops/bench_deepseek_nsa_fwd.py: 3 warnings
benchmarks/ops/bench_deepseek_nsa_topk.py: 3 warnings
benchmarks/ops/bench_dropout.py: 9 warnings
benchmarks/ops/bench_elementwise_fp8.py: 24 warnings
benchmarks/ops/bench_engram_bwd.py: 4 warnings
benchmarks/ops/bench_engram_decode.py: 3 warnings
benchmarks/ops/bench_engram_fwd.py: 4 warnings
benchmarks/ops/bench_fft.py: 5 warnings
benchmarks/ops/bench_fft_lut.py: 14 warnings
benchmarks/ops/bench_fp8_lighting_indexer.py: 2 warnings
benchmarks/ops/bench_fp8_quant.py: 4 warnings
benchmarks/ops/bench_fused_add_layer_norm.py: 4 warnings
benchmarks/ops/bench_fused_add_rmsnorm.py: 4 warnings
benchmarks/ops/bench_gated_deltanet_chunkwise.py: 32 warnings
benchmarks/ops/bench_gated_deltanet_recurrence.py: 13 warnings
benchmarks/ops/bench_gemm.py: 5 warnings
benchmarks/ops/bench_gla_chunkwise.py: 24 warnings
benchmarks/ops/bench_gla_recurrence.py: 16 warnings
benchmarks/ops/bench_gqa.py: 6 warnings
benchmarks/ops/bench_gqa_decode.py: 3 warnings
benchmarks/ops/bench_gqa_decode_paged.py: 4 warnings
benchmarks/ops/bench_gqa_sliding_window_fwd.py: 5 warnings
benchmarks/ops/bench_gqa_sliding_window_varlen_fwd.py: 5 warnings
benchmarks/ops/bench_group_norm.py: 4 warnings
benchmarks/ops/bench_grouped_gemm.py: 8 warnings
benchmarks/ops/bench_independent_elementwise.py: 132 warnings
benchmarks/ops/bench_instance_norm.py: 4 warnings
benchmarks/ops/bench_layer_norm.py: 4 warnings
benchmarks/ops/bench_logical_reduce.py: 15 warnings
benchmarks/ops/bench_mean_pooling.py: 4 warnings
benchmarks/ops/bench_mha.py: 6 warnings
benchmarks/ops/bench_mha_decode.py: 3 warnings
benchmarks/ops/bench_mha_decode_paged.py: 4 warnings
benchmarks/ops/bench_mhc_post.py: 3 warnings
benchmarks/ops/bench_mhc_pre.py: 3 warnings
benchmarks/ops/bench_moe_permute.py: 9 warnings
benchmarks/ops/bench_moe_permute_align.py: 10 warnings
benchmarks/ops/bench_reduce.py: 10 warnings
benchmarks/ops/bench_rms_norm.py: 4 warnings
benchmarks/ops/bench_rope.py: 9 warnings
benchmarks/ops/bench_softmax.py: 12 warnings
benchmarks/ops/bench_topk_selector.py: 4 warnings
benchmarks/ops/bench_unary_elementwise.py: 21 warnings
benchmarks/ops/bench_unary_strategy.py: 27 warnings
benchmarks/ops/bench_vector_norm.py: 12 warnings
  /home/ci-runner/workdir/_tool/tileops_benchmarks_venv_db4a9af2f48857c5/lib/python3.11/site-packages/torch/profiler/profiler.py:509: UserWarning: Profiler won't be using warmup, this can skew profiler results
    warn("Profiler won't be using warmup, this can skew profiler results")

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
814 passed, 825 warnings in 3173.30s (0:52:53)

===== profile_run.log summary =====
# TileOPs Benchmark Report
Generated: 2026-03-19 20:14:33

## Environment

- **Torch version**: 2.9.1+cu128
- **CUDA version (torch)**: 12.8
- **GPU model**: NVIDIA H200
- **Driver version**: 575.57.08

## r2_small_tensor_unary

### tileops

| n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 4096 | torch.float16 | 0.00 | 0.00 | 0.01 |

### baseline

| n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 4096 | torch.float16 | 0.00 | 0.00 | 0.01 |

## r3_jit_cost

### relu_jit

| n_total | cold_ms | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 1000 | 21.13 | 0.00 | 0.00 | 0.00 |
| 2000 | 20.84 | 0.00 | 0.00 | 0.00 |
| 4000 | 26.71 | 0.00 | 0.00 | 0.01 |
| 8000 | 23.2 | 0.00 | 0.00 | 0.02 |
| 16000 | 21.19 | 0.00 | 0.01 | 0.03 |
| 32000 | 21.93 | 0.00 | 0.02 | 0.06 |
| 64000 | 22.51 | 0.00 | 0.03 | 0.12 |
| 128000 | 22.38 | 0.00 | 0.06 | 0.23 |
| 256000 | 20.55 | 0.00 | 0.11 | 0.45 |
| 512000 | 20.76 | 0.00 | 0.19 | 0.78 |

## r4_strategy_unary

### relu_direct

| n_total | size_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | torch.float16 | direct | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.bfloat16 | direct | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.float32 | direct | 0.00 | 0.00 | 0.02 |
| 4194304 | 4M | torch.float16 | direct | 0.01 | 0.29 | 1.15 |
| 4194304 | 4M | torch.bfloat16 | direct | 0.01 | 0.29 | 1.15 |
| 4194304 | 4M | torch.float32 | direct | 0.02 | 0.26 | 2.10 |
| 16777216 | 16M | torch.float16 | direct | 0.05 | 0.32 | 1.29 |
| 16777216 | 16M | torch.bfloat16 | direct | 0.05 | 0.32 | 1.29 |
| 16777216 | 16M | torch.float32 | direct | 0.06 | 0.29 | 2.34 |

### relu_explicit_parallel

| n_total | size_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | torch.float16 | explicit_parallel | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.bfloat16 | explicit_parallel | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.float32 | explicit_parallel | 0.00 | 0.00 | 0.02 |
| 4194304 | 4M | torch.float16 | explicit_parallel | 0.01 | 0.60 | 2.40 |
| 4194304 | 4M | torch.bfloat16 | explicit_parallel | 0.01 | 0.60 | 2.40 |
| 4194304 | 4M | torch.float32 | explicit_parallel | 0.01 | 0.38 | 3.03 |
| 16777216 | 16M | torch.float16 | explicit_parallel | 0.02 | 0.79 | 3.18 |
| 16777216 | 16M | torch.bfloat16 | explicit_parallel | 0.02 | 0.80 | 3.18 |
| 16777216 | 16M | torch.float32 | explicit_parallel | 0.03 | 0.48 | 3.85 |

### relu_register_copy

| n_total | size_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | torch.float16 | register_copy | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.bfloat16 | register_copy | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.float32 | register_copy | 0.00 | 0.00 | 0.02 |
| 4194304 | 4M | torch.float16 | register_copy | 0.01 | 0.65 | 2.61 |
| 4194304 | 4M | torch.bfloat16 | register_copy | 0.01 | 0.65 | 2.61 |
| 4194304 | 4M | torch.float32 | register_copy | 0.01 | 0.39 | 3.12 |
| 16777216 | 16M | torch.float16 | register_copy | 0.02 | 0.91 | 3.63 |
| 16777216 | 16M | torch.bfloat16 | register_copy | 0.02 | 0.91 | 3.63 |
| 16777216 | 16M | torch.float32 | register_copy | 0.03 | 0.49 | 3.91 |

## r4_strategy_gelu

### gelu_direct

| n_total | size_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | torch.float16 | direct | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.bfloat16 | direct | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.float32 | direct | 0.00 | 0.00 | 0.02 |
| 4194304 | 4M | torch.float16 | direct | 0.02 | 0.27 | 1.06 |
| 4194304 | 4M | torch.bfloat16 | direct | 0.02 | 0.26 | 1.06 |
| 4194304 | 4M | torch.float32 | direct | 0.02 | 0.25 | 1.96 |
| 16777216 | 16M | torch.float16 | direct | 0.06 | 0.29 | 1.17 |
| 16777216 | 16M | torch.bfloat16 | direct | 0.06 | 0.29 | 1.17 |
| 16777216 | 16M | torch.float32 | direct | 0.06 | 0.27 | 2.18 |

### gelu_explicit_parallel

| n_total | size_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | torch.float16 | explicit_parallel | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.bfloat16 | explicit_parallel | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.float32 | explicit_parallel | 0.00 | 0.00 | 0.02 |
| 4194304 | 4M | torch.float16 | explicit_parallel | 0.01 | 0.44 | 1.76 |
| 4194304 | 4M | torch.bfloat16 | explicit_parallel | 0.01 | 0.44 | 1.74 |
| 4194304 | 4M | torch.float32 | explicit_parallel | 0.01 | 0.38 | 3.03 |
| 16777216 | 16M | torch.float16 | explicit_parallel | 0.03 | 0.53 | 2.12 |
| 16777216 | 16M | torch.bfloat16 | explicit_parallel | 0.03 | 0.52 | 2.10 |
| 16777216 | 16M | torch.float32 | explicit_parallel | 0.04 | 0.46 | 3.68 |

### gelu_register_copy

| n_total | size_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | torch.float16 | register_copy | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.bfloat16 | register_copy | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.float32 | register_copy | 0.00 | 0.00 | 0.02 |
| 4194304 | 4M | torch.float16 | register_copy | 0.01 | 0.47 | 1.90 |
| 4194304 | 4M | torch.bfloat16 | register_copy | 0.01 | 0.46 | 1.86 |
| 4194304 | 4M | torch.float32 | register_copy | 0.01 | 0.39 | 3.10 |
| 16777216 | 16M | torch.float16 | register_copy | 0.03 | 0.60 | 2.41 |
| 16777216 | 16M | torch.bfloat16 | register_copy | 0.03 | 0.58 | 2.33 |
| 16777216 | 16M | torch.float32 | register_copy | 0.04 | 0.47 | 3.78 |

## r5_boundary

### relu_aligned

| n_total | align_label | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 2048000 | aligned | 0.00 | 0.45 | 1.78 |

### relu_unaligned_plus_1

| n_total | align_label | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 2048001 | unaligned_plus_1 | 0.01 | 0.29 | 1.15 |

### relu_unaligned_plus_127

| n_total | align_label | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 2048127 | unaligned_plus_127 | 0.01 | 0.29 | 1.15 |

## r6_threads

### relu_t128

| n_total | size_label | op_name | threads | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | relu | 128 | 0.00 | 0.00 | 0.01 |
| 4194304 | 4M | relu | 128 | 0.02 | 0.21 | 0.85 |
| 16777216 | 16M | relu | 128 | 0.07 | 0.26 | 1.02 |

### relu_t256

| n_total | size_label | op_name | threads | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | relu | 256 | 0.00 | 0.00 | 0.01 |
| 4194304 | 4M | relu | 256 | 0.02 | 0.21 | 0.86 |
| 16777216 | 16M | relu | 256 | 0.07 | 0.25 | 0.98 |

### erf_t128

| n_total | size_label | op_name | threads | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | erf | 128 | 0.00 | 0.00 | 0.01 |
| 4194304 | 4M | erf | 128 | 0.01 | 0.48 | 1.93 |
| 16777216 | 16M | erf | 128 | 0.03 | 0.60 | 2.40 |

### erf_t256

| n_total | size_label | op_name | threads | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | erf | 256 | 0.00 | 0.00 | 0.01 |
| 4194304 | 4M | erf | 256 | 0.01 | 0.48 | 1.91 |
| 16777216 | 16M | erf | 256 | 0.03 | 0.60 | 2.40 |

### mish_t128

| n_total | size_label | op_name | threads | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | mish | 128 | 0.00 | 0.00 | 0.01 |
| 4194304 | 4M | mish | 128 | 0.01 | 0.36 | 1.44 |
| 16777216 | 16M | mish | 128 | 0.04 | 0.42 | 1.69 |

### mish_t256

| n_total | size_label | op_name | threads | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | mish | 256 | 0.00 | 0.00 | 0.01 |
| 4194304 | 4M | mish | 256 | 0.01 | 0.35 | 1.42 |
| 16777216 | 16M | mish | 256 | 0.04 | 0.42 | 1.68 |

## r7_dtype_npt

### relu_fp32_npt4

| dtype_label | num_per_thread | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| fp32 | 4 | 0.00 | 0.22 | 1.76 |

### relu_fp32_npt8

| dtype_label | num_per_thread | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| fp32 | 8 | 0.00 | 0.22 | 1.74 |

### relu_fp16_npt4

| dtype_label | num_per_thread | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| fp16 | 4 | 0.00 | 0.28 | 1.12 |

### relu_fp16_npt8

| dtype_label | num_per_thread | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| fp16 | 8 | 0.00 | 0.22 | 0.89 |

## relu

### tileops

| n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 4194304 | torch.float16 | 0.01 | 0.65 | 2.61 |
| 4194304 | torch.bfloat16 | 0.01 | 0.65 | 2.61 |
| 4194304 | torch.float32 | 0.01 | 0.39 | 3.12 |

### baseline

| n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 4194304 | torch.float16 | 0.01 | 0.63 | 2.51 |
| 4194304 | torch.bfloat16 | 0.01 | 0.64 | 2.56 |
| 4194304 | torch.float32 | 0.01 | 0.39 | 3.12 |

## ada_layer_norm

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.02 | 1.34 | 2.14 |
| 4096 | 4096 | torch.bfloat16 | 0.05 | 1.60 | 2.57 |
| 1024 | 3000 | torch.float16 | 0.04 | 0.38 | 0.60 |
| 1025 | 4096 | torch.float16 | 0.02 | 1.34 | 2.14 |

### baseline

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.02 | 0.85 | 1.36 |
| 4096 | 4096 | torch.bfloat16 | 0.08 | 1.05 | 1.68 |
| 1024 | 3000 | torch.float16 | 0.02 | 0.75 | 1.21 |
| 1025 | 4096 | torch.float16 | 0.03 | 0.83 | 1.33 |

## ada_layer_norm_zero

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.02 | 1.29 | 2.15 |
| 4096 | 4096 | torch.bfloat16 | 0.06 | 1.58 | 2.63 |
| 1024 | 3000 | torch.float16 | 0.05 | 0.35 | 0.58 |
| 1025 | 4096 | torch.float16 | 0.02 | 1.29 | 2.15 |

### baseline

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.03 | 0.80 | 1.33 |
| 4096 | 4096 | torch.bfloat16 | 0.11 | 0.96 | 1.59 |
| 1024 | 3000 | torch.float16 | 0.03 | 0.72 | 1.20 |
| 1025 | 4096 | torch.float16 | 0.03 | 0.80 | 1.33 |

## argreduce

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | argmax | 2.98 | 0.00 | 0.00 |
| 1024 | 4096 | torch.bfloat16 | argmax | 2.98 | 0.00 | 0.00 |
| 4096 | 4096 | torch.float16 | argmax | 9.50 | 0.00 | 0.00 |
| 1024 | 4096 | torch.float16 | argmin | 3.34 | 0.00 | 0.00 |
| 1024 | 4096 | torch.bfloat16 | argmin | 3.36 | 0.00 | 0.00 |
| 4096 | 4096 | torch.float16 | argmin | 10.61 | 0.00 | 0.00 |

### baseline

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | argmax | 0.02 | 0.19 | 0.37 |
| 1024 | 4096 | torch.bfloat16 | argmax | 0.02 | 0.19 | 0.37 |
| 4096 | 4096 | torch.float16 | argmax | 0.03 | 0.59 | 1.18 |
| 1024 | 4096 | torch.float16 | argmin | 0.02 | 0.19 | 0.37 |
| 1024 | 4096 | torch.bfloat16 | argmin | 0.02 | 0.19 | 0.37 |
| 4096 | 4096 | torch.float16 | argmin | 0.03 | 0.59 | 1.18 |

## batch_norm_fwd

### tileops

| N | C | spatial | dtype | training | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 64 | () | torch.float16 | True | 0.01 | 0.00 | 0.00 |
| 8 | 64 | (32, 32) | torch.float16 | True | 0.01 | 0.43 | 0.17 |
| 4 | 128 | (32, 32) | torch.float16 | True | 0.01 | 0.43 | 0.17 |
| 4 | 256 | (28, 28) | torch.float16 | True | 0.02 | 0.51 | 0.20 |
| 4 | 128 | (1024, 1024) | torch.float16 | True | 3.86 | 1.39 | 0.56 |
| 4 | 256 | (1024, 1024) | torch.float16 | True | 7.06 | 1.52 | 0.61 |

### torch_cudnn

| N | C | spatial | dtype | training | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 64 | () | torch.float16 | True | 0.02 | 0.00 | 0.00 |
| 8 | 64 | (32, 32) | torch.float16 | True | 0.01 | 0.37 | 0.15 |
| 4 | 128 | (32, 32) | torch.float16 | True | 0.01 | 0.39 | 0.16 |
| 4 | 256 | (28, 28) | torch.float16 | True | 0.01 | 0.57 | 0.23 |
| 4 | 128 | (1024, 1024) | torch.float16 | True | 4.04 | 1.33 | 0.53 |
| 4 | 256 | (1024, 1024) | torch.float16 | True | 7.01 | 1.53 | 0.61 |

## batch_norm_bwd

### tileops

| N | C | spatial | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 64 | () | torch.float16 | 0.01 | 0.00 | 0.00 |
| 8 | 64 | (32, 32) | torch.float16 | 0.02 | 0.26 | 0.19 |
| 4 | 128 | (32, 32) | torch.float16 | 0.02 | 0.26 | 0.20 |
| 4 | 256 | (28, 28) | torch.float16 | 0.02 | 0.32 | 0.24 |
| 4 | 128 | (1024, 1024) | torch.float16 | 6.03 | 0.71 | 0.53 |
| 4 | 256 | (1024, 1024) | torch.float16 | 10.65 | 0.81 | 0.61 |

### torch_autograd

| N | C | spatial | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 64 | () | torch.float16 | 0.02 | 0.00 | 0.00 |
| 8 | 64 | (32, 32) | torch.float16 | 0.03 | 0.16 | 0.12 |
| 4 | 128 | (32, 32) | torch.float16 | 0.02 | 0.19 | 0.14 |
| 4 | 256 | (28, 28) | torch.float16 | 0.02 | 0.27 | 0.20 |
| 4 | 128 | (1024, 1024) | torch.float16 | 11.97 | 0.36 | 0.27 |
| 4 | 256 | (1024, 1024) | torch.float16 | 17.88 | 0.48 | 0.36 |

## r1_vectorization

### add_same_shape_1d

| pattern_name | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| same_shape_1d | 1000000 | 0.00 | 0.24 | 1.44 |

### baseline_same_shape_1d

| pattern_name | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| same_shape_1d | 1000000 | 0.00 | 0.25 | 1.49 |

### add_bias_add_2d

| pattern_name | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| bias_add_2d | 1000000 | 0.00 | 0.26 | 1.06 |

### baseline_bias_add_2d

| pattern_name | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| bias_add_2d | 1000000 | 0.01 | 0.17 | 0.67 |

### add_interleaved_3d

| pattern_name | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| interleaved_3d | 1048576 | 0.00 | 0.41 | 0.84 |

### baseline_interleaved_3d

| pattern_name | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| interleaved_3d | 1048576 | 0.01 | 0.19 | 0.38 |

## r2_small_tensor_binary

### add_same_shape

| pattern_name | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| same_shape | 4096 | 0.00 | 0.00 | 0.01 |

### baseline_same_shape

| pattern_name | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| same_shape | 4096 | 0.00 | 0.00 | 0.01 |

### add_broadcast_3d

| pattern_name | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| broadcast_3d | 4096 | 0.00 | 0.00 | 0.01 |

### baseline_broadcast_3d

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
| 1M | same_shape | direct | 1048576 | 0.01 | 0.18 | 1.10 |
| 1M | same_shape | direct | 1048576 | 0.01 | 0.18 | 1.10 |
| 1M | same_shape | direct | 1048576 | 0.01 | 0.16 | 1.96 |
| 16M | same_shape | direct | 16777216 | 0.06 | 0.29 | 1.76 |
| 16M | same_shape | direct | 16777216 | 0.06 | 0.29 | 1.76 |
| 16M | same_shape | direct | 16777216 | 0.07 | 0.25 | 3.03 |

### add_direct_bias_add_2d

| size_label | pattern_name | strategy | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4K | bias_add_2d | direct | 4096 | 0.00 | 0.00 | 0.01 |
| 4K | bias_add_2d | direct | 4096 | 0.00 | 0.00 | 0.01 |
| 4K | bias_add_2d | direct | 4096 | 0.00 | 0.00 | 0.02 |
| 1M | bias_add_2d | direct | 1048576 | 0.01 | 0.20 | 0.80 |
| 1M | bias_add_2d | direct | 1048576 | 0.01 | 0.20 | 0.80 |
| 1M | bias_add_2d | direct | 1048576 | 0.01 | 0.19 | 1.50 |
| 16M | bias_add_2d | direct | 16777216 | 0.05 | 0.32 | 1.27 |
| 16M | bias_add_2d | direct | 16777216 | 0.05 | 0.32 | 1.27 |
| 16M | bias_add_2d | direct | 16777216 | 0.06 | 0.29 | 2.31 |

### add_direct_interleaved_3d

| size_label | pattern_name | strategy | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4K | interleaved_3d | direct | 4096 | 0.00 | 0.00 | 0.01 |
| 4K | interleaved_3d | direct | 4096 | 0.00 | 0.00 | 0.01 |
| 4K | interleaved_3d | direct | 4096 | 0.00 | 0.00 | 0.01 |
| 1M | interleaved_3d | direct | 1048576 | 0.00 | 0.23 | 0.47 |
| 1M | interleaved_3d | direct | 1048576 | 0.00 | 0.23 | 0.47 |
| 1M | interleaved_3d | direct | 1048576 | 0.00 | 0.23 | 0.90 |
| 16M | interleaved_3d | direct | 16777216 | 0.04 | 0.40 | 0.80 |
| 16M | interleaved_3d | direct | 16777216 | 0.04 | 0.40 | 0.80 |
| 16M | interleaved_3d | direct | 16777216 | 0.04 | 0.39 | 1.57 |

### add_explicit_parallel_same_shape

| size_label | pattern_name | strategy | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4K | same_shape | explicit_parallel | 4096 | 0.00 | 0.00 | 0.01 |
| 4K | same_shape | explicit_parallel | 4096 | 0.00 | 0.00 | 0.01 |
| 4K | same_shape | explicit_parallel | 4096 | 0.00 | 0.00 | 0.03 |
| 1M | same_shape | explicit_parallel | 1048576 | 0.00 | 0.26 | 1.56 |
| 1M | same_shape | explicit_parallel | 1048576 | 0.00 | 0.26 | 1.56 |
| 1M | same_shape | explicit_parallel | 1048576 | 0.01 | 0.19 | 2.24 |
| 16M | same_shape | explicit_parallel | 16777216 | 0.03 | 0.59 | 3.57 |
| 16M | same_shape | explicit_parallel | 16777216 | 0.03 | 0.60 | 3.57 |
| 16M | same_shape | explicit_parallel | 16777216 | 0.05 | 0.34 | 4.03 |

### add_explicit_parallel_bias_add_2d

| size_label | pattern_name | strategy | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4K | bias_add_2d | explicit_parallel | 4096 | 0.00 | 0.00 | 0.01 |
| 4K | bias_add_2d | explicit_parallel | 4096 | 0.00 | 0.00 | 0.01 |
| 4K | bias_add_2d | explicit_parallel | 4096 | 0.00 | 0.00 | 0.02 |
| 1M | bias_add_2d | explicit_parallel | 1048576 | 0.00 | 0.31 | 1.26 |
| 1M | bias_add_2d | explicit_parallel | 1048576 | 0.00 | 0.31 | 1.26 |
| 1M | bias_add_2d | explicit_parallel | 1048576 | 0.00 | 0.24 | 1.89 |
| 16M | bias_add_2d | explicit_parallel | 16777216 | 0.02 | 0.81 | 3.24 |
| 16M | bias_add_2d | explicit_parallel | 16777216 | 0.02 | 0.81 | 3.24 |
| 16M | bias_add_2d | explicit_parallel | 16777216 | 0.03 | 0.49 | 3.92 |

### add_explicit_parallel_interleaved_3d

| size_label | pattern_name | strategy | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4K | interleaved_3d | explicit_parallel | 4096 | 0.00 | 0.00 | 0.01 |
| 4K | interleaved_3d | explicit_parallel | 4096 | 0.00 | 0.00 | 0.01 |
| 4K | interleaved_3d | explicit_parallel | 4096 | 0.00 | 0.00 | 0.01 |
| 1M | interleaved_3d | explicit_parallel | 1048576 | 0.00 | 0.42 | 0.84 |
| 1M | interleaved_3d | explicit_parallel | 1048576 | 0.00 | 0.42 | 0.84 |
| 1M | interleaved_3d | explicit_parallel | 1048576 | 0.00 | 0.38 | 1.54 |
| 16M | interleaved_3d | explicit_parallel | 16777216 | 0.01 | 1.18 | 2.35 |
| 16M | interleaved_3d | explicit_parallel | 16777216 | 0.01 | 1.18 | 2.36 |
| 16M | interleaved_3d | explicit_parallel | 16777216 | 0.02 | 0.96 | 3.84 |

## r4_where

### tileops_where

| n_total | size_label | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 4096 | 4K | 0.00 | 0.00 | 0.01 |
| 1048576 | 1M | 0.00 | 0.24 | 1.68 |
| 16777216 | 16M | 0.03 | 0.53 | 3.72 |

### baseline_where

| n_total | size_label | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 4096 | 4K | 0.00 | 0.00 | 0.01 |
| 1048576 | 1M | 0.00 | 0.24 | 1.65 |
| 16777216 | 16M | 0.03 | 0.54 | 3.75 |

## add

### tileops

| n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 4194304 | torch.float16 | 0.01 | 0.47 | 2.83 |
| 4194304 | torch.bfloat16 | 0.01 | 0.47 | 2.83 |
| 4194304 | torch.float32 | 0.02 | 0.27 | 3.29 |

### baseline

| n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 4194304 | torch.float16 | 0.01 | 0.47 | 2.79 |
| 4194304 | torch.bfloat16 | 0.01 | 0.47 | 2.79 |
| 4194304 | torch.float32 | 0.02 | 0.28 | 3.31 |

## sub

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| sub | torch.float16 | torch.float16 | 0.01 | 0.47 | 2.84 |
| sub | torch.float16 | torch.float16 | 0.02 | 0.57 | 3.44 |
| sub | torch.float16 | torch.float16 | 0.03 | 0.64 | 3.84 |

### baseline

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| sub | torch.float16 | torch.float16 | 0.01 | 0.47 | 2.79 |
| sub | torch.float16 | torch.float16 | 0.02 | 0.57 | 3.42 |
| sub | torch.float16 | torch.float16 | 0.03 | 0.64 | 3.86 |

## mul

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| mul | torch.float16 | torch.float16 | 0.01 | 0.47 | 2.83 |
| mul | torch.float16 | torch.float16 | 0.02 | 0.57 | 3.44 |
| mul | torch.float16 | torch.float16 | 0.03 | 0.64 | 3.84 |

### baseline

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| mul | torch.float16 | torch.float16 | 0.01 | 0.47 | 2.79 |
| mul | torch.float16 | torch.float16 | 0.02 | 0.57 | 3.42 |
| mul | torch.float16 | torch.float16 | 0.03 | 0.64 | 3.86 |

## div

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| div | torch.float16 | torch.float16 | 0.01 | 0.45 | 2.72 |
| div | torch.float16 | torch.float16 | 0.02 | 0.57 | 3.45 |
| div | torch.float16 | torch.float16 | 0.03 | 0.63 | 3.78 |

### baseline

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| div | torch.float16 | torch.float16 | 0.01 | 0.44 | 2.67 |
| div | torch.float16 | torch.float16 | 0.02 | 0.57 | 3.41 |
| div | torch.float16 | torch.float16 | 0.03 | 0.63 | 3.75 |

## remainder

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| remainder | torch.float16 | torch.float16 | 0.01 | 0.45 | 2.68 |
| remainder | torch.float16 | torch.float16 | 0.02 | 0.57 | 3.42 |

### baseline

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| remainder | torch.float16 | torch.float16 | 0.01 | 0.40 | 2.43 |
| remainder | torch.float16 | torch.float16 | 0.02 | 0.51 | 3.06 |

## pow

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| pow | torch.float16 | torch.float16 | 0.02 | 0.24 | 1.45 |
| pow | torch.float16 | torch.float16 | 0.04 | 0.28 | 1.67 |

### baseline

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| pow | torch.float16 | torch.float16 | 0.02 | 0.24 | 1.45 |
| pow | torch.float16 | torch.float16 | 0.04 | 0.28 | 1.66 |

## floor_divide

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| floor_divide | torch.float16 | torch.float16 | 0.01 | 0.45 | 2.68 |
| floor_divide | torch.float16 | torch.float16 | 0.02 | 0.57 | 3.43 |

### baseline

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| floor_divide | torch.float16 | torch.float16 | 0.02 | 0.18 | 1.11 |
| floor_divide | torch.float16 | torch.float16 | 0.05 | 0.21 | 1.25 |

## lerp

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| lerp | torch.float16 | torch.float16 | 0.01 | 0.47 | 2.83 |
| lerp | torch.float16 | torch.float16 | 0.02 | 0.57 | 3.44 |

### baseline

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| lerp | torch.float16 | torch.float16 | 0.01 | 0.47 | 2.83 |
| lerp | torch.float16 | torch.float16 | 0.02 | 0.57 | 3.44 |

## maximum

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| maximum | torch.float16 | torch.float16 | 0.01 | 0.46 | 2.77 |
| maximum | torch.float16 | torch.float16 | 0.02 | 0.57 | 3.39 |
| maximum | torch.float16 | torch.float16 | 0.03 | 0.64 | 3.82 |

### baseline

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| maximum | torch.float16 | torch.float16 | 0.01 | 0.46 | 2.76 |
| maximum | torch.float16 | torch.float16 | 0.02 | 0.57 | 3.39 |
| maximum | torch.float16 | torch.float16 | 0.03 | 0.64 | 3.82 |

## minimum

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| minimum | torch.float16 | torch.float16 | 0.01 | 0.46 | 2.77 |
| minimum | torch.float16 | torch.float16 | 0.02 | 0.57 | 3.39 |
| minimum | torch.float16 | torch.float16 | 0.03 | 0.64 | 3.82 |

### baseline

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| minimum | torch.float16 | torch.float16 | 0.01 | 0.46 | 2.76 |
| minimum | torch.float16 | torch.float16 | 0.02 | 0.57 | 3.39 |
| minimum | torch.float16 | torch.float16 | 0.03 | 0.64 | 3.82 |

## cmp_eq

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| eq | torch.float16 | 0.02 | 0.22 | 1.11 |
| eq | torch.float16 | 0.04 | 0.26 | 1.32 |

### baseline

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| eq | torch.float16 | 0.01 | 0.52 | 2.58 |
| eq | torch.float16 | 0.02 | 0.64 | 3.22 |

## cmp_ne

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| ne | torch.float16 | 0.02 | 0.22 | 1.11 |

### baseline

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| ne | torch.float16 | 0.01 | 0.52 | 2.58 |

## cmp_gt

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| gt | torch.float16 | 0.02 | 0.22 | 1.11 |

### baseline

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| gt | torch.float16 | 0.01 | 0.51 | 2.56 |

## cmp_lt

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| lt | torch.float16 | 0.02 | 0.22 | 1.11 |

### baseline

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| lt | torch.float16 | 0.01 | 0.51 | 2.55 |

## cmp_ge

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| ge | torch.float16 | 0.02 | 0.22 | 1.11 |

### baseline

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| ge | torch.float16 | 0.01 | 0.51 | 2.56 |

## cmp_le

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| le | torch.float16 | 0.02 | 0.22 | 1.11 |

### baseline

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| le | torch.float16 | 0.01 | 0.51 | 2.55 |

## logical_and

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| logical_and | torch.float16 | 0.02 | 0.22 | 1.11 |
| logical_and | torch.float16 | 0.04 | 0.26 | 1.32 |

### baseline

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| logical_and | torch.float16 | 0.01 | 0.71 | 3.54 |
| logical_and | torch.float16 | 0.01 | 0.93 | 4.67 |

## logical_or

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| logical_or | torch.float16 | 0.02 | 0.22 | 1.11 |
| logical_or | torch.float16 | 0.04 | 0.26 | 1.32 |

### baseline

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| logical_or | torch.float16 | 0.01 | 0.71 | 3.56 |
| logical_or | torch.float16 | 0.01 | 0.94 | 4.70 |

## bitwise_and

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| bitwise_and | torch.int32 | 0.02 | 0.27 | 3.23 |
| bitwise_and | torch.int32 | 0.03 | 0.31 | 3.77 |

### baseline

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| bitwise_and | torch.int32 | 0.02 | 0.28 | 3.31 |
| bitwise_and | torch.int32 | 0.03 | 0.32 | 3.84 |

## bitwise_or

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| bitwise_or | torch.int32 | 0.02 | 0.27 | 3.22 |

### baseline

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| bitwise_or | torch.int32 | 0.02 | 0.28 | 3.31 |

## bitwise_xor

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| bitwise_xor | torch.int32 | 0.02 | 0.27 | 3.22 |

### baseline

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| bitwise_xor | torch.int32 | 0.02 | 0.28 | 3.31 |

## gelu_and_mul

### tileops

| op_name | M | N | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| gelu_and_mul | 1024 | 4096 | torch.float16 | 0.01 | 0.77 | 2.31 |
| gelu_and_mul | 1024 | 10240 | torch.float16 | 0.02 | 0.89 | 2.66 |
| gelu_and_mul | 1024 | 20480 | torch.float16 | 0.04 | 0.96 | 2.89 |

### baseline

| op_name | M | N | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| gelu_and_mul | 1024 | 4096 | torch.float16 | 0.03 | 0.27 | 0.80 |
| gelu_and_mul | 1024 | 10240 | torch.float16 | 0.07 | 0.30 | 0.91 |
| gelu_and_mul | 1024 | 20480 | torch.float16 | 0.13 | 0.32 | 0.96 |

## gelu_tanh_and_mul

### tileops

| op_name | M | N | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| gelu_tanh_and_mul | 1024 | 4096 | torch.float16 | 0.01 | 0.90 | 2.69 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.float16 | 0.02 | 1.06 | 3.19 |
| gelu_tanh_and_mul | 1024 | 20480 | torch.float16 | 0.04 | 1.17 | 3.50 |

### baseline

| op_name | M | N | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| gelu_tanh_and_mul | 1024 | 4096 | torch.float16 | 0.03 | 0.28 | 0.83 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.float16 | 0.07 | 0.32 | 0.95 |
| gelu_tanh_and_mul | 1024 | 20480 | torch.float16 | 0.12 | 0.34 | 1.01 |

## silu_and_mul_strategy

### tileops_direct

| op_name | M | N | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| silu_and_mul | 1024 | 4096 | torch.float16 | direct | 0.02 | 0.49 | 1.48 |
| silu_and_mul | 1024 | 4096 | torch.bfloat16 | direct | 0.02 | 0.49 | 1.48 |
| silu_and_mul | 1024 | 4096 | torch.float32 | direct | 0.02 | 0.43 | 2.56 |
| silu_and_mul | 1024 | 10240 | torch.float16 | direct | 0.04 | 0.53 | 1.60 |
| silu_and_mul | 1024 | 10240 | torch.bfloat16 | direct | 0.04 | 0.54 | 1.61 |
| silu_and_mul | 1024 | 10240 | torch.float32 | direct | 0.04 | 0.47 | 2.84 |
| silu_and_mul | 4096 | 4096 | torch.float16 | direct | 0.06 | 0.55 | 1.66 |
| silu_and_mul | 4096 | 4096 | torch.bfloat16 | direct | 0.06 | 0.56 | 1.67 |
| silu_and_mul | 4096 | 4096 | torch.float32 | direct | 0.07 | 0.49 | 2.95 |

### tileops_explicit_parallel

| op_name | M | N | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| silu_and_mul | 1024 | 4096 | torch.float16 | explicit_parallel | 0.01 | 0.90 | 2.69 |
| silu_and_mul | 1024 | 4096 | torch.bfloat16 | explicit_parallel | 0.01 | 0.89 | 2.67 |
| silu_and_mul | 1024 | 4096 | torch.float32 | explicit_parallel | 0.02 | 0.55 | 3.29 |
| silu_and_mul | 1024 | 10240 | torch.float16 | explicit_parallel | 0.02 | 1.07 | 3.20 |
| silu_and_mul | 1024 | 10240 | torch.bfloat16 | explicit_parallel | 0.02 | 1.05 | 3.14 |
| silu_and_mul | 1024 | 10240 | torch.float32 | explicit_parallel | 0.03 | 0.64 | 3.81 |
| silu_and_mul | 4096 | 4096 | torch.float16 | explicit_parallel | 0.03 | 1.13 | 3.40 |
| silu_and_mul | 4096 | 4096 | torch.bfloat16 | explicit_parallel | 0.03 | 1.11 | 3.34 |
| silu_and_mul | 4096 | 4096 | torch.float32 | explicit_parallel | 0.05 | 0.67 | 4.02 |

## gelu_and_mul_strategy

### tileops_direct

| op_name | M | N | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gelu_and_mul | 1024 | 4096 | torch.float16 | direct | 0.02 | 0.49 | 1.46 |
| gelu_and_mul | 1024 | 4096 | torch.bfloat16 | direct | 0.02 | 0.49 | 1.46 |
| gelu_and_mul | 1024 | 4096 | torch.float32 | direct | 0.02 | 0.43 | 2.57 |
| gelu_and_mul | 1024 | 10240 | torch.float16 | direct | 0.04 | 0.53 | 1.59 |
| gelu_and_mul | 1024 | 10240 | torch.bfloat16 | direct | 0.04 | 0.53 | 1.58 |
| gelu_and_mul | 1024 | 10240 | torch.float32 | direct | 0.04 | 0.48 | 2.85 |
| gelu_and_mul | 4096 | 4096 | torch.float16 | direct | 0.06 | 0.55 | 1.64 |
| gelu_and_mul | 4096 | 4096 | torch.bfloat16 | direct | 0.06 | 0.55 | 1.64 |
| gelu_and_mul | 4096 | 4096 | torch.float32 | direct | 0.07 | 0.49 | 2.97 |

### tileops_explicit_parallel

| op_name | M | N | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gelu_and_mul | 1024 | 4096 | torch.float16 | explicit_parallel | 0.01 | 0.77 | 2.31 |
| gelu_and_mul | 1024 | 4096 | torch.bfloat16 | explicit_parallel | 0.01 | 0.77 | 2.31 |
| gelu_and_mul | 1024 | 4096 | torch.float32 | explicit_parallel | 0.02 | 0.55 | 3.29 |
| gelu_and_mul | 1024 | 10240 | torch.float16 | explicit_parallel | 0.02 | 0.89 | 2.66 |
| gelu_and_mul | 1024 | 10240 | torch.bfloat16 | explicit_parallel | 0.02 | 0.88 | 2.64 |
| gelu_and_mul | 1024 | 10240 | torch.float32 | explicit_parallel | 0.03 | 0.62 | 3.74 |
| gelu_and_mul | 4096 | 4096 | torch.float16 | explicit_parallel | 0.03 | 0.98 | 2.95 |
| gelu_and_mul | 4096 | 4096 | torch.bfloat16 | explicit_parallel | 0.03 | 0.98 | 2.95 |
| gelu_and_mul | 4096 | 4096 | torch.float32 | explicit_parallel | 0.05 | 0.67 | 4.01 |

## gelu_tanh_and_mul_strategy

### tileops_direct

| op_name | M | N | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gelu_tanh_and_mul | 1024 | 4096 | torch.float16 | direct | 0.02 | 0.49 | 1.48 |
| gelu_tanh_and_mul | 1024 | 4096 | torch.bfloat16 | direct | 0.02 | 0.49 | 1.48 |
| gelu_tanh_and_mul | 1024 | 4096 | torch.float32 | direct | 0.02 | 0.43 | 2.60 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.float16 | direct | 0.04 | 0.54 | 1.61 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.bfloat16 | direct | 0.04 | 0.54 | 1.61 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.float32 | direct | 0.04 | 0.48 | 2.89 |
| gelu_tanh_and_mul | 4096 | 4096 | torch.float16 | direct | 0.06 | 0.56 | 1.67 |
| gelu_tanh_and_mul | 4096 | 4096 | torch.bfloat16 | direct | 0.06 | 0.56 | 1.67 |
| gelu_tanh_and_mul | 4096 | 4096 | torch.float32 | direct | 0.07 | 0.50 | 3.01 |

### tileops_explicit_parallel

| op_name | M | N | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gelu_tanh_and_mul | 1024 | 4096 | torch.float16 | explicit_parallel | 0.01 | 0.90 | 2.69 |
| gelu_tanh_and_mul | 1024 | 4096 | torch.bfloat16 | explicit_parallel | 0.01 | 0.90 | 2.69 |
| gelu_tanh_and_mul | 1024 | 4096 | torch.float32 | explicit_parallel | 0.02 | 0.55 | 3.31 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.float16 | explicit_parallel | 0.02 | 1.06 | 3.19 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.bfloat16 | explicit_parallel | 0.02 | 1.06 | 3.17 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.float32 | explicit_parallel | 0.03 | 0.64 | 3.83 |
| gelu_tanh_and_mul | 4096 | 4096 | torch.float16 | explicit_parallel | 0.03 | 1.13 | 3.38 |
| gelu_tanh_and_mul | 4096 | 4096 | torch.bfloat16 | explicit_parallel | 0.03 | 1.12 | 3.37 |
| gelu_tanh_and_mul | 4096 | 4096 | torch.float32 | explicit_parallel | 0.05 | 0.67 | 4.03 |

## sub_bcast

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| sub | torch.float16 | 0.01 | 0.61 | 2.46 |
| sub | torch.float16 | 0.01 | 0.76 | 3.03 |
| sub | torch.float16 | 0.03 | 0.83 | 3.33 |

### baseline

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| sub | torch.float16 | 0.01 | 0.29 | 1.15 |
| sub | torch.float16 | 0.03 | 0.34 | 1.36 |
| sub | torch.float16 | 0.06 | 0.37 | 1.47 |

## mul_bcast

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| mul | torch.float16 | 0.01 | 0.61 | 2.46 |
| mul | torch.float16 | 0.01 | 0.76 | 3.03 |
| mul | torch.float16 | 0.03 | 0.83 | 3.33 |

### baseline

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| mul | torch.float16 | 0.01 | 0.29 | 1.15 |
| mul | torch.float16 | 0.03 | 0.34 | 1.36 |
| mul | torch.float16 | 0.06 | 0.37 | 1.47 |

## div_bcast

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| div | torch.float16 | 0.01 | 0.58 | 2.31 |
| div | torch.float16 | 0.01 | 0.72 | 2.90 |
| div | torch.float16 | 0.03 | 0.79 | 3.16 |

### baseline

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| div | torch.float16 | 0.02 | 0.27 | 1.08 |
| div | torch.float16 | 0.03 | 0.32 | 1.27 |
| div | torch.float16 | 0.06 | 0.34 | 1.37 |

## binary_strategy

### add_direct

| n_total | shape_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4194304 | 1024x4096 | torch.float16 | direct | 0.02 | 0.26 | 1.58 |
| 4194304 | 1024x4096 | torch.bfloat16 | direct | 0.02 | 0.26 | 1.58 |
| 4194304 | 1024x4096 | torch.float32 | direct | 0.02 | 0.22 | 2.59 |
| 10485760 | 1024x10240 | torch.float16 | direct | 0.04 | 0.28 | 1.69 |
| 10485760 | 1024x10240 | torch.bfloat16 | direct | 0.04 | 0.28 | 1.69 |
| 10485760 | 1024x10240 | torch.float32 | direct | 0.04 | 0.24 | 2.92 |
| 20971520 | 1024x20480 | torch.float16 | direct | 0.07 | 0.30 | 1.79 |
| 20971520 | 1024x20480 | torch.bfloat16 | direct | 0.07 | 0.30 | 1.79 |
| 20971520 | 1024x20480 | torch.float32 | direct | 0.08 | 0.26 | 3.10 |

### add_explicit_parallel

| n_total | shape_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4194304 | 1024x4096 | torch.float16 | explicit_parallel | 0.01 | 0.46 | 2.75 |
| 4194304 | 1024x4096 | torch.bfloat16 | explicit_parallel | 0.01 | 0.46 | 2.76 |
| 4194304 | 1024x4096 | torch.float32 | explicit_parallel | 0.02 | 0.27 | 3.30 |
| 10485760 | 1024x10240 | torch.float16 | explicit_parallel | 0.02 | 0.55 | 3.30 |
| 10485760 | 1024x10240 | torch.bfloat16 | explicit_parallel | 0.02 | 0.55 | 3.30 |
| 10485760 | 1024x10240 | torch.float32 | explicit_parallel | 0.03 | 0.32 | 3.85 |
| 20971520 | 1024x20480 | torch.float16 | explicit_parallel | 0.03 | 0.62 | 3.70 |
| 20971520 | 1024x20480 | torch.bfloat16 | explicit_parallel | 0.03 | 0.62 | 3.70 |
| 20971520 | 1024x20480 | torch.float32 | explicit_parallel | 0.06 | 0.34 | 4.12 |

## cumulative

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | cumsum | 0.06 | 0.07 | 0.28 |
| 1024 | 4096 | torch.bfloat16 | cumsum | 0.06 | 0.07 | 0.28 |
| 4096 | 4096 | torch.float16 | cumsum | 0.12 | 0.14 | 0.56 |
| 1024 | 4096 | torch.float16 | cumprod | 0.06 | 0.07 | 0.28 |
| 1024 | 4096 | torch.bfloat16 | cumprod | 0.06 | 0.07 | 0.28 |
| 4096 | 4096 | torch.float16 | cumprod | 0.12 | 0.14 | 0.56 |

### baseline

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | cumsum | 0.10 | 0.04 | 0.18 |
| 1024 | 4096 | torch.bfloat16 | cumsum | 0.10 | 0.04 | 0.18 |
| 4096 | 4096 | torch.float16 | cumsum | 0.25 | 0.07 | 0.27 |
| 1024 | 4096 | torch.float16 | cumprod | 0.10 | 0.04 | 0.18 |
| 1024 | 4096 | torch.bfloat16 | cumprod | 0.10 | 0.04 | 0.18 |
| 4096 | 4096 | torch.float16 | cumprod | 0.25 | 0.07 | 0.27 |

## dsa_decode

### tileops

| batch | heads | seq_len_q | seq_len_kv | dim | dim_tail | topk | stride_kv | heads_kv | q_start_index_s | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 128 | 1024 | 2048 | 512 | 64 | 2048 | 1 | 1 | 1024 | torch.float16 | 1.08 | 539.71 | 0.27 |
| 1 | 128 | 512 | 4096 | 512 | 64 | 1024 | 1 | 1 | 512 | torch.float16 | 0.30 | 479.73 | 0.49 |

### baseline

| batch | heads | seq_len_q | seq_len_kv | dim | dim_tail | topk | stride_kv | heads_kv | q_start_index_s | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 128 | 1024 | 2048 | 512 | 64 | 2048 | 1 | 1 | 1024 | torch.float16 | 15.11 | 38.66 | 0.02 |
| 1 | 128 | 512 | 4096 | 512 | 64 | 1024 | 1 | 1 | 512 | torch.float16 | 15.19 | 9.62 | 0.01 |

## mla_decode

### tileops

| batch | heads | heads_kv | seq_len_kv | dim | dim_pe | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 128 | 1 | 8192 | 512 | 64 | torch.float16 | 0.16 | 443.24 | 1.86 |
| 16 | 128 | 1 | 4096 | 512 | 64 | torch.float16 | 0.05 | 374.74 | 1.60 |

### baseline

| batch | heads | heads_kv | seq_len_kv | dim | dim_pe | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 128 | 1 | 8192 | 512 | 64 | torch.float16 | 0.71 | 102.29 | 0.43 |
| 16 | 128 | 1 | 4096 | 512 | 64 | torch.float16 | 0.19 | 96.16 | 0.41 |

## nsa_cmp_fwd

### tileops

| seq_num | c_seq_len | heads | dim_k | dim_v | group | scale | bc | bs | bk | bv | dtype | accum_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 9 | 8192 | 32 | 128 | 128 | 16 | 0.08838834764831845 | 32 | 32 | 128 | 128 | torch.float16 | torch.float32 | 0.31 | 55.53 | 7.16 |
| 16 | 16384 | 32 | 128 | 128 | 16 | 0.08838834764831845 | 32 | 32 | 128 | 128 | torch.float16 | torch.float32 | 0.34 | 203.68 | 25.86 |

### baseline

| seq_num | c_seq_len | heads | dim_k | dim_v | group | scale | bc | bs | bk | bv | dtype | accum_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 9 | 8192 | 32 | 128 | 128 | 16 | 0.08838834764831845 | 32 | 32 | 128 | 128 | torch.float16 | torch.float32 | 304.66 | 0.06 | 0.01 |
| 16 | 16384 | 32 | 128 | 128 | 16 | 0.08838834764831845 | 32 | 32 | 128 | 128 | torch.float16 | torch.float32 | 591.79 | 0.12 | 0.01 |

## nsa_fwd

### tileops

| batch | heads | c_seq_len | dim | is_causal | scale | block_size | groups | selected_blocks | dtype | accum_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 16 | 1024 | 64 | True | 0.1 | 32 | 16 | 1 | torch.float16 | torch.float32 | 0.01 | 15.10 | 0.50 |
| 4 | 16 | 8192 | 64 | True | 0.1 | 32 | 16 | 1 | torch.float16 | torch.float32 | 0.04 | 28.38 | 0.94 |
| 2 | 16 | 8192 | 64 | True | 0.1 | 32 | 16 | 4 | torch.float16 | torch.float32 | 0.05 | 78.33 | 0.65 |

### baseline

| batch | heads | c_seq_len | dim | is_causal | scale | block_size | groups | selected_blocks | dtype | accum_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 16 | 1024 | 64 | True | 0.1 | 32 | 16 | 1 | torch.float16 | torch.float32 | 64.86 | 0.00 | 0.00 |
| 4 | 16 | 8192 | 64 | True | 0.1 | 32 | 16 | 1 | torch.float16 | torch.float32 | 518.95 | 0.00 | 0.00 |
| 2 | 16 | 8192 | 64 | True | 0.1 | 32 | 16 | 4 | torch.float16 | torch.float32 | 478.04 | 0.01 | 0.00 |

## nsa_topk

### tileops

| seq_num | c_seq_len | heads | dim | group | scale | selected_block_num | bc | bs | bk | dtype | accum_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 1024 | 32 | 128 | 16 | 1 | 16 | 32 | 32 | 128 | torch.float16 | torch.float32 | 0.04 | 6.93 | 0.65 |
| 3 | 512 | 32 | 128 | 16 | 1 | 16 | 32 | 32 | 128 | torch.float16 | torch.float32 | 0.02 | 2.76 | 0.35 |
| 9 | 8192 | 32 | 128 | 16 | 1 | 16 | 32 | 32 | 128 | torch.float16 | torch.float32 | 0.38 | 45.58 | 3.03 |

### baseline

| seq_num | c_seq_len | heads | dim | group | scale | selected_block_num | bc | bs | bk | dtype | accum_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 1024 | 32 | 128 | 16 | 1 | 16 | 32 | 32 | 128 | torch.float16 | torch.float32 | 240.27 | 0.00 | 0.00 |
| 3 | 512 | 32 | 128 | 16 | 1 | 16 | 32 | 32 | 128 | torch.float16 | torch.float32 | 119.50 | 0.00 | 0.00 |
| 9 | 8192 | 32 | 128 | 16 | 1 | 16 | 32 | 32 | 128 | torch.float16 | torch.float32 | 2492.99 | 0.01 | 0.00 |

## dropout

### tileops

| dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| torch.float16 | 4194304 | 0.01 | 0.64 | 2.57 |
| torch.bfloat16 | 4194304 | 0.01 | 0.64 | 2.56 |
| torch.float32 | 4194304 | 0.01 | 0.39 | 3.11 |
| torch.float16 | 10485760 | 0.01 | 0.83 | 3.31 |
| torch.bfloat16 | 10485760 | 0.01 | 0.83 | 3.31 |
| torch.float32 | 10485760 | 0.02 | 0.48 | 3.81 |
| torch.float16 | 20971520 | 0.02 | 0.95 | 3.81 |
| torch.bfloat16 | 20971520 | 0.02 | 0.95 | 3.81 |
| torch.float32 | 20971520 | 0.04 | 0.51 | 4.10 |

### baseline

| dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| torch.float16 | 4194304 | 0.01 | 0.39 | 1.54 |
| torch.bfloat16 | 4194304 | 0.01 | 0.38 | 1.52 |
| torch.float32 | 4194304 | 0.01 | 0.28 | 2.26 |
| torch.float16 | 10485760 | 0.02 | 0.50 | 2.01 |
| torch.bfloat16 | 10485760 | 0.02 | 0.50 | 1.98 |
| torch.float32 | 10485760 | 0.03 | 0.33 | 2.62 |
| torch.float16 | 20971520 | 0.04 | 0.55 | 2.21 |
| torch.bfloat16 | 20971520 | 0.04 | 0.54 | 2.17 |
| torch.float32 | 20971520 | 0.06 | 0.35 | 2.78 |

## relu_fp8

### tileops

| op_name | n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| relu_fp8 | 8388608 | torch.float8_e4m3fn | 0.01 | 0.57 | 1.14 |
| relu_fp8 | 8388608 | torch.float8_e5m2 | 0.02 | 0.45 | 0.91 |
| relu_fp8 | 67108864 | torch.float8_e4m3fn | 0.09 | 0.72 | 1.44 |
| relu_fp8 | 67108864 | torch.float8_e5m2 | 0.12 | 0.55 | 1.10 |
| relu_fp8 | 134217728 | torch.float8_e4m3fn | 0.18 | 0.75 | 1.49 |
| relu_fp8 | 134217728 | torch.float8_e5m2 | 0.24 | 0.57 | 1.14 |

### baseline

| op_name | n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| relu_fp8 | 8388608 | torch.float8_e4m3fn | 0.05 | 0.19 | 0.37 |
| relu_fp8 | 8388608 | torch.float8_e5m2 | 0.04 | 0.19 | 0.38 |
| relu_fp8 | 67108864 | torch.float8_e4m3fn | 0.33 | 0.20 | 0.40 |
| relu_fp8 | 67108864 | torch.float8_e5m2 | 0.32 | 0.21 | 0.41 |
| relu_fp8 | 134217728 | torch.float8_e4m3fn | 0.64 | 0.21 | 0.42 |
| relu_fp8 | 134217728 | torch.float8_e5m2 | 0.62 | 0.22 | 0.43 |

## exp_fp8

### tileops

| op_name | n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| exp_fp8 | 8388608 | torch.float8_e4m3fn | 0.01 | 0.98 | 1.96 |
| exp_fp8 | 8388608 | torch.float8_e5m2 | 0.02 | 0.46 | 0.91 |
| exp_fp8 | 67108864 | torch.float8_e4m3fn | 0.05 | 1.33 | 2.67 |
| exp_fp8 | 67108864 | torch.float8_e5m2 | 0.12 | 0.55 | 1.09 |
| exp_fp8 | 134217728 | torch.float8_e4m3fn | 0.10 | 1.40 | 2.79 |
| exp_fp8 | 134217728 | torch.float8_e5m2 | 0.24 | 0.57 | 1.13 |

### baseline

| op_name | n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| exp_fp8 | 8388608 | torch.float8_e4m3fn | 0.05 | 0.19 | 0.37 |
| exp_fp8 | 8388608 | torch.float8_e5m2 | 0.04 | 0.19 | 0.38 |
| exp_fp8 | 67108864 | torch.float8_e4m3fn | 0.33 | 0.20 | 0.41 |
| exp_fp8 | 67108864 | torch.float8_e5m2 | 0.32 | 0.21 | 0.42 |
| exp_fp8 | 134217728 | torch.float8_e4m3fn | 0.63 | 0.21 | 0.43 |
| exp_fp8 | 134217728 | torch.float8_e5m2 | 0.61 | 0.22 | 0.44 |

## add_fp8

### tileops

| op_name | n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| add_fp8 | 8388608 | torch.float8_e4m3fn | 0.01 | 0.86 | 2.57 |
| add_fp8 | 8388608 | torch.float8_e5m2 | 0.02 | 0.41 | 1.24 |
| add_fp8 | 67108864 | torch.float8_e4m3fn | 0.06 | 1.16 | 3.48 |
| add_fp8 | 67108864 | torch.float8_e5m2 | 0.13 | 0.51 | 1.52 |
| add_fp8 | 134217728 | torch.float8_e4m3fn | 0.11 | 1.22 | 3.65 |
| add_fp8 | 134217728 | torch.float8_e5m2 | 0.25 | 0.53 | 1.58 |

### baseline

| op_name | n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| add_fp8 | 8388608 | torch.float8_e4m3fn | 0.08 | 0.11 | 0.33 |
| add_fp8 | 8388608 | torch.float8_e5m2 | 0.07 | 0.11 | 0.34 |
| add_fp8 | 67108864 | torch.float8_e4m3fn | 0.55 | 0.12 | 0.37 |
| add_fp8 | 67108864 | torch.float8_e5m2 | 0.53 | 0.13 | 0.38 |
| add_fp8 | 134217728 | torch.float8_e4m3fn | 1.04 | 0.13 | 0.39 |
| add_fp8 | 134217728 | torch.float8_e5m2 | 1.01 | 0.13 | 0.40 |

## silu_and_mul_fp8

### tileops

| op_name | M | N | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| silu_and_mul_fp8 | 2048 | 11008 | torch.float8_e4m3fn | 0.04 | 2.59 | 1.56 |
| silu_and_mul_fp8 | 2048 | 11008 | torch.float8_e5m2 | 0.06 | 1.77 | 1.06 |
| silu_and_mul_fp8 | 16384 | 11008 | torch.float8_e4m3fn | 0.30 | 2.98 | 1.79 |
| silu_and_mul_fp8 | 16384 | 11008 | torch.float8_e5m2 | 0.44 | 2.05 | 1.23 |
| silu_and_mul_fp8 | 16384 | 28672 | torch.float8_e4m3fn | 0.72 | 3.26 | 1.95 |
| silu_and_mul_fp8 | 16384 | 28672 | torch.float8_e5m2 | 1.04 | 2.26 | 1.36 |

### baseline

| op_name | M | N | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| silu_and_mul_fp8 | 2048 | 11008 | torch.float8_e4m3fn | 0.29 | 0.39 | 0.23 |
| silu_and_mul_fp8 | 2048 | 11008 | torch.float8_e5m2 | 0.28 | 0.40 | 0.24 |
| silu_and_mul_fp8 | 16384 | 11008 | torch.float8_e4m3fn | 1.90 | 0.48 | 0.29 |
| silu_and_mul_fp8 | 16384 | 11008 | torch.float8_e5m2 | 1.86 | 0.49 | 0.29 |
| silu_and_mul_fp8 | 16384 | 28672 | torch.float8_e4m3fn | 3.32 | 0.71 | 0.42 |
| silu_and_mul_fp8 | 16384 | 28672 | torch.float8_e5m2 | 3.30 | 0.71 | 0.43 |

## engram_gate_conv_bwd

### tileops

| M | seq_len | d | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 256 | torch.float16 | 0.01 | 0.07 | 0.03 |
| 2 | 64 | 512 | torch.float16 | 0.01 | 0.33 | 0.11 |
| 1 | 128 | 256 | torch.bfloat16 | 0.01 | 0.19 | 0.06 |
| 2 | 16 | 256 | torch.bfloat16 | 0.01 | 0.07 | 0.02 |

## engram_decode

### tileops

| batch | d_mem | d | max_conv_len | conv_kernel_size | dilation | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 512 | 256 | 12 | 4 | 3 | torch.float16 | 0.03 | 0.02 | 0.02 |
| 4 | 1024 | 512 | 20 | 4 | 5 | torch.float16 | 0.07 | 0.12 | 0.03 |
| 8 | 512 | 256 | 18 | 4 | 3 | torch.bfloat16 | 0.03 | 0.14 | 0.02 |

### baseline

| batch | d_mem | d | max_conv_len | conv_kernel_size | dilation | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 512 | 256 | 12 | 4 | 3 | torch.float16 | 0.10 | 0.01 | 0.01 |
| 4 | 1024 | 512 | 20 | 4 | 5 | torch.float16 | 0.13 | 0.06 | 0.02 |
| 8 | 512 | 256 | 18 | 4 | 3 | torch.bfloat16 | 0.12 | 0.04 | 0.01 |

## engram_gate_conv_fwd

### tileops

| M | seq_len | d | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 256 | torch.float16 | 0.00 | 0.05 | 0.02 |
| 2 | 64 | 512 | torch.float16 | 0.01 | 0.30 | 0.13 |
| 1 | 128 | 256 | torch.bfloat16 | 0.00 | 0.16 | 0.07 |
| 2 | 16 | 256 | torch.bfloat16 | 0.00 | 0.05 | 0.02 |

### baseline

| M | seq_len | d | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 256 | torch.float16 | 0.08 | 0.00 | 0.00 |
| 2 | 64 | 512 | torch.float16 | 0.09 | 0.02 | 0.01 |
| 1 | 128 | 256 | torch.bfloat16 | 0.09 | 0.01 | 0.00 |
| 2 | 16 | 256 | torch.bfloat16 | 0.08 | 0.00 | 0.00 |

## fft_c2c

### tileops

| n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 64 | torch.complex64 | 0.02 | 0.00 | 0.00 |
| 128 | torch.complex64 | 0.02 | 0.00 | 0.00 |
| 256 | torch.complex64 | 0.03 | 0.00 | 0.00 |
| 512 | torch.complex64 | 0.03 | 0.00 | 0.00 |
| 1024 | torch.complex64 | 0.03 | 0.00 | 0.00 |

### baseline

| n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 64 | torch.complex64 | 0.00 | 0.00 | 0.00 |
| 128 | torch.complex64 | 0.00 | 0.00 | 0.00 |
| 256 | torch.complex64 | 0.00 | 0.00 | 0.00 |
| 512 | torch.complex64 | 0.00 | 0.01 | 0.00 |
| 1024 | torch.complex64 | 0.00 | 0.02 | 0.01 |

## fft_c2c_lut

### tileops-lut

| n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 64 | torch.complex64 | 0.01 | 0.00 | 0.00 |
| 128 | torch.complex64 | 0.01 | 0.00 | 0.00 |
| 256 | torch.complex64 | 0.01 | 0.00 | 0.00 |
| 512 | torch.complex64 | 0.01 | 0.00 | 0.00 |
| 1024 | torch.complex64 | 0.01 | 0.00 | 0.00 |
| 4096 | torch.complex64 | 0.01 | 0.02 | 0.01 |
| 16384 | torch.complex64 | 0.01 | 0.08 | 0.02 |

### tileops-base

| n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 64 | torch.complex64 | 0.02 | 0.00 | 0.00 |
| 128 | torch.complex64 | 0.02 | 0.00 | 0.00 |
| 256 | torch.complex64 | 0.03 | 0.00 | 0.00 |
| 512 | torch.complex64 | 0.03 | 0.00 | 0.00 |
| 1024 | torch.complex64 | 0.03 | 0.00 | 0.00 |
| 4096 | torch.complex64 | 0.03 | 0.01 | 0.00 |
| 16384 | torch.complex64 | 0.04 | 0.03 | 0.01 |

### baseline

| n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 64 | torch.complex64 | 0.00 | 0.00 | 0.00 |
| 128 | torch.complex64 | 0.00 | 0.00 | 0.00 |
| 256 | torch.complex64 | 0.00 | 0.00 | 0.00 |
| 512 | torch.complex64 | 0.00 | 0.01 | 0.00 |
| 1024 | torch.complex64 | 0.00 | 0.02 | 0.01 |
| 4096 | torch.complex64 | 0.01 | 0.05 | 0.01 |
| 16384 | torch.complex64 | 0.01 | 0.10 | 0.02 |

## fp8_lighting_indexer

### tileops

| batch | seq_len | heads | index_dim | seq_len_kv | kv_group | clean_logits | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 4096 | 32 | 64 | 8192 | 1 | True | 0.19 | N/A | 0.77 |
| 1 | 2048 | 16 | 64 | 4096 | 1 | True | 0.12 | N/A | 0.30 |

### baseline

| batch | seq_len | heads | index_dim | seq_len_kv | kv_group | clean_logits | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 4096 | 32 | 64 | 8192 | 1 | True | 8.92 | N/A | 0.02 |
| 1 | 2048 | 16 | 64 | 4096 | 1 | True | 1.51 | N/A | 0.02 |

## fp8_quant

### tileops

| batch | seq_len_kv | kv_group | index_dim | in_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8192 | 1 | 64 | torch.float16 | 0.00 | 1.05 | 0.35 |
| 1 | 8192 | 1 | 64 | torch.bfloat16 | 0.00 | 1.04 | 0.35 |
| 1 | 4096 | 1 | 128 | torch.float32 | 0.00 | 0.78 | 0.52 |
| 1 | 16384 | 1 | 32 | torch.float32 | 0.00 | 0.98 | 0.65 |

### baseline

| batch | seq_len_kv | kv_group | index_dim | in_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8192 | 1 | 64 | torch.float16 | 0.02 | 0.18 | 0.06 |
| 1 | 8192 | 1 | 64 | torch.bfloat16 | 0.02 | 0.18 | 0.06 |
| 1 | 4096 | 1 | 128 | torch.float32 | 0.02 | 0.18 | 0.12 |
| 1 | 16384 | 1 | 32 | torch.float32 | 0.02 | 0.15 | 0.10 |

## fused_add_layer_norm

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.02 | 1.25 | 1.67 |
| 4096 | 4096 | torch.bfloat16 | 0.07 | 1.53 | 2.04 |
| 1024 | 3000 | torch.float16 | 0.04 | 0.52 | 0.70 |
| 1025 | 4096 | torch.float16 | 0.02 | 1.26 | 1.68 |

### baseline

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.06 | 0.44 | 0.59 |
| 4096 | 4096 | torch.bfloat16 | 0.21 | 0.48 | 0.64 |
| 1024 | 3000 | torch.float16 | 0.04 | 0.42 | 0.56 |
| 1025 | 4096 | torch.float16 | 0.06 | 0.44 | 0.59 |

## fused_add_rmsnorm

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.02 | 1.14 | 1.82 |
| 4096 | 4096 | torch.bfloat16 | 0.06 | 1.43 | 2.28 |
| 2048 | 5120 | torch.float16 | 0.04 | 1.37 | 2.20 |
| 1025 | 4096 | torch.float16 | 0.02 | 1.14 | 1.82 |

### baseline

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.13 | 0.17 | 0.27 |
| 4096 | 4096 | torch.bfloat16 | 0.48 | 0.17 | 0.28 |
| 2048 | 5120 | torch.float16 | 0.32 | 0.17 | 0.26 |
| 1025 | 4096 | torch.float16 | 0.13 | 0.17 | 0.27 |

## gated_deltanet_fwd

### tileops

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 0.19 | 1.42 | 0.09 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 0.20 | 1.37 | 0.09 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 0.08 | 1.66 | 0.10 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 0.14 | 1.94 | 0.12 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 0.28 | 1.89 | 0.12 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 0.53 | 2.02 | 0.13 |
| 2 | 32768 | 4 | 64 | 64 | 64 | torch.float16 | 1.01 | 2.13 | 0.13 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.08 | 1.65 | 0.10 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.14 | 1.95 | 0.12 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.28 | 1.91 | 0.12 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.53 | 2.03 | 0.13 |
| 2 | 32768 | 4 | 64 | 64 | 64 | torch.bfloat16 | 1.00 | 2.14 | 0.13 |

### baseline

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 131.52 | 0.00 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 130.63 | 0.00 | 0.00 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 45.11 | 0.00 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 90.87 | 0.00 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 183.24 | 0.00 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 364.95 | 0.00 | 0.00 |
| 2 | 32768 | 4 | 64 | 64 | 64 | torch.float16 | 730.33 | 0.00 | 0.00 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 45.34 | 0.00 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 90.81 | 0.00 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 183.02 | 0.00 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 366.81 | 0.00 | 0.00 |
| 2 | 32768 | 4 | 64 | 64 | 64 | torch.bfloat16 | 732.22 | 0.00 | 0.00 |

## gated_deltanet_bwd

### tileops

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | H | S | DK | DV | BC | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.42 | 1.27 | 0.07 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.42 | 1.28 | 0.07 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.19 | 1.43 | 0.08 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.34 | 1.56 | 0.09 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.63 | 1.71 | 0.09 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 16384 | 64 | 64 | 64 | 1.19 | 1.80 | 0.10 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.19 | 1.42 | 0.08 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.35 | 1.55 | 0.09 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.63 | 1.70 | 0.09 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 16384 | 64 | 64 | 64 | 1.20 | 1.79 | 0.10 |

### baseline

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | H | S | DK | DV | BC | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 32 | 49.99 | 0.01 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 32 | 49.99 | 0.01 | 0.00 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 2048 | 64 | 64 | 64 | 19.20 | 0.01 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 64 | 39.49 | 0.01 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 8192 | 64 | 64 | 64 | 82.34 | 0.01 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 16384 | 64 | 64 | 64 | 175.23 | 0.01 | 0.00 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 2048 | 64 | 64 | 64 | 19.20 | 0.01 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 64 | 39.51 | 0.01 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 8192 | 64 | 64 | 64 | 82.45 | 0.01 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 16384 | 64 | 64 | 64 | 175.28 | 0.01 | 0.00 |

## gated_deltanet_fwdbwd

### tileops

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | H | S | DK | DV | BC | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.60 | 1.35 | 0.08 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.60 | 1.34 | 0.08 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.26 | 1.55 | 0.09 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.47 | 1.70 | 0.10 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.89 | 1.81 | 0.10 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 16384 | 64 | 64 | 64 | 1.65 | 1.96 | 0.11 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.26 | 1.53 | 0.09 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.47 | 1.70 | 0.10 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.89 | 1.81 | 0.10 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 16384 | 64 | 64 | 64 | 1.65 | 1.95 | 0.11 |

### baseline

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | H | S | DK | DV | BC | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 32 | 49.93 | 0.02 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 32 | 49.99 | 0.02 | 0.00 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 2048 | 64 | 64 | 64 | 19.21 | 0.02 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 64 | 39.51 | 0.02 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 8192 | 64 | 64 | 64 | 82.44 | 0.02 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 16384 | 64 | 64 | 64 | 175.35 | 0.02 | 0.00 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 2048 | 64 | 64 | 64 | 19.23 | 0.02 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 64 | 39.50 | 0.02 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 8192 | 64 | 64 | 64 | 82.44 | 0.02 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 16384 | 64 | 64 | 64 | 175.40 | 0.02 | 0.00 |

## gated_deltanet_decode

### tileops

| batch | heads | dim_k | dim_v | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 128 | 128 | torch.float32 | 0.01 | 0.30 | 0.41 |
| 1 | 32 | 128 | 128 | torch.bfloat16 | 0.01 | 0.31 | 0.21 |
| 8 | 32 | 128 | 128 | torch.float32 | 0.02 | 1.18 | 1.59 |
| 8 | 32 | 128 | 128 | torch.bfloat16 | 0.01 | 2.10 | 1.42 |
| 16 | 32 | 128 | 128 | torch.float32 | 0.04 | 1.28 | 1.74 |
| 16 | 32 | 128 | 128 | torch.bfloat16 | 0.02 | 3.04 | 2.06 |
| 32 | 32 | 128 | 128 | torch.float32 | 0.08 | 1.34 | 1.81 |
| 32 | 32 | 128 | 128 | torch.bfloat16 | 0.03 | 3.29 | 2.22 |
| 1 | 32 | 64 | 64 | torch.float32 | 0.01 | 0.14 | 0.19 |
| 8 | 32 | 64 | 64 | torch.float32 | 0.01 | 0.60 | 0.82 |
| 32 | 32 | 64 | 64 | torch.float32 | 0.04 | 0.71 | 0.97 |
| 64 | 32 | 128 | 128 | torch.float32 | 0.14 | 1.40 | 1.89 |
| 64 | 32 | 128 | 128 | torch.bfloat16 | 0.05 | 3.69 | 2.49 |

### baseline

| batch | heads | dim_k | dim_v | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 128 | 128 | torch.float32 | 0.03 | 0.09 | 0.13 |
| 1 | 32 | 128 | 128 | torch.bfloat16 | 0.05 | 0.06 | 0.04 |
| 8 | 32 | 128 | 128 | torch.float32 | 0.08 | 0.31 | 0.42 |
| 8 | 32 | 128 | 128 | torch.bfloat16 | 0.11 | 0.23 | 0.16 |
| 16 | 32 | 128 | 128 | torch.float32 | 0.12 | 0.41 | 0.56 |
| 16 | 32 | 128 | 128 | torch.bfloat16 | 0.18 | 0.29 | 0.19 |
| 32 | 32 | 128 | 128 | torch.float32 | 0.21 | 0.47 | 0.64 |
| 32 | 32 | 128 | 128 | torch.bfloat16 | 0.31 | 0.33 | 0.22 |
| 1 | 32 | 64 | 64 | torch.float32 | 0.03 | 0.03 | 0.04 |
| 8 | 32 | 64 | 64 | torch.float32 | 0.04 | 0.16 | 0.22 |
| 32 | 32 | 64 | 64 | torch.float32 | 0.08 | 0.32 | 0.44 |
| 64 | 32 | 128 | 128 | torch.float32 | 0.39 | 0.52 | 0.70 |
| 64 | 32 | 128 | 128 | torch.bfloat16 | 0.55 | 0.36 | 0.25 |

## gemm

### tileops

| m | n | k | dtype | trans_a | trans_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1024 | 1024 | 1024 | torch.float16 | False | False | 0.01 | 241.90 | 0.71 |
| 1 | 7168 | 16384 | torch.float16 | False | True | 0.06 | 3.67 | 3.67 |
| 1 | 18432 | 7168 | torch.bfloat16 | False | True | 0.07 | 3.85 | 3.85 |
| 7168 | 1 | 16384 | torch.float16 | False | False | 0.06 | 3.67 | 3.68 |
| 18432 | 1 | 7168 | torch.bfloat16 | False | False | 0.07 | 3.84 | 3.84 |

### baseline

| m | n | k | dtype | trans_a | trans_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1024 | 1024 | 1024 | torch.float16 | False | False | 0.01 | 324.51 | 0.95 |
| 1 | 7168 | 16384 | torch.float16 | False | True | 0.07 | 3.44 | 3.44 |
| 1 | 18432 | 7168 | torch.bfloat16 | False | True | 0.07 | 3.64 | 3.64 |
| 7168 | 1 | 16384 | torch.float16 | False | False | 0.07 | 3.44 | 3.44 |
| 18432 | 1 | 7168 | torch.bfloat16 | False | False | 0.07 | 3.64 | 3.64 |

## gla_fwd

### tileops

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.08 | 1.73 | 0.11 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.14 | 1.99 | 0.12 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.26 | 2.05 | 0.13 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.49 | 2.17 | 0.14 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.08 | 1.66 | 0.10 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.14 | 1.98 | 0.12 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.26 | 2.06 | 0.13 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.50 | 2.16 | 0.14 |

### baseline

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 2.00 | 0.07 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 4.05 | 0.07 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 8.10 | 0.07 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 16.19 | 0.07 | 0.00 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 1.99 | 0.07 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 4.06 | 0.07 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 8.11 | 0.07 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 16.17 | 0.07 | 0.00 |

## gla_bwd

### tileops

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | T | H | K | V | BC | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 2048 | 4 | 64 | 64 | 64 | 0.125 | 0.15 | 1.75 | 0.10 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4096 | 4 | 64 | 64 | 64 | 0.125 | 0.30 | 1.82 | 0.10 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 8192 | 4 | 64 | 64 | 64 | 0.125 | 0.55 | 1.94 | 0.11 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 16384 | 4 | 64 | 64 | 64 | 0.125 | 1.01 | 2.13 | 0.12 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 2048 | 4 | 64 | 64 | 64 | 0.125 | 0.15 | 1.76 | 0.10 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4096 | 4 | 64 | 64 | 64 | 0.125 | 0.30 | 1.80 | 0.10 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 8192 | 4 | 64 | 64 | 64 | 0.125 | 0.57 | 1.89 | 0.10 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 16384 | 4 | 64 | 64 | 64 | 0.125 | 1.05 | 2.04 | 0.11 |

### baseline

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | T | H | K | V | BC | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 2048 | 4 | 64 | 64 | 64 | 0.125 | 5.81 | 0.05 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4096 | 4 | 64 | 64 | 64 | 0.125 | 12.48 | 0.04 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 8192 | 4 | 64 | 64 | 64 | 0.125 | 28.58 | 0.04 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 16384 | 4 | 64 | 64 | 64 | 0.125 | 70.20 | 0.03 | 0.00 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 2048 | 4 | 64 | 64 | 64 | 0.125 | 5.94 | 0.05 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4096 | 4 | 64 | 64 | 64 | 0.125 | 12.49 | 0.04 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 8192 | 4 | 64 | 64 | 64 | 0.125 | 28.59 | 0.04 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 16384 | 4 | 64 | 64 | 64 | 0.125 | 70.26 | 0.03 | 0.00 |

## gla_fwdbwd

### tileops

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | T | B | BC | H | K | V | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2048 | 2 | 64 | 4 | 64 | 64 | 0.125 | 0.22 | 1.79 | 0.10 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 4096 | 2 | 64 | 4 | 64 | 64 | 0.125 | 0.42 | 1.90 | 0.11 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 8192 | 2 | 64 | 4 | 64 | 64 | 0.125 | 0.79 | 2.04 | 0.12 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 16384 | 2 | 64 | 4 | 64 | 64 | 0.125 | 1.39 | 2.31 | 0.13 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2048 | 2 | 64 | 4 | 64 | 64 | 0.125 | 0.23 | 1.78 | 0.10 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 4096 | 2 | 64 | 4 | 64 | 64 | 0.125 | 0.43 | 1.89 | 0.11 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 8192 | 2 | 64 | 4 | 64 | 64 | 0.125 | 0.80 | 2.01 | 0.12 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 16384 | 2 | 64 | 4 | 64 | 64 | 0.125 | 1.46 | 2.20 | 0.13 |

### baseline

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | T | B | BC | H | K | V | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2048 | 2 | 64 | 4 | 64 | 64 | 0.125 | 5.95 | 0.07 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 4096 | 2 | 64 | 4 | 64 | 64 | 0.125 | 12.49 | 0.06 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 8192 | 2 | 64 | 4 | 64 | 64 | 0.125 | 28.56 | 0.06 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 16384 | 2 | 64 | 4 | 64 | 64 | 0.125 | 70.34 | 0.05 | 0.00 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2048 | 2 | 64 | 4 | 64 | 64 | 0.125 | 5.81 | 0.07 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 4096 | 2 | 64 | 4 | 64 | 64 | 0.125 | 12.47 | 0.06 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 8192 | 2 | 64 | 4 | 64 | 64 | 0.125 | 28.55 | 0.06 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 16384 | 2 | 64 | 4 | 64 | 64 | 0.125 | 70.19 | 0.05 | 0.00 |

## gla_decode

### tileops

| batch | heads | dim_k | dim_v | dtype | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 64 | 64 | torch.float32 | 0.125 | 0.01 | 0.07 | 0.14 |
| 1 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.02 | 0.13 | 0.26 |
| 1 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.01 | 0.17 | 0.17 |
| 1 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.01 | 0.17 | 0.17 |
| 8 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.03 | 0.50 | 1.01 |
| 8 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.01 | 1.20 | 1.22 |
| 8 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.01 | 1.18 | 1.20 |
| 16 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.06 | 0.52 | 1.06 |
| 16 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.02 | 1.86 | 1.89 |
| 16 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.02 | 1.84 | 1.87 |
| 32 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.12 | 0.55 | 1.11 |
| 32 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.03 | 1.95 | 1.98 |
| 32 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.03 | 1.93 | 1.96 |
| 64 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.24 | 0.57 | 1.16 |
| 64 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.06 | 2.18 | 2.22 |
| 64 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.06 | 2.16 | 2.20 |

### baseline

| batch | heads | dim_k | dim_v | dtype | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 64 | 64 | torch.float32 | 0.125 | 0.01 | 0.04 | 0.08 |
| 1 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.02 | 0.13 | 0.26 |
| 1 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.04 | 0.06 | 0.06 |
| 1 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.04 | 0.06 | 0.06 |
| 8 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.05 | 0.32 | 0.66 |
| 8 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.09 | 0.20 | 0.20 |
| 8 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.09 | 0.20 | 0.20 |
| 16 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.09 | 0.36 | 0.74 |
| 16 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.15 | 0.23 | 0.23 |
| 16 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.15 | 0.23 | 0.23 |
| 32 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.18 | 0.38 | 0.78 |
| 32 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.27 | 0.25 | 0.26 |
| 32 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.27 | 0.25 | 0.25 |
| 64 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.33 | 0.41 | 0.83 |
| 64 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.50 | 0.27 | 0.28 |
| 64 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.50 | 0.27 | 0.28 |

## gqa_fwd

### tileops

| batch | seq_len | heads | heads_kv | dim | causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 8 | 4 | 64 | False | torch.float16 | 0.01 | 210.22 | 0.31 |
| 4 | 2048 | 64 | 4 | 128 | False | torch.float16 | 0.76 | 723.11 | 0.38 |
| 4 | 2048 | 64 | 4 | 128 | False | torch.bfloat16 | 0.74 | 744.35 | 0.39 |

## gqa_bwd

### tileops

| batch | seq_len | heads | heads_kv | dim | causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 8 | 4 | 64 | False | torch.float16 | 0.05 | 102.20 | 0.10 |
| 4 | 2048 | 64 | 4 | 128 | False | torch.float16 | 3.31 | 414.74 | 0.13 |
| 4 | 2048 | 64 | 4 | 128 | False | torch.bfloat16 | 3.19 | 431.29 | 0.14 |

## gqa_decode

### tileops

| batch | heads | heads_kv | seq_len_kv | dim | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 8 | 8192 | 128 | torch.float16 | 0.03 | 3.99 | 1.00 |
| 4 | 32 | 4 | 4096 | 128 | torch.bfloat16 | 0.04 | 6.74 | 0.84 |
| 8 | 64 | 16 | 8192 | 128 | torch.float16 | 0.17 | 12.94 | 3.24 |

### baseline

| batch | heads | heads_kv | seq_len_kv | dim | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 8 | 8192 | 128 | torch.float16 | 0.43 | 0.31 | 0.08 |
| 4 | 32 | 4 | 4096 | 128 | torch.bfloat16 | 0.81 | 0.33 | 0.04 |
| 8 | 64 | 16 | 8192 | 128 | torch.float16 | 5.65 | 0.38 | 0.10 |

## gqa_decode_paged

### tileops

| batch | heads | heads_kv | seqlen_kv | dim | page_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 16 | 8 | 512 | 128 | 128 | torch.float16 | 0.02 | 0.21 | 0.10 |
| 2 | 8 | 4 | 1024 | 64 | 256 | torch.float16 | 0.02 | 0.20 | 0.05 |
| 1 | 16 | 4 | 2048 | 128 | 512 | torch.float16 | 0.02 | 0.79 | 0.20 |
| 1 | 32 | 16 | 512 | 64 | 128 | torch.float16 | 0.02 | 0.22 | 0.11 |

### baseline

| batch | heads | heads_kv | seqlen_kv | dim | page_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 16 | 8 | 512 | 128 | 128 | torch.float16 | 0.07 | 0.06 | 0.03 |
| 2 | 8 | 4 | 1024 | 64 | 256 | torch.float16 | 0.13 | 0.03 | 0.01 |
| 1 | 16 | 4 | 2048 | 128 | 512 | torch.float16 | 0.07 | 0.25 | 0.06 |
| 1 | 32 | 16 | 512 | 64 | 128 | torch.float16 | 0.07 | 0.06 | 0.03 |

## gqa_sliding_window_fwd

### tileops

| batch | seq | heads | heads_kv | dim | is_causal | wl | wr | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 512 | 8 | 2 | 64 | True | -1 | -1 | torch.float16 | 0.01 | 72.30 | 0.35 |
| 2 | 512 | 8 | 2 | 64 | True | 128 | -1 | torch.float16 | 0.01 | 41.67 | 0.46 |
| 2 | 768 | 8 | 2 | 64 | False | 256 | -1 | torch.float16 | 0.01 | 150.24 | 0.31 |
| 2 | 512 | 8 | 2 | 64 | False | 64 | 64 | torch.bfloat16 | 0.01 | 44.40 | 0.46 |
| 1 | 2048 | 8 | 2 | 64 | True | 512 | -1 | torch.float16 | 0.01 | 153.75 | 0.43 |

## gqa_sliding_window_varlen_fwd

### tileops

| batch | heads | heads_kv | dim | is_causal | wl | wr | dtype | max_seqlen_q | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 8 | 128 | True | -1 | -1 | torch.float16 | 3000 | 0.21 | 343.20 | 0.29 |
| 2 | 32 | 8 | 128 | True | 2000 | -1 | torch.float16 | 3073 | 0.34 | 254.89 | 0.28 |
| 4 | 32 | 8 | 128 | False | 1500 | 1500 | torch.float16 | 3500 | 0.90 | 381.37 | 0.23 |
| 2 | 32 | 8 | 128 | True | -1 | -1 | torch.float16 | 1024 | 0.39 | 411.42 | 0.19 |
| 2 | 32 | 8 | 128 | True | -1 | -1 | torch.float16 | 8193 | 1.86 | 406.27 | 0.15 |

## group_norm

### tileops

| n | c | g | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 128 | 32 | torch.float16 | 0.01 | 0.36 | 0.29 |
| 8 | 128 | 32 | torch.bfloat16 | 0.01 | 0.36 | 0.29 |
| 4 | 256 | 32 | torch.float16 | 0.02 | 0.19 | 0.15 |
| 4 | 128 | 16 | torch.float16 | 0.02 | 0.12 | 0.10 |

### baseline

| n | c | g | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 128 | 32 | torch.float16 | 0.01 | 0.35 | 0.28 |
| 8 | 128 | 32 | torch.bfloat16 | 0.01 | 0.35 | 0.28 |
| 4 | 256 | 32 | torch.float16 | 0.02 | 0.25 | 0.20 |
| 4 | 128 | 16 | torch.float16 | 0.02 | 0.15 | 0.12 |

## grouped_gemm_nt

### tileops

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_nt | 16384 | 4 | 4864 | 4096 | torch.float16 | False | True | 0.95 | 690.53 | 0.48 |

### baseline

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_nt | 16384 | 4 | 4864 | 4096 | torch.float16 | False | True | 1.62 | 403.39 | 0.28 |

## grouped_gemm_nn

### tileops

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_nn | 16384 | 4 | 4864 | 4096 | torch.float16 | False | False | 0.97 | 673.71 | 0.47 |

### baseline

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_nn | 16384 | 4 | 4864 | 4096 | torch.float16 | False | False | 1.14 | 573.83 | 0.40 |

## grouped_gemm_tn

### tileops

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_tn | 16384 | 4 | 4864 | 4096 | torch.float16 | True | False | 9.23 | 70.70 | 0.05 |

### baseline

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_tn | 16384 | 4 | 4864 | 4096 | torch.float16 | True | False | 1.00 | 649.61 | 0.45 |

## grouped_gemm_tt

### tileops

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_tt | 16384 | 4 | 4864 | 4096 | torch.float16 | True | True | 9.06 | 72.06 | 0.05 |

### baseline

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_tt | 16384 | 4 | 4864 | 4096 | torch.float16 | True | True | 1.01 | 646.13 | 0.45 |

## grouped_gemm_complete

### tileops

| batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16384 | 4 | 4864 | 4096 | torch.float16 | True | False | 20.29 | 96.53 | N/A |

### baseline

| batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16384 | 4 | 4864 | 4096 | torch.float16 | True | False | 4.79 | 408.53 | N/A |

## leaky_relu

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| leaky_relu | torch.float16 | 4194304 | 0.01 | 0.69 | 2.75 |
| leaky_relu | torch.bfloat16 | 4194304 | 0.01 | 0.69 | 2.75 |
| leaky_relu | torch.float32 | 4194304 | 0.01 | 0.40 | 3.17 |
| leaky_relu | torch.float16 | 10485760 | 0.01 | 0.82 | 3.28 |
| leaky_relu | torch.bfloat16 | 10485760 | 0.01 | 0.82 | 3.28 |
| leaky_relu | torch.float32 | 10485760 | 0.02 | 0.48 | 3.82 |
| leaky_relu | torch.float16 | 20971520 | 0.02 | 0.92 | 3.66 |
| leaky_relu | torch.bfloat16 | 20971520 | 0.02 | 0.92 | 3.66 |
| leaky_relu | torch.float32 | 20971520 | 0.04 | 0.52 | 4.14 |

### baseline

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| leaky_relu | torch.float16 | 4194304 | 0.01 | 0.66 | 2.62 |
| leaky_relu | torch.bfloat16 | 4194304 | 0.01 | 0.65 | 2.62 |
| leaky_relu | torch.float32 | 4194304 | 0.01 | 0.40 | 3.19 |
| leaky_relu | torch.float16 | 10485760 | 0.01 | 0.84 | 3.36 |
| leaky_relu | torch.bfloat16 | 10485760 | 0.01 | 0.84 | 3.36 |
| leaky_relu | torch.float32 | 10485760 | 0.02 | 0.47 | 3.77 |
| leaky_relu | torch.float16 | 20971520 | 0.02 | 0.95 | 3.80 |
| leaky_relu | torch.bfloat16 | 20971520 | 0.02 | 0.95 | 3.80 |
| leaky_relu | torch.float32 | 20971520 | 0.04 | 0.51 | 4.08 |

## elu

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| elu | torch.float16 | 4194304 | 0.01 | 0.61 | 2.45 |
| elu | torch.bfloat16 | 4194304 | 0.01 | 0.62 | 2.46 |
| elu | torch.float32 | 4194304 | 0.01 | 0.39 | 3.14 |
| elu | torch.float16 | 10485760 | 0.01 | 0.80 | 3.20 |
| elu | torch.bfloat16 | 10485760 | 0.01 | 0.80 | 3.19 |
| elu | torch.float32 | 10485760 | 0.02 | 0.47 | 3.77 |
| elu | torch.float16 | 20971520 | 0.02 | 0.90 | 3.61 |
| elu | torch.bfloat16 | 20971520 | 0.02 | 0.90 | 3.61 |
| elu | torch.float32 | 20971520 | 0.04 | 0.51 | 4.05 |

### baseline

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| elu | torch.float16 | 4194304 | 0.01 | 0.43 | 1.71 |
| elu | torch.bfloat16 | 4194304 | 0.01 | 0.42 | 1.69 |
| elu | torch.float32 | 4194304 | 0.01 | 0.37 | 2.98 |
| elu | torch.float16 | 10485760 | 0.02 | 0.52 | 2.08 |
| elu | torch.bfloat16 | 10485760 | 0.02 | 0.51 | 2.05 |
| elu | torch.float32 | 10485760 | 0.02 | 0.46 | 3.67 |
| elu | torch.float16 | 20971520 | 0.04 | 0.57 | 2.29 |
| elu | torch.bfloat16 | 20971520 | 0.04 | 0.56 | 2.26 |
| elu | torch.float32 | 20971520 | 0.04 | 0.50 | 4.02 |

## hardtanh

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| hardtanh | torch.float16 | 4194304 | 0.01 | 0.68 | 2.72 |
| hardtanh | torch.bfloat16 | 4194304 | 0.01 | 0.68 | 2.71 |
| hardtanh | torch.float32 | 4194304 | 0.01 | 0.40 | 3.19 |
| hardtanh | torch.float16 | 10485760 | 0.01 | 0.81 | 3.25 |
| hardtanh | torch.bfloat16 | 10485760 | 0.01 | 0.81 | 3.25 |
| hardtanh | torch.float32 | 10485760 | 0.02 | 0.48 | 3.83 |
| hardtanh | torch.float16 | 20971520 | 0.02 | 0.91 | 3.64 |
| hardtanh | torch.bfloat16 | 20971520 | 0.02 | 0.90 | 3.60 |
| hardtanh | torch.float32 | 20971520 | 0.04 | 0.52 | 4.15 |

### baseline

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| hardtanh | torch.float16 | 4194304 | 0.01 | 0.62 | 2.50 |
| hardtanh | torch.bfloat16 | 4194304 | 0.01 | 0.64 | 2.57 |
| hardtanh | torch.float32 | 4194304 | 0.01 | 0.40 | 3.16 |
| hardtanh | torch.float16 | 10485760 | 0.01 | 0.79 | 3.17 |
| hardtanh | torch.bfloat16 | 10485760 | 0.01 | 0.83 | 3.32 |
| hardtanh | torch.float32 | 10485760 | 0.02 | 0.47 | 3.76 |
| hardtanh | torch.float16 | 20971520 | 0.02 | 0.88 | 3.51 |
| hardtanh | torch.bfloat16 | 20971520 | 0.02 | 0.94 | 3.77 |
| hardtanh | torch.float32 | 20971520 | 0.04 | 0.51 | 4.08 |

## softplus

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| softplus | torch.float16 | 4194304 | 0.01 | 0.42 | 1.66 |
| softplus | torch.bfloat16 | 4194304 | 0.01 | 0.41 | 1.63 |
| softplus | torch.float32 | 4194304 | 0.01 | 0.38 | 3.03 |
| softplus | torch.float16 | 10485760 | 0.02 | 0.53 | 2.13 |
| softplus | torch.bfloat16 | 10485760 | 0.02 | 0.52 | 2.08 |
| softplus | torch.float32 | 10485760 | 0.02 | 0.43 | 3.47 |
| softplus | torch.float16 | 20971520 | 0.04 | 0.59 | 2.36 |
| softplus | torch.bfloat16 | 20971520 | 0.04 | 0.57 | 2.29 |
| softplus | torch.float32 | 20971520 | 0.05 | 0.46 | 3.71 |

### baseline

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| softplus | torch.float16 | 4194304 | 0.01 | 0.36 | 1.44 |
| softplus | torch.bfloat16 | 4194304 | 0.01 | 0.36 | 1.43 |
| softplus | torch.float32 | 4194304 | 0.01 | 0.34 | 2.76 |
| softplus | torch.float16 | 10485760 | 0.02 | 0.43 | 1.72 |
| softplus | torch.bfloat16 | 10485760 | 0.02 | 0.43 | 1.71 |
| softplus | torch.float32 | 10485760 | 0.03 | 0.42 | 3.34 |
| softplus | torch.float16 | 20971520 | 0.04 | 0.47 | 1.86 |
| softplus | torch.bfloat16 | 20971520 | 0.05 | 0.46 | 1.84 |
| softplus | torch.float32 | 20971520 | 0.05 | 0.45 | 3.63 |

## clamp

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| clamp | torch.float16 | 4194304 | 0.01 | 0.68 | 2.72 |
| clamp | torch.bfloat16 | 4194304 | 0.01 | 0.68 | 2.70 |
| clamp | torch.float32 | 4194304 | 0.01 | 0.40 | 3.18 |
| clamp | torch.float16 | 10485760 | 0.01 | 0.82 | 3.26 |
| clamp | torch.bfloat16 | 10485760 | 0.01 | 0.81 | 3.25 |
| clamp | torch.float32 | 10485760 | 0.02 | 0.48 | 3.83 |
| clamp | torch.float16 | 20971520 | 0.02 | 0.90 | 3.62 |
| clamp | torch.bfloat16 | 20971520 | 0.02 | 0.91 | 3.63 |
| clamp | torch.float32 | 20971520 | 0.04 | 0.51 | 4.10 |

### baseline

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| clamp | torch.float16 | 4194304 | 0.01 | 0.63 | 2.50 |
| clamp | torch.bfloat16 | 4194304 | 0.01 | 0.64 | 2.58 |
| clamp | torch.float32 | 4194304 | 0.01 | 0.40 | 3.17 |
| clamp | torch.float16 | 10485760 | 0.01 | 0.79 | 3.15 |
| clamp | torch.bfloat16 | 10485760 | 0.01 | 0.83 | 3.33 |
| clamp | torch.float32 | 10485760 | 0.02 | 0.47 | 3.76 |
| clamp | torch.float16 | 20971520 | 0.02 | 0.88 | 3.51 |
| clamp | torch.bfloat16 | 20971520 | 0.02 | 0.94 | 3.78 |
| clamp | torch.float32 | 20971520 | 0.04 | 0.51 | 4.08 |

## nan_to_num

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| nan_to_num | torch.float16 | 4194304 | 0.01 | 0.66 | 2.63 |
| nan_to_num | torch.bfloat16 | 4194304 | 0.01 | 0.66 | 2.62 |
| nan_to_num | torch.float32 | 4194304 | 0.01 | 0.40 | 3.17 |
| nan_to_num | torch.float16 | 10485760 | 0.01 | 0.80 | 3.19 |
| nan_to_num | torch.bfloat16 | 10485760 | 0.01 | 0.80 | 3.19 |
| nan_to_num | torch.float32 | 10485760 | 0.02 | 0.48 | 3.83 |
| nan_to_num | torch.float16 | 20971520 | 0.02 | 0.90 | 3.61 |
| nan_to_num | torch.bfloat16 | 20971520 | 0.02 | 0.90 | 3.60 |
| nan_to_num | torch.float32 | 20971520 | 0.04 | 0.52 | 4.15 |

### baseline

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| nan_to_num | torch.float16 | 4194304 | 0.01 | 0.64 | 2.55 |
| nan_to_num | torch.bfloat16 | 4194304 | 0.01 | 0.64 | 2.54 |
| nan_to_num | torch.float32 | 4194304 | 0.01 | 0.40 | 3.19 |
| nan_to_num | torch.float16 | 10485760 | 0.01 | 0.82 | 3.30 |
| nan_to_num | torch.bfloat16 | 10485760 | 0.01 | 0.82 | 3.30 |
| nan_to_num | torch.float32 | 10485760 | 0.02 | 0.47 | 3.77 |
| nan_to_num | torch.float16 | 20971520 | 0.02 | 0.94 | 3.74 |
| nan_to_num | torch.bfloat16 | 20971520 | 0.02 | 0.94 | 3.74 |
| nan_to_num | torch.float32 | 20971520 | 0.04 | 0.51 | 4.09 |

## prelu

### tileops

| num_channels | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 128 | torch.float16 | 131072 | 0.00 | 0.06 | 0.25 |
| 128 | torch.bfloat16 | 131072 | 0.00 | 0.06 | 0.25 |
| 128 | torch.float32 | 131072 | 0.00 | 0.06 | 0.46 |
| 4096 | torch.float16 | 4194304 | 0.01 | 0.68 | 2.73 |
| 4096 | torch.bfloat16 | 4194304 | 0.01 | 0.68 | 2.73 |
| 4096 | torch.float32 | 4194304 | 0.01 | 0.40 | 3.20 |
| 10240 | torch.float16 | 10485760 | 0.01 | 0.82 | 3.27 |
| 10240 | torch.bfloat16 | 10485760 | 0.01 | 0.82 | 3.27 |
| 10240 | torch.float32 | 10485760 | 0.02 | 0.48 | 3.81 |
| 20480 | torch.float16 | 20971520 | 0.02 | 0.91 | 3.65 |
| 20480 | torch.bfloat16 | 20971520 | 0.02 | 0.91 | 3.65 |
| 20480 | torch.float32 | 20971520 | 0.04 | 0.51 | 4.09 |

### baseline

| num_channels | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 128 | torch.float16 | 131072 | 0.00 | 0.03 | 0.12 |
| 128 | torch.bfloat16 | 131072 | 0.00 | 0.03 | 0.11 |
| 128 | torch.float32 | 131072 | 0.00 | 0.04 | 0.29 |
| 4096 | torch.float16 | 4194304 | 0.01 | 0.29 | 1.15 |
| 4096 | torch.bfloat16 | 4194304 | 0.01 | 0.28 | 1.14 |
| 4096 | torch.float32 | 4194304 | 0.02 | 0.25 | 2.03 |
| 10240 | torch.float16 | 10485760 | 0.03 | 0.34 | 1.35 |
| 10240 | torch.bfloat16 | 10485760 | 0.03 | 0.33 | 1.34 |
| 10240 | torch.float32 | 10485760 | 0.04 | 0.29 | 2.33 |
| 20480 | torch.float16 | 20971520 | 0.06 | 0.37 | 1.47 |
| 20480 | torch.bfloat16 | 20971520 | 0.06 | 0.36 | 1.45 |
| 20480 | torch.float32 | 20971520 | 0.07 | 0.31 | 2.48 |

## where

### tileops

| dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| torch.float16 | 4194304 | 0.01 | 0.41 | 2.86 |
| torch.bfloat16 | 4194304 | 0.01 | 0.41 | 2.86 |
| torch.float32 | 4194304 | 0.02 | 0.26 | 3.32 |
| torch.float16 | 10485760 | 0.02 | 0.50 | 3.50 |
| torch.bfloat16 | 10485760 | 0.02 | 0.50 | 3.50 |
| torch.float32 | 10485760 | 0.04 | 0.30 | 3.88 |
| torch.float16 | 20971520 | 0.04 | 0.56 | 3.92 |
| torch.bfloat16 | 20971520 | 0.04 | 0.56 | 3.93 |
| torch.float32 | 20971520 | 0.06 | 0.32 | 4.22 |

### baseline

| dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| torch.float16 | 4194304 | 0.01 | 0.41 | 2.89 |
| torch.bfloat16 | 4194304 | 0.01 | 0.41 | 2.89 |
| torch.float32 | 4194304 | 0.02 | 0.26 | 3.36 |
| torch.float16 | 10485760 | 0.02 | 0.50 | 3.53 |
| torch.bfloat16 | 10485760 | 0.02 | 0.50 | 3.53 |
| torch.float32 | 10485760 | 0.03 | 0.30 | 3.93 |
| torch.float16 | 20971520 | 0.04 | 0.56 | 3.95 |
| torch.bfloat16 | 20971520 | 0.04 | 0.56 | 3.95 |
| torch.float32 | 20971520 | 0.06 | 0.33 | 4.24 |

## masked_fill

### tileops

| dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| torch.float16 | 4194304 | 0.01 | 0.55 | 2.75 |
| torch.bfloat16 | 4194304 | 0.01 | 0.55 | 2.76 |
| torch.float32 | 4194304 | 0.01 | 0.35 | 3.14 |
| torch.float16 | 10485760 | 0.02 | 0.68 | 3.38 |
| torch.bfloat16 | 10485760 | 0.02 | 0.67 | 3.36 |
| torch.float32 | 10485760 | 0.02 | 0.42 | 3.81 |
| torch.float16 | 20971520 | 0.03 | 0.77 | 3.84 |
| torch.bfloat16 | 20971520 | 0.03 | 0.77 | 3.84 |
| torch.float32 | 20971520 | 0.05 | 0.46 | 4.14 |

### baseline

| dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| torch.float16 | 4194304 | 0.01 | 0.36 | 1.78 |
| torch.bfloat16 | 4194304 | 0.01 | 0.36 | 1.78 |
| torch.float32 | 4194304 | 0.02 | 0.23 | 2.03 |
| torch.float16 | 10485760 | 0.02 | 0.45 | 2.25 |
| torch.bfloat16 | 10485760 | 0.02 | 0.45 | 2.25 |
| torch.float32 | 10485760 | 0.05 | 0.23 | 2.08 |
| torch.float16 | 20971520 | 0.05 | 0.44 | 2.18 |
| torch.bfloat16 | 20971520 | 0.05 | 0.44 | 2.18 |
| torch.float32 | 20971520 | 0.09 | 0.24 | 2.18 |

## alibi

### tileops

| op_name | seq_len | dim | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| alibi | 512 | 64 | torch.float16 | 0.03 | 0.53 | 1.05 |
| alibi | 512 | 64 | torch.bfloat16 | 0.03 | 0.53 | 1.05 |
| alibi | 512 | 64 | torch.float32 | 0.02 | 0.90 | 3.59 |
| alibi | 2048 | 64 | torch.float16 | 0.26 | 1.03 | 2.06 |
| alibi | 2048 | 64 | torch.bfloat16 | 0.27 | 1.00 | 2.01 |
| alibi | 2048 | 64 | torch.float32 | 0.25 | 1.07 | 4.28 |
| alibi | 4096 | 128 | torch.float16 | 0.80 | 2.67 | 5.34 |
| alibi | 4096 | 128 | torch.bfloat16 | 0.79 | 2.73 | 5.45 |
| alibi | 4096 | 128 | torch.float32 | 1.03 | 2.09 | 8.34 |

### baseline

| op_name | seq_len | dim | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| alibi | 512 | 64 | torch.float16 | 0.08 | 0.20 | 0.41 |
| alibi | 512 | 64 | torch.bfloat16 | 0.08 | 0.21 | 0.41 |
| alibi | 512 | 64 | torch.float32 | 0.06 | 0.30 | 1.22 |
| alibi | 2048 | 64 | torch.float16 | 1.00 | 0.27 | 0.54 |
| alibi | 2048 | 64 | torch.bfloat16 | 1.00 | 0.27 | 0.54 |
| alibi | 2048 | 64 | torch.float32 | 0.65 | 0.41 | 1.64 |
| alibi | 4096 | 128 | torch.float16 | 7.72 | 0.28 | 0.56 |
| alibi | 4096 | 128 | torch.bfloat16 | 7.26 | 0.30 | 0.59 |
| alibi | 4096 | 128 | torch.float32 | 5.16 | 0.42 | 1.67 |

## sinusoidal

### tileops

| op_name | seq_len | dim | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| sinusoidal | 512 | 256 | torch.float16 | 0.00 | 0.04 | 0.09 |
| sinusoidal | 512 | 256 | torch.bfloat16 | 0.00 | 0.04 | 0.09 |
| sinusoidal | 512 | 256 | torch.float32 | 0.00 | 0.05 | 0.21 |
| sinusoidal | 2048 | 300 | torch.float16 | 0.00 | 0.15 | 0.29 |
| sinusoidal | 2048 | 300 | torch.bfloat16 | 0.00 | 0.15 | 0.29 |
| sinusoidal | 2048 | 300 | torch.float32 | 0.00 | 0.13 | 0.52 |
| sinusoidal | 4096 | 512 | torch.float16 | 0.01 | 0.31 | 0.61 |
| sinusoidal | 4096 | 512 | torch.bfloat16 | 0.01 | 0.31 | 0.61 |
| sinusoidal | 4096 | 512 | torch.float32 | 0.01 | 0.19 | 0.75 |

### baseline

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
| sinusoidal | 4096 | 512 | torch.float32 | 0.03 | 0.07 | 0.30 |

## leaky_relu_fp8

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| leaky_relu | torch.float8_e4m3fn | 4194304 | 0.01 | 0.49 | 0.98 |
| leaky_relu | torch.float8_e5m2 | 4194304 | 0.03 | 0.16 | 0.32 |
| leaky_relu | torch.float8_e4m3fn | 10485760 | 0.02 | 0.57 | 1.14 |
| leaky_relu | torch.float8_e5m2 | 10485760 | 0.05 | 0.20 | 0.41 |
| leaky_relu | torch.float8_e4m3fn | 20971520 | 0.03 | 0.62 | 1.24 |
| leaky_relu | torch.float8_e5m2 | 20971520 | 0.10 | 0.22 | 0.44 |

### baseline

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| leaky_relu | torch.float8_e4m3fn | 4194304 | 0.03 | 0.16 | 0.33 |
| leaky_relu | torch.float8_e5m2 | 4194304 | 0.02 | 0.17 | 0.34 |
| leaky_relu | torch.float8_e4m3fn | 10485760 | 0.05 | 0.19 | 0.38 |
| leaky_relu | torch.float8_e5m2 | 10485760 | 0.05 | 0.20 | 0.40 |
| leaky_relu | torch.float8_e4m3fn | 20971520 | 0.11 | 0.19 | 0.38 |
| leaky_relu | torch.float8_e5m2 | 20971520 | 0.11 | 0.20 | 0.39 |

## elu_fp8

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| elu | torch.float8_e4m3fn | 4194304 | 0.01 | 0.66 | 1.32 |
| elu | torch.float8_e5m2 | 4194304 | 0.02 | 0.26 | 0.52 |
| elu | torch.float8_e4m3fn | 10485760 | 0.01 | 0.86 | 1.71 |
| elu | torch.float8_e5m2 | 10485760 | 0.03 | 0.31 | 0.62 |
| elu | torch.float8_e4m3fn | 20971520 | 0.02 | 0.96 | 1.92 |
| elu | torch.float8_e5m2 | 20971520 | 0.06 | 0.32 | 0.65 |

### baseline

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| elu | torch.float8_e4m3fn | 4194304 | 0.03 | 0.14 | 0.28 |
| elu | torch.float8_e5m2 | 4194304 | 0.03 | 0.14 | 0.29 |
| elu | torch.float8_e4m3fn | 10485760 | 0.06 | 0.16 | 0.33 |
| elu | torch.float8_e5m2 | 10485760 | 0.06 | 0.17 | 0.34 |
| elu | torch.float8_e4m3fn | 20971520 | 0.12 | 0.17 | 0.34 |
| elu | torch.float8_e5m2 | 20971520 | 0.12 | 0.17 | 0.34 |

## clamp_fp8

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| clamp | torch.float8_e4m3fn | 4194304 | 0.00 | 0.99 | 1.98 |
| clamp | torch.float8_e5m2 | 4194304 | 0.01 | 0.40 | 0.79 |
| clamp | torch.float8_e4m3fn | 10485760 | 0.01 | 1.33 | 2.65 |
| clamp | torch.float8_e5m2 | 10485760 | 0.02 | 0.50 | 0.99 |
| clamp | torch.float8_e4m3fn | 20971520 | 0.01 | 1.57 | 3.14 |
| clamp | torch.float8_e5m2 | 20971520 | 0.04 | 0.51 | 1.02 |

### baseline

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| clamp | torch.float8_e4m3fn | 4194304 | 0.03 | 0.16 | 0.32 |
| clamp | torch.float8_e5m2 | 4194304 | 0.03 | 0.17 | 0.33 |
| clamp | torch.float8_e4m3fn | 10485760 | 0.06 | 0.19 | 0.38 |
| clamp | torch.float8_e5m2 | 10485760 | 0.05 | 0.19 | 0.39 |
| clamp | torch.float8_e4m3fn | 20971520 | 0.11 | 0.19 | 0.37 |
| clamp | torch.float8_e5m2 | 20971520 | 0.11 | 0.19 | 0.38 |

## where_fp8

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| where | torch.float8_e4m3fn | 4194304 | 0.01 | 0.51 | 2.04 |
| where | torch.float8_e5m2 | 4194304 | 0.01 | 0.51 | 2.04 |
| where | torch.float8_e4m3fn | 10485760 | 0.02 | 0.57 | 2.29 |
| where | torch.float8_e5m2 | 10485760 | 0.02 | 0.57 | 2.29 |
| where | torch.float8_e4m3fn | 20971520 | 0.03 | 0.68 | 2.71 |
| where | torch.float8_e5m2 | 20971520 | 0.03 | 0.68 | 2.71 |

### baseline

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| where | torch.float8_e4m3fn | 4194304 | 0.01 | 0.59 | 2.37 |
| where | torch.float8_e5m2 | 4194304 | 0.01 | 0.59 | 2.37 |
| where | torch.float8_e4m3fn | 10485760 | 0.01 | 0.75 | 3.00 |
| where | torch.float8_e5m2 | 10485760 | 0.01 | 0.75 | 3.00 |
| where | torch.float8_e4m3fn | 20971520 | 0.02 | 0.85 | 3.40 |
| where | torch.float8_e5m2 | 20971520 | 0.02 | 0.85 | 3.41 |

## masked_fill_fp8

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| masked_fill | torch.float8_e4m3fn | 4194304 | 0.01 | 0.58 | 1.74 |
| masked_fill | torch.float8_e5m2 | 4194304 | 0.01 | 0.32 | 0.96 |
| masked_fill | torch.float8_e4m3fn | 10485760 | 0.01 | 0.75 | 2.26 |
| masked_fill | torch.float8_e5m2 | 10485760 | 0.03 | 0.40 | 1.21 |
| masked_fill | torch.float8_e4m3fn | 20971520 | 0.03 | 0.83 | 2.49 |
| masked_fill | torch.float8_e5m2 | 20971520 | 0.05 | 0.42 | 1.25 |

### baseline

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| masked_fill | torch.float8_e4m3fn | 4194304 | 0.03 | 0.14 | 0.41 |
| masked_fill | torch.float8_e5m2 | 4194304 | 0.03 | 0.14 | 0.42 |
| masked_fill | torch.float8_e4m3fn | 10485760 | 0.07 | 0.16 | 0.48 |
| masked_fill | torch.float8_e5m2 | 10485760 | 0.06 | 0.16 | 0.49 |
| masked_fill | torch.float8_e4m3fn | 20971520 | 0.14 | 0.15 | 0.46 |
| masked_fill | torch.float8_e5m2 | 20971520 | 0.13 | 0.16 | 0.47 |

## instance_norm

### tileops

| n | c | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 8 | 128 | torch.float16 | 0.02 | 0.35 | 0.28 |
| 8 | 128 | torch.bfloat16 | 0.01 | 0.36 | 0.28 |
| 4 | 256 | torch.float16 | 0.02 | 0.22 | 0.18 |
| 4 | 64 | torch.float16 | 0.01 | 0.08 | 0.07 |

### baseline

| n | c | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 8 | 128 | torch.float16 | 0.02 | 0.25 | 0.20 |
| 8 | 128 | torch.bfloat16 | 0.02 | 0.25 | 0.20 |
| 4 | 256 | torch.float16 | 0.02 | 0.19 | 0.16 |
| 4 | 64 | torch.float16 | 0.01 | 0.09 | 0.07 |

## layer_norm

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.01 | 1.93 | 1.55 |
| 4096 | 4096 | torch.bfloat16 | 0.04 | 2.29 | 1.83 |
| 2048 | 5120 | torch.float16 | 0.03 | 2.07 | 1.66 |
| 1025 | 4096 | torch.float16 | 0.01 | 1.83 | 1.46 |

### baseline

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.01 | 1.65 | 1.32 |
| 4096 | 4096 | torch.bfloat16 | 0.03 | 2.59 | 2.07 |
| 2048 | 5120 | torch.float16 | 0.02 | 2.36 | 1.89 |
| 1025 | 4096 | torch.float16 | 0.01 | 1.65 | 1.32 |

## logical_reduce

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | any | 0.01 | 0.40 | 0.81 |
| 1024 | 4096 | torch.bfloat16 | any | 0.01 | 0.40 | 0.80 |
| 1024 | 4096 | torch.float32 | any | 0.01 | 0.29 | 1.15 |
| 1024 | 4096 | torch.int32 | any | 0.03 | 0.16 | 0.62 |
| 4096 | 4096 | torch.float16 | any | 0.03 | 0.63 | 1.25 |
| 1024 | 4096 | torch.float16 | all | 0.01 | 0.40 | 0.81 |
| 1024 | 4096 | torch.bfloat16 | all | 0.01 | 0.40 | 0.80 |
| 1024 | 4096 | torch.float32 | all | 0.01 | 0.29 | 1.15 |
| 1024 | 4096 | torch.int32 | all | 0.03 | 0.16 | 0.62 |
| 4096 | 4096 | torch.float16 | all | 0.03 | 0.63 | 1.25 |
| 1024 | 4096 | torch.float16 | count_nonzero | 0.01 | 0.53 | 1.06 |
| 1024 | 4096 | torch.bfloat16 | count_nonzero | 0.01 | 0.53 | 1.07 |
| 1024 | 4096 | torch.float32 | count_nonzero | 0.01 | 0.35 | 1.38 |
| 1024 | 4096 | torch.int32 | count_nonzero | 0.02 | 0.17 | 0.69 |
| 4096 | 4096 | torch.float16 | count_nonzero | 0.02 | 0.70 | 1.40 |

### baseline

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | any | 0.03 | 0.16 | 0.32 |
| 1024 | 4096 | torch.bfloat16 | any | 0.03 | 0.16 | 0.32 |
| 1024 | 4096 | torch.float32 | any | 0.03 | 0.16 | 0.63 |
| 1024 | 4096 | torch.int32 | any | 0.03 | 0.16 | 0.63 |
| 4096 | 4096 | torch.float16 | any | 0.07 | 0.25 | 0.51 |
| 1024 | 4096 | torch.float16 | all | 0.03 | 0.16 | 0.33 |
| 1024 | 4096 | torch.bfloat16 | all | 0.03 | 0.16 | 0.33 |
| 1024 | 4096 | torch.float32 | all | 0.03 | 0.16 | 0.63 |
| 1024 | 4096 | torch.int32 | all | 0.03 | 0.16 | 0.64 |
| 4096 | 4096 | torch.float16 | all | 0.07 | 0.26 | 0.51 |
| 1024 | 4096 | torch.float16 | count_nonzero | 0.03 | 0.12 | 0.25 |
| 1024 | 4096 | torch.bfloat16 | count_nonzero | 0.03 | 0.12 | 0.25 |
| 1024 | 4096 | torch.float32 | count_nonzero | 0.04 | 0.11 | 0.46 |
| 1024 | 4096 | torch.int32 | count_nonzero | 0.04 | 0.11 | 0.45 |
| 4096 | 4096 | torch.float16 | count_nonzero | 0.11 | 0.16 | 0.31 |

## mean_pooling

### tileops

| batch_size | seq_len | heads | dim | chunk_size | dtype | accum_dtype | chunks_per_bacth | seq_num | use_offsets | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8192 | 64 | 128 | 64 | torch.float16 | torch.float32 | 128 | 1 | 0 | 0.13 | 0.52 | 1.06 |
| 2 | 2048 | 64 | 128 | 64 | torch.float16 | torch.float32 | 32 | 2 | 0 | 0.07 | 0.48 | 0.97 |
| 1 | 8192 | 64 | 128 | 64 | torch.float16 | torch.float32 | 128 | 4 | 1 | 0.19 | 0.36 | 0.73 |
| 1 | 1000 | 64 | 128 | 32 | torch.float16 | torch.float32 | 34 | 4 | 1 | 0.02 | 0.40 | 0.75 |

### baseline

| batch_size | seq_len | heads | dim | chunk_size | dtype | accum_dtype | chunks_per_bacth | seq_num | use_offsets | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8192 | 64 | 128 | 64 | torch.float16 | torch.float32 | 128 | 1 | 0 | 0.77 | 0.09 | 0.18 |
| 2 | 2048 | 64 | 128 | 64 | torch.float16 | torch.float32 | 32 | 2 | 0 | 0.27 | 0.12 | 0.25 |
| 1 | 8192 | 64 | 128 | 64 | torch.float16 | torch.float32 | 128 | 4 | 1 | 0.80 | 0.08 | 0.17 |
| 1 | 1000 | 64 | 128 | 32 | torch.float16 | torch.float32 | 34 | 4 | 1 | 0.27 | 0.03 | 0.06 |

## mha_fwd

### tileops

| batch | seq_len | heads | dim | causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 8 | 64 | False | torch.float16 | 0.01 | 207.54 | 0.41 |
| 16 | 2048 | 16 | 128 | False | torch.float16 | 0.75 | 737.45 | 0.72 |
| 4 | 4096 | 16 | 128 | False | torch.bfloat16 | 0.70 | 783.15 | 0.38 |

## mha_bwd

### tileops

| batch | seq_len | heads | dim | causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 8 | 64 | False | torch.float16 | 0.04 | 125.60 | 0.17 |
| 16 | 2048 | 16 | 128 | False | torch.float16 | 2.25 | 610.30 | 0.42 |
| 4 | 4096 | 16 | 128 | False | torch.bfloat16 | 2.17 | 633.80 | 0.22 |

## mha_decode

### tileops

| b | h | s_q | s_kv | d | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 128 | 8192 | 128 | torch.float16 | 0.06 | 292.97 | 2.32 |
| 1 | 32 | 128 | 8192 | 128 | torch.bfloat16 | 0.06 | 294.16 | 2.33 |
| 1 | 32 | 128 | 5 | 128 | torch.float16 | 0.01 | 1.01 | 0.21 |

### baseline

| b | h | s_q | s_kv | d | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 128 | 8192 | 128 | torch.float16 | 0.07 | 259.68 | 2.06 |
| 1 | 32 | 128 | 8192 | 128 | torch.bfloat16 | 0.07 | 262.36 | 2.08 |
| 1 | 32 | 128 | 5 | 128 | torch.float16 | 0.01 | 1.47 | 0.31 |

## mha_decode_paged

### tileops

| batch | heads | seqlen_q | seqlen_kv | dim | page_size | is_causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 16 | 1 | 512 | 128 | 128 | False | torch.float16 | 0.02 | 0.18 | 0.18 |
| 2 | 8 | 1 | 1024 | 64 | 256 | False | torch.float16 | 0.02 | 0.18 | 0.09 |
| 1 | 8 | 1 | 1024 | 64 | 256 | False | torch.float16 | 0.02 | 0.09 | 0.09 |
| 1 | 8 | 1 | 512 | 64 | 256 | False | torch.float16 | 0.02 | 0.05 | 0.05 |

### baseline

| batch | heads | seqlen_q | seqlen_kv | dim | page_size | is_causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 16 | 1 | 512 | 128 | 128 | False | torch.float16 | 0.04 | 0.10 | 0.10 |
| 2 | 8 | 1 | 1024 | 64 | 256 | False | torch.float16 | 0.08 | 0.05 | 0.03 |
| 1 | 8 | 1 | 1024 | 64 | 256 | False | torch.float16 | 0.04 | 0.05 | 0.05 |
| 1 | 8 | 1 | 512 | 64 | 256 | False | torch.float16 | 0.03 | 0.03 | 0.03 |

## mhc_post

### tileops

| batch | n_expand | c_x | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | 1280 | torch.bfloat16 | 0.00 | 28.28 | 0.01 |
| 2 | 4 | 1920 | torch.bfloat16 | 0.00 | 123.40 | 0.01 |
| 4 | 4 | 2560 | torch.bfloat16 | 0.00 | 417.43 | 0.01 |

### baseline

| batch | n_expand | c_x | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | 1280 | torch.bfloat16 | 0.01 | 3.95 | 0.00 |
| 2 | 4 | 1920 | torch.bfloat16 | 0.01 | 16.68 | 0.00 |
| 4 | 4 | 2560 | torch.bfloat16 | 0.01 | 56.30 | 0.00 |

## mhc_pre

### tileops

| batch | n_expand | c_x | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | 1280 | torch.bfloat16 | 0.04 | 30.63 | 0.00 |
| 2 | 4 | 1920 | torch.bfloat16 | 0.05 | 103.98 | 0.00 |
| 4 | 4 | 2560 | torch.bfloat16 | 0.07 | 292.69 | 0.00 |

### baseline

| batch | n_expand | c_x | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | 1280 | torch.bfloat16 | 0.27 | 4.59 | 0.00 |
| 2 | 4 | 1920 | torch.bfloat16 | 0.30 | 18.78 | 0.00 |
| 4 | 4 | 2560 | torch.bfloat16 | 0.34 | 60.04 | 0.00 |

## moe_permute

### tileops

| total_tokens | top_k | num_experts | hidden_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 2 | 8 | 4096 | torch.bfloat16 | 0.01 | N/A | 1.52 |
| 2048 | 2 | 8 | 4096 | torch.bfloat16 | 0.02 | N/A | 2.36 |
| 4096 | 2 | 8 | 4096 | torch.bfloat16 | 0.04 | N/A | 2.35 |
| 512 | 8 | 256 | 7168 | torch.bfloat16 | 0.04 | N/A | 1.75 |
| 2048 | 8 | 256 | 7168 | torch.bfloat16 | 0.16 | N/A | 1.70 |
| 4096 | 8 | 256 | 7168 | torch.bfloat16 | 0.32 | N/A | 1.65 |
| 512 | 8 | 128 | 2048 | torch.bfloat16 | 0.02 | N/A | 0.98 |
| 2048 | 8 | 128 | 2048 | torch.bfloat16 | 0.06 | N/A | 1.16 |
| 4096 | 8 | 128 | 2048 | torch.bfloat16 | 0.13 | N/A | 1.18 |

### pytorch-ref

| total_tokens | top_k | num_experts | hidden_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 2 | 8 | 4096 | torch.bfloat16 | 0.01 | N/A | 0.88 |
| 2048 | 2 | 8 | 4096 | torch.bfloat16 | 0.03 | N/A | 1.46 |
| 4096 | 2 | 8 | 4096 | torch.bfloat16 | 0.06 | N/A | 1.58 |
| 512 | 8 | 256 | 7168 | torch.bfloat16 | 0.04 | N/A | 1.78 |
| 2048 | 8 | 256 | 7168 | torch.bfloat16 | 0.12 | N/A | 2.18 |
| 4096 | 8 | 256 | 7168 | torch.bfloat16 | 0.18 | N/A | 2.88 |
| 512 | 8 | 128 | 2048 | torch.bfloat16 | 0.03 | N/A | 0.75 |
| 2048 | 8 | 128 | 2048 | torch.bfloat16 | 0.07 | N/A | 1.01 |
| 4096 | 8 | 128 | 2048 | torch.bfloat16 | 0.07 | N/A | 2.25 |

## permute_align

### tileops

| total_tokens | top_k | num_experts | block_size | numel | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 2 | 8 | 16 | 1024 | 0.01 | N/A | 0.00 |
| 2048 | 2 | 8 | 16 | 4096 | 0.01 | N/A | 0.00 |
| 4096 | 2 | 8 | 16 | 8192 | 0.01 | N/A | 0.01 |
| 512 | 6 | 64 | 64 | 3072 | 0.01 | N/A | 0.01 |
| 2048 | 6 | 64 | 64 | 12288 | 0.01 | N/A | 0.01 |
| 4096 | 6 | 64 | 128 | 24576 | 0.02 | N/A | 0.01 |
| 8192 | 6 | 64 | 128 | 49152 | 0.03 | N/A | 0.01 |
| 2048 | 6 | 256 | 128 | 12288 | 0.02 | N/A | 0.02 |
| 8192 | 6 | 256 | 128 | 49152 | 0.03 | N/A | 0.02 |

### triton

| total_tokens | top_k | num_experts | block_size | numel | max_num_blocks | max_padded | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 2 | 8 | 16 | 1024 | 73 | 1159 | 0.03 | N/A | 0.00 |
| 2048 | 2 | 8 | 16 | 4096 | 265 | 4231 | 0.10 | N/A | 0.00 |
| 4096 | 2 | 8 | 16 | 8192 | 521 | 8327 | 0.20 | N/A | 0.00 |
| 512 | 6 | 64 | 64 | 3072 | 112 | 7167 | 0.02 | N/A | 0.00 |
| 2048 | 6 | 64 | 64 | 12288 | 256 | 16383 | 0.05 | N/A | 0.00 |
| 4096 | 6 | 64 | 128 | 24576 | 257 | 32831 | 0.09 | N/A | 0.00 |
| 8192 | 6 | 64 | 128 | 49152 | 449 | 57407 | 0.16 | N/A | 0.00 |
| 2048 | 6 | 256 | 128 | 12288 | 351 | 44927 | 0.05 | N/A | 0.00 |
| 8192 | 6 | 256 | 128 | 49152 | 639 | 81791 | 0.08 | N/A | 0.01 |

## reduce

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | sum | 0.01 | 0.53 | 1.06 |
| 1024 | 4096 | torch.bfloat16 | sum | 0.01 | 0.54 | 1.07 |
| 4096 | 4096 | torch.float16 | sum | 0.02 | 0.70 | 1.40 |
| 1024 | 4096 | torch.float16 | mean | 0.01 | 0.53 | 1.06 |
| 1024 | 4096 | torch.float16 | amax | 0.01 | 0.54 | 1.07 |
| 1024 | 4096 | torch.float16 | amin | 0.01 | 0.54 | 1.07 |
| 1024 | 4096 | torch.float16 | prod | 0.02 | 0.24 | 0.48 |
| 1024 | 4096 | torch.float16 | std | 0.01 | 1.39 | 0.93 |
| 1024 | 4096 | torch.float16 | var | 0.01 | 1.40 | 0.93 |
| 1024 | 4096 | torch.float16 | var_mean | 0.01 | 1.52 | 1.01 |

### baseline

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | sum | 0.03 | 0.16 | 0.32 |
| 1024 | 4096 | torch.bfloat16 | sum | 0.03 | 0.16 | 0.32 |
| 4096 | 4096 | torch.float16 | sum | 0.08 | 0.21 | 0.43 |
| 1024 | 4096 | torch.float16 | mean | 0.03 | 0.16 | 0.32 |
| 1024 | 4096 | torch.float16 | amax | 0.03 | 0.16 | 0.32 |
| 1024 | 4096 | torch.float16 | amin | 0.03 | 0.16 | 0.32 |
| 1024 | 4096 | torch.float16 | prod | 0.03 | 0.16 | 0.32 |
| 1024 | 4096 | torch.float16 | std | 0.04 | 0.33 | 0.22 |
| 1024 | 4096 | torch.float16 | var | 0.04 | 0.33 | 0.22 |
| 1024 | 4096 | torch.float16 | var_mean | 0.05 | 0.26 | 0.17 |

## rms_norm

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.01 | 1.72 | 1.72 |
| 4096 | 4096 | torch.bfloat16 | 0.03 | 2.11 | 2.11 |
| 2048 | 5120 | torch.float16 | 0.02 | 2.00 | 2.00 |
| 1025 | 4096 | torch.float16 | 0.01 | 1.68 | 1.68 |

### baseline

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.07 | 0.24 | 0.24 |
| 4096 | 4096 | torch.bfloat16 | 0.25 | 0.27 | 0.27 |
| 2048 | 5120 | torch.float16 | 0.17 | 0.25 | 0.25 |
| 1025 | 4096 | torch.float16 | 0.07 | 0.24 | 0.24 |

## rope_neox

### tileops

| seq_len | head_dim | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 2048 | 64 | torch.float16 | 0.00 | 0.24 | 0.36 |
| 2048 | 64 | torch.bfloat16 | 0.00 | 0.24 | 0.36 |
| 2048 | 64 | torch.float32 | 0.00 | 0.22 | 0.65 |
| 2048 | 128 | torch.float16 | 0.00 | 0.41 | 0.62 |
| 2048 | 128 | torch.bfloat16 | 0.00 | 0.41 | 0.62 |
| 2048 | 128 | torch.float32 | 0.00 | 0.34 | 1.01 |
| 4096 | 128 | torch.float16 | 0.00 | 0.67 | 1.00 |
| 4096 | 128 | torch.bfloat16 | 0.00 | 0.67 | 1.00 |
| 4096 | 128 | torch.float32 | 0.00 | 0.54 | 1.62 |

### baseline

| seq_len | head_dim | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 2048 | 64 | torch.float16 | 0.01 | 0.04 | 0.06 |
| 2048 | 64 | torch.bfloat16 | 0.01 | 0.04 | 0.06 |
| 2048 | 64 | torch.float32 | 0.01 | 0.04 | 0.12 |
| 2048 | 128 | torch.float16 | 0.01 | 0.07 | 0.11 |
| 2048 | 128 | torch.bfloat16 | 0.01 | 0.07 | 0.11 |
| 2048 | 128 | torch.float32 | 0.01 | 0.07 | 0.22 |
| 4096 | 128 | torch.float16 | 0.02 | 0.13 | 0.20 |
| 4096 | 128 | torch.bfloat16 | 0.02 | 0.13 | 0.20 |
| 4096 | 128 | torch.float32 | 0.02 | 0.13 | 0.38 |

## softmax

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.02 | 1.05 | 1.05 |
| 4096 | 4096 | torch.bfloat16 | 0.06 | 1.18 | 1.18 |
| 1024 | 3000 | torch.float16 | 0.02 | 0.50 | 0.50 |
| 1025 | 4096 | torch.float16 | 0.02 | 1.05 | 1.05 |

### baseline

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.02 | 1.01 | 1.01 |
| 4096 | 4096 | torch.bfloat16 | 0.06 | 1.15 | 1.15 |
| 1024 | 3000 | torch.float16 | 0.01 | 0.84 | 0.84 |
| 1025 | 4096 | torch.float16 | 0.02 | 0.97 | 0.97 |

## log_softmax

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.02 | 1.02 | 0.81 |
| 4096 | 4096 | torch.bfloat16 | 0.08 | 1.11 | 0.89 |
| 1024 | 3000 | torch.float16 | 0.03 | 0.53 | 0.42 |
| 1025 | 4096 | torch.float16 | 0.02 | 1.02 | 0.81 |

### baseline

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.02 | 1.37 | 1.10 |
| 4096 | 4096 | torch.bfloat16 | 0.05 | 1.61 | 1.29 |
| 1024 | 3000 | torch.float16 | 0.01 | 1.16 | 0.93 |
| 1025 | 4096 | torch.float16 | 0.02 | 1.37 | 1.10 |

## logsumexp

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.01 | 1.30 | 0.87 |
| 4096 | 4096 | torch.bfloat16 | 0.03 | 1.60 | 1.07 |
| 1024 | 3000 | torch.float16 | 0.02 | 0.57 | 0.38 |
| 1025 | 4096 | torch.float16 | 0.01 | 1.28 | 0.86 |

### baseline

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.05 | 0.25 | 0.17 |
| 4096 | 4096 | torch.bfloat16 | 0.10 | 0.50 | 0.33 |
| 1024 | 3000 | torch.float16 | 0.04 | 0.21 | 0.14 |
| 1025 | 4096 | torch.float16 | 0.05 | 0.24 | 0.16 |

## topk_selector

### tileops

| batch | seq_len | seq_len_kv | kv_group | topk | in_dtype | out_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32768 | 65536 | 1 | 1024 | torch.float32 | torch.int32 | 13.53 | N/A | 0.64 |
| 1 | 32768 | 65536 | 1 | 2048 | torch.float32 | torch.int32 | 14.00 | N/A | 0.63 |
| 1 | 65535 | 131072 | 1 | 1024 | torch.float32 | torch.int32 | 44.56 | N/A | 0.78 |
| 1 | 65535 | 131072 | 1 | 2048 | torch.float32 | torch.int32 | 44.97 | N/A | 0.78 |

### baseline

| batch | seq_len | seq_len_kv | kv_group | topk | in_dtype | out_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32768 | 65536 | 1 | 1024 | torch.float32 | torch.int32 | 15.66 | N/A | 0.56 |
| 1 | 32768 | 65536 | 1 | 2048 | torch.float32 | torch.int32 | 14.91 | N/A | 0.59 |
| 1 | 65535 | 131072 | 1 | 1024 | torch.float32 | torch.int32 | 109.41 | N/A | 0.32 |
| 1 | 65535 | 131072 | 1 | 2048 | torch.float32 | torch.int32 | 112.16 | N/A | 0.31 |

## exp

### tileops

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| exp | 262144 | torch.float16 | torch.float16 | 0.00 | 0.11 | 0.46 |
| exp | 1048576 | torch.float16 | torch.float16 | 0.00 | 0.33 | 1.31 |
| exp | 4000000 | torch.float16 | torch.float16 | 0.01 | 0.63 | 2.53 |
| exp | 262144 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.11 | 0.46 |
| exp | 1048576 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.33 | 1.32 |
| exp | 4000000 | torch.bfloat16 | torch.bfloat16 | 0.01 | 0.63 | 2.52 |

### baseline

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| exp | 262144 | torch.float16 | torch.float16 | 0.00 | 0.11 | 0.44 |
| exp | 1048576 | torch.float16 | torch.float16 | 0.00 | 0.32 | 1.29 |
| exp | 4000000 | torch.float16 | torch.float16 | 0.01 | 0.63 | 2.53 |
| exp | 262144 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.11 | 0.44 |
| exp | 1048576 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.33 | 1.33 |
| exp | 4000000 | torch.bfloat16 | torch.bfloat16 | 0.01 | 0.63 | 2.54 |

## gelu

### tileops

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| gelu | 262144 | torch.float16 | torch.float16 | 0.00 | 0.11 | 0.44 |
| gelu | 1048576 | torch.float16 | torch.float16 | 0.00 | 0.30 | 1.18 |
| gelu | 4000000 | torch.float16 | torch.float16 | 0.01 | 0.48 | 1.91 |
| gelu | 262144 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.11 | 0.44 |
| gelu | 1048576 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.29 | 1.17 |
| gelu | 4000000 | torch.bfloat16 | torch.bfloat16 | 0.01 | 0.46 | 1.85 |

### baseline

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| gelu | 262144 | torch.float16 | torch.float16 | 0.00 | 0.10 | 0.42 |
| gelu | 1048576 | torch.float16 | torch.float16 | 0.00 | 0.29 | 1.15 |
| gelu | 4000000 | torch.float16 | torch.float16 | 0.01 | 0.52 | 2.07 |
| gelu | 262144 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.10 | 0.41 |
| gelu | 1048576 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.28 | 1.12 |
| gelu | 4000000 | torch.bfloat16 | torch.bfloat16 | 0.01 | 0.50 | 1.99 |

## logical_not

### tileops

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| logical_not | 262144 | torch.float16 | torch.bool | 0.00 | 0.11 | 0.32 |
| logical_not | 1048576 | torch.float16 | torch.bool | 0.00 | 0.21 | 0.64 |
| logical_not | 4000000 | torch.float16 | torch.bool | 0.01 | 0.30 | 0.91 |

### baseline

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| logical_not | 262144 | torch.float16 | torch.bool | 0.00 | 0.11 | 0.34 |
| logical_not | 1048576 | torch.float16 | torch.bool | 0.00 | 0.34 | 1.01 |
| logical_not | 4000000 | torch.float16 | torch.bool | 0.01 | 0.72 | 2.17 |

## bitwise_not

### tileops

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| bitwise_not | 262144 | torch.int32 | torch.int32 | 0.00 | 0.10 | 0.77 |
| bitwise_not | 1048576 | torch.int32 | torch.int32 | 0.01 | 0.20 | 1.56 |
| bitwise_not | 4000000 | torch.int32 | torch.int32 | 0.02 | 0.27 | 2.13 |

### baseline

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| bitwise_not | 262144 | torch.int32 | torch.int32 | 0.00 | 0.10 | 0.83 |
| bitwise_not | 1048576 | torch.int32 | torch.int32 | 0.00 | 0.25 | 1.96 |
| bitwise_not | 4000000 | torch.int32 | torch.int32 | 0.01 | 0.40 | 3.17 |

## isnan

### tileops

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| isnan | 262144 | torch.float16 | torch.bool | 0.00 | 0.11 | 0.32 |
| isnan | 1048576 | torch.float16 | torch.bool | 0.00 | 0.21 | 0.64 |
| isnan | 4000000 | torch.float16 | torch.bool | 0.01 | 0.30 | 0.91 |

### baseline

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| isnan | 262144 | torch.float16 | torch.bool | 0.00 | 0.11 | 0.34 |
| isnan | 1048576 | torch.float16 | torch.bool | 0.00 | 0.34 | 1.01 |
| isnan | 4000000 | torch.float16 | torch.bool | 0.01 | 0.72 | 2.17 |

## unary_strategy

### relu_direct

| n_total | shape_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4194304 | 1024x4096 | torch.float16 | direct | 0.01 | 0.29 | 1.18 |
| 4194304 | 1024x4096 | torch.bfloat16 | direct | 0.01 | 0.29 | 1.17 |
| 4194304 | 1024x4096 | torch.float32 | direct | 0.02 | 0.27 | 2.15 |
| 10485760 | 1024x10240 | torch.float16 | direct | 0.03 | 0.32 | 1.29 |
| 10485760 | 1024x10240 | torch.bfloat16 | direct | 0.03 | 0.32 | 1.29 |
| 10485760 | 1024x10240 | torch.float32 | direct | 0.04 | 0.29 | 2.34 |
| 20971520 | 1024x20480 | torch.float16 | direct | 0.06 | 0.34 | 1.35 |
| 20971520 | 1024x20480 | torch.bfloat16 | direct | 0.06 | 0.34 | 1.35 |
| 20971520 | 1024x20480 | torch.float32 | direct | 0.07 | 0.31 | 2.46 |

### relu_explicit_parallel

| n_total | shape_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4194304 | 1024x4096 | torch.float16 | explicit_parallel | 0.01 | 0.61 | 2.45 |
| 4194304 | 1024x4096 | torch.bfloat16 | explicit_parallel | 0.01 | 0.61 | 2.46 |
| 4194304 | 1024x4096 | torch.float32 | explicit_parallel | 0.01 | 0.39 | 3.09 |
| 10485760 | 1024x10240 | torch.float16 | explicit_parallel | 0.01 | 0.76 | 3.05 |
| 10485760 | 1024x10240 | torch.bfloat16 | explicit_parallel | 0.01 | 0.76 | 3.06 |
| 10485760 | 1024x10240 | torch.float32 | explicit_parallel | 0.02 | 0.47 | 3.76 |
| 20971520 | 1024x20480 | torch.float16 | explicit_parallel | 0.03 | 0.83 | 3.33 |
| 20971520 | 1024x20480 | torch.bfloat16 | explicit_parallel | 0.02 | 0.84 | 3.36 |
| 20971520 | 1024x20480 | torch.float32 | explicit_parallel | 0.04 | 0.51 | 4.06 |

### relu_register_copy

| n_total | shape_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4194304 | 1024x4096 | torch.float16 | register_copy | 0.01 | 0.66 | 2.66 |
| 4194304 | 1024x4096 | torch.bfloat16 | register_copy | 0.01 | 0.67 | 2.66 |
| 4194304 | 1024x4096 | torch.float32 | register_copy | 0.01 | 0.40 | 3.18 |
| 10485760 | 1024x10240 | torch.float16 | register_copy | 0.01 | 0.85 | 3.41 |
| 10485760 | 1024x10240 | torch.bfloat16 | register_copy | 0.01 | 0.85 | 3.41 |
| 10485760 | 1024x10240 | torch.float32 | register_copy | 0.02 | 0.48 | 3.84 |
| 20971520 | 1024x20480 | torch.float16 | register_copy | 0.02 | 0.96 | 3.84 |
| 20971520 | 1024x20480 | torch.bfloat16 | register_copy | 0.02 | 0.96 | 3.84 |
| 20971520 | 1024x20480 | torch.float32 | register_copy | 0.04 | 0.52 | 4.14 |

## vector_norm

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | l1 | 0.01 | 0.53 | 1.06 |
| 1024 | 4096 | torch.bfloat16 | l1 | 0.01 | 0.54 | 1.07 |
| 1024 | 4096 | torch.float32 | l1 | 0.01 | 0.35 | 1.38 |
| 4096 | 4096 | torch.float16 | l1 | 0.02 | 0.70 | 1.41 |
| 1024 | 4096 | torch.float16 | l2 | 0.01 | 0.53 | 1.05 |
| 1024 | 4096 | torch.bfloat16 | l2 | 0.01 | 0.53 | 1.06 |
| 1024 | 4096 | torch.float32 | l2 | 0.01 | 0.34 | 1.36 |
| 4096 | 4096 | torch.float16 | l2 | 0.02 | 0.70 | 1.39 |
| 1024 | 4096 | torch.float16 | inf | 0.03 | 0.15 | 0.30 |
| 1024 | 4096 | torch.bfloat16 | inf | 0.03 | 0.15 | 0.30 |
| 1024 | 4096 | torch.float32 | inf | 0.03 | 0.12 | 0.50 |
| 4096 | 4096 | torch.float16 | inf | 0.06 | 0.30 | 0.59 |

### baseline

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | l1 | 0.02 | 0.25 | 0.50 |
| 1024 | 4096 | torch.bfloat16 | l1 | 0.02 | 0.25 | 0.50 |
| 1024 | 4096 | torch.float32 | l1 | 0.02 | 0.22 | 0.89 |
| 4096 | 4096 | torch.float16 | l1 | 0.02 | 0.77 | 1.53 |
| 1024 | 4096 | torch.float16 | l2 | 0.02 | 0.25 | 0.50 |
| 1024 | 4096 | torch.bfloat16 | l2 | 0.02 | 0.25 | 0.50 |
| 1024 | 4096 | torch.float32 | l2 | 0.02 | 0.22 | 0.88 |
| 4096 | 4096 | torch.float16 | l2 | 0.02 | 0.77 | 1.53 |
| 1024 | 4096 | torch.float16 | inf | 0.02 | 0.25 | 0.49 |
| 1024 | 4096 | torch.bfloat16 | inf | 0.02 | 0.25 | 0.49 |
| 1024 | 4096 | torch.float32 | inf | 0.02 | 0.22 | 0.87 |
| 4096 | 4096 | torch.float16 | inf | 0.02 | 0.76 | 1.52 |

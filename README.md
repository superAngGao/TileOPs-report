<div align="center">

# TileOPs Operator Tracking

**22/186** operators complete (11%)

[View full HTML report](https://superanggao.github.io/TileOPs-report/nightly/)

Last updated: `2026-03-21T22:21:59Z`

</div>

---

## Overview

| | Count | Bar |
|:--|------:|:----|
| Implemented | **31**/186 | `██░░░░░░░░░░░░░` 31/186 |
| Test Passed | **23**/186 | `██░░░░░░░░░░░░░` 23/186 |
| Bench Passed | **30**/186 | `██░░░░░░░░░░░░░` 30/186 |
| **Done** | **22**/186 | `██░░░░░░░░░░░░░` 22/186 |

### Status Breakdown

| Test | | Bench | |
|:-----|----:|:------|----:|
| Passed | 23 | Qualified (ratio >= 0.8) | 3 |
| Failed | 0 | Passed (no ratio) | 27 |
| Missing | 163 | Underperforming | 1 |
| | | Failed | 0 |
| | | Missing | 155 |

## Assessment

> TileOPs is in early development with only 31 of 186 operators implemented (16.7%), concentrated almost entirely in the Reduce and GEMM categories. The implemented operators show strong quality — Reduce achieves 100% test coverage with all tests passing, and GEMM demonstrates excellent performance (gemm_fp16 at 1.07x baseline) — but 8 of 11 implemented GEMM ops lack test coverage, creating correctness risk. The vast majority of the library (155 ops across 9 categories) remains completely unimplemented, with the highest-impact gaps being Flash Attention, Elementwise, Norm, and MoE, all of which are essential for production LLM inference.

**Recommendations:**

1. Address the log_softmax performance regression (0.67 ratio) in Reduce immediately, and add test coverage for all 8 implemented-but-untested GEMM operators (fp8 variants, bmm, small_batch) to eliminate correctness risk in the most mature categories.
2. Prioritize implementing Flash Attention (flash_prefill_fwd, flash_decode_fwd) and Norm (rms_norm, layer_norm) as the next development sprint, since these are foundational to every transformer-based LLM and will unlock the most practical use cases.
3. Implement the Elementwise category in a systematic batch approach starting with unary_math and binary_arith subcategories, as these primitives are prerequisites for many higher-level operators and their absence blocks 72 ops from being testable or benchmarkable.


## Categories

### Elementwise

| | Progress | |
|:--|:---------|:--|
| Impl | `░░░░░░░░░░░░░░░` 0/72 | |
| Test | `░░░░░░░░░░░░░░░` 0/72 | |
| Bench | `░░░░░░░░░░░░░░░` 0/72 | |

> **Issues:** No operators implemented. All 72 ops are missing tests and benchmarks.

> **Evaluation:** Elementwise is entirely unimplemented, representing the largest gap in the library (72 ops). Prioritize implementing foundational subcategories first: unary_math and binary_arith ops, as they underpin nearly all other categories. Consider batch-implementing common activation functions (relu, gelu, silu) as a second wave.

<details>
<summary>0/72 done - click to expand</summary>

| | Operator | Test | Bench | Ratio |
|:--|:---------|:----:|:-----:|------:|
| ⬜ | add | missing | missing | - |
| ⬜ | sub | missing | missing | - |
| ⬜ | mul | missing | missing | - |
| ⬜ | div | missing | missing | - |
| ⬜ | remainder | missing | missing | - |
| ⬜ | pow | missing | missing | - |
| ⬜ | floor_divide | missing | missing | - |
| ⬜ | lerp | missing | missing | - |
| ⬜ | maximum | missing | missing | - |
| ⬜ | minimum | missing | missing | - |
| ⬜ | exp | missing | missing | - |
| ⬜ | log | missing | missing | - |
| ⬜ | sqrt | missing | missing | - |
| ⬜ | rsqrt | missing | missing | - |
| ⬜ | abs | missing | missing | - |
| ⬜ | neg | missing | missing | - |
| ⬜ | reciprocal | missing | missing | - |
| ⬜ | sign | missing | missing | - |
| ⬜ | sin | missing | missing | - |
| ⬜ | cos | missing | missing | - |
| ⬜ | floor | missing | missing | - |
| ⬜ | ceil | missing | missing | - |
| ⬜ | round | missing | missing | - |
| ⬜ | trunc | missing | missing | - |
| ⬜ | erf | missing | missing | - |
| ⬜ | log1p | missing | missing | - |
| ⬜ | expm1 | missing | missing | - |
| ⬜ | relu | missing | missing | - |
| ⬜ | gelu | missing | missing | - |
| ⬜ | silu | missing | missing | - |
| ⬜ | sigmoid | missing | missing | - |
| ⬜ | tanh | missing | missing | - |
| ⬜ | leaky_relu | missing | missing | - |
| ⬜ | elu | missing | missing | - |
| ⬜ | selu | missing | missing | - |
| ⬜ | hardswish | missing | missing | - |
| ⬜ | hardsigmoid | missing | missing | - |
| ⬜ | hardtanh | missing | missing | - |
| ⬜ | softplus | missing | missing | - |
| ⬜ | mish | missing | missing | - |
| ⬜ | prelu | missing | missing | - |
| ⬜ | silu_and_mul | missing | missing | - |
| ⬜ | gelu_and_mul | missing | missing | - |
| ⬜ | gelu_tanh_and_mul | missing | missing | - |
| ⬜ | eq | missing | missing | - |
| ⬜ | ne | missing | missing | - |
| ⬜ | gt | missing | missing | - |
| ⬜ | lt | missing | missing | - |
| ⬜ | ge | missing | missing | - |
| ⬜ | le | missing | missing | - |
| ⬜ | bitwise_and | missing | missing | - |
| ⬜ | bitwise_or | missing | missing | - |
| ⬜ | bitwise_xor | missing | missing | - |
| ⬜ | bitwise_not | missing | missing | - |
| ⬜ | logical_not | missing | missing | - |
| ⬜ | logical_and | missing | missing | - |
| ⬜ | logical_or | missing | missing | - |
| ⬜ | where | missing | missing | - |
| ⬜ | clamp | missing | missing | - |
| ⬜ | masked_fill | missing | missing | - |
| ⬜ | nan_to_num | missing | missing | - |
| ⬜ | isnan | missing | missing | - |
| ⬜ | isinf | missing | missing | - |
| ⬜ | isfinite | missing | missing | - |
| ⬜ | dropout | missing | missing | - |
| ⬜ | rope_neox | missing | missing | - |
| ⬜ | rope_non_neox | missing | missing | - |
| ⬜ | rope_llama31 | missing | missing | - |
| ⬜ | yarn_rope | missing | missing | - |
| ⬜ | longrope | missing | missing | - |
| ⬜ | alibi | missing | missing | - |
| ⬜ | sinusoidal | missing | missing | - |

</details>

### Reduce | Func: ⭐⭐⭐⭐⭐ (5/5) | Perf: ⭐⭐⭐⭐☆ (4/5)

| | Progress | |
|:--|:---------|:--|
| Impl | `███████████████` 20/20 | |
| Test | `███████████████` 20/20 | |
| Bench | `██████████████░` 19/20 | |

> **Issues:** log_softmax is underperforming at 0.67 ratio, well below the 0.80 threshold. All other benchmarked ops use 'passed' status without explicit ratios, suggesting they meet baseline but ratio data is unavailable for most.

> **Evaluation:** Reduce is the most mature category with 100% test coverage and all ops implemented. The primary concern is log_softmax performance (0.67 ratio). Investigate the log_softmax kernel for numerical stability trade-offs that may be causing slowdowns, and consider fusing the log and softmax passes. The logsumexp result (3.43x) is excellent and may indicate optimization techniques transferable to log_softmax.

<details>
<summary>19/20 done - click to expand</summary>

| | Operator | Test | Bench | Ratio |
|:--|:---------|:----:|:-----:|------:|
| ✅ | sum | passed | passed | - |
| ✅ | mean | passed | passed | - |
| ✅ | amin | passed | passed | - |
| ✅ | amax | passed | passed | - |
| ✅ | prod | passed | passed | - |
| ✅ | std | passed | passed | - |
| ✅ | var | passed | passed | - |
| ✅ | var_mean | passed | passed | - |
| ✅ | softmax | passed | qualified | 0.92 |
| 🟡 | log_softmax | passed | underperforming | 0.67 |
| ✅ | logsumexp | passed | qualified | 3.43 |
| ✅ | argmax | passed | passed | - |
| ✅ | argmin | passed | passed | - |
| ✅ | cumsum | passed | passed | - |
| ✅ | cumprod | passed | passed | - |
| ✅ | any | passed | passed | - |
| ✅ | all | passed | passed | - |
| ✅ | l1_norm | passed | passed | - |
| ✅ | l2_norm | passed | passed | - |
| ✅ | inf_norm | passed | passed | - |

</details>

### Norm

| | Progress | |
|:--|:---------|:--|
| Impl | `░░░░░░░░░░░░░░░` 0/10 | |
| Test | `░░░░░░░░░░░░░░░` 0/10 | |
| Bench | `░░░░░░░░░░░░░░░` 0/10 | |

> **Issues:** No operators implemented. All 10 norm ops are missing tests and benchmarks.

> **Evaluation:** Norm is entirely unimplemented despite being critical for LLM inference (rms_norm, layer_norm are used in nearly every transformer model). Prioritize rms_norm and layer_norm as they are the most commonly used. Fused variants (fused_add_rmsnorm, fused_add_layer_norm) should follow as they provide significant memory bandwidth savings.

<details>
<summary>0/10 done - click to expand</summary>

| | Operator | Test | Bench | Ratio |
|:--|:---------|:----:|:-----:|------:|
| ⬜ | layer_norm | missing | missing | - |
| ⬜ | batch_norm | missing | missing | - |
| ⬜ | group_norm | missing | missing | - |
| ⬜ | instance_norm | missing | missing | - |
| ⬜ | rms_norm | missing | missing | - |
| ⬜ | qk_norm | missing | missing | - |
| ⬜ | ada_layer_norm | missing | missing | - |
| ⬜ | ada_layer_norm_zero | missing | missing | - |
| ⬜ | fused_add_layer_norm | missing | missing | - |
| ⬜ | fused_add_rmsnorm | missing | missing | - |

</details>

### Conv & Pooling

| | Progress | |
|:--|:---------|:--|
| Impl | `░░░░░░░░░░░░░░░` 0/16 | |
| Test | `░░░░░░░░░░░░░░░` 0/16 | |
| Bench | `░░░░░░░░░░░░░░░` 0/16 | |

> **Issues:** No operators implemented. All 16 conv and pooling ops are missing tests and benchmarks.

> **Evaluation:** Conv & Pooling is entirely unimplemented. Given TileOPs is focused on LLM operators, this category may be lower priority than Norm, Flash Attention, or MoE. Recommend deprioritizing until core LLM ops are complete, then starting with conv1d and conv2d as foundational building blocks.

<details>
<summary>0/16 done - click to expand</summary>

| | Operator | Test | Bench | Ratio |
|:--|:---------|:----:|:-----:|------:|
| ⬜ | conv1d | missing | missing | - |
| ⬜ | conv2d | missing | missing | - |
| ⬜ | conv3d | missing | missing | - |
| ⬜ | conv_transpose1d | missing | missing | - |
| ⬜ | conv_transpose2d | missing | missing | - |
| ⬜ | depthwise_conv2d | missing | missing | - |
| ⬜ | grouped_conv2d | missing | missing | - |
| ⬜ | dilated_conv2d | missing | missing | - |
| ⬜ | max_pool1d | missing | missing | - |
| ⬜ | max_pool2d | missing | missing | - |
| ⬜ | max_pool3d | missing | missing | - |
| ⬜ | avg_pool1d | missing | missing | - |
| ⬜ | avg_pool2d | missing | missing | - |
| ⬜ | avg_pool3d | missing | missing | - |
| ⬜ | adaptive_avg_pool2d | missing | missing | - |
| ⬜ | adaptive_max_pool2d | missing | missing | - |

</details>

### GEMM | Func: ⭐⭐⭐⭐☆ (4/5) | Perf: ⭐⭐⭐⭐⭐ (5/5)

| | Progress | |
|:--|:---------|:--|
| Impl | `█████████░░░░░░` 11/19 | |
| Test | `██░░░░░░░░░░░░░` 3/19 | |
| Bench | `█████████░░░░░░` 11/19 | |

> **Issues:** 8 implemented ops lack test coverage (gemm_fp8, gemm_fp8_block_scaled, gemv_fp8, small_batch_gemm_fp16, small_batch_gemm_fp8, bmm_fp16, bmm_fp8, groupgemm_fp8). 8 unimplemented ops have no benchmarks. Only gemm_fp16 has an explicit benchmark ratio (1.07).

> **Evaluation:** GEMM shows strong performance where measured (gemm_fp16 at 1.07x baseline) and all implemented ops pass benchmarks. The critical gap is test coverage — 8 implemented ops have no tests, creating correctness risk. Immediately add test mappings for all implemented GEMM variants, especially fp8 variants which are complex. Then prioritize implementing lowbit_gemm ops (w4a16, w8a8) as they are essential for quantized inference.

<details>
<summary>3/19 done - click to expand</summary>

| | Operator | Test | Bench | Ratio |
|:--|:---------|:----:|:-----:|------:|
| ✅ | gemm_fp16 | passed | qualified | 1.07 |
| 🟦 | gemm_fp8 | missing | passed | - |
| 🟦 | gemm_fp8_block_scaled | missing | passed | - |
| ✅ | gemv_fp16 | passed | passed | - |
| 🟦 | gemv_fp8 | missing | passed | - |
| 🟦 | small_batch_gemm_fp16 | missing | passed | - |
| 🟦 | small_batch_gemm_fp8 | missing | passed | - |
| 🟦 | bmm_fp16 | missing | passed | - |
| 🟦 | bmm_fp8 | missing | passed | - |
| ✅ | groupgemm_fp16 | passed | passed | - |
| 🟦 | groupgemm_fp8 | missing | passed | - |
| ⬜ | outer | missing | missing | - |
| ⬜ | w4a16 | missing | missing | - |
| ⬜ | w8a8 | missing | missing | - |
| ⬜ | w8a8_int8 | missing | missing | - |
| ⬜ | weight_only_int4 | missing | missing | - |
| ⬜ | fp4 | missing | missing | - |
| ⬜ | sparse_gemm_fp16 | missing | missing | - |
| ⬜ | sparse_gemm_fp8 | missing | missing | - |

</details>

### Quantize

| | Progress | |
|:--|:---------|:--|
| Impl | `░░░░░░░░░░░░░░░` 0/10 | |
| Test | `░░░░░░░░░░░░░░░` 0/10 | |
| Bench | `░░░░░░░░░░░░░░░` 0/10 | |

> **Issues:** No operators implemented. All 10 quantize ops are missing tests and benchmarks.

> **Evaluation:** Quantize is entirely unimplemented despite being tightly coupled with the fp8 and int8 GEMM ops already implemented. Prioritize fp8_per_tensor and fp8_per_block to complement existing fp8 GEMM kernels, then int8_per_tensor and int8_per_channel for smooth_quant support.

<details>
<summary>0/10 done - click to expand</summary>

| | Operator | Test | Bench | Ratio |
|:--|:---------|:----:|:-----:|------:|
| ⬜ | int8_per_tensor | missing | missing | - |
| ⬜ | int8_per_channel | missing | missing | - |
| ⬜ | int8_per_block | missing | missing | - |
| ⬜ | smooth_quant | missing | missing | - |
| ⬜ | int4_per_channel | missing | missing | - |
| ⬜ | int4_per_block | missing | missing | - |
| ⬜ | nf4 | missing | missing | - |
| ⬜ | fp8_per_tensor | missing | missing | - |
| ⬜ | fp8_per_block | missing | missing | - |
| ⬜ | fp8_cast_transpose | missing | missing | - |

</details>

### Sampling

| | Progress | |
|:--|:---------|:--|
| Impl | `░░░░░░░░░░░░░░░` 0/7 | |
| Test | `░░░░░░░░░░░░░░░` 0/7 | |
| Bench | `░░░░░░░░░░░░░░░` 0/7 | |

> **Issues:** No operators implemented. All 7 sampling ops are missing tests and benchmarks.

> **Evaluation:** Sampling is entirely unimplemented. These ops are needed for LLM decoding pipelines. Recommend starting with top_k and top_p as they are the most commonly used, then top_k_top_p as a fused variant. chain_speculative_sampling is lower priority but valuable for speculative decoding workflows.

<details>
<summary>0/7 done - click to expand</summary>

| | Operator | Test | Bench | Ratio |
|:--|:---------|:----:|:-----:|------:|
| ⬜ | top_k | missing | missing | - |
| ⬜ | top_p | missing | missing | - |
| ⬜ | min_p | missing | missing | - |
| ⬜ | top_k_top_p | missing | missing | - |
| ⬜ | temperature_scale | missing | missing | - |
| ⬜ | sampling_from_probs | missing | missing | - |
| ⬜ | chain_speculative_sampling | missing | missing | - |

</details>

### Flash Attention

| | Progress | |
|:--|:---------|:--|
| Impl | `░░░░░░░░░░░░░░░` 0/16 | |
| Test | `░░░░░░░░░░░░░░░` 0/16 | |
| Bench | `░░░░░░░░░░░░░░░` 0/16 | |

> **Issues:** No operators implemented. All 16 flash attention ops are missing tests and benchmarks.

> **Evaluation:** Flash Attention is entirely unimplemented and represents the highest-impact gap for LLM inference performance. Prioritize flash_prefill_fwd and flash_decode_fwd as the foundational kernels, then varlen variants. MLA variants (mla_prefill_fwd, mla_decode_fwd) are critical for DeepSeek-style models and should be a near-term target.

<details>
<summary>0/16 done - click to expand</summary>

| | Operator | Test | Bench | Ratio |
|:--|:---------|:----:|:-----:|------:|
| ⬜ | flash_prefill_fwd | missing | missing | - |
| ⬜ | flash_prefill_bwd | missing | missing | - |
| ⬜ | flash_prefill_varlen_fwd | missing | missing | - |
| ⬜ | flash_prefill_varlen_bwd | missing | missing | - |
| ⬜ | flash_decode_fwd | missing | missing | - |
| ⬜ | flash_decode_paged_fwd | missing | missing | - |
| ⬜ | flash_decode_varlen_fwd | missing | missing | - |
| ⬜ | flash_chunked_prefill_fwd | missing | missing | - |
| ⬜ | mla_prefill_fwd | missing | missing | - |
| ⬜ | mla_prefill_bwd | missing | missing | - |
| ⬜ | mla_decode_fwd | missing | missing | - |
| ⬜ | mla_decode_paged_fwd | missing | missing | - |
| ⬜ | nsa_prefill_fwd | missing | missing | - |
| ⬜ | nsa_decode_fwd | missing | missing | - |
| ⬜ | dsa_prefill_fwd | missing | missing | - |
| ⬜ | dsa_decode_fwd | missing | missing | - |

</details>

### MoE

| | Progress | |
|:--|:---------|:--|
| Impl | `░░░░░░░░░░░░░░░` 0/6 | |
| Test | `░░░░░░░░░░░░░░░` 0/6 | |
| Bench | `░░░░░░░░░░░░░░░` 0/6 | |

> **Issues:** No operators implemented. All 6 MoE ops are missing tests and benchmarks.

> **Evaluation:** MoE is entirely unimplemented. Given the prevalence of MoE architectures (DeepSeek, Qwen, GLM), this is a high-priority gap. Recommend starting with fused_moe_deepseek as a reference implementation, then adapting for other model families. permute_align and unpermute_depad are prerequisite utilities that should be implemented first.

<details>
<summary>0/6 done - click to expand</summary>

| | Operator | Test | Bench | Ratio |
|:--|:---------|:----:|:-----:|------:|
| ⬜ | permute_align | missing | missing | - |
| ⬜ | unpermute_depad | missing | missing | - |
| ⬜ | fused_moe_deepseek | missing | missing | - |
| ⬜ | fused_moe_glm | missing | missing | - |
| ⬜ | fused_moe_kimi | missing | missing | - |
| ⬜ | fused_moe_qwen | missing | missing | - |

</details>

### Linear Attention

| | Progress | |
|:--|:---------|:--|
| Impl | `░░░░░░░░░░░░░░░` 0/8 | |
| Test | `░░░░░░░░░░░░░░░` 0/8 | |
| Bench | `░░░░░░░░░░░░░░░` 0/8 | |

> **Issues:** No operators implemented. All 8 linear attention ops are missing tests and benchmarks.

> **Evaluation:** Linear Attention is entirely unimplemented. This is a more specialized category; recommend implementing after Flash Attention is stable. Start with deltanet_chunkwise and gla_chunkwise as representative chunkwise algorithms, then add recurrence variants.

<details>
<summary>0/8 done - click to expand</summary>

| | Operator | Test | Bench | Ratio |
|:--|:---------|:----:|:-----:|------:|
| ⬜ | gated_deltanet_chunkwise | missing | missing | - |
| ⬜ | gated_deltanet_recurrence | missing | missing | - |
| ⬜ | deltanet_chunkwise | missing | missing | - |
| ⬜ | deltanet_recurrence | missing | missing | - |
| ⬜ | gla_chunkwise | missing | missing | - |
| ⬜ | gla_recurrence | missing | missing | - |
| ⬜ | retnet_chunkwise | missing | missing | - |
| ⬜ | retnet_recurrence | missing | missing | - |

</details>

### SSM

| | Progress | |
|:--|:---------|:--|
| Impl | `░░░░░░░░░░░░░░░` 0/2 | |
| Test | `░░░░░░░░░░░░░░░` 0/2 | |
| Bench | `░░░░░░░░░░░░░░░` 0/2 | |

> **Issues:** No operators implemented. Both mamba1 and mamba2 are missing tests and benchmarks.

> **Evaluation:** SSM is entirely unimplemented with only 2 ops needed. While a small category, Mamba models have significant adoption. Recommend implementing mamba2 first as it is more widely used in recent models, then mamba1 for legacy support.

<details>
<summary>0/2 done - click to expand</summary>

| | Operator | Test | Bench | Ratio |
|:--|:---------|:----:|:-----:|------:|
| ⬜ | mamba1 | missing | missing | - |
| ⬜ | mamba2 | missing | missing | - |

</details>


---

<sub>Auto-generated by the [TileOPs Report](https://superanggao.github.io/TileOPs-report/nightly/) pipeline from [`op_registry.json`](scripts/op_registry.json)</sub>

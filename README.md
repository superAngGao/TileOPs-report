<div align="center">

# TileOPs Operator Tracking

**96/186** operators complete (51%)

[View full HTML report](https://superanggao.github.io/TileOPs-report/nightly/)

Last updated: `2026-03-23T22:30:11Z`

</div>

---

## Overview

| | Count | Bar |
|:--|------:|:----|
| Implemented | **125**/186 | `██████████░░░░░` 125/186 |
| Test Passed | **106**/186 | `█████████░░░░░░` 106/186 |
| Bench Passed | **112**/186 | `█████████░░░░░░` 112/186 |
| **Done** | **96**/186 | `████████░░░░░░░` 96/186 |

### Status Breakdown

| Test | | Bench | |
|:-----|----:|:------|----:|
| Passed | 106 | Qualified (ratio >= 0.8) | 21 |
| Failed | 4 | Passed (no ratio) | 91 |
| Missing | 76 | Underperforming | 3 |
| | | Failed | 0 |
| | | Missing | 71 |

## Assessment

> TileOPs shows strong progress in core categories — Reduce (100% complete), Norm (90%), Flash Attention (50% with all passing), and Linear Attention (50% with all passing) — but has significant gaps in Conv & Pooling (0% implemented), Sampling (14%), Quantize (20%), and MoE (17%). Of 186 total ops, 125 are implemented, 106 tested, and 96 fully done, with only 4 test failures concentrated in RoPE and dropout. The main concerns are the 3 severely underperforming elementwise ops (bitwise_not, logical_and, logical_or at 0.33x baseline) and the complete absence of Conv & Pooling and SSM implementations.

**Recommendations:**

1. Fix the 4 failing tests (dropout, rope_neox, rope_non_neox, rope_llama31) and investigate the 3 underperforming bitwise/logical ops (ratio=0.33) as these represent correctness and performance regressions in production-critical ops.
2. Prioritize implementing top_k, top_p, and temperature_scale in Sampling, and at least 2 fused_moe variants (deepseek, qwen) in MoE, as these are essential for end-to-end LLM inference pipelines.
3. Add tests for the 6 implemented-but-untested GEMM ops (gemm_fp8, gemv_fp8, small_batch_gemm variants, groupgemm_fp8) and fix benchmark ratio reporting for Reduce, Norm, Flash Attention, GEMM, and Linear Attention categories where ratios are not being captured in the structured log.


## Categories

### Elementwise | Func: ⭐⭐⭐⭐☆ (4/5) | Perf: ⭐⭐⭐⭐☆ (4/5)

| | Progress | |
|:--|:---------|:--|
| Impl | `███████████████` 70/72 | |
| Test | `████████████░░░` 59/72 | |
| Bench | `████████████░░░` 58/72 | |

> **Issues:** dropout, rope_neox, rope_non_neox, rope_llama31 tests failing. bitwise_not, logical_and, logical_or severely underperforming (ratio=0.33). 11 ops missing benchmarks (where, clamp, masked_fill, nan_to_num, isnan, isinf, isfinite, alibi, sinusoidal, yarn_rope, longrope). 5 ops missing tests (leaky_relu, elu, hardtanh, softplus, prelu).

> **Evaluation:** The category is largely functional with strong benchmark results for most ops, but RoPE variants and dropout have test failures that need immediate investigation. Fix the 3 underperforming logical/bitwise ops and add missing benchmarks for special_elementwise ops. Prioritize resolving RoPE test failures as they impact LLM inference correctness.

<details>
<summary>49/72 done - click to expand</summary>

| | Operator | Test | Bench | Ratio |
|:--|:---------|:----:|:-----:|------:|
| ✅ | add | passed | passed | - |
| ✅ | sub | passed | qualified | 1.00 |
| ✅ | mul | passed | qualified | 1.00 |
| ✅ | div | passed | qualified | 1.00 |
| ✅ | remainder | passed | qualified | 1.00 |
| ✅ | pow | passed | qualified | 1.00 |
| ✅ | floor_divide | passed | qualified | 2.33 |
| ✅ | lerp | passed | qualified | 1.00 |
| ✅ | maximum | passed | qualified | 1.00 |
| ✅ | minimum | passed | qualified | 1.00 |
| ✅ | exp | passed | qualified | 1.00 |
| ✅ | log | passed | passed | - |
| ✅ | sqrt | passed | passed | - |
| ✅ | rsqrt | passed | passed | - |
| ✅ | abs | passed | passed | - |
| ✅ | neg | passed | passed | - |
| ✅ | reciprocal | passed | passed | - |
| ✅ | sign | passed | passed | - |
| ✅ | sin | passed | passed | - |
| ✅ | cos | passed | passed | - |
| ✅ | floor | passed | passed | - |
| ✅ | ceil | passed | passed | - |
| ✅ | round | passed | passed | - |
| ✅ | trunc | passed | passed | - |
| ✅ | erf | passed | passed | - |
| ✅ | log1p | passed | passed | - |
| ✅ | expm1 | passed | passed | - |
| ✅ | relu | passed | passed | - |
| ✅ | gelu | passed | qualified | 1.00 |
| ✅ | silu | passed | passed | - |
| ✅ | sigmoid | passed | passed | - |
| ✅ | tanh | passed | passed | - |
| 🟦 | leaky_relu | missing | qualified | 1.00 |
| 🟦 | elu | missing | qualified | 1.40 |
| ✅ | selu | passed | passed | - |
| ✅ | hardswish | passed | passed | - |
| ✅ | hardsigmoid | passed | passed | - |
| 🟦 | hardtanh | missing | qualified | 1.00 |
| 🟦 | softplus | missing | qualified | 1.14 |
| ✅ | mish | passed | passed | - |
| 🟦 | prelu | missing | passed | - |
| ✅ | silu_and_mul | passed | passed | - |
| ✅ | gelu_and_mul | passed | qualified | 3.29 |
| ✅ | gelu_tanh_and_mul | passed | qualified | 3.29 |
| ✅ | eq | passed | passed | - |
| ✅ | ne | passed | passed | - |
| ✅ | gt | passed | passed | - |
| ✅ | lt | passed | passed | - |
| ✅ | ge | passed | passed | - |
| ✅ | le | passed | passed | - |
| ✅ | bitwise_and | passed | qualified | 1.00 |
| ✅ | bitwise_or | passed | qualified | 1.00 |
| ✅ | bitwise_xor | passed | qualified | 1.00 |
| 🟡 | bitwise_not | passed | underperforming | 0.33 |
| ✅ | logical_not | passed | qualified | 1.00 |
| 🟡 | logical_and | passed | underperforming | 0.33 |
| 🟡 | logical_or | passed | underperforming | 0.33 |
| 🟡 | where | passed | missing | - |
| 🟡 | clamp | passed | missing | - |
| 🟡 | masked_fill | passed | missing | - |
| 🟡 | nan_to_num | passed | missing | - |
| 🟡 | isnan | passed | missing | - |
| 🟡 | isinf | passed | missing | - |
| 🟡 | isfinite | passed | missing | - |
| ❌ | dropout | failed | passed | - |
| ❌ | rope_neox | failed | passed | - |
| ❌ | rope_non_neox | failed | passed | - |
| ❌ | rope_llama31 | failed | passed | - |
| ⬜ | yarn_rope | missing | missing | - |
| ⬜ | longrope | missing | missing | - |
| 🟦 | alibi | missing | missing | - |
| 🟦 | sinusoidal | missing | missing | - |

</details>

### Reduce | Func: ⭐⭐⭐⭐⭐ (5/5)

| | Progress | |
|:--|:---------|:--|
| Impl | `███████████████` 20/20 | |
| Test | `███████████████` 20/20 | |
| Bench | `███████████████` 20/20 | |

> **Issues:** All 20 ops pass tests, but benchmark log shows 0 qualified and 0 missing — bench statuses are listed as 'passed' in the operator list without ratio data, suggesting benchmarks ran but ratios were not captured in the structured log.

> **Evaluation:** Reduce is the most complete category with 100% test pass rate and full implementation. Benchmark data appears present but ratios are not surfaced in the structured log. Ensure benchmark ratio extraction is working for this category to enable performance tracking.

<details>
<summary>20/20 done - click to expand</summary>

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
| ✅ | softmax | passed | passed | - |
| ✅ | log_softmax | passed | passed | - |
| ✅ | logsumexp | passed | passed | - |
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

### Norm | Func: ⭐⭐⭐⭐⭐ (5/5)

| | Progress | |
|:--|:---------|:--|
| Impl | `███████████████` 10/10 | |
| Test | `██████████████░` 9/10 | |
| Bench | `██████████████░` 9/10 | |

> **Issues:** qk_norm has no test or benchmark mapping. Benchmark ratios not captured in structured log (all show 'passed' without ratio).

> **Evaluation:** Norm is nearly complete with 9/10 ops fully passing. Add test and benchmark coverage for qk_norm to complete the category. Ensure benchmark ratio reporting is consistent with other categories.

<details>
<summary>9/10 done - click to expand</summary>

| | Operator | Test | Bench | Ratio |
|:--|:---------|:----:|:-----:|------:|
| ✅ | layer_norm | passed | passed | - |
| ✅ | batch_norm | passed | passed | - |
| ✅ | group_norm | passed | passed | - |
| ✅ | instance_norm | passed | passed | - |
| ✅ | rms_norm | passed | passed | - |
| 🟦 | qk_norm | missing | missing | - |
| ✅ | ada_layer_norm | passed | passed | - |
| ✅ | ada_layer_norm_zero | passed | passed | - |
| ✅ | fused_add_layer_norm | passed | passed | - |
| ✅ | fused_add_rmsnorm | passed | passed | - |

</details>

### Conv & Pooling

| | Progress | |
|:--|:---------|:--|
| Impl | `░░░░░░░░░░░░░░░` 0/16 | |
| Test | `░░░░░░░░░░░░░░░` 0/16 | |
| Bench | `░░░░░░░░░░░░░░░` 0/16 | |

> **Issues:** Entire category unimplemented — 0/16 ops have any implementation, tests, or benchmarks.

> **Evaluation:** Conv & Pooling is a complete gap in the library. This is a significant missing feature area. Prioritize at minimum conv2d and max_pool2d/avg_pool2d as the most commonly used ops, then expand to other variants.

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

### GEMM | Func: ⭐⭐⭐⭐☆ (4/5)

| | Progress | |
|:--|:---------|:--|
| Impl | `███████░░░░░░░░` 9/19 | |
| Test | `██░░░░░░░░░░░░░` 3/19 | |
| Bench | `███████░░░░░░░░` 9/19 | |

> **Issues:** Only 9/19 ops implemented. 6 implemented ops (gemm_fp8, gemm_fp8_block_scaled, gemv_fp8, small_batch_gemm_fp16, small_batch_gemm_fp8, groupgemm_fp8) have benchmarks but no tests. All lowbit_gemm, bmm, outer, and sparse_gemm variants unimplemented. Benchmark ratios not captured in structured log.

> **Evaluation:** GEMM has a solid foundation with passing tests for core ops, but test coverage for implemented ops is critically low. Add tests for the 6 implemented-but-untested ops immediately. Plan implementation of bmm and lowbit_gemm variants as they are essential for LLM workloads.

<details>
<summary>3/19 done - click to expand</summary>

| | Operator | Test | Bench | Ratio |
|:--|:---------|:----:|:-----:|------:|
| ✅ | gemm_fp16 | passed | passed | - |
| 🟦 | gemm_fp8 | missing | passed | - |
| 🟦 | gemm_fp8_block_scaled | missing | passed | - |
| ✅ | gemv_fp16 | passed | passed | - |
| 🟦 | gemv_fp8 | missing | passed | - |
| 🟦 | small_batch_gemm_fp16 | missing | passed | - |
| 🟦 | small_batch_gemm_fp8 | missing | passed | - |
| ⬜ | bmm_fp16 | missing | missing | - |
| ⬜ | bmm_fp8 | missing | missing | - |
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

### Quantize | Func: ⭐⭐⭐⭐☆ (4/5)

| | Progress | |
|:--|:---------|:--|
| Impl | `███░░░░░░░░░░░░` 2/10 | |
| Test | `██░░░░░░░░░░░░░` 1/10 | |
| Bench | `███░░░░░░░░░░░░` 2/10 | |

> **Issues:** Only 2/10 ops implemented (fp8_per_tensor, fp8_per_block). All int8 and int4 quantization variants unimplemented. fp8_per_block has benchmark but no test. fp8_cast_transpose unimplemented.

> **Evaluation:** Quantize coverage is minimal, limited to FP8 only. Given the importance of quantization for LLM deployment, int8_per_tensor and int8_per_channel should be prioritized next. Add a test for fp8_per_block to complete coverage of implemented ops.

<details>
<summary>1/10 done - click to expand</summary>

| | Operator | Test | Bench | Ratio |
|:--|:---------|:----:|:-----:|------:|
| ⬜ | int8_per_tensor | missing | missing | - |
| ⬜ | int8_per_channel | missing | missing | - |
| ⬜ | int8_per_block | missing | missing | - |
| ⬜ | smooth_quant | missing | missing | - |
| ⬜ | int4_per_channel | missing | missing | - |
| ⬜ | int4_per_block | missing | missing | - |
| ⬜ | nf4 | missing | missing | - |
| ✅ | fp8_per_tensor | passed | passed | - |
| 🟦 | fp8_per_block | missing | passed | - |
| ⬜ | fp8_cast_transpose | missing | missing | - |

</details>

### Sampling | Func: ⭐⭐⭐⭐⭐ (5/5)

| | Progress | |
|:--|:---------|:--|
| Impl | `██░░░░░░░░░░░░░` 1/7 | |
| Test | `██░░░░░░░░░░░░░` 1/7 | |
| Bench | `██░░░░░░░░░░░░░` 1/7 | |

> **Issues:** Only chain_speculative_sampling implemented. top_k, top_p, min_p, top_k_top_p, temperature_scale, sampling_from_probs all unimplemented.

> **Evaluation:** Sampling is severely underdeveloped with only 1/7 ops implemented. top_k and top_p are fundamental sampling ops used in virtually all LLM inference pipelines and should be implemented immediately. temperature_scale is also a quick win.

<details>
<summary>1/7 done - click to expand</summary>

| | Operator | Test | Bench | Ratio |
|:--|:---------|:----:|:-----:|------:|
| ⬜ | top_k | missing | missing | - |
| ⬜ | top_p | missing | missing | - |
| ⬜ | min_p | missing | missing | - |
| ⬜ | top_k_top_p | missing | missing | - |
| ⬜ | temperature_scale | missing | missing | - |
| ⬜ | sampling_from_probs | missing | missing | - |
| ✅ | chain_speculative_sampling | passed | passed | - |

</details>

### Flash Attention | Func: ⭐⭐⭐⭐⭐ (5/5)

| | Progress | |
|:--|:---------|:--|
| Impl | `████████░░░░░░░` 8/16 | |
| Test | `████████░░░░░░░` 8/16 | |
| Bench | `████████░░░░░░░` 8/16 | |

> **Issues:** 8/16 ops unimplemented: flash_prefill_varlen_bwd, flash_decode_varlen_fwd, flash_chunked_prefill_fwd, mla_prefill_fwd/bwd, mla_decode_paged_fwd, nsa_decode_fwd, dsa_prefill_fwd. Benchmark ratios not captured in structured log for implemented ops.

> **Evaluation:** All implemented Flash Attention ops pass tests, which is a strong result. The missing ops are important for production use (varlen backward, chunked prefill, MLA variants). Prioritize flash_prefill_varlen_bwd and flash_chunked_prefill_fwd as they complete the core attention suite.

<details>
<summary>8/16 done - click to expand</summary>

| | Operator | Test | Bench | Ratio |
|:--|:---------|:----:|:-----:|------:|
| ✅ | flash_prefill_fwd | passed | passed | - |
| ✅ | flash_prefill_bwd | passed | passed | - |
| ✅ | flash_prefill_varlen_fwd | passed | passed | - |
| ⬜ | flash_prefill_varlen_bwd | missing | missing | - |
| ✅ | flash_decode_fwd | passed | passed | - |
| ✅ | flash_decode_paged_fwd | passed | passed | - |
| ⬜ | flash_decode_varlen_fwd | missing | missing | - |
| ⬜ | flash_chunked_prefill_fwd | missing | missing | - |
| ⬜ | mla_prefill_fwd | missing | missing | - |
| ⬜ | mla_prefill_bwd | missing | missing | - |
| ✅ | mla_decode_fwd | passed | passed | - |
| ⬜ | mla_decode_paged_fwd | missing | missing | - |
| ✅ | nsa_prefill_fwd | passed | passed | - |
| ⬜ | nsa_decode_fwd | missing | missing | - |
| ⬜ | dsa_prefill_fwd | missing | missing | - |
| ✅ | dsa_decode_fwd | passed | passed | - |

</details>

### MoE | Func: ⭐⭐⭐⭐⭐ (5/5)

| | Progress | |
|:--|:---------|:--|
| Impl | `██░░░░░░░░░░░░░` 1/6 | |
| Test | `██░░░░░░░░░░░░░` 1/6 | |
| Bench | `██░░░░░░░░░░░░░` 1/6 | |

> **Issues:** Only permute_align implemented. unpermute_depad and all 4 fused_moe variants (deepseek, glm, kimi, qwen) unimplemented.

> **Evaluation:** MoE support is minimal with only the permute utility op implemented. fused_moe_deepseek and fused_moe_qwen are high-priority targets given their prevalence in production MoE models. Implement unpermute_depad alongside fused_moe to complete the MoE pipeline.

<details>
<summary>1/6 done - click to expand</summary>

| | Operator | Test | Bench | Ratio |
|:--|:---------|:----:|:-----:|------:|
| ✅ | permute_align | passed | passed | - |
| ⬜ | unpermute_depad | missing | missing | - |
| ⬜ | fused_moe_deepseek | missing | missing | - |
| ⬜ | fused_moe_glm | missing | missing | - |
| ⬜ | fused_moe_kimi | missing | missing | - |
| ⬜ | fused_moe_qwen | missing | missing | - |

</details>

### Linear Attention | Func: ⭐⭐⭐⭐⭐ (5/5)

| | Progress | |
|:--|:---------|:--|
| Impl | `████████░░░░░░░` 4/8 | |
| Test | `████████░░░░░░░` 4/8 | |
| Bench | `████████░░░░░░░` 4/8 | |

> **Issues:** deltanet_chunkwise, deltanet_recurrence, retnet_chunkwise, retnet_recurrence unimplemented. Benchmark ratios not captured in structured log.

> **Evaluation:** Implemented Linear Attention ops (gated_deltanet and GLA variants) all pass tests. Complete the category by implementing deltanet and retnet variants. Ensure benchmark ratio reporting is enabled for this category.

<details>
<summary>4/8 done - click to expand</summary>

| | Operator | Test | Bench | Ratio |
|:--|:---------|:----:|:-----:|------:|
| ✅ | gated_deltanet_chunkwise | passed | passed | - |
| ✅ | gated_deltanet_recurrence | passed | passed | - |
| ⬜ | deltanet_chunkwise | missing | missing | - |
| ⬜ | deltanet_recurrence | missing | missing | - |
| ✅ | gla_chunkwise | passed | passed | - |
| ✅ | gla_recurrence | passed | passed | - |
| ⬜ | retnet_chunkwise | missing | missing | - |
| ⬜ | retnet_recurrence | missing | missing | - |

</details>

### SSM

| | Progress | |
|:--|:---------|:--|
| Impl | `░░░░░░░░░░░░░░░` 0/2 | |
| Test | `░░░░░░░░░░░░░░░` 0/2 | |
| Bench | `░░░░░░░░░░░░░░░` 0/2 | |

> **Issues:** Both mamba1 and mamba2 unimplemented.

> **Evaluation:** SSM category is entirely unimplemented. Mamba2 is more relevant for current state-space model research and should be prioritized. Consider implementing alongside SSM-adjacent linear attention ops to share infrastructure.

<details>
<summary>0/2 done - click to expand</summary>

| | Operator | Test | Bench | Ratio |
|:--|:---------|:----:|:-----:|------:|
| ⬜ | mamba1 | missing | missing | - |
| ⬜ | mamba2 | missing | missing | - |

</details>


---

<sub>Auto-generated by the [TileOPs Report](https://superanggao.github.io/TileOPs-report/nightly/) pipeline from [`op_registry.json`](scripts/op_registry.json)</sub>

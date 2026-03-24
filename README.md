<div align="center">

# TileOPs Operator Tracking

**96/186** operators complete (51%)

[View full HTML report](https://superanggao.github.io/TileOPs-report/nightly/)

Last updated: `2026-03-24T22:28:59Z`

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

> TileOPs shows strong maturity in foundational categories — Elementwise, Reduce, Norm, Flash Attention, and Linear Attention all demonstrate high functional correctness — but the project has significant coverage gaps in Conv & Pooling (0% implemented), Sampling (14% implemented), MoE (17% implemented), and Quantize (20% implemented) that limit its utility for production LLM deployment. Performance data quality is inconsistent: many categories report benchmark 'passed' without ratio values, making it impossible to detect regressions, while the Elementwise category reveals concerning underperformance (ratio=0.33) in boolean/bitwise ops that warrants investigation.

**Recommendations:**

1. Fix the 4 failing Elementwise tests (dropout, rope_neox, rope_non_neox, rope_llama31) and investigate the 0.33 performance ratio for bitwise_not, logical_and, and logical_or, as these suggest systematic correctness and performance issues in a high-usage operator group.
2. Standardize benchmark reporting to always emit ratio values instead of bare 'passed' status — Reduce, Norm, Flash Attention, Linear Attention, GEMM, Quantize, Sampling, and MoE categories currently have no quantitative performance data, making regression detection impossible.
3. Prioritize implementation of high-impact missing operator groups: (1) top_k/top_p/temperature_scale in Sampling for LLM text generation, (2) fused_moe_deepseek + unpermute_depad in MoE for MoE model inference, and (3) int8_per_tensor/int8_per_channel in Quantize for deployment-critical quantization support.


## Categories

### Elementwise | Func: ⭐⭐⭐⭐☆ (4/5) | Perf: ⭐⭐⭐⭐⭐ (5/5)

| | Progress | |
|:--|:---------|:--|
| Impl | `███████████████` 70/72 | |
| Test | `████████████░░░` 59/72 | |
| Bench | `████████████░░░` 58/72 | |

> **Issues:** dropout, rope_neox, rope_non_neox, rope_llama31 tests failing. bitwise_not, logical_and, logical_or underperforming at ratio=0.33. 9 ops missing tests (leaky_relu, elu, hardtanh, softplus, prelu, alibi, sinusoidal, yarn_rope, longrope). 11 ops missing benchmarks including where, clamp, masked_fill, nan_to_num, isnan, isinf, isfinite, alibi, sinusoidal, yarn_rope, longrope.

> **Evaluation:** Elementwise is the most mature category with strong coverage and mostly excellent performance. Fix the 4 failing RoPE/dropout tests as a priority, then investigate the 0.33 ratio underperformers (bitwise_not, logical_and, logical_or) which suggest a systematic issue with boolean/bitwise ops. Add benchmark coverage for the 7 special_elementwise ops (where, clamp, masked_fill, etc.).

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

> **Issues:** All 20 benchmarks report 'passed' status without ratio data, so quantitative performance scoring is unavailable.

> **Evaluation:** Reduce is functionally complete with 100% test pass rate. However, all benchmark entries show 'passed' without ratio values, meaning performance cannot be quantitatively assessed. Add ratio-based benchmark reporting for all reduce ops to enable proper performance tracking.

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

> **Issues:** qk_norm is missing both test and benchmark. All other benchmarks report 'passed' without ratio data.

> **Evaluation:** Norm is nearly complete with 9/10 ops fully passing. Implement and test qk_norm to close the gap. As with Reduce, benchmark ratio data is absent — add quantitative ratio reporting to enable performance regression detection.

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

> **Evaluation:** Conv & Pooling is a complete gap in the library. Prioritize implementing the most commonly used ops first: conv2d, depthwise_conv2d, max_pool2d, and avg_pool2d. These are foundational for CNN-based model support and should be scheduled for the next development sprint.

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

> **Issues:** Only 9/19 ops implemented. 16 ops missing tests including gemm_fp8, gemm_fp8_block_scaled, gemv_fp8, small_batch_gemm variants, and all lowbit/sparse/bmm ops. All benchmarks report 'passed' without ratio data.

> **Evaluation:** GEMM has a solid foundation with passing tests for gemm_fp16, gemv_fp16, and groupgemm_fp16, but coverage is critically low. Prioritize adding tests for the 6 already-implemented but untested ops (gemm_fp8, gemm_fp8_block_scaled, gemv_fp8, small_batch variants, groupgemm_fp8), then plan implementation of bmm and lowbit GEMM variants which are essential for LLM inference.

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

> **Issues:** Only 2/10 ops implemented (fp8_per_tensor, fp8_per_block). All int8, int4, nf4, and fp8_cast_transpose ops are unimplemented. fp8_per_block is implemented but missing tests.

> **Evaluation:** Quantize coverage is very limited, with only FP8 ops partially implemented. Add a test for fp8_per_block immediately since it is implemented but untested. Then prioritize int8_per_tensor and int8_per_channel as they are the most widely used quantization schemes for LLM deployment.

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

> **Issues:** Only chain_speculative_sampling implemented. top_k, top_p, min_p, top_k_top_p, temperature_scale, and sampling_from_probs are all unimplemented.

> **Evaluation:** Sampling is critically underdeveloped with only 1/7 ops implemented. top_k and top_p are essential for LLM text generation and should be the immediate implementation priority. temperature_scale is also straightforward and high-value.

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

> **Issues:** 8/16 ops unimplemented: flash_prefill_varlen_bwd, flash_decode_varlen_fwd, flash_chunked_prefill_fwd, mla_prefill_fwd/bwd, mla_decode_paged_fwd, nsa_decode_fwd, dsa_prefill_fwd. All benchmarks report 'passed' without ratio data.

> **Evaluation:** Flash Attention has excellent functional quality with 8/8 implemented ops passing all tests. The missing ops represent advanced features (varlen backward, chunked prefill, MLA variants). Add ratio-based benchmark reporting to detect performance regressions. Next implementation priority should be flash_chunked_prefill_fwd and flash_decode_varlen_fwd for broader model compatibility.

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

> **Issues:** Only permute_align implemented. unpermute_depad and all 4 fused_moe variants (deepseek, glm, kimi, qwen) are unimplemented.

> **Evaluation:** MoE support is minimal with only the permute utility implemented. The fused_moe variants are critical for MoE model inference performance. Prioritize implementing fused_moe_deepseek and unpermute_depad as they form the core MoE dispatch pipeline, then add the other model-specific variants.

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

> **Issues:** deltanet_chunkwise, deltanet_recurrence, retnet_chunkwise, retnet_recurrence are unimplemented. All benchmarks report 'passed' without ratio data.

> **Evaluation:** Linear Attention has solid coverage for gated_deltanet and GLA variants with 100% pass rate. Implement the base deltanet and retnet variants to complete the category. Add ratio-based benchmark reporting to enable performance tracking across all implemented ops.

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

> **Issues:** Both mamba1 and mamba2 are unimplemented with no tests or benchmarks.

> **Evaluation:** SSM is entirely unimplemented. Given the growing importance of Mamba-based architectures, schedule mamba2 implementation first as it is the more modern and widely adopted variant, followed by mamba1 for backward compatibility.

<details>
<summary>0/2 done - click to expand</summary>

| | Operator | Test | Bench | Ratio |
|:--|:---------|:----:|:-----:|------:|
| ⬜ | mamba1 | missing | missing | - |
| ⬜ | mamba2 | missing | missing | - |

</details>


---

<sub>Auto-generated by the [TileOPs Report](https://superanggao.github.io/TileOPs-report/nightly/) pipeline from [`op_registry.json`](scripts/op_registry.json)</sub>

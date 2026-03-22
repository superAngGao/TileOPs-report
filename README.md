<div align="center">

# TileOPs Operator Tracking

**94/186** operators complete (50%)

[View full HTML report](https://superanggao.github.io/TileOPs-report/nightly/)

Last updated: `2026-03-22T22:23:01Z`

</div>

---

## Overview

| | Count | Bar |
|:--|------:|:----|
| Implemented | **125**/186 | `██████████░░░░░` 125/186 |
| Test Passed | **106**/186 | `█████████░░░░░░` 106/186 |
| Bench Passed | **110**/186 | `█████████░░░░░░` 110/186 |
| **Done** | **94**/186 | `████████░░░░░░░` 94/186 |

### Status Breakdown

| Test | | Bench | |
|:-----|----:|:------|----:|
| Passed | 106 | Qualified (ratio >= 0.8) | 40 |
| Failed | 4 | Passed (no ratio) | 70 |
| Missing | 76 | Underperforming | 5 |
| | | Failed | 0 |
| | | Missing | 71 |

## Assessment

> TileOPs shows strong progress in core LLM operator categories — Elementwise, Reduce, Norm, Flash Attention, and Linear Attention are largely functional and performant, with 94/186 ops fully done. However, significant gaps remain: Conv & Pooling (0/16), SSM (0/2), Quantize (2/10), Sampling (1/7), and MoE (1/6) are severely underdeveloped, limiting the library's production readiness for diverse model architectures. The 4 failing Elementwise tests (dropout, rope variants) and 3 underperforming bitwise/logical ops are the most urgent correctness issues in otherwise healthy categories.

**Recommendations:**

1. Fix the 4 failing Elementwise tests (dropout, rope_neox, rope_non_neox, rope_llama31) and investigate the 3x performance regression in bitwise_not/logical_and/logical_or, as these are foundational ops used across many models.
2. Prioritize implementing the critical inference primitives missing from Quantize (int8_per_tensor, int8_per_channel), Sampling (top_k, top_p), and MoE (fused_moe_deepseek, unpermute_depad) to unblock end-to-end LLM deployment use cases.
3. Begin Conv & Pooling implementation starting with conv2d and avg_pool2d/max_pool2d, and add test mappings for the 6 implemented-but-untested GEMM ops (gemm_fp8, gemv_fp8, small_batch variants, groupgemm_fp8) to improve coverage visibility.


## Categories

### Elementwise | Func: ⭐⭐⭐⭐☆ (4/5) | Perf: ⭐⭐⭐⭐☆ (4/5)

| | Progress | |
|:--|:---------|:--|
| Impl | `███████████████` 70/72 | |
| Test | `████████████░░░` 59/72 | |
| Bench | `████████████░░░` 58/72 | |

> **Issues:** dropout, rope_neox, rope_non_neox, rope_llama31 tests failing. bitwise_not, logical_and, logical_or underperforming at ratio=0.33. 7 special_elementwise ops and alibi/sinusoidal lack bench coverage.

> **Evaluation:** Strong overall coverage with most ops passing and meeting performance targets. Fix the 4 failing test cases (dropout and rope variants) as a priority, then investigate the 3x performance gap in bitwise_not/logical_and/logical_or. Add benchmark mappings for the 7 special_elementwise ops (where, clamp, masked_fill, etc.).

<details>
<summary>49/72 done - click to expand</summary>

| | Operator | Test | Bench | Ratio |
|:--|:---------|:----:|:-----:|------:|
| ✅ | add | passed | qualified | 1.00 |
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
| ✅ | relu | passed | qualified | 1.00 |
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
| 🟦 | prelu | missing | qualified | 2.20 |
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
| ❌ | dropout | failed | qualified | 1.60 |
| ❌ | rope_neox | failed | passed | - |
| ❌ | rope_non_neox | failed | passed | - |
| ❌ | rope_llama31 | failed | passed | - |
| ⬜ | yarn_rope | missing | missing | - |
| ⬜ | longrope | missing | missing | - |
| 🟦 | alibi | missing | missing | - |
| 🟦 | sinusoidal | missing | missing | - |

</details>

### Reduce | Func: ⭐⭐⭐⭐⭐ (5/5) | Perf: ⭐⭐⭐⭐☆ (4/5)

| | Progress | |
|:--|:---------|:--|
| Impl | `███████████████` 20/20 | |
| Test | `███████████████` 20/20 | |
| Bench | `██████████████░` 19/20 | |

> **Issues:** log_softmax underperforming at ratio=0.67. 17 ops have bench=passed (no ratio data) so full performance picture is incomplete.

> **Evaluation:** Perfect functional correctness across all 20 ops. log_softmax is the only confirmed underperformer; investigate kernel tuning. The majority of reduce ops report bench=passed without a ratio, so adding ratio tracking would improve visibility into performance regressions.

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

### Norm | Func: ⭐⭐⭐⭐⭐ (5/5) | Perf: ⭐⭐⭐⭐⭐ (5/5)

| | Progress | |
|:--|:---------|:--|
| Impl | `███████████████` 10/10 | |
| Test | `██████████████░` 9/10 | |
| Bench | `████████████░░░` 8/10 | |

> **Issues:** layer_norm slightly underperforming at ratio=0.78. qk_norm has no test or bench mapping.

> **Evaluation:** Excellent category with strong performance (rms_norm at 8x, fused_add_rmsnorm at 7.57x). Address layer_norm performance gap with kernel tuning. Implement and add test/bench mappings for qk_norm to complete the category.

<details>
<summary>8/10 done - click to expand</summary>

| | Operator | Test | Bench | Ratio |
|:--|:---------|:----:|:-----:|------:|
| 🟡 | layer_norm | passed | underperforming | 0.78 |
| ✅ | batch_norm | passed | passed | - |
| ✅ | group_norm | passed | qualified | 1.00 |
| ✅ | instance_norm | passed | qualified | 1.17 |
| ✅ | rms_norm | passed | qualified | 8.00 |
| 🟦 | qk_norm | missing | missing | - |
| ✅ | ada_layer_norm | passed | qualified | 1.15 |
| ✅ | ada_layer_norm_zero | passed | qualified | 1.33 |
| ✅ | fused_add_layer_norm | passed | qualified | 2.47 |
| ✅ | fused_add_rmsnorm | passed | qualified | 7.57 |

</details>

### Conv & Pooling

| | Progress | |
|:--|:---------|:--|
| Impl | `░░░░░░░░░░░░░░░` 0/16 | |
| Test | `░░░░░░░░░░░░░░░` 0/16 | |
| Bench | `░░░░░░░░░░░░░░░` 0/16 | |

> **Issues:** Entire category unimplemented — 0/16 ops have any implementation, tests, or benchmarks.

> **Evaluation:** This is the largest gap in the library. Prioritize at least conv2d and max_pool2d/avg_pool2d as foundational ops. Consider a phased plan: implement basic 2D conv and pooling first, then extend to 1D/3D and specialized variants.

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

> **Issues:** Only 9/19 ops implemented. 6 implemented ops (gemm_fp8, gemm_fp8_block_scaled, gemv_fp8, small_batch_gemm variants, groupgemm_fp8) lack test mappings. All 9 benched ops report bench=passed without ratio data. 10 ops entirely unimplemented including lowbit_gemm variants and sparse_gemm.

> **Evaluation:** The implemented GEMM ops pass their tests but coverage is thin. Add test mappings for the 6 implemented-but-untested ops immediately. Prioritize implementing bmm_fp16/fp8 and w4a16/w8a8 lowbit variants as they are critical for LLM inference workloads.

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

> **Issues:** Only 2/10 ops implemented (fp8_per_tensor, fp8_per_block). All int8 and int4 quantization ops are unimplemented. fp8_per_block lacks a test mapping.

> **Evaluation:** Severely underdeveloped category critical for LLM deployment. fp8_per_tensor passes its test and bench. Add test mapping for fp8_per_block, then prioritize implementing int8_per_tensor and int8_per_channel as the most commonly needed quantization ops.

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

### Sampling | Func: ⭐⭐⭐⭐⭐ (5/5) | Perf: ⭐⭐⭐⭐⭐ (5/5)

| | Progress | |
|:--|:---------|:--|
| Impl | `██░░░░░░░░░░░░░` 1/7 | |
| Test | `██░░░░░░░░░░░░░` 1/7 | |
| Bench | `██░░░░░░░░░░░░░` 1/7 | |

> **Issues:** Only 1/7 ops implemented. top_k, top_p, min_p, top_k_top_p, temperature_scale, sampling_from_probs all unimplemented.

> **Evaluation:** chain_speculative_sampling is well-implemented (2.16x baseline) but the category is nearly empty. Implement top_k and top_p as the highest-priority sampling primitives since they are required for most LLM decoding strategies.

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
| ✅ | chain_speculative_sampling | passed | qualified | 2.16 |

</details>

### Flash Attention | Func: ⭐⭐⭐⭐⭐ (5/5) | Perf: ⭐⭐⭐⭐⭐ (5/5)

| | Progress | |
|:--|:---------|:--|
| Impl | `████████░░░░░░░` 8/16 | |
| Test | `████████░░░░░░░` 8/16 | |
| Bench | `████████░░░░░░░` 8/16 | |

> **Issues:** 8/16 ops unimplemented including flash_prefill_varlen_bwd, flash_decode_varlen_fwd, flash_chunked_prefill_fwd, all MLA prefill/paged variants, nsa_decode_fwd, and dsa_prefill_fwd. 5 benched ops report bench=passed without ratio.

> **Evaluation:** All implemented ops pass tests and benchmarks exceed baseline (up to 4.14x for mla_decode_fwd). Focus next on flash_prefill_varlen_bwd and flash_decode_varlen_fwd to complete variable-length attention support, then mla_decode_paged_fwd for production paged KV-cache use cases.

<details>
<summary>8/16 done - click to expand</summary>

| | Operator | Test | Bench | Ratio |
|:--|:---------|:----:|:-----:|------:|
| ✅ | flash_prefill_fwd | passed | passed | - |
| ✅ | flash_prefill_bwd | passed | passed | - |
| ✅ | flash_prefill_varlen_fwd | passed | passed | - |
| ⬜ | flash_prefill_varlen_bwd | missing | missing | - |
| ✅ | flash_decode_fwd | passed | qualified | 1.15 |
| ✅ | flash_decode_paged_fwd | passed | qualified | 2.38 |
| ⬜ | flash_decode_varlen_fwd | missing | missing | - |
| ⬜ | flash_chunked_prefill_fwd | missing | missing | - |
| ⬜ | mla_prefill_fwd | missing | missing | - |
| ⬜ | mla_prefill_bwd | missing | missing | - |
| ✅ | mla_decode_fwd | passed | qualified | 4.14 |
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

> **Issues:** Only 1/6 ops implemented (permute_align). All fused_moe variants (deepseek, glm, kimi, qwen) and unpermute_depad are unimplemented.

> **Evaluation:** permute_align passes its test and bench but the category is critically incomplete for MoE model support. Implement fused_moe_deepseek as the highest priority since DeepSeek-style MoE is widely deployed, followed by unpermute_depad to complete the permute/unpermute pair.

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

### Linear Attention | Func: ⭐⭐⭐⭐⭐ (5/5) | Perf: ⭐⭐⭐⭐⭐ (5/5)

| | Progress | |
|:--|:---------|:--|
| Impl | `████████░░░░░░░` 4/8 | |
| Test | `████████░░░░░░░` 4/8 | |
| Bench | `████████░░░░░░░` 4/8 | |

> **Issues:** deltanet and retnet variants (4 ops) are unimplemented. 2 benched ops report bench=passed without ratio.

> **Evaluation:** All implemented ops pass tests and recurrence variants show strong performance (gated_deltanet_recurrence 4.47x, gla_recurrence 3.61x). Implement deltanet_chunkwise and deltanet_recurrence next to complete the deltanet family, then add ratio tracking to the bench=passed ops.

<details>
<summary>4/8 done - click to expand</summary>

| | Operator | Test | Bench | Ratio |
|:--|:---------|:----:|:-----:|------:|
| ✅ | gated_deltanet_chunkwise | passed | passed | - |
| ✅ | gated_deltanet_recurrence | passed | qualified | 4.47 |
| ⬜ | deltanet_chunkwise | missing | missing | - |
| ⬜ | deltanet_recurrence | missing | missing | - |
| ✅ | gla_chunkwise | passed | passed | - |
| ✅ | gla_recurrence | passed | qualified | 3.61 |
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

> **Evaluation:** Category is entirely unimplemented. Given Mamba's relevance to efficient sequence modeling, implement mamba2 first as it is the more modern and widely adopted variant, then backport to mamba1.

<details>
<summary>0/2 done - click to expand</summary>

| | Operator | Test | Bench | Ratio |
|:--|:---------|:----:|:-----:|------:|
| ⬜ | mamba1 | missing | missing | - |
| ⬜ | mamba2 | missing | missing | - |

</details>


---

<sub>Auto-generated by the [TileOPs Report](https://superanggao.github.io/TileOPs-report/nightly/) pipeline from [`op_registry.json`](scripts/op_registry.json)</sub>

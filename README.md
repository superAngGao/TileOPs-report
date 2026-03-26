<div align="center">

# TileOPs Operator Tracking

**96/186** operators complete (51%)

[View full HTML report](https://superanggao.github.io/TileOPs-report/nightly/)

Last updated: `2026-03-26T22:27:13Z`

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

> TileOPs shows strong functional correctness in implemented categories (106/110 tests passing, 96% pass rate) but has significant coverage gaps. Three categories (Conv & Pooling, SSM, Sampling) are critically underdeveloped. Performance validation is incomplete with most benchmarks showing 'passed' status but lacking ratio data for analysis.

**Recommendations:**

1. Fix 4 failing tests in Elementwise (dropout, rope variants) and investigate 3 underperforming logical/bitwise ops
2. Implement missing core operators: conv2d/pooling (0/16), int8/int4 quantization (2/10), top_k/top_p sampling (1/7), and mamba SSM (0/2)
3. Add comprehensive benchmark mappings for Reduce, Norm, Flash Attention, and GEMM categories to validate performance claims and identify optimization opportunities


## Categories

### Elementwise | Func: ⭐⭐⭐⭐☆ (4/5) | Perf: ⭐⭐⭐⭐☆ (4/5)

| | Progress | |
|:--|:---------|:--|
| Impl | `███████████████` 70/72 | |
| Test | `████████████░░░` 59/72 | |
| Bench | `████████████░░░` 58/72 | |

> **Issues:** 4 test failures (dropout, rope variants), 3 underperforming ops (bitwise_not, logical_and, logical_or at 0.33 ratio), 11 ops missing benchmarks

> **Evaluation:** Strong category with 82% test pass rate and most benchmarks qualified. Fix rope and dropout test failures, investigate logical/bitwise underperformance, add missing benchmarks for special_elementwise ops.

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

> **Evaluation:** Perfect functional correctness with all 20 ops passing tests. No benchmark data available despite implementations existing. Add benchmark mappings to validate performance.

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

> **Issues:** qk_norm missing test and benchmark mapping

> **Evaluation:** Excellent functional status with 90% coverage. Add test and benchmark mapping for qk_norm to complete the category.

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

> **Issues:** Entire category unimplemented (0/16 ops)

> **Evaluation:** Critical gap with zero implementation. Prioritize conv2d, max_pool2d, and avg_pool2d as foundational ops for CNN workloads.

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

### GEMM | Func: ⭐⭐⭐☆☆ (3/5)

| | Progress | |
|:--|:---------|:--|
| Impl | `███████░░░░░░░░` 9/19 | |
| Test | `██░░░░░░░░░░░░░` 3/19 | |
| Bench | `███████░░░░░░░░` 9/19 | |

> **Issues:** Only 3/19 ops tested, 10/19 missing implementations, 6 implemented ops lack test mappings

> **Evaluation:** Mixed progress with core GEMM ops implemented but undertested. Add test mappings for fp8 variants and small_batch ops, implement lowbit quantized GEMM for inference optimization.

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

### Quantize | Func: ⭐⭐☆☆☆ (2/5)

| | Progress | |
|:--|:---------|:--|
| Impl | `███░░░░░░░░░░░░` 2/10 | |
| Test | `██░░░░░░░░░░░░░` 1/10 | |
| Bench | `███░░░░░░░░░░░░` 2/10 | |

> **Issues:** Only 2/10 ops implemented, 9/10 missing tests, critical int8/int4 quantization absent

> **Evaluation:** Severely underdeveloped category. Implement int8_per_tensor and int4_per_block as priorities for model compression and inference acceleration.

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

### Sampling | Func: ⭐⭐☆☆☆ (2/5)

| | Progress | |
|:--|:---------|:--|
| Impl | `██░░░░░░░░░░░░░` 1/7 | |
| Test | `██░░░░░░░░░░░░░` 1/7 | |
| Bench | `██░░░░░░░░░░░░░` 1/7 | |

> **Issues:** Only 1/7 ops implemented (chain_speculative_sampling), core sampling ops missing

> **Evaluation:** Critical gap for LLM inference. Implement top_k, top_p, and temperature_scale as essential decoding primitives.

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

### Flash Attention | Func: ⭐⭐⭐⭐☆ (4/5)

| | Progress | |
|:--|:---------|:--|
| Impl | `████████░░░░░░░` 8/16 | |
| Test | `████████░░░░░░░` 8/16 | |
| Bench | `████████░░░░░░░` 8/16 | |

> **Issues:** 8/16 ops missing (varlen_bwd, chunked_prefill, MLA/NSA/DSA decode variants)

> **Evaluation:** Solid foundation with core prefill/decode ops working. Complete varlen and paged variants for production serving, add benchmark mappings to validate performance claims.

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

### MoE | Func: ⭐⭐☆☆☆ (2/5)

| | Progress | |
|:--|:---------|:--|
| Impl | `██░░░░░░░░░░░░░` 1/6 | |
| Test | `██░░░░░░░░░░░░░` 1/6 | |
| Bench | `██░░░░░░░░░░░░░` 1/6 | |

> **Issues:** Only permute_align implemented, all fused_moe variants missing

> **Evaluation:** Minimal coverage for critical MoE workloads. Implement fused_moe_deepseek and fused_moe_qwen to support popular MoE architectures.

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

### Linear Attention | Func: ⭐⭐⭐⭐☆ (4/5)

| | Progress | |
|:--|:---------|:--|
| Impl | `████████░░░░░░░` 4/8 | |
| Test | `████████░░░░░░░` 4/8 | |
| Bench | `████████░░░░░░░` 4/8 | |

> **Issues:** 4/8 ops missing (deltanet and retnet variants)

> **Evaluation:** Good progress on gated_deltanet and GLA. Complete deltanet and retnet implementations for broader linear attention architecture support.

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

> **Issues:** Entire category unimplemented (0/2 ops)

> **Evaluation:** Complete gap for state space models. Implement mamba1 and mamba2 to support efficient sequence modeling architectures.

<details>
<summary>0/2 done - click to expand</summary>

| | Operator | Test | Bench | Ratio |
|:--|:---------|:----:|:-----:|------:|
| ⬜ | mamba1 | missing | missing | - |
| ⬜ | mamba2 | missing | missing | - |

</details>


---

<sub>Auto-generated by the [TileOPs Report](https://superanggao.github.io/TileOPs-report/nightly/) pipeline from [`op_registry.json`](scripts/op_registry.json)</sub>

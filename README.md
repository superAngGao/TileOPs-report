<div align="center">

# TileOPs Operator Tracking

**99/186** operators complete (53%)

[View full HTML report](https://superanggao.github.io/TileOPs-report/nightly/)

Last updated: `2026-03-28T22:26:58Z`

</div>

---

## Overview

| | Count | Bar |
|:--|------:|:----|
| Implemented | **125**/186 | `██████████░░░░░` 125/186 |
| Test Passed | **106**/186 | `█████████░░░░░░` 106/186 |
| Bench Passed | **115**/186 | `█████████░░░░░░` 115/186 |
| **Done** | **99**/186 | `████████░░░░░░░` 99/186 |

### Status Breakdown

| Test | | Bench | |
|:-----|----:|:------|----:|
| Passed | 106 | Qualified (ratio >= 0.8) | 0 |
| Failed | 4 | Passed (no ratio) | 115 |
| Missing | 76 | Underperforming | 0 |
| | | Failed | 0 |
| | | Missing | 71 |

## Assessment

> Analysis unavailable.


## Categories

### Elementwise

| | Progress | |
|:--|:---------|:--|
| Impl | `███████████████` 70/72 | |
| Test | `████████████░░░` 59/72 | |
| Bench | `█████████████░░` 61/72 | |

<details>
<summary>52/72 done - click to expand</summary>

| | Operator | Test | Bench | Ratio |
|:--|:---------|:----:|:-----:|------:|
| ✅ | add | passed | passed | - |
| ✅ | sub | passed | passed | - |
| ✅ | mul | passed | passed | - |
| ✅ | div | passed | passed | - |
| ✅ | remainder | passed | passed | - |
| ✅ | pow | passed | passed | - |
| ✅ | floor_divide | passed | passed | - |
| ✅ | lerp | passed | passed | - |
| ✅ | maximum | passed | passed | - |
| ✅ | minimum | passed | passed | - |
| ✅ | exp | passed | passed | - |
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
| ✅ | gelu | passed | passed | - |
| ✅ | silu | passed | passed | - |
| ✅ | sigmoid | passed | passed | - |
| ✅ | tanh | passed | passed | - |
| 🟦 | leaky_relu | missing | passed | - |
| 🟦 | elu | missing | passed | - |
| ✅ | selu | passed | passed | - |
| ✅ | hardswish | passed | passed | - |
| ✅ | hardsigmoid | passed | passed | - |
| 🟦 | hardtanh | missing | passed | - |
| 🟦 | softplus | missing | passed | - |
| ✅ | mish | passed | passed | - |
| 🟦 | prelu | missing | passed | - |
| ✅ | silu_and_mul | passed | passed | - |
| ✅ | gelu_and_mul | passed | passed | - |
| ✅ | gelu_tanh_and_mul | passed | passed | - |
| ✅ | eq | passed | passed | - |
| ✅ | ne | passed | passed | - |
| ✅ | gt | passed | passed | - |
| ✅ | lt | passed | passed | - |
| ✅ | ge | passed | passed | - |
| ✅ | le | passed | passed | - |
| ✅ | bitwise_and | passed | passed | - |
| ✅ | bitwise_or | passed | passed | - |
| ✅ | bitwise_xor | passed | passed | - |
| ✅ | bitwise_not | passed | passed | - |
| ✅ | logical_not | passed | passed | - |
| ✅ | logical_and | passed | passed | - |
| ✅ | logical_or | passed | passed | - |
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

### Reduce

| | Progress | |
|:--|:---------|:--|
| Impl | `███████████████` 20/20 | |
| Test | `███████████████` 20/20 | |
| Bench | `███████████████` 20/20 | |

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

### Norm

| | Progress | |
|:--|:---------|:--|
| Impl | `███████████████` 10/10 | |
| Test | `██████████████░` 9/10 | |
| Bench | `██████████████░` 9/10 | |

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

### GEMM

| | Progress | |
|:--|:---------|:--|
| Impl | `███████░░░░░░░░` 9/19 | |
| Test | `██░░░░░░░░░░░░░` 3/19 | |
| Bench | `███████░░░░░░░░` 9/19 | |

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

### Quantize

| | Progress | |
|:--|:---------|:--|
| Impl | `███░░░░░░░░░░░░` 2/10 | |
| Test | `██░░░░░░░░░░░░░` 1/10 | |
| Bench | `███░░░░░░░░░░░░` 2/10 | |

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

### Sampling

| | Progress | |
|:--|:---------|:--|
| Impl | `██░░░░░░░░░░░░░` 1/7 | |
| Test | `██░░░░░░░░░░░░░` 1/7 | |
| Bench | `██░░░░░░░░░░░░░` 1/7 | |

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

### Flash Attention

| | Progress | |
|:--|:---------|:--|
| Impl | `████████░░░░░░░` 8/16 | |
| Test | `████████░░░░░░░` 8/16 | |
| Bench | `████████░░░░░░░` 8/16 | |

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

### MoE

| | Progress | |
|:--|:---------|:--|
| Impl | `██░░░░░░░░░░░░░` 1/6 | |
| Test | `██░░░░░░░░░░░░░` 1/6 | |
| Bench | `██░░░░░░░░░░░░░` 1/6 | |

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

### Linear Attention

| | Progress | |
|:--|:---------|:--|
| Impl | `████████░░░░░░░` 4/8 | |
| Test | `████████░░░░░░░` 4/8 | |
| Bench | `████████░░░░░░░` 4/8 | |

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

<details>
<summary>0/2 done - click to expand</summary>

| | Operator | Test | Bench | Ratio |
|:--|:---------|:----:|:-----:|------:|
| ⬜ | mamba1 | missing | missing | - |
| ⬜ | mamba2 | missing | missing | - |

</details>


---

<sub>Auto-generated by the [TileOPs Report](https://superanggao.github.io/TileOPs-report/nightly/) pipeline from [`op_registry.json`](scripts/op_registry.json)</sub>

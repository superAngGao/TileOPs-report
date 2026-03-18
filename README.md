# TileOPs Nightly Report

> Auto-generated from [`op_registry.json`](scripts/op_registry.json) | Last updated: 2026-03-18T11:08:28Z

[View full HTML report](https://superanggao.github.io/TileOPs-report/nightly/)

## Overall Progress

| Total | Implemented | Test Passed | Bench Passed | Done |
|:-----:|:-----------:|:-----------:|:------------:|:----:|
| 186 | 110 | 82 | 93 | **73** (39%) |

### Test Status

| Passed | Failed | Missing |
|:------:|:------:|:-------:|
| 82 | 7 | 97 |

### Benchmark Status

| Qualified | Passed (no ratio) | Underperforming | Failed | Missing |
|:---------:|:-----------------:|:---------------:|:------:|:-------:|
| 31 | 62 | 4 | 3 | 86 |

## Category Progress

### Elementwise

| | Progress | Count |
|:--|:---------|------:|
| Impl | `###################.` 97% | 70/72 |
| Test | `###############.....` 76% | 55/72 |
| Bench | `################....` 80% | 58/72 |
| **Done** | | **49/72** |

<details>
<summary>Operators (49/72 done)</summary>

| Operator | Impl | Test | Bench |
|:---------|:----:|:----:|:-----:|
| add | Y | Y | Y |
| sub | Y | Y | Y |
| mul | Y | Y | Y |
| div | Y | Y | Y |
| remainder | Y | Y | Y |
| pow | Y | Y | Y |
| floor_divide | Y | Y | Y |
| lerp | Y | Y | Y |
| maximum | Y | Y | Y |
| minimum | Y | Y | Y |
| exp | Y | Y | Y |
| log | Y | Y | Y |
| sqrt | Y | Y | Y |
| rsqrt | Y | Y | Y |
| abs | Y | Y | Y |
| neg | Y | Y | Y |
| reciprocal | Y | Y | Y |
| sign | Y | Y | Y |
| sin | Y | Y | Y |
| cos | Y | Y | Y |
| floor | Y | Y | Y |
| ceil | Y | Y | Y |
| round | Y | Y | Y |
| trunc | Y | Y | Y |
| erf | Y | Y | Y |
| log1p | Y | Y | Y |
| expm1 | Y | Y | Y |
| relu | Y | Y | Y |
| gelu | Y | Y | Y |
| silu | Y | Y | Y |
| sigmoid | Y | Y | Y |
| tanh | Y | Y | Y |
| leaky_relu | Y | - | Y |
| elu | Y | - | Y |
| selu | Y | Y | Y |
| hardswish | Y | Y | Y |
| hardsigmoid | Y | Y | Y |
| hardtanh | Y | - | Y |
| softplus | Y | - | Y |
| mish | Y | Y | Y |
| prelu | Y | - | Y |
| silu_and_mul | Y | Y | Y |
| gelu_and_mul | Y | Y | Y |
| gelu_tanh_and_mul | Y | Y | Y |
| eq | Y | Y | Y |
| ne | Y | Y | Y |
| gt | Y | Y | Y |
| lt | Y | Y | Y |
| ge | Y | Y | Y |
| le | Y | Y | Y |
| bitwise_and | Y | Y | Y |
| bitwise_or | Y | Y | Y |
| bitwise_xor | Y | Y | Y |
| bitwise_not | Y | Y | FAIL |
| logical_not | Y | Y | Y |
| logical_and | Y | Y | FAIL |
| logical_or | Y | Y | FAIL |
| where | Y | - | - |
| clamp | Y | - | - |
| masked_fill | Y | - | - |
| nan_to_num | Y | - | - |
| isnan | Y | Y | - |
| isinf | Y | Y | - |
| isfinite | Y | Y | - |
| dropout | Y | FAIL | Y |
| rope_neox | Y | FAIL | Y |
| rope_non_neox | Y | FAIL | Y |
| rope_llama31 | Y | FAIL | Y |
| yarn_rope | - | - | - |
| longrope | - | - | - |
| alibi | Y | - | - |
| sinusoidal | Y | - | - |

</details>

### Reduce

| | Progress | Count |
|:--|:---------|------:|
| Impl | `###.................` 15% | 3/20 |
| Test | `....................` 0% | 0/20 |
| Bench | `###.................` 15% | 3/20 |
| **Done** | | **0/20** |

<details>
<summary>Operators (0/20 done)</summary>

| Operator | Impl | Test | Bench |
|:---------|:----:|:----:|:-----:|
| sum | - | - | - |
| mean | - | - | - |
| amin | - | - | - |
| amax | - | - | - |
| prod | - | - | - |
| std | - | - | - |
| var | - | - | - |
| var_mean | - | - | - |
| softmax | - | - | - |
| log_softmax | - | - | - |
| logsumexp | - | - | - |
| argmax | - | - | - |
| argmin | - | - | - |
| cumsum | - | - | - |
| cumprod | - | - | - |
| any | - | - | - |
| all | - | - | - |
| l1_norm | Y | FAIL | Y |
| l2_norm | Y | FAIL | Y |
| inf_norm | Y | FAIL | Y |

</details>

### Norm

| | Progress | Count |
|:--|:---------|------:|
| Impl | `####################` 100% | 10/10 |
| Test | `##################..` 90% | 9/10 |
| Bench | `##################..` 90% | 9/10 |
| **Done** | | **9/10** |

<details>
<summary>Operators (9/10 done)</summary>

| Operator | Impl | Test | Bench |
|:---------|:----:|:----:|:-----:|
| layer_norm | Y | Y | Y |
| batch_norm | Y | Y | Y |
| group_norm | Y | Y | Y |
| instance_norm | Y | Y | Y |
| rms_norm | Y | Y | Y |
| qk_norm | Y | - | - |
| ada_layer_norm | Y | Y | Y |
| ada_layer_norm_zero | Y | Y | Y |
| fused_add_layer_norm | Y | Y | Y |
| fused_add_rmsnorm | Y | Y | Y |

</details>

### Conv & Pooling

| | Progress | Count |
|:--|:---------|------:|
| Impl | `....................` 0% | 0/16 |
| Test | `....................` 0% | 0/16 |
| Bench | `....................` 0% | 0/16 |
| **Done** | | **0/16** |

<details>
<summary>Operators (0/16 done)</summary>

| Operator | Impl | Test | Bench |
|:---------|:----:|:----:|:-----:|
| conv1d | - | - | - |
| conv2d | - | - | - |
| conv3d | - | - | - |
| conv_transpose1d | - | - | - |
| conv_transpose2d | - | - | - |
| depthwise_conv2d | - | - | - |
| grouped_conv2d | - | - | - |
| dilated_conv2d | - | - | - |
| max_pool1d | - | - | - |
| max_pool2d | - | - | - |
| max_pool3d | - | - | - |
| avg_pool1d | - | - | - |
| avg_pool2d | - | - | - |
| avg_pool3d | - | - | - |
| adaptive_avg_pool2d | - | - | - |
| adaptive_max_pool2d | - | - | - |

</details>

### GEMM

| | Progress | Count |
|:--|:---------|------:|
| Impl | `############........` 57% | 11/19 |
| Test | `###.................` 15% | 3/19 |
| Bench | `############........` 57% | 11/19 |
| **Done** | | **3/19** |

<details>
<summary>Operators (3/19 done)</summary>

| Operator | Impl | Test | Bench |
|:---------|:----:|:----:|:-----:|
| gemm_fp16 | Y | Y | Y |
| gemm_fp8 | Y | - | Y |
| gemm_fp8_block_scaled | Y | - | Y |
| gemv_fp16 | Y | Y | Y |
| gemv_fp8 | Y | - | Y |
| small_batch_gemm_fp16 | Y | - | Y |
| small_batch_gemm_fp8 | Y | - | Y |
| bmm_fp16 | Y | - | Y |
| bmm_fp8 | Y | - | Y |
| groupgemm_fp16 | Y | Y | Y |
| groupgemm_fp8 | Y | - | Y |
| outer | - | - | - |
| w4a16 | - | - | - |
| w8a8 | - | - | - |
| w8a8_int8 | - | - | - |
| weight_only_int4 | - | - | - |
| fp4 | - | - | - |
| sparse_gemm_fp16 | - | - | - |
| sparse_gemm_fp8 | - | - | - |

</details>

### Quantize

| | Progress | Count |
|:--|:---------|------:|
| Impl | `####................` 20% | 2/10 |
| Test | `##..................` 10% | 1/10 |
| Bench | `....................` 0% | 0/10 |
| **Done** | | **0/10** |

<details>
<summary>Operators (0/10 done)</summary>

| Operator | Impl | Test | Bench |
|:---------|:----:|:----:|:-----:|
| int8_per_tensor | - | - | - |
| int8_per_channel | - | - | - |
| int8_per_block | - | - | - |
| smooth_quant | - | - | - |
| int4_per_channel | - | - | - |
| int4_per_block | - | - | - |
| nf4 | - | - | - |
| fp8_per_tensor | Y | Y | FAIL |
| fp8_per_block | Y | - | FAIL |
| fp8_cast_transpose | - | - | - |

</details>

### Sampling

| | Progress | Count |
|:--|:---------|------:|
| Impl | `###.................` 14% | 1/7 |
| Test | `###.................` 14% | 1/7 |
| Bench | `....................` 0% | 0/7 |
| **Done** | | **0/7** |

<details>
<summary>Operators (0/7 done)</summary>

| Operator | Impl | Test | Bench |
|:---------|:----:|:----:|:-----:|
| top_k | - | - | - |
| top_p | - | - | - |
| min_p | - | - | - |
| top_k_top_p | - | - | - |
| temperature_scale | - | - | - |
| sampling_from_probs | - | - | - |
| chain_speculative_sampling | Y | Y | FAIL |

</details>

### Flash Attention

| | Progress | Count |
|:--|:---------|------:|
| Impl | `##########..........` 50% | 8/16 |
| Test | `##########..........` 50% | 8/16 |
| Bench | `##########..........` 50% | 8/16 |
| **Done** | | **8/16** |

<details>
<summary>Operators (8/16 done)</summary>

| Operator | Impl | Test | Bench |
|:---------|:----:|:----:|:-----:|
| flash_prefill_fwd | Y | Y | Y |
| flash_prefill_bwd | Y | Y | Y |
| flash_prefill_varlen_fwd | Y | Y | Y |
| flash_prefill_varlen_bwd | - | - | - |
| flash_decode_fwd | Y | Y | Y |
| flash_decode_paged_fwd | Y | Y | Y |
| flash_decode_varlen_fwd | - | - | - |
| flash_chunked_prefill_fwd | - | - | - |
| mla_prefill_fwd | - | - | - |
| mla_prefill_bwd | - | - | - |
| mla_decode_fwd | Y | Y | Y |
| mla_decode_paged_fwd | - | - | - |
| nsa_prefill_fwd | Y | Y | Y |
| nsa_decode_fwd | - | - | - |
| dsa_prefill_fwd | - | - | - |
| dsa_decode_fwd | Y | Y | Y |

</details>

### MoE

| | Progress | Count |
|:--|:---------|------:|
| Impl | `###.................` 16% | 1/6 |
| Test | `###.................` 16% | 1/6 |
| Bench | `###.................` 16% | 1/6 |
| **Done** | | **1/6** |

<details>
<summary>Operators (1/6 done)</summary>

| Operator | Impl | Test | Bench |
|:---------|:----:|:----:|:-----:|
| permute_align | Y | Y | Y |
| unpermute_depad | - | - | - |
| fused_moe_deepseek | - | - | - |
| fused_moe_glm | - | - | - |
| fused_moe_kimi | - | - | - |
| fused_moe_qwen | - | - | - |

</details>

### Linear Attention

| | Progress | Count |
|:--|:---------|------:|
| Impl | `##########..........` 50% | 4/8 |
| Test | `##########..........` 50% | 4/8 |
| Bench | `########............` 37% | 3/8 |
| **Done** | | **3/8** |

<details>
<summary>Operators (3/8 done)</summary>

| Operator | Impl | Test | Bench |
|:---------|:----:|:----:|:-----:|
| gated_deltanet_chunkwise | Y | Y | FAIL |
| gated_deltanet_recurrence | Y | Y | Y |
| deltanet_chunkwise | - | - | - |
| deltanet_recurrence | - | - | - |
| gla_chunkwise | Y | Y | Y |
| gla_recurrence | Y | Y | Y |
| retnet_chunkwise | - | - | - |
| retnet_recurrence | - | - | - |

</details>

### SSM

| | Progress | Count |
|:--|:---------|------:|
| Impl | `....................` 0% | 0/2 |
| Test | `....................` 0% | 0/2 |
| Bench | `....................` 0% | 0/2 |
| **Done** | | **0/2** |

<details>
<summary>Operators (0/2 done)</summary>

| Operator | Impl | Test | Bench |
|:---------|:----:|:----:|:-----:|
| mamba1 | - | - | - |
| mamba2 | - | - | - |

</details>

---

*Generated by [TileOPs Report](https://superanggao.github.io/TileOPs-report/nightly/) pipeline*
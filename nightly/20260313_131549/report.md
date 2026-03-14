# TileOPs 夜测综合报告

- **日期**: 20260313_131549
- **Commit**: `4d65362c5878dc31c77fcc7cc3c7932026c97f04`
- **生成时间**: 2026-03-13 13:15:50

## 总体状态

| 项目 | 状态 |
|------|------|
| 测试（pytest） | ❌ FAIL |
| Benchmark | ❌ FAIL（或已跳过） |
| **综合** | **❌ FAIL** |

## 开发进度

> 数据来源：[issue #407](https://github.com/tile-ai/TileOPs/issues/407)  
> 更新时间：2026-03-13T13:15:50Z

**总进度：0 / 186 ops 已完成（0.0%）**  （已实现: 102，已通过测试: 0）

| # | Category | Ops | Difficulty | Issue | 状态 |
|---|---|---|---|---|---|
| 1 | Elementwise | 72 | ⭐ | #397 | 🔧 In Progress (0/72) |
| 2 | Reduce | 20 | ⭐ | #398 | 🔧 In Progress (0/20) |
| 3 | Norm | 10 | ⭐ | #399 | 🔧 In Progress (0/10) |
| 4 | Conv & Pooling | 16 | ⭐ | #402 | 🔲 Not Started |
| 5 | GEMM | 19 | ⭐⭐ | #400 | 🔧 In Progress (0/19) |
| 6 | Quantize | 10 | ⭐⭐ | #401 | 🔧 In Progress (0/10) |
| 7 | Sampling | 7 | ⭐⭐ | #426 | 🔧 In Progress (0/7) |
| 8 | Flash Attention | 16 | ⭐⭐⭐ | #403 | 🔧 In Progress (0/16) |
| 9 | MoE | 6 | ⭐⭐⭐ | #404 | 🔲 Not Started |
| 10 | Linear Attention | 8 | ⭐⭐⭐ | #405 | 🔧 In Progress (0/8) |
| 11 | SSM | 2 | ⭐⭐⭐ | #406 | 🔲 Not Started |

### 各分类 op 详情

<details>
<summary><b>Elementwise</b> — 0/72 ops (0%) [🔧 In Progress (0/72)]</summary>

| Op | Sub-category | 实现 | 测试 | Benchmark |
|---|---|:---:|:---:|:---:|
| `add` | binary_arith | ✅ | — | — |
| `sub` | binary_arith | ✅ | — | — |
| `mul` | binary_arith | ✅ | — | — |
| `div` | binary_arith | ✅ | — | — |
| `remainder` | binary_arith | ✅ | — | — |
| `pow` | binary_arith | ✅ | — | — |
| `floor_divide` | binary_arith | ✅ | — | — |
| `lerp` | binary_arith | ✅ | — | — |
| `maximum` | binary_arith | ✅ | — | — |
| `minimum` | binary_arith | ✅ | — | — |
| `exp` | unary_math | ✅ | — | — |
| `log` | unary_math | ✅ | — | — |
| `sqrt` | unary_math | ✅ | — | — |
| `rsqrt` | unary_math | ✅ | — | — |
| `abs` | unary_math | ✅ | — | — |
| `neg` | unary_math | ✅ | — | — |
| `reciprocal` | unary_math | ✅ | — | — |
| `sign` | unary_math | ✅ | — | — |
| `sin` | unary_math | ✅ | — | — |
| `cos` | unary_math | ✅ | — | — |
| `floor` | unary_math | ✅ | — | — |
| `ceil` | unary_math | ✅ | — | — |
| `round` | unary_math | ✅ | — | — |
| `trunc` | unary_math | ✅ | — | — |
| `erf` | unary_math | ✅ | — | — |
| `log1p` | unary_math | ✅ | — | — |
| `expm1` | unary_math | ✅ | — | — |
| `relu` | activation | ✅ | — | — |
| `gelu` | activation | ✅ | — | — |
| `silu` | activation | ✅ | — | — |
| `sigmoid` | activation | ✅ | — | — |
| `tanh` | activation | ✅ | — | — |
| `leaky_relu` | activation | 🔲 | — | — |
| `elu` | activation | 🔲 | — | — |
| `selu` | activation | ✅ | — | — |
| `hardswish` | activation | ✅ | — | — |
| `hardsigmoid` | activation | ✅ | — | — |
| `hardtanh` | activation | 🔲 | — | — |
| `softplus` | activation | 🔲 | — | — |
| `mish` | activation | ✅ | — | — |
| `prelu` | activation | 🔲 | — | — |
| `silu_and_mul` | fused_gated_activation | ✅ | — | — |
| `gelu_and_mul` | fused_gated_activation | ✅ | — | — |
| `gelu_tanh_and_mul` | fused_gated_activation | ✅ | — | — |
| `eq` | comparison | ✅ | — | — |
| `ne` | comparison | ✅ | — | — |
| `gt` | comparison | ✅ | — | — |
| `lt` | comparison | ✅ | — | — |
| `ge` | comparison | ✅ | — | — |
| `le` | comparison | ✅ | — | — |
| `bitwise_and` | bitwise | ✅ | — | — |
| `bitwise_or` | bitwise | ✅ | — | — |
| `bitwise_xor` | bitwise | ✅ | — | — |
| `bitwise_not` | bitwise | ✅ | — | — |
| `logical_not` | logical | ✅ | — | — |
| `logical_and` | logical | ✅ | — | — |
| `logical_or` | logical | ✅ | — | — |
| `where` | special_elementwise | 🔲 | — | — |
| `clamp` | special_elementwise | 🔲 | — | — |
| `masked_fill` | special_elementwise | 🔲 | — | — |
| `nan_to_num` | special_elementwise | 🔲 | — | — |
| `isnan` | special_elementwise | ✅ | — | — |
| `isinf` | special_elementwise | ✅ | — | — |
| `isfinite` | special_elementwise | ✅ | — | — |
| `dropout` | dropout | 🔲 | — | — |
| `rope_neox` | rope | 🔲 | — | — |
| `rope_non_neox` | rope | 🔲 | — | — |
| `rope_llama31` | rope | 🔲 | — | — |
| `yarn_rope` | rope | 🔲 | — | — |
| `longrope` | rope | 🔲 | — | — |
| `alibi` | alibi | 🔲 | — | — |
| `sinusoidal` | sinusoidal | 🔲 | — | — |

</details>

<details>
<summary><b>Reduce</b> — 0/20 ops (0%) [🔧 In Progress (0/20)]</summary>

| Op | Sub-category | 实现 | 测试 | Benchmark |
|---|---|:---:|:---:|:---:|
| `sum` | reduce | ✅ | — | — |
| `mean` | reduce | ✅ | — | — |
| `amin` | reduce | ✅ | — | — |
| `amax` | reduce | ✅ | — | — |
| `prod` | reduce | ✅ | — | — |
| `std` | reduce | ✅ | — | — |
| `var` | reduce | ✅ | — | — |
| `var_mean` | reduce | ✅ | — | — |
| `softmax` | softmax | ✅ | — | — |
| `log_softmax` | softmax | ✅ | — | — |
| `logsumexp` | softmax | ✅ | — | — |
| `argmax` | argreduce | ✅ | — | — |
| `argmin` | argreduce | ✅ | — | — |
| `cumsum` | cumulative | ✅ | — | — |
| `cumprod` | cumulative | ✅ | — | — |
| `any` | logical_reduce | 🔲 | — | — |
| `all` | logical_reduce | 🔲 | — | — |
| `l1_norm` | linalg_vector_norm | 🔲 | — | — |
| `l2_norm` | linalg_vector_norm | 🔲 | — | — |
| `inf_norm` | linalg_vector_norm | 🔲 | — | — |

</details>

<details>
<summary><b>Norm</b> — 0/10 ops (0%) [🔧 In Progress (0/10)]</summary>

| Op | Sub-category | 实现 | 测试 | Benchmark |
|---|---|:---:|:---:|:---:|
| `layer_norm` | layer_norm | ✅ | — | — |
| `batch_norm` | batch_norm | ✅ | — | — |
| `group_norm` | group_norm | ✅ | — | — |
| `instance_norm` | instance_norm | ✅ | — | — |
| `rms_norm` | rms_norm | ✅ | — | — |
| `qk_norm` | qk_norm | 🔲 | — | — |
| `ada_layer_norm` | ada_layer_norm | ✅ | — | — |
| `ada_layer_norm_zero` | ada_layer_norm_zero | ✅ | — | — |
| `fused_add_layer_norm` | fused_add_norm | ✅ | — | — |
| `fused_add_rmsnorm` | fused_add_norm | ✅ | — | — |

</details>

<details>
<summary><b>Conv & Pooling</b> — 0/16 ops (0%) [🔲 Not Started]</summary>

| Op | Sub-category | 实现 | 测试 | Benchmark |
|---|---|:---:|:---:|:---:|
| `conv1d` | conv | 🔲 | — | — |
| `conv2d` | conv | 🔲 | — | — |
| `conv3d` | conv | 🔲 | — | — |
| `conv_transpose1d` | conv_transpose | 🔲 | — | — |
| `conv_transpose2d` | conv_transpose | 🔲 | — | — |
| `depthwise_conv2d` | depthwise | 🔲 | — | — |
| `grouped_conv2d` | grouped | 🔲 | — | — |
| `dilated_conv2d` | dilated | 🔲 | — | — |
| `max_pool1d` | max_pool | 🔲 | — | — |
| `max_pool2d` | max_pool | 🔲 | — | — |
| `max_pool3d` | max_pool | 🔲 | — | — |
| `avg_pool1d` | avg_pool | 🔲 | — | — |
| `avg_pool2d` | avg_pool | 🔲 | — | — |
| `avg_pool3d` | avg_pool | 🔲 | — | — |
| `adaptive_avg_pool2d` | adaptive_pool | 🔲 | — | — |
| `adaptive_max_pool2d` | adaptive_pool | 🔲 | — | — |

</details>

<details>
<summary><b>GEMM</b> — 0/19 ops (0%) [🔧 In Progress (0/19)]</summary>

| Op | Sub-category | 实现 | 测试 | Benchmark |
|---|---|:---:|:---:|:---:|
| `gemm_fp16` | gemm | ✅ | — | — |
| `gemm_fp8` | gemm | ✅ | — | — |
| `gemm_fp8_block_scaled` | gemm | ✅ | — | — |
| `gemv_fp16` | gemm | ✅ | — | — |
| `gemv_fp8` | gemm | ✅ | — | — |
| `small_batch_gemm_fp16` | gemm | ✅ | — | — |
| `small_batch_gemm_fp8` | gemm | ✅ | — | — |
| `bmm_fp16` | bmm | 🔲 | — | — |
| `bmm_fp8` | bmm | 🔲 | — | — |
| `groupgemm_fp16` | groupgemm | ✅ | — | — |
| `groupgemm_fp8` | groupgemm | ✅ | — | — |
| `outer` | outer | 🔲 | — | — |
| `w4a16` | lowbit_gemm | 🔲 | — | — |
| `w8a8` | lowbit_gemm | 🔲 | — | — |
| `w8a8_int8` | lowbit_gemm | 🔲 | — | — |
| `weight_only_int4` | lowbit_gemm | 🔲 | — | — |
| `fp4` | lowbit_gemm | 🔲 | — | — |
| `sparse_gemm_fp16` | sparse_gemm_2_4 | 🔲 | — | — |
| `sparse_gemm_fp8` | sparse_gemm_2_4 | 🔲 | — | — |

</details>

<details>
<summary><b>Quantize</b> — 0/10 ops (0%) [🔧 In Progress (0/10)]</summary>

| Op | Sub-category | 实现 | 测试 | Benchmark |
|---|---|:---:|:---:|:---:|
| `int8_per_tensor` | int8_quantize | 🔲 | — | — |
| `int8_per_channel` | int8_quantize | 🔲 | — | — |
| `int8_per_block` | int8_quantize | 🔲 | — | — |
| `smooth_quant` | int8_quantize | 🔲 | — | — |
| `int4_per_channel` | int4_quantize | 🔲 | — | — |
| `int4_per_block` | int4_quantize | 🔲 | — | — |
| `nf4` | int4_quantize | 🔲 | — | — |
| `fp8_per_tensor` | fp8_quantize | ✅ | — | — |
| `fp8_per_block` | fp8_quantize | 🔲 | — | — |
| `fp8_cast_transpose` | fp8_quantize | 🔲 | — | — |

</details>

<details>
<summary><b>Sampling</b> — 0/7 ops (0%) [🔧 In Progress (0/7)]</summary>

| Op | Sub-category | 实现 | 测试 | Benchmark |
|---|---|:---:|:---:|:---:|
| `top_k` | top_k | 🔲 | — | — |
| `top_p` | top_p | 🔲 | — | — |
| `min_p` | min_p | 🔲 | — | — |
| `top_k_top_p` | top_k_top_p | 🔲 | — | — |
| `temperature_scale` | temperature_scale | 🔲 | — | — |
| `sampling_from_probs` | sampling_from_probs | 🔲 | — | — |
| `chain_speculative_sampling` | chain_speculative_sampling | ✅ | — | — |

</details>

<details>
<summary><b>Flash Attention</b> — 0/16 ops (0%) [🔧 In Progress (0/16)]</summary>

| Op | Sub-category | 实现 | 测试 | Benchmark |
|---|---|:---:|:---:|:---:|
| `flash_prefill_fwd` | flash_attention | ✅ | — | — |
| `flash_prefill_bwd` | flash_attention | ✅ | — | — |
| `flash_prefill_varlen_fwd` | flash_attention | ✅ | — | — |
| `flash_prefill_varlen_bwd` | flash_attention | 🔲 | — | — |
| `flash_decode_fwd` | flash_attention | ✅ | — | — |
| `flash_decode_paged_fwd` | flash_attention | ✅ | — | — |
| `flash_decode_varlen_fwd` | flash_attention | 🔲 | — | — |
| `flash_chunked_prefill_fwd` | flash_attention | 🔲 | — | — |
| `mla_prefill_fwd` | mla | 🔲 | — | — |
| `mla_prefill_bwd` | mla | 🔲 | — | — |
| `mla_decode_fwd` | mla | ✅ | — | — |
| `mla_decode_paged_fwd` | mla | 🔲 | — | — |
| `nsa_prefill_fwd` | nsa | ✅ | — | — |
| `nsa_decode_fwd` | nsa | 🔲 | — | — |
| `dsa_prefill_fwd` | dsa | 🔲 | — | — |
| `dsa_decode_fwd` | dsa | ✅ | — | — |

</details>

<details>
<summary><b>MoE</b> — 0/6 ops (0%) [🔲 Not Started]</summary>

| Op | Sub-category | 实现 | 测试 | Benchmark |
|---|---|:---:|:---:|:---:|
| `permute_align` | permute_align | 🔲 | — | — |
| `unpermute_depad` | unpermute_depad | 🔲 | — | — |
| `fused_moe_deepseek` | fused_moe | 🔲 | — | — |
| `fused_moe_glm` | fused_moe | 🔲 | — | — |
| `fused_moe_kimi` | fused_moe | 🔲 | — | — |
| `fused_moe_qwen` | fused_moe | 🔲 | — | — |

</details>

<details>
<summary><b>Linear Attention</b> — 0/8 ops (0%) [🔧 In Progress (0/8)]</summary>

| Op | Sub-category | 实现 | 测试 | Benchmark |
|---|---|:---:|:---:|:---:|
| `gated_deltanet_chunkwise` | gated_deltanet | ✅ | — | — |
| `gated_deltanet_recurrence` | gated_deltanet | ✅ | — | — |
| `deltanet_chunkwise` | deltanet | ✅ | — | — |
| `deltanet_recurrence` | deltanet | ✅ | — | — |
| `gla_chunkwise` | gla | 🔲 | — | — |
| `gla_recurrence` | gla | 🔲 | — | — |
| `retnet_chunkwise` | retnet | 🔲 | — | — |
| `retnet_recurrence` | retnet | 🔲 | — | — |

</details>

<details>
<summary><b>SSM</b> — 0/2 ops (0%) [🔲 Not Started]</summary>

| Op | Sub-category | 实现 | 测试 | Benchmark |
|---|---|:---:|:---:|:---:|
| `mamba1` | mamba1 | 🔲 | — | — |
| `mamba2` | mamba2 | 🔲 | — | — |

</details>

## 测试结果

> 测试结果文件未找到
## Benchmark 结果

> Benchmark 报告未生成（可能已跳过或运行失败）。

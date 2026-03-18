<div align="center">

# TileOPs Operator Tracking

**73/186** operators complete (39%)

[![Report](https://superanggao.github.io/TileOPs-report/nightly/report.html)](https://superanggao.github.io/TileOPs-report/nightly/)

Last updated: `2026-03-18T11:15:24Z`

</div>

---

## Overview

| | Count | Bar |
|:--|------:|:----|
| Implemented | **110**/186 | `█████████░░░░░░` 110/186 |
| Test Passed | **82**/186 | `███████░░░░░░░░` 82/186 |
| Bench Passed | **93**/186 | `████████░░░░░░░` 93/186 |
| **Done** | **73**/186 | `██████░░░░░░░░░` 73/186 |

### Status Breakdown

| Test | | Bench | |
|:-----|----:|:------|----:|
| Passed | 82 | Qualified (ratio >= 0.8) | 31 |
| Failed | 7 | Passed (no ratio) | 62 |
| Missing | 97 | Underperforming | 4 |
| | | Failed | 3 |
| | | Missing | 86 |

## Assessment

> TileOPs is at roughly 59% implementation coverage (110/186 ops) with strong results in mature categories like Norm, Flash Attention, and Elementwise, but significant gaps remain in Reduce, Conv & Pooling, Quantize, and MoE. Test coverage lags implementation — 28 implemented ops have no test mappings — and several benchmark pipelines are broken or missing ratio data, making performance qualification incomplete. The project has a solid foundation but needs focused effort on closing test/bench gaps and unblocking the failing kernels before broader rollout.

**Recommendations:**

1. Fix the 3 failing rope ops (rope_neox, rope_non_neox, rope_llama31) and the 3 failing linalg_vector_norm ops (l1_norm, l2_norm, inf_norm) — these are regressions in implemented ops that should be green.
2. Investigate and resolve gated_deltanet_chunkwise performance (ratio=0.12) and the logical_and/logical_or/bitwise_not underperformance — these suggest kernel-level bugs rather than tuning issues.
3. Add test and benchmark mappings for all 8 implemented-but-untested GEMM ops (fp8 and bmm variants) and fix the broken benchmark harness for fp8_quantize and chain_speculative_sampling to restore full CI signal.

## Categories

### Elementwise | Func: ⭐⭐⭐⭐☆ (4/5) | Perf: ⭐⭐⭐⭐☆ (4/5)

| | Progress | |
|:--|:---------|:--|
| Impl | `███████████████` 70/72 | |
| Test | `███████████░░░░` 55/72 | |
| Bench | `████████████░░░` 58/72 | |

> **Issues:** dropout, rope_neox, rope_non_neox, rope_llama31 tests failing. logical_and(0.33), logical_or(0.33), bitwise_not(0.50) significantly underperforming. 13 ops lack test coverage (where, clamp, masked_fill, nan_to_num, alibi, sinusoidal, etc.). bench_status for many unary_math and activation ops recorded as 'passed' rather than qualified/underperforming — ratio data missing.

> **Evaluation:** Strongest category overall with broad implementation coverage. Fix the 3 failing rope ops and dropout tests as a priority. Investigate logical_and/logical_or/bitwise_not performance regressions — ratios of 0.33 suggest a likely kernel or dispatch issue. Add test and bench mappings for the 13 uncovered special_elementwise and positional encoding ops.

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
| 🟦 | elu | missing | qualified | 0.91 |
| ✅ | selu | passed | passed | - |
| ✅ | hardswish | passed | passed | - |
| ✅ | hardsigmoid | passed | passed | - |
| 🟦 | hardtanh | missing | qualified | 1.00 |
| 🟦 | softplus | missing | qualified | 0.96 |
| ✅ | mish | passed | passed | - |
| 🟦 | prelu | missing | qualified | 2.20 |
| ✅ | silu_and_mul | passed | passed | - |
| ✅ | gelu_and_mul | passed | qualified | 3.29 |
| ✅ | gelu_tanh_and_mul | passed | qualified | 3.83 |
| ✅ | eq | passed | passed | - |
| ✅ | ne | passed | passed | - |
| ✅ | gt | passed | passed | - |
| ✅ | lt | passed | passed | - |
| ✅ | ge | passed | passed | - |
| ✅ | le | passed | passed | - |
| ✅ | bitwise_and | passed | qualified | 1.00 |
| ✅ | bitwise_or | passed | qualified | 1.00 |
| ✅ | bitwise_xor | passed | qualified | 1.00 |
| 🟡 | bitwise_not | passed | underperforming | 0.50 |
| ✅ | logical_not | passed | qualified | 1.00 |
| 🟡 | logical_and | passed | underperforming | 0.33 |
| 🟡 | logical_or | passed | underperforming | 0.33 |
| 🟦 | where | missing | missing | - |
| 🟦 | clamp | missing | missing | - |
| 🟦 | masked_fill | missing | missing | - |
| 🟦 | nan_to_num | missing | missing | - |
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

### Reduce | Func: ⭐☆☆☆☆ (1/5)

| | Progress | |
|:--|:---------|:--|
| Impl | `██░░░░░░░░░░░░░` 3/20 | |
| Test | `░░░░░░░░░░░░░░░` 0/20 | |
| Bench | `██░░░░░░░░░░░░░` 3/20 | |

> **Issues:** Only 3 of 20 ops implemented (l1_norm, l2_norm, inf_norm), all 3 fail tests. No benchmark data for any op. 17 ops entirely unimplemented.

> **Evaluation:** Category is effectively non-functional. Fix the 3 linalg_vector_norm test failures immediately since those are the only implemented ops. The remaining 17 ops (sum, mean, softmax, argmax, cumsum, etc.) are critical for a complete LLM operator library and should be prioritized in the implementation roadmap.

<details>
<summary>0/20 done - click to expand</summary>

| | Operator | Test | Bench | Ratio |
|:--|:---------|:----:|:-----:|------:|
| ⬜ | sum | missing | missing | - |
| ⬜ | mean | missing | missing | - |
| ⬜ | amin | missing | missing | - |
| ⬜ | amax | missing | missing | - |
| ⬜ | prod | missing | missing | - |
| ⬜ | std | missing | missing | - |
| ⬜ | var | missing | missing | - |
| ⬜ | var_mean | missing | missing | - |
| ⬜ | softmax | missing | missing | - |
| ⬜ | log_softmax | missing | missing | - |
| ⬜ | logsumexp | missing | missing | - |
| ⬜ | argmax | missing | missing | - |
| ⬜ | argmin | missing | missing | - |
| ⬜ | cumsum | missing | missing | - |
| ⬜ | cumprod | missing | missing | - |
| ⬜ | any | missing | missing | - |
| ⬜ | all | missing | missing | - |
| ❌ | l1_norm | failed | passed | - |
| ❌ | l2_norm | failed | passed | - |
| ❌ | inf_norm | failed | passed | - |

</details>

### Norm | Func: ⭐⭐⭐⭐⭐ (5/5) | Perf: ⭐⭐⭐⭐⭐ (5/5)

| | Progress | |
|:--|:---------|:--|
| Impl | `███████████████` 10/10 | |
| Test | `██████████████░` 9/10 | |
| Bench | `██████████████░` 9/10 | |

> **Issues:** qk_norm lacks both test and bench mapping. Several ops have bench status 'passed' without ratio data.

> **Evaluation:** Healthiest category in the project — 9/10 ops passing with no failures and ada_layer_norm variants exceeding baseline. Add test and bench mapping for qk_norm to reach full coverage.

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
| ✅ | ada_layer_norm | passed | qualified | 1.36 |
| ✅ | ada_layer_norm_zero | passed | qualified | 1.33 |
| ✅ | fused_add_layer_norm | passed | passed | - |
| ✅ | fused_add_rmsnorm | passed | passed | - |

</details>

### Conv & Pooling

| | Progress | |
|:--|:---------|:--|
| Impl | `░░░░░░░░░░░░░░░` 0/16 | |
| Test | `░░░░░░░░░░░░░░░` 0/16 | |
| Bench | `░░░░░░░░░░░░░░░` 0/16 | |

> **Issues:** Zero implementation across all 16 ops. No tests, no benchmarks.

> **Evaluation:** Category has not been started. If Conv & Pooling is in scope for TileOPs, it needs to be scheduled. If it is out of scope for the current milestone, formally mark it as deferred to avoid noise in progress tracking.

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
| Impl | `█████████░░░░░░` 11/19 | |
| Test | `██░░░░░░░░░░░░░` 3/19 | |
| Bench | `█████████░░░░░░` 11/19 | |

> **Issues:** 11 ops implemented but only 3 have test mappings (gemm_fp16, gemv_fp16, groupgemm_fp16). 8 implemented ops have bench results recorded as 'passed' without ratio data, so qualified/underperforming status is unknown. 8 lowbit and sparse ops unimplemented.

> **Evaluation:** Functional correctness looks solid for tested ops but test coverage is critically low — 8 implemented ops are untested. Add test mappings for all fp8 and bmm variants. Ensure benchmark results capture ratio data so performance qualification can be properly assessed.

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

### Quantize | Func: ⭐⭐⭐⭐☆ (4/5)

| | Progress | |
|:--|:---------|:--|
| Impl | `███░░░░░░░░░░░░` 2/10 | |
| Test | `██░░░░░░░░░░░░░` 1/10 | |
| Bench | `░░░░░░░░░░░░░░░` 0/10 | |

> **Issues:** fp8_per_tensor and fp8_per_block benchmarks exist but produced no results (failed). fp8_per_block and fp8_cast_transpose lack test mappings. 8 int8/int4 ops unimplemented.

> **Evaluation:** Only 2 of 10 ops implemented and benchmark pipeline is broken for both fp8 ops. Fix the benchmark runner for fp8_quantize group — likely an environment or harness issue. Add test mapping for fp8_per_block.

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
| 🟡 | fp8_per_tensor | passed | failed | - |
| 🟦 | fp8_per_block | missing | failed | - |
| ⬜ | fp8_cast_transpose | missing | missing | - |

</details>

### Sampling | Func: ⭐⭐⭐⭐☆ (4/5)

| | Progress | |
|:--|:---------|:--|
| Impl | `██░░░░░░░░░░░░░` 1/7 | |
| Test | `██░░░░░░░░░░░░░` 1/7 | |
| Bench | `░░░░░░░░░░░░░░░` 0/7 | |

> **Issues:** chain_speculative_sampling benchmark mapping exists but produced no results. 6 of 7 ops unimplemented.

> **Evaluation:** Only chain_speculative_sampling is implemented and it passes tests but its benchmark is broken. Fix the benchmark harness for this op. The remaining 6 sampling ops (top_k, top_p, min_p, etc.) are important for inference pipelines and should be scheduled for implementation.

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
| 🟡 | chain_speculative_sampling | passed | failed | - |

</details>

### Flash Attention | Func: ⭐⭐⭐⭐⭐ (5/5) | Perf: ⭐⭐⭐⭐⭐ (5/5)

| | Progress | |
|:--|:---------|:--|
| Impl | `████████░░░░░░░` 8/16 | |
| Test | `████████░░░░░░░` 8/16 | |
| Bench | `████████░░░░░░░` 8/16 | |

> **Issues:** 8 ops unimplemented (flash_prefill_varlen_bwd, flash_decode_varlen_fwd, flash_chunked_prefill_fwd, all MLA paged/prefill, NSA decode, DSA prefill). Several implemented ops have bench status 'passed' without ratio data.

> **Evaluation:** All implemented ops pass tests and mla_decode_fwd(4.33) and nsa_prefill_fwd(2.43) significantly exceed baseline — strong results. Focus next on implementing flash_chunked_prefill_fwd and flash_decode_varlen_fwd as they are high-priority for production LLM serving.

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
| ✅ | mla_decode_fwd | passed | qualified | 4.33 |
| ⬜ | mla_decode_paged_fwd | missing | missing | - |
| ✅ | nsa_prefill_fwd | passed | qualified | 2.43 |
| ⬜ | nsa_decode_fwd | missing | missing | - |
| ⬜ | dsa_prefill_fwd | missing | missing | - |
| ✅ | dsa_decode_fwd | passed | passed | - |

</details>

### MoE | Func: ⭐⭐⭐⭐⭐ (5/5) | Perf: ⭐⭐⭐⭐⭐ (5/5)

| | Progress | |
|:--|:---------|:--|
| Impl | `██░░░░░░░░░░░░░` 1/6 | |
| Test | `██░░░░░░░░░░░░░` 1/6 | |
| Bench | `██░░░░░░░░░░░░░` 1/6 | |

> **Issues:** Only permute_align implemented. fused_moe variants for deepseek, glm, kimi, qwen all unimplemented — these are the high-value ops for the category.

> **Evaluation:** The one implemented op is healthy but the category is essentially a stub. The fused_moe variants are critical for MoE inference performance and should be the top implementation priority for this category.

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

### Linear Attention | Func: ⭐⭐⭐⭐⭐ (5/5) | Perf: ⭐⭐⭐☆☆ (3/5)

| | Progress | |
|:--|:---------|:--|
| Impl | `████████░░░░░░░` 4/8 | |
| Test | `████████░░░░░░░` 4/8 | |
| Bench | `██████░░░░░░░░░` 3/8 | |

> **Issues:** gated_deltanet_chunkwise severely underperforming at ratio=0.12 — well below the 0.80 threshold. deltanet and retnet variants unimplemented.

> **Evaluation:** All 4 implemented ops pass tests, and gated_deltanet_recurrence(4.47) and gla_recurrence(3.79) are strong. However gated_deltanet_chunkwise at 0.12 is a critical performance regression that needs immediate investigation — likely a tiling or memory access pattern issue in the chunkwise kernel.

<details>
<summary>3/8 done - click to expand</summary>

| | Operator | Test | Bench | Ratio |
|:--|:---------|:----:|:-----:|------:|
| 🟡 | gated_deltanet_chunkwise | passed | underperforming | 0.12 |
| ✅ | gated_deltanet_recurrence | passed | qualified | 4.47 |
| ⬜ | deltanet_chunkwise | missing | missing | - |
| ⬜ | deltanet_recurrence | missing | missing | - |
| ✅ | gla_chunkwise | passed | passed | - |
| ✅ | gla_recurrence | passed | qualified | 3.79 |
| ⬜ | retnet_chunkwise | missing | missing | - |
| ⬜ | retnet_recurrence | missing | missing | - |

</details>

### SSM

| | Progress | |
|:--|:---------|:--|
| Impl | `░░░░░░░░░░░░░░░` 0/2 | |
| Test | `░░░░░░░░░░░░░░░` 0/2 | |
| Bench | `░░░░░░░░░░░░░░░` 0/2 | |

> **Issues:** mamba1 and mamba2 both unimplemented.

> **Evaluation:** Category not started. Mamba kernels are relevant for SSM-based model support. Schedule implementation if in scope for the current milestone.

<details>
<summary>0/2 done - click to expand</summary>

| | Operator | Test | Bench | Ratio |
|:--|:---------|:----:|:-----:|------:|
| ⬜ | mamba1 | missing | missing | - |
| ⬜ | mamba2 | missing | missing | - |

</details>

---

<sub>Auto-generated by the [TileOPs Report](https://superanggao.github.io/TileOPs-report/nightly/) pipeline from [`op_registry.json`](scripts/op_registry.json)</sub>
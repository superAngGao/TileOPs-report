........................................................................ [  9%]
........................................................................ [ 18%]
........................................................................ [ 27%]
........................................................................ [ 37%]
........................................................................ [ 46%]
.......................FFFFFF........................................... [ 55%]
........................................................................ [ 65%]
........................................................................ [ 74%]
........................................................................ [ 83%]
...............................................................FFFF..... [ 92%]
.......................................................                  [100%]Benchmark report saved to profile_run.log

=================================== FAILURES ===================================
_______________ test_fp8_lighting_indexer_bench[default-config] ________________

seq_len = 4096, heads = 32, index_dim = 64, seq_len_kv = 8192
clean_logits = True, config = None, tune = False

    @pytest.mark.parametrize(
        "seq_len, heads, index_dim, seq_len_kv, clean_logits, config, tune",
        _FP8_LIGHTING_INDEXER_BENCH_PARAMS,
    )
    def test_fp8_lighting_indexer_bench(seq_len: int, heads: int, index_dim: int, seq_len_kv: int,
                                        clean_logits: bool, config: Optional[dict],
                                        tune: bool) -> None:
        test = Fp8LightingIndexerTest(seq_len, heads, index_dim, seq_len_kv, clean_logits, config)
        bm = Fp8LightingIndexerBenchmark(test)
>       inputs = test.gen_inputs()
                 ^^^^^^^^^^^^^^^^^

benchmarks/ops/bench_fp8_lighting_indexer.py:50: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <tests.ops.test_fp8_lighting_indexer.Fp8LightingIndexerTest object at 0x7f2f8723b290>
params = None

    def gen_inputs(
            self,
            params=None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        IndexQ = torch.randn(
            self.batch,
            self.seq_len,
            self.heads,
            self.index_dim,
            device='cuda',
            dtype=torch.bfloat16)
>       IndexK = torch.randn(
            self.batch,
            self.seq_len_kv,
            self.kv_group,
            self.index_dim,
            device='cuda',
            dtype=torch.bfloat16)
E       TypeError: randn(): argument 'size' failed to unpack the object at pos 3 with error "type must be tuple of ints,but got NoneType"

tests/ops/test_fp8_lighting_indexer.py:170: TypeError
__________________ test_fp8_lighting_indexer_bench[mid-shape] __________________

seq_len = 2048, heads = 16, index_dim = 64, seq_len_kv = 4096
clean_logits = True, config = None, tune = False

    @pytest.mark.parametrize(
        "seq_len, heads, index_dim, seq_len_kv, clean_logits, config, tune",
        _FP8_LIGHTING_INDEXER_BENCH_PARAMS,
    )
    def test_fp8_lighting_indexer_bench(seq_len: int, heads: int, index_dim: int, seq_len_kv: int,
                                        clean_logits: bool, config: Optional[dict],
                                        tune: bool) -> None:
        test = Fp8LightingIndexerTest(seq_len, heads, index_dim, seq_len_kv, clean_logits, config)
        bm = Fp8LightingIndexerBenchmark(test)
>       inputs = test.gen_inputs()
                 ^^^^^^^^^^^^^^^^^

benchmarks/ops/bench_fp8_lighting_indexer.py:50: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <tests.ops.test_fp8_lighting_indexer.Fp8LightingIndexerTest object at 0x7f2f874d0b50>
params = None

    def gen_inputs(
            self,
            params=None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
>       IndexQ = torch.randn(
            self.batch,
            self.seq_len,
            self.heads,
            self.index_dim,
            device='cuda',
            dtype=torch.bfloat16)
E       torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 16.00 GiB. GPU 0 has a total capacity of 140.06 GiB of which 3.12 GiB is free. Process 3989842 has 136.94 GiB memory in use. Of the allocated memory 128.03 GiB is allocated by PyTorch, and 216.00 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

tests/ops/test_fp8_lighting_indexer.py:163: OutOfMemoryError
____________________ test_fp8_quant_bench[mainstream-fp16] _____________________

seq_len_kv = 8192, index_dim = 64, in_dtype = torch.float16, tune = True

    @pytest.mark.parametrize("seq_len_kv, index_dim, in_dtype, tune", _FP8_QUANT_BENCH_PARAMS)
    def test_fp8_quant_bench(seq_len_kv: int, index_dim: int, in_dtype: torch.dtype,
                             tune: bool) -> None:
>       test = Fp8QuantTest(seq_len_kv, index_dim, in_dtype)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       TypeError: Fp8QuantTest.__init__() missing 2 required positional arguments: 'index_dim' and 'in_dtype'

benchmarks/ops/bench_fp8_quant.py:34: TypeError
____________________ test_fp8_quant_bench[mainstream-bf16] _____________________

seq_len_kv = 8192, index_dim = 64, in_dtype = torch.bfloat16, tune = True

    @pytest.mark.parametrize("seq_len_kv, index_dim, in_dtype, tune", _FP8_QUANT_BENCH_PARAMS)
    def test_fp8_quant_bench(seq_len_kv: int, index_dim: int, in_dtype: torch.dtype,
                             tune: bool) -> None:
>       test = Fp8QuantTest(seq_len_kv, index_dim, in_dtype)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       TypeError: Fp8QuantTest.__init__() missing 2 required positional arguments: 'index_dim' and 'in_dtype'

benchmarks/ops/bench_fp8_quant.py:34: TypeError
______________________ test_fp8_quant_bench[wider-index] _______________________

seq_len_kv = 4096, index_dim = 128, in_dtype = torch.float32, tune = True

    @pytest.mark.parametrize("seq_len_kv, index_dim, in_dtype, tune", _FP8_QUANT_BENCH_PARAMS)
    def test_fp8_quant_bench(seq_len_kv: int, index_dim: int, in_dtype: torch.dtype,
                             tune: bool) -> None:
>       test = Fp8QuantTest(seq_len_kv, index_dim, in_dtype)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       TypeError: Fp8QuantTest.__init__() missing 2 required positional arguments: 'index_dim' and 'in_dtype'

benchmarks/ops/bench_fp8_quant.py:34: TypeError
_____________________ test_fp8_quant_bench[long-sequence] ______________________

seq_len_kv = 16384, index_dim = 32, in_dtype = torch.float32, tune = True

    @pytest.mark.parametrize("seq_len_kv, index_dim, in_dtype, tune", _FP8_QUANT_BENCH_PARAMS)
    def test_fp8_quant_bench(seq_len_kv: int, index_dim: int, in_dtype: torch.dtype,
                             tune: bool) -> None:
>       test = Fp8QuantTest(seq_len_kv, index_dim, in_dtype)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       TypeError: Fp8QuantTest.__init__() missing 2 required positional arguments: 'index_dim' and 'in_dtype'

benchmarks/ops/bench_fp8_quant.py:34: TypeError
___________________ test_topk_selector_bench[base-topk1024] ____________________

batch = 64, seq_len = 32768, topk = 1024, in_dtype_str = 'float32'
out_dtype_str = 'int32', tune = True

    @pytest.mark.parametrize(
        "batch, seq_len, topk, in_dtype_str, out_dtype_str, tune",
        _TOPK_SELECTOR_BENCH_PARAMS,
    )
    def test_topk_selector_bench(batch: int, seq_len: int, topk: int, in_dtype_str: str,
                                  out_dtype_str: str, tune: bool) -> None:
        in_dtype = str2dtype[in_dtype_str]
        out_dtype = str2dtype[out_dtype_str]
>       test = TopkSelectorTest(batch, seq_len, topk, in_dtype, out_dtype)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       TypeError: TopkSelectorTest.__init__() missing 2 required positional arguments: 'in_dtype' and 'out_dtype'

benchmarks/ops/bench_topk_selector.py:42: TypeError
___________________ test_topk_selector_bench[base-topk2048] ____________________

batch = 64, seq_len = 32768, topk = 2048, in_dtype_str = 'float32'
out_dtype_str = 'int32', tune = True

    @pytest.mark.parametrize(
        "batch, seq_len, topk, in_dtype_str, out_dtype_str, tune",
        _TOPK_SELECTOR_BENCH_PARAMS,
    )
    def test_topk_selector_bench(batch: int, seq_len: int, topk: int, in_dtype_str: str,
                                  out_dtype_str: str, tune: bool) -> None:
        in_dtype = str2dtype[in_dtype_str]
        out_dtype = str2dtype[out_dtype_str]
>       test = TopkSelectorTest(batch, seq_len, topk, in_dtype, out_dtype)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       TypeError: TopkSelectorTest.__init__() missing 2 required positional arguments: 'in_dtype' and 'out_dtype'

benchmarks/ops/bench_topk_selector.py:42: TypeError
________________ test_topk_selector_bench[large-batch-topk1024] ________________

batch = 128, seq_len = 65536, topk = 1024, in_dtype_str = 'float32'
out_dtype_str = 'int32', tune = True

    @pytest.mark.parametrize(
        "batch, seq_len, topk, in_dtype_str, out_dtype_str, tune",
        _TOPK_SELECTOR_BENCH_PARAMS,
    )
    def test_topk_selector_bench(batch: int, seq_len: int, topk: int, in_dtype_str: str,
                                  out_dtype_str: str, tune: bool) -> None:
        in_dtype = str2dtype[in_dtype_str]
        out_dtype = str2dtype[out_dtype_str]
>       test = TopkSelectorTest(batch, seq_len, topk, in_dtype, out_dtype)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       TypeError: TopkSelectorTest.__init__() missing 2 required positional arguments: 'in_dtype' and 'out_dtype'

benchmarks/ops/bench_topk_selector.py:42: TypeError
________________ test_topk_selector_bench[large-batch-topk2048] ________________

batch = 128, seq_len = 65536, topk = 2048, in_dtype_str = 'float32'
out_dtype_str = 'int32', tune = True

    @pytest.mark.parametrize(
        "batch, seq_len, topk, in_dtype_str, out_dtype_str, tune",
        _TOPK_SELECTOR_BENCH_PARAMS,
    )
    def test_topk_selector_bench(batch: int, seq_len: int, topk: int, in_dtype_str: str,
                                  out_dtype_str: str, tune: bool) -> None:
        in_dtype = str2dtype[in_dtype_str]
        out_dtype = str2dtype[out_dtype_str]
>       test = TopkSelectorTest(batch, seq_len, topk, in_dtype, out_dtype)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       TypeError: TopkSelectorTest.__init__() missing 2 required positional arguments: 'in_dtype' and 'out_dtype'

benchmarks/ops/bench_topk_selector.py:42: TypeError
=============================== warnings summary ===============================
benchmarks/ops/bench_activation.py: 93 warnings
benchmarks/ops/bench_ada_layer_norm.py: 16 warnings
benchmarks/ops/bench_argreduce.py: 12 warnings
benchmarks/ops/bench_batch_norm.py: 12 warnings
benchmarks/ops/bench_binary_arith.py: 65 warnings
benchmarks/ops/bench_binary_elementwise.py: 107 warnings
benchmarks/ops/bench_binary_strategy.py: 18 warnings
benchmarks/ops/bench_cumulative.py: 12 warnings
benchmarks/ops/bench_deepseek_dsa_decode.py: 4 warnings
benchmarks/ops/bench_deepseek_mla_decode.py: 4 warnings
benchmarks/ops/bench_deepseek_nsa_cmp_fwd.py: 4 warnings
benchmarks/ops/bench_deepseek_nsa_fwd.py: 6 warnings
benchmarks/ops/bench_deepseek_nsa_topk.py: 6 warnings
benchmarks/ops/bench_dropout.py: 9 warnings
benchmarks/ops/bench_elementwise_fp8.py: 24 warnings
benchmarks/ops/bench_engram_bwd.py: 4 warnings
benchmarks/ops/bench_engram_decode.py: 6 warnings
benchmarks/ops/bench_engram_fwd.py: 8 warnings
benchmarks/ops/bench_fft.py: 10 warnings
benchmarks/ops/bench_fft_lut.py: 21 warnings
benchmarks/ops/bench_fused_add_layer_norm.py: 8 warnings
benchmarks/ops/bench_fused_add_rmsnorm.py: 8 warnings
benchmarks/ops/bench_gated_deltanet_decode.py: 13 warnings
benchmarks/ops/bench_gated_deltanet_vs_fla.py: 64 warnings
benchmarks/ops/bench_gemm.py: 6 warnings
benchmarks/ops/bench_gla.py: 24 warnings
benchmarks/ops/bench_gla_decode.py: 16 warnings
benchmarks/ops/bench_gqa.py: 6 warnings
benchmarks/ops/bench_gqa_decode.py: 6 warnings
benchmarks/ops/bench_gqa_decode_paged.py: 8 warnings
benchmarks/ops/bench_gqa_sliding_window_fwd.py: 5 warnings
benchmarks/ops/bench_gqa_sliding_window_varlen_fwd.py: 5 warnings
benchmarks/ops/bench_group_norm.py: 8 warnings
benchmarks/ops/bench_grouped_gemm.py: 16 warnings
benchmarks/ops/bench_independent_elementwise.py: 102 warnings
benchmarks/ops/bench_instance_norm.py: 8 warnings
benchmarks/ops/bench_layer_norm.py: 8 warnings
benchmarks/ops/bench_logical_reduce.py: 30 warnings
benchmarks/ops/bench_mean_pooling.py: 8 warnings
benchmarks/ops/bench_mha.py: 6 warnings
benchmarks/ops/bench_mha_decode.py: 6 warnings
benchmarks/ops/bench_mha_decode_paged.py: 8 warnings
benchmarks/ops/bench_mhc_post.py: 6 warnings
benchmarks/ops/bench_mhc_pre.py: 6 warnings
benchmarks/ops/bench_moe_permute_align.py: 10 warnings
benchmarks/ops/bench_reduce.py: 20 warnings
benchmarks/ops/bench_rms_norm.py: 8 warnings
benchmarks/ops/bench_rope.py: 9 warnings
benchmarks/ops/bench_softmax.py: 24 warnings
benchmarks/ops/bench_unary_elementwise.py: 21 warnings
benchmarks/ops/bench_unary_strategy.py: 27 warnings
benchmarks/ops/bench_vector_norm.py: 24 warnings
  /home/ci-runner/workdir/_tool/tileops_benchmarks_venv_db4a9af2f48857c5/lib/python3.11/site-packages/torch/profiler/profiler.py:509: UserWarning: Profiler won't be using warmup, this can skew profiler results
    warn("Profiler won't be using warmup, this can skew profiler results")

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED benchmarks/ops/bench_fp8_lighting_indexer.py::test_fp8_lighting_indexer_bench[default-config] - TypeError: randn(): argument 'size' failed to unpack the object at pos 3 with error "type must be tuple of ints,but got NoneType"
FAILED benchmarks/ops/bench_fp8_lighting_indexer.py::test_fp8_lighting_indexer_bench[mid-shape] - torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 16.00 GiB. GPU 0 has a total capacity of 140.06 GiB of which 3.12 GiB is free. Process 3989842 has 136.94 GiB memory in use. Of the allocated memory 128.03 GiB is allocated by PyTorch, and 216.00 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
FAILED benchmarks/ops/bench_fp8_quant.py::test_fp8_quant_bench[mainstream-fp16] - TypeError: Fp8QuantTest.__init__() missing 2 required positional arguments: 'index_dim' and 'in_dtype'
FAILED benchmarks/ops/bench_fp8_quant.py::test_fp8_quant_bench[mainstream-bf16] - TypeError: Fp8QuantTest.__init__() missing 2 required positional arguments: 'index_dim' and 'in_dtype'
FAILED benchmarks/ops/bench_fp8_quant.py::test_fp8_quant_bench[wider-index] - TypeError: Fp8QuantTest.__init__() missing 2 required positional arguments: 'index_dim' and 'in_dtype'
FAILED benchmarks/ops/bench_fp8_quant.py::test_fp8_quant_bench[long-sequence] - TypeError: Fp8QuantTest.__init__() missing 2 required positional arguments: 'index_dim' and 'in_dtype'
FAILED benchmarks/ops/bench_topk_selector.py::test_topk_selector_bench[base-topk1024] - TypeError: TopkSelectorTest.__init__() missing 2 required positional arguments: 'in_dtype' and 'out_dtype'
FAILED benchmarks/ops/bench_topk_selector.py::test_topk_selector_bench[base-topk2048] - TypeError: TopkSelectorTest.__init__() missing 2 required positional arguments: 'in_dtype' and 'out_dtype'
FAILED benchmarks/ops/bench_topk_selector.py::test_topk_selector_bench[large-batch-topk1024] - TypeError: TopkSelectorTest.__init__() missing 2 required positional arguments: 'in_dtype' and 'out_dtype'
FAILED benchmarks/ops/bench_topk_selector.py::test_topk_selector_bench[large-batch-topk2048] - TypeError: TopkSelectorTest.__init__() missing 2 required positional arguments: 'in_dtype' and 'out_dtype'
10 failed, 765 passed, 965 warnings in 3257.93s (0:54:17)

===== profile_run.log summary =====
# TileOPs Benchmark Report
Generated: 2026-03-18 07:09:58

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
| 1000 | 23.09 | 0.00 | 0.00 | 0.00 |
| 2000 | 22.88 | 0.00 | 0.00 | 0.00 |
| 4000 | 23.47 | 0.00 | 0.00 | 0.01 |
| 8000 | 21.88 | 0.00 | 0.00 | 0.02 |
| 16000 | 23.09 | 0.00 | 0.01 | 0.03 |
| 32000 | 25.25 | 0.00 | 0.02 | 0.06 |
| 64000 | 22.78 | 0.00 | 0.03 | 0.12 |
| 128000 | 23.91 | 0.00 | 0.06 | 0.23 |
| 256000 | 23.76 | 0.00 | 0.11 | 0.45 |
| 512000 | 218.5 | 0.00 | 0.19 | 0.77 |

## r4_strategy_unary

### relu_direct

| n_total | size_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | torch.float16 | direct | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.bfloat16 | direct | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.float32 | direct | 0.00 | 0.00 | 0.02 |
| 4194304 | 4M | torch.float16 | direct | 0.01 | 0.29 | 1.15 |
| 4194304 | 4M | torch.bfloat16 | direct | 0.01 | 0.29 | 1.15 |
| 4194304 | 4M | torch.float32 | direct | 0.02 | 0.26 | 2.09 |
| 16777216 | 16M | torch.float16 | direct | 0.05 | 0.32 | 1.29 |
| 16777216 | 16M | torch.bfloat16 | direct | 0.05 | 0.32 | 1.29 |
| 16777216 | 16M | torch.float32 | direct | 0.06 | 0.29 | 2.34 |

### relu_explicit_parallel

| n_total | size_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | torch.float16 | explicit_parallel | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.bfloat16 | explicit_parallel | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.float32 | explicit_parallel | 0.00 | 0.00 | 0.02 |
| 4194304 | 4M | torch.float16 | explicit_parallel | 0.02 | 0.21 | 0.86 |
| 4194304 | 4M | torch.bfloat16 | explicit_parallel | 0.02 | 0.21 | 0.86 |
| 4194304 | 4M | torch.float32 | explicit_parallel | 0.01 | 0.38 | 3.03 |
| 16777216 | 16M | torch.float16 | explicit_parallel | 0.07 | 0.25 | 0.98 |
| 16777216 | 16M | torch.bfloat16 | explicit_parallel | 0.07 | 0.25 | 0.98 |
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
| 16777216 | 16M | torch.float16 | register_copy | 0.02 | 0.91 | 3.62 |
| 16777216 | 16M | torch.bfloat16 | register_copy | 0.02 | 0.90 | 3.62 |
| 16777216 | 16M | torch.float32 | register_copy | 0.03 | 0.49 | 3.91 |

## r4_strategy_gelu

### gelu_direct

| n_total | size_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | torch.float16 | direct | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.bfloat16 | direct | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.float32 | direct | 0.00 | 0.00 | 0.02 |
| 4194304 | 4M | torch.float16 | direct | 0.02 | 0.26 | 1.05 |
| 4194304 | 4M | torch.bfloat16 | direct | 0.02 | 0.26 | 1.05 |
| 4194304 | 4M | torch.float32 | direct | 0.02 | 0.25 | 1.96 |
| 16777216 | 16M | torch.float16 | direct | 0.06 | 0.29 | 1.17 |
| 16777216 | 16M | torch.bfloat16 | direct | 0.06 | 0.29 | 1.16 |
| 16777216 | 16M | torch.float32 | direct | 0.06 | 0.27 | 2.18 |

### gelu_explicit_parallel

| n_total | size_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | torch.float16 | explicit_parallel | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.bfloat16 | explicit_parallel | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.float32 | explicit_parallel | 0.00 | 0.00 | 0.02 |
| 4194304 | 4M | torch.float16 | explicit_parallel | 0.01 | 0.46 | 1.82 |
| 4194304 | 4M | torch.bfloat16 | explicit_parallel | 0.01 | 0.45 | 1.81 |
| 4194304 | 4M | torch.float32 | explicit_parallel | 0.01 | 0.38 | 3.03 |
| 16777216 | 16M | torch.float16 | explicit_parallel | 0.03 | 0.57 | 2.26 |
| 16777216 | 16M | torch.bfloat16 | explicit_parallel | 0.03 | 0.56 | 2.24 |
| 16777216 | 16M | torch.float32 | explicit_parallel | 0.04 | 0.46 | 3.67 |

### gelu_register_copy

| n_total | size_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | torch.float16 | register_copy | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.bfloat16 | register_copy | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.float32 | register_copy | 0.00 | 0.00 | 0.02 |
| 4194304 | 4M | torch.float16 | register_copy | 0.01 | 0.47 | 1.89 |
| 4194304 | 4M | torch.bfloat16 | register_copy | 0.01 | 0.46 | 1.86 |
| 4194304 | 4M | torch.float32 | register_copy | 0.01 | 0.39 | 3.10 |
| 16777216 | 16M | torch.float16 | register_copy | 0.03 | 0.60 | 2.40 |
| 16777216 | 16M | torch.bfloat16 | register_copy | 0.03 | 0.58 | 2.33 |
| 16777216 | 16M | torch.float32 | register_copy | 0.04 | 0.47 | 3.78 |

## r5_boundary

### relu_aligned

| n_total | align_label | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 2048000 | aligned | 0.01 | 0.17 | 0.67 |

### relu_unaligned_plus_1

| n_total | align_label | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 2048001 | unaligned_plus_1 | 0.01 | 0.17 | 0.67 |

### relu_unaligned_plus_127

| n_total | align_label | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 2048127 | unaligned_plus_127 | 0.01 | 0.17 | 0.67 |

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
| 4194304 | 4M | relu | 256 | 0.02 | 0.22 | 0.86 |
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
| 4194304 | 4M | erf | 256 | 0.01 | 0.48 | 1.90 |
| 16777216 | 16M | erf | 256 | 0.03 | 0.60 | 2.39 |

### mish_t128

| n_total | size_label | op_name | threads | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | mish | 128 | 0.00 | 0.00 | 0.01 |
| 4194304 | 4M | mish | 128 | 0.01 | 0.36 | 1.44 |
| 16777216 | 16M | mish | 128 | 0.04 | 0.42 | 1.68 |

### mish_t256

| n_total | size_label | op_name | threads | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | mish | 256 | 0.00 | 0.00 | 0.01 |
| 4194304 | 4M | mish | 256 | 0.01 | 0.35 | 1.42 |
| 16777216 | 16M | mish | 256 | 0.04 | 0.42 | 1.67 |

## r7_dtype_npt

### relu_fp32_npt4

| dtype_label | num_per_thread | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| fp32 | 4 | 0.00 | 0.22 | 1.75 |

### relu_fp32_npt8

| dtype_label | num_per_thread | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| fp32 | 8 | 0.00 | 0.22 | 1.73 |

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
| 4194304 | torch.float16 | 0.01 | 0.63 | 2.50 |
| 4194304 | torch.bfloat16 | 0.01 | 0.64 | 2.55 |
| 4194304 | torch.float32 | 0.01 | 0.39 | 3.12 |

## ada_layer_norm

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.01 | 1.41 | 2.26 |
| 4096 | 4096 | torch.bfloat16 | 0.05 | 1.62 | 2.59 |
| 1024 | 3000 | torch.float16 | 0.04 | 0.38 | 0.60 |
| 1025 | 4096 | torch.float16 | 0.01 | 1.41 | 2.26 |

### baseline

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.02 | 0.85 | 1.36 |
| 4096 | 4096 | torch.bfloat16 | 0.08 | 1.05 | 1.67 |
| 1024 | 3000 | torch.float16 | 0.02 | 0.76 | 1.21 |
| 1025 | 4096 | torch.float16 | 0.03 | 0.83 | 1.33 |

## ada_layer_norm_zero

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.02 | 1.35 | 2.25 |
| 4096 | 4096 | torch.bfloat16 | 0.06 | 1.58 | 2.63 |
| 1024 | 3000 | torch.float16 | 0.05 | 0.35 | 0.58 |
| 1025 | 4096 | torch.float16 | 0.02 | 1.33 | 2.21 |

### baseline

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.03 | 0.80 | 1.34 |
| 4096 | 4096 | torch.bfloat16 | 0.11 | 0.96 | 1.59 |
| 1024 | 3000 | torch.float16 | 0.03 | 0.72 | 1.20 |
| 1025 | 4096 | torch.float16 | 0.03 | 0.80 | 1.33 |

## argreduce

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | argmax | 3.07 | 0.00 | 0.00 |
| 1024 | 4096 | torch.bfloat16 | argmax | 3.05 | 0.00 | 0.00 |
| 4096 | 4096 | torch.float16 | argmax | 10.67 | 0.00 | 0.00 |
| 1024 | 4096 | torch.float16 | argmin | 3.46 | 0.00 | 0.00 |
| 1024 | 4096 | torch.bfloat16 | argmin | 3.47 | 0.00 | 0.00 |
| 4096 | 4096 | torch.float16 | argmin | 12.14 | 0.00 | 0.00 |

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
| 4 | 128 | (1024, 1024) | torch.float16 | True | 3.94 | 1.36 | 0.55 |
| 4 | 256 | (1024, 1024) | torch.float16 | True | 7.33 | 1.46 | 0.59 |

### torch_cudnn

| N | C | spatial | dtype | training | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 64 | () | torch.float16 | True | 0.02 | 0.00 | 0.00 |
| 8 | 64 | (32, 32) | torch.float16 | True | 0.01 | 0.37 | 0.15 |
| 4 | 128 | (32, 32) | torch.float16 | True | 0.01 | 0.39 | 0.16 |
| 4 | 256 | (28, 28) | torch.float16 | True | 0.01 | 0.56 | 0.23 |
| 4 | 128 | (1024, 1024) | torch.float16 | True | 4.13 | 1.30 | 0.52 |
| 4 | 256 | (1024, 1024) | torch.float16 | True | 7.28 | 1.48 | 0.59 |

## batch_norm_bwd

### tileops

| N | C | spatial | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 64 | () | torch.float16 | 0.01 | 0.00 | 0.00 |
| 8 | 64 | (32, 32) | torch.float16 | 0.02 | 0.26 | 0.19 |
| 4 | 128 | (32, 32) | torch.float16 | 0.02 | 0.26 | 0.20 |
| 4 | 256 | (28, 28) | torch.float16 | 0.02 | 0.32 | 0.24 |
| 4 | 128 | (1024, 1024) | torch.float16 | 6.23 | 0.69 | 0.52 |
| 4 | 256 | (1024, 1024) | torch.float16 | 11.27 | 0.76 | 0.57 |

### torch_autograd

| N | C | spatial | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 64 | () | torch.float16 | 0.02 | 0.00 | 0.00 |
| 8 | 64 | (32, 32) | torch.float16 | 0.03 | 0.16 | 0.12 |
| 4 | 128 | (32, 32) | torch.float16 | 0.02 | 0.19 | 0.14 |
| 4 | 256 | (28, 28) | torch.float16 | 0.02 | 0.27 | 0.20 |
| 4 | 128 | (1024, 1024) | torch.float16 | 12.12 | 0.35 | 0.27 |
| 4 | 256 | (1024, 1024) | torch.float16 | 18.14 | 0.47 | 0.36 |

## r1_vectorization

### add_same_shape_1d

| pattern_name | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| same_shape_1d | 1000000 | 0.00 | 0.24 | 1.47 |

### baseline_same_shape_1d

| pattern_name | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| same_shape_1d | 1000000 | 0.00 | 0.25 | 1.50 |

### add_bias_add_2d

| pattern_name | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| bias_add_2d | 1000000 | 0.00 | 0.27 | 1.08 |

### baseline_bias_add_2d

| pattern_name | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| bias_add_2d | 1000000 | 0.01 | 0.17 | 0.67 |

### add_interleaved_3d

| pattern_name | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| interleaved_3d | 1048576 | 0.00 | 0.46 | 0.93 |

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
| 1M | same_shape | direct | 1048576 | 0.01 | 0.18 | 1.11 |
| 1M | same_shape | direct | 1048576 | 0.01 | 0.18 | 1.11 |
| 1M | same_shape | direct | 1048576 | 0.01 | 0.16 | 1.95 |
| 16M | same_shape | direct | 16777216 | 0.06 | 0.29 | 1.75 |
| 16M | same_shape | direct | 16777216 | 0.06 | 0.29 | 1.75 |
| 16M | same_shape | direct | 16777216 | 0.07 | 0.25 | 3.02 |

### add_direct_bias_add_2d

| size_label | pattern_name | strategy | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4K | bias_add_2d | direct | 4096 | 0.00 | 0.00 | 0.01 |
| 4K | bias_add_2d | direct | 4096 | 0.00 | 0.00 | 0.01 |
| 4K | bias_add_2d | direct | 4096 | 0.00 | 0.00 | 0.02 |
| 1M | bias_add_2d | direct | 1048576 | 0.01 | 0.20 | 0.80 |
| 1M | bias_add_2d | direct | 1048576 | 0.01 | 0.20 | 0.80 |
| 1M | bias_add_2d | direct | 1048576 | 0.01 | 0.19 | 1.51 |
| 16M | bias_add_2d | direct | 16777216 | 0.05 | 0.32 | 1.27 |
| 16M | bias_add_2d | direct | 16777216 | 0.05 | 0.32 | 1.27 |
| 16M | bias_add_2d | direct | 16777216 | 0.06 | 0.29 | 2.30 |

### add_direct_interleaved_3d

| size_label | pattern_name | strategy | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4K | interleaved_3d | direct | 4096 | 0.00 | 0.00 | 0.01 |
| 4K | interleaved_3d | direct | 4096 | 0.00 | 0.00 | 0.01 |
| 4K | interleaved_3d | direct | 4096 | 0.00 | 0.00 | 0.01 |
| 1M | interleaved_3d | direct | 1048576 | 0.00 | 0.23 | 0.47 |
| 1M | interleaved_3d | direct | 1048576 | 0.00 | 0.23 | 0.46 |
| 1M | interleaved_3d | direct | 1048576 | 0.00 | 0.22 | 0.90 |
| 16M | interleaved_3d | direct | 16777216 | 0.04 | 0.40 | 0.80 |
| 16M | interleaved_3d | direct | 16777216 | 0.04 | 0.40 | 0.80 |
| 16M | interleaved_3d | direct | 16777216 | 0.04 | 0.39 | 1.57 |

### add_explicit_parallel_same_shape

| size_label | pattern_name | strategy | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4K | same_shape | explicit_parallel | 4096 | 0.00 | 0.00 | 0.01 |
| 4K | same_shape | explicit_parallel | 4096 | 0.00 | 0.00 | 0.01 |
| 4K | same_shape | explicit_parallel | 4096 | 0.00 | 0.00 | 0.03 |
| 1M | same_shape | explicit_parallel | 1048576 | 0.00 | 0.27 | 1.61 |
| 1M | same_shape | explicit_parallel | 1048576 | 0.00 | 0.27 | 1.60 |
| 1M | same_shape | explicit_parallel | 1048576 | 0.01 | 0.19 | 2.25 |
| 16M | same_shape | explicit_parallel | 16777216 | 0.03 | 0.62 | 3.71 |
| 16M | same_shape | explicit_parallel | 16777216 | 0.03 | 0.62 | 3.71 |
| 16M | same_shape | explicit_parallel | 16777216 | 0.05 | 0.33 | 4.01 |

### add_explicit_parallel_bias_add_2d

| size_label | pattern_name | strategy | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4K | bias_add_2d | explicit_parallel | 4096 | 0.00 | 0.00 | 0.01 |
| 4K | bias_add_2d | explicit_parallel | 4096 | 0.00 | 0.00 | 0.01 |
| 4K | bias_add_2d | explicit_parallel | 4096 | 0.00 | 0.00 | 0.02 |
| 1M | bias_add_2d | explicit_parallel | 1048576 | 0.00 | 0.32 | 1.27 |
| 1M | bias_add_2d | explicit_parallel | 1048576 | 0.00 | 0.32 | 1.27 |
| 1M | bias_add_2d | explicit_parallel | 1048576 | 0.00 | 0.24 | 1.90 |
| 16M | bias_add_2d | explicit_parallel | 16777216 | 0.02 | 0.90 | 3.61 |
| 16M | bias_add_2d | explicit_parallel | 16777216 | 0.02 | 0.90 | 3.61 |
| 16M | bias_add_2d | explicit_parallel | 16777216 | 0.03 | 0.49 | 3.91 |

### add_explicit_parallel_interleaved_3d

| size_label | pattern_name | strategy | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4K | interleaved_3d | explicit_parallel | 4096 | 0.00 | 0.00 | 0.01 |
| 4K | interleaved_3d | explicit_parallel | 4096 | 0.00 | 0.00 | 0.01 |
| 4K | interleaved_3d | explicit_parallel | 4096 | 0.00 | 0.00 | 0.01 |
| 1M | interleaved_3d | explicit_parallel | 1048576 | 0.00 | 0.47 | 0.95 |
| 1M | interleaved_3d | explicit_parallel | 1048576 | 0.00 | 0.47 | 0.95 |
| 1M | interleaved_3d | explicit_parallel | 1048576 | 0.00 | 0.38 | 1.53 |
| 16M | interleaved_3d | explicit_parallel | 16777216 | 0.01 | 1.71 | 3.43 |
| 16M | interleaved_3d | explicit_parallel | 16777216 | 0.01 | 1.71 | 3.43 |
| 16M | interleaved_3d | explicit_parallel | 16777216 | 0.02 | 0.96 | 3.83 |

## r4_where

### tileops_where

| n_total | size_label | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 4096 | 4K | 0.00 | 0.00 | 0.01 |
| 1048576 | 1M | 0.00 | 0.24 | 1.68 |
| 16777216 | 16M | 0.03 | 0.53 | 3.71 |

### baseline_where

| n_total | size_label | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 4096 | 4K | 0.00 | 0.00 | 0.01 |
| 1048576 | 1M | 0.00 | 0.24 | 1.65 |
| 16777216 | 16M | 0.03 | 0.53 | 3.74 |

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
| 4194304 | torch.float16 | 0.01 | 0.46 | 2.79 |
| 4194304 | torch.bfloat16 | 0.01 | 0.46 | 2.79 |
| 4194304 | torch.float32 | 0.02 | 0.28 | 3.31 |

## sub

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| sub | torch.float16 | torch.float16 | 0.01 | 0.47 | 2.83 |
| sub | torch.float16 | torch.float16 | 0.02 | 0.57 | 3.43 |
| sub | torch.float16 | torch.float16 | 0.03 | 0.64 | 3.83 |

### baseline

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| sub | torch.float16 | torch.float16 | 0.01 | 0.46 | 2.79 |
| sub | torch.float16 | torch.float16 | 0.02 | 0.57 | 3.41 |
| sub | torch.float16 | torch.float16 | 0.03 | 0.64 | 3.85 |

## mul

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| mul | torch.float16 | torch.float16 | 0.01 | 0.47 | 2.83 |
| mul | torch.float16 | torch.float16 | 0.02 | 0.57 | 3.43 |
| mul | torch.float16 | torch.float16 | 0.03 | 0.64 | 3.83 |

### baseline

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| mul | torch.float16 | torch.float16 | 0.01 | 0.46 | 2.79 |
| mul | torch.float16 | torch.float16 | 0.02 | 0.57 | 3.41 |
| mul | torch.float16 | torch.float16 | 0.03 | 0.64 | 3.85 |

## div

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| div | torch.float16 | torch.float16 | 0.01 | 0.45 | 2.72 |
| div | torch.float16 | torch.float16 | 0.02 | 0.57 | 3.44 |
| div | torch.float16 | torch.float16 | 0.03 | 0.63 | 3.77 |

### baseline

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| div | torch.float16 | torch.float16 | 0.01 | 0.44 | 2.67 |
| div | torch.float16 | torch.float16 | 0.02 | 0.57 | 3.40 |
| div | torch.float16 | torch.float16 | 0.03 | 0.62 | 3.74 |

## remainder

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| remainder | torch.float16 | torch.float16 | 0.01 | 0.45 | 2.67 |
| remainder | torch.float16 | torch.float16 | 0.02 | 0.57 | 3.42 |

### baseline

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| remainder | torch.float16 | torch.float16 | 0.01 | 0.40 | 2.43 |
| remainder | torch.float16 | torch.float16 | 0.02 | 0.51 | 3.05 |

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
| floor_divide | torch.float16 | torch.float16 | 0.02 | 0.57 | 3.42 |

### baseline

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| floor_divide | torch.float16 | torch.float16 | 0.02 | 0.18 | 1.10 |
| floor_divide | torch.float16 | torch.float16 | 0.05 | 0.21 | 1.24 |

## lerp

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| lerp | torch.float16 | torch.float16 | 0.01 | 0.47 | 2.82 |
| lerp | torch.float16 | torch.float16 | 0.02 | 0.57 | 3.43 |

### baseline

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| lerp | torch.float16 | torch.float16 | 0.01 | 0.47 | 2.82 |
| lerp | torch.float16 | torch.float16 | 0.02 | 0.57 | 3.43 |

## maximum

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| maximum | torch.float16 | torch.float16 | 0.01 | 0.46 | 2.77 |
| maximum | torch.float16 | torch.float16 | 0.02 | 0.56 | 3.39 |
| maximum | torch.float16 | torch.float16 | 0.03 | 0.64 | 3.81 |

### baseline

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| maximum | torch.float16 | torch.float16 | 0.01 | 0.46 | 2.75 |
| maximum | torch.float16 | torch.float16 | 0.02 | 0.57 | 3.39 |
| maximum | torch.float16 | torch.float16 | 0.03 | 0.63 | 3.81 |

## minimum

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| minimum | torch.float16 | torch.float16 | 0.01 | 0.46 | 2.77 |
| minimum | torch.float16 | torch.float16 | 0.02 | 0.57 | 3.39 |
| minimum | torch.float16 | torch.float16 | 0.03 | 0.64 | 3.81 |

### baseline

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| minimum | torch.float16 | torch.float16 | 0.01 | 0.46 | 2.76 |
| minimum | torch.float16 | torch.float16 | 0.02 | 0.57 | 3.39 |
| minimum | torch.float16 | torch.float16 | 0.03 | 0.63 | 3.81 |

## cmp_eq

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| eq | torch.float16 | 0.02 | 0.22 | 1.11 |
| eq | torch.float16 | 0.04 | 0.26 | 1.31 |

### baseline

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| eq | torch.float16 | 0.01 | 0.52 | 2.60 |
| eq | torch.float16 | 0.02 | 0.64 | 3.22 |

## cmp_ne

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| ne | torch.float16 | 0.02 | 0.22 | 1.11 |

### baseline

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| ne | torch.float16 | 0.01 | 0.52 | 2.60 |

## cmp_gt

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| gt | torch.float16 | 0.02 | 0.22 | 1.10 |

### baseline

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| gt | torch.float16 | 0.01 | 0.51 | 2.57 |

## cmp_lt

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| lt | torch.float16 | 0.02 | 0.22 | 1.10 |

### baseline

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| lt | torch.float16 | 0.01 | 0.51 | 2.57 |

## cmp_ge

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| ge | torch.float16 | 0.02 | 0.22 | 1.10 |

### baseline

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| ge | torch.float16 | 0.01 | 0.52 | 2.58 |

## cmp_le

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| le | torch.float16 | 0.02 | 0.22 | 1.10 |

### baseline

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| le | torch.float16 | 0.01 | 0.52 | 2.58 |

## logical_and

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| logical_and | torch.float16 | 0.02 | 0.22 | 1.10 |
| logical_and | torch.float16 | 0.04 | 0.26 | 1.31 |

### baseline

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| logical_and | torch.float16 | 0.01 | 0.71 | 3.54 |
| logical_and | torch.float16 | 0.01 | 0.93 | 4.66 |

## logical_or

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| logical_or | torch.float16 | 0.02 | 0.22 | 1.11 |
| logical_or | torch.float16 | 0.04 | 0.26 | 1.31 |

### baseline

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| logical_or | torch.float16 | 0.01 | 0.71 | 3.56 |
| logical_or | torch.float16 | 0.01 | 0.94 | 4.69 |

## bitwise_and

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| bitwise_and | torch.int32 | 0.02 | 0.27 | 3.22 |
| bitwise_and | torch.int32 | 0.03 | 0.31 | 3.75 |

### baseline

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| bitwise_and | torch.int32 | 0.02 | 0.28 | 3.31 |
| bitwise_and | torch.int32 | 0.03 | 0.32 | 3.83 |

## bitwise_or

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| bitwise_or | torch.int32 | 0.02 | 0.27 | 3.22 |

### baseline

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| bitwise_or | torch.int32 | 0.02 | 0.28 | 3.30 |

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
| gelu_and_mul | 1024 | 4096 | torch.float16 | 0.01 | 0.78 | 2.34 |
| gelu_and_mul | 1024 | 10240 | torch.float16 | 0.02 | 0.92 | 2.75 |
| gelu_and_mul | 1024 | 20480 | torch.float16 | 0.04 | 1.04 | 3.11 |

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
| gelu_tanh_and_mul | 1024 | 4096 | torch.float16 | 0.01 | 0.91 | 2.74 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.float16 | 0.02 | 1.14 | 3.42 |
| gelu_tanh_and_mul | 1024 | 20480 | torch.float16 | 0.03 | 1.24 | 3.72 |

### baseline

| op_name | M | N | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| gelu_tanh_and_mul | 1024 | 4096 | torch.float16 | 0.03 | 0.28 | 0.83 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.float16 | 0.07 | 0.31 | 0.94 |
| gelu_tanh_and_mul | 1024 | 20480 | torch.float16 | 0.13 | 0.33 | 1.00 |

## silu_and_mul_strategy

### tileops_direct

| op_name | M | N | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| silu_and_mul | 1024 | 4096 | torch.float16 | direct | 0.02 | 0.49 | 1.48 |
| silu_and_mul | 1024 | 4096 | torch.bfloat16 | direct | 0.02 | 0.49 | 1.48 |
| silu_and_mul | 1024 | 4096 | torch.float32 | direct | 0.02 | 0.42 | 2.55 |
| silu_and_mul | 1024 | 10240 | torch.float16 | direct | 0.04 | 0.53 | 1.60 |
| silu_and_mul | 1024 | 10240 | torch.bfloat16 | direct | 0.04 | 0.53 | 1.60 |
| silu_and_mul | 1024 | 10240 | torch.float32 | direct | 0.04 | 0.47 | 2.82 |
| silu_and_mul | 4096 | 4096 | torch.float16 | direct | 0.06 | 0.55 | 1.66 |
| silu_and_mul | 4096 | 4096 | torch.bfloat16 | direct | 0.06 | 0.55 | 1.66 |
| silu_and_mul | 4096 | 4096 | torch.float32 | direct | 0.07 | 0.49 | 2.94 |

### tileops_explicit_parallel

| op_name | M | N | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| silu_and_mul | 1024 | 4096 | torch.float16 | explicit_parallel | 0.01 | 0.85 | 2.54 |
| silu_and_mul | 1024 | 4096 | torch.bfloat16 | explicit_parallel | 0.01 | 0.91 | 2.73 |
| silu_and_mul | 1024 | 4096 | torch.float32 | explicit_parallel | 0.02 | 0.55 | 3.28 |
| silu_and_mul | 1024 | 10240 | torch.float16 | explicit_parallel | 0.02 | 0.99 | 2.98 |
| silu_and_mul | 1024 | 10240 | torch.bfloat16 | explicit_parallel | 0.02 | 1.15 | 3.45 |
| silu_and_mul | 1024 | 10240 | torch.float32 | explicit_parallel | 0.03 | 0.63 | 3.80 |
| silu_and_mul | 4096 | 4096 | torch.float16 | explicit_parallel | 0.03 | 1.06 | 3.18 |
| silu_and_mul | 4096 | 4096 | torch.bfloat16 | explicit_parallel | 0.03 | 1.23 | 3.68 |
| silu_and_mul | 4096 | 4096 | torch.float32 | explicit_parallel | 0.05 | 0.67 | 4.00 |

## gelu_and_mul_strategy

### tileops_direct

| op_name | M | N | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gelu_and_mul | 1024 | 4096 | torch.float16 | direct | 0.02 | 0.48 | 1.45 |
| gelu_and_mul | 1024 | 4096 | torch.bfloat16 | direct | 0.02 | 0.48 | 1.45 |
| gelu_and_mul | 1024 | 4096 | torch.float32 | direct | 0.02 | 0.43 | 2.57 |
| gelu_and_mul | 1024 | 10240 | torch.float16 | direct | 0.04 | 0.53 | 1.58 |
| gelu_and_mul | 1024 | 10240 | torch.bfloat16 | direct | 0.04 | 0.53 | 1.58 |
| gelu_and_mul | 1024 | 10240 | torch.float32 | direct | 0.04 | 0.47 | 2.84 |
| gelu_and_mul | 4096 | 4096 | torch.float16 | direct | 0.06 | 0.54 | 1.63 |
| gelu_and_mul | 4096 | 4096 | torch.bfloat16 | direct | 0.06 | 0.54 | 1.63 |
| gelu_and_mul | 4096 | 4096 | torch.float32 | direct | 0.07 | 0.49 | 2.95 |

### tileops_explicit_parallel

| op_name | M | N | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gelu_and_mul | 1024 | 4096 | torch.float16 | explicit_parallel | 0.01 | 0.78 | 2.34 |
| gelu_and_mul | 1024 | 4096 | torch.bfloat16 | explicit_parallel | 0.01 | 0.75 | 2.24 |
| gelu_and_mul | 1024 | 4096 | torch.float32 | explicit_parallel | 0.02 | 0.55 | 3.28 |
| gelu_and_mul | 1024 | 10240 | torch.float16 | explicit_parallel | 0.02 | 0.92 | 2.75 |
| gelu_and_mul | 1024 | 10240 | torch.bfloat16 | explicit_parallel | 0.02 | 0.91 | 2.74 |
| gelu_and_mul | 1024 | 10240 | torch.float32 | explicit_parallel | 0.03 | 0.62 | 3.72 |
| gelu_and_mul | 4096 | 4096 | torch.float16 | explicit_parallel | 0.03 | 1.00 | 2.99 |
| gelu_and_mul | 4096 | 4096 | torch.bfloat16 | explicit_parallel | 0.03 | 0.96 | 2.88 |
| gelu_and_mul | 4096 | 4096 | torch.float32 | explicit_parallel | 0.05 | 0.67 | 3.99 |

## gelu_tanh_and_mul_strategy

### tileops_direct

| op_name | M | N | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gelu_tanh_and_mul | 1024 | 4096 | torch.float16 | direct | 0.02 | 0.49 | 1.48 |
| gelu_tanh_and_mul | 1024 | 4096 | torch.bfloat16 | direct | 0.02 | 0.49 | 1.48 |
| gelu_tanh_and_mul | 1024 | 4096 | torch.float32 | direct | 0.02 | 0.43 | 2.59 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.float16 | direct | 0.04 | 0.54 | 1.61 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.bfloat16 | direct | 0.04 | 0.54 | 1.61 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.float32 | direct | 0.04 | 0.48 | 2.87 |
| gelu_tanh_and_mul | 4096 | 4096 | torch.float16 | direct | 0.06 | 0.56 | 1.67 |
| gelu_tanh_and_mul | 4096 | 4096 | torch.bfloat16 | direct | 0.06 | 0.56 | 1.67 |
| gelu_tanh_and_mul | 4096 | 4096 | torch.float32 | direct | 0.07 | 0.50 | 2.99 |

### tileops_explicit_parallel

| op_name | M | N | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gelu_tanh_and_mul | 1024 | 4096 | torch.float16 | explicit_parallel | 0.01 | 0.91 | 2.74 |
| gelu_tanh_and_mul | 1024 | 4096 | torch.bfloat16 | explicit_parallel | 0.01 | 0.91 | 2.72 |
| gelu_tanh_and_mul | 1024 | 4096 | torch.float32 | explicit_parallel | 0.02 | 0.55 | 3.30 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.float16 | explicit_parallel | 0.02 | 1.14 | 3.42 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.bfloat16 | explicit_parallel | 0.02 | 1.12 | 3.36 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.float32 | explicit_parallel | 0.03 | 0.64 | 3.81 |
| gelu_tanh_and_mul | 4096 | 4096 | torch.float16 | explicit_parallel | 0.03 | 1.20 | 3.59 |
| gelu_tanh_and_mul | 4096 | 4096 | torch.bfloat16 | explicit_parallel | 0.03 | 1.18 | 3.54 |
| gelu_tanh_and_mul | 4096 | 4096 | torch.float32 | explicit_parallel | 0.05 | 0.67 | 4.01 |

## sub_bcast

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| sub | torch.float16 | 0.01 | 0.65 | 2.60 |
| sub | torch.float16 | 0.01 | 0.83 | 3.31 |
| sub | torch.float16 | 0.02 | 0.93 | 3.72 |

### baseline

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| sub | torch.float16 | 0.01 | 0.29 | 1.15 |
| sub | torch.float16 | 0.03 | 0.34 | 1.35 |
| sub | torch.float16 | 0.06 | 0.36 | 1.46 |

## mul_bcast

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| mul | torch.float16 | 0.01 | 0.65 | 2.60 |
| mul | torch.float16 | 0.01 | 0.83 | 3.30 |
| mul | torch.float16 | 0.02 | 0.93 | 3.72 |

### baseline

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| mul | torch.float16 | 0.01 | 0.29 | 1.15 |
| mul | torch.float16 | 0.03 | 0.34 | 1.35 |
| mul | torch.float16 | 0.06 | 0.37 | 1.46 |

## div_bcast

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| div | torch.float16 | 0.01 | 0.60 | 2.42 |
| div | torch.float16 | 0.01 | 0.77 | 3.08 |
| div | torch.float16 | 0.02 | 0.85 | 3.38 |

### baseline

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| div | torch.float16 | 0.02 | 0.27 | 1.08 |
| div | torch.float16 | 0.03 | 0.32 | 1.26 |
| div | torch.float16 | 0.06 | 0.34 | 1.36 |

## binary_strategy

### add_direct

| n_total | shape_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4194304 | 1024x4096 | torch.float16 | direct | 0.02 | 0.26 | 1.55 |
| 4194304 | 1024x4096 | torch.bfloat16 | direct | 0.02 | 0.26 | 1.55 |
| 4194304 | 1024x4096 | torch.float32 | direct | 0.02 | 0.22 | 2.59 |
| 10485760 | 1024x10240 | torch.float16 | direct | 0.04 | 0.28 | 1.68 |
| 10485760 | 1024x10240 | torch.bfloat16 | direct | 0.04 | 0.28 | 1.68 |
| 10485760 | 1024x10240 | torch.float32 | direct | 0.04 | 0.24 | 2.91 |
| 20971520 | 1024x20480 | torch.float16 | direct | 0.07 | 0.30 | 1.78 |
| 20971520 | 1024x20480 | torch.bfloat16 | direct | 0.07 | 0.30 | 1.78 |
| 20971520 | 1024x20480 | torch.float32 | direct | 0.08 | 0.26 | 3.07 |

### add_explicit_parallel

| n_total | shape_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4194304 | 1024x4096 | torch.float16 | explicit_parallel | 0.01 | 0.47 | 2.83 |
| 4194304 | 1024x4096 | torch.bfloat16 | explicit_parallel | 0.01 | 0.47 | 2.82 |
| 4194304 | 1024x4096 | torch.float32 | explicit_parallel | 0.02 | 0.27 | 3.29 |
| 10485760 | 1024x10240 | torch.float16 | explicit_parallel | 0.02 | 0.57 | 3.43 |
| 10485760 | 1024x10240 | torch.bfloat16 | explicit_parallel | 0.02 | 0.57 | 3.43 |
| 10485760 | 1024x10240 | torch.float32 | explicit_parallel | 0.03 | 0.32 | 3.83 |
| 20971520 | 1024x20480 | torch.float16 | explicit_parallel | 0.03 | 0.64 | 3.83 |
| 20971520 | 1024x20480 | torch.bfloat16 | explicit_parallel | 0.03 | 0.64 | 3.83 |
| 20971520 | 1024x20480 | torch.float32 | explicit_parallel | 0.06 | 0.34 | 4.09 |

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
| 1024 | 4096 | torch.float16 | cumsum | 0.10 | 0.04 | 0.17 |
| 1024 | 4096 | torch.bfloat16 | cumsum | 0.10 | 0.04 | 0.17 |
| 4096 | 4096 | torch.float16 | cumsum | 0.26 | 0.07 | 0.26 |
| 1024 | 4096 | torch.float16 | cumprod | 0.10 | 0.04 | 0.17 |
| 1024 | 4096 | torch.bfloat16 | cumprod | 0.10 | 0.04 | 0.17 |
| 4096 | 4096 | torch.float16 | cumprod | 0.26 | 0.07 | 0.26 |

## dsa_decode

### tileops

| batch | heads | seq_len_q | seq_len_kv | dim | dim_tail | topk | stride_kv | heads_kv | q_start_index_s | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 128 | 1024 | 2048 | 512 | 64 | 2048 | 1 | 1 | 1024 | torch.float16 | 1.14 | 510.70 | 0.26 |
| 1 | 128 | 512 | 4096 | 512 | 64 | 1024 | 1 | 1 | 512 | torch.float16 | 0.32 | 461.56 | 0.47 |

### baseline

| batch | heads | seq_len_q | seq_len_kv | dim | dim_tail | topk | stride_kv | heads_kv | q_start_index_s | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 128 | 1024 | 2048 | 512 | 64 | 2048 | 1 | 1 | 1024 | torch.float16 | 16.51 | 35.38 | 0.02 |
| 1 | 128 | 512 | 4096 | 512 | 64 | 1024 | 1 | 1 | 512 | torch.float16 | 16.73 | 8.73 | 0.01 |

## mla_decode

### tileops

| batch | heads | heads_kv | seq_len_kv | dim | dim_pe | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 128 | 1 | 8192 | 512 | 64 | torch.float16 | 0.16 | 443.30 | 1.86 |
| 16 | 128 | 1 | 4096 | 512 | 64 | torch.float16 | 0.05 | 379.02 | 1.62 |

### baseline

| batch | heads | heads_kv | seq_len_kv | dim | dim_pe | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 128 | 1 | 8192 | 512 | 64 | torch.float16 | 0.72 | 101.17 | 0.42 |
| 16 | 128 | 1 | 4096 | 512 | 64 | torch.float16 | 0.19 | 95.69 | 0.41 |

## nsa_cmp_fwd

### tileops

| seq_num | c_seq_len | heads | dim_k | dim_v | group | scale | bc | bs | bk | bv | dtype | accum_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 9 | 8192 | 32 | 128 | 128 | 16 | 0.08838834764831845 | 32 | 32 | 128 | 128 | torch.float16 | torch.float32 | 0.32 | 54.25 | 6.99 |
| 16 | 16384 | 32 | 128 | 128 | 16 | 0.08838834764831845 | 32 | 32 | 128 | 128 | torch.float16 | torch.float32 | 157.72 | 0.44 | 0.06 |

### baseline

| seq_num | c_seq_len | heads | dim_k | dim_v | group | scale | bc | bs | bk | bv | dtype | accum_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 9 | 8192 | 32 | 128 | 128 | 16 | 0.08838834764831845 | 32 | 32 | 128 | 128 | torch.float16 | torch.float32 | 305.88 | 0.06 | 0.01 |
| 16 | 16384 | 32 | 128 | 128 | 16 | 0.08838834764831845 | 32 | 32 | 128 | 128 | torch.float16 | torch.float32 | 594.30 | 0.12 | 0.01 |

## nsa_fwd

### tileops

| batch | heads | c_seq_len | dim | is_causal | scale | block_size | groups | selected_blocks | dtype | accum_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 16 | 1024 | 64 | True | 0.1 | 32 | 16 | 1 | torch.float16 | torch.float32 | 147.70 | 0.00 | 0.00 |
| 4 | 16 | 8192 | 64 | True | 0.1 | 32 | 16 | 1 | torch.float16 | torch.float32 | 145.07 | 0.01 | 0.00 |
| 2 | 16 | 8192 | 64 | True | 0.1 | 32 | 16 | 4 | torch.float16 | torch.float32 | 143.73 | 0.03 | 0.00 |

### baseline

| batch | heads | c_seq_len | dim | is_causal | scale | block_size | groups | selected_blocks | dtype | accum_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 16 | 1024 | 64 | True | 0.1 | 32 | 16 | 1 | torch.float16 | torch.float32 | 64.79 | 0.00 | 0.00 |
| 4 | 16 | 8192 | 64 | True | 0.1 | 32 | 16 | 1 | torch.float16 | torch.float32 | 518.28 | 0.00 | 0.00 |
| 2 | 16 | 8192 | 64 | True | 0.1 | 32 | 16 | 4 | torch.float16 | torch.float32 | 478.20 | 0.01 | 0.00 |

## nsa_topk

### tileops

| seq_num | c_seq_len | heads | dim | group | scale | selected_block_num | bc | bs | bk | dtype | accum_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 1024 | 32 | 128 | 16 | 1 | 16 | 32 | 32 | 128 | torch.float16 | torch.float32 | 249.72 | 0.00 | 0.00 |
| 3 | 512 | 32 | 128 | 16 | 1 | 16 | 32 | 32 | 128 | torch.float16 | torch.float32 | 391.31 | 0.00 | 0.00 |
| 9 | 8192 | 32 | 128 | 16 | 1 | 16 | 32 | 32 | 128 | torch.float16 | torch.float32 | 249.74 | 0.07 | 0.00 |

### baseline

| seq_num | c_seq_len | heads | dim | group | scale | selected_block_num | bc | bs | bk | dtype | accum_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 1024 | 32 | 128 | 16 | 1 | 16 | 32 | 32 | 128 | torch.float16 | torch.float32 | 240.70 | 0.00 | 0.00 |
| 3 | 512 | 32 | 128 | 16 | 1 | 16 | 32 | 32 | 128 | torch.float16 | torch.float32 | 119.67 | 0.00 | 0.00 |
| 9 | 8192 | 32 | 128 | 16 | 1 | 16 | 32 | 32 | 128 | torch.float16 | torch.float32 | 2499.59 | 0.01 | 0.00 |

## dropout

### tileops

| dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| torch.float16 | 4194304 | 0.01 | 0.64 | 2.56 |
| torch.bfloat16 | 4194304 | 0.01 | 0.64 | 2.56 |
| torch.float32 | 4194304 | 0.01 | 0.39 | 3.10 |
| torch.float16 | 10485760 | 0.01 | 0.83 | 3.32 |
| torch.bfloat16 | 10485760 | 0.01 | 0.83 | 3.31 |
| torch.float32 | 10485760 | 0.02 | 0.47 | 3.79 |
| torch.float16 | 20971520 | 0.02 | 0.95 | 3.79 |
| torch.bfloat16 | 20971520 | 0.02 | 0.95 | 3.79 |
| torch.float32 | 20971520 | 0.04 | 0.51 | 4.11 |

### baseline

| dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| torch.float16 | 4194304 | 0.01 | 0.38 | 1.53 |
| torch.bfloat16 | 4194304 | 0.01 | 0.38 | 1.51 |
| torch.float32 | 4194304 | 0.01 | 0.28 | 2.25 |
| torch.float16 | 10485760 | 0.02 | 0.50 | 2.01 |
| torch.bfloat16 | 10485760 | 0.02 | 0.49 | 1.97 |
| torch.float32 | 10485760 | 0.03 | 0.33 | 2.60 |
| torch.float16 | 20971520 | 0.04 | 0.55 | 2.20 |
| torch.bfloat16 | 20971520 | 0.04 | 0.54 | 2.16 |
| torch.float32 | 20971520 | 0.06 | 0.34 | 2.76 |

## relu_fp8

### tileops

| op_name | n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| relu_fp8 | 8388608 | torch.float8_e4m3fn | 0.01 | 0.57 | 1.14 |
| relu_fp8 | 8388608 | torch.float8_e5m2 | 0.02 | 0.45 | 0.91 |
| relu_fp8 | 67108864 | torch.float8_e4m3fn | 0.09 | 0.71 | 1.43 |
| relu_fp8 | 67108864 | torch.float8_e5m2 | 0.12 | 0.54 | 1.09 |
| relu_fp8 | 134217728 | torch.float8_e4m3fn | 0.18 | 0.74 | 1.47 |
| relu_fp8 | 134217728 | torch.float8_e5m2 | 0.24 | 0.56 | 1.13 |

### baseline

| op_name | n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| relu_fp8 | 8388608 | torch.float8_e4m3fn | 0.05 | 0.19 | 0.37 |
| relu_fp8 | 8388608 | torch.float8_e5m2 | 0.04 | 0.19 | 0.38 |
| relu_fp8 | 67108864 | torch.float8_e4m3fn | 0.34 | 0.20 | 0.40 |
| relu_fp8 | 67108864 | torch.float8_e5m2 | 0.33 | 0.20 | 0.41 |
| relu_fp8 | 134217728 | torch.float8_e4m3fn | 0.65 | 0.21 | 0.41 |
| relu_fp8 | 134217728 | torch.float8_e5m2 | 0.63 | 0.21 | 0.42 |

## exp_fp8

### tileops

| op_name | n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| exp_fp8 | 8388608 | torch.float8_e4m3fn | 0.01 | 0.98 | 1.95 |
| exp_fp8 | 8388608 | torch.float8_e5m2 | 0.02 | 0.46 | 0.91 |
| exp_fp8 | 67108864 | torch.float8_e4m3fn | 0.05 | 1.33 | 2.65 |
| exp_fp8 | 67108864 | torch.float8_e5m2 | 0.12 | 0.54 | 1.08 |
| exp_fp8 | 134217728 | torch.float8_e4m3fn | 0.10 | 1.38 | 2.77 |
| exp_fp8 | 134217728 | torch.float8_e5m2 | 0.24 | 0.56 | 1.12 |

### baseline

| op_name | n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| exp_fp8 | 8388608 | torch.float8_e4m3fn | 0.05 | 0.19 | 0.37 |
| exp_fp8 | 8388608 | torch.float8_e5m2 | 0.04 | 0.19 | 0.38 |
| exp_fp8 | 67108864 | torch.float8_e4m3fn | 0.33 | 0.20 | 0.40 |
| exp_fp8 | 67108864 | torch.float8_e5m2 | 0.32 | 0.21 | 0.42 |
| exp_fp8 | 134217728 | torch.float8_e4m3fn | 0.64 | 0.21 | 0.42 |
| exp_fp8 | 134217728 | torch.float8_e5m2 | 0.62 | 0.22 | 0.43 |

## add_fp8

### tileops

| op_name | n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| add_fp8 | 8388608 | torch.float8_e4m3fn | 0.01 | 0.85 | 2.56 |
| add_fp8 | 8388608 | torch.float8_e5m2 | 0.02 | 0.41 | 1.24 |
| add_fp8 | 67108864 | torch.float8_e4m3fn | 0.06 | 1.15 | 3.46 |
| add_fp8 | 67108864 | torch.float8_e5m2 | 0.13 | 0.50 | 1.50 |
| add_fp8 | 134217728 | torch.float8_e4m3fn | 0.11 | 1.21 | 3.62 |
| add_fp8 | 134217728 | torch.float8_e5m2 | 0.26 | 0.52 | 1.56 |

### baseline

| op_name | n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| add_fp8 | 8388608 | torch.float8_e4m3fn | 0.08 | 0.11 | 0.32 |
| add_fp8 | 8388608 | torch.float8_e5m2 | 0.07 | 0.11 | 0.34 |
| add_fp8 | 67108864 | torch.float8_e4m3fn | 0.56 | 0.12 | 0.36 |
| add_fp8 | 67108864 | torch.float8_e5m2 | 0.54 | 0.12 | 0.37 |
| add_fp8 | 134217728 | torch.float8_e4m3fn | 1.07 | 0.13 | 0.38 |
| add_fp8 | 134217728 | torch.float8_e5m2 | 1.10 | 0.12 | 0.37 |

## silu_and_mul_fp8

### tileops

| op_name | M | N | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| silu_and_mul_fp8 | 2048 | 11008 | torch.float8_e4m3fn | 0.04 | 2.59 | 1.56 |
| silu_and_mul_fp8 | 2048 | 11008 | torch.float8_e5m2 | 0.06 | 1.78 | 1.07 |
| silu_and_mul_fp8 | 16384 | 11008 | torch.float8_e4m3fn | 0.30 | 2.98 | 1.79 |
| silu_and_mul_fp8 | 16384 | 11008 | torch.float8_e5m2 | 0.45 | 2.03 | 1.22 |
| silu_and_mul_fp8 | 16384 | 28672 | torch.float8_e4m3fn | 0.75 | 3.15 | 1.89 |
| silu_and_mul_fp8 | 16384 | 28672 | torch.float8_e5m2 | 1.11 | 2.11 | 1.27 |

### baseline

| op_name | M | N | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| silu_and_mul_fp8 | 2048 | 11008 | torch.float8_e4m3fn | 0.28 | 0.41 | 0.24 |
| silu_and_mul_fp8 | 2048 | 11008 | torch.float8_e5m2 | 0.29 | 0.39 | 0.24 |
| silu_and_mul_fp8 | 16384 | 11008 | torch.float8_e4m3fn | 2.00 | 0.45 | 0.27 |
| silu_and_mul_fp8 | 16384 | 11008 | torch.float8_e5m2 | 1.92 | 0.47 | 0.28 |
| silu_and_mul_fp8 | 16384 | 28672 | torch.float8_e4m3fn | 5.96 | 0.39 | 0.24 |
| silu_and_mul_fp8 | 16384 | 28672 | torch.float8_e5m2 | 3.96 | 0.59 | 0.36 |

## engram_gate_conv_bwd

### tileops

| M | seq_len | d | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 256 | torch.float16 | 435.24 | 0.00 | 0.00 |
| 2 | 64 | 512 | torch.float16 | 543.94 | 0.00 | 0.00 |
| 1 | 128 | 256 | torch.bfloat16 | 711.46 | 0.00 | 0.00 |
| 2 | 16 | 256 | torch.bfloat16 | 432.89 | 0.00 | 0.00 |

## engram_decode

### tileops

| batch | d_mem | d | max_conv_len | conv_kernel_size | dilation | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 512 | 256 | 12 | 4 | 3 | torch.float16 | 0.06 | 0.01 | 0.01 |
| 4 | 1024 | 512 | 20 | 4 | 5 | torch.float16 | 251.86 | 0.00 | 0.00 |
| 8 | 512 | 256 | 18 | 4 | 3 | torch.bfloat16 | 225.94 | 0.00 | 0.00 |

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
| 1 | 32 | 256 | torch.float16 | 500.09 | 0.00 | 0.00 |
| 2 | 64 | 512 | torch.float16 | 276.77 | 0.00 | 0.00 |
| 1 | 128 | 256 | torch.bfloat16 | 0.01 | 0.15 | 0.06 |
| 2 | 16 | 256 | torch.bfloat16 | 202.28 | 0.00 | 0.00 |

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
| 64 | torch.complex64 | 190.58 | 0.00 | 0.00 |
| 128 | torch.complex64 | 197.70 | 0.00 | 0.00 |
| 256 | torch.complex64 | 201.91 | 0.00 | 0.00 |
| 512 | torch.complex64 | 167.22 | 0.00 | 0.00 |
| 1024 | torch.complex64 | 192.50 | 0.00 | 0.00 |

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
| 64 | torch.complex64 | 934.04 | 0.00 | 0.00 |
| 128 | torch.complex64 | 817.48 | 0.00 | 0.00 |
| 256 | torch.complex64 | 840.49 | 0.00 | 0.00 |
| 512 | torch.complex64 | 0.01 | 0.00 | 0.00 |
| 1024 | torch.complex64 | 1431.71 | 0.00 | 0.00 |
| 4096 | torch.complex64 | 980.26 | 0.00 | 0.00 |
| 16384 | torch.complex64 | 841.95 | 0.00 | 0.00 |

### tileops-base

| n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 64 | torch.complex64 | 225.62 | 0.00 | 0.00 |
| 128 | torch.complex64 | 165.89 | 0.00 | 0.00 |
| 256 | torch.complex64 | 207.14 | 0.00 | 0.00 |
| 512 | torch.complex64 | 220.96 | 0.00 | 0.00 |
| 1024 | torch.complex64 | 354.00 | 0.00 | 0.00 |
| 4096 | torch.complex64 | 213.09 | 0.00 | 0.00 |
| 16384 | torch.complex64 | 171.94 | 0.00 | 0.00 |

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

## fused_add_layer_norm

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 99.54 | 0.00 | 0.00 |
| 4096 | 4096 | torch.bfloat16 | 100.50 | 0.00 | 0.00 |
| 1024 | 3000 | torch.float16 | 100.39 | 0.00 | 0.00 |
| 1025 | 4096 | torch.float16 | 102.76 | 0.00 | 0.00 |

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
| 1024 | 4096 | torch.float16 | 72.37 | 0.00 | 0.00 |
| 4096 | 4096 | torch.bfloat16 | 69.45 | 0.00 | 0.00 |
| 2048 | 5120 | torch.float16 | 69.18 | 0.00 | 0.00 |
| 1025 | 4096 | torch.float16 | 69.45 | 0.00 | 0.00 |

### baseline

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.13 | 0.17 | 0.27 |
| 4096 | 4096 | torch.bfloat16 | 0.48 | 0.17 | 0.28 |
| 2048 | 5120 | torch.float16 | 0.32 | 0.16 | 0.26 |
| 1025 | 4096 | torch.float16 | 0.13 | 0.17 | 0.27 |

## gated_deltanet_decode

### tileops

| batch | heads | dim_k | dim_v | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 128 | 128 | torch.float32 | 0.01 | 0.30 | 0.41 |
| 1 | 32 | 128 | 128 | torch.bfloat16 | 0.01 | 0.31 | 0.21 |
| 8 | 32 | 128 | 128 | torch.float32 | 0.02 | 1.19 | 1.60 |
| 8 | 32 | 128 | 128 | torch.bfloat16 | 0.01 | 2.10 | 1.42 |
| 16 | 32 | 128 | 128 | torch.float32 | 0.04 | 1.27 | 1.71 |
| 16 | 32 | 128 | 128 | torch.bfloat16 | 0.02 | 3.05 | 2.06 |
| 32 | 32 | 128 | 128 | torch.float32 | 0.08 | 1.33 | 1.80 |
| 32 | 32 | 128 | 128 | torch.bfloat16 | 0.03 | 3.32 | 2.24 |
| 1 | 32 | 64 | 64 | torch.float32 | 0.01 | 0.14 | 0.19 |
| 8 | 32 | 64 | 64 | torch.float32 | 0.01 | 0.59 | 0.81 |
| 32 | 32 | 64 | 64 | torch.float32 | 0.04 | 0.71 | 0.97 |
| 64 | 32 | 128 | 128 | torch.float32 | 0.15 | 1.38 | 1.87 |
| 64 | 32 | 128 | 128 | torch.bfloat16 | 0.06 | 3.66 | 2.47 |

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
| 64 | 32 | 128 | 128 | torch.bfloat16 | 0.56 | 0.36 | 0.24 |

## gated_deltanet_fwd

### tileops

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 400.89 | 0.00 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 260.21 | 0.00 | 0.00 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 262.56 | 0.00 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 258.88 | 0.00 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 260.59 | 0.00 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 708.86 | 0.00 | 0.00 |
| 2 | 32768 | 4 | 64 | 64 | 64 | torch.float16 | 261.53 | 0.01 | 0.00 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 261.11 | 0.00 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 260.59 | 0.00 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 402.25 | 0.00 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 466.90 | 0.00 | 0.00 |
| 2 | 32768 | 4 | 64 | 64 | 64 | torch.bfloat16 | 263.89 | 0.01 | 0.00 |

### baseline

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 130.32 | 0.00 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 131.67 | 0.00 | 0.00 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 45.25 | 0.00 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 90.84 | 0.00 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 181.44 | 0.00 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 364.17 | 0.00 | 0.00 |
| 2 | 32768 | 4 | 64 | 64 | 64 | torch.float16 | 729.21 | 0.00 | 0.00 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 45.33 | 0.00 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 90.83 | 0.00 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 182.39 | 0.00 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 364.86 | 0.00 | 0.00 |
| 2 | 32768 | 4 | 64 | 64 | 64 | torch.bfloat16 | 728.64 | 0.00 | 0.00 |

## gated_deltanet_bwd

### tileops

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | H | S | DK | DV | BC | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 32 | 810.26 | 0.00 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 32 | 811.32 | 0.00 | 0.00 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 2048 | 64 | 64 | 64 | 890.87 | 0.00 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.35 | 1.54 | 0.09 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 8192 | 64 | 64 | 64 | 948.66 | 0.00 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 16384 | 64 | 64 | 64 | 812.69 | 0.00 | 0.00 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.19 | 1.41 | 0.08 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.35 | 1.54 | 0.08 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 8192 | 64 | 64 | 64 | 990.61 | 0.00 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 16384 | 64 | 64 | 64 | 819.64 | 0.00 | 0.00 |

### baseline

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | H | S | DK | DV | BC | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 32 | 49.85 | 0.01 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 32 | 49.76 | 0.01 | 0.00 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 2048 | 64 | 64 | 64 | 19.24 | 0.01 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 64 | 39.54 | 0.01 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 8192 | 64 | 64 | 64 | 82.34 | 0.01 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 16384 | 64 | 64 | 64 | 175.28 | 0.01 | 0.00 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 2048 | 64 | 64 | 64 | 19.09 | 0.01 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 64 | 39.40 | 0.01 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 8192 | 64 | 64 | 64 | 82.25 | 0.01 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 16384 | 64 | 64 | 64 | 175.27 | 0.01 | 0.00 |

## gated_deltanet_fwdbwd

### tileops

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | H | S | DK | DV | BC | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 32 | 953.77 | 0.00 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 32 | 946.19 | 0.00 | 0.00 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 2048 | 64 | 64 | 64 | 1021.44 | 0.00 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 64 | 977.08 | 0.00 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 8192 | 64 | 64 | 64 | 947.53 | 0.00 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 16384 | 64 | 64 | 64 | 1156.55 | 0.00 | 0.00 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 2048 | 64 | 64 | 64 | 1063.54 | 0.00 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 64 | 946.55 | 0.00 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 8192 | 64 | 64 | 64 | 951.40 | 0.00 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 16384 | 64 | 64 | 64 | 1060.79 | 0.00 | 0.00 |

### baseline

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | H | S | DK | DV | BC | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 32 | 49.72 | 0.02 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 32 | 49.80 | 0.02 | 0.00 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 2048 | 64 | 64 | 64 | 19.08 | 0.02 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 64 | 39.61 | 0.02 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 8192 | 64 | 64 | 64 | 82.24 | 0.02 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 16384 | 64 | 64 | 64 | 175.34 | 0.02 | 0.00 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 2048 | 64 | 64 | 64 | 19.04 | 0.02 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 64 | 39.34 | 0.02 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 8192 | 64 | 64 | 64 | 82.19 | 0.02 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 16384 | 64 | 64 | 64 | 175.23 | 0.02 | 0.00 |

## gemm

### tileops

| m | n | k | dtype | trans_a | trans_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1024 | 1024 | 1024 | torch.float16 | False | False | 42.67 | 0.05 | 0.00 |
| 1 | 7168 | 16384 | torch.float16 | False | True | 0.07 | 3.61 | 3.61 |
| 1 | 18432 | 7168 | torch.bfloat16 | False | True | 0.07 | 3.87 | 3.87 |
| 7168 | 1 | 16384 | torch.float16 | False | False | 0.07 | 3.61 | 3.61 |
| 18432 | 1 | 7168 | torch.bfloat16 | False | False | 0.07 | 3.87 | 3.87 |

### baseline

| m | n | k | dtype | trans_a | trans_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1024 | 1024 | 1024 | torch.float16 | False | False | 0.01 | 325.07 | 0.95 |
| 1 | 7168 | 16384 | torch.float16 | False | True | 0.07 | 3.43 | 3.43 |
| 1 | 18432 | 7168 | torch.bfloat16 | False | True | 0.08 | 3.50 | 3.50 |
| 7168 | 1 | 16384 | torch.float16 | False | False | 0.07 | 3.43 | 3.43 |
| 18432 | 1 | 7168 | torch.bfloat16 | False | False | 0.08 | 3.50 | 3.50 |

## gla_fwd

### tileops

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.08 | 1.75 | 0.11 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.13 | 2.01 | 0.13 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.26 | 2.08 | 0.13 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.49 | 2.21 | 0.14 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.08 | 1.68 | 0.11 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.13 | 2.01 | 0.13 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.26 | 2.08 | 0.13 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.49 | 2.20 | 0.14 |

### baseline

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 1.99 | 0.07 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 4.04 | 0.07 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 8.09 | 0.07 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 16.11 | 0.07 | 0.00 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 1.99 | 0.07 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 4.04 | 0.07 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 8.09 | 0.07 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 16.11 | 0.07 | 0.00 |

## gla_bwd

### tileops

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | T | H | K | V | BC | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 2048 | 4 | 64 | 64 | 64 | 0.125 | 0.15 | 1.77 | 0.10 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4096 | 4 | 64 | 64 | 64 | 0.125 | 0.29 | 1.84 | 0.10 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 8192 | 4 | 64 | 64 | 64 | 0.125 | 0.54 | 1.98 | 0.11 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 16384 | 4 | 64 | 64 | 64 | 0.125 | 0.98 | 2.20 | 0.12 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 2048 | 4 | 64 | 64 | 64 | 0.125 | 0.15 | 1.76 | 0.10 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4096 | 4 | 64 | 64 | 64 | 0.125 | 0.30 | 1.82 | 0.10 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 8192 | 4 | 64 | 64 | 64 | 0.125 | 0.56 | 1.92 | 0.10 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 16384 | 4 | 64 | 64 | 64 | 0.125 | 1.00 | 2.14 | 0.12 |

### baseline

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | T | H | K | V | BC | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 2048 | 4 | 64 | 64 | 64 | 0.125 | 5.91 | 0.05 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4096 | 4 | 64 | 64 | 64 | 0.125 | 12.38 | 0.04 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 8192 | 4 | 64 | 64 | 64 | 0.125 | 28.47 | 0.04 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 16384 | 4 | 64 | 64 | 64 | 0.125 | 70.09 | 0.03 | 0.00 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 2048 | 4 | 64 | 64 | 64 | 0.125 | 5.91 | 0.05 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4096 | 4 | 64 | 64 | 64 | 0.125 | 12.38 | 0.04 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 8192 | 4 | 64 | 64 | 64 | 0.125 | 28.46 | 0.04 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 16384 | 4 | 64 | 64 | 64 | 0.125 | 70.10 | 0.03 | 0.00 |

## gla_fwdbwd

### tileops

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | T | B | BC | H | K | V | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2048 | 2 | 64 | 4 | 64 | 64 | 0.125 | 0.22 | 1.82 | 0.10 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 4096 | 2 | 64 | 4 | 64 | 64 | 0.125 | 0.42 | 1.92 | 0.11 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 8192 | 2 | 64 | 4 | 64 | 64 | 0.125 | 0.77 | 2.09 | 0.12 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 16384 | 2 | 64 | 4 | 64 | 64 | 0.125 | 1.33 | 2.41 | 0.14 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2048 | 2 | 64 | 4 | 64 | 64 | 0.125 | 0.22 | 1.80 | 0.10 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 4096 | 2 | 64 | 4 | 64 | 64 | 0.125 | 0.42 | 1.92 | 0.11 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 8192 | 2 | 64 | 4 | 64 | 64 | 0.125 | 0.78 | 2.06 | 0.12 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 16384 | 2 | 64 | 4 | 64 | 64 | 0.125 | 1.36 | 2.36 | 0.14 |

### baseline

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | T | B | BC | H | K | V | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2048 | 2 | 64 | 4 | 64 | 64 | 0.125 | 5.76 | 0.07 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 4096 | 2 | 64 | 4 | 64 | 64 | 0.125 | 12.40 | 0.06 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 8192 | 2 | 64 | 4 | 64 | 64 | 0.125 | 28.73 | 0.06 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 16384 | 2 | 64 | 4 | 64 | 64 | 0.125 | 70.04 | 0.05 | 0.00 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2048 | 2 | 64 | 4 | 64 | 64 | 0.125 | 5.91 | 0.07 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 4096 | 2 | 64 | 4 | 64 | 64 | 0.125 | 12.46 | 0.06 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 8192 | 2 | 64 | 4 | 64 | 64 | 0.125 | 28.46 | 0.06 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 16384 | 2 | 64 | 4 | 64 | 64 | 0.125 | 70.03 | 0.05 | 0.00 |

## gla_decode

### tileops

| batch | heads | dim_k | dim_v | dtype | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 64 | 64 | torch.float32 | 0.125 | 0.01 | 0.07 | 0.14 |
| 1 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.02 | 0.13 | 0.26 |
| 1 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.01 | 0.17 | 0.17 |
| 1 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.01 | 0.17 | 0.17 |
| 8 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.03 | 0.50 | 1.02 |
| 8 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.01 | 1.20 | 1.22 |
| 8 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.01 | 1.19 | 1.20 |
| 16 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.06 | 0.52 | 1.06 |
| 16 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.02 | 1.87 | 1.90 |
| 16 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.02 | 1.85 | 1.88 |
| 32 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.12 | 0.55 | 1.12 |
| 32 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.03 | 1.96 | 1.99 |
| 32 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.03 | 1.94 | 1.97 |
| 64 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.23 | 0.58 | 1.17 |
| 64 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.06 | 2.20 | 2.24 |
| 64 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.06 | 2.18 | 2.22 |

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
| 32 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.18 | 0.38 | 0.77 |
| 32 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.27 | 0.25 | 0.26 |
| 32 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.27 | 0.25 | 0.26 |
| 64 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.33 | 0.41 | 0.83 |
| 64 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.49 | 0.27 | 0.28 |
| 64 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.50 | 0.27 | 0.28 |

## gqa_fwd

### tileops

| batch | seq_len | heads | heads_kv | dim | causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 8 | 4 | 64 | False | torch.float16 | 147.53 | 0.01 | 0.00 |
| 4 | 2048 | 64 | 4 | 128 | False | torch.float16 | 146.07 | 3.76 | 0.00 |
| 4 | 2048 | 64 | 4 | 128 | False | torch.bfloat16 | 145.59 | 3.78 | 0.00 |

## gqa_bwd

### tileops

| batch | seq_len | heads | heads_kv | dim | causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 8 | 4 | 64 | False | torch.float16 | 0.05 | 102.47 | 0.10 |
| 4 | 2048 | 64 | 4 | 128 | False | torch.float16 | 3.08 | 446.13 | 0.14 |
| 4 | 2048 | 64 | 4 | 128 | False | torch.bfloat16 | 3.08 | 446.32 | 0.14 |

## gqa_decode

### tileops

| batch | heads | heads_kv | seq_len_kv | dim | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 8 | 8192 | 128 | torch.float16 | 349.05 | 0.00 | 0.00 |
| 4 | 32 | 4 | 4096 | 128 | torch.bfloat16 | 349.58 | 0.00 | 0.00 |
| 8 | 64 | 16 | 8192 | 128 | torch.float16 | 0.17 | 12.82 | 3.21 |

### baseline

| batch | heads | heads_kv | seq_len_kv | dim | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 8 | 8192 | 128 | torch.float16 | 0.43 | 0.31 | 0.08 |
| 4 | 32 | 4 | 4096 | 128 | torch.bfloat16 | 0.81 | 0.33 | 0.04 |
| 8 | 64 | 16 | 8192 | 128 | torch.float16 | 5.88 | 0.37 | 0.09 |

## gqa_decode_paged

### tileops

| batch | heads | heads_kv | seqlen_kv | dim | page_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 16 | 8 | 512 | 128 | 128 | torch.float16 | 379.52 | 0.00 | 0.00 |
| 2 | 8 | 4 | 1024 | 64 | 256 | torch.float16 | 383.18 | 0.00 | 0.00 |
| 1 | 16 | 4 | 2048 | 128 | 512 | torch.float16 | 383.53 | 0.00 | 0.00 |
| 1 | 32 | 16 | 512 | 64 | 128 | torch.float16 | 563.90 | 0.00 | 0.00 |

### baseline

| batch | heads | heads_kv | seqlen_kv | dim | page_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 16 | 8 | 512 | 128 | 128 | torch.float16 | 0.07 | 0.06 | 0.03 |
| 2 | 8 | 4 | 1024 | 64 | 256 | torch.float16 | 0.13 | 0.03 | 0.01 |
| 1 | 16 | 4 | 2048 | 128 | 512 | torch.float16 | 0.07 | 0.25 | 0.06 |
| 1 | 32 | 16 | 512 | 64 | 128 | torch.float16 | 0.06 | 0.06 | 0.03 |

## gqa_sliding_window_fwd

### tileops

| batch | seq | heads | heads_kv | dim | is_causal | wl | wr | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 512 | 8 | 2 | 64 | True | -1 | -1 | torch.float16 | 219.34 | 0.00 | 0.00 |
| 2 | 512 | 8 | 2 | 64 | True | 128 | -1 | torch.float16 | 199.59 | 0.00 | 0.00 |
| 2 | 768 | 8 | 2 | 64 | False | 256 | -1 | torch.float16 | 204.27 | 0.01 | 0.00 |
| 2 | 512 | 8 | 2 | 64 | False | 64 | 64 | torch.bfloat16 | 326.71 | 0.00 | 0.00 |
| 1 | 2048 | 8 | 2 | 64 | True | 512 | -1 | torch.float16 | 199.39 | 0.01 | 0.00 |

## gqa_sliding_window_varlen_fwd

### tileops

| batch | heads | heads_kv | dim | is_causal | wl | wr | dtype | max_seqlen_q | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 8 | 128 | True | -1 | -1 | torch.float16 | 3000 | 0.21 | 345.57 | 0.29 |
| 2 | 32 | 8 | 128 | True | 2000 | -1 | torch.float16 | 3073 | 0.34 | 254.58 | 0.28 |
| 4 | 32 | 8 | 128 | False | 1500 | 1500 | torch.float16 | 3500 | 0.91 | 378.19 | 0.23 |
| 2 | 32 | 8 | 128 | True | -1 | -1 | torch.float16 | 1024 | 0.39 | 411.50 | 0.19 |
| 2 | 32 | 8 | 128 | True | -1 | -1 | torch.float16 | 8193 | 1.81 | 416.07 | 0.15 |

## group_norm

### tileops

| n | c | g | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 128 | 32 | torch.float16 | 90.19 | 0.00 | 0.00 |
| 8 | 128 | 32 | torch.bfloat16 | 90.09 | 0.00 | 0.00 |
| 4 | 256 | 32 | torch.float16 | 89.53 | 0.00 | 0.00 |
| 4 | 128 | 16 | torch.float16 | 89.41 | 0.00 | 0.00 |

### baseline

| n | c | g | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 128 | 32 | torch.float16 | 0.01 | 0.35 | 0.28 |
| 8 | 128 | 32 | torch.bfloat16 | 0.01 | 0.36 | 0.29 |
| 4 | 256 | 32 | torch.float16 | 0.02 | 0.25 | 0.20 |
| 4 | 128 | 16 | torch.float16 | 0.02 | 0.15 | 0.12 |

## grouped_gemm_nt

### tileops

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_nt | 16384 | 4 | 4864 | 4096 | torch.float16 | False | True | 127.47 | 5.12 | 0.00 |

### baseline

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_nt | 16384 | 4 | 4864 | 4096 | torch.float16 | False | True | 1.51 | 432.74 | 0.30 |

## grouped_gemm_nn

### tileops

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_nn | 16384 | 4 | 4864 | 4096 | torch.float16 | False | False | 125.80 | 5.19 | 0.00 |

### baseline

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_nn | 16384 | 4 | 4864 | 4096 | torch.float16 | False | False | 1.02 | 638.85 | 0.44 |

## grouped_gemm_tn

### tileops

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_tn | 16384 | 4 | 4864 | 4096 | torch.float16 | True | False | 123.28 | 5.30 | 0.00 |

### baseline

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_tn | 16384 | 4 | 4864 | 4096 | torch.float16 | True | False | 1.00 | 652.34 | 0.45 |

## grouped_gemm_tt

### tileops

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_tt | 16384 | 4 | 4864 | 4096 | torch.float16 | True | True | 126.66 | 5.15 | 0.00 |

### baseline

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_tt | 16384 | 4 | 4864 | 4096 | torch.float16 | True | True | 0.98 | 664.25 | 0.46 |

## grouped_gemm_complete

### tileops

| batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16384 | 4 | 4864 | 4096 | torch.float16 | True | False | 487.30 | 4.02 | N/A |

### baseline

| batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16384 | 4 | 4864 | 4096 | torch.float16 | True | False | 4.48 | 437.12 | N/A |

## leaky_relu

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| leaky_relu | torch.float16 | 4194304 | 0.01 | 0.69 | 2.77 |
| leaky_relu | torch.bfloat16 | 4194304 | 0.01 | 0.69 | 2.77 |
| leaky_relu | torch.float32 | 4194304 | 0.01 | 0.40 | 3.19 |
| leaky_relu | torch.float16 | 10485760 | 0.01 | 0.82 | 3.29 |
| leaky_relu | torch.bfloat16 | 10485760 | 0.01 | 0.82 | 3.29 |
| leaky_relu | torch.float32 | 10485760 | 0.02 | 0.48 | 3.83 |
| leaky_relu | torch.float16 | 20971520 | 0.02 | 0.92 | 3.69 |
| leaky_relu | torch.bfloat16 | 20971520 | 0.02 | 0.92 | 3.69 |
| leaky_relu | torch.float32 | 20971520 | 0.04 | 0.52 | 4.12 |

### baseline

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| leaky_relu | torch.float16 | 4194304 | 0.01 | 0.66 | 2.63 |
| leaky_relu | torch.bfloat16 | 4194304 | 0.01 | 0.65 | 2.62 |
| leaky_relu | torch.float32 | 4194304 | 0.01 | 0.40 | 3.19 |
| leaky_relu | torch.float16 | 10485760 | 0.01 | 0.85 | 3.39 |
| leaky_relu | torch.bfloat16 | 10485760 | 0.01 | 0.85 | 3.38 |
| leaky_relu | torch.float32 | 10485760 | 0.02 | 0.47 | 3.77 |
| leaky_relu | torch.float16 | 20971520 | 0.02 | 0.95 | 3.81 |
| leaky_relu | torch.bfloat16 | 20971520 | 0.02 | 0.95 | 3.81 |
| leaky_relu | torch.float32 | 20971520 | 0.04 | 0.51 | 4.08 |

## elu

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| elu | torch.float16 | 4194304 | 0.01 | 0.41 | 1.62 |
| elu | torch.bfloat16 | 4194304 | 0.01 | 0.40 | 1.61 |
| elu | torch.float32 | 4194304 | 0.01 | 0.31 | 2.49 |
| elu | torch.float16 | 10485760 | 0.02 | 0.49 | 1.97 |
| elu | torch.bfloat16 | 10485760 | 0.02 | 0.49 | 1.95 |
| elu | torch.float32 | 10485760 | 0.03 | 0.36 | 2.90 |
| elu | torch.float16 | 20971520 | 0.04 | 0.54 | 2.15 |
| elu | torch.bfloat16 | 20971520 | 0.04 | 0.53 | 2.14 |
| elu | torch.float32 | 20971520 | 0.05 | 0.38 | 3.06 |

### baseline

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| elu | torch.float16 | 4194304 | 0.01 | 0.43 | 1.71 |
| elu | torch.bfloat16 | 4194304 | 0.01 | 0.42 | 1.69 |
| elu | torch.float32 | 4194304 | 0.01 | 0.37 | 3.00 |
| elu | torch.float16 | 10485760 | 0.02 | 0.52 | 2.08 |
| elu | torch.bfloat16 | 10485760 | 0.02 | 0.51 | 2.06 |
| elu | torch.float32 | 10485760 | 0.02 | 0.46 | 3.68 |
| elu | torch.float16 | 20971520 | 0.04 | 0.57 | 2.28 |
| elu | torch.bfloat16 | 20971520 | 0.04 | 0.57 | 2.26 |
| elu | torch.float32 | 20971520 | 0.04 | 0.50 | 4.03 |

## hardtanh

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| hardtanh | torch.float16 | 4194304 | 0.01 | 0.66 | 2.65 |
| hardtanh | torch.bfloat16 | 4194304 | 0.01 | 0.66 | 2.65 |
| hardtanh | torch.float32 | 4194304 | 0.01 | 0.40 | 3.19 |
| hardtanh | torch.float16 | 10485760 | 0.01 | 0.85 | 3.39 |
| hardtanh | torch.bfloat16 | 10485760 | 0.01 | 0.85 | 3.39 |
| hardtanh | torch.float32 | 10485760 | 0.02 | 0.48 | 3.83 |
| hardtanh | torch.float16 | 20971520 | 0.02 | 0.96 | 3.83 |
| hardtanh | torch.bfloat16 | 20971520 | 0.02 | 0.96 | 3.85 |
| hardtanh | torch.float32 | 20971520 | 0.04 | 0.52 | 4.14 |

### baseline

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| hardtanh | torch.float16 | 4194304 | 0.01 | 0.63 | 2.51 |
| hardtanh | torch.bfloat16 | 4194304 | 0.01 | 0.65 | 2.59 |
| hardtanh | torch.float32 | 4194304 | 0.01 | 0.40 | 3.17 |
| hardtanh | torch.float16 | 10485760 | 0.01 | 0.79 | 3.16 |
| hardtanh | torch.bfloat16 | 10485760 | 0.01 | 0.84 | 3.34 |
| hardtanh | torch.float32 | 10485760 | 0.02 | 0.47 | 3.77 |
| hardtanh | torch.float16 | 20971520 | 0.02 | 0.88 | 3.52 |
| hardtanh | torch.bfloat16 | 20971520 | 0.02 | 0.95 | 3.79 |
| hardtanh | torch.float32 | 20971520 | 0.04 | 0.51 | 4.08 |

## softplus

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| softplus | torch.float16 | 4194304 | 0.01 | 0.32 | 1.30 |
| softplus | torch.bfloat16 | 4194304 | 0.01 | 0.32 | 1.30 |
| softplus | torch.float32 | 4194304 | 0.01 | 0.37 | 2.98 |
| softplus | torch.float16 | 10485760 | 0.03 | 0.38 | 1.50 |
| softplus | torch.bfloat16 | 10485760 | 0.03 | 0.38 | 1.50 |
| softplus | torch.float32 | 10485760 | 0.02 | 0.44 | 3.53 |
| softplus | torch.float16 | 20971520 | 0.05 | 0.40 | 1.61 |
| softplus | torch.bfloat16 | 20971520 | 0.05 | 0.40 | 1.61 |
| softplus | torch.float32 | 20971520 | 0.04 | 0.47 | 3.73 |

### baseline

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| softplus | torch.float16 | 4194304 | 0.01 | 0.36 | 1.44 |
| softplus | torch.bfloat16 | 4194304 | 0.01 | 0.36 | 1.43 |
| softplus | torch.float32 | 4194304 | 0.01 | 0.34 | 2.71 |
| softplus | torch.float16 | 10485760 | 0.02 | 0.45 | 1.80 |
| softplus | torch.bfloat16 | 10485760 | 0.02 | 0.43 | 1.71 |
| softplus | torch.float32 | 10485760 | 0.03 | 0.42 | 3.34 |
| softplus | torch.float16 | 20971520 | 0.04 | 0.47 | 1.87 |
| softplus | torch.bfloat16 | 20971520 | 0.05 | 0.46 | 1.85 |
| softplus | torch.float32 | 20971520 | 0.05 | 0.45 | 3.63 |

## clamp

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| clamp | torch.float16 | 4194304 | 0.01 | 0.66 | 2.65 |
| clamp | torch.bfloat16 | 4194304 | 0.01 | 0.66 | 2.65 |
| clamp | torch.float32 | 4194304 | 0.01 | 0.40 | 3.19 |
| clamp | torch.float16 | 10485760 | 0.01 | 0.85 | 3.39 |
| clamp | torch.bfloat16 | 10485760 | 0.01 | 0.85 | 3.39 |
| clamp | torch.float32 | 10485760 | 0.02 | 0.48 | 3.83 |
| clamp | torch.float16 | 20971520 | 0.02 | 0.94 | 3.77 |
| clamp | torch.bfloat16 | 20971520 | 0.02 | 0.96 | 3.82 |
| clamp | torch.float32 | 20971520 | 0.04 | 0.52 | 4.13 |

### baseline

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| clamp | torch.float16 | 4194304 | 0.01 | 0.63 | 2.51 |
| clamp | torch.bfloat16 | 4194304 | 0.01 | 0.65 | 2.58 |
| clamp | torch.float32 | 4194304 | 0.01 | 0.40 | 3.16 |
| clamp | torch.float16 | 10485760 | 0.01 | 0.79 | 3.16 |
| clamp | torch.bfloat16 | 10485760 | 0.01 | 0.84 | 3.35 |
| clamp | torch.float32 | 10485760 | 0.02 | 0.47 | 3.77 |
| clamp | torch.float16 | 20971520 | 0.02 | 0.88 | 3.52 |
| clamp | torch.bfloat16 | 20971520 | 0.02 | 0.94 | 3.78 |
| clamp | torch.float32 | 20971520 | 0.04 | 0.51 | 4.05 |

## nan_to_num

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| nan_to_num | torch.float16 | 4194304 | 0.01 | 0.66 | 2.64 |
| nan_to_num | torch.bfloat16 | 4194304 | 0.01 | 0.66 | 2.63 |
| nan_to_num | torch.float32 | 4194304 | 0.01 | 0.40 | 3.17 |
| nan_to_num | torch.float16 | 10485760 | 0.01 | 0.80 | 3.21 |
| nan_to_num | torch.bfloat16 | 10485760 | 0.01 | 0.80 | 3.21 |
| nan_to_num | torch.float32 | 10485760 | 0.02 | 0.48 | 3.83 |
| nan_to_num | torch.float16 | 20971520 | 0.02 | 0.90 | 3.62 |
| nan_to_num | torch.bfloat16 | 20971520 | 0.02 | 0.90 | 3.62 |
| nan_to_num | torch.float32 | 20971520 | 0.04 | 0.52 | 4.13 |

### baseline

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| nan_to_num | torch.float16 | 4194304 | 0.01 | 0.64 | 2.55 |
| nan_to_num | torch.bfloat16 | 4194304 | 0.01 | 0.64 | 2.55 |
| nan_to_num | torch.float32 | 4194304 | 0.01 | 0.40 | 3.17 |
| nan_to_num | torch.float16 | 10485760 | 0.01 | 0.83 | 3.31 |
| nan_to_num | torch.bfloat16 | 10485760 | 0.01 | 0.83 | 3.31 |
| nan_to_num | torch.float32 | 10485760 | 0.02 | 0.47 | 3.78 |
| nan_to_num | torch.float16 | 20971520 | 0.02 | 0.93 | 3.73 |
| nan_to_num | torch.bfloat16 | 20971520 | 0.02 | 0.94 | 3.75 |
| nan_to_num | torch.float32 | 20971520 | 0.04 | 0.51 | 4.08 |

## prelu

### tileops

| num_channels | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 128 | torch.float16 | 131072 | 0.00 | 0.06 | 0.25 |
| 128 | torch.bfloat16 | 131072 | 0.00 | 0.06 | 0.25 |
| 128 | torch.float32 | 131072 | 0.00 | 0.06 | 0.46 |
| 4096 | torch.float16 | 4194304 | 0.01 | 0.69 | 2.74 |
| 4096 | torch.bfloat16 | 4194304 | 0.01 | 0.68 | 2.74 |
| 4096 | torch.float32 | 4194304 | 0.01 | 0.40 | 3.20 |
| 10240 | torch.float16 | 10485760 | 0.01 | 0.82 | 3.29 |
| 10240 | torch.bfloat16 | 10485760 | 0.01 | 0.82 | 3.27 |
| 10240 | torch.float32 | 10485760 | 0.02 | 0.47 | 3.78 |
| 20480 | torch.float16 | 20971520 | 0.02 | 0.92 | 3.67 |
| 20480 | torch.bfloat16 | 20971520 | 0.02 | 0.91 | 3.66 |
| 20480 | torch.float32 | 20971520 | 0.04 | 0.50 | 4.03 |

### baseline

| num_channels | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 128 | torch.float16 | 131072 | 0.00 | 0.03 | 0.12 |
| 128 | torch.bfloat16 | 131072 | 0.00 | 0.03 | 0.12 |
| 128 | torch.float32 | 131072 | 0.00 | 0.04 | 0.29 |
| 4096 | torch.float16 | 4194304 | 0.01 | 0.29 | 1.15 |
| 4096 | torch.bfloat16 | 4194304 | 0.01 | 0.28 | 1.14 |
| 4096 | torch.float32 | 4194304 | 0.02 | 0.25 | 2.04 |
| 10240 | torch.float16 | 10485760 | 0.03 | 0.34 | 1.36 |
| 10240 | torch.bfloat16 | 10485760 | 0.03 | 0.33 | 1.34 |
| 10240 | torch.float32 | 10485760 | 0.04 | 0.29 | 2.33 |
| 20480 | torch.float16 | 20971520 | 0.06 | 0.37 | 1.47 |
| 20480 | torch.bfloat16 | 20971520 | 0.06 | 0.36 | 1.45 |
| 20480 | torch.float32 | 20971520 | 0.07 | 0.31 | 2.47 |

## where

### tileops

| dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| torch.float16 | 4194304 | 0.01 | 0.41 | 2.85 |
| torch.bfloat16 | 4194304 | 0.01 | 0.41 | 2.87 |
| torch.float32 | 4194304 | 0.02 | 0.26 | 3.33 |
| torch.float16 | 10485760 | 0.02 | 0.50 | 3.51 |
| torch.bfloat16 | 10485760 | 0.02 | 0.50 | 3.51 |
| torch.float32 | 10485760 | 0.03 | 0.30 | 3.90 |
| torch.float16 | 20971520 | 0.04 | 0.56 | 3.94 |
| torch.bfloat16 | 20971520 | 0.04 | 0.56 | 3.91 |
| torch.float32 | 20971520 | 0.06 | 0.32 | 4.22 |

### baseline

| dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| torch.float16 | 4194304 | 0.01 | 0.41 | 2.89 |
| torch.bfloat16 | 4194304 | 0.01 | 0.41 | 2.89 |
| torch.float32 | 4194304 | 0.02 | 0.26 | 3.37 |
| torch.float16 | 10485760 | 0.02 | 0.51 | 3.54 |
| torch.bfloat16 | 10485760 | 0.02 | 0.51 | 3.54 |
| torch.float32 | 10485760 | 0.03 | 0.30 | 3.93 |
| torch.float16 | 20971520 | 0.04 | 0.56 | 3.95 |
| torch.bfloat16 | 20971520 | 0.04 | 0.56 | 3.95 |
| torch.float32 | 20971520 | 0.06 | 0.33 | 4.23 |

## masked_fill

### tileops

| dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| torch.float16 | 4194304 | 0.01 | 0.55 | 2.75 |
| torch.bfloat16 | 4194304 | 0.01 | 0.55 | 2.74 |
| torch.float32 | 4194304 | 0.01 | 0.35 | 3.15 |
| torch.float16 | 10485760 | 0.02 | 0.68 | 3.38 |
| torch.bfloat16 | 10485760 | 0.02 | 0.68 | 3.38 |
| torch.float32 | 10485760 | 0.02 | 0.42 | 3.82 |
| torch.float16 | 20971520 | 0.03 | 0.77 | 3.87 |
| torch.bfloat16 | 20971520 | 0.03 | 0.78 | 3.88 |
| torch.float32 | 20971520 | 0.05 | 0.46 | 4.13 |

### baseline

| dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| torch.float16 | 4194304 | 0.01 | 0.36 | 1.79 |
| torch.bfloat16 | 4194304 | 0.01 | 0.36 | 1.78 |
| torch.float32 | 4194304 | 0.02 | 0.23 | 2.03 |
| torch.float16 | 10485760 | 0.02 | 0.45 | 2.25 |
| torch.bfloat16 | 10485760 | 0.02 | 0.45 | 2.25 |
| torch.float32 | 10485760 | 0.05 | 0.23 | 2.09 |
| torch.float16 | 20971520 | 0.05 | 0.44 | 2.22 |
| torch.bfloat16 | 20971520 | 0.05 | 0.43 | 2.16 |
| torch.float32 | 20971520 | 0.09 | 0.24 | 2.20 |

## alibi

### tileops

| op_name | seq_len | dim | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| alibi | 512 | 64 | torch.float16 | 0.03 | 0.53 | 1.06 |
| alibi | 512 | 64 | torch.bfloat16 | 0.03 | 0.53 | 1.06 |
| alibi | 512 | 64 | torch.float32 | 0.02 | 0.90 | 3.60 |
| alibi | 2048 | 64 | torch.float16 | 0.26 | 1.04 | 2.08 |
| alibi | 2048 | 64 | torch.bfloat16 | 0.29 | 0.93 | 1.87 |
| alibi | 2048 | 64 | torch.float32 | 0.24 | 1.10 | 4.39 |
| alibi | 4096 | 128 | torch.float16 | 0.74 | 2.91 | 5.83 |
| alibi | 4096 | 128 | torch.bfloat16 | 0.74 | 2.91 | 5.82 |
| alibi | 4096 | 128 | torch.float32 | 2.06 | 1.04 | 4.17 |

### baseline

| op_name | seq_len | dim | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| alibi | 512 | 64 | torch.float16 | 0.08 | 0.21 | 0.41 |
| alibi | 512 | 64 | torch.bfloat16 | 0.08 | 0.20 | 0.41 |
| alibi | 512 | 64 | torch.float32 | 0.05 | 0.31 | 1.22 |
| alibi | 2048 | 64 | torch.float16 | 0.99 | 0.27 | 0.54 |
| alibi | 2048 | 64 | torch.bfloat16 | 0.99 | 0.27 | 0.54 |
| alibi | 2048 | 64 | torch.float32 | 0.65 | 0.41 | 1.64 |
| alibi | 4096 | 128 | torch.float16 | 9.51 | 0.23 | 0.45 |
| alibi | 4096 | 128 | torch.bfloat16 | 6.81 | 0.32 | 0.63 |
| alibi | 4096 | 128 | torch.float32 | 4.90 | 0.44 | 1.75 |

## sinusoidal

### tileops

| op_name | seq_len | dim | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| sinusoidal | 512 | 256 | torch.float16 | 0.00 | 0.04 | 0.09 |
| sinusoidal | 512 | 256 | torch.bfloat16 | 0.00 | 0.05 | 0.09 |
| sinusoidal | 512 | 256 | torch.float32 | 0.00 | 0.05 | 0.21 |
| sinusoidal | 2048 | 300 | torch.float16 | 0.00 | 0.15 | 0.30 |
| sinusoidal | 2048 | 300 | torch.bfloat16 | 0.00 | 0.15 | 0.30 |
| sinusoidal | 2048 | 300 | torch.float32 | 0.00 | 0.13 | 0.51 |
| sinusoidal | 4096 | 512 | torch.float16 | 0.01 | 0.30 | 0.61 |
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
| sinusoidal | 4096 | 512 | torch.float32 | 0.03 | 0.07 | 0.29 |

## instance_norm

### tileops

| n | c | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 8 | 128 | torch.float16 | 89.21 | 0.00 | 0.00 |
| 8 | 128 | torch.bfloat16 | 90.26 | 0.00 | 0.00 |
| 4 | 256 | torch.float16 | 89.37 | 0.00 | 0.00 |
| 4 | 64 | torch.float16 | 89.19 | 0.00 | 0.00 |

### baseline

| n | c | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 8 | 128 | torch.float16 | 0.02 | 0.25 | 0.20 |
| 8 | 128 | torch.bfloat16 | 0.02 | 0.25 | 0.20 |
| 4 | 256 | torch.float16 | 0.02 | 0.20 | 0.16 |
| 4 | 64 | torch.float16 | 0.01 | 0.09 | 0.07 |

## layer_norm

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 88.42 | 0.00 | 0.00 |
| 4096 | 4096 | torch.bfloat16 | 89.51 | 0.00 | 0.00 |
| 2048 | 5120 | torch.float16 | 88.37 | 0.00 | 0.00 |
| 1025 | 4096 | torch.float16 | 89.13 | 0.00 | 0.00 |

### baseline

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.01 | 1.66 | 1.33 |
| 4096 | 4096 | torch.bfloat16 | 0.03 | 2.59 | 2.07 |
| 2048 | 5120 | torch.float16 | 0.02 | 2.36 | 1.89 |
| 1025 | 4096 | torch.float16 | 0.01 | 1.66 | 1.33 |

## logical_reduce

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | any | 56.73 | 0.00 | 0.00 |
| 1024 | 4096 | torch.bfloat16 | any | 56.62 | 0.00 | 0.00 |
| 1024 | 4096 | torch.float32 | any | 56.77 | 0.00 | 0.00 |
| 1024 | 4096 | torch.int32 | any | 57.56 | 0.00 | 0.00 |
| 4096 | 4096 | torch.float16 | any | 56.80 | 0.00 | 0.00 |
| 1024 | 4096 | torch.float16 | all | 56.57 | 0.00 | 0.00 |
| 1024 | 4096 | torch.bfloat16 | all | 56.25 | 0.00 | 0.00 |
| 1024 | 4096 | torch.float32 | all | 56.88 | 0.00 | 0.00 |
| 1024 | 4096 | torch.int32 | all | 56.77 | 0.00 | 0.00 |
| 4096 | 4096 | torch.float16 | all | 57.03 | 0.00 | 0.00 |
| 1024 | 4096 | torch.float16 | count_nonzero | 53.84 | 0.00 | 0.00 |
| 1024 | 4096 | torch.bfloat16 | count_nonzero | 53.63 | 0.00 | 0.00 |
| 1024 | 4096 | torch.float32 | count_nonzero | 54.12 | 0.00 | 0.00 |
| 1024 | 4096 | torch.int32 | count_nonzero | 55.30 | 0.00 | 0.00 |
| 4096 | 4096 | torch.float16 | count_nonzero | 53.77 | 0.00 | 0.00 |

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
| 1024 | 4096 | torch.float32 | all | 0.03 | 0.16 | 0.64 |
| 1024 | 4096 | torch.int32 | all | 0.03 | 0.16 | 0.64 |
| 4096 | 4096 | torch.float16 | all | 0.07 | 0.26 | 0.51 |
| 1024 | 4096 | torch.float16 | count_nonzero | 0.03 | 0.12 | 0.25 |
| 1024 | 4096 | torch.bfloat16 | count_nonzero | 0.03 | 0.12 | 0.25 |
| 1024 | 4096 | torch.float32 | count_nonzero | 0.04 | 0.11 | 0.45 |
| 1024 | 4096 | torch.int32 | count_nonzero | 0.04 | 0.11 | 0.45 |
| 4096 | 4096 | torch.float16 | count_nonzero | 0.11 | 0.16 | 0.31 |

## mean_pooling

### tileops

| batch_size | seq_len | heads | dim | chunk_size | dtype | accum_dtype | chunks_per_bacth | seq_num | use_offsets | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8192 | 64 | 128 | 64 | torch.float16 | torch.float32 | 128 | 1 | 0 | 62.04 | 0.00 | 0.00 |
| 2 | 2048 | 64 | 128 | 64 | torch.float16 | torch.float32 | 32 | 2 | 0 | 61.17 | 0.00 | 0.00 |
| 1 | 8192 | 64 | 128 | 64 | torch.float16 | torch.float32 | 128 | 4 | 1 | 62.26 | 0.00 | 0.00 |
| 1 | 1000 | 64 | 128 | 32 | torch.float16 | torch.float32 | 34 | 4 | 1 | 62.60 | 0.00 | 0.00 |

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
| 1 | 1024 | 8 | 64 | False | torch.float16 | 146.89 | 0.01 | 0.00 |
| 16 | 2048 | 16 | 128 | False | torch.float16 | 149.03 | 3.69 | 0.00 |
| 4 | 4096 | 16 | 128 | False | torch.bfloat16 | 146.85 | 3.74 | 0.00 |

## mha_bwd

### tileops

| batch | seq_len | heads | dim | causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 8 | 64 | False | torch.float16 | 0.04 | 124.29 | 0.17 |
| 16 | 2048 | 16 | 128 | False | torch.float16 | 2.31 | 594.88 | 0.41 |
| 4 | 4096 | 16 | 128 | False | torch.bfloat16 | 2.13 | 646.25 | 0.22 |

## mha_decode

### tileops

| b | h | s_q | s_kv | d | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 128 | 8192 | 128 | torch.float16 | 423.56 | 0.04 | 0.00 |
| 1 | 32 | 128 | 8192 | 128 | torch.bfloat16 | 570.23 | 0.03 | 0.00 |
| 1 | 32 | 128 | 5 | 128 | torch.float16 | 581.26 | 0.00 | 0.00 |

### baseline

| b | h | s_q | s_kv | d | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 128 | 8192 | 128 | torch.float16 | 0.07 | 253.80 | 2.01 |
| 1 | 32 | 128 | 8192 | 128 | torch.bfloat16 | 0.07 | 256.55 | 2.04 |
| 1 | 32 | 128 | 5 | 128 | torch.float16 | 0.01 | 1.45 | 0.30 |

## mha_decode_paged

### tileops

| batch | heads | seqlen_q | seqlen_kv | dim | page_size | is_causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 16 | 1 | 512 | 128 | 128 | False | torch.float16 | 467.11 | 0.00 | 0.00 |
| 2 | 8 | 1 | 1024 | 64 | 256 | False | torch.float16 | 471.02 | 0.00 | 0.00 |
| 1 | 8 | 1 | 1024 | 64 | 256 | False | torch.float16 | 470.88 | 0.00 | 0.00 |
| 1 | 8 | 1 | 512 | 64 | 256 | False | torch.float16 | 615.77 | 0.00 | 0.00 |

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
| 1 | 4 | 1280 | torch.bfloat16 | 41.81 | 0.00 | 0.00 |
| 2 | 4 | 1920 | torch.bfloat16 | 41.99 | 0.01 | 0.00 |
| 4 | 4 | 2560 | torch.bfloat16 | 42.01 | 0.02 | 0.00 |

### baseline

| batch | n_expand | c_x | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | 1280 | torch.bfloat16 | 0.01 | 3.97 | 0.00 |
| 2 | 4 | 1920 | torch.bfloat16 | 0.01 | 16.64 | 0.00 |
| 4 | 4 | 2560 | torch.bfloat16 | 0.01 | 56.08 | 0.00 |

## mhc_pre

### tileops

| batch | n_expand | c_x | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | 1280 | torch.bfloat16 | 324.42 | 0.00 | 0.00 |
| 2 | 4 | 1920 | torch.bfloat16 | 326.59 | 0.02 | 0.00 |
| 4 | 4 | 2560 | torch.bfloat16 | 326.91 | 0.06 | 0.00 |

### baseline

| batch | n_expand | c_x | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | 1280 | torch.bfloat16 | 0.27 | 4.60 | 0.00 |
| 2 | 4 | 1920 | torch.bfloat16 | 0.30 | 18.84 | 0.00 |
| 4 | 4 | 2560 | torch.bfloat16 | 0.34 | 59.99 | 0.00 |

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
| 1024 | 4096 | torch.float16 | sum | 72.54 | 0.00 | 0.00 |
| 1024 | 4096 | torch.bfloat16 | sum | 112.44 | 0.00 | 0.00 |
| 4096 | 4096 | torch.float16 | sum | 144.10 | 0.00 | 0.00 |
| 1024 | 4096 | torch.float16 | mean | 72.89 | 0.00 | 0.00 |
| 1024 | 4096 | torch.float16 | amax | 65.92 | 0.00 | 0.00 |
| 1024 | 4096 | torch.float16 | amin | 64.02 | 0.00 | 0.00 |
| 1024 | 4096 | torch.float16 | prod | 80.24 | 0.00 | 0.00 |
| 1024 | 4096 | torch.float16 | std | 112.67 | 0.00 | 0.00 |
| 1024 | 4096 | torch.float16 | var | 114.14 | 0.00 | 0.00 |
| 1024 | 4096 | torch.float16 | var_mean | 111.45 | 0.00 | 0.00 |

### baseline

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | sum | 0.03 | 0.16 | 0.32 |
| 1024 | 4096 | torch.bfloat16 | sum | 0.03 | 0.16 | 0.32 |
| 4096 | 4096 | torch.float16 | sum | 0.08 | 0.22 | 0.44 |
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
| 1024 | 4096 | torch.float16 | 55.77 | 0.00 | 0.00 |
| 4096 | 4096 | torch.bfloat16 | 56.46 | 0.00 | 0.00 |
| 2048 | 5120 | torch.float16 | 56.20 | 0.00 | 0.00 |
| 1025 | 4096 | torch.float16 | 56.00 | 0.00 | 0.00 |

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
| 2048 | 128 | torch.float16 | 0.00 | 0.42 | 0.63 |
| 2048 | 128 | torch.bfloat16 | 0.00 | 0.42 | 0.63 |
| 2048 | 128 | torch.float32 | 0.00 | 0.35 | 1.04 |
| 4096 | 128 | torch.float16 | 0.00 | 0.69 | 1.04 |
| 4096 | 128 | torch.bfloat16 | 0.00 | 0.68 | 1.02 |
| 4096 | 128 | torch.float32 | 0.00 | 0.54 | 1.61 |

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
| 1024 | 4096 | torch.float16 | 80.66 | 0.00 | 0.00 |
| 4096 | 4096 | torch.bfloat16 | 80.05 | 0.00 | 0.00 |
| 1024 | 3000 | torch.float16 | 79.63 | 0.00 | 0.00 |
| 1025 | 4096 | torch.float16 | 79.69 | 0.00 | 0.00 |

### baseline

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.02 | 0.98 | 0.98 |
| 4096 | 4096 | torch.bfloat16 | 0.06 | 1.15 | 1.15 |
| 1024 | 3000 | torch.float16 | 0.01 | 0.84 | 0.84 |
| 1025 | 4096 | torch.float16 | 0.02 | 0.97 | 0.97 |

## log_softmax

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 80.12 | 0.00 | 0.00 |
| 4096 | 4096 | torch.bfloat16 | 79.79 | 0.00 | 0.00 |
| 1024 | 3000 | torch.float16 | 80.20 | 0.00 | 0.00 |
| 1025 | 4096 | torch.float16 | 79.82 | 0.00 | 0.00 |

### baseline

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.02 | 1.38 | 1.11 |
| 4096 | 4096 | torch.bfloat16 | 0.05 | 1.61 | 1.29 |
| 1024 | 3000 | torch.float16 | 0.01 | 1.16 | 0.93 |
| 1025 | 4096 | torch.float16 | 0.02 | 1.38 | 1.10 |

## logsumexp

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 73.60 | 0.00 | 0.00 |
| 4096 | 4096 | torch.bfloat16 | 73.28 | 0.00 | 0.00 |
| 1024 | 3000 | torch.float16 | 73.20 | 0.00 | 0.00 |
| 1025 | 4096 | torch.float16 | 75.02 | 0.00 | 0.00 |

### baseline

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.05 | 0.25 | 0.17 |
| 4096 | 4096 | torch.bfloat16 | 0.10 | 0.50 | 0.33 |
| 1024 | 3000 | torch.float16 | 0.04 | 0.21 | 0.14 |
| 1025 | 4096 | torch.float16 | 0.05 | 0.24 | 0.16 |

## exp

### tileops

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| exp | 262144 | torch.float16 | torch.float16 | 0.00 | 0.11 | 0.46 |
| exp | 1048576 | torch.float16 | torch.float16 | 0.00 | 0.33 | 1.32 |
| exp | 4000000 | torch.float16 | torch.float16 | 0.01 | 0.63 | 2.53 |
| exp | 262144 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.11 | 0.46 |
| exp | 1048576 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.33 | 1.32 |
| exp | 4000000 | torch.bfloat16 | torch.bfloat16 | 0.01 | 0.63 | 2.53 |

### baseline

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| exp | 262144 | torch.float16 | torch.float16 | 0.00 | 0.11 | 0.44 |
| exp | 1048576 | torch.float16 | torch.float16 | 0.00 | 0.32 | 1.29 |
| exp | 4000000 | torch.float16 | torch.float16 | 0.01 | 0.63 | 2.54 |
| exp | 262144 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.11 | 0.44 |
| exp | 1048576 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.32 | 1.28 |
| exp | 4000000 | torch.bfloat16 | torch.bfloat16 | 0.01 | 0.63 | 2.53 |

## gelu

### tileops

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| gelu | 262144 | torch.float16 | torch.float16 | 0.00 | 0.11 | 0.44 |
| gelu | 1048576 | torch.float16 | torch.float16 | 0.00 | 0.30 | 1.18 |
| gelu | 4000000 | torch.float16 | torch.float16 | 0.01 | 0.48 | 1.91 |
| gelu | 262144 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.11 | 0.44 |
| gelu | 1048576 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.29 | 1.16 |
| gelu | 4000000 | torch.bfloat16 | torch.bfloat16 | 0.01 | 0.46 | 1.83 |

### baseline

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| gelu | 262144 | torch.float16 | torch.float16 | 0.00 | 0.10 | 0.41 |
| gelu | 1048576 | torch.float16 | torch.float16 | 0.00 | 0.29 | 1.15 |
| gelu | 4000000 | torch.float16 | torch.float16 | 0.01 | 0.52 | 2.08 |
| gelu | 262144 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.10 | 0.41 |
| gelu | 1048576 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.28 | 1.12 |
| gelu | 4000000 | torch.bfloat16 | torch.bfloat16 | 0.01 | 0.51 | 2.02 |

## logical_not

### tileops

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| logical_not | 262144 | torch.float16 | torch.bool | 0.00 | 0.11 | 0.33 |
| logical_not | 1048576 | torch.float16 | torch.bool | 0.00 | 0.22 | 0.65 |
| logical_not | 4000000 | torch.float16 | torch.bool | 0.01 | 0.30 | 0.91 |

### baseline

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| logical_not | 262144 | torch.float16 | torch.bool | 0.00 | 0.11 | 0.34 |
| logical_not | 1048576 | torch.float16 | torch.bool | 0.00 | 0.33 | 1.00 |
| logical_not | 4000000 | torch.float16 | torch.bool | 0.01 | 0.72 | 2.17 |

## bitwise_not

### tileops

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| bitwise_not | 262144 | torch.int32 | torch.int32 | 0.00 | 0.10 | 0.76 |
| bitwise_not | 1048576 | torch.int32 | torch.int32 | 0.01 | 0.19 | 1.56 |
| bitwise_not | 4000000 | torch.int32 | torch.int32 | 0.01 | 0.27 | 2.14 |

### baseline

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| bitwise_not | 262144 | torch.int32 | torch.int32 | 0.00 | 0.10 | 0.82 |
| bitwise_not | 1048576 | torch.int32 | torch.int32 | 0.00 | 0.24 | 1.95 |
| bitwise_not | 4000000 | torch.int32 | torch.int32 | 0.01 | 0.40 | 3.17 |

## isnan

### tileops

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| isnan | 262144 | torch.float16 | torch.bool | 0.00 | 0.11 | 0.33 |
| isnan | 1048576 | torch.float16 | torch.bool | 0.00 | 0.21 | 0.64 |
| isnan | 4000000 | torch.float16 | torch.bool | 0.01 | 0.30 | 0.91 |

### baseline

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| isnan | 262144 | torch.float16 | torch.bool | 0.00 | 0.11 | 0.34 |
| isnan | 1048576 | torch.float16 | torch.bool | 0.00 | 0.34 | 1.01 |
| isnan | 4000000 | torch.float16 | torch.bool | 0.01 | 0.73 | 2.18 |

## unary_strategy

### relu_direct

| n_total | shape_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4194304 | 1024x4096 | torch.float16 | direct | 0.01 | 0.30 | 1.18 |
| 4194304 | 1024x4096 | torch.bfloat16 | direct | 0.01 | 0.30 | 1.18 |
| 4194304 | 1024x4096 | torch.float32 | direct | 0.02 | 0.27 | 2.15 |
| 10485760 | 1024x10240 | torch.float16 | direct | 0.03 | 0.32 | 1.29 |
| 10485760 | 1024x10240 | torch.bfloat16 | direct | 0.03 | 0.32 | 1.29 |
| 10485760 | 1024x10240 | torch.float32 | direct | 0.04 | 0.29 | 2.35 |
| 20971520 | 1024x20480 | torch.float16 | direct | 0.06 | 0.34 | 1.35 |
| 20971520 | 1024x20480 | torch.bfloat16 | direct | 0.06 | 0.33 | 1.33 |
| 20971520 | 1024x20480 | torch.float32 | direct | 0.07 | 0.31 | 2.47 |

### relu_explicit_parallel

| n_total | shape_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4194304 | 1024x4096 | torch.float16 | explicit_parallel | 0.02 | 0.22 | 0.88 |
| 4194304 | 1024x4096 | torch.bfloat16 | explicit_parallel | 0.02 | 0.22 | 0.88 |
| 4194304 | 1024x4096 | torch.float32 | explicit_parallel | 0.01 | 0.39 | 3.11 |
| 10485760 | 1024x10240 | torch.float16 | explicit_parallel | 0.04 | 0.25 | 0.99 |
| 10485760 | 1024x10240 | torch.bfloat16 | explicit_parallel | 0.04 | 0.25 | 0.99 |
| 10485760 | 1024x10240 | torch.float32 | explicit_parallel | 0.02 | 0.47 | 3.77 |
| 20971520 | 1024x20480 | torch.float16 | explicit_parallel | 0.08 | 0.26 | 1.03 |
| 20971520 | 1024x20480 | torch.bfloat16 | explicit_parallel | 0.08 | 0.26 | 1.04 |
| 20971520 | 1024x20480 | torch.float32 | explicit_parallel | 0.04 | 0.51 | 4.05 |

### relu_register_copy

| n_total | shape_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4194304 | 1024x4096 | torch.float16 | register_copy | 0.01 | 0.67 | 2.67 |
| 4194304 | 1024x4096 | torch.bfloat16 | register_copy | 0.01 | 0.67 | 2.67 |
| 4194304 | 1024x4096 | torch.float32 | register_copy | 0.01 | 0.40 | 3.19 |
| 10485760 | 1024x10240 | torch.float16 | register_copy | 0.01 | 0.85 | 3.39 |
| 10485760 | 1024x10240 | torch.bfloat16 | register_copy | 0.01 | 0.84 | 3.37 |
| 10485760 | 1024x10240 | torch.float32 | register_copy | 0.02 | 0.48 | 3.83 |
| 20971520 | 1024x20480 | torch.float16 | register_copy | 0.02 | 0.95 | 3.81 |
| 20971520 | 1024x20480 | torch.bfloat16 | register_copy | 0.02 | 0.96 | 3.85 |
| 20971520 | 1024x20480 | torch.float32 | register_copy | 0.04 | 0.52 | 4.13 |

## vector_norm

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | l1 | 65.81 | 0.00 | 0.00 |
| 1024 | 4096 | torch.bfloat16 | l1 | 66.13 | 0.00 | 0.00 |
| 1024 | 4096 | torch.float32 | l1 | 65.63 | 0.00 | 0.00 |
| 4096 | 4096 | torch.float16 | l1 | 65.60 | 0.00 | 0.00 |
| 1024 | 4096 | torch.float16 | l2 | 65.59 | 0.00 | 0.00 |
| 1024 | 4096 | torch.bfloat16 | l2 | 65.69 | 0.00 | 0.00 |
| 1024 | 4096 | torch.float32 | l2 | 65.99 | 0.00 | 0.00 |
| 4096 | 4096 | torch.float16 | l2 | 66.12 | 0.00 | 0.00 |
| 1024 | 4096 | torch.float16 | inf | 65.69 | 0.00 | 0.00 |
| 1024 | 4096 | torch.bfloat16 | inf | 66.10 | 0.00 | 0.00 |
| 1024 | 4096 | torch.float32 | inf | 65.81 | 0.00 | 0.00 |
| 4096 | 4096 | torch.float16 | inf | 66.07 | 0.00 | 0.00 |

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
| 1024 | 4096 | torch.bfloat16 | inf | 0.02 | 0.24 | 0.49 |
| 1024 | 4096 | torch.float32 | inf | 0.02 | 0.22 | 0.87 |
| 4096 | 4096 | torch.float16 | inf | 0.02 | 0.76 | 1.52 |

........................................................................ [  8%]
........................................................................ [ 17%]
........................................................................ [ 25%]
........................................................................ [ 34%]
........................................................................ [ 43%]
........................................................................ [ 51%]
........................................................................ [ 60%]
........................................................................ [ 69%]
........................................................................ [ 77%]
....................................FF.................................. [ 86%]
........................................................................ [ 95%]
........................................                                 [100%]Benchmark report saved to profile_run.log

=================================== FAILURES ===================================
____________________ test_mha_decode_bench[fp16-long-cache] ____________________

b = 1, h = 32, s_q = 128, s_kv = 8192, d = 128, dtype = torch.float16
tune = True

    @pytest.mark.parametrize("b, h, s_q, s_kv, d, dtype, tune", _MHA_DECODE_BENCH_PARAMS)
    def test_mha_decode_bench(b: int, h: int, s_q: int, s_kv: int, d: int, dtype: torch.dtype,
                              tune: bool) -> None:
        test = MhaDecodeTest(b, h, s_q, s_kv, d, dtype)
        bm = MhaDecodeBenchmark(test)
        inputs = test.gen_inputs()
    
        op = MultiHeadAttentionDecodeWithKVCacheOp(b, h, s_q, s_kv, d, dtype, tune=tune)
>       result = bm.profile(op, *inputs)
                 ^^^^^^^^^^^^^^^^^^^^^^^

benchmarks/ops/bench_mha_decode.py:43: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
benchmarks/benchmark.py:148: in profile
    latency = do_bench(bench_fn, warmup=warmup, rep=rep, backend='cupti')
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
../../_tool/tileops_benchmarks_venv_db4a9af2f48857c5/lib/python3.11/site-packages/tilelang/profiler/bench.py:100: in do_bench
    fn()
benchmarks/benchmark.py:141: in bench_fn
    return functor(*inputs)
           ^^^^^^^^^^^^^^^^
tileops/ops/op.py:79: in __call__
    return self.forward(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tileops/ops/mha_decode.py:49: in forward
    return self.kernel(q, k, v, real_seqlen_kv)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tileops/kernels/kernel.py:64: in __call__
    return self.forward(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tileops/kernels/flash_decode/mha_decode.py:448: in forward
    return _mha_decode_wrapped_kernel(self.batch, self.heads, self.seqlen_q, self.seqlen_kv,
../../_tool/tileops_benchmarks_venv_db4a9af2f48857c5/lib/python3.11/site-packages/torch/_library/custom_ops.py:676: in __call__
    return self._opoverload(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
../../_tool/tileops_benchmarks_venv_db4a9af2f48857c5/lib/python3.11/site-packages/torch/_ops.py:841: in __call__
    return self._op(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
../../_tool/tileops_benchmarks_venv_db4a9af2f48857c5/lib/python3.11/site-packages/torch/_library/autograd.py:111: in autograd_impl
    result = forward_no_grad(*args, Metadata(keyset, keyword_only_args))
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
../../_tool/tileops_benchmarks_venv_db4a9af2f48857c5/lib/python3.11/site-packages/torch/_library/autograd.py:40: in forward_no_grad
    result = op.redispatch(keyset & _C._after_autograd_keyset, *args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
../../_tool/tileops_benchmarks_venv_db4a9af2f48857c5/lib/python3.11/site-packages/torch/_ops.py:848: in redispatch
    return self._handle.redispatch_boxed(keyset, *args, **kwargs)  # type: ignore[return-value]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
../../_tool/tileops_benchmarks_venv_db4a9af2f48857c5/lib/python3.11/site-packages/torch/_library/custom_ops.py:343: in backend_impl
    result = self._backend_fns[device_type](*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
../../_tool/tileops_benchmarks_venv_db4a9af2f48857c5/lib/python3.11/site-packages/torch/_compile.py:53: in inner
    return disable_fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
../../_tool/tileops_benchmarks_venv_db4a9af2f48857c5/lib/python3.11/site-packages/torch/_dynamo/eval_frame.py:1044: in _fn
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
../../_tool/tileops_benchmarks_venv_db4a9af2f48857c5/lib/python3.11/site-packages/torch/_library/custom_ops.py:376: in wrapped_fn
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
tileops/kernels/flash_decode/mha_decode.py:336: in _mha_decode_wrapped_kernel
    return _mha_decode_kernel(batch, heads, seqlen_q, seqlen_kv, dim, is_causal,
../../_tool/tileops_benchmarks_venv_db4a9af2f48857c5/lib/python3.11/site-packages/tilelang/jit/kernel.py:207: in __call__
    return self.torch_function(*args, **kwds)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
../../_tool/tileops_benchmarks_venv_db4a9af2f48857c5/lib/python3.11/site-packages/tilelang/jit/adapter/tvm_ffi.py:244: in func
    executable(*tensor_list)
../../_tool/tileops_benchmarks_venv_db4a9af2f48857c5/lib/python3.11/site-packages/tilelang/3rdparty/tvm/python/tvm/runtime/executable.py:42: in __call__
    return self.jit().main(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

>   ???
E   RuntimeError: kernel mha_decode_split input Output_partial ndim expected 5, but got 4

python/tvm_ffi/cython/function.pxi:929: RuntimeError
----------------------------- Captured stdout call -----------------------------
Start autotuning mha_decode_kernel...
2026-03-23 19:54:10,586 INFO:Auto-tuning with 0.9 CPU utilizations, 240 CPUs available, 216 CPUs will be used
2026-03-23 19:54:19  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[-1]`
2026-03-23 19:54:20  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_no_split` with `out_idx=[4]`
2026-03-23 19:54:20  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-23 19:54:20  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-23 19:54:20  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_no_split` with `out_idx=[4]`
2026-03-23 19:54:20  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_no_split` with `out_idx=[4]`
2026-03-23 19:54:20  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_no_split` with `out_idx=[4]`
2026-03-23 19:54:20  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-23 19:54:20  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-23 19:54:20  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_no_split` with `out_idx=[4]`
2026-03-23 19:54:20  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_no_split` with `out_idx=[4]`
2026-03-23 19:54:20  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-23 19:54:20  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_no_split` with `out_idx=[4]`
2026-03-23 19:54:20  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-23 19:54:20  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-23 19:54:20  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_no_split` with `out_idx=[4]`
2026-03-23 19:54:20  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-23 19:54:20  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-23 19:54:20  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-23 19:54:20  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_no_split` with `out_idx=[4]`
2026-03-23 19:54:20  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-23 19:54:20  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-23 19:54:20  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-23 19:54:20  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-23 19:54:20  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_no_split` with `out_idx=[4]`
2026-03-23 19:54:20  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-23 19:54:20  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_no_split` with `out_idx=[4]`
2026-03-23 19:54:20  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-23 19:54:20  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_no_split` with `out_idx=[4]`
2026-03-23 19:54:20  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-23 19:54:20  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_no_split` with `out_idx=[4]`
2026-03-23 19:54:20  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-23 19:54:20  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-23 19:54:20  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-23 19:54:20  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-23 19:54:20  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_no_split` with `out_idx=[4]`
2026-03-23 19:54:20  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-23 19:54:20  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-23 19:54:20  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-23 19:54:20  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_no_split` with `out_idx=[4]`
2026-03-23 19:54:20  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-23 19:54:20  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-23 19:54:20  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-23 19:54:20  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-23 19:54:20  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-23 19:54:20  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-23 19:54:20  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-23 19:54:33  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_no_split`
2026-03-23 19:54:36  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_no_split`
2026-03-23 19:54:36  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_no_split`
2026-03-23 19:54:40  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_no_split`
2026-03-23 19:54:41  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_no_split`
2026-03-23 19:54:42  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_no_split`
2026-03-23 19:54:42  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_no_split`
2026-03-23 19:54:43  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_no_split`
2026-03-23 19:54:43  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_no_split`
2026-03-23 19:54:43  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_no_split`
2026-03-23 19:54:43  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_no_split`
2026-03-23 19:54:51  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-23 19:54:51  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-23 19:54:52  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-23 19:54:52  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-23 19:54:53  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-23 19:54:53  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-23 19:54:54  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-23 19:54:54  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-23 19:54:56  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-23 19:54:58  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-23 19:54:58  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-23 19:54:59  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-23 19:54:59  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-23 19:54:59  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-23 19:54:59  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-23 19:55:00  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-23 19:55:00  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-23 19:55:00  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-23 19:55:00  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-23 19:55:00  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-23 19:55:00  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-23 19:55:00  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-23 19:55:00  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-23 19:55:00  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-23 19:55:00  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-23 19:55:00  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-23 19:55:00  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-23 19:55:01  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-23 19:55:01  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-23 19:55:01  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-23 19:55:01  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-23 19:55:01  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
Tuned Latency 0.1449422985315323 with config {'block_M': 128, 'block_N': 64, 'num_split': 1, 'num_stages': 2, 'threads': 256} at index 0
Tuned Latency 0.17942790687084198 with config {'block_M': 128, 'block_N': 64, 'num_split': 1, 'num_stages': 3, 'threads': 128} at index 1
Tuned Latency 0.11045517772436142 with config {'block_M': 64, 'block_N': 64, 'num_split': 1, 'num_stages': 2, 'threads': 128} at index 2
Tuned Latency 0.178262859582901 with config {'block_M': 128, 'block_N': 64, 'num_split': 1, 'num_stages': 2, 'threads': 128} at index 3
Tuned Latency 0.09215757250785828 with config {'block_M': 64, 'block_N': 128, 'num_split': 1, 'num_stages': 2, 'threads': 128} at index 4
Tuned Latency 0.10815734416246414 with config {'block_M': 64, 'block_N': 64, 'num_split': 1, 'num_stages': 3, 'threads': 128} at index 5
Tuned Latency 0.12696711719036102 with config {'block_M': 128, 'block_N': 128, 'num_split': 1, 'num_stages': 3, 'threads': 256} at index 6
Tuned Latency 0.12265308201313019 with config {'block_M': 128, 'block_N': 128, 'num_split': 1, 'num_stages': 2, 'threads': 256} at index 7
Tuned Latency 0.09141381084918976 with config {'block_M': 64, 'block_N': 128, 'num_split': 1, 'num_stages': 3, 'threads': 128} at index 8
Tuned Latency 0.2876514494419098 with config {'block_M': 128, 'block_N': 128, 'num_split': 1, 'num_stages': 2, 'threads': 128} at index 9
Tuned Latency 0.1573488712310791 with config {'block_M': 128, 'block_N': 64, 'num_split': 1, 'num_stages': 3, 'threads': 256} at index 10
Tuned Latency 0.2916765809059143 with config {'block_M': 128, 'block_N': 128, 'num_split': 1, 'num_stages': 3, 'threads': 128} at index 11
Tuned Latency 0.01926920935511589 with config {'block_M': 128, 'block_N': 64, 'num_split': 4, 'num_stages': 3, 'threads': 128} at index 12
Tuned Latency 0.0194972213357687 with config {'block_M': 128, 'block_N': 64, 'num_split': 4, 'num_stages': 3, 'threads': 256} at index 13
Tuned Latency 0.015371058136224747 with config {'block_M': 64, 'block_N': 64, 'num_split': 2, 'num_stages': 2, 'threads': 256} at index 14
Tuned Latency 0.017309091985225677 with config {'block_M': 128, 'block_N': 64, 'num_split': 2, 'num_stages': 2, 'threads': 128} at index 15
Tuned Latency 0.015924233943223953 with config {'block_M': 64, 'block_N': 128, 'num_split': 2, 'num_stages': 3, 'threads': 256} at index 16
Tuned Latency 0.015307140536606312 with config {'block_M': 128, 'block_N': 128, 'num_split': 2, 'num_stages': 2, 'threads': 256} at index 17
Tuned Latency 0.018414059653878212 with config {'block_M': 64, 'block_N': 64, 'num_split': 4, 'num_stages': 2, 'threads': 256} at index 18
Tuned Latency 0.013161828741431236 with config {'block_M': 64, 'block_N': 64, 'num_split': 2, 'num_stages': 2, 'threads': 128} at index 19
Tuned Latency 0.016688726842403412 with config {'block_M': 128, 'block_N': 64, 'num_split': 4, 'num_stages': 2, 'threads': 256} at index 20
Tuned Latency 0.018028458580374718 with config {'block_M': 64, 'block_N': 64, 'num_split': 4, 'num_stages': 2, 'threads': 128} at index 21
Tuned Latency 0.017297083511948586 with config {'block_M': 128, 'block_N': 128, 'num_split': 4, 'num_stages': 2, 'threads': 128} at index 22
Tuned Latency 0.015229884535074234 with config {'block_M': 64, 'block_N': 64, 'num_split': 2, 'num_stages': 3, 'threads': 128} at index 23
Tuned Latency 0.02209858037531376 with config {'block_M': 64, 'block_N': 128, 'num_split': 4, 'num_stages': 3, 'threads': 256} at index 24
Tuned Latency 0.0160419549793005 with config {'block_M': 64, 'block_N': 128, 'num_split': 2, 'num_stages': 2, 'threads': 128} at index 25
Tuned Latency 0.013362186960875988 with config {'block_M': 64, 'block_N': 64, 'num_split': 2, 'num_stages': 3, 'threads': 256} at index 26
Tuned Latency 0.020022423937916756 with config {'block_M': 64, 'block_N': 64, 'num_split': 4, 'num_stages': 3, 'threads': 256} at index 27
Tuned Latency 0.01960979774594307 with config {'block_M': 128, 'block_N': 64, 'num_split': 4, 'num_stages': 2, 'threads': 128} at index 28
Tuned Latency 0.016079407185316086 with config {'block_M': 64, 'block_N': 64, 'num_split': 4, 'num_stages': 3, 'threads': 128} at index 29
Tuned Latency 0.013291969895362854 with config {'block_M': 64, 'block_N': 128, 'num_split': 2, 'num_stages': 2, 'threads': 256} at index 30
Tuned Latency 0.019700586795806885 with config {'block_M': 64, 'block_N': 128, 'num_split': 4, 'num_stages': 2, 'threads': 256} at index 31
Tuned Latency 0.025384338572621346 with config {'block_M': 128, 'block_N': 128, 'num_split': 4, 'num_stages': 3, 'threads': 128} at index 32
Tuned Latency 0.019041478633880615 with config {'block_M': 64, 'block_N': 128, 'num_split': 4, 'num_stages': 2, 'threads': 128} at index 33
Tuned Latency 0.016638796776533127 with config {'block_M': 64, 'block_N': 128, 'num_split': 2, 'num_stages': 3, 'threads': 128} at index 34
Tuned Latency 0.019710250198841095 with config {'block_M': 64, 'block_N': 128, 'num_split': 4, 'num_stages': 3, 'threads': 128} at index 35
Tuned Latency 0.018088549375534058 with config {'block_M': 128, 'block_N': 64, 'num_split': 2, 'num_stages': 3, 'threads': 128} at index 36
Tuned Latency 0.014750608243048191 with config {'block_M': 128, 'block_N': 64, 'num_split': 2, 'num_stages': 2, 'threads': 256} at index 37
Tuned Latency 0.017892850562930107 with config {'block_M': 128, 'block_N': 64, 'num_split': 2, 'num_stages': 3, 'threads': 256} at index 38
Tuned Latency 0.022540748119354248 with config {'block_M': 128, 'block_N': 128, 'num_split': 2, 'num_stages': 3, 'threads': 128} at index 39
Tuned Latency 0.02486453391611576 with config {'block_M': 128, 'block_N': 128, 'num_split': 4, 'num_stages': 3, 'threads': 256} at index 40
Tuned Latency 0.02409154362976551 with config {'block_M': 128, 'block_N': 128, 'num_split': 4, 'num_stages': 2, 'threads': 256} at index 41
Tuned Latency 0.015541176311671734 with config {'block_M': 128, 'block_N': 128, 'num_split': 2, 'num_stages': 3, 'threads': 256} at index 42
Tuned Latency 0.015371764078736305 with config {'block_M': 128, 'block_N': 128, 'num_split': 2, 'num_stages': 2, 'threads': 128} at index 43
Best config: {'block_M': 64, 'block_N': 64, 'num_split': 2, 'num_stages': 2, 'threads': 128}
mha_decode_kernel initialized with config: {'block_M': 64, 'block_N': 64, 'num_split': 2, 'num_stages': 2, 'threads': 128}
----------------------------- Captured stderr call -----------------------------
Compiling configurations:   0%|          | 0/48 [00:00<?, ?it/s]                                                                Compiling configurations:   0%|          | 0/48 [00:05<?, ?it/s]Compiling configurations:   2%|▏         | 1/48 [00:05<04:29,  5.74s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:05<04:29,  5.74s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:05<04:29,  5.74s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:05<04:29,  5.74s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:05<04:29,  5.74s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:05<04:29,  5.74s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:05<04:29,  5.74s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:05<04:29,  5.74s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:05<04:29,  5.74s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:05<04:29,  5.74s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:05<04:29,  5.74s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:05<04:29,  5.74s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:05<04:29,  5.74s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:05<04:29,  5.74s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:05<04:29,  5.74s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:05<04:29,  5.74s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:05<04:29,  5.74s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:05<04:29,  5.74s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:05<04:29,  5.74s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:05<04:29,  5.74s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:05<04:29,  5.74s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:05<04:29,  5.74s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:05<04:29,  5.74s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:05<04:29,  5.74s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:05<04:29,  5.74s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:05<04:29,  5.74s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:05<04:29,  5.74s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:05<04:29,  5.74s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:05<04:29,  5.74s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:05<04:29,  5.74s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:05<04:29,  5.74s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:05<04:29,  5.74s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:05<04:29,  5.74s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:05<04:29,  5.74s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:05<04:29,  5.74s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:05<04:29,  5.74s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:05<04:29,  5.74s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:05<04:29,  5.74s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:05<04:29,  5.74s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:05<04:29,  5.74s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:05<04:29,  5.74s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:05<04:29,  5.74s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:05<04:29,  5.74s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:05<04:29,  5.74s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:05<04:29,  5.74s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:05<04:29,  5.74s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:05<04:29,  5.74s/it]Compiling configurations:   4%|▍         | 2/48 [00:06<01:56,  2.53s/it]Compiling configurations:   6%|▋         | 3/48 [00:06<01:05,  1.46s/it]Compiling configurations:   8%|▊         | 4/48 [00:06<00:40,  1.08it/s]Compiling configurations:  10%|█         | 5/48 [00:06<00:27,  1.58it/s]                                                                        Compiling configurations:  10%|█         | 5/48 [00:19<00:27,  1.58it/s]Compiling configurations:  12%|█▎        | 6/48 [00:19<03:24,  4.86s/it]                                                                        Compiling configurations:  12%|█▎        | 6/48 [00:21<03:24,  4.86s/it]                                                                        Compiling configurations:  12%|█▎        | 6/48 [00:22<03:24,  4.86s/it]Compiling configurations:  15%|█▍        | 7/48 [00:22<02:55,  4.27s/it]Compiling configurations:  17%|█▋        | 8/48 [00:23<02:10,  3.26s/it]                                                                        Compiling configurations:  17%|█▋        | 8/48 [00:26<02:10,  3.26s/it]                                                                        Compiling configurations:  17%|█▋        | 8/48 [00:26<02:10,  3.26s/it]Compiling configurations:  19%|█▉        | 9/48 [00:27<02:09,  3.32s/it]Compiling configurations:  21%|██        | 10/48 [00:27<01:37,  2.56s/it]                                                                         Compiling configurations:  21%|██        | 10/48 [00:28<01:37,  2.56s/it]                                                                         Compiling configurations:  21%|██        | 10/48 [00:28<01:37,  2.56s/it]                                                                         Compiling configurations:  21%|██        | 10/48 [00:28<01:37,  2.56s/it]Compiling configurations:  23%|██▎       | 11/48 [00:28<01:15,  2.05s/it]                                                                         Compiling configurations:  23%|██▎       | 11/48 [00:29<01:15,  2.05s/it]                                                                         Compiling configurations:  23%|██▎       | 11/48 [00:29<01:15,  2.05s/it]Compiling configurations:  25%|██▌       | 12/48 [00:29<00:57,  1.59s/it]                                                                         Compiling configurations:  25%|██▌       | 12/48 [00:29<00:57,  1.59s/it]Compiling configurations:  27%|██▋       | 13/48 [00:29<00:43,  1.25s/it]Compiling configurations:  29%|██▉       | 14/48 [00:30<00:34,  1.02s/it]Compiling configurations:  31%|███▏      | 15/48 [00:30<00:27,  1.20it/s]Compiling configurations:  33%|███▎      | 16/48 [00:31<00:23,  1.38it/s]                                                                         Compiling configurations:  33%|███▎      | 16/48 [00:37<00:23,  1.38it/s]                                                                         Compiling configurations:  33%|███▎      | 16/48 [00:37<00:23,  1.38it/s]                                                                         Compiling configurations:  33%|███▎      | 16/48 [00:37<00:23,  1.38it/s]Compiling configurations:  35%|███▌      | 17/48 [00:38<01:19,  2.56s/it]                                                                         Compiling configurations:  35%|███▌      | 17/48 [00:38<01:19,  2.56s/it]Compiling configurations:  38%|███▊      | 18/48 [00:38<00:58,  1.94s/it]                                                                         Compiling configurations:  38%|███▊      | 18/48 [00:38<00:58,  1.94s/it]                                                                         Compiling configurations:  38%|███▊      | 18/48 [00:38<00:58,  1.94s/it]Compiling configurations:  40%|███▉      | 19/48 [00:38<00:43,  1.49s/it]Compiling configurations:  42%|████▏     | 20/48 [00:39<00:33,  1.19s/it]Compiling configurations:  44%|████▍     | 21/48 [00:39<00:26,  1.03it/s]                                                                         Compiling configurations:  44%|████▍     | 21/48 [00:40<00:26,  1.03it/s]Compiling configurations:  46%|████▌     | 22/48 [00:40<00:21,  1.18it/s]                                                                         Compiling configurations:  46%|████▌     | 22/48 [00:40<00:21,  1.18it/s]Compiling configurations:  48%|████▊     | 23/48 [00:40<00:18,  1.37it/s]Compiling configurations:  50%|█████     | 24/48 [00:41<00:15,  1.56it/s]                                                                         Compiling configurations:  50%|█████     | 24/48 [00:41<00:15,  1.56it/s]Compiling configurations:  52%|█████▏    | 25/48 [00:43<00:24,  1.06s/it]                                                                         Compiling configurations:  52%|█████▏    | 25/48 [00:44<00:24,  1.06s/it]                                                                         Compiling configurations:  52%|█████▏    | 25/48 [00:44<00:24,  1.06s/it]                                                                         Compiling configurations:  52%|█████▏    | 25/48 [00:45<00:24,  1.06s/it]Compiling configurations:  54%|█████▍    | 26/48 [00:45<00:28,  1.27s/it]                                                                         Compiling configurations:  54%|█████▍    | 26/48 [00:45<00:28,  1.27s/it]                                                                         Compiling configurations:  54%|█████▍    | 26/48 [00:45<00:28,  1.27s/it]                                                                         Compiling configurations:  54%|█████▍    | 26/48 [00:45<00:28,  1.27s/it]                                                                         Compiling configurations:  54%|█████▍    | 26/48 [00:45<00:28,  1.27s/it]                                                                         Compiling configurations:  54%|█████▍    | 26/48 [00:45<00:28,  1.27s/it]                                                                         Compiling configurations:  54%|█████▍    | 26/48 [00:45<00:28,  1.27s/it]Compiling configurations:  56%|█████▋    | 27/48 [00:45<00:22,  1.09s/it]                                                                         Compiling configurations:  56%|█████▋    | 27/48 [00:45<00:22,  1.09s/it]                                                                         Compiling configurations:  56%|█████▋    | 27/48 [00:45<00:22,  1.09s/it]                                                                         Compiling configurations:  56%|█████▋    | 27/48 [00:46<00:22,  1.09s/it]                                                                         Compiling configurations:  56%|█████▋    | 27/48 [00:46<00:22,  1.09s/it]                                                                         Compiling configurations:  56%|█████▋    | 27/48 [00:46<00:22,  1.09s/it]                                                                         Compiling configurations:  56%|█████▋    | 27/48 [00:46<00:22,  1.09s/it]                                                                         Compiling configurations:  56%|█████▋    | 27/48 [00:46<00:22,  1.09s/it]                                                                         Compiling configurations:  56%|█████▋    | 27/48 [00:46<00:22,  1.09s/it]Compiling configurations:  58%|█████▊    | 28/48 [00:46<00:18,  1.10it/s]                                                                         Compiling configurations:  58%|█████▊    | 28/48 [00:46<00:18,  1.10it/s]                                                                         Compiling configurations:  58%|█████▊    | 28/48 [00:46<00:18,  1.10it/s]                                                                         Compiling configurations:  58%|█████▊    | 28/48 [00:46<00:18,  1.10it/s]                                                                         Compiling configurations:  58%|█████▊    | 28/48 [00:46<00:18,  1.10it/s]                                                                         Compiling configurations:  58%|█████▊    | 28/48 [00:46<00:18,  1.10it/s]Compiling configurations:  60%|██████    | 29/48 [00:46<00:14,  1.29it/s]                                                                         Compiling configurations:  60%|██████    | 29/48 [00:46<00:14,  1.29it/s]Compiling configurations:  62%|██████▎   | 30/48 [00:47<00:12,  1.47it/s]Compiling configurations:  65%|██████▍   | 31/48 [00:47<00:10,  1.66it/s]Compiling configurations:  67%|██████▋   | 32/48 [00:48<00:08,  1.82it/s]Compiling configurations:  69%|██████▉   | 33/48 [00:48<00:07,  1.89it/s]Compiling configurations:  71%|███████   | 34/48 [00:49<00:07,  2.00it/s]Compiling configurations:  73%|███████▎  | 35/48 [00:49<00:06,  2.06it/s]Compiling configurations:  75%|███████▌  | 36/48 [00:49<00:05,  2.10it/s]Compiling configurations:  77%|███████▋  | 37/48 [00:50<00:05,  2.03it/s]Compiling configurations:  79%|███████▉  | 38/48 [00:50<00:04,  2.09it/s]Compiling configurations:  81%|████████▏ | 39/48 [00:51<00:04,  2.14it/s]Compiling configurations:  83%|████████▎ | 40/48 [00:51<00:03,  2.16it/s]Compiling configurations:  85%|████████▌ | 41/48 [00:52<00:03,  2.14it/s]Compiling configurations:  88%|████████▊ | 42/48 [00:52<00:02,  2.13it/s]Compiling configurations:  90%|████████▉ | 43/48 [00:53<00:02,  2.12it/s]Compiling configurations:  92%|█████████▏| 44/48 [00:53<00:01,  2.04it/s]Compiling configurations:  94%|█████████▍| 45/48 [00:54<00:01,  2.00it/s]Compiling configurations:  96%|█████████▌| 46/48 [00:54<00:01,  1.96it/s]Compiling configurations:  98%|█████████▊| 47/48 [00:55<00:00,  1.94it/s]Compiling configurations: 100%|██████████| 48/48 [00:55<00:00,  1.92it/s]Compiling configurations: 100%|██████████| 48/48 [00:55<00:00,  1.16s/it]
Bench configurations:   0%|          | 0/44 [00:00<?, ?it/s]Bench configurations:   0%|          | 0/44 [00:00<?, ?it/s, best_latency=0.145]                                                                                Bench configurations:   0%|          | 0/44 [00:00<?, ?it/s, best_latency=0.145]Bench configurations:   0%|          | 0/44 [00:00<?, ?it/s, best_latency=0.145]                                                                                Bench configurations:   0%|          | 0/44 [00:00<?, ?it/s, best_latency=0.145]Bench configurations:   5%|▍         | 2/44 [00:00<00:08,  5.04it/s, best_latency=0.145]Bench configurations:   5%|▍         | 2/44 [00:00<00:08,  5.04it/s, best_latency=0.11]                                                                                        Bench configurations:   5%|▍         | 2/44 [00:00<00:08,  5.04it/s, best_latency=0.11]Bench configurations:   7%|▋         | 3/44 [00:00<00:10,  4.00it/s, best_latency=0.11]Bench configurations:   7%|▋         | 3/44 [00:01<00:10,  4.00it/s, best_latency=0.11]                                                                                       Bench configurations:   7%|▋         | 3/44 [00:01<00:10,  4.00it/s, best_latency=0.11]Bench configurations:   9%|▉         | 4/44 [00:01<00:11,  3.41it/s, best_latency=0.11]Bench configurations:   9%|▉         | 4/44 [00:01<00:11,  3.41it/s, best_latency=0.0922]                                                                                         Bench configurations:   9%|▉         | 4/44 [00:01<00:11,  3.41it/s, best_latency=0.0922]Bench configurations:  11%|█▏        | 5/44 [00:01<00:12,  3.22it/s, best_latency=0.0922]Bench configurations:  11%|█▏        | 5/44 [00:01<00:12,  3.22it/s, best_latency=0.0922]                                                                                         Bench configurations:  11%|█▏        | 5/44 [00:01<00:12,  3.22it/s, best_latency=0.0922]Bench configurations:  14%|█▎        | 6/44 [00:01<00:11,  3.17it/s, best_latency=0.0922]Bench configurations:  14%|█▎        | 6/44 [00:02<00:11,  3.17it/s, best_latency=0.0922]                                                                                         Bench configurations:  14%|█▎        | 6/44 [00:02<00:11,  3.17it/s, best_latency=0.0922]Bench configurations:  16%|█▌        | 7/44 [00:02<00:12,  3.07it/s, best_latency=0.0922]Bench configurations:  16%|█▌        | 7/44 [00:02<00:12,  3.07it/s, best_latency=0.0922]                                                                                         Bench configurations:  16%|█▌        | 7/44 [00:02<00:12,  3.07it/s, best_latency=0.0922]Bench configurations:  18%|█▊        | 8/44 [00:02<00:11,  3.00it/s, best_latency=0.0922]Bench configurations:  18%|█▊        | 8/44 [00:02<00:11,  3.00it/s, best_latency=0.0914]                                                                                         Bench configurations:  18%|█▊        | 8/44 [00:02<00:11,  3.00it/s, best_latency=0.0914]Bench configurations:  20%|██        | 9/44 [00:02<00:11,  2.97it/s, best_latency=0.0914]Bench configurations:  20%|██        | 9/44 [00:03<00:11,  2.97it/s, best_latency=0.0914]                                                                                         Bench configurations:  20%|██        | 9/44 [00:03<00:11,  2.97it/s, best_latency=0.0914]Bench configurations:  23%|██▎       | 10/44 [00:03<00:12,  2.78it/s, best_latency=0.0914]Bench configurations:  23%|██▎       | 10/44 [00:03<00:12,  2.78it/s, best_latency=0.0914]                                                                                          Bench configurations:  23%|██▎       | 10/44 [00:03<00:12,  2.78it/s, best_latency=0.0914]Bench configurations:  25%|██▌       | 11/44 [00:03<00:11,  2.87it/s, best_latency=0.0914]Bench configurations:  25%|██▌       | 11/44 [00:03<00:11,  2.87it/s, best_latency=0.0914]                                                                                          Bench configurations:  25%|██▌       | 11/44 [00:03<00:11,  2.87it/s, best_latency=0.0914]Bench configurations:  27%|██▋       | 12/44 [00:03<00:11,  2.71it/s, best_latency=0.0914]Bench configurations:  27%|██▋       | 12/44 [00:04<00:11,  2.71it/s, best_latency=0.0193]                                                                                          Bench configurations:  27%|██▋       | 12/44 [00:04<00:11,  2.71it/s, best_latency=0.0193]Bench configurations:  30%|██▉       | 13/44 [00:04<00:12,  2.46it/s, best_latency=0.0193]Bench configurations:  30%|██▉       | 13/44 [00:04<00:12,  2.46it/s, best_latency=0.0193]                                                                                          Bench configurations:  30%|██▉       | 13/44 [00:04<00:12,  2.46it/s, best_latency=0.0193]Bench configurations:  32%|███▏      | 14/44 [00:04<00:12,  2.31it/s, best_latency=0.0193]Bench configurations:  32%|███▏      | 14/44 [00:05<00:12,  2.31it/s, best_latency=0.0154]                                                                                          Bench configurations:  32%|███▏      | 14/44 [00:05<00:12,  2.31it/s, best_latency=0.0154]Bench configurations:  34%|███▍      | 15/44 [00:05<00:12,  2.31it/s, best_latency=0.0154]Bench configurations:  34%|███▍      | 15/44 [00:05<00:12,  2.31it/s, best_latency=0.0154]                                                                                          Bench configurations:  34%|███▍      | 15/44 [00:05<00:12,  2.31it/s, best_latency=0.0154]Bench configurations:  36%|███▋      | 16/44 [00:05<00:12,  2.23it/s, best_latency=0.0154]Bench configurations:  36%|███▋      | 16/44 [00:06<00:12,  2.23it/s, best_latency=0.0154]                                                                                          Bench configurations:  36%|███▋      | 16/44 [00:06<00:12,  2.23it/s, best_latency=0.0154]Bench configurations:  39%|███▊      | 17/44 [00:06<00:12,  2.22it/s, best_latency=0.0154]Bench configurations:  39%|███▊      | 17/44 [00:06<00:12,  2.22it/s, best_latency=0.0153]                                                                                          Bench configurations:  39%|███▊      | 17/44 [00:06<00:12,  2.22it/s, best_latency=0.0153]Bench configurations:  41%|████      | 18/44 [00:06<00:12,  2.10it/s, best_latency=0.0153]Bench configurations:  41%|████      | 18/44 [00:07<00:12,  2.10it/s, best_latency=0.0153]                                                                                          Bench configurations:  41%|████      | 18/44 [00:07<00:12,  2.10it/s, best_latency=0.0153]Bench configurations:  43%|████▎     | 19/44 [00:07<00:11,  2.15it/s, best_latency=0.0153]Bench configurations:  43%|████▎     | 19/44 [00:07<00:11,  2.15it/s, best_latency=0.0132]                                                                                          Bench configurations:  43%|████▎     | 19/44 [00:07<00:11,  2.15it/s, best_latency=0.0132]Bench configurations:  45%|████▌     | 20/44 [00:07<00:10,  2.19it/s, best_latency=0.0132]Bench configurations:  45%|████▌     | 20/44 [00:08<00:10,  2.19it/s, best_latency=0.0132]                                                                                          Bench configurations:  45%|████▌     | 20/44 [00:08<00:10,  2.19it/s, best_latency=0.0132]Bench configurations:  48%|████▊     | 21/44 [00:08<00:10,  2.15it/s, best_latency=0.0132]Bench configurations:  48%|████▊     | 21/44 [00:08<00:10,  2.15it/s, best_latency=0.0132]                                                                                          Bench configurations:  48%|████▊     | 21/44 [00:08<00:10,  2.15it/s, best_latency=0.0132]Bench configurations:  50%|█████     | 22/44 [00:08<00:10,  2.19it/s, best_latency=0.0132]Bench configurations:  50%|█████     | 22/44 [00:09<00:10,  2.19it/s, best_latency=0.0132]                                                                                          Bench configurations:  50%|█████     | 22/44 [00:09<00:10,  2.19it/s, best_latency=0.0132]Bench configurations:  52%|█████▏    | 23/44 [00:09<00:10,  2.08it/s, best_latency=0.0132]Bench configurations:  52%|█████▏    | 23/44 [00:09<00:10,  2.08it/s, best_latency=0.0132]                                                                                          Bench configurations:  52%|█████▏    | 23/44 [00:09<00:10,  2.08it/s, best_latency=0.0132]Bench configurations:  55%|█████▍    | 24/44 [00:09<00:09,  2.14it/s, best_latency=0.0132]Bench configurations:  55%|█████▍    | 24/44 [00:10<00:09,  2.14it/s, best_latency=0.0132]                                                                                          Bench configurations:  55%|█████▍    | 24/44 [00:10<00:09,  2.14it/s, best_latency=0.0132]Bench configurations:  57%|█████▋    | 25/44 [00:10<00:08,  2.15it/s, best_latency=0.0132]Bench configurations:  57%|█████▋    | 25/44 [00:10<00:08,  2.15it/s, best_latency=0.0132]                                                                                          Bench configurations:  57%|█████▋    | 25/44 [00:10<00:08,  2.15it/s, best_latency=0.0132]Bench configurations:  59%|█████▉    | 26/44 [00:10<00:08,  2.17it/s, best_latency=0.0132]Bench configurations:  59%|█████▉    | 26/44 [00:10<00:08,  2.17it/s, best_latency=0.0132]                                                                                          Bench configurations:  59%|█████▉    | 26/44 [00:10<00:08,  2.17it/s, best_latency=0.0132]Bench configurations:  61%|██████▏   | 27/44 [00:10<00:07,  2.21it/s, best_latency=0.0132]Bench configurations:  61%|██████▏   | 27/44 [00:11<00:07,  2.21it/s, best_latency=0.0132]                                                                                          Bench configurations:  61%|██████▏   | 27/44 [00:11<00:07,  2.21it/s, best_latency=0.0132]Bench configurations:  64%|██████▎   | 28/44 [00:11<00:07,  2.23it/s, best_latency=0.0132]Bench configurations:  64%|██████▎   | 28/44 [00:11<00:07,  2.23it/s, best_latency=0.0132]                                                                                          Bench configurations:  64%|██████▎   | 28/44 [00:11<00:07,  2.23it/s, best_latency=0.0132]Bench configurations:  66%|██████▌   | 29/44 [00:11<00:06,  2.18it/s, best_latency=0.0132]Bench configurations:  66%|██████▌   | 29/44 [00:12<00:06,  2.18it/s, best_latency=0.0132]                                                                                          Bench configurations:  66%|██████▌   | 29/44 [00:12<00:06,  2.18it/s, best_latency=0.0132]Bench configurations:  68%|██████▊   | 30/44 [00:12<00:06,  2.21it/s, best_latency=0.0132]Bench configurations:  68%|██████▊   | 30/44 [00:12<00:06,  2.21it/s, best_latency=0.0132]                                                                                          Bench configurations:  68%|██████▊   | 30/44 [00:12<00:06,  2.21it/s, best_latency=0.0132]Bench configurations:  70%|███████   | 31/44 [00:12<00:05,  2.21it/s, best_latency=0.0132]Bench configurations:  70%|███████   | 31/44 [00:13<00:05,  2.21it/s, best_latency=0.0132]                                                                                          Bench configurations:  70%|███████   | 31/44 [00:13<00:05,  2.21it/s, best_latency=0.0132]Bench configurations:  73%|███████▎  | 32/44 [00:13<00:05,  2.20it/s, best_latency=0.0132]Bench configurations:  73%|███████▎  | 32/44 [00:13<00:05,  2.20it/s, best_latency=0.0132]                                                                                          Bench configurations:  73%|███████▎  | 32/44 [00:13<00:05,  2.20it/s, best_latency=0.0132]Bench configurations:  75%|███████▌  | 33/44 [00:13<00:05,  2.08it/s, best_latency=0.0132]Bench configurations:  75%|███████▌  | 33/44 [00:14<00:05,  2.08it/s, best_latency=0.0132]                                                                                          Bench configurations:  75%|███████▌  | 33/44 [00:14<00:05,  2.08it/s, best_latency=0.0132]Bench configurations:  77%|███████▋  | 34/44 [00:14<00:04,  2.11it/s, best_latency=0.0132]Bench configurations:  77%|███████▋  | 34/44 [00:14<00:04,  2.11it/s, best_latency=0.0132]                                                                                          Bench configurations:  77%|███████▋  | 34/44 [00:14<00:04,  2.11it/s, best_latency=0.0132]Bench configurations:  80%|███████▉  | 35/44 [00:14<00:04,  2.14it/s, best_latency=0.0132]Bench configurations:  80%|███████▉  | 35/44 [00:15<00:04,  2.14it/s, best_latency=0.0132]                                                                                          Bench configurations:  80%|███████▉  | 35/44 [00:15<00:04,  2.14it/s, best_latency=0.0132]Bench configurations:  82%|████████▏ | 36/44 [00:15<00:03,  2.15it/s, best_latency=0.0132]Bench configurations:  82%|████████▏ | 36/44 [00:15<00:03,  2.15it/s, best_latency=0.0132]                                                                                          Bench configurations:  82%|████████▏ | 36/44 [00:15<00:03,  2.15it/s, best_latency=0.0132]Bench configurations:  84%|████████▍ | 37/44 [00:15<00:03,  2.12it/s, best_latency=0.0132]Bench configurations:  84%|████████▍ | 37/44 [00:16<00:03,  2.12it/s, best_latency=0.0132]                                                                                          Bench configurations:  84%|████████▍ | 37/44 [00:16<00:03,  2.12it/s, best_latency=0.0132]Bench configurations:  86%|████████▋ | 38/44 [00:16<00:02,  2.10it/s, best_latency=0.0132]Bench configurations:  86%|████████▋ | 38/44 [00:16<00:02,  2.10it/s, best_latency=0.0132]                                                                                          Bench configurations:  86%|████████▋ | 38/44 [00:16<00:02,  2.10it/s, best_latency=0.0132]Bench configurations:  89%|████████▊ | 39/44 [00:16<00:02,  2.09it/s, best_latency=0.0132]Bench configurations:  89%|████████▊ | 39/44 [00:17<00:02,  2.09it/s, best_latency=0.0132]                                                                                          Bench configurations:  89%|████████▊ | 39/44 [00:17<00:02,  2.09it/s, best_latency=0.0132]Bench configurations:  91%|█████████ | 40/44 [00:17<00:01,  2.02it/s, best_latency=0.0132]Bench configurations:  91%|█████████ | 40/44 [00:17<00:01,  2.02it/s, best_latency=0.0132]                                                                                          Bench configurations:  91%|█████████ | 40/44 [00:17<00:01,  2.02it/s, best_latency=0.0132]Bench configurations:  93%|█████████▎| 41/44 [00:17<00:01,  1.97it/s, best_latency=0.0132]Bench configurations:  93%|█████████▎| 41/44 [00:18<00:01,  1.97it/s, best_latency=0.0132]                                                                                          Bench configurations:  93%|█████████▎| 41/44 [00:18<00:01,  1.97it/s, best_latency=0.0132]Bench configurations:  95%|█████████▌| 42/44 [00:18<00:01,  1.93it/s, best_latency=0.0132]Bench configurations:  95%|█████████▌| 42/44 [00:18<00:01,  1.93it/s, best_latency=0.0132]                                                                                          Bench configurations:  95%|█████████▌| 42/44 [00:18<00:01,  1.93it/s, best_latency=0.0132]Bench configurations:  98%|█████████▊| 43/44 [00:18<00:00,  1.91it/s, best_latency=0.0132]Bench configurations:  98%|█████████▊| 43/44 [00:19<00:00,  1.91it/s, best_latency=0.0132]                                                                                          Bench configurations:  98%|█████████▊| 43/44 [00:19<00:00,  1.91it/s, best_latency=0.0132]Bench configurations: 100%|██████████| 44/44 [00:19<00:00,  1.89it/s, best_latency=0.0132]Bench configurations: 100%|██████████| 44/44 [00:19<00:00,  2.28it/s, best_latency=0.0132]
____________________ test_mha_decode_bench[bf16-long-cache] ____________________

b = 1, h = 32, s_q = 128, s_kv = 8192, d = 128, dtype = torch.bfloat16
tune = True

    @pytest.mark.parametrize("b, h, s_q, s_kv, d, dtype, tune", _MHA_DECODE_BENCH_PARAMS)
    def test_mha_decode_bench(b: int, h: int, s_q: int, s_kv: int, d: int, dtype: torch.dtype,
                              tune: bool) -> None:
        test = MhaDecodeTest(b, h, s_q, s_kv, d, dtype)
        bm = MhaDecodeBenchmark(test)
        inputs = test.gen_inputs()
    
        op = MultiHeadAttentionDecodeWithKVCacheOp(b, h, s_q, s_kv, d, dtype, tune=tune)
>       result = bm.profile(op, *inputs)
                 ^^^^^^^^^^^^^^^^^^^^^^^

benchmarks/ops/bench_mha_decode.py:43: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
benchmarks/benchmark.py:148: in profile
    latency = do_bench(bench_fn, warmup=warmup, rep=rep, backend='cupti')
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
../../_tool/tileops_benchmarks_venv_db4a9af2f48857c5/lib/python3.11/site-packages/tilelang/profiler/bench.py:100: in do_bench
    fn()
benchmarks/benchmark.py:141: in bench_fn
    return functor(*inputs)
           ^^^^^^^^^^^^^^^^
tileops/ops/op.py:79: in __call__
    return self.forward(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tileops/ops/mha_decode.py:49: in forward
    return self.kernel(q, k, v, real_seqlen_kv)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tileops/kernels/kernel.py:64: in __call__
    return self.forward(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tileops/kernels/flash_decode/mha_decode.py:448: in forward
    return _mha_decode_wrapped_kernel(self.batch, self.heads, self.seqlen_q, self.seqlen_kv,
../../_tool/tileops_benchmarks_venv_db4a9af2f48857c5/lib/python3.11/site-packages/torch/_library/custom_ops.py:676: in __call__
    return self._opoverload(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
../../_tool/tileops_benchmarks_venv_db4a9af2f48857c5/lib/python3.11/site-packages/torch/_ops.py:841: in __call__
    return self._op(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
../../_tool/tileops_benchmarks_venv_db4a9af2f48857c5/lib/python3.11/site-packages/torch/_library/autograd.py:111: in autograd_impl
    result = forward_no_grad(*args, Metadata(keyset, keyword_only_args))
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
../../_tool/tileops_benchmarks_venv_db4a9af2f48857c5/lib/python3.11/site-packages/torch/_library/autograd.py:40: in forward_no_grad
    result = op.redispatch(keyset & _C._after_autograd_keyset, *args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
../../_tool/tileops_benchmarks_venv_db4a9af2f48857c5/lib/python3.11/site-packages/torch/_ops.py:848: in redispatch
    return self._handle.redispatch_boxed(keyset, *args, **kwargs)  # type: ignore[return-value]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
../../_tool/tileops_benchmarks_venv_db4a9af2f48857c5/lib/python3.11/site-packages/torch/_library/custom_ops.py:343: in backend_impl
    result = self._backend_fns[device_type](*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
../../_tool/tileops_benchmarks_venv_db4a9af2f48857c5/lib/python3.11/site-packages/torch/_compile.py:53: in inner
    return disable_fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
../../_tool/tileops_benchmarks_venv_db4a9af2f48857c5/lib/python3.11/site-packages/torch/_dynamo/eval_frame.py:1044: in _fn
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
../../_tool/tileops_benchmarks_venv_db4a9af2f48857c5/lib/python3.11/site-packages/torch/_library/custom_ops.py:376: in wrapped_fn
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
tileops/kernels/flash_decode/mha_decode.py:336: in _mha_decode_wrapped_kernel
    return _mha_decode_kernel(batch, heads, seqlen_q, seqlen_kv, dim, is_causal,
../../_tool/tileops_benchmarks_venv_db4a9af2f48857c5/lib/python3.11/site-packages/tilelang/jit/kernel.py:207: in __call__
    return self.torch_function(*args, **kwds)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
../../_tool/tileops_benchmarks_venv_db4a9af2f48857c5/lib/python3.11/site-packages/tilelang/jit/adapter/tvm_ffi.py:244: in func
    executable(*tensor_list)
../../_tool/tileops_benchmarks_venv_db4a9af2f48857c5/lib/python3.11/site-packages/tilelang/3rdparty/tvm/python/tvm/runtime/executable.py:42: in __call__
    return self.jit().main(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

>   ???
E   RuntimeError: kernel mha_decode_split input Output_partial ndim expected 5, but got 4

python/tvm_ffi/cython/function.pxi:929: RuntimeError
----------------------------- Captured stdout call -----------------------------
Start autotuning mha_decode_kernel...
2026-03-23 19:55:30,482 INFO:Auto-tuning with 0.9 CPU utilizations, 240 CPUs available, 216 CPUs will be used
2026-03-23 19:55:39  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[-1]`
2026-03-23 19:55:39  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-23 19:55:39  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-23 19:55:39  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_no_split` with `out_idx=[4]`
2026-03-23 19:55:40  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-23 19:55:40  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-23 19:55:40  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-23 19:55:40  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-23 19:55:40  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-23 19:55:40  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-23 19:55:40  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-23 19:55:40  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_no_split` with `out_idx=[4]`
2026-03-23 19:55:40  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_no_split` with `out_idx=[4]`
2026-03-23 19:55:40  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-23 19:55:40  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-23 19:55:40  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-23 19:55:40  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-23 19:55:40  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_no_split` with `out_idx=[4]`
2026-03-23 19:55:40  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-23 19:55:40  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-23 19:55:40  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-23 19:55:40  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-23 19:55:40  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-23 19:55:40  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_no_split` with `out_idx=[4]`
2026-03-23 19:55:40  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-23 19:55:40  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-23 19:55:40  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-23 19:55:40  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-23 19:55:40  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-23 19:55:40  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-23 19:55:40  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-23 19:55:40  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-23 19:55:40  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-23 19:55:40  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-23 19:55:40  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-23 19:55:40  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-23 19:55:40  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-23 19:55:51  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_no_split`
2026-03-23 19:56:07  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-23 19:56:08  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-23 19:56:09  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-23 19:56:09  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-23 19:56:09  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-23 19:56:09  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-23 19:56:09  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-23 19:56:09  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-23 19:56:09  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-23 19:56:09  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-23 19:56:09  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-23 19:56:09  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-23 19:56:09  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-23 19:56:09  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-23 19:56:10  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-23 19:56:10  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-23 19:56:10  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-23 19:56:10  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-23 19:56:10  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-23 19:56:10  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-23 19:56:10  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-23 19:56:10  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-23 19:56:10  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-23 19:56:10  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-23 19:56:10  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-23 19:56:10  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-23 19:56:10  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-23 19:56:11  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-23 19:56:11  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-23 19:56:11  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-23 19:56:11  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-23 19:56:11  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
Tuned Latency 0.1073061153292656 with config {'block_M': 64, 'block_N': 64, 'num_split': 1, 'num_stages': 3, 'threads': 128} at index 0
Tuned Latency 0.11044555902481079 with config {'block_M': 64, 'block_N': 64, 'num_split': 1, 'num_stages': 2, 'threads': 128} at index 1
Tuned Latency 0.09143737703561783 with config {'block_M': 64, 'block_N': 128, 'num_split': 1, 'num_stages': 2, 'threads': 128} at index 2
Tuned Latency 0.14581550657749176 with config {'block_M': 128, 'block_N': 64, 'num_split': 1, 'num_stages': 2, 'threads': 256} at index 3
Tuned Latency 0.2885257303714752 with config {'block_M': 128, 'block_N': 128, 'num_split': 1, 'num_stages': 2, 'threads': 128} at index 4
Tuned Latency 0.12282996624708176 with config {'block_M': 128, 'block_N': 128, 'num_split': 1, 'num_stages': 2, 'threads': 256} at index 5
Tuned Latency 0.15655232965946198 with config {'block_M': 128, 'block_N': 64, 'num_split': 1, 'num_stages': 3, 'threads': 256} at index 6
Tuned Latency 0.1789843738079071 with config {'block_M': 128, 'block_N': 64, 'num_split': 1, 'num_stages': 3, 'threads': 128} at index 7
Tuned Latency 0.29198169708251953 with config {'block_M': 128, 'block_N': 128, 'num_split': 1, 'num_stages': 3, 'threads': 128} at index 8
Tuned Latency 0.12632237374782562 with config {'block_M': 128, 'block_N': 128, 'num_split': 1, 'num_stages': 3, 'threads': 256} at index 9
Tuned Latency 0.177635058760643 with config {'block_M': 128, 'block_N': 64, 'num_split': 1, 'num_stages': 2, 'threads': 128} at index 10
Tuned Latency 0.09139345586299896 with config {'block_M': 64, 'block_N': 128, 'num_split': 1, 'num_stages': 3, 'threads': 128} at index 11
Tuned Latency 0.013526791706681252 with config {'block_M': 64, 'block_N': 64, 'num_split': 2, 'num_stages': 3, 'threads': 128} at index 12
Tuned Latency 0.012707631103694439 with config {'block_M': 64, 'block_N': 64, 'num_split': 2, 'num_stages': 2, 'threads': 128} at index 13
Tuned Latency 0.01640167087316513 with config {'block_M': 64, 'block_N': 64, 'num_split': 4, 'num_stages': 3, 'threads': 128} at index 14
Tuned Latency 0.01871429570019245 with config {'block_M': 128, 'block_N': 64, 'num_split': 4, 'num_stages': 3, 'threads': 256} at index 15
Tuned Latency 0.01762642338871956 with config {'block_M': 128, 'block_N': 64, 'num_split': 2, 'num_stages': 2, 'threads': 128} at index 16
Tuned Latency 0.015367940999567509 with config {'block_M': 128, 'block_N': 128, 'num_split': 2, 'num_stages': 3, 'threads': 256} at index 17
Tuned Latency 0.013195199891924858 with config {'block_M': 64, 'block_N': 64, 'num_split': 2, 'num_stages': 2, 'threads': 256} at index 18
Tuned Latency 0.016525372862815857 with config {'block_M': 64, 'block_N': 128, 'num_split': 2, 'num_stages': 2, 'threads': 256} at index 19
Tuned Latency 0.013097142800688744 with config {'block_M': 64, 'block_N': 64, 'num_split': 2, 'num_stages': 3, 'threads': 256} at index 20
Tuned Latency 0.02406585030257702 with config {'block_M': 128, 'block_N': 128, 'num_split': 4, 'num_stages': 3, 'threads': 256} at index 21
Tuned Latency 0.02317849174141884 with config {'block_M': 128, 'block_N': 128, 'num_split': 2, 'num_stages': 2, 'threads': 128} at index 22
Tuned Latency 0.02130713500082493 with config {'block_M': 64, 'block_N': 128, 'num_split': 4, 'num_stages': 2, 'threads': 128} at index 23
Tuned Latency 0.013504457660019398 with config {'block_M': 64, 'block_N': 128, 'num_split': 2, 'num_stages': 3, 'threads': 128} at index 24
Tuned Latency 0.015953540802001953 with config {'block_M': 64, 'block_N': 128, 'num_split': 2, 'num_stages': 3, 'threads': 256} at index 25
Tuned Latency 0.01838449202477932 with config {'block_M': 64, 'block_N': 64, 'num_split': 4, 'num_stages': 2, 'threads': 128} at index 26
Tuned Latency 0.01632811687886715 with config {'block_M': 64, 'block_N': 128, 'num_split': 4, 'num_stages': 2, 'threads': 256} at index 27
Tuned Latency 0.018337463960051537 with config {'block_M': 64, 'block_N': 64, 'num_split': 4, 'num_stages': 2, 'threads': 256} at index 28
Tuned Latency 0.02022274024784565 with config {'block_M': 64, 'block_N': 64, 'num_split': 4, 'num_stages': 3, 'threads': 256} at index 29
Tuned Latency 0.01341764722019434 with config {'block_M': 64, 'block_N': 128, 'num_split': 2, 'num_stages': 2, 'threads': 128} at index 30
Tuned Latency 0.022764744237065315 with config {'block_M': 64, 'block_N': 128, 'num_split': 4, 'num_stages': 3, 'threads': 256} at index 31
Tuned Latency 0.019611718133091927 with config {'block_M': 128, 'block_N': 64, 'num_split': 4, 'num_stages': 3, 'threads': 128} at index 32
Tuned Latency 0.01873413845896721 with config {'block_M': 128, 'block_N': 64, 'num_split': 4, 'num_stages': 2, 'threads': 128} at index 33
Tuned Latency 0.022744258865714073 with config {'block_M': 64, 'block_N': 128, 'num_split': 4, 'num_stages': 3, 'threads': 128} at index 34
Tuned Latency 0.017336968332529068 with config {'block_M': 128, 'block_N': 64, 'num_split': 2, 'num_stages': 2, 'threads': 256} at index 35
Tuned Latency 0.01909686252474785 with config {'block_M': 128, 'block_N': 64, 'num_split': 4, 'num_stages': 2, 'threads': 256} at index 36
Tuned Latency 0.017482668161392212 with config {'block_M': 128, 'block_N': 64, 'num_split': 2, 'num_stages': 3, 'threads': 128} at index 37
Tuned Latency 0.017566289752721786 with config {'block_M': 128, 'block_N': 64, 'num_split': 2, 'num_stages': 3, 'threads': 256} at index 38
Tuned Latency 0.022273290902376175 with config {'block_M': 128, 'block_N': 128, 'num_split': 2, 'num_stages': 2, 'threads': 256} at index 39
Tuned Latency 0.0225225780159235 with config {'block_M': 128, 'block_N': 128, 'num_split': 2, 'num_stages': 3, 'threads': 128} at index 40
Tuned Latency 0.0246217530220747 with config {'block_M': 128, 'block_N': 128, 'num_split': 4, 'num_stages': 3, 'threads': 128} at index 41
Tuned Latency 0.02417401596903801 with config {'block_M': 128, 'block_N': 128, 'num_split': 4, 'num_stages': 2, 'threads': 256} at index 42
Tuned Latency 0.02456938661634922 with config {'block_M': 128, 'block_N': 128, 'num_split': 4, 'num_stages': 2, 'threads': 128} at index 43
Best config: {'block_M': 64, 'block_N': 64, 'num_split': 2, 'num_stages': 2, 'threads': 128}
mha_decode_kernel initialized with config: {'block_M': 64, 'block_N': 64, 'num_split': 2, 'num_stages': 2, 'threads': 128}
----------------------------- Captured stderr call -----------------------------
Compiling configurations:   0%|          | 0/48 [00:00<?, ?it/s]                                                                Compiling configurations:   0%|          | 0/48 [00:06<?, ?it/s]Compiling configurations:   2%|▏         | 1/48 [00:06<04:54,  6.26s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<04:54,  6.26s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<04:54,  6.26s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<04:54,  6.26s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<04:54,  6.26s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<04:54,  6.26s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<04:54,  6.26s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<04:54,  6.26s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<04:54,  6.26s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<04:54,  6.26s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<04:54,  6.26s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<04:54,  6.26s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<04:54,  6.26s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<04:54,  6.26s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<04:54,  6.26s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<04:54,  6.26s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<04:54,  6.26s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<04:54,  6.26s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<04:54,  6.26s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<04:54,  6.26s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<04:54,  6.26s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<04:54,  6.26s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<04:54,  6.26s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<04:54,  6.26s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<04:54,  6.26s/it]Compiling configurations:  17%|█▋        | 8/48 [00:06<00:23,  1.70it/s]                                                                        Compiling configurations:  17%|█▋        | 8/48 [00:06<00:23,  1.70it/s]                                                                        Compiling configurations:  17%|█▋        | 8/48 [00:06<00:23,  1.70it/s]                                                                        Compiling configurations:  17%|█▋        | 8/48 [00:06<00:23,  1.70it/s]                                                                        Compiling configurations:  17%|█▋        | 8/48 [00:06<00:23,  1.70it/s]                                                                        Compiling configurations:  17%|█▋        | 8/48 [00:06<00:23,  1.70it/s]                                                                        Compiling configurations:  17%|█▋        | 8/48 [00:06<00:23,  1.70it/s]                                                                        Compiling configurations:  17%|█▋        | 8/48 [00:06<00:23,  1.70it/s]                                                                        Compiling configurations:  17%|█▋        | 8/48 [00:06<00:23,  1.70it/s]                                                                        Compiling configurations:  17%|█▋        | 8/48 [00:06<00:23,  1.70it/s]                                                                        Compiling configurations:  17%|█▋        | 8/48 [00:06<00:23,  1.70it/s]                                                                        Compiling configurations:  17%|█▋        | 8/48 [00:06<00:23,  1.70it/s]                                                                        Compiling configurations:  17%|█▋        | 8/48 [00:06<00:23,  1.70it/s]Compiling configurations:  27%|██▋       | 13/48 [00:06<00:11,  3.03it/s]                                                                         Compiling configurations:  27%|██▋       | 13/48 [00:17<00:11,  3.03it/s]Compiling configurations:  33%|███▎      | 16/48 [00:17<00:42,  1.31s/it]                                                                         Compiling configurations:  33%|███▎      | 16/48 [00:34<00:42,  1.31s/it]                                                                         Compiling configurations:  33%|███▎      | 16/48 [00:34<00:42,  1.31s/it]Compiling configurations:  35%|███▌      | 17/48 [00:34<01:39,  3.21s/it]Compiling configurations:  38%|███▊      | 18/48 [00:35<01:23,  2.80s/it]                                                                         Compiling configurations:  38%|███▊      | 18/48 [00:35<01:23,  2.80s/it]                                                                         Compiling configurations:  38%|███▊      | 18/48 [00:35<01:23,  2.80s/it]                                                                         Compiling configurations:  38%|███▊      | 18/48 [00:35<01:23,  2.80s/it]                                                                         Compiling configurations:  38%|███▊      | 18/48 [00:35<01:23,  2.80s/it]                                                                         Compiling configurations:  38%|███▊      | 18/48 [00:35<01:23,  2.80s/it]                                                                         Compiling configurations:  38%|███▊      | 18/48 [00:35<01:23,  2.80s/it]                                                                         Compiling configurations:  38%|███▊      | 18/48 [00:36<01:23,  2.80s/it]                                                                         Compiling configurations:  38%|███▊      | 18/48 [00:36<01:23,  2.80s/it]                                                                         Compiling configurations:  38%|███▊      | 18/48 [00:36<01:23,  2.80s/it]                                                                         Compiling configurations:  38%|███▊      | 18/48 [00:36<01:23,  2.80s/it]                                                                         Compiling configurations:  38%|███▊      | 18/48 [00:36<01:23,  2.80s/it]                                                                         Compiling configurations:  38%|███▊      | 18/48 [00:36<01:23,  2.80s/it]Compiling configurations:  42%|████▏     | 20/48 [00:36<01:00,  2.15s/it]                                                                         Compiling configurations:  42%|████▏     | 20/48 [00:36<01:00,  2.15s/it]                                                                         Compiling configurations:  42%|████▏     | 20/48 [00:36<01:00,  2.15s/it]                                                                         Compiling configurations:  42%|████▏     | 20/48 [00:36<01:00,  2.15s/it]                                                                         Compiling configurations:  42%|████▏     | 20/48 [00:36<01:00,  2.15s/it]                                                                         Compiling configurations:  42%|████▏     | 20/48 [00:36<01:00,  2.15s/it]                                                                         Compiling configurations:  42%|████▏     | 20/48 [00:36<01:00,  2.15s/it]                                                                         Compiling configurations:  42%|████▏     | 20/48 [00:36<01:00,  2.15s/it]                                                                         Compiling configurations:  42%|████▏     | 20/48 [00:37<01:00,  2.15s/it]                                                                         Compiling configurations:  42%|████▏     | 20/48 [00:37<01:00,  2.15s/it]                                                                         Compiling configurations:  42%|████▏     | 20/48 [00:37<01:00,  2.15s/it]                                                                         Compiling configurations:  42%|████▏     | 20/48 [00:37<01:00,  2.15s/it]                                                                         Compiling configurations:  42%|████▏     | 20/48 [00:37<01:00,  2.15s/it]                                                                         Compiling configurations:  42%|████▏     | 20/48 [00:37<01:00,  2.15s/it]                                                                         Compiling configurations:  42%|████▏     | 20/48 [00:37<01:00,  2.15s/it]Compiling configurations:  46%|████▌     | 22/48 [00:37<00:43,  1.67s/it]                                                                         Compiling configurations:  46%|████▌     | 22/48 [00:37<00:43,  1.67s/it]                                                                         Compiling configurations:  46%|████▌     | 22/48 [00:37<00:43,  1.67s/it]                                                                         Compiling configurations:  46%|████▌     | 22/48 [00:37<00:43,  1.67s/it]                                                                         Compiling configurations:  46%|████▌     | 22/48 [00:37<00:43,  1.67s/it]Compiling configurations:  48%|████▊     | 23/48 [00:37<00:36,  1.45s/it]Compiling configurations:  50%|█████     | 24/48 [00:38<00:30,  1.25s/it]Compiling configurations:  52%|█████▏    | 25/48 [00:38<00:24,  1.07s/it]Compiling configurations:  54%|█████▍    | 26/48 [00:39<00:20,  1.07it/s]Compiling configurations:  56%|█████▋    | 27/48 [00:39<00:17,  1.20it/s]Compiling configurations:  58%|█████▊    | 28/48 [00:40<00:14,  1.37it/s]Compiling configurations:  60%|██████    | 29/48 [00:40<00:12,  1.53it/s]Compiling configurations:  62%|██████▎   | 30/48 [00:41<00:10,  1.68it/s]Compiling configurations:  65%|██████▍   | 31/48 [00:41<00:09,  1.83it/s]Compiling configurations:  67%|██████▋   | 32/48 [00:41<00:08,  1.93it/s]Compiling configurations:  69%|██████▉   | 33/48 [00:42<00:07,  2.03it/s]Compiling configurations:  71%|███████   | 34/48 [00:42<00:06,  2.12it/s]Compiling configurations:  73%|███████▎  | 35/48 [00:43<00:06,  2.16it/s]Compiling configurations:  75%|███████▌  | 36/48 [00:43<00:05,  2.18it/s]Compiling configurations:  77%|███████▋  | 37/48 [00:44<00:05,  2.16it/s]Compiling configurations:  79%|███████▉  | 38/48 [00:44<00:04,  2.14it/s]Compiling configurations:  81%|████████▏ | 39/48 [00:45<00:04,  2.18it/s]Compiling configurations:  83%|████████▎ | 40/48 [00:45<00:03,  2.15it/s]Compiling configurations:  85%|████████▌ | 41/48 [00:46<00:03,  2.14it/s]Compiling configurations:  88%|████████▊ | 42/48 [00:46<00:02,  2.13it/s]Compiling configurations:  90%|████████▉ | 43/48 [00:46<00:02,  2.13it/s]Compiling configurations:  92%|█████████▏| 44/48 [00:47<00:01,  2.05it/s]Compiling configurations:  94%|█████████▍| 45/48 [00:48<00:01,  2.01it/s]Compiling configurations:  96%|█████████▌| 46/48 [00:48<00:01,  1.97it/s]Compiling configurations:  98%|█████████▊| 47/48 [00:49<00:00,  1.94it/s]Compiling configurations: 100%|██████████| 48/48 [00:49<00:00,  1.93it/s]Compiling configurations: 100%|██████████| 48/48 [00:49<00:00,  1.03s/it]
Bench configurations:   0%|          | 0/44 [00:00<?, ?it/s]Bench configurations:   0%|          | 0/44 [00:00<?, ?it/s, best_latency=0.107]                                                                                Bench configurations:   0%|          | 0/44 [00:00<?, ?it/s, best_latency=0.107]Bench configurations:   0%|          | 0/44 [00:00<?, ?it/s, best_latency=0.107]                                                                                Bench configurations:   0%|          | 0/44 [00:00<?, ?it/s, best_latency=0.107]Bench configurations:   0%|          | 0/44 [00:00<?, ?it/s, best_latency=0.0914]                                                                                 Bench configurations:   0%|          | 0/44 [00:00<?, ?it/s, best_latency=0.0914]Bench configurations:   0%|          | 0/44 [00:00<?, ?it/s, best_latency=0.0914]                                                                                 Bench configurations:   0%|          | 0/44 [00:00<?, ?it/s, best_latency=0.0914]Bench configurations:   0%|          | 0/44 [00:00<?, ?it/s, best_latency=0.0914]                                                                                 Bench configurations:   0%|          | 0/44 [00:00<?, ?it/s, best_latency=0.0914]Bench configurations:   0%|          | 0/44 [00:00<?, ?it/s, best_latency=0.0914]                                                                                 Bench configurations:   0%|          | 0/44 [00:00<?, ?it/s, best_latency=0.0914]Bench configurations:  14%|█▎        | 6/44 [00:00<00:00, 51.54it/s, best_latency=0.0914]Bench configurations:  14%|█▎        | 6/44 [00:00<00:00, 51.54it/s, best_latency=0.0914]                                                                                         Bench configurations:  14%|█▎        | 6/44 [00:00<00:00, 51.54it/s, best_latency=0.0914]Bench configurations:  14%|█▎        | 6/44 [00:00<00:00, 51.54it/s, best_latency=0.0914]                                                                                         Bench configurations:  14%|█▎        | 6/44 [00:00<00:00, 51.54it/s, best_latency=0.0914]Bench configurations:  14%|█▎        | 6/44 [00:00<00:00, 51.54it/s, best_latency=0.0914]                                                                                         Bench configurations:  14%|█▎        | 6/44 [00:00<00:00, 51.54it/s, best_latency=0.0914]Bench configurations:  14%|█▎        | 6/44 [00:00<00:00, 51.54it/s, best_latency=0.0914]                                                                                         Bench configurations:  14%|█▎        | 6/44 [00:00<00:00, 51.54it/s, best_latency=0.0914]Bench configurations:  14%|█▎        | 6/44 [00:00<00:00, 51.54it/s, best_latency=0.0914]                                                                                         Bench configurations:  14%|█▎        | 6/44 [00:00<00:00, 51.54it/s, best_latency=0.0914]Bench configurations:  14%|█▎        | 6/44 [00:00<00:00, 51.54it/s, best_latency=0.0914]                                                                                         Bench configurations:  14%|█▎        | 6/44 [00:00<00:00, 51.54it/s, best_latency=0.0914]Bench configurations:  27%|██▋       | 12/44 [00:00<00:01, 18.82it/s, best_latency=0.0914]Bench configurations:  27%|██▋       | 12/44 [00:01<00:01, 18.82it/s, best_latency=0.0135]                                                                                          Bench configurations:  27%|██▋       | 12/44 [00:01<00:01, 18.82it/s, best_latency=0.0135]Bench configurations:  27%|██▋       | 12/44 [00:01<00:01, 18.82it/s, best_latency=0.0127]                                                                                          Bench configurations:  27%|██▋       | 12/44 [00:01<00:01, 18.82it/s, best_latency=0.0127]Bench configurations:  27%|██▋       | 12/44 [00:01<00:01, 18.82it/s, best_latency=0.0127]                                                                                          Bench configurations:  27%|██▋       | 12/44 [00:01<00:01, 18.82it/s, best_latency=0.0127]Bench configurations:  34%|███▍      | 15/44 [00:01<00:04,  6.01it/s, best_latency=0.0127]Bench configurations:  34%|███▍      | 15/44 [00:02<00:04,  6.01it/s, best_latency=0.0127]                                                                                          Bench configurations:  34%|███▍      | 15/44 [00:02<00:04,  6.01it/s, best_latency=0.0127]Bench configurations:  34%|███▍      | 15/44 [00:02<00:04,  6.01it/s, best_latency=0.0127]                                                                                          Bench configurations:  34%|███▍      | 15/44 [00:02<00:04,  6.01it/s, best_latency=0.0127]Bench configurations:  39%|███▊      | 17/44 [00:02<00:06,  4.22it/s, best_latency=0.0127]Bench configurations:  39%|███▊      | 17/44 [00:03<00:06,  4.22it/s, best_latency=0.0127]                                                                                          Bench configurations:  39%|███▊      | 17/44 [00:03<00:06,  4.22it/s, best_latency=0.0127]Bench configurations:  39%|███▊      | 17/44 [00:03<00:06,  4.22it/s, best_latency=0.0127]                                                                                          Bench configurations:  39%|███▊      | 17/44 [00:03<00:06,  4.22it/s, best_latency=0.0127]Bench configurations:  43%|████▎     | 19/44 [00:03<00:07,  3.38it/s, best_latency=0.0127]Bench configurations:  43%|████▎     | 19/44 [00:04<00:07,  3.38it/s, best_latency=0.0127]                                                                                          Bench configurations:  43%|████▎     | 19/44 [00:04<00:07,  3.38it/s, best_latency=0.0127]Bench configurations:  45%|████▌     | 20/44 [00:04<00:07,  3.14it/s, best_latency=0.0127]Bench configurations:  45%|████▌     | 20/44 [00:04<00:07,  3.14it/s, best_latency=0.0127]                                                                                          Bench configurations:  45%|████▌     | 20/44 [00:04<00:07,  3.14it/s, best_latency=0.0127]Bench configurations:  48%|████▊     | 21/44 [00:04<00:07,  2.96it/s, best_latency=0.0127]Bench configurations:  48%|████▊     | 21/44 [00:05<00:07,  2.96it/s, best_latency=0.0127]                                                                                          Bench configurations:  48%|████▊     | 21/44 [00:05<00:07,  2.96it/s, best_latency=0.0127]Bench configurations:  50%|█████     | 22/44 [00:05<00:08,  2.65it/s, best_latency=0.0127]Bench configurations:  50%|█████     | 22/44 [00:05<00:08,  2.65it/s, best_latency=0.0127]                                                                                          Bench configurations:  50%|█████     | 22/44 [00:05<00:08,  2.65it/s, best_latency=0.0127]Bench configurations:  52%|█████▏    | 23/44 [00:05<00:08,  2.43it/s, best_latency=0.0127]Bench configurations:  52%|█████▏    | 23/44 [00:06<00:08,  2.43it/s, best_latency=0.0127]                                                                                          Bench configurations:  52%|█████▏    | 23/44 [00:06<00:08,  2.43it/s, best_latency=0.0127]Bench configurations:  55%|█████▍    | 24/44 [00:06<00:08,  2.37it/s, best_latency=0.0127]Bench configurations:  55%|█████▍    | 24/44 [00:06<00:08,  2.37it/s, best_latency=0.0127]                                                                                          Bench configurations:  55%|█████▍    | 24/44 [00:06<00:08,  2.37it/s, best_latency=0.0127]Bench configurations:  57%|█████▋    | 25/44 [00:06<00:08,  2.33it/s, best_latency=0.0127]Bench configurations:  57%|█████▋    | 25/44 [00:07<00:08,  2.33it/s, best_latency=0.0127]                                                                                          Bench configurations:  57%|█████▋    | 25/44 [00:07<00:08,  2.33it/s, best_latency=0.0127]Bench configurations:  59%|█████▉    | 26/44 [00:07<00:07,  2.30it/s, best_latency=0.0127]Bench configurations:  59%|█████▉    | 26/44 [00:07<00:07,  2.30it/s, best_latency=0.0127]                                                                                          Bench configurations:  59%|█████▉    | 26/44 [00:07<00:07,  2.30it/s, best_latency=0.0127]Bench configurations:  61%|██████▏   | 27/44 [00:07<00:07,  2.30it/s, best_latency=0.0127]Bench configurations:  61%|██████▏   | 27/44 [00:08<00:07,  2.30it/s, best_latency=0.0127]                                                                                          Bench configurations:  61%|██████▏   | 27/44 [00:08<00:07,  2.30it/s, best_latency=0.0127]Bench configurations:  64%|██████▎   | 28/44 [00:08<00:07,  2.27it/s, best_latency=0.0127]Bench configurations:  64%|██████▎   | 28/44 [00:08<00:07,  2.27it/s, best_latency=0.0127]                                                                                          Bench configurations:  64%|██████▎   | 28/44 [00:08<00:07,  2.27it/s, best_latency=0.0127]Bench configurations:  66%|██████▌   | 29/44 [00:08<00:06,  2.28it/s, best_latency=0.0127]Bench configurations:  66%|██████▌   | 29/44 [00:08<00:06,  2.28it/s, best_latency=0.0127]                                                                                          Bench configurations:  66%|██████▌   | 29/44 [00:08<00:06,  2.28it/s, best_latency=0.0127]Bench configurations:  68%|██████▊   | 30/44 [00:08<00:06,  2.29it/s, best_latency=0.0127]Bench configurations:  68%|██████▊   | 30/44 [00:09<00:06,  2.29it/s, best_latency=0.0127]                                                                                          Bench configurations:  68%|██████▊   | 30/44 [00:09<00:06,  2.29it/s, best_latency=0.0127]Bench configurations:  70%|███████   | 31/44 [00:09<00:05,  2.27it/s, best_latency=0.0127]Bench configurations:  70%|███████   | 31/44 [00:09<00:05,  2.27it/s, best_latency=0.0127]                                                                                          Bench configurations:  70%|███████   | 31/44 [00:09<00:05,  2.27it/s, best_latency=0.0127]Bench configurations:  73%|███████▎  | 32/44 [00:09<00:05,  2.25it/s, best_latency=0.0127]Bench configurations:  73%|███████▎  | 32/44 [00:10<00:05,  2.25it/s, best_latency=0.0127]                                                                                          Bench configurations:  73%|███████▎  | 32/44 [00:10<00:05,  2.25it/s, best_latency=0.0127]Bench configurations:  75%|███████▌  | 33/44 [00:10<00:05,  2.16it/s, best_latency=0.0127]Bench configurations:  75%|███████▌  | 33/44 [00:10<00:05,  2.16it/s, best_latency=0.0127]                                                                                          Bench configurations:  75%|███████▌  | 33/44 [00:10<00:05,  2.16it/s, best_latency=0.0127]Bench configurations:  77%|███████▋  | 34/44 [00:10<00:04,  2.12it/s, best_latency=0.0127]Bench configurations:  77%|███████▋  | 34/44 [00:11<00:04,  2.12it/s, best_latency=0.0127]                                                                                          Bench configurations:  77%|███████▋  | 34/44 [00:11<00:04,  2.12it/s, best_latency=0.0127]Bench configurations:  80%|███████▉  | 35/44 [00:11<00:04,  2.13it/s, best_latency=0.0127]Bench configurations:  80%|███████▉  | 35/44 [00:11<00:04,  2.13it/s, best_latency=0.0127]                                                                                          Bench configurations:  80%|███████▉  | 35/44 [00:11<00:04,  2.13it/s, best_latency=0.0127]Bench configurations:  82%|████████▏ | 36/44 [00:11<00:03,  2.11it/s, best_latency=0.0127]Bench configurations:  82%|████████▏ | 36/44 [00:12<00:03,  2.11it/s, best_latency=0.0127]                                                                                          Bench configurations:  82%|████████▏ | 36/44 [00:12<00:03,  2.11it/s, best_latency=0.0127]Bench configurations:  84%|████████▍ | 37/44 [00:12<00:03,  2.09it/s, best_latency=0.0127]Bench configurations:  84%|████████▍ | 37/44 [00:12<00:03,  2.09it/s, best_latency=0.0127]                                                                                          Bench configurations:  84%|████████▍ | 37/44 [00:12<00:03,  2.09it/s, best_latency=0.0127]Bench configurations:  86%|████████▋ | 38/44 [00:12<00:02,  2.08it/s, best_latency=0.0127]Bench configurations:  86%|████████▋ | 38/44 [00:13<00:02,  2.08it/s, best_latency=0.0127]                                                                                          Bench configurations:  86%|████████▋ | 38/44 [00:13<00:02,  2.08it/s, best_latency=0.0127]Bench configurations:  89%|████████▊ | 39/44 [00:13<00:02,  2.08it/s, best_latency=0.0127]Bench configurations:  89%|████████▊ | 39/44 [00:13<00:02,  2.08it/s, best_latency=0.0127]                                                                                          Bench configurations:  89%|████████▊ | 39/44 [00:13<00:02,  2.08it/s, best_latency=0.0127]Bench configurations:  91%|█████████ | 40/44 [00:13<00:01,  2.01it/s, best_latency=0.0127]Bench configurations:  91%|█████████ | 40/44 [00:14<00:01,  2.01it/s, best_latency=0.0127]                                                                                          Bench configurations:  91%|█████████ | 40/44 [00:14<00:01,  2.01it/s, best_latency=0.0127]Bench configurations:  93%|█████████▎| 41/44 [00:14<00:01,  1.97it/s, best_latency=0.0127]Bench configurations:  93%|█████████▎| 41/44 [00:14<00:01,  1.97it/s, best_latency=0.0127]                                                                                          Bench configurations:  93%|█████████▎| 41/44 [00:14<00:01,  1.97it/s, best_latency=0.0127]Bench configurations:  95%|█████████▌| 42/44 [00:14<00:01,  1.94it/s, best_latency=0.0127]Bench configurations:  95%|█████████▌| 42/44 [00:15<00:01,  1.94it/s, best_latency=0.0127]                                                                                          Bench configurations:  95%|█████████▌| 42/44 [00:15<00:01,  1.94it/s, best_latency=0.0127]Bench configurations:  98%|█████████▊| 43/44 [00:15<00:00,  1.91it/s, best_latency=0.0127]Bench configurations:  98%|█████████▊| 43/44 [00:15<00:00,  1.91it/s, best_latency=0.0127]                                                                                          Bench configurations:  98%|█████████▊| 43/44 [00:15<00:00,  1.91it/s, best_latency=0.0127]Bench configurations: 100%|██████████| 44/44 [00:15<00:00,  1.89it/s, best_latency=0.0127]Bench configurations: 100%|██████████| 44/44 [00:15<00:00,  2.77it/s, best_latency=0.0127]
=============================== warnings summary ===============================
benchmarks/ops/bench_activation.py: 93 warnings
benchmarks/ops/bench_ada_layer_norm.py: 8 warnings
benchmarks/ops/bench_argreduce.py: 6 warnings
benchmarks/ops/bench_batch_norm.py: 12 warnings
benchmarks/ops/bench_binary_arith.py: 65 warnings
benchmarks/ops/bench_binary_elementwise.py: 107 warnings
benchmarks/ops/bench_binary_strategy.py: 18 warnings
benchmarks/ops/bench_cumulative.py: 6 warnings
benchmarks/ops/bench_deepseek_dsa_decode.py: 2 warnings
benchmarks/ops/bench_deepseek_mla_decode.py: 2 warnings
benchmarks/ops/bench_deepseek_nsa_cmp_fwd.py: 2 warnings
benchmarks/ops/bench_deepseek_nsa_fwd.py: 3 warnings
benchmarks/ops/bench_deepseek_nsa_topk.py: 3 warnings
benchmarks/ops/bench_dropout.py: 9 warnings
benchmarks/ops/bench_elementwise_fp8.py: 24 warnings
benchmarks/ops/bench_engram_bwd.py: 4 warnings
benchmarks/ops/bench_engram_decode.py: 3 warnings
benchmarks/ops/bench_engram_fwd.py: 4 warnings
benchmarks/ops/bench_fft.py: 5 warnings
benchmarks/ops/bench_fft_lut.py: 14 warnings
benchmarks/ops/bench_fp8_lighting_indexer.py: 2 warnings
benchmarks/ops/bench_fp8_quant.py: 4 warnings
benchmarks/ops/bench_fused_add_layer_norm.py: 4 warnings
benchmarks/ops/bench_fused_add_rmsnorm.py: 4 warnings
benchmarks/ops/bench_gated_deltanet_chunkwise.py: 32 warnings
benchmarks/ops/bench_gated_deltanet_recurrence.py: 13 warnings
benchmarks/ops/bench_gemm.py: 5 warnings
benchmarks/ops/bench_gla_chunkwise.py: 24 warnings
benchmarks/ops/bench_gla_recurrence.py: 16 warnings
benchmarks/ops/bench_gqa.py: 6 warnings
benchmarks/ops/bench_gqa_decode.py: 3 warnings
benchmarks/ops/bench_gqa_decode_paged.py: 4 warnings
benchmarks/ops/bench_gqa_sliding_window_fwd.py: 5 warnings
benchmarks/ops/bench_gqa_sliding_window_varlen_fwd.py: 5 warnings
benchmarks/ops/bench_group_norm.py: 4 warnings
benchmarks/ops/bench_grouped_gemm.py: 8 warnings
benchmarks/ops/bench_independent_elementwise.py: 132 warnings
benchmarks/ops/bench_instance_norm.py: 4 warnings
benchmarks/ops/bench_layer_norm.py: 4 warnings
benchmarks/ops/bench_logical_reduce.py: 15 warnings
benchmarks/ops/bench_mean_pooling.py: 4 warnings
benchmarks/ops/bench_mha.py: 6 warnings
benchmarks/ops/bench_mha_decode.py: 1 warning
benchmarks/ops/bench_mha_decode_paged.py: 4 warnings
benchmarks/ops/bench_mhc_post.py: 3 warnings
benchmarks/ops/bench_mhc_pre.py: 3 warnings
benchmarks/ops/bench_moe_fused_topk.py: 9 warnings
benchmarks/ops/bench_moe_permute.py: 9 warnings
benchmarks/ops/bench_moe_permute_align.py: 10 warnings
benchmarks/ops/bench_moe_unpermute.py: 9 warnings
benchmarks/ops/bench_reduce.py: 10 warnings
benchmarks/ops/bench_rms_norm.py: 4 warnings
benchmarks/ops/bench_rope.py: 9 warnings
benchmarks/ops/bench_softmax.py: 12 warnings
benchmarks/ops/bench_topk_selector.py: 4 warnings
benchmarks/ops/bench_unary_elementwise.py: 21 warnings
benchmarks/ops/bench_unary_strategy.py: 27 warnings
benchmarks/ops/bench_vector_norm.py: 12 warnings
  /home/ci-runner/workdir/_tool/tileops_benchmarks_venv_db4a9af2f48857c5/lib/python3.11/site-packages/torch/profiler/profiler.py:509: UserWarning: Profiler won't be using warmup, this can skew profiler results
    warn("Profiler won't be using warmup, this can skew profiler results")

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED benchmarks/ops/bench_mha_decode.py::test_mha_decode_bench[fp16-long-cache] - RuntimeError: kernel mha_decode_split input Output_partial ndim expected 5, but got 4
FAILED benchmarks/ops/bench_mha_decode.py::test_mha_decode_bench[bf16-long-cache] - RuntimeError: kernel mha_decode_split input Output_partial ndim expected 5, but got 4
2 failed, 830 passed, 841 warnings in 3035.78s (0:50:35)

===== profile_run.log summary =====
# TileOPs Benchmark Report
Generated: 2026-03-23 20:04:18

## Environment

- **Torch version**: 2.9.1+cu128
- **CUDA version (torch)**: 12.8
- **GPU model**: NVIDIA H200
- **Driver version**: 575.57.08

## ReluOp

### tileops

| n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 4096 | torch.float16 | 0.00 | 0.00 | 0.01 |
| 4194304 | torch.float16 | 0.01 | 0.65 | 2.61 |
| 4194304 | torch.bfloat16 | 0.01 | 0.65 | 2.61 |
| 4194304 | torch.float32 | 0.01 | 0.39 | 3.12 |

### baseline

| n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 4096 | torch.float16 | 0.00 | 0.00 | 0.01 |
| 4194304 | torch.float16 | 0.01 | 0.63 | 2.51 |
| 4194304 | torch.bfloat16 | 0.01 | 0.64 | 2.56 |
| 4194304 | torch.float32 | 0.01 | 0.39 | 3.12 |

## r3_jit_cost

### relu_jit

| n_total | cold_ms | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 1000 | 21.98 | 0.00 | 0.00 | 0.00 |
| 2000 | 24.43 | 0.00 | 0.00 | 0.00 |
| 4000 | 23.36 | 0.00 | 0.00 | 0.01 |
| 8000 | 20.99 | 0.00 | 0.00 | 0.02 |
| 16000 | 23.41 | 0.00 | 0.01 | 0.03 |
| 32000 | 21.51 | 0.00 | 0.02 | 0.06 |
| 64000 | 22.25 | 0.00 | 0.03 | 0.12 |
| 128000 | 20.59 | 0.00 | 0.06 | 0.23 |
| 256000 | 21.29 | 0.00 | 0.11 | 0.45 |
| 512000 | 20.82 | 0.00 | 0.19 | 0.77 |

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
| 4194304 | 4M | torch.float16 | explicit_parallel | 0.01 | 0.60 | 2.40 |
| 4194304 | 4M | torch.bfloat16 | explicit_parallel | 0.01 | 0.60 | 2.41 |
| 4194304 | 4M | torch.float32 | explicit_parallel | 0.01 | 0.38 | 3.03 |
| 16777216 | 16M | torch.float16 | explicit_parallel | 0.02 | 0.79 | 3.18 |
| 16777216 | 16M | torch.bfloat16 | explicit_parallel | 0.02 | 0.79 | 3.18 |
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
| 16777216 | 16M | torch.float16 | register_copy | 0.02 | 0.91 | 3.63 |
| 16777216 | 16M | torch.bfloat16 | register_copy | 0.02 | 0.91 | 3.62 |
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
| 4194304 | 4M | torch.float16 | explicit_parallel | 0.01 | 0.44 | 1.76 |
| 4194304 | 4M | torch.bfloat16 | explicit_parallel | 0.01 | 0.44 | 1.74 |
| 4194304 | 4M | torch.float32 | explicit_parallel | 0.01 | 0.38 | 3.03 |
| 16777216 | 16M | torch.float16 | explicit_parallel | 0.03 | 0.53 | 2.12 |
| 16777216 | 16M | torch.bfloat16 | explicit_parallel | 0.03 | 0.52 | 2.10 |
| 16777216 | 16M | torch.float32 | explicit_parallel | 0.04 | 0.46 | 3.67 |

### gelu_register_copy

| n_total | size_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | torch.float16 | register_copy | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.bfloat16 | register_copy | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.float32 | register_copy | 0.00 | 0.00 | 0.02 |
| 4194304 | 4M | torch.float16 | register_copy | 0.01 | 0.47 | 1.90 |
| 4194304 | 4M | torch.bfloat16 | register_copy | 0.01 | 0.46 | 1.86 |
| 4194304 | 4M | torch.float32 | register_copy | 0.01 | 0.39 | 3.10 |
| 16777216 | 16M | torch.float16 | register_copy | 0.03 | 0.60 | 2.41 |
| 16777216 | 16M | torch.bfloat16 | register_copy | 0.03 | 0.58 | 2.32 |
| 16777216 | 16M | torch.float32 | register_copy | 0.04 | 0.47 | 3.78 |

## r5_boundary

### relu_aligned

| n_total | align_label | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 2048000 | aligned | 0.00 | 0.44 | 1.77 |

### relu_unaligned_plus_1

| n_total | align_label | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 2048001 | unaligned_plus_1 | 0.01 | 0.29 | 1.16 |

### relu_unaligned_plus_127

| n_total | align_label | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 2048127 | unaligned_plus_127 | 0.01 | 0.29 | 1.15 |

## r6_threads

### relu_t128

| n_total | size_label | op_name | threads | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | relu | 128 | 0.00 | 0.00 | 0.01 |
| 4194304 | 4M | relu | 128 | 0.02 | 0.21 | 0.84 |
| 16777216 | 16M | relu | 128 | 0.07 | 0.26 | 1.02 |

### relu_t256

| n_total | size_label | op_name | threads | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | relu | 256 | 0.00 | 0.00 | 0.01 |
| 4194304 | 4M | relu | 256 | 0.02 | 0.22 | 0.86 |
| 16777216 | 16M | relu | 256 | 0.07 | 0.24 | 0.98 |

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
| 4194304 | 4M | erf | 256 | 0.01 | 0.48 | 1.91 |
| 16777216 | 16M | erf | 256 | 0.03 | 0.60 | 2.40 |

### mish_t128

| n_total | size_label | op_name | threads | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | mish | 128 | 0.00 | 0.00 | 0.01 |
| 4194304 | 4M | mish | 128 | 0.01 | 0.36 | 1.44 |
| 16777216 | 16M | mish | 128 | 0.04 | 0.42 | 1.69 |

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
| fp16 | 4 | 0.00 | 0.28 | 1.11 |

### relu_fp16_npt8

| dtype_label | num_per_thread | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| fp16 | 8 | 0.00 | 0.22 | 0.89 |

## AdaLayerNormOp

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.02 | 1.34 | 2.14 |
| 4096 | 4096 | torch.bfloat16 | 0.05 | 1.60 | 2.56 |
| 1024 | 3000 | torch.float16 | 0.04 | 0.38 | 0.61 |
| 1025 | 4096 | torch.float16 | 0.02 | 1.34 | 2.14 |

### baseline

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.02 | 0.85 | 1.36 |
| 4096 | 4096 | torch.bfloat16 | 0.08 | 1.05 | 1.68 |
| 1024 | 3000 | torch.float16 | 0.02 | 0.75 | 1.21 |
| 1025 | 4096 | torch.float16 | 0.03 | 0.84 | 1.34 |

## AdaLayerNormZeroOp

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.02 | 1.29 | 2.15 |
| 4096 | 4096 | torch.bfloat16 | 0.06 | 1.57 | 2.62 |
| 1024 | 3000 | torch.float16 | 0.05 | 0.35 | 0.58 |
| 1025 | 4096 | torch.float16 | 0.02 | 1.29 | 2.15 |

### baseline

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.03 | 0.80 | 1.33 |
| 4096 | 4096 | torch.bfloat16 | 0.11 | 0.96 | 1.59 |
| 1024 | 3000 | torch.float16 | 0.03 | 0.72 | 1.20 |
| 1025 | 4096 | torch.float16 | 0.03 | 0.80 | 1.33 |

## ArgmaxOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | argmax | 3.07 | 0.00 | 0.00 |
| 1024 | 4096 | torch.bfloat16 | argmax | 3.08 | 0.00 | 0.00 |
| 4096 | 4096 | torch.float16 | argmax | 10.67 | 0.00 | 0.00 |

### baseline

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | argmax | 0.02 | 0.19 | 0.37 |
| 1024 | 4096 | torch.bfloat16 | argmax | 0.02 | 0.19 | 0.37 |
| 4096 | 4096 | torch.float16 | argmax | 0.03 | 0.59 | 1.18 |

## ArgminOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | argmin | 3.47 | 0.00 | 0.00 |
| 1024 | 4096 | torch.bfloat16 | argmin | 3.48 | 0.00 | 0.00 |
| 4096 | 4096 | torch.float16 | argmin | 12.12 | 0.00 | 0.00 |

### baseline

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | argmin | 0.02 | 0.19 | 0.37 |
| 1024 | 4096 | torch.bfloat16 | argmin | 0.02 | 0.19 | 0.37 |
| 4096 | 4096 | torch.float16 | argmin | 0.03 | 0.59 | 1.18 |

## BatchNormFwdOp

### tileops

| N | C | spatial | dtype | training | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 64 | () | torch.float16 | True | 0.01 | 0.00 | 0.00 |
| 8 | 64 | (32, 32) | torch.float16 | True | 0.01 | 0.42 | 0.17 |
| 4 | 128 | (32, 32) | torch.float16 | True | 0.01 | 0.43 | 0.17 |
| 4 | 256 | (28, 28) | torch.float16 | True | 0.02 | 0.50 | 0.20 |
| 4 | 128 | (1024, 1024) | torch.float16 | True | 3.94 | 1.36 | 0.55 |
| 4 | 256 | (1024, 1024) | torch.float16 | True | 7.33 | 1.46 | 0.59 |

### torch_cudnn

| N | C | spatial | dtype | training | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 64 | () | torch.float16 | True | 0.02 | 0.00 | 0.00 |
| 8 | 64 | (32, 32) | torch.float16 | True | 0.01 | 0.37 | 0.15 |
| 4 | 128 | (32, 32) | torch.float16 | True | 0.01 | 0.39 | 0.16 |
| 4 | 256 | (28, 28) | torch.float16 | True | 0.01 | 0.56 | 0.23 |
| 4 | 128 | (1024, 1024) | torch.float16 | True | 4.12 | 1.30 | 0.52 |
| 4 | 256 | (1024, 1024) | torch.float16 | True | 7.28 | 1.47 | 0.59 |

## BatchNormBwdOp

### tileops

| N | C | spatial | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 64 | () | torch.float16 | 0.01 | 0.00 | 0.00 |
| 8 | 64 | (32, 32) | torch.float16 | 0.02 | 0.26 | 0.19 |
| 4 | 128 | (32, 32) | torch.float16 | 0.02 | 0.26 | 0.20 |
| 4 | 256 | (28, 28) | torch.float16 | 0.02 | 0.32 | 0.24 |
| 4 | 128 | (1024, 1024) | torch.float16 | 6.23 | 0.69 | 0.52 |
| 4 | 256 | (1024, 1024) | torch.float16 | 11.28 | 0.76 | 0.57 |

### torch_autograd

| N | C | spatial | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 64 | () | torch.float16 | 0.02 | 0.00 | 0.00 |
| 8 | 64 | (32, 32) | torch.float16 | 0.03 | 0.16 | 0.12 |
| 4 | 128 | (32, 32) | torch.float16 | 0.02 | 0.19 | 0.14 |
| 4 | 256 | (28, 28) | torch.float16 | 0.02 | 0.27 | 0.20 |
| 4 | 128 | (1024, 1024) | torch.float16 | 12.07 | 0.36 | 0.27 |
| 4 | 256 | (1024, 1024) | torch.float16 | 18.16 | 0.47 | 0.35 |

## r1_vectorization

### add_same_shape_1d

| pattern_name | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| same_shape_1d | 1000000 | 0.00 | 0.24 | 1.44 |

### baseline_same_shape_1d

| pattern_name | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| same_shape_1d | 1000000 | 0.00 | 0.25 | 1.49 |

### add_bias_add_2d

| pattern_name | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| bias_add_2d | 1000000 | 0.00 | 0.27 | 1.06 |

### baseline_bias_add_2d

| pattern_name | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| bias_add_2d | 1000000 | 0.01 | 0.17 | 0.67 |

### add_interleaved_3d

| pattern_name | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| interleaved_3d | 1048576 | 0.00 | 0.41 | 0.83 |

### baseline_interleaved_3d

| pattern_name | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| interleaved_3d | 1048576 | 0.01 | 0.19 | 0.37 |

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
| 1M | same_shape | direct | 1048576 | 0.01 | 0.18 | 1.10 |
| 1M | same_shape | direct | 1048576 | 0.01 | 0.18 | 1.10 |
| 1M | same_shape | direct | 1048576 | 0.01 | 0.16 | 1.96 |
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
| 1M | interleaved_3d | direct | 1048576 | 0.00 | 0.23 | 0.46 |
| 1M | interleaved_3d | direct | 1048576 | 0.00 | 0.23 | 0.47 |
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
| 1M | same_shape | explicit_parallel | 1048576 | 0.00 | 0.26 | 1.56 |
| 1M | same_shape | explicit_parallel | 1048576 | 0.00 | 0.26 | 1.57 |
| 1M | same_shape | explicit_parallel | 1048576 | 0.01 | 0.19 | 2.24 |
| 16M | same_shape | explicit_parallel | 16777216 | 0.03 | 0.59 | 3.55 |
| 16M | same_shape | explicit_parallel | 16777216 | 0.03 | 0.59 | 3.55 |
| 16M | same_shape | explicit_parallel | 16777216 | 0.05 | 0.33 | 4.01 |

### add_explicit_parallel_bias_add_2d

| size_label | pattern_name | strategy | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4K | bias_add_2d | explicit_parallel | 4096 | 0.00 | 0.00 | 0.01 |
| 4K | bias_add_2d | explicit_parallel | 4096 | 0.00 | 0.00 | 0.01 |
| 4K | bias_add_2d | explicit_parallel | 4096 | 0.00 | 0.00 | 0.02 |
| 1M | bias_add_2d | explicit_parallel | 1048576 | 0.00 | 0.31 | 1.24 |
| 1M | bias_add_2d | explicit_parallel | 1048576 | 0.00 | 0.31 | 1.25 |
| 1M | bias_add_2d | explicit_parallel | 1048576 | 0.00 | 0.24 | 1.90 |
| 16M | bias_add_2d | explicit_parallel | 16777216 | 0.02 | 0.81 | 3.23 |
| 16M | bias_add_2d | explicit_parallel | 16777216 | 0.02 | 0.81 | 3.23 |
| 16M | bias_add_2d | explicit_parallel | 16777216 | 0.03 | 0.49 | 3.91 |

### add_explicit_parallel_interleaved_3d

| size_label | pattern_name | strategy | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4K | interleaved_3d | explicit_parallel | 4096 | 0.00 | 0.00 | 0.01 |
| 4K | interleaved_3d | explicit_parallel | 4096 | 0.00 | 0.00 | 0.01 |
| 4K | interleaved_3d | explicit_parallel | 4096 | 0.00 | 0.00 | 0.01 |
| 1M | interleaved_3d | explicit_parallel | 1048576 | 0.00 | 0.41 | 0.83 |
| 1M | interleaved_3d | explicit_parallel | 1048576 | 0.00 | 0.41 | 0.83 |
| 1M | interleaved_3d | explicit_parallel | 1048576 | 0.00 | 0.38 | 1.54 |
| 16M | interleaved_3d | explicit_parallel | 16777216 | 0.01 | 1.17 | 2.35 |
| 16M | interleaved_3d | explicit_parallel | 16777216 | 0.01 | 1.17 | 2.35 |
| 16M | interleaved_3d | explicit_parallel | 16777216 | 0.02 | 0.95 | 3.82 |

## r4_where

### tileops_where

| n_total | size_label | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 4096 | 4K | 0.00 | 0.00 | 0.01 |
| 1048576 | 1M | 0.00 | 0.24 | 1.68 |
| 16777216 | 16M | 0.03 | 0.53 | 3.70 |

### baseline_where

| n_total | size_label | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 4096 | 4K | 0.00 | 0.00 | 0.01 |
| 1048576 | 1M | 0.00 | 0.24 | 1.65 |
| 16777216 | 16M | 0.03 | 0.53 | 3.74 |

## AddOp

### tileops

| n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 4194304 | torch.float16 | 0.01 | 0.47 | 2.83 |
| 4194304 | torch.bfloat16 | 0.01 | 0.47 | 2.83 |
| 4194304 | torch.float32 | 0.02 | 0.27 | 3.28 |

### baseline

| n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 4194304 | torch.float16 | 0.01 | 0.46 | 2.79 |
| 4194304 | torch.bfloat16 | 0.01 | 0.46 | 2.79 |
| 4194304 | torch.float32 | 0.02 | 0.28 | 3.30 |

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
| div | torch.float16 | torch.float16 | 0.01 | 0.44 | 2.66 |
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
| lerp | torch.float16 | torch.float16 | 0.01 | 0.47 | 2.83 |
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
| maximum | torch.float16 | torch.float16 | 0.02 | 0.57 | 3.39 |
| maximum | torch.float16 | torch.float16 | 0.03 | 0.64 | 3.81 |

### baseline

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| maximum | torch.float16 | torch.float16 | 0.01 | 0.46 | 2.76 |
| maximum | torch.float16 | torch.float16 | 0.02 | 0.56 | 3.39 |
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
| minimum | torch.float16 | torch.float16 | 0.02 | 0.56 | 3.39 |
| minimum | torch.float16 | torch.float16 | 0.03 | 0.63 | 3.81 |

## cmp_eq

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| eq | torch.float16 | 0.02 | 0.22 | 1.11 |
| eq | torch.float16 | 0.04 | 0.26 | 1.32 |

### baseline

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| eq | torch.float16 | 0.01 | 0.52 | 2.59 |
| eq | torch.float16 | 0.02 | 0.65 | 3.23 |

## cmp_ne

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| ne | torch.float16 | 0.02 | 0.22 | 1.11 |

### baseline

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| ne | torch.float16 | 0.01 | 0.52 | 2.59 |

## cmp_gt

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| gt | torch.float16 | 0.02 | 0.22 | 1.11 |

### baseline

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| gt | torch.float16 | 0.01 | 0.51 | 2.57 |

## cmp_lt

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| lt | torch.float16 | 0.02 | 0.22 | 1.11 |

### baseline

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| lt | torch.float16 | 0.01 | 0.51 | 2.56 |

## cmp_ge

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| ge | torch.float16 | 0.02 | 0.22 | 1.11 |

### baseline

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| ge | torch.float16 | 0.01 | 0.52 | 2.58 |

## cmp_le

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| le | torch.float16 | 0.02 | 0.22 | 1.11 |

### baseline

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| le | torch.float16 | 0.01 | 0.51 | 2.56 |

## logical_and

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| logical_and | torch.float16 | 0.02 | 0.22 | 1.11 |
| logical_and | torch.float16 | 0.04 | 0.26 | 1.32 |

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
| logical_or | torch.float16 | 0.04 | 0.26 | 1.32 |

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
| bitwise_and | torch.int32 | 0.02 | 0.28 | 3.30 |
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
| bitwise_xor | torch.int32 | 0.02 | 0.28 | 3.30 |

## gelu_and_mul

### tileops

| op_name | M | N | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| gelu_and_mul | 1024 | 4096 | torch.float16 | 0.01 | 0.77 | 2.30 |
| gelu_and_mul | 1024 | 10240 | torch.float16 | 0.02 | 0.89 | 2.66 |
| gelu_and_mul | 1024 | 20480 | torch.float16 | 0.04 | 0.96 | 2.88 |

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
| gelu_tanh_and_mul | 1024 | 4096 | torch.float16 | 0.01 | 0.89 | 2.68 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.float16 | 0.02 | 1.06 | 3.18 |
| gelu_tanh_and_mul | 1024 | 20480 | torch.float16 | 0.04 | 1.16 | 3.48 |

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
| silu_and_mul | 1024 | 4096 | torch.float16 | direct | 0.02 | 0.49 | 1.47 |
| silu_and_mul | 1024 | 4096 | torch.bfloat16 | direct | 0.02 | 0.49 | 1.47 |
| silu_and_mul | 1024 | 4096 | torch.float32 | direct | 0.02 | 0.42 | 2.55 |
| silu_and_mul | 1024 | 10240 | torch.float16 | direct | 0.04 | 0.53 | 1.60 |
| silu_and_mul | 1024 | 10240 | torch.bfloat16 | direct | 0.04 | 0.53 | 1.60 |
| silu_and_mul | 1024 | 10240 | torch.float32 | direct | 0.04 | 0.47 | 2.83 |
| silu_and_mul | 4096 | 4096 | torch.float16 | direct | 0.06 | 0.55 | 1.66 |
| silu_and_mul | 4096 | 4096 | torch.bfloat16 | direct | 0.06 | 0.55 | 1.66 |
| silu_and_mul | 4096 | 4096 | torch.float32 | direct | 0.07 | 0.49 | 2.94 |

### tileops_explicit_parallel

| op_name | M | N | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| silu_and_mul | 1024 | 4096 | torch.float16 | explicit_parallel | 0.01 | 0.89 | 2.68 |
| silu_and_mul | 1024 | 4096 | torch.bfloat16 | explicit_parallel | 0.01 | 0.89 | 2.66 |
| silu_and_mul | 1024 | 4096 | torch.float32 | explicit_parallel | 0.02 | 0.55 | 3.28 |
| silu_and_mul | 1024 | 10240 | torch.float16 | explicit_parallel | 0.02 | 1.06 | 3.18 |
| silu_and_mul | 1024 | 10240 | torch.bfloat16 | explicit_parallel | 0.02 | 1.04 | 3.13 |
| silu_and_mul | 1024 | 10240 | torch.float32 | explicit_parallel | 0.03 | 0.63 | 3.80 |
| silu_and_mul | 4096 | 4096 | torch.float16 | explicit_parallel | 0.03 | 1.13 | 3.38 |
| silu_and_mul | 4096 | 4096 | torch.bfloat16 | explicit_parallel | 0.03 | 1.11 | 3.33 |
| silu_and_mul | 4096 | 4096 | torch.float32 | explicit_parallel | 0.05 | 0.67 | 4.00 |

## gelu_and_mul_strategy

### tileops_direct

| op_name | M | N | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gelu_and_mul | 1024 | 4096 | torch.float16 | direct | 0.02 | 0.48 | 1.45 |
| gelu_and_mul | 1024 | 4096 | torch.bfloat16 | direct | 0.02 | 0.48 | 1.45 |
| gelu_and_mul | 1024 | 4096 | torch.float32 | direct | 0.02 | 0.43 | 2.56 |
| gelu_and_mul | 1024 | 10240 | torch.float16 | direct | 0.04 | 0.53 | 1.58 |
| gelu_and_mul | 1024 | 10240 | torch.bfloat16 | direct | 0.04 | 0.53 | 1.58 |
| gelu_and_mul | 1024 | 10240 | torch.float32 | direct | 0.04 | 0.47 | 2.84 |
| gelu_and_mul | 4096 | 4096 | torch.float16 | direct | 0.06 | 0.54 | 1.63 |
| gelu_and_mul | 4096 | 4096 | torch.bfloat16 | direct | 0.06 | 0.54 | 1.63 |
| gelu_and_mul | 4096 | 4096 | torch.float32 | direct | 0.07 | 0.49 | 2.95 |

### tileops_explicit_parallel

| op_name | M | N | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gelu_and_mul | 1024 | 4096 | torch.float16 | explicit_parallel | 0.01 | 0.77 | 2.30 |
| gelu_and_mul | 1024 | 4096 | torch.bfloat16 | explicit_parallel | 0.01 | 0.77 | 2.30 |
| gelu_and_mul | 1024 | 4096 | torch.float32 | explicit_parallel | 0.02 | 0.55 | 3.28 |
| gelu_and_mul | 1024 | 10240 | torch.float16 | explicit_parallel | 0.02 | 0.89 | 2.66 |
| gelu_and_mul | 1024 | 10240 | torch.bfloat16 | explicit_parallel | 0.02 | 0.88 | 2.63 |
| gelu_and_mul | 1024 | 10240 | torch.float32 | explicit_parallel | 0.03 | 0.62 | 3.73 |
| gelu_and_mul | 4096 | 4096 | torch.float16 | explicit_parallel | 0.03 | 0.98 | 2.93 |
| gelu_and_mul | 4096 | 4096 | torch.bfloat16 | explicit_parallel | 0.03 | 0.98 | 2.93 |
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
| gelu_tanh_and_mul | 4096 | 4096 | torch.bfloat16 | direct | 0.06 | 0.55 | 1.66 |
| gelu_tanh_and_mul | 4096 | 4096 | torch.float32 | direct | 0.07 | 0.50 | 2.99 |

### tileops_explicit_parallel

| op_name | M | N | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gelu_tanh_and_mul | 1024 | 4096 | torch.float16 | explicit_parallel | 0.01 | 0.89 | 2.68 |
| gelu_tanh_and_mul | 1024 | 4096 | torch.bfloat16 | explicit_parallel | 0.01 | 0.89 | 2.67 |
| gelu_tanh_and_mul | 1024 | 4096 | torch.float32 | explicit_parallel | 0.02 | 0.55 | 3.30 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.float16 | explicit_parallel | 0.02 | 1.06 | 3.18 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.bfloat16 | explicit_parallel | 0.02 | 1.05 | 3.16 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.float32 | explicit_parallel | 0.03 | 0.63 | 3.81 |
| gelu_tanh_and_mul | 4096 | 4096 | torch.float16 | explicit_parallel | 0.03 | 1.12 | 3.36 |
| gelu_tanh_and_mul | 4096 | 4096 | torch.bfloat16 | explicit_parallel | 0.03 | 1.12 | 3.35 |
| gelu_tanh_and_mul | 4096 | 4096 | torch.float32 | explicit_parallel | 0.05 | 0.67 | 4.01 |

## sub_bcast

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| sub | torch.float16 | 0.01 | 0.61 | 2.45 |
| sub | torch.float16 | 0.01 | 0.75 | 3.01 |
| sub | torch.float16 | 0.03 | 0.83 | 3.31 |

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
| mul | torch.float16 | 0.01 | 0.61 | 2.45 |
| mul | torch.float16 | 0.01 | 0.75 | 3.01 |
| mul | torch.float16 | 0.03 | 0.83 | 3.31 |

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
| div | torch.float16 | 0.01 | 0.57 | 2.30 |
| div | torch.float16 | 0.01 | 0.72 | 2.88 |
| div | torch.float16 | 0.03 | 0.79 | 3.14 |

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
| 4194304 | 1024x4096 | torch.float32 | direct | 0.02 | 0.22 | 2.58 |
| 10485760 | 1024x10240 | torch.float16 | direct | 0.04 | 0.28 | 1.68 |
| 10485760 | 1024x10240 | torch.bfloat16 | direct | 0.04 | 0.28 | 1.68 |
| 10485760 | 1024x10240 | torch.float32 | direct | 0.04 | 0.24 | 2.91 |
| 20971520 | 1024x20480 | torch.float16 | direct | 0.07 | 0.30 | 1.78 |
| 20971520 | 1024x20480 | torch.bfloat16 | direct | 0.07 | 0.30 | 1.78 |
| 20971520 | 1024x20480 | torch.float32 | direct | 0.08 | 0.26 | 3.07 |

### add_explicit_parallel

| n_total | shape_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4194304 | 1024x4096 | torch.float16 | explicit_parallel | 0.01 | 0.46 | 2.75 |
| 4194304 | 1024x4096 | torch.bfloat16 | explicit_parallel | 0.01 | 0.46 | 2.74 |
| 4194304 | 1024x4096 | torch.float32 | explicit_parallel | 0.02 | 0.27 | 3.28 |
| 10485760 | 1024x10240 | torch.float16 | explicit_parallel | 0.02 | 0.55 | 3.29 |
| 10485760 | 1024x10240 | torch.bfloat16 | explicit_parallel | 0.02 | 0.55 | 3.29 |
| 10485760 | 1024x10240 | torch.float32 | explicit_parallel | 0.03 | 0.32 | 3.83 |
| 20971520 | 1024x20480 | torch.float16 | explicit_parallel | 0.03 | 0.61 | 3.67 |
| 20971520 | 1024x20480 | torch.bfloat16 | explicit_parallel | 0.03 | 0.61 | 3.67 |
| 20971520 | 1024x20480 | torch.float32 | explicit_parallel | 0.06 | 0.34 | 4.09 |

## CumsumOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | cumsum | 0.06 | 0.07 | 0.28 |
| 1024 | 4096 | torch.bfloat16 | cumsum | 0.06 | 0.07 | 0.28 |
| 4096 | 4096 | torch.float16 | cumsum | 0.12 | 0.14 | 0.56 |

### baseline

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | cumsum | 0.10 | 0.04 | 0.17 |
| 1024 | 4096 | torch.bfloat16 | cumsum | 0.10 | 0.04 | 0.17 |
| 4096 | 4096 | torch.float16 | cumsum | 0.26 | 0.07 | 0.26 |

## CumprodOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | cumprod | 0.06 | 0.07 | 0.28 |
| 1024 | 4096 | torch.bfloat16 | cumprod | 0.06 | 0.07 | 0.28 |
| 4096 | 4096 | torch.float16 | cumprod | 0.12 | 0.14 | 0.56 |

### baseline

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | cumprod | 0.10 | 0.04 | 0.17 |
| 1024 | 4096 | torch.bfloat16 | cumprod | 0.10 | 0.04 | 0.17 |
| 4096 | 4096 | torch.float16 | cumprod | 0.26 | 0.07 | 0.26 |

## DeepSeekSparseAttentionDecodeWithKVCacheOp

### tileops

| batch | heads | seq_len_q | seq_len_kv | dim | dim_tail | topk | stride_kv | heads_kv | q_start_index_s | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 128 | 1024 | 2048 | 512 | 64 | 2048 | 1 | 1 | 1024 | torch.float16 | 1.16 | 501.85 | 0.25 |
| 1 | 128 | 512 | 4096 | 512 | 64 | 1024 | 1 | 1 | 512 | torch.float16 | 0.31 | 468.80 | 0.48 |

### baseline

| batch | heads | seq_len_q | seq_len_kv | dim | dim_tail | topk | stride_kv | heads_kv | q_start_index_s | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 128 | 1024 | 2048 | 512 | 64 | 2048 | 1 | 1 | 1024 | torch.float16 | 16.50 | 35.40 | 0.02 |
| 1 | 128 | 512 | 4096 | 512 | 64 | 1024 | 1 | 1 | 512 | torch.float16 | 16.72 | 8.73 | 0.01 |

## MultiHeadLatentAttentionDecodeWithKVCacheOp

### tileops

| batch | heads | heads_kv | seq_len_kv | dim | dim_pe | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 128 | 1 | 8192 | 512 | 64 | torch.float16 | 0.17 | 438.10 | 1.84 |
| 16 | 128 | 1 | 4096 | 512 | 64 | torch.float16 | 0.05 | 373.99 | 1.60 |

### baseline

| batch | heads | heads_kv | seq_len_kv | dim | dim_pe | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 128 | 1 | 8192 | 512 | 64 | torch.float16 | 0.72 | 101.13 | 0.42 |
| 16 | 128 | 1 | 4096 | 512 | 64 | torch.float16 | 0.19 | 95.71 | 0.41 |

## NSACmpFwdVarlenOp

### tileops

| seq_num | c_seq_len | heads | dim_k | dim_v | group | scale | bc | bs | bk | bv | dtype | accum_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 9 | 8192 | 32 | 128 | 128 | 16 | 0.08838834764831845 | 32 | 32 | 128 | 128 | torch.float16 | torch.float32 | 0.32 | 54.28 | 7.00 |
| 16 | 16384 | 32 | 128 | 128 | 16 | 0.08838834764831845 | 32 | 32 | 128 | 128 | torch.float16 | torch.float32 | 0.35 | 198.13 | 25.15 |

### baseline

| seq_num | c_seq_len | heads | dim_k | dim_v | group | scale | bc | bs | bk | bv | dtype | accum_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 9 | 8192 | 32 | 128 | 128 | 16 | 0.08838834764831845 | 32 | 32 | 128 | 128 | torch.float16 | torch.float32 | 305.05 | 0.06 | 0.01 |
| 16 | 16384 | 32 | 128 | 128 | 16 | 0.08838834764831845 | 32 | 32 | 128 | 128 | torch.float16 | torch.float32 | 591.81 | 0.12 | 0.01 |

## NSAFwdVarlenOp

### tileops

| batch | heads | c_seq_len | dim | is_causal | scale | block_size | groups | selected_blocks | dtype | accum_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 16 | 1024 | 64 | True | 0.1 | 32 | 16 | 1 | torch.float16 | torch.float32 | 0.01 | 14.97 | 0.50 |
| 4 | 16 | 8192 | 64 | True | 0.1 | 32 | 16 | 1 | torch.float16 | torch.float32 | 0.04 | 28.09 | 0.93 |
| 2 | 16 | 8192 | 64 | True | 0.1 | 32 | 16 | 4 | torch.float16 | torch.float32 | 0.06 | 77.35 | 0.64 |

### baseline

| batch | heads | c_seq_len | dim | is_causal | scale | block_size | groups | selected_blocks | dtype | accum_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 16 | 1024 | 64 | True | 0.1 | 32 | 16 | 1 | torch.float16 | torch.float32 | 64.82 | 0.00 | 0.00 |
| 4 | 16 | 8192 | 64 | True | 0.1 | 32 | 16 | 1 | torch.float16 | torch.float32 | 519.41 | 0.00 | 0.00 |
| 2 | 16 | 8192 | 64 | True | 0.1 | 32 | 16 | 4 | torch.float16 | torch.float32 | 478.34 | 0.01 | 0.00 |

## NSATopkVarlenOp

### tileops

| seq_num | c_seq_len | heads | dim | group | scale | selected_block_num | bc | bs | bk | dtype | accum_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 1024 | 32 | 128 | 16 | 1 | 16 | 32 | 32 | 128 | torch.float16 | torch.float32 | 0.04 | 6.88 | 0.65 |
| 3 | 512 | 32 | 128 | 16 | 1 | 16 | 32 | 32 | 128 | torch.float16 | torch.float32 | 0.02 | 2.74 | 0.35 |
| 9 | 8192 | 32 | 128 | 16 | 1 | 16 | 32 | 32 | 128 | torch.float16 | torch.float32 | 0.37 | 46.55 | 3.09 |

### baseline

| seq_num | c_seq_len | heads | dim | group | scale | selected_block_num | bc | bs | bk | dtype | accum_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 1024 | 32 | 128 | 16 | 1 | 16 | 32 | 32 | 128 | torch.float16 | torch.float32 | 240.30 | 0.00 | 0.00 |
| 3 | 512 | 32 | 128 | 16 | 1 | 16 | 32 | 32 | 128 | torch.float16 | torch.float32 | 119.56 | 0.00 | 0.00 |
| 9 | 8192 | 32 | 128 | 16 | 1 | 16 | 32 | 32 | 128 | torch.float16 | torch.float32 | 2492.96 | 0.01 | 0.00 |

## DropoutOp

### tileops

| dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| torch.float16 | 4194304 | 0.01 | 0.64 | 2.56 |
| torch.bfloat16 | 4194304 | 0.01 | 0.64 | 2.56 |
| torch.float32 | 4194304 | 0.01 | 0.39 | 3.09 |
| torch.float16 | 10485760 | 0.01 | 0.83 | 3.30 |
| torch.bfloat16 | 10485760 | 0.01 | 0.83 | 3.31 |
| torch.float32 | 10485760 | 0.02 | 0.47 | 3.78 |
| torch.float16 | 20971520 | 0.02 | 0.94 | 3.78 |
| torch.bfloat16 | 20971520 | 0.02 | 0.95 | 3.78 |
| torch.float32 | 20971520 | 0.04 | 0.51 | 4.07 |

### baseline

| dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| torch.float16 | 4194304 | 0.01 | 0.38 | 1.53 |
| torch.bfloat16 | 4194304 | 0.01 | 0.38 | 1.51 |
| torch.float32 | 4194304 | 0.01 | 0.28 | 2.25 |
| torch.float16 | 10485760 | 0.02 | 0.50 | 2.00 |
| torch.bfloat16 | 10485760 | 0.02 | 0.49 | 1.97 |
| torch.float32 | 10485760 | 0.03 | 0.32 | 2.60 |
| torch.float16 | 20971520 | 0.04 | 0.55 | 2.19 |
| torch.bfloat16 | 20971520 | 0.04 | 0.54 | 2.16 |
| torch.float32 | 20971520 | 0.06 | 0.34 | 2.75 |

## relu_fp8

### tileops

| op_name | n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| relu_fp8 | 8388608 | torch.float8_e4m3fn | 0.01 | 0.57 | 1.14 |
| relu_fp8 | 8388608 | torch.float8_e5m2 | 0.02 | 0.45 | 0.91 |
| relu_fp8 | 67108864 | torch.float8_e4m3fn | 0.09 | 0.71 | 1.42 |
| relu_fp8 | 67108864 | torch.float8_e5m2 | 0.12 | 0.54 | 1.09 |
| relu_fp8 | 134217728 | torch.float8_e4m3fn | 0.18 | 0.73 | 1.46 |
| relu_fp8 | 134217728 | torch.float8_e5m2 | 0.24 | 0.56 | 1.12 |

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
| exp_fp8 | 134217728 | torch.float8_e4m3fn | 0.10 | 1.38 | 2.76 |
| exp_fp8 | 134217728 | torch.float8_e5m2 | 0.24 | 0.56 | 1.12 |

### baseline

| op_name | n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| exp_fp8 | 8388608 | torch.float8_e4m3fn | 0.05 | 0.19 | 0.37 |
| exp_fp8 | 8388608 | torch.float8_e5m2 | 0.04 | 0.19 | 0.38 |
| exp_fp8 | 67108864 | torch.float8_e4m3fn | 0.33 | 0.20 | 0.40 |
| exp_fp8 | 67108864 | torch.float8_e5m2 | 0.32 | 0.21 | 0.41 |
| exp_fp8 | 134217728 | torch.float8_e4m3fn | 0.64 | 0.21 | 0.42 |
| exp_fp8 | 134217728 | torch.float8_e5m2 | 0.63 | 0.21 | 0.43 |

## add_fp8

### tileops

| op_name | n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| add_fp8 | 8388608 | torch.float8_e4m3fn | 0.01 | 0.85 | 2.56 |
| add_fp8 | 8388608 | torch.float8_e5m2 | 0.02 | 0.41 | 1.24 |
| add_fp8 | 67108864 | torch.float8_e4m3fn | 0.06 | 1.15 | 3.46 |
| add_fp8 | 67108864 | torch.float8_e5m2 | 0.13 | 0.50 | 1.50 |
| add_fp8 | 134217728 | torch.float8_e4m3fn | 0.11 | 1.20 | 3.61 |
| add_fp8 | 134217728 | torch.float8_e5m2 | 0.26 | 0.52 | 1.56 |

### baseline

| op_name | n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| add_fp8 | 8388608 | torch.float8_e4m3fn | 0.08 | 0.11 | 0.32 |
| add_fp8 | 8388608 | torch.float8_e5m2 | 0.07 | 0.11 | 0.34 |
| add_fp8 | 67108864 | torch.float8_e4m3fn | 0.56 | 0.12 | 0.36 |
| add_fp8 | 67108864 | torch.float8_e5m2 | 0.54 | 0.12 | 0.37 |
| add_fp8 | 134217728 | torch.float8_e4m3fn | 1.07 | 0.12 | 0.37 |
| add_fp8 | 134217728 | torch.float8_e5m2 | 1.04 | 0.13 | 0.39 |

## silu_and_mul_fp8

### tileops

| op_name | M | N | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| silu_and_mul_fp8 | 2048 | 11008 | torch.float8_e4m3fn | 0.04 | 2.58 | 1.55 |
| silu_and_mul_fp8 | 2048 | 11008 | torch.float8_e5m2 | 0.06 | 1.76 | 1.06 |
| silu_and_mul_fp8 | 16384 | 11008 | torch.float8_e4m3fn | 0.31 | 2.90 | 1.74 |
| silu_and_mul_fp8 | 16384 | 11008 | torch.float8_e5m2 | 0.45 | 2.00 | 1.20 |
| silu_and_mul_fp8 | 16384 | 28672 | torch.float8_e4m3fn | 0.77 | 3.06 | 1.84 |
| silu_and_mul_fp8 | 16384 | 28672 | torch.float8_e5m2 | 1.10 | 2.14 | 1.28 |

### baseline

| op_name | M | N | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| silu_and_mul_fp8 | 2048 | 11008 | torch.float8_e4m3fn | 0.29 | 0.38 | 0.23 |
| silu_and_mul_fp8 | 2048 | 11008 | torch.float8_e5m2 | 0.29 | 0.39 | 0.24 |
| silu_and_mul_fp8 | 16384 | 11008 | torch.float8_e4m3fn | 2.02 | 0.45 | 0.27 |
| silu_and_mul_fp8 | 16384 | 11008 | torch.float8_e5m2 | 1.97 | 0.46 | 0.28 |
| silu_and_mul_fp8 | 16384 | 28672 | torch.float8_e4m3fn | 4.11 | 0.57 | 0.34 |
| silu_and_mul_fp8 | 16384 | 28672 | torch.float8_e5m2 | 4.05 | 0.58 | 0.35 |

## EngramGateConvBwdOp

### tileops

| M | seq_len | d | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 256 | torch.float16 | 0.01 | 0.07 | 0.03 |
| 2 | 64 | 512 | torch.float16 | 0.01 | 0.33 | 0.11 |
| 1 | 128 | 256 | torch.bfloat16 | 0.01 | 0.19 | 0.06 |
| 2 | 16 | 256 | torch.bfloat16 | 0.01 | 0.07 | 0.02 |

### torch

| M | seq_len | d | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 256 | torch.float16 | 0.20 | 0.00 | 0.00 |
| 2 | 64 | 512 | torch.float16 | 0.23 | 0.02 | 0.01 |
| 1 | 128 | 256 | torch.bfloat16 | 0.21 | 0.01 | 0.00 |
| 2 | 16 | 256 | torch.bfloat16 | 0.20 | 0.00 | 0.00 |

## EngramDecodeOp

### tileops

| batch | d_mem | d | max_conv_len | conv_kernel_size | dilation | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 512 | 256 | 12 | 4 | 3 | torch.float16 | 0.03 | 0.02 | 0.02 |
| 4 | 1024 | 512 | 20 | 4 | 5 | torch.float16 | 0.07 | 0.12 | 0.03 |
| 8 | 512 | 256 | 18 | 4 | 3 | torch.bfloat16 | 0.03 | 0.14 | 0.02 |

### baseline

| batch | d_mem | d | max_conv_len | conv_kernel_size | dilation | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 512 | 256 | 12 | 4 | 3 | torch.float16 | 0.10 | 0.01 | 0.01 |
| 4 | 1024 | 512 | 20 | 4 | 5 | torch.float16 | 0.13 | 0.06 | 0.02 |
| 8 | 512 | 256 | 18 | 4 | 3 | torch.bfloat16 | 0.12 | 0.04 | 0.01 |

## EngramGateConvFwdOp

### tileops

| M | seq_len | d | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 256 | torch.float16 | 0.00 | 0.05 | 0.02 |
| 2 | 64 | 512 | torch.float16 | 0.01 | 0.30 | 0.13 |
| 1 | 128 | 256 | torch.bfloat16 | 0.00 | 0.16 | 0.07 |
| 2 | 16 | 256 | torch.bfloat16 | 0.00 | 0.05 | 0.02 |

### baseline

| M | seq_len | d | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 256 | torch.float16 | 0.08 | 0.00 | 0.00 |
| 2 | 64 | 512 | torch.float16 | 0.09 | 0.02 | 0.01 |
| 1 | 128 | 256 | torch.bfloat16 | 0.09 | 0.01 | 0.00 |
| 2 | 16 | 256 | torch.bfloat16 | 0.08 | 0.00 | 0.00 |

## FFTC2COp

### tileops

| n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 64 | torch.complex64 | 0.02 | 0.00 | 0.00 |
| 128 | torch.complex64 | 0.02 | 0.00 | 0.00 |
| 256 | torch.complex64 | 0.03 | 0.00 | 0.00 |
| 512 | torch.complex64 | 0.03 | 0.00 | 0.00 |
| 1024 | torch.complex64 | 0.03 | 0.00 | 0.00 |

### baseline

| n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 64 | torch.complex64 | 0.00 | 0.00 | 0.00 |
| 128 | torch.complex64 | 0.00 | 0.00 | 0.00 |
| 256 | torch.complex64 | 0.00 | 0.00 | 0.00 |
| 512 | torch.complex64 | 0.00 | 0.01 | 0.00 |
| 1024 | torch.complex64 | 0.00 | 0.02 | 0.01 |

### tileops-base

| n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 64 | torch.complex64 | 0.02 | 0.00 | 0.00 |
| 128 | torch.complex64 | 0.02 | 0.00 | 0.00 |
| 256 | torch.complex64 | 0.03 | 0.00 | 0.00 |
| 512 | torch.complex64 | 0.03 | 0.00 | 0.00 |
| 1024 | torch.complex64 | 0.03 | 0.00 | 0.00 |
| 4096 | torch.complex64 | 0.03 | 0.01 | 0.00 |
| 16384 | torch.complex64 | 0.04 | 0.03 | 0.01 |

## FFTC2CLUTOp

### tileops-lut

| n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 64 | torch.complex64 | 0.01 | 0.00 | 0.00 |
| 128 | torch.complex64 | 0.01 | 0.00 | 0.00 |
| 256 | torch.complex64 | 0.01 | 0.00 | 0.00 |
| 512 | torch.complex64 | 0.01 | 0.00 | 0.00 |
| 1024 | torch.complex64 | 0.01 | 0.00 | 0.00 |
| 4096 | torch.complex64 | 0.01 | 0.02 | 0.01 |
| 16384 | torch.complex64 | 0.01 | 0.08 | 0.02 |

### baseline

| n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 64 | torch.complex64 | 0.00 | 0.00 | 0.00 |
| 128 | torch.complex64 | 0.00 | 0.00 | 0.00 |
| 256 | torch.complex64 | 0.00 | 0.00 | 0.00 |
| 512 | torch.complex64 | 0.00 | 0.01 | 0.00 |
| 1024 | torch.complex64 | 0.00 | 0.02 | 0.01 |
| 4096 | torch.complex64 | 0.01 | 0.05 | 0.01 |
| 16384 | torch.complex64 | 0.01 | 0.09 | 0.02 |

## Fp8LightingIndexerOp

### tileops

| batch | seq_len | heads | index_dim | seq_len_kv | kv_group | clean_logits | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 4096 | 32 | 64 | 8192 | 1 | True | 0.19 | N/A | 0.77 |
| 1 | 2048 | 16 | 64 | 4096 | 1 | True | 0.12 | N/A | 0.30 |

### baseline

| batch | seq_len | heads | index_dim | seq_len_kv | kv_group | clean_logits | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 4096 | 32 | 64 | 8192 | 1 | True | 9.74 | N/A | 0.01 |
| 1 | 2048 | 16 | 64 | 4096 | 1 | True | 1.53 | N/A | 0.02 |

## Fp8QuantOp

### tileops

| batch | seq_len_kv | kv_group | index_dim | in_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8192 | 1 | 64 | torch.float16 | 0.00 | 1.04 | 0.35 |
| 1 | 8192 | 1 | 64 | torch.bfloat16 | 0.00 | 1.03 | 0.34 |
| 1 | 4096 | 1 | 128 | torch.float32 | 0.00 | 0.78 | 0.52 |
| 1 | 16384 | 1 | 32 | torch.float32 | 0.00 | 0.98 | 0.65 |

### baseline

| batch | seq_len_kv | kv_group | index_dim | in_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8192 | 1 | 64 | torch.float16 | 0.02 | 0.18 | 0.06 |
| 1 | 8192 | 1 | 64 | torch.bfloat16 | 0.02 | 0.17 | 0.06 |
| 1 | 4096 | 1 | 128 | torch.float32 | 0.02 | 0.18 | 0.12 |
| 1 | 16384 | 1 | 32 | torch.float32 | 0.02 | 0.15 | 0.10 |

## FusedAddLayerNormOp

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.02 | 1.25 | 1.66 |
| 4096 | 4096 | torch.bfloat16 | 0.07 | 1.52 | 2.02 |
| 1024 | 3000 | torch.float16 | 0.04 | 0.52 | 0.70 |
| 1025 | 4096 | torch.float16 | 0.02 | 1.25 | 1.67 |

### baseline

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.06 | 0.44 | 0.59 |
| 4096 | 4096 | torch.bfloat16 | 0.21 | 0.48 | 0.64 |
| 1024 | 3000 | torch.float16 | 0.04 | 0.42 | 0.56 |
| 1025 | 4096 | torch.float16 | 0.06 | 0.44 | 0.59 |

## FusedAddRmsNormOp

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.02 | 1.13 | 1.81 |
| 4096 | 4096 | torch.bfloat16 | 0.06 | 1.42 | 2.26 |
| 2048 | 5120 | torch.float16 | 0.04 | 1.37 | 2.19 |
| 1025 | 4096 | torch.float16 | 0.02 | 1.13 | 1.81 |

### baseline

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.13 | 0.17 | 0.27 |
| 4096 | 4096 | torch.bfloat16 | 0.48 | 0.17 | 0.28 |
| 2048 | 5120 | torch.float16 | 0.32 | 0.16 | 0.26 |
| 1025 | 4096 | torch.float16 | 0.13 | 0.17 | 0.27 |

## GatedDeltaNetFwdOp

### tileops

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 0.19 | 1.41 | 0.09 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 0.20 | 1.36 | 0.09 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 0.08 | 1.65 | 0.10 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 0.14 | 1.93 | 0.12 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 0.29 | 1.88 | 0.12 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 0.54 | 1.99 | 0.13 |
| 2 | 32768 | 4 | 64 | 64 | 64 | torch.float16 | 1.03 | 2.08 | 0.13 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.08 | 1.65 | 0.10 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.14 | 1.94 | 0.12 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.28 | 1.90 | 0.12 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.53 | 2.01 | 0.13 |
| 2 | 32768 | 4 | 64 | 64 | 64 | torch.bfloat16 | 1.02 | 2.11 | 0.13 |

### baseline

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 130.97 | 0.00 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 131.26 | 0.00 | 0.00 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 45.12 | 0.00 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 90.62 | 0.00 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 181.88 | 0.00 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 366.24 | 0.00 | 0.00 |
| 2 | 32768 | 4 | 64 | 64 | 64 | torch.float16 | 730.82 | 0.00 | 0.00 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 45.07 | 0.00 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 90.97 | 0.00 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 183.48 | 0.00 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 363.49 | 0.00 | 0.00 |
| 2 | 32768 | 4 | 64 | 64 | 64 | torch.bfloat16 | 728.69 | 0.00 | 0.00 |

## GatedDeltaNetBwdOp

### tileops

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | H | S | DK | DV | BC | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.42 | 1.27 | 0.07 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.42 | 1.28 | 0.07 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.19 | 1.43 | 0.08 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.34 | 1.56 | 0.09 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.63 | 1.70 | 0.09 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 16384 | 64 | 64 | 64 | 1.20 | 1.78 | 0.10 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.19 | 1.41 | 0.08 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.35 | 1.55 | 0.09 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.63 | 1.69 | 0.09 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 16384 | 64 | 64 | 64 | 1.21 | 1.78 | 0.10 |

### baseline

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | H | S | DK | DV | BC | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 32 | 50.09 | 0.01 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 32 | 50.08 | 0.01 | 0.00 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 2048 | 64 | 64 | 64 | 19.43 | 0.01 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 64 | 39.72 | 0.01 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 8192 | 64 | 64 | 64 | 82.66 | 0.01 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 16384 | 64 | 64 | 64 | 175.67 | 0.01 | 0.00 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 2048 | 64 | 64 | 64 | 19.43 | 0.01 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 64 | 39.73 | 0.01 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 8192 | 64 | 64 | 64 | 82.62 | 0.01 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 16384 | 64 | 64 | 64 | 175.60 | 0.01 | 0.00 |

## GatedDeltaNetOp

### tileops

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | H | S | DK | DV | BC | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.60 | 1.34 | 0.08 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.61 | 1.33 | 0.08 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.26 | 1.54 | 0.09 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.48 | 1.69 | 0.10 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.90 | 1.79 | 0.10 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 16384 | 64 | 64 | 64 | 1.68 | 1.92 | 0.11 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.26 | 1.52 | 0.09 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.48 | 1.69 | 0.10 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.90 | 1.79 | 0.10 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 16384 | 64 | 64 | 64 | 1.69 | 1.91 | 0.11 |

### baseline

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | H | S | DK | DV | BC | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 32 | 50.15 | 0.02 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 32 | 50.15 | 0.02 | 0.00 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 2048 | 64 | 64 | 64 | 19.44 | 0.02 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 64 | 39.74 | 0.02 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 8192 | 64 | 64 | 64 | 82.64 | 0.02 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 16384 | 64 | 64 | 64 | 175.58 | 0.02 | 0.00 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 2048 | 64 | 64 | 64 | 19.43 | 0.02 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 64 | 39.72 | 0.02 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 8192 | 64 | 64 | 64 | 82.72 | 0.02 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 16384 | 64 | 64 | 64 | 175.67 | 0.02 | 0.00 |

## GatedDeltaNetDecodeOp

### tileops

| batch | heads | dim_k | dim_v | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 128 | 128 | torch.float32 | 0.01 | 0.30 | 0.41 |
| 1 | 32 | 128 | 128 | torch.bfloat16 | 0.01 | 0.31 | 0.21 |
| 8 | 32 | 128 | 128 | torch.float32 | 0.02 | 1.17 | 1.58 |
| 8 | 32 | 128 | 128 | torch.bfloat16 | 0.01 | 2.10 | 1.42 |
| 16 | 32 | 128 | 128 | torch.float32 | 0.04 | 1.27 | 1.72 |
| 16 | 32 | 128 | 128 | torch.bfloat16 | 0.02 | 3.03 | 2.05 |
| 32 | 32 | 128 | 128 | torch.float32 | 0.08 | 1.33 | 1.79 |
| 32 | 32 | 128 | 128 | torch.bfloat16 | 0.03 | 3.27 | 2.21 |
| 1 | 32 | 64 | 64 | torch.float32 | 0.01 | 0.14 | 0.19 |
| 8 | 32 | 64 | 64 | torch.float32 | 0.01 | 0.59 | 0.81 |
| 32 | 32 | 64 | 64 | torch.float32 | 0.04 | 0.71 | 0.97 |
| 64 | 32 | 128 | 128 | torch.float32 | 0.15 | 1.38 | 1.86 |
| 64 | 32 | 128 | 128 | torch.bfloat16 | 0.06 | 3.66 | 2.47 |

### baseline

| batch | heads | dim_k | dim_v | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 128 | 128 | torch.float32 | 0.03 | 0.09 | 0.13 |
| 1 | 32 | 128 | 128 | torch.bfloat16 | 0.05 | 0.06 | 0.04 |
| 8 | 32 | 128 | 128 | torch.float32 | 0.08 | 0.31 | 0.42 |
| 8 | 32 | 128 | 128 | torch.bfloat16 | 0.11 | 0.23 | 0.15 |
| 16 | 32 | 128 | 128 | torch.float32 | 0.12 | 0.41 | 0.56 |
| 16 | 32 | 128 | 128 | torch.bfloat16 | 0.18 | 0.29 | 0.19 |
| 32 | 32 | 128 | 128 | torch.float32 | 0.21 | 0.47 | 0.64 |
| 32 | 32 | 128 | 128 | torch.bfloat16 | 0.31 | 0.33 | 0.22 |
| 1 | 32 | 64 | 64 | torch.float32 | 0.03 | 0.03 | 0.04 |
| 8 | 32 | 64 | 64 | torch.float32 | 0.04 | 0.16 | 0.22 |
| 32 | 32 | 64 | 64 | torch.float32 | 0.08 | 0.32 | 0.44 |
| 64 | 32 | 128 | 128 | torch.float32 | 0.39 | 0.52 | 0.70 |
| 64 | 32 | 128 | 128 | torch.bfloat16 | 0.56 | 0.36 | 0.24 |

## GemmOp

### tileops

| m | n | k | dtype | trans_a | trans_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1024 | 1024 | 1024 | torch.float16 | False | False | 0.01 | 240.04 | 0.70 |
| 1 | 7168 | 16384 | torch.float16 | False | True | 0.06 | 3.63 | 3.63 |
| 1 | 18432 | 7168 | torch.bfloat16 | False | True | 0.07 | 3.81 | 3.81 |
| 7168 | 1 | 16384 | torch.float16 | False | False | 0.06 | 3.63 | 3.64 |
| 18432 | 1 | 7168 | torch.bfloat16 | False | False | 0.07 | 3.81 | 3.81 |

### baseline

| m | n | k | dtype | trans_a | trans_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1024 | 1024 | 1024 | torch.float16 | False | False | 0.01 | 322.94 | 0.95 |
| 1 | 7168 | 16384 | torch.float16 | False | True | 0.07 | 3.43 | 3.43 |
| 1 | 18432 | 7168 | torch.bfloat16 | False | True | 0.07 | 3.61 | 3.61 |
| 7168 | 1 | 16384 | torch.float16 | False | False | 0.07 | 3.42 | 3.43 |
| 18432 | 1 | 7168 | torch.bfloat16 | False | False | 0.07 | 3.60 | 3.60 |

## GLAFwdOp

### tileops

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.08 | 1.73 | 0.11 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.14 | 1.97 | 0.12 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.26 | 2.03 | 0.13 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.50 | 2.14 | 0.13 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.08 | 1.66 | 0.10 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.14 | 1.97 | 0.12 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.26 | 2.04 | 0.13 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.51 | 2.11 | 0.13 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.23 | 1.78 | 0.10 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.43 | 1.86 | 0.11 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.81 | 1.98 | 0.11 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 1.50 | 2.15 | 0.12 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.23 | 1.76 | 0.10 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.43 | 1.86 | 0.11 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.83 | 1.94 | 0.11 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 1.52 | 2.11 | 0.12 |

### baseline

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 1.99 | 0.07 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 4.05 | 0.07 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 8.09 | 0.07 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 16.15 | 0.07 | 0.00 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 1.99 | 0.07 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 4.04 | 0.07 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 8.09 | 0.07 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 16.15 | 0.07 | 0.00 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 5.98 | 0.07 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 12.54 | 0.06 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 28.62 | 0.06 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 70.33 | 0.05 | 0.00 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 5.98 | 0.07 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 12.55 | 0.06 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 28.62 | 0.06 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 70.40 | 0.05 | 0.00 |

## GLABwdOp

### tileops

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | T | H | K | V | BC | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 2048 | 4 | 64 | 64 | 64 | 0.125 | 0.15 | 1.73 | 0.09 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4096 | 4 | 64 | 64 | 64 | 0.125 | 0.30 | 1.79 | 0.10 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 8192 | 4 | 64 | 64 | 64 | 0.125 | 0.57 | 1.89 | 0.10 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 16384 | 4 | 64 | 64 | 64 | 0.125 | 1.07 | 2.01 | 0.11 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 2048 | 4 | 64 | 64 | 64 | 0.125 | 0.15 | 1.74 | 0.10 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4096 | 4 | 64 | 64 | 64 | 0.125 | 0.30 | 1.78 | 0.10 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 8192 | 4 | 64 | 64 | 64 | 0.125 | 0.60 | 1.80 | 0.10 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 16384 | 4 | 64 | 64 | 64 | 0.125 | 1.10 | 1.95 | 0.11 |

### baseline

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | T | H | K | V | BC | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 2048 | 4 | 64 | 64 | 64 | 0.125 | 5.98 | 0.04 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4096 | 4 | 64 | 64 | 64 | 0.125 | 12.55 | 0.04 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 8192 | 4 | 64 | 64 | 64 | 0.125 | 28.62 | 0.04 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 16384 | 4 | 64 | 64 | 64 | 0.125 | 70.21 | 0.03 | 0.00 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 2048 | 4 | 64 | 64 | 64 | 0.125 | 5.98 | 0.04 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4096 | 4 | 64 | 64 | 64 | 0.125 | 12.54 | 0.04 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 8192 | 4 | 64 | 64 | 64 | 0.125 | 28.62 | 0.04 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 16384 | 4 | 64 | 64 | 64 | 0.125 | 70.22 | 0.03 | 0.00 |

## GLADecodeOp

### tileops

| batch | heads | dim_k | dim_v | dtype | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 64 | 64 | torch.float32 | 0.125 | 0.01 | 0.07 | 0.14 |
| 1 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.02 | 0.13 | 0.26 |
| 1 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.01 | 0.17 | 0.17 |
| 1 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.01 | 0.17 | 0.17 |
| 8 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.03 | 0.49 | 1.00 |
| 8 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.01 | 1.19 | 1.21 |
| 8 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.01 | 1.18 | 1.19 |
| 16 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.07 | 0.52 | 1.05 |
| 16 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.02 | 1.85 | 1.88 |
| 16 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.02 | 1.83 | 1.86 |
| 32 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.13 | 0.54 | 1.09 |
| 32 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.03 | 1.94 | 1.97 |
| 32 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.04 | 1.92 | 1.95 |
| 64 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.24 | 0.55 | 1.12 |
| 64 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.06 | 2.16 | 2.20 |
| 64 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.06 | 2.14 | 2.17 |

### baseline

| batch | heads | dim_k | dim_v | dtype | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 64 | 64 | torch.float32 | 0.125 | 0.01 | 0.04 | 0.08 |
| 1 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.02 | 0.13 | 0.25 |
| 1 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.04 | 0.06 | 0.06 |
| 1 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.04 | 0.06 | 0.06 |
| 8 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.05 | 0.32 | 0.66 |
| 8 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.09 | 0.20 | 0.20 |
| 8 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.09 | 0.20 | 0.20 |
| 16 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.09 | 0.36 | 0.73 |
| 16 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.15 | 0.23 | 0.23 |
| 16 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.15 | 0.22 | 0.23 |
| 32 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.18 | 0.38 | 0.78 |
| 32 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.27 | 0.25 | 0.25 |
| 32 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.27 | 0.25 | 0.25 |
| 64 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.33 | 0.40 | 0.82 |
| 64 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.50 | 0.27 | 0.27 |
| 64 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.50 | 0.27 | 0.27 |

## GroupQueryAttentionFwdOp

### tileops

| batch | seq_len | heads | heads_kv | dim | causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 8 | 4 | 64 | False | torch.float16 | 0.01 | 208.88 | 0.31 |
| 4 | 2048 | 64 | 4 | 128 | False | torch.float16 | 0.83 | 659.97 | 0.34 |
| 4 | 2048 | 64 | 4 | 128 | False | torch.bfloat16 | 0.81 | 677.86 | 0.35 |

### torch

| batch | seq_len | heads | heads_kv | dim | causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 8 | 4 | 64 | False | torch.float16 | 0.02 | 117.36 | 0.17 |
| 4 | 2048 | 64 | 4 | 128 | False | torch.float16 | 1.25 | 438.56 | 0.23 |
| 4 | 2048 | 64 | 4 | 128 | False | torch.bfloat16 | 1.31 | 420.53 | 0.22 |

## GroupQueryAttentionBwdOp

### tileops

| batch | seq_len | heads | heads_kv | dim | causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 8 | 4 | 64 | False | torch.float16 | 0.05 | 101.08 | 0.10 |
| 4 | 2048 | 64 | 4 | 128 | False | torch.float16 | 3.57 | 384.53 | 0.12 |
| 4 | 2048 | 64 | 4 | 128 | False | torch.bfloat16 | 3.45 | 398.38 | 0.13 |

### torch

| batch | seq_len | heads | heads_kv | dim | causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 8 | 4 | 64 | False | torch.float16 | 0.08 | 69.35 | 0.07 |
| 4 | 2048 | 64 | 4 | 128 | False | torch.float16 | 4.72 | 290.97 | 0.09 |
| 4 | 2048 | 64 | 4 | 128 | False | torch.bfloat16 | 4.88 | 281.59 | 0.09 |

## GroupQueryAttentionDecodeWithKVCacheOp

### tileops

| batch | heads | heads_kv | seq_len_kv | dim | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 8 | 8192 | 128 | torch.float16 | 0.03 | 3.97 | 0.99 |
| 4 | 32 | 4 | 4096 | 128 | torch.bfloat16 | 0.04 | 6.73 | 0.84 |
| 8 | 64 | 16 | 8192 | 128 | torch.float16 | 0.17 | 12.89 | 3.22 |

### baseline

| batch | heads | heads_kv | seq_len_kv | dim | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 8 | 8192 | 128 | torch.float16 | 0.43 | 0.31 | 0.08 |
| 4 | 32 | 4 | 4096 | 128 | torch.bfloat16 | 0.82 | 0.33 | 0.04 |
| 8 | 64 | 16 | 8192 | 128 | torch.float16 | 6.16 | 0.35 | 0.09 |

## GroupQueryAttentionDecodePagedWithKVCacheOp

### tileops

| batch | heads | heads_kv | seqlen_kv | dim | page_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 16 | 8 | 512 | 128 | 128 | torch.float16 | 0.02 | 0.21 | 0.10 |
| 2 | 8 | 4 | 1024 | 64 | 256 | torch.float16 | 0.02 | 0.20 | 0.05 |
| 1 | 16 | 4 | 2048 | 128 | 512 | torch.float16 | 0.02 | 0.79 | 0.20 |
| 1 | 32 | 16 | 512 | 64 | 128 | torch.float16 | 0.02 | 0.22 | 0.11 |

### baseline

| batch | heads | heads_kv | seqlen_kv | dim | page_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 16 | 8 | 512 | 128 | 128 | torch.float16 | 0.07 | 0.06 | 0.03 |
| 2 | 8 | 4 | 1024 | 64 | 256 | torch.float16 | 0.13 | 0.03 | 0.01 |
| 1 | 16 | 4 | 2048 | 128 | 512 | torch.float16 | 0.07 | 0.25 | 0.06 |
| 1 | 32 | 16 | 512 | 64 | 128 | torch.float16 | 0.07 | 0.06 | 0.03 |

## GqaSlidingWindowFwdOp

### tileops

| batch | seq | heads | heads_kv | dim | is_causal | wl | wr | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 512 | 8 | 2 | 64 | True | -1 | -1 | torch.float16 | 0.01 | 71.71 | 0.35 |
| 2 | 512 | 8 | 2 | 64 | True | 128 | -1 | torch.float16 | 0.01 | 41.52 | 0.46 |
| 2 | 768 | 8 | 2 | 64 | False | 256 | -1 | torch.float16 | 0.01 | 149.46 | 0.31 |
| 2 | 512 | 8 | 2 | 64 | False | 64 | 64 | torch.bfloat16 | 0.01 | 44.09 | 0.46 |
| 1 | 2048 | 8 | 2 | 64 | True | 512 | -1 | torch.float16 | 0.01 | 151.86 | 0.42 |

### torch

| batch | seq | heads | heads_kv | dim | is_causal | wl | wr | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 512 | 8 | 2 | 64 | True | -1 | -1 | torch.float16 | 0.15 | 3.59 | 0.02 |
| 2 | 512 | 8 | 2 | 64 | True | 128 | -1 | torch.float16 | 0.16 | 1.50 | 0.02 |
| 2 | 768 | 8 | 2 | 64 | False | 256 | -1 | torch.float16 | 0.28 | 6.83 | 0.01 |
| 2 | 512 | 8 | 2 | 64 | False | 64 | 64 | torch.bfloat16 | 0.16 | 1.59 | 0.02 |
| 1 | 2048 | 8 | 2 | 64 | True | 512 | -1 | torch.float16 | 0.81 | 2.31 | 0.01 |

## GqaSlidingWindowVarlenFwdOp

### tileops

| batch | heads | heads_kv | dim | is_causal | wl | wr | dtype | max_seqlen_q | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 8 | 128 | True | -1 | -1 | torch.float16 | 3000 | 0.22 | 340.90 | 0.28 |
| 2 | 32 | 8 | 128 | True | 2000 | -1 | torch.float16 | 3073 | 0.35 | 251.88 | 0.27 |
| 4 | 32 | 8 | 128 | False | 1500 | 1500 | torch.float16 | 3500 | 0.91 | 377.43 | 0.22 |
| 2 | 32 | 8 | 128 | True | -1 | -1 | torch.float16 | 1024 | 0.40 | 407.71 | 0.19 |
| 2 | 32 | 8 | 128 | True | -1 | -1 | torch.float16 | 8193 | 1.90 | 397.50 | 0.14 |

### torch

| batch | heads | heads_kv | dim | is_causal | wl | wr | dtype | max_seqlen_q | max_seqlen_k | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 8 | 128 | True | -1 | -1 | torch.float16 | 3000 | 3000 | 7.90 | 9.34 | 0.01 |
| 2 | 32 | 8 | 128 | True | 2000 | -1 | torch.float16 | 3073 | 3073 | 16.36 | 5.34 | 0.01 |
| 4 | 32 | 8 | 128 | False | 1500 | 1500 | torch.float16 | 3500 | 3500 | 36.06 | 9.54 | 0.01 |
| 2 | 32 | 8 | 128 | True | -1 | -1 | torch.float16 | 1024 | 8192 | 14.93 | 10.79 | 0.01 |
| 2 | 32 | 8 | 128 | True | -1 | -1 | torch.float16 | 8193 | 8193 | 53.84 | 14.02 | 0.01 |

## GroupNormOp

### tileops

| n | c | g | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 128 | 32 | torch.float16 | 0.01 | 0.36 | 0.29 |
| 8 | 128 | 32 | torch.bfloat16 | 0.01 | 0.36 | 0.29 |
| 4 | 256 | 32 | torch.float16 | 0.02 | 0.19 | 0.15 |
| 4 | 128 | 16 | torch.float16 | 0.02 | 0.12 | 0.10 |

### baseline

| n | c | g | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 128 | 32 | torch.float16 | 0.01 | 0.35 | 0.28 |
| 8 | 128 | 32 | torch.bfloat16 | 0.01 | 0.35 | 0.28 |
| 4 | 256 | 32 | torch.float16 | 0.02 | 0.25 | 0.20 |
| 4 | 128 | 16 | torch.float16 | 0.02 | 0.14 | 0.12 |

## grouped_gemm_nt

### tileops

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_nt | 16384 | 4 | 4864 | 4096 | torch.float16 | False | True | 1.08 | 605.97 | 0.42 |

### baseline

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_nt | 16384 | 4 | 4864 | 4096 | torch.float16 | False | True | 1.65 | 394.98 | 0.27 |

## grouped_gemm_nn

### tileops

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_nn | 16384 | 4 | 4864 | 4096 | torch.float16 | False | False | 1.11 | 586.01 | 0.41 |

### baseline

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_nn | 16384 | 4 | 4864 | 4096 | torch.float16 | False | False | 1.17 | 559.63 | 0.39 |

## grouped_gemm_tn

### tileops

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_tn | 16384 | 4 | 4864 | 4096 | torch.float16 | True | False | 9.30 | 70.19 | 0.05 |

### baseline

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_tn | 16384 | 4 | 4864 | 4096 | torch.float16 | True | False | 1.02 | 638.95 | 0.44 |

## grouped_gemm_tt

### tileops

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_tt | 16384 | 4 | 4864 | 4096 | torch.float16 | True | True | 9.04 | 72.25 | 0.05 |

### baseline

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_tt | 16384 | 4 | 4864 | 4096 | torch.float16 | True | True | 1.02 | 638.58 | 0.44 |

## GroupedGemmOp

### tileops

| batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16384 | 4 | 4864 | 4096 | torch.float16 | True | False | 20.84 | 93.99 | N/A |

### baseline

| batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16384 | 4 | 4864 | 4096 | torch.float16 | True | False | 4.90 | 399.38 | N/A |

## leaky_relu

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| leaky_relu | torch.float16 | 4194304 | 0.01 | 0.66 | 2.65 |
| leaky_relu | torch.bfloat16 | 4194304 | 0.01 | 0.66 | 2.65 |
| leaky_relu | torch.float32 | 4194304 | 0.01 | 0.39 | 3.16 |
| leaky_relu | torch.float16 | 10485760 | 0.01 | 0.84 | 3.38 |
| leaky_relu | torch.bfloat16 | 10485760 | 0.01 | 0.84 | 3.37 |
| leaky_relu | torch.float32 | 10485760 | 0.02 | 0.47 | 3.79 |
| leaky_relu | torch.float16 | 20971520 | 0.02 | 0.95 | 3.80 |
| leaky_relu | torch.bfloat16 | 20971520 | 0.02 | 0.95 | 3.80 |
| leaky_relu | torch.float32 | 20971520 | 0.04 | 0.51 | 4.11 |

### baseline

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| leaky_relu | torch.float16 | 4194304 | 0.01 | 0.65 | 2.61 |
| leaky_relu | torch.bfloat16 | 4194304 | 0.01 | 0.65 | 2.60 |
| leaky_relu | torch.float32 | 4194304 | 0.01 | 0.40 | 3.17 |
| leaky_relu | torch.float16 | 10485760 | 0.01 | 0.84 | 3.34 |
| leaky_relu | torch.bfloat16 | 10485760 | 0.01 | 0.84 | 3.34 |
| leaky_relu | torch.float32 | 10485760 | 0.02 | 0.47 | 3.74 |
| leaky_relu | torch.float16 | 20971520 | 0.02 | 0.95 | 3.78 |
| leaky_relu | torch.bfloat16 | 20971520 | 0.02 | 0.95 | 3.78 |
| leaky_relu | torch.float32 | 20971520 | 0.04 | 0.51 | 4.05 |

## elu

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| elu | torch.float16 | 4194304 | 0.01 | 0.61 | 2.45 |
| elu | torch.bfloat16 | 4194304 | 0.01 | 0.61 | 2.45 |
| elu | torch.float32 | 4194304 | 0.01 | 0.39 | 3.11 |
| elu | torch.float16 | 10485760 | 0.01 | 0.77 | 3.06 |
| elu | torch.bfloat16 | 10485760 | 0.01 | 0.76 | 3.04 |
| elu | torch.float32 | 10485760 | 0.02 | 0.47 | 3.73 |
| elu | torch.float16 | 20971520 | 0.02 | 0.86 | 3.42 |
| elu | torch.bfloat16 | 20971520 | 0.02 | 0.86 | 3.45 |
| elu | torch.float32 | 20971520 | 0.04 | 0.50 | 4.01 |

### baseline

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| elu | torch.float16 | 4194304 | 0.01 | 0.42 | 1.70 |
| elu | torch.bfloat16 | 4194304 | 0.01 | 0.42 | 1.68 |
| elu | torch.float32 | 4194304 | 0.01 | 0.37 | 2.97 |
| elu | torch.float16 | 10485760 | 0.02 | 0.52 | 2.06 |
| elu | torch.bfloat16 | 10485760 | 0.02 | 0.51 | 2.04 |
| elu | torch.float32 | 10485760 | 0.02 | 0.46 | 3.64 |
| elu | torch.float16 | 20971520 | 0.04 | 0.57 | 2.27 |
| elu | torch.bfloat16 | 20971520 | 0.04 | 0.56 | 2.24 |
| elu | torch.float32 | 20971520 | 0.04 | 0.50 | 3.99 |

## hardtanh

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| hardtanh | torch.float16 | 4194304 | 0.01 | 0.66 | 2.63 |
| hardtanh | torch.bfloat16 | 4194304 | 0.01 | 0.66 | 2.63 |
| hardtanh | torch.float32 | 4194304 | 0.01 | 0.39 | 3.16 |
| hardtanh | torch.float16 | 10485760 | 0.01 | 0.84 | 3.37 |
| hardtanh | torch.bfloat16 | 10485760 | 0.01 | 0.84 | 3.38 |
| hardtanh | torch.float32 | 10485760 | 0.02 | 0.47 | 3.79 |
| hardtanh | torch.float16 | 20971520 | 0.02 | 0.95 | 3.80 |
| hardtanh | torch.bfloat16 | 20971520 | 0.02 | 0.95 | 3.79 |
| hardtanh | torch.float32 | 20971520 | 0.04 | 0.52 | 4.13 |

### baseline

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| hardtanh | torch.float16 | 4194304 | 0.01 | 0.62 | 2.49 |
| hardtanh | torch.bfloat16 | 4194304 | 0.01 | 0.64 | 2.56 |
| hardtanh | torch.float32 | 4194304 | 0.01 | 0.39 | 3.15 |
| hardtanh | torch.float16 | 10485760 | 0.01 | 0.78 | 3.13 |
| hardtanh | torch.bfloat16 | 10485760 | 0.01 | 0.83 | 3.31 |
| hardtanh | torch.float32 | 10485760 | 0.02 | 0.47 | 3.73 |
| hardtanh | torch.float16 | 20971520 | 0.02 | 0.87 | 3.48 |
| hardtanh | torch.bfloat16 | 20971520 | 0.02 | 0.94 | 3.75 |
| hardtanh | torch.float32 | 20971520 | 0.04 | 0.51 | 4.05 |

## softplus

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| softplus | torch.float16 | 4194304 | 0.01 | 0.42 | 1.70 |
| softplus | torch.bfloat16 | 4194304 | 0.01 | 0.42 | 1.67 |
| softplus | torch.float32 | 4194304 | 0.01 | 0.38 | 3.00 |
| softplus | torch.float16 | 10485760 | 0.02 | 0.52 | 2.07 |
| softplus | torch.bfloat16 | 10485760 | 0.02 | 0.51 | 2.03 |
| softplus | torch.float32 | 10485760 | 0.02 | 0.43 | 3.46 |
| softplus | torch.float16 | 20971520 | 0.04 | 0.57 | 2.27 |
| softplus | torch.bfloat16 | 20971520 | 0.04 | 0.56 | 2.22 |
| softplus | torch.float32 | 20971520 | 0.05 | 0.46 | 3.70 |

### baseline

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| softplus | torch.float16 | 4194304 | 0.01 | 0.36 | 1.43 |
| softplus | torch.bfloat16 | 4194304 | 0.01 | 0.36 | 1.42 |
| softplus | torch.float32 | 4194304 | 0.01 | 0.34 | 2.69 |
| softplus | torch.float16 | 10485760 | 0.02 | 0.43 | 1.71 |
| softplus | torch.bfloat16 | 10485760 | 0.02 | 0.42 | 1.69 |
| softplus | torch.float32 | 10485760 | 0.03 | 0.41 | 3.32 |
| softplus | torch.float16 | 20971520 | 0.05 | 0.46 | 1.85 |
| softplus | torch.bfloat16 | 20971520 | 0.05 | 0.46 | 1.83 |
| softplus | torch.float32 | 20971520 | 0.05 | 0.45 | 3.60 |

## clamp

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| clamp | torch.float16 | 4194304 | 0.01 | 0.66 | 2.63 |
| clamp | torch.bfloat16 | 4194304 | 0.01 | 0.66 | 2.63 |
| clamp | torch.float32 | 4194304 | 0.01 | 0.40 | 3.16 |
| clamp | torch.float16 | 10485760 | 0.01 | 0.84 | 3.38 |
| clamp | torch.bfloat16 | 10485760 | 0.01 | 0.84 | 3.37 |
| clamp | torch.float32 | 10485760 | 0.02 | 0.47 | 3.79 |
| clamp | torch.float16 | 20971520 | 0.02 | 0.95 | 3.80 |
| clamp | torch.bfloat16 | 20971520 | 0.02 | 0.95 | 3.80 |
| clamp | torch.float32 | 20971520 | 0.04 | 0.52 | 4.14 |

### baseline

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| clamp | torch.float16 | 4194304 | 0.01 | 0.62 | 2.49 |
| clamp | torch.bfloat16 | 4194304 | 0.01 | 0.64 | 2.57 |
| clamp | torch.float32 | 4194304 | 0.01 | 0.39 | 3.15 |
| clamp | torch.float16 | 10485760 | 0.01 | 0.78 | 3.14 |
| clamp | torch.bfloat16 | 10485760 | 0.01 | 0.83 | 3.31 |
| clamp | torch.float32 | 10485760 | 0.02 | 0.47 | 3.73 |
| clamp | torch.float16 | 20971520 | 0.02 | 0.87 | 3.49 |
| clamp | torch.bfloat16 | 20971520 | 0.02 | 0.94 | 3.75 |
| clamp | torch.float32 | 20971520 | 0.04 | 0.51 | 4.05 |

## nan_to_num

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| nan_to_num | torch.float16 | 4194304 | 0.01 | 0.64 | 2.57 |
| nan_to_num | torch.bfloat16 | 4194304 | 0.01 | 0.64 | 2.57 |
| nan_to_num | torch.float32 | 4194304 | 0.01 | 0.39 | 3.16 |
| nan_to_num | torch.float16 | 10485760 | 0.01 | 0.83 | 3.31 |
| nan_to_num | torch.bfloat16 | 10485760 | 0.01 | 0.83 | 3.31 |
| nan_to_num | torch.float32 | 10485760 | 0.02 | 0.47 | 3.79 |
| nan_to_num | torch.float16 | 20971520 | 0.02 | 0.93 | 3.73 |
| nan_to_num | torch.bfloat16 | 20971520 | 0.02 | 0.93 | 3.74 |
| nan_to_num | torch.float32 | 20971520 | 0.04 | 0.51 | 4.12 |

### baseline

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| nan_to_num | torch.float16 | 4194304 | 0.01 | 0.63 | 2.53 |
| nan_to_num | torch.bfloat16 | 4194304 | 0.01 | 0.63 | 2.53 |
| nan_to_num | torch.float32 | 4194304 | 0.01 | 0.40 | 3.17 |
| nan_to_num | torch.float16 | 10485760 | 0.01 | 0.82 | 3.28 |
| nan_to_num | torch.bfloat16 | 10485760 | 0.01 | 0.82 | 3.28 |
| nan_to_num | torch.float32 | 10485760 | 0.02 | 0.47 | 3.74 |
| nan_to_num | torch.float16 | 20971520 | 0.02 | 0.93 | 3.72 |
| nan_to_num | torch.bfloat16 | 20971520 | 0.02 | 0.93 | 3.72 |
| nan_to_num | torch.float32 | 20971520 | 0.04 | 0.51 | 4.05 |

## PreluOp

### tileops

| num_channels | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 128 | torch.float16 | 131072 | 0.00 | 0.06 | 0.25 |
| 128 | torch.bfloat16 | 131072 | 0.00 | 0.06 | 0.25 |
| 128 | torch.float32 | 131072 | 0.00 | 0.06 | 0.47 |
| 4096 | torch.float16 | 4194304 | 0.01 | 0.66 | 2.64 |
| 4096 | torch.bfloat16 | 4194304 | 0.01 | 0.66 | 2.64 |
| 4096 | torch.float32 | 4194304 | 0.01 | 0.40 | 3.18 |
| 10240 | torch.float16 | 10485760 | 0.01 | 0.84 | 3.37 |
| 10240 | torch.bfloat16 | 10485760 | 0.01 | 0.84 | 3.38 |
| 10240 | torch.float32 | 10485760 | 0.02 | 0.47 | 3.79 |
| 20480 | torch.float16 | 20971520 | 0.02 | 0.95 | 3.79 |
| 20480 | torch.bfloat16 | 20971520 | 0.02 | 0.95 | 3.79 |
| 20480 | torch.float32 | 20971520 | 0.04 | 0.51 | 4.06 |

### baseline

| num_channels | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 128 | torch.float16 | 131072 | 0.00 | 0.03 | 0.11 |
| 128 | torch.bfloat16 | 131072 | 0.00 | 0.03 | 0.11 |
| 128 | torch.float32 | 131072 | 0.00 | 0.04 | 0.29 |
| 4096 | torch.float16 | 4194304 | 0.01 | 0.29 | 1.15 |
| 4096 | torch.bfloat16 | 4194304 | 0.01 | 0.28 | 1.13 |
| 4096 | torch.float32 | 4194304 | 0.02 | 0.25 | 2.02 |
| 10240 | torch.float16 | 10485760 | 0.03 | 0.34 | 1.35 |
| 10240 | torch.bfloat16 | 10485760 | 0.03 | 0.33 | 1.33 |
| 10240 | torch.float32 | 10485760 | 0.04 | 0.29 | 2.31 |
| 20480 | torch.float16 | 20971520 | 0.06 | 0.36 | 1.46 |
| 20480 | torch.bfloat16 | 20971520 | 0.06 | 0.36 | 1.44 |
| 20480 | torch.float32 | 20971520 | 0.07 | 0.31 | 2.46 |

## WhereOp

### tileops

| dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| torch.float16 | 4194304 | 0.01 | 0.41 | 2.85 |
| torch.bfloat16 | 4194304 | 0.01 | 0.41 | 2.85 |
| torch.float32 | 4194304 | 0.02 | 0.25 | 3.30 |
| torch.float16 | 10485760 | 0.02 | 0.50 | 3.47 |
| torch.bfloat16 | 10485760 | 0.02 | 0.50 | 3.47 |
| torch.float32 | 10485760 | 0.04 | 0.30 | 3.86 |
| torch.float16 | 20971520 | 0.04 | 0.56 | 3.90 |
| torch.bfloat16 | 20971520 | 0.04 | 0.56 | 3.90 |
| torch.float32 | 20971520 | 0.07 | 0.32 | 4.18 |
| torch.float8_e4m3fn | 4194304 | 0.01 | 0.51 | 2.02 |
| torch.float8_e5m2 | 4194304 | 0.01 | 0.51 | 2.03 |
| torch.float8_e4m3fn | 10485760 | 0.02 | 0.58 | 2.32 |
| torch.float8_e5m2 | 10485760 | 0.02 | 0.58 | 2.32 |
| torch.float8_e4m3fn | 20971520 | 0.03 | 0.67 | 2.69 |
| torch.float8_e5m2 | 20971520 | 0.03 | 0.67 | 2.69 |

### baseline

| dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| torch.float16 | 4194304 | 0.01 | 0.41 | 2.88 |
| torch.bfloat16 | 4194304 | 0.01 | 0.41 | 2.88 |
| torch.float32 | 4194304 | 0.02 | 0.26 | 3.34 |
| torch.float16 | 10485760 | 0.02 | 0.51 | 3.57 |
| torch.bfloat16 | 10485760 | 0.02 | 0.50 | 3.51 |
| torch.float32 | 10485760 | 0.03 | 0.30 | 3.90 |
| torch.float16 | 20971520 | 0.04 | 0.56 | 3.92 |
| torch.bfloat16 | 20971520 | 0.04 | 0.56 | 3.92 |
| torch.float32 | 20971520 | 0.06 | 0.32 | 4.20 |
| torch.float8_e4m3fn | 4194304 | 0.01 | 0.59 | 2.36 |
| torch.float8_e5m2 | 4194304 | 0.01 | 0.59 | 2.36 |
| torch.float8_e4m3fn | 10485760 | 0.01 | 0.75 | 2.99 |
| torch.float8_e5m2 | 10485760 | 0.01 | 0.75 | 2.99 |
| torch.float8_e4m3fn | 20971520 | 0.02 | 0.85 | 3.39 |
| torch.float8_e5m2 | 20971520 | 0.02 | 0.85 | 3.40 |

## MaskedFillOp

### tileops

| dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| torch.float16 | 4194304 | 0.01 | 0.55 | 2.75 |
| torch.bfloat16 | 4194304 | 0.01 | 0.55 | 2.74 |
| torch.float32 | 4194304 | 0.01 | 0.35 | 3.13 |
| torch.float16 | 10485760 | 0.02 | 0.67 | 3.35 |
| torch.bfloat16 | 10485760 | 0.02 | 0.67 | 3.34 |
| torch.float32 | 10485760 | 0.02 | 0.42 | 3.78 |
| torch.float16 | 20971520 | 0.03 | 0.76 | 3.82 |
| torch.bfloat16 | 20971520 | 0.03 | 0.76 | 3.82 |
| torch.float32 | 20971520 | 0.05 | 0.46 | 4.11 |
| torch.float8_e4m3fn | 4194304 | 0.01 | 0.58 | 1.73 |
| torch.float8_e5m2 | 4194304 | 0.01 | 0.32 | 0.96 |
| torch.float8_e4m3fn | 10485760 | 0.01 | 0.75 | 2.26 |
| torch.float8_e5m2 | 10485760 | 0.03 | 0.40 | 1.20 |
| torch.float8_e4m3fn | 20971520 | 0.03 | 0.82 | 2.47 |
| torch.float8_e5m2 | 20971520 | 0.05 | 0.42 | 1.25 |

### baseline

| dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| torch.float16 | 4194304 | 0.01 | 0.36 | 1.78 |
| torch.bfloat16 | 4194304 | 0.01 | 0.36 | 1.78 |
| torch.float32 | 4194304 | 0.02 | 0.22 | 2.02 |
| torch.float16 | 10485760 | 0.02 | 0.45 | 2.24 |
| torch.bfloat16 | 10485760 | 0.02 | 0.45 | 2.23 |
| torch.float32 | 10485760 | 0.05 | 0.23 | 2.08 |
| torch.float16 | 20971520 | 0.05 | 0.44 | 2.18 |
| torch.bfloat16 | 20971520 | 0.05 | 0.44 | 2.18 |
| torch.float32 | 20971520 | 0.09 | 0.24 | 2.18 |
| torch.float8_e4m3fn | 4194304 | 0.03 | 0.14 | 0.41 |
| torch.float8_e5m2 | 4194304 | 0.03 | 0.14 | 0.42 |
| torch.float8_e4m3fn | 10485760 | 0.07 | 0.16 | 0.48 |
| torch.float8_e5m2 | 10485760 | 0.06 | 0.16 | 0.49 |
| torch.float8_e4m3fn | 20971520 | 0.14 | 0.15 | 0.46 |
| torch.float8_e5m2 | 20971520 | 0.14 | 0.16 | 0.47 |

## alibi

### tileops

| op_name | seq_len | dim | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| alibi | 512 | 64 | torch.float16 | 0.03 | 0.52 | 1.04 |
| alibi | 512 | 64 | torch.bfloat16 | 0.03 | 0.52 | 1.05 |
| alibi | 512 | 64 | torch.float32 | 0.02 | 0.89 | 3.57 |
| alibi | 2048 | 64 | torch.float16 | 0.27 | 1.00 | 2.01 |
| alibi | 2048 | 64 | torch.bfloat16 | 0.27 | 1.00 | 2.00 |
| alibi | 2048 | 64 | torch.float32 | 0.25 | 1.06 | 4.24 |
| alibi | 4096 | 128 | torch.float16 | 0.97 | 2.22 | 4.45 |
| alibi | 4096 | 128 | torch.bfloat16 | 0.84 | 2.54 | 5.09 |
| alibi | 4096 | 128 | torch.float32 | 1.15 | 1.87 | 7.47 |

### baseline

| op_name | seq_len | dim | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| alibi | 512 | 64 | torch.float16 | 0.08 | 0.21 | 0.41 |
| alibi | 512 | 64 | torch.bfloat16 | 0.08 | 0.20 | 0.41 |
| alibi | 512 | 64 | torch.float32 | 0.06 | 0.30 | 1.22 |
| alibi | 2048 | 64 | torch.float16 | 1.01 | 0.27 | 0.53 |
| alibi | 2048 | 64 | torch.bfloat16 | 1.01 | 0.27 | 0.53 |
| alibi | 2048 | 64 | torch.float32 | 0.66 | 0.41 | 1.63 |
| alibi | 4096 | 128 | torch.float16 | 7.72 | 0.28 | 0.56 |
| alibi | 4096 | 128 | torch.bfloat16 | 7.71 | 0.28 | 0.56 |
| alibi | 4096 | 128 | torch.float32 | 5.53 | 0.39 | 1.55 |

## sinusoidal

### tileops

| op_name | seq_len | dim | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| sinusoidal | 512 | 256 | torch.float16 | 0.00 | 0.04 | 0.09 |
| sinusoidal | 512 | 256 | torch.bfloat16 | 0.00 | 0.04 | 0.09 |
| sinusoidal | 512 | 256 | torch.float32 | 0.00 | 0.05 | 0.21 |
| sinusoidal | 2048 | 300 | torch.float16 | 0.00 | 0.15 | 0.29 |
| sinusoidal | 2048 | 300 | torch.bfloat16 | 0.00 | 0.15 | 0.29 |
| sinusoidal | 2048 | 300 | torch.float32 | 0.00 | 0.13 | 0.52 |
| sinusoidal | 4096 | 512 | torch.float16 | 0.01 | 0.30 | 0.61 |
| sinusoidal | 4096 | 512 | torch.bfloat16 | 0.01 | 0.30 | 0.61 |
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

## leaky_relu_fp8

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| leaky_relu | torch.float8_e4m3fn | 4194304 | 0.01 | 0.49 | 0.98 |
| leaky_relu | torch.float8_e5m2 | 4194304 | 0.03 | 0.16 | 0.32 |
| leaky_relu | torch.float8_e4m3fn | 10485760 | 0.02 | 0.56 | 1.13 |
| leaky_relu | torch.float8_e5m2 | 10485760 | 0.05 | 0.20 | 0.40 |
| leaky_relu | torch.float8_e4m3fn | 20971520 | 0.03 | 0.62 | 1.24 |
| leaky_relu | torch.float8_e5m2 | 20971520 | 0.10 | 0.22 | 0.43 |

### baseline

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| leaky_relu | torch.float8_e4m3fn | 4194304 | 0.03 | 0.16 | 0.33 |
| leaky_relu | torch.float8_e5m2 | 4194304 | 0.02 | 0.17 | 0.34 |
| leaky_relu | torch.float8_e4m3fn | 10485760 | 0.05 | 0.19 | 0.38 |
| leaky_relu | torch.float8_e5m2 | 10485760 | 0.05 | 0.20 | 0.40 |
| leaky_relu | torch.float8_e4m3fn | 20971520 | 0.11 | 0.19 | 0.38 |
| leaky_relu | torch.float8_e5m2 | 20971520 | 0.11 | 0.19 | 0.39 |

## elu_fp8

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| elu | torch.float8_e4m3fn | 4194304 | 0.01 | 0.66 | 1.31 |
| elu | torch.float8_e5m2 | 4194304 | 0.02 | 0.26 | 0.52 |
| elu | torch.float8_e4m3fn | 10485760 | 0.01 | 0.85 | 1.70 |
| elu | torch.float8_e5m2 | 10485760 | 0.03 | 0.31 | 0.62 |
| elu | torch.float8_e4m3fn | 20971520 | 0.02 | 0.95 | 1.91 |
| elu | torch.float8_e5m2 | 20971520 | 0.07 | 0.32 | 0.64 |

### baseline

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| elu | torch.float8_e4m3fn | 4194304 | 0.03 | 0.14 | 0.28 |
| elu | torch.float8_e5m2 | 4194304 | 0.03 | 0.14 | 0.29 |
| elu | torch.float8_e4m3fn | 10485760 | 0.06 | 0.16 | 0.33 |
| elu | torch.float8_e5m2 | 10485760 | 0.06 | 0.17 | 0.34 |
| elu | torch.float8_e4m3fn | 20971520 | 0.13 | 0.17 | 0.33 |
| elu | torch.float8_e5m2 | 20971520 | 0.12 | 0.17 | 0.34 |

## clamp_fp8

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| clamp | torch.float8_e4m3fn | 4194304 | 0.00 | 0.98 | 1.97 |
| clamp | torch.float8_e5m2 | 4194304 | 0.01 | 0.39 | 0.79 |
| clamp | torch.float8_e4m3fn | 10485760 | 0.01 | 1.33 | 2.65 |
| clamp | torch.float8_e5m2 | 10485760 | 0.02 | 0.49 | 0.99 |
| clamp | torch.float8_e4m3fn | 20971520 | 0.01 | 1.57 | 3.13 |
| clamp | torch.float8_e5m2 | 20971520 | 0.04 | 0.51 | 1.02 |

### baseline

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| clamp | torch.float8_e4m3fn | 4194304 | 0.03 | 0.16 | 0.32 |
| clamp | torch.float8_e5m2 | 4194304 | 0.03 | 0.16 | 0.33 |
| clamp | torch.float8_e4m3fn | 10485760 | 0.06 | 0.19 | 0.38 |
| clamp | torch.float8_e5m2 | 10485760 | 0.05 | 0.19 | 0.39 |
| clamp | torch.float8_e4m3fn | 20971520 | 0.11 | 0.19 | 0.37 |
| clamp | torch.float8_e5m2 | 20971520 | 0.11 | 0.19 | 0.38 |

## InstanceNormOp

### tileops

| n | c | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 8 | 128 | torch.float16 | 0.02 | 0.35 | 0.28 |
| 8 | 128 | torch.bfloat16 | 0.01 | 0.35 | 0.28 |
| 4 | 256 | torch.float16 | 0.02 | 0.22 | 0.18 |
| 4 | 64 | torch.float16 | 0.01 | 0.08 | 0.07 |

### baseline

| n | c | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 8 | 128 | torch.float16 | 0.02 | 0.25 | 0.20 |
| 8 | 128 | torch.bfloat16 | 0.02 | 0.25 | 0.20 |
| 4 | 256 | torch.float16 | 0.02 | 0.20 | 0.16 |
| 4 | 64 | torch.float16 | 0.01 | 0.09 | 0.07 |

## LayerNormOp

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.01 | 1.92 | 1.54 |
| 4096 | 4096 | torch.bfloat16 | 0.04 | 2.28 | 1.82 |
| 2048 | 5120 | torch.float16 | 0.03 | 2.04 | 1.63 |
| 1025 | 4096 | torch.float16 | 0.01 | 1.82 | 1.46 |

### baseline

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.01 | 1.64 | 1.32 |
| 4096 | 4096 | torch.bfloat16 | 0.03 | 2.58 | 2.06 |
| 2048 | 5120 | torch.float16 | 0.02 | 2.39 | 1.91 |
| 1025 | 4096 | torch.float16 | 0.01 | 1.64 | 1.32 |

## AnyOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | any | 0.01 | 0.40 | 0.80 |
| 1024 | 4096 | torch.bfloat16 | any | 0.01 | 0.40 | 0.80 |
| 1024 | 4096 | torch.float32 | any | 0.01 | 0.28 | 1.13 |
| 1024 | 4096 | torch.int32 | any | 0.03 | 0.16 | 0.62 |
| 4096 | 4096 | torch.float16 | any | 0.03 | 0.62 | 1.24 |

### baseline

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | any | 0.03 | 0.16 | 0.32 |
| 1024 | 4096 | torch.bfloat16 | any | 0.03 | 0.16 | 0.32 |
| 1024 | 4096 | torch.float32 | any | 0.03 | 0.16 | 0.62 |
| 1024 | 4096 | torch.int32 | any | 0.03 | 0.16 | 0.62 |
| 4096 | 4096 | torch.float16 | any | 0.07 | 0.25 | 0.51 |

## AllOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | all | 0.01 | 0.40 | 0.80 |
| 1024 | 4096 | torch.bfloat16 | all | 0.01 | 0.40 | 0.80 |
| 1024 | 4096 | torch.float32 | all | 0.01 | 0.28 | 1.13 |
| 1024 | 4096 | torch.int32 | all | 0.03 | 0.15 | 0.62 |
| 4096 | 4096 | torch.float16 | all | 0.03 | 0.62 | 1.25 |

### baseline

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | all | 0.03 | 0.16 | 0.33 |
| 1024 | 4096 | torch.bfloat16 | all | 0.03 | 0.16 | 0.33 |
| 1024 | 4096 | torch.float32 | all | 0.03 | 0.16 | 0.63 |
| 1024 | 4096 | torch.int32 | all | 0.03 | 0.16 | 0.63 |
| 4096 | 4096 | torch.float16 | all | 0.07 | 0.25 | 0.51 |

## CountNonzeroOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | count_nonzero | 0.01 | 0.53 | 1.07 |
| 1024 | 4096 | torch.bfloat16 | count_nonzero | 0.01 | 0.53 | 1.07 |
| 1024 | 4096 | torch.float32 | count_nonzero | 0.01 | 0.34 | 1.38 |
| 1024 | 4096 | torch.int32 | count_nonzero | 0.02 | 0.17 | 0.69 |
| 4096 | 4096 | torch.float16 | count_nonzero | 0.02 | 0.70 | 1.39 |

### baseline

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | count_nonzero | 0.03 | 0.12 | 0.25 |
| 1024 | 4096 | torch.bfloat16 | count_nonzero | 0.03 | 0.12 | 0.24 |
| 1024 | 4096 | torch.float32 | count_nonzero | 0.04 | 0.11 | 0.45 |
| 1024 | 4096 | torch.int32 | count_nonzero | 0.04 | 0.11 | 0.45 |
| 4096 | 4096 | torch.float16 | count_nonzero | 0.11 | 0.15 | 0.31 |

## MeanPoolingForwardOp

### tileops

| batch_size | seq_len | heads | dim | chunk_size | dtype | accum_dtype | chunks_per_bacth | seq_num | use_offsets | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8192 | 64 | 128 | 64 | torch.float16 | torch.float32 | 128 | 1 | 0 | 0.13 | 0.52 | 1.05 |
| 2 | 2048 | 64 | 128 | 64 | torch.float16 | torch.float32 | 32 | 2 | 0 | 0.07 | 0.47 | 0.96 |
| 1 | 8192 | 64 | 128 | 64 | torch.float16 | torch.float32 | 128 | 4 | 1 | 0.19 | 0.36 | 0.73 |
| 1 | 1000 | 64 | 128 | 32 | torch.float16 | torch.float32 | 34 | 4 | 1 | 0.02 | 0.39 | 0.75 |

### baseline

| batch_size | seq_len | heads | dim | chunk_size | dtype | accum_dtype | chunks_per_bacth | seq_num | use_offsets | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8192 | 64 | 128 | 64 | torch.float16 | torch.float32 | 128 | 1 | 0 | 0.77 | 0.09 | 0.18 |
| 2 | 2048 | 64 | 128 | 64 | torch.float16 | torch.float32 | 32 | 2 | 0 | 0.27 | 0.12 | 0.25 |
| 1 | 8192 | 64 | 128 | 64 | torch.float16 | torch.float32 | 128 | 4 | 1 | 0.80 | 0.08 | 0.17 |
| 1 | 1000 | 64 | 128 | 32 | torch.float16 | torch.float32 | 34 | 4 | 1 | 0.27 | 0.03 | 0.06 |

## MultiHeadAttentionFwdOp

### tileops

| batch | seq_len | heads | dim | causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 8 | 64 | False | torch.float16 | 0.01 | 206.08 | 0.40 |
| 16 | 2048 | 16 | 128 | False | torch.float16 | 0.81 | 674.98 | 0.66 |
| 4 | 4096 | 16 | 128 | False | torch.bfloat16 | 0.79 | 697.08 | 0.34 |

### torch

| batch | seq_len | heads | dim | causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 8 | 64 | False | torch.float16 | 0.02 | 116.48 | 0.23 |
| 16 | 2048 | 16 | 128 | False | torch.float16 | 1.20 | 457.23 | 0.45 |
| 4 | 4096 | 16 | 128 | False | torch.bfloat16 | 1.14 | 482.78 | 0.24 |

## MultiHeadAttentionBwdOp

### tileops

| batch | seq_len | heads | dim | causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 8 | 64 | False | torch.float16 | 0.04 | 120.45 | 0.16 |
| 16 | 2048 | 16 | 128 | False | torch.float16 | 3.01 | 457.15 | 0.31 |
| 4 | 4096 | 16 | 128 | False | torch.bfloat16 | 2.58 | 532.49 | 0.18 |

### torch

| batch | seq_len | heads | dim | causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 8 | 64 | False | torch.float16 | 0.07 | 79.82 | 0.11 |
| 16 | 2048 | 16 | 128 | False | torch.float16 | 3.42 | 402.08 | 0.27 |
| 4 | 4096 | 16 | 128 | False | torch.bfloat16 | 3.38 | 406.29 | 0.14 |

## MultiHeadAttentionDecodeWithKVCacheOp

### tileops

| b | h | s_q | s_kv | d | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 128 | 5 | 128 | torch.float16 | 0.01 | 1.69 | 0.35 |

### torch

| b | h | s_q | s_kv | d | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 128 | 5 | 128 | torch.float16 | 0.01 | 1.48 | 0.31 |

## MultiHeadAttentionDecodePagedWithKVCacheOp

### tileops

| batch | heads | seqlen_q | seqlen_kv | dim | page_size | is_causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 16 | 1 | 512 | 128 | 128 | False | torch.float16 | 0.02 | 0.18 | 0.18 |
| 2 | 8 | 1 | 1024 | 64 | 256 | False | torch.float16 | 0.02 | 0.18 | 0.09 |
| 1 | 8 | 1 | 1024 | 64 | 256 | False | torch.float16 | 0.02 | 0.09 | 0.09 |
| 1 | 8 | 1 | 512 | 64 | 256 | False | torch.float16 | 0.02 | 0.05 | 0.05 |

### baseline

| batch | heads | seqlen_q | seqlen_kv | dim | page_size | is_causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 16 | 1 | 512 | 128 | 128 | False | torch.float16 | 0.04 | 0.10 | 0.10 |
| 2 | 8 | 1 | 1024 | 64 | 256 | False | torch.float16 | 0.08 | 0.05 | 0.03 |
| 1 | 8 | 1 | 1024 | 64 | 256 | False | torch.float16 | 0.04 | 0.05 | 0.05 |
| 1 | 8 | 1 | 512 | 64 | 256 | False | torch.float16 | 0.03 | 0.03 | 0.03 |

## ManifoldConstrainedHyperConnectionPostOp

### tileops

| batch | n_expand | c_x | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | 1280 | torch.bfloat16 | 0.00 | 27.94 | 0.01 |
| 2 | 4 | 1920 | torch.bfloat16 | 0.00 | 123.27 | 0.01 |
| 4 | 4 | 2560 | torch.bfloat16 | 0.00 | 417.19 | 0.01 |

### baseline

| batch | n_expand | c_x | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | 1280 | torch.bfloat16 | 0.01 | 3.99 | 0.00 |
| 2 | 4 | 1920 | torch.bfloat16 | 0.01 | 16.77 | 0.00 |
| 4 | 4 | 2560 | torch.bfloat16 | 0.01 | 56.46 | 0.00 |

## ManifoldConstrainedHyperConnectionPreOp

### tileops

| batch | n_expand | c_x | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | 1280 | torch.bfloat16 | 0.04 | 30.58 | 0.00 |
| 2 | 4 | 1920 | torch.bfloat16 | 0.05 | 103.58 | 0.00 |
| 4 | 4 | 2560 | torch.bfloat16 | 0.07 | 292.12 | 0.00 |

### baseline

| batch | n_expand | c_x | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | 1280 | torch.bfloat16 | 0.27 | 4.59 | 0.00 |
| 2 | 4 | 1920 | torch.bfloat16 | 0.30 | 18.80 | 0.00 |
| 4 | 4 | 2560 | torch.bfloat16 | 0.34 | 60.01 | 0.00 |

## fused_topk

### tileops

| num_tokens | num_experts | top_k | scoring_func | renormalize | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 128 | 8 | softmax | False | torch.bfloat16 | 0.01 | 0.15 | 0.02 |
| 2048 | 128 | 8 | softmax | False | torch.bfloat16 | 0.01 | 0.58 | 0.08 |
| 4096 | 128 | 8 | softmax | False | torch.bfloat16 | 0.01 | 0.91 | 0.13 |
| 512 | 256 | 8 | softmax | True | torch.bfloat16 | 0.01 | 0.17 | 0.02 |
| 2048 | 256 | 8 | softmax | True | torch.bfloat16 | 0.01 | 0.64 | 0.08 |
| 4096 | 256 | 8 | softmax | True | torch.bfloat16 | 0.02 | 1.01 | 0.13 |
| 512 | 256 | 8 | sigmoid | True | torch.bfloat16 | 0.02 | 0.15 | 0.02 |
| 2048 | 256 | 8 | sigmoid | True | torch.bfloat16 | 0.02 | 0.58 | 0.07 |
| 4096 | 256 | 8 | sigmoid | True | torch.bfloat16 | 0.02 | 0.96 | 0.12 |

### pytorch-ref

| num_tokens | num_experts | top_k | scoring_func | renormalize | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 128 | 8 | softmax | False | torch.bfloat16 | 0.02 | 0.06 | 0.01 |
| 2048 | 128 | 8 | softmax | False | torch.bfloat16 | 0.03 | 0.16 | 0.02 |
| 4096 | 128 | 8 | softmax | False | torch.bfloat16 | 0.04 | 0.21 | 0.03 |
| 512 | 256 | 8 | softmax | True | torch.bfloat16 | 0.03 | 0.08 | 0.01 |
| 2048 | 256 | 8 | softmax | True | torch.bfloat16 | 0.05 | 0.18 | 0.02 |
| 4096 | 256 | 8 | softmax | True | torch.bfloat16 | 0.08 | 0.23 | 0.03 |
| 512 | 256 | 8 | sigmoid | True | torch.bfloat16 | 0.03 | 0.08 | 0.01 |
| 2048 | 256 | 8 | sigmoid | True | torch.bfloat16 | 0.05 | 0.17 | 0.02 |
| 4096 | 256 | 8 | sigmoid | True | torch.bfloat16 | 0.09 | 0.22 | 0.03 |

## moe_permute

### tileops

| total_tokens | top_k | num_experts | hidden_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 2 | 8 | 4096 | torch.bfloat16 | 0.01 | N/A | 1.54 |
| 2048 | 2 | 8 | 4096 | torch.bfloat16 | 0.02 | N/A | 2.35 |
| 4096 | 2 | 8 | 4096 | torch.bfloat16 | 0.04 | N/A | 2.40 |
| 512 | 8 | 256 | 7168 | torch.bfloat16 | 0.04 | N/A | 1.78 |
| 2048 | 8 | 256 | 7168 | torch.bfloat16 | 0.16 | N/A | 1.70 |
| 4096 | 8 | 256 | 7168 | torch.bfloat16 | 0.32 | N/A | 1.66 |
| 512 | 8 | 128 | 2048 | torch.bfloat16 | 0.02 | N/A | 0.99 |
| 2048 | 8 | 128 | 2048 | torch.bfloat16 | 0.06 | N/A | 1.17 |
| 4096 | 8 | 128 | 2048 | torch.bfloat16 | 0.13 | N/A | 1.17 |

### pytorch-ref

| total_tokens | top_k | num_experts | hidden_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 2 | 8 | 4096 | torch.bfloat16 | 0.01 | N/A | 0.86 |
| 2048 | 2 | 8 | 4096 | torch.bfloat16 | 0.04 | N/A | 1.43 |
| 4096 | 2 | 8 | 4096 | torch.bfloat16 | 0.07 | N/A | 1.46 |
| 512 | 8 | 256 | 7168 | torch.bfloat16 | 0.04 | N/A | 1.79 |
| 2048 | 8 | 256 | 7168 | torch.bfloat16 | 0.12 | N/A | 2.17 |
| 4096 | 8 | 256 | 7168 | torch.bfloat16 | 0.22 | N/A | 2.38 |
| 512 | 8 | 128 | 2048 | torch.bfloat16 | 0.03 | N/A | 0.73 |
| 2048 | 8 | 128 | 2048 | torch.bfloat16 | 0.08 | N/A | 0.94 |
| 4096 | 8 | 128 | 2048 | torch.bfloat16 | 0.12 | N/A | 1.29 |

## MoePermuteAlignOp

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

## moe_unpermute

### tileops

| total_tokens | top_k | hidden_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 512 | 2 | 4096 | torch.bfloat16 | 0.01 | 1.38 | 2.08 |
| 2048 | 2 | 4096 | torch.bfloat16 | 0.02 | 2.02 | 3.04 |
| 4096 | 2 | 4096 | torch.bfloat16 | 0.03 | 2.33 | 3.49 |
| 512 | 8 | 7168 | torch.bfloat16 | 0.02 | 2.37 | 2.66 |
| 2048 | 8 | 7168 | torch.bfloat16 | 0.07 | 3.28 | 3.69 |
| 4096 | 8 | 7168 | torch.bfloat16 | 0.13 | 3.61 | 4.07 |
| 512 | 8 | 2048 | torch.bfloat16 | 0.01 | 2.07 | 2.33 |
| 2048 | 8 | 2048 | torch.bfloat16 | 0.02 | 2.84 | 3.20 |
| 4096 | 8 | 2048 | torch.bfloat16 | 0.04 | 3.20 | 3.61 |

### pytorch-vec

| total_tokens | top_k | hidden_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 512 | 2 | 4096 | torch.bfloat16 | 0.06 | 0.15 | 0.22 |
| 2048 | 2 | 4096 | torch.bfloat16 | 0.20 | 0.17 | 0.25 |
| 4096 | 2 | 4096 | torch.bfloat16 | 0.38 | 0.18 | 0.27 |
| 512 | 8 | 7168 | torch.bfloat16 | 0.31 | 0.19 | 0.21 |
| 2048 | 8 | 7168 | torch.bfloat16 | 1.14 | 0.21 | 0.23 |
| 4096 | 8 | 7168 | torch.bfloat16 | 2.11 | 0.22 | 0.25 |
| 512 | 8 | 2048 | torch.bfloat16 | 0.10 | 0.17 | 0.19 |
| 2048 | 8 | 2048 | torch.bfloat16 | 0.36 | 0.19 | 0.21 |
| 4096 | 8 | 2048 | torch.bfloat16 | 0.68 | 0.20 | 0.22 |

## SumOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | sum | 0.01 | 0.53 | 1.06 |
| 1024 | 4096 | torch.bfloat16 | sum | 0.01 | 0.54 | 1.08 |
| 4096 | 4096 | torch.float16 | sum | 0.02 | 0.70 | 1.40 |

### baseline

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | sum | 0.03 | 0.16 | 0.32 |
| 1024 | 4096 | torch.bfloat16 | sum | 0.03 | 0.16 | 0.32 |
| 4096 | 4096 | torch.float16 | sum | 0.08 | 0.21 | 0.43 |

## MeanOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | mean | 0.01 | 0.53 | 1.07 |

### baseline

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | mean | 0.03 | 0.16 | 0.32 |

## AmaxOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | amax | 0.01 | 0.54 | 1.08 |

### baseline

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | amax | 0.03 | 0.16 | 0.32 |

## AminOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | amin | 0.01 | 0.54 | 1.08 |

### baseline

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | amin | 0.03 | 0.16 | 0.32 |

## ProdOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | prod | 0.02 | 0.24 | 0.48 |

### baseline

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | prod | 0.03 | 0.16 | 0.32 |

## StdOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | std | 0.01 | 1.39 | 0.93 |

### baseline

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | std | 0.04 | 0.33 | 0.22 |

## VarOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | var | 0.01 | 1.39 | 0.93 |

### baseline

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | var | 0.04 | 0.33 | 0.22 |

## VarMeanOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | var_mean | 0.01 | 1.51 | 1.01 |

### baseline

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | var_mean | 0.05 | 0.26 | 0.17 |

## RmsNormOp

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.01 | 1.72 | 1.72 |
| 4096 | 4096 | torch.bfloat16 | 0.03 | 2.10 | 2.10 |
| 2048 | 5120 | torch.float16 | 0.02 | 1.99 | 1.99 |
| 1025 | 4096 | torch.float16 | 0.01 | 1.67 | 1.67 |

### baseline

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.07 | 0.24 | 0.24 |
| 4096 | 4096 | torch.bfloat16 | 0.25 | 0.27 | 0.27 |
| 2048 | 5120 | torch.float16 | 0.17 | 0.24 | 0.24 |
| 1025 | 4096 | torch.float16 | 0.07 | 0.24 | 0.24 |

## RopeNeoxOp

### tileops

| seq_len | head_dim | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 2048 | 64 | torch.float16 | 0.00 | 0.24 | 0.36 |
| 2048 | 64 | torch.bfloat16 | 0.00 | 0.24 | 0.36 |
| 2048 | 64 | torch.float32 | 0.00 | 0.21 | 0.64 |
| 2048 | 128 | torch.float16 | 0.00 | 0.41 | 0.61 |
| 2048 | 128 | torch.bfloat16 | 0.00 | 0.41 | 0.61 |
| 2048 | 128 | torch.float32 | 0.00 | 0.34 | 1.01 |
| 4096 | 128 | torch.float16 | 0.00 | 0.67 | 1.00 |
| 4096 | 128 | torch.bfloat16 | 0.00 | 0.67 | 1.00 |
| 4096 | 128 | torch.float32 | 0.00 | 0.54 | 1.62 |

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
| 4096 | 128 | torch.float32 | 0.02 | 0.12 | 0.37 |

## SoftmaxOp

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.02 | 1.05 | 1.05 |
| 4096 | 4096 | torch.bfloat16 | 0.06 | 1.17 | 1.17 |
| 1024 | 3000 | torch.float16 | 0.02 | 0.50 | 0.50 |
| 1025 | 4096 | torch.float16 | 0.02 | 1.05 | 1.05 |

### baseline

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.02 | 0.97 | 0.97 |
| 4096 | 4096 | torch.bfloat16 | 0.06 | 1.14 | 1.14 |
| 1024 | 3000 | torch.float16 | 0.01 | 0.84 | 0.84 |
| 1025 | 4096 | torch.float16 | 0.02 | 0.97 | 0.97 |

## LogSoftmaxOp

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.02 | 1.01 | 0.81 |
| 4096 | 4096 | torch.bfloat16 | 0.08 | 1.10 | 0.88 |
| 1024 | 3000 | torch.float16 | 0.03 | 0.54 | 0.43 |
| 1025 | 4096 | torch.float16 | 0.02 | 1.01 | 0.81 |

### baseline

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.02 | 1.38 | 1.10 |
| 4096 | 4096 | torch.bfloat16 | 0.05 | 1.61 | 1.29 |
| 1024 | 3000 | torch.float16 | 0.01 | 1.16 | 0.93 |
| 1025 | 4096 | torch.float16 | 0.02 | 1.38 | 1.10 |

## LogSumExpOp

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.01 | 1.29 | 0.86 |
| 4096 | 4096 | torch.bfloat16 | 0.03 | 1.59 | 1.06 |
| 1024 | 3000 | torch.float16 | 0.02 | 0.57 | 0.38 |
| 1025 | 4096 | torch.float16 | 0.01 | 1.29 | 0.86 |

### baseline

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.05 | 0.25 | 0.17 |
| 4096 | 4096 | torch.bfloat16 | 0.10 | 0.50 | 0.33 |
| 1024 | 3000 | torch.float16 | 0.04 | 0.21 | 0.14 |
| 1025 | 4096 | torch.float16 | 0.05 | 0.24 | 0.16 |

## TopkSelectorOp

### tileops

| batch | seq_len | seq_len_kv | kv_group | topk | in_dtype | out_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32768 | 65536 | 1 | 1024 | torch.float32 | torch.int32 | 13.53 | N/A | 0.64 |
| 1 | 32768 | 65536 | 1 | 2048 | torch.float32 | torch.int32 | 14.00 | N/A | 0.63 |
| 1 | 65535 | 131072 | 1 | 1024 | torch.float32 | torch.int32 | 44.56 | N/A | 0.78 |
| 1 | 65535 | 131072 | 1 | 2048 | torch.float32 | torch.int32 | 44.97 | N/A | 0.78 |

### baseline

| batch | seq_len | seq_len_kv | kv_group | topk | in_dtype | out_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32768 | 65536 | 1 | 1024 | torch.float32 | torch.int32 | 14.47 | N/A | 0.60 |
| 1 | 32768 | 65536 | 1 | 2048 | torch.float32 | torch.int32 | 7.11 | N/A | 1.25 |
| 1 | 65535 | 131072 | 1 | 1024 | torch.float32 | torch.int32 | 109.43 | N/A | 0.32 |
| 1 | 65535 | 131072 | 1 | 2048 | torch.float32 | torch.int32 | 112.13 | N/A | 0.31 |

## exp

### tileops

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| exp | 262144 | torch.float16 | torch.float16 | 0.00 | 0.11 | 0.46 |
| exp | 1048576 | torch.float16 | torch.float16 | 0.00 | 0.33 | 1.31 |
| exp | 4000000 | torch.float16 | torch.float16 | 0.01 | 0.63 | 2.51 |
| exp | 262144 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.11 | 0.46 |
| exp | 1048576 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.33 | 1.31 |
| exp | 4000000 | torch.bfloat16 | torch.bfloat16 | 0.01 | 0.62 | 2.49 |

### baseline

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| exp | 262144 | torch.float16 | torch.float16 | 0.00 | 0.11 | 0.44 |
| exp | 1048576 | torch.float16 | torch.float16 | 0.00 | 0.32 | 1.28 |
| exp | 4000000 | torch.float16 | torch.float16 | 0.01 | 0.63 | 2.53 |
| exp | 262144 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.11 | 0.44 |
| exp | 1048576 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.32 | 1.28 |
| exp | 4000000 | torch.bfloat16 | torch.bfloat16 | 0.01 | 0.63 | 2.52 |

## gelu

### tileops

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| gelu | 262144 | torch.float16 | torch.float16 | 0.00 | 0.11 | 0.44 |
| gelu | 1048576 | torch.float16 | torch.float16 | 0.00 | 0.29 | 1.18 |
| gelu | 4000000 | torch.float16 | torch.float16 | 0.01 | 0.48 | 1.90 |
| gelu | 262144 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.11 | 0.44 |
| gelu | 1048576 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.29 | 1.17 |
| gelu | 4000000 | torch.bfloat16 | torch.bfloat16 | 0.01 | 0.46 | 1.84 |

### baseline

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| gelu | 262144 | torch.float16 | torch.float16 | 0.00 | 0.11 | 0.42 |
| gelu | 1048576 | torch.float16 | torch.float16 | 0.00 | 0.29 | 1.15 |
| gelu | 4000000 | torch.float16 | torch.float16 | 0.01 | 0.51 | 2.03 |
| gelu | 262144 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.10 | 0.41 |
| gelu | 1048576 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.28 | 1.13 |
| gelu | 4000000 | torch.bfloat16 | torch.bfloat16 | 0.01 | 0.50 | 2.02 |

## logical_not

### tileops

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| logical_not | 262144 | torch.float16 | torch.bool | 0.00 | 0.11 | 0.32 |
| logical_not | 1048576 | torch.float16 | torch.bool | 0.00 | 0.21 | 0.64 |
| logical_not | 4000000 | torch.float16 | torch.bool | 0.01 | 0.30 | 0.91 |

### baseline

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| logical_not | 262144 | torch.float16 | torch.bool | 0.00 | 0.11 | 0.34 |
| logical_not | 1048576 | torch.float16 | torch.bool | 0.00 | 0.34 | 1.01 |
| logical_not | 4000000 | torch.float16 | torch.bool | 0.01 | 0.72 | 2.17 |

## bitwise_not

### tileops

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| bitwise_not | 262144 | torch.int32 | torch.int32 | 0.00 | 0.10 | 0.77 |
| bitwise_not | 1048576 | torch.int32 | torch.int32 | 0.01 | 0.19 | 1.56 |
| bitwise_not | 4000000 | torch.int32 | torch.int32 | 0.02 | 0.26 | 2.12 |

### baseline

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| bitwise_not | 262144 | torch.int32 | torch.int32 | 0.00 | 0.10 | 0.83 |
| bitwise_not | 1048576 | torch.int32 | torch.int32 | 0.00 | 0.24 | 1.95 |
| bitwise_not | 4000000 | torch.int32 | torch.int32 | 0.01 | 0.39 | 3.15 |

## isnan

### tileops

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| isnan | 262144 | torch.float16 | torch.bool | 0.00 | 0.11 | 0.32 |
| isnan | 1048576 | torch.float16 | torch.bool | 0.00 | 0.21 | 0.64 |
| isnan | 4000000 | torch.float16 | torch.bool | 0.01 | 0.30 | 0.91 |

### baseline

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| isnan | 262144 | torch.float16 | torch.bool | 0.00 | 0.11 | 0.34 |
| isnan | 1048576 | torch.float16 | torch.bool | 0.00 | 0.34 | 1.02 |
| isnan | 4000000 | torch.float16 | torch.bool | 0.01 | 0.73 | 2.18 |

## unary_strategy

### relu_direct

| n_total | shape_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4194304 | 1024x4096 | torch.float16 | direct | 0.01 | 0.29 | 1.17 |
| 4194304 | 1024x4096 | torch.bfloat16 | direct | 0.01 | 0.29 | 1.17 |
| 4194304 | 1024x4096 | torch.float32 | direct | 0.02 | 0.27 | 2.13 |
| 10485760 | 1024x10240 | torch.float16 | direct | 0.03 | 0.32 | 1.28 |
| 10485760 | 1024x10240 | torch.bfloat16 | direct | 0.03 | 0.32 | 1.28 |
| 10485760 | 1024x10240 | torch.float32 | direct | 0.04 | 0.29 | 2.33 |
| 20971520 | 1024x20480 | torch.float16 | direct | 0.06 | 0.34 | 1.34 |
| 20971520 | 1024x20480 | torch.bfloat16 | direct | 0.06 | 0.34 | 1.34 |
| 20971520 | 1024x20480 | torch.float32 | direct | 0.07 | 0.31 | 2.44 |

### relu_explicit_parallel

| n_total | shape_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4194304 | 1024x4096 | torch.float16 | explicit_parallel | 0.01 | 0.61 | 2.45 |
| 4194304 | 1024x4096 | torch.bfloat16 | explicit_parallel | 0.01 | 0.61 | 2.45 |
| 4194304 | 1024x4096 | torch.float32 | explicit_parallel | 0.01 | 0.39 | 3.09 |
| 10485760 | 1024x10240 | torch.float16 | explicit_parallel | 0.01 | 0.76 | 3.03 |
| 10485760 | 1024x10240 | torch.bfloat16 | explicit_parallel | 0.01 | 0.76 | 3.03 |
| 10485760 | 1024x10240 | torch.float32 | explicit_parallel | 0.02 | 0.47 | 3.74 |
| 20971520 | 1024x20480 | torch.float16 | explicit_parallel | 0.03 | 0.83 | 3.33 |
| 20971520 | 1024x20480 | torch.bfloat16 | explicit_parallel | 0.03 | 0.83 | 3.34 |
| 20971520 | 1024x20480 | torch.float32 | explicit_parallel | 0.04 | 0.51 | 4.04 |

### relu_register_copy

| n_total | shape_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4194304 | 1024x4096 | torch.float16 | register_copy | 0.01 | 0.66 | 2.66 |
| 4194304 | 1024x4096 | torch.bfloat16 | register_copy | 0.01 | 0.66 | 2.66 |
| 4194304 | 1024x4096 | torch.float32 | register_copy | 0.01 | 0.40 | 3.18 |
| 10485760 | 1024x10240 | torch.float16 | register_copy | 0.01 | 0.85 | 3.39 |
| 10485760 | 1024x10240 | torch.bfloat16 | register_copy | 0.01 | 0.85 | 3.40 |
| 10485760 | 1024x10240 | torch.float32 | register_copy | 0.02 | 0.48 | 3.81 |
| 20971520 | 1024x20480 | torch.float16 | register_copy | 0.02 | 0.95 | 3.82 |
| 20971520 | 1024x20480 | torch.bfloat16 | register_copy | 0.02 | 0.96 | 3.82 |
| 20971520 | 1024x20480 | torch.float32 | register_copy | 0.04 | 0.52 | 4.14 |

## L1NormOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | l1 | 0.01 | 0.53 | 1.06 |
| 1024 | 4096 | torch.bfloat16 | l1 | 0.01 | 0.54 | 1.08 |
| 1024 | 4096 | torch.float32 | l1 | 0.01 | 0.35 | 1.38 |
| 4096 | 4096 | torch.float16 | l1 | 0.02 | 0.70 | 1.40 |

### baseline

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | l1 | 0.02 | 0.25 | 0.50 |
| 1024 | 4096 | torch.bfloat16 | l1 | 0.02 | 0.25 | 0.50 |
| 1024 | 4096 | torch.float32 | l1 | 0.02 | 0.22 | 0.88 |
| 4096 | 4096 | torch.float16 | l1 | 0.02 | 0.76 | 1.53 |

## L2NormOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | l2 | 0.01 | 0.53 | 1.05 |
| 1024 | 4096 | torch.bfloat16 | l2 | 0.01 | 0.53 | 1.06 |
| 1024 | 4096 | torch.float32 | l2 | 0.01 | 0.34 | 1.35 |
| 4096 | 4096 | torch.float16 | l2 | 0.02 | 0.69 | 1.38 |

### baseline

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | l2 | 0.02 | 0.25 | 0.49 |
| 1024 | 4096 | torch.bfloat16 | l2 | 0.02 | 0.25 | 0.50 |
| 1024 | 4096 | torch.float32 | l2 | 0.02 | 0.22 | 0.88 |
| 4096 | 4096 | torch.float16 | l2 | 0.02 | 0.76 | 1.52 |

## InfNormOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | inf | 0.03 | 0.15 | 0.30 |
| 1024 | 4096 | torch.bfloat16 | inf | 0.03 | 0.15 | 0.30 |
| 1024 | 4096 | torch.float32 | inf | 0.03 | 0.12 | 0.50 |
| 4096 | 4096 | torch.float16 | inf | 0.06 | 0.30 | 0.59 |

### baseline

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | inf | 0.02 | 0.24 | 0.49 |
| 1024 | 4096 | torch.bfloat16 | inf | 0.02 | 0.24 | 0.49 |
| 1024 | 4096 | torch.float32 | inf | 0.02 | 0.22 | 0.87 |
| 4096 | 4096 | torch.float16 | inf | 0.02 | 0.76 | 1.51 |

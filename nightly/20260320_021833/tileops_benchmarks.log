........................................................................ [  8%]
........................................................................ [ 17%]
........................................................................ [ 26%]
........................................................................ [ 35%]
........................................................................ [ 44%]
........................................................................ [ 53%]
........................................................................ [ 62%]
........................................................................ [ 71%]
........................................................................ [ 80%]
....................................FF.................................. [ 89%]
........................................................................ [ 98%]
.............                                                            [100%]Benchmark report saved to profile_run.log

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
benchmarks/benchmark.py:140: in profile
    latency = do_bench(bench_fn, warmup=warmup, rep=rep, backend='cupti')
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
../../_tool/tileops_benchmarks_venv_db4a9af2f48857c5/lib/python3.11/site-packages/tilelang/profiler/bench.py:100: in do_bench
    fn()
benchmarks/benchmark.py:133: in bench_fn
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
tileops/kernels/flash_decode/mha_decode.py:446: in forward
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
tileops/kernels/flash_decode/mha_decode.py:334: in _mha_decode_wrapped_kernel
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
2026-03-19 10:14:43,666 INFO:Auto-tuning with 0.9 CPU utilizations, 240 CPUs available, 216 CPUs will be used
2026-03-19 10:14:52  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[-1]`
2026-03-19 10:14:53  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_no_split` with `out_idx=[-1]`
2026-03-19 10:14:53  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[-1]`
2026-03-19 10:14:53  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[-1]`
2026-03-19 10:14:53  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[-1]`
2026-03-19 10:14:53  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_no_split` with `out_idx=[-1]`
2026-03-19 10:14:53  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[-1]`
2026-03-19 10:14:53  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_no_split` with `out_idx=[-1]`
2026-03-19 10:14:53  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[-1]`
2026-03-19 10:14:53  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_no_split` with `out_idx=[-1]`
2026-03-19 10:14:53  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_no_split` with `out_idx=[-1]`
2026-03-19 10:14:53  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[-1]`
2026-03-19 10:14:53  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[-1]`
2026-03-19 10:14:53  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[-1]`
2026-03-19 10:14:53  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[-1]`
2026-03-19 10:14:53  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_no_split` with `out_idx=[-1]`
2026-03-19 10:14:53  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[-1]`
2026-03-19 10:14:53  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[-1]`
2026-03-19 10:14:53  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_no_split` with `out_idx=[-1]`
2026-03-19 10:14:53  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[-1]`
2026-03-19 10:14:53  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_no_split` with `out_idx=[-1]`
2026-03-19 10:14:53  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[-1]`
2026-03-19 10:14:53  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[-1]`
2026-03-19 10:14:53  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[-1]`
2026-03-19 10:14:53  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[-1]`
2026-03-19 10:14:53  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[-1]`
2026-03-19 10:14:53  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[-1]`
2026-03-19 10:14:53  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[-1]`
2026-03-19 10:14:53  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[-1]`
2026-03-19 10:14:53  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_no_split` with `out_idx=[-1]`
2026-03-19 10:14:53  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_no_split` with `out_idx=[-1]`
2026-03-19 10:14:53  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_no_split` with `out_idx=[-1]`
2026-03-19 10:14:53  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[-1]`
2026-03-19 10:14:53  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[-1]`
2026-03-19 10:14:53  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[-1]`
2026-03-19 10:14:53  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_no_split` with `out_idx=[-1]`
2026-03-19 10:14:53  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_no_split` with `out_idx=[-1]`
2026-03-19 10:14:53  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_no_split` with `out_idx=[-1]`
2026-03-19 10:14:53  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_no_split` with `out_idx=[-1]`
2026-03-19 10:14:53  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[-1]`
2026-03-19 10:14:53  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_no_split` with `out_idx=[-1]`
2026-03-19 10:14:53  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[-1]`
2026-03-19 10:14:53  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[-1]`
2026-03-19 10:14:53  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[-1]`
2026-03-19 10:14:53  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[-1]`
2026-03-19 10:14:53  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[-1]`
2026-03-19 10:14:53  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[-1]`
2026-03-19 10:14:53  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[-1]`
2026-03-19 10:15:06  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_no_split`
2026-03-19 10:15:07  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_no_split`
2026-03-19 10:15:07  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_no_split`
2026-03-19 10:15:08  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_no_split`
2026-03-19 10:15:08  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_no_split`
2026-03-19 10:15:08  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_no_split`
2026-03-19 10:15:08  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_no_split`
2026-03-19 10:15:08  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_no_split`
2026-03-19 10:15:08  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_no_split`
2026-03-19 10:15:08  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_no_split`
2026-03-19 10:15:08  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_no_split`
2026-03-19 10:15:11  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_no_split`
2026-03-19 10:15:26  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-19 10:15:27  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-19 10:15:27  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-19 10:15:27  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-19 10:15:27  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-19 10:15:27  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-19 10:15:27  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-19 10:15:27  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-19 10:15:27  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-19 10:15:27  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-19 10:15:27  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-19 10:15:27  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-19 10:15:27  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-19 10:15:27  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-19 10:15:27  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-19 10:15:28  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-19 10:15:28  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-19 10:15:28  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-19 10:15:28  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-19 10:15:28  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-19 10:15:28  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-19 10:15:28  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-19 10:15:28  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-19 10:15:28  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-19 10:15:28  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-19 10:15:28  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-19 10:15:28  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-19 10:15:31  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-19 10:15:31  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-19 10:15:31  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-19 10:15:31  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-19 10:15:31  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
Tuned Latency 0.10725293308496475 with config {'block_M': 64, 'block_N': 64, 'num_split': 1, 'num_stages': 3, 'threads': 128} at index 0
Tuned Latency 0.10970963537693024 with config {'block_M': 64, 'block_N': 64, 'num_split': 1, 'num_stages': 2, 'threads': 128} at index 1
Tuned Latency 0.14471691846847534 with config {'block_M': 128, 'block_N': 64, 'num_split': 1, 'num_stages': 2, 'threads': 256} at index 2
Tuned Latency 0.1785527914762497 with config {'block_M': 128, 'block_N': 64, 'num_split': 1, 'num_stages': 3, 'threads': 128} at index 3
Tuned Latency 0.09116499871015549 with config {'block_M': 64, 'block_N': 128, 'num_split': 1, 'num_stages': 3, 'threads': 128} at index 4
Tuned Latency 0.09117729216814041 with config {'block_M': 64, 'block_N': 128, 'num_split': 1, 'num_stages': 2, 'threads': 128} at index 5
Tuned Latency 0.12609630823135376 with config {'block_M': 128, 'block_N': 128, 'num_split': 1, 'num_stages': 3, 'threads': 256} at index 6
Tuned Latency 0.15650762617588043 with config {'block_M': 128, 'block_N': 64, 'num_split': 1, 'num_stages': 3, 'threads': 256} at index 7
Tuned Latency 0.12242461740970612 with config {'block_M': 128, 'block_N': 128, 'num_split': 1, 'num_stages': 2, 'threads': 256} at index 8
Tuned Latency 0.17724722623825073 with config {'block_M': 128, 'block_N': 64, 'num_split': 1, 'num_stages': 2, 'threads': 128} at index 9
Tuned Latency 0.2874411642551422 with config {'block_M': 128, 'block_N': 128, 'num_split': 1, 'num_stages': 2, 'threads': 128} at index 10
Tuned Latency 0.2918720245361328 with config {'block_M': 128, 'block_N': 128, 'num_split': 1, 'num_stages': 3, 'threads': 128} at index 11
Tuned Latency 0.012287496589124203 with config {'block_M': 64, 'block_N': 128, 'num_split': 2, 'num_stages': 2, 'threads': 128} at index 12
Tuned Latency 0.017774643376469612 with config {'block_M': 128, 'block_N': 64, 'num_split': 4, 'num_stages': 3, 'threads': 128} at index 13
Tuned Latency 0.018077826127409935 with config {'block_M': 128, 'block_N': 64, 'num_split': 4, 'num_stages': 3, 'threads': 256} at index 14
Tuned Latency 0.018897777423262596 with config {'block_M': 64, 'block_N': 64, 'num_split': 4, 'num_stages': 3, 'threads': 256} at index 15
Tuned Latency 0.01773606427013874 with config {'block_M': 64, 'block_N': 64, 'num_split': 4, 'num_stages': 3, 'threads': 128} at index 16
Tuned Latency 0.01788966916501522 with config {'block_M': 128, 'block_N': 64, 'num_split': 4, 'num_stages': 2, 'threads': 256} at index 17
Tuned Latency 0.014307227917015553 with config {'block_M': 128, 'block_N': 128, 'num_split': 2, 'num_stages': 3, 'threads': 256} at index 18
Tuned Latency 0.016107017174363136 with config {'block_M': 128, 'block_N': 64, 'num_split': 2, 'num_stages': 3, 'threads': 128} at index 19
Tuned Latency 0.01835678704082966 with config {'block_M': 64, 'block_N': 128, 'num_split': 4, 'num_stages': 2, 'threads': 256} at index 20
Tuned Latency 0.01781700924038887 with config {'block_M': 128, 'block_N': 64, 'num_split': 4, 'num_stages': 2, 'threads': 128} at index 21
Tuned Latency 0.015832973644137383 with config {'block_M': 64, 'block_N': 128, 'num_split': 4, 'num_stages': 3, 'threads': 128} at index 22
Tuned Latency 0.016963796690106392 with config {'block_M': 64, 'block_N': 128, 'num_split': 4, 'num_stages': 2, 'threads': 128} at index 23
Tuned Latency 0.014433289878070354 with config {'block_M': 64, 'block_N': 64, 'num_split': 2, 'num_stages': 2, 'threads': 128} at index 24
Tuned Latency 0.017352625727653503 with config {'block_M': 64, 'block_N': 64, 'num_split': 4, 'num_stages': 2, 'threads': 256} at index 25
Tuned Latency 0.023499637842178345 with config {'block_M': 128, 'block_N': 128, 'num_split': 4, 'num_stages': 2, 'threads': 128} at index 26
Tuned Latency 0.01209238264709711 with config {'block_M': 64, 'block_N': 128, 'num_split': 2, 'num_stages': 3, 'threads': 128} at index 27
Tuned Latency 0.016207603737711906 with config {'block_M': 128, 'block_N': 64, 'num_split': 2, 'num_stages': 2, 'threads': 256} at index 28
Tuned Latency 0.018021423369646072 with config {'block_M': 64, 'block_N': 128, 'num_split': 4, 'num_stages': 3, 'threads': 256} at index 29
Tuned Latency 0.01228850707411766 with config {'block_M': 64, 'block_N': 128, 'num_split': 2, 'num_stages': 3, 'threads': 256} at index 30
Tuned Latency 0.016123075038194656 with config {'block_M': 128, 'block_N': 64, 'num_split': 2, 'num_stages': 2, 'threads': 128} at index 31
Tuned Latency 0.021362125873565674 with config {'block_M': 128, 'block_N': 128, 'num_split': 2, 'num_stages': 2, 'threads': 128} at index 32
Tuned Latency 0.01716799847781658 with config {'block_M': 64, 'block_N': 64, 'num_split': 4, 'num_stages': 2, 'threads': 128} at index 33
Tuned Latency 0.024298567324876785 with config {'block_M': 128, 'block_N': 128, 'num_split': 4, 'num_stages': 3, 'threads': 128} at index 34
Tuned Latency 0.014237378723919392 with config {'block_M': 128, 'block_N': 128, 'num_split': 2, 'num_stages': 3, 'threads': 128} at index 35
Tuned Latency 0.024191128090023994 with config {'block_M': 128, 'block_N': 128, 'num_split': 4, 'num_stages': 2, 'threads': 256} at index 36
Tuned Latency 0.0242000725120306 with config {'block_M': 128, 'block_N': 128, 'num_split': 4, 'num_stages': 3, 'threads': 256} at index 37
Tuned Latency 0.014361652545630932 with config {'block_M': 128, 'block_N': 128, 'num_split': 2, 'num_stages': 2, 'threads': 256} at index 38
Tuned Latency 0.01217725034803152 with config {'block_M': 64, 'block_N': 64, 'num_split': 2, 'num_stages': 3, 'threads': 256} at index 39
Tuned Latency 0.015094968490302563 with config {'block_M': 64, 'block_N': 128, 'num_split': 2, 'num_stages': 2, 'threads': 256} at index 40
Tuned Latency 0.014093568548560143 with config {'block_M': 64, 'block_N': 64, 'num_split': 2, 'num_stages': 2, 'threads': 256} at index 41
Tuned Latency 0.013360507786273956 with config {'block_M': 128, 'block_N': 64, 'num_split': 2, 'num_stages': 3, 'threads': 256} at index 42
Tuned Latency 0.0121652502566576 with config {'block_M': 64, 'block_N': 64, 'num_split': 2, 'num_stages': 3, 'threads': 128} at index 43
Best config: {'block_M': 64, 'block_N': 128, 'num_split': 2, 'num_stages': 3, 'threads': 128}
mha_decode_kernel initialized with config: {'block_M': 64, 'block_N': 128, 'num_split': 2, 'num_stages': 3, 'threads': 128}
----------------------------- Captured stderr call -----------------------------
Compiling configurations:   0%|          | 0/48 [00:00<?, ?it/s]                                                                Compiling configurations:   0%|          | 0/48 [00:05<?, ?it/s]                                                                Compiling configurations:   0%|          | 0/48 [00:06<?, ?it/s]                                                                Compiling configurations:   0%|          | 0/48 [00:06<?, ?it/s]                                                                Compiling configurations:   0%|          | 0/48 [00:06<?, ?it/s]                                                                Compiling configurations:   0%|          | 0/48 [00:06<?, ?it/s]                                                                Compiling configurations:   0%|          | 0/48 [00:06<?, ?it/s]                                                                Compiling configurations:   0%|          | 0/48 [00:06<?, ?it/s]                                                                Compiling configurations:   0%|          | 0/48 [00:06<?, ?it/s]                                                                Compiling configurations:   0%|          | 0/48 [00:06<?, ?it/s]                                                                Compiling configurations:   0%|          | 0/48 [00:06<?, ?it/s]                                                                Compiling configurations:   0%|          | 0/48 [00:06<?, ?it/s]                                                                Compiling configurations:   0%|          | 0/48 [00:06<?, ?it/s]                                                                Compiling configurations:   0%|          | 0/48 [00:06<?, ?it/s]                                                                Compiling configurations:   0%|          | 0/48 [00:06<?, ?it/s]                                                                Compiling configurations:   0%|          | 0/48 [00:06<?, ?it/s]                                                                Compiling configurations:   0%|          | 0/48 [00:06<?, ?it/s]                                                                Compiling configurations:   0%|          | 0/48 [00:06<?, ?it/s]                                                                Compiling configurations:   0%|          | 0/48 [00:06<?, ?it/s]                                                                Compiling configurations:   0%|          | 0/48 [00:06<?, ?it/s]                                                                Compiling configurations:   0%|          | 0/48 [00:06<?, ?it/s]                                                                Compiling configurations:   0%|          | 0/48 [00:06<?, ?it/s]                                                                Compiling configurations:   0%|          | 0/48 [00:06<?, ?it/s]                                                                Compiling configurations:   0%|          | 0/48 [00:06<?, ?it/s]                                                                Compiling configurations:   0%|          | 0/48 [00:06<?, ?it/s]                                                                Compiling configurations:   0%|          | 0/48 [00:06<?, ?it/s]                                                                Compiling configurations:   0%|          | 0/48 [00:06<?, ?it/s]                                                                Compiling configurations:   0%|          | 0/48 [00:06<?, ?it/s]                                                                Compiling configurations:   0%|          | 0/48 [00:06<?, ?it/s]                                                                Compiling configurations:   0%|          | 0/48 [00:06<?, ?it/s]                                                                Compiling configurations:   0%|          | 0/48 [00:06<?, ?it/s]                                                                Compiling configurations:   0%|          | 0/48 [00:06<?, ?it/s]                                                                Compiling configurations:   0%|          | 0/48 [00:06<?, ?it/s]                                                                Compiling configurations:   0%|          | 0/48 [00:06<?, ?it/s]                                                                Compiling configurations:   0%|          | 0/48 [00:06<?, ?it/s]                                                                Compiling configurations:   0%|          | 0/48 [00:06<?, ?it/s]                                                                Compiling configurations:   0%|          | 0/48 [00:06<?, ?it/s]                                                                Compiling configurations:   0%|          | 0/48 [00:06<?, ?it/s]                                                                Compiling configurations:   0%|          | 0/48 [00:06<?, ?it/s]                                                                Compiling configurations:   0%|          | 0/48 [00:06<?, ?it/s]                                                                Compiling configurations:   0%|          | 0/48 [00:06<?, ?it/s]                                                                Compiling configurations:   0%|          | 0/48 [00:06<?, ?it/s]                                                                Compiling configurations:   0%|          | 0/48 [00:06<?, ?it/s]                                                                Compiling configurations:   0%|          | 0/48 [00:06<?, ?it/s]                                                                Compiling configurations:   0%|          | 0/48 [00:06<?, ?it/s]                                                                Compiling configurations:   0%|          | 0/48 [00:06<?, ?it/s]                                                                Compiling configurations:   0%|          | 0/48 [00:06<?, ?it/s]                                                                Compiling configurations:   0%|          | 0/48 [00:06<?, ?it/s]                                                                Compiling configurations:   0%|          | 0/48 [00:06<?, ?it/s]Compiling configurations:   2%|▏         | 1/48 [00:06<05:05,  6.50s/it]Compiling configurations:   6%|▋         | 3/48 [00:07<01:24,  1.88s/it]                                                                        Compiling configurations:   6%|▋         | 3/48 [00:19<01:24,  1.88s/it]Compiling configurations:  10%|█         | 5/48 [00:19<02:59,  4.17s/it]                                                                        Compiling configurations:  10%|█         | 5/48 [00:20<02:59,  4.17s/it]                                                                        Compiling configurations:  10%|█         | 5/48 [00:20<02:59,  4.17s/it]                                                                        Compiling configurations:  10%|█         | 5/48 [00:20<02:59,  4.17s/it]Compiling configurations:  12%|█▎        | 6/48 [00:20<02:21,  3.36s/it]                                                                        Compiling configurations:  12%|█▎        | 6/48 [00:20<02:21,  3.36s/it]                                                                        Compiling configurations:  12%|█▎        | 6/48 [00:21<02:21,  3.36s/it]                                                                        Compiling configurations:  12%|█▎        | 6/48 [00:21<02:21,  3.36s/it]                                                                        Compiling configurations:  12%|█▎        | 6/48 [00:21<02:21,  3.36s/it]Compiling configurations:  15%|█▍        | 7/48 [00:21<01:43,  2.52s/it]                                                                        Compiling configurations:  15%|█▍        | 7/48 [00:21<01:43,  2.52s/it]                                                                        Compiling configurations:  15%|█▍        | 7/48 [00:21<01:43,  2.52s/it]                                                                        Compiling configurations:  15%|█▍        | 7/48 [00:21<01:43,  2.52s/it]Compiling configurations:  17%|█▋        | 8/48 [00:21<01:16,  1.91s/it]Compiling configurations:  19%|█▉        | 9/48 [00:21<00:56,  1.46s/it]Compiling configurations:  21%|██        | 10/48 [00:22<00:42,  1.13s/it]Compiling configurations:  23%|██▎       | 11/48 [00:22<00:33,  1.11it/s]Compiling configurations:  25%|██▌       | 12/48 [00:22<00:26,  1.36it/s]Compiling configurations:  27%|██▋       | 13/48 [00:23<00:21,  1.62it/s]Compiling configurations:  29%|██▉       | 14/48 [00:23<00:18,  1.84it/s]                                                                         Compiling configurations:  29%|██▉       | 14/48 [00:23<00:18,  1.84it/s]Compiling configurations:  31%|███▏      | 15/48 [00:24<00:16,  1.98it/s]Compiling configurations:  33%|███▎      | 16/48 [00:24<00:15,  2.09it/s]                                                                         Compiling configurations:  33%|███▎      | 16/48 [00:38<00:15,  2.09it/s]Compiling configurations:  35%|███▌      | 17/48 [00:39<02:28,  4.80s/it]                                                                         Compiling configurations:  35%|███▌      | 17/48 [00:39<02:28,  4.80s/it]                                                                         Compiling configurations:  35%|███▌      | 17/48 [00:39<02:28,  4.80s/it]                                                                         Compiling configurations:  35%|███▌      | 17/48 [00:40<02:28,  4.80s/it]                                                                         Compiling configurations:  35%|███▌      | 17/48 [00:40<02:28,  4.80s/it]                                                                         Compiling configurations:  35%|███▌      | 17/48 [00:40<02:28,  4.80s/it]Compiling configurations:  38%|███▊      | 18/48 [00:40<01:48,  3.61s/it]                                                                         Compiling configurations:  38%|███▊      | 18/48 [00:40<01:48,  3.61s/it]                                                                         Compiling configurations:  38%|███▊      | 18/48 [00:40<01:48,  3.61s/it]                                                                         Compiling configurations:  38%|███▊      | 18/48 [00:40<01:48,  3.61s/it]                                                                         Compiling configurations:  38%|███▊      | 18/48 [00:40<01:48,  3.61s/it]                                                                         Compiling configurations:  38%|███▊      | 18/48 [00:40<01:48,  3.61s/it]                                                                         Compiling configurations:  38%|███▊      | 18/48 [00:40<01:48,  3.61s/it]                                                                         Compiling configurations:  38%|███▊      | 18/48 [00:40<01:48,  3.61s/it]                                                                         Compiling configurations:  38%|███▊      | 18/48 [00:40<01:48,  3.61s/it]                                                                         Compiling configurations:  38%|███▊      | 18/48 [00:40<01:48,  3.61s/it]                                                                         Compiling configurations:  38%|███▊      | 18/48 [00:40<01:48,  3.61s/it]                                                                         Compiling configurations:  38%|███▊      | 18/48 [00:40<01:48,  3.61s/it]                                                                         Compiling configurations:  38%|███▊      | 18/48 [00:40<01:48,  3.61s/it]Compiling configurations:  40%|███▉      | 19/48 [00:40<01:17,  2.68s/it]                                                                         Compiling configurations:  40%|███▉      | 19/48 [00:40<01:17,  2.68s/it]                                                                         Compiling configurations:  40%|███▉      | 19/48 [00:40<01:17,  2.68s/it]                                                                         Compiling configurations:  40%|███▉      | 19/48 [00:40<01:17,  2.68s/it]                                                                         Compiling configurations:  40%|███▉      | 19/48 [00:40<01:17,  2.68s/it]                                                                         Compiling configurations:  40%|███▉      | 19/48 [00:41<01:17,  2.68s/it]                                                                         Compiling configurations:  40%|███▉      | 19/48 [00:41<01:17,  2.68s/it]Compiling configurations:  42%|████▏     | 20/48 [00:41<00:56,  2.02s/it]                                                                         Compiling configurations:  42%|████▏     | 20/48 [00:41<00:56,  2.02s/it]                                                                         Compiling configurations:  42%|████▏     | 20/48 [00:41<00:56,  2.02s/it]                                                                         Compiling configurations:  42%|████▏     | 20/48 [00:41<00:56,  2.02s/it]Compiling configurations:  44%|████▍     | 21/48 [00:41<00:41,  1.55s/it]Compiling configurations:  46%|████▌     | 22/48 [00:42<00:31,  1.23s/it]Compiling configurations:  48%|████▊     | 23/48 [00:42<00:25,  1.02s/it]Compiling configurations:  50%|█████     | 24/48 [00:43<00:20,  1.15it/s]Compiling configurations:  52%|█████▏    | 25/48 [00:43<00:17,  1.34it/s]                                                                         Compiling configurations:  52%|█████▏    | 25/48 [00:43<00:17,  1.34it/s]                                                                         Compiling configurations:  52%|█████▏    | 25/48 [00:43<00:17,  1.34it/s]                                                                         Compiling configurations:  52%|█████▏    | 25/48 [00:44<00:17,  1.34it/s]Compiling configurations:  54%|█████▍    | 26/48 [00:44<00:14,  1.50it/s]                                                                         Compiling configurations:  54%|█████▍    | 26/48 [00:44<00:14,  1.50it/s]                                                                         Compiling configurations:  54%|█████▍    | 26/48 [00:44<00:14,  1.50it/s]Compiling configurations:  56%|█████▋    | 27/48 [00:44<00:12,  1.65it/s]Compiling configurations:  58%|█████▊    | 28/48 [00:45<00:11,  1.77it/s]Compiling configurations:  60%|██████    | 29/48 [00:45<00:09,  1.90it/s]Compiling configurations:  62%|██████▎   | 30/48 [00:45<00:08,  2.00it/s]Compiling configurations:  65%|██████▍   | 31/48 [00:46<00:08,  1.94it/s]Compiling configurations:  67%|██████▋   | 32/48 [00:46<00:07,  2.00it/s]Compiling configurations:  69%|██████▉   | 33/48 [00:47<00:07,  2.02it/s]Compiling configurations:  71%|███████   | 34/48 [00:47<00:06,  2.07it/s]Compiling configurations:  73%|███████▎  | 35/48 [00:48<00:06,  2.09it/s]Compiling configurations:  75%|███████▌  | 36/48 [00:48<00:05,  2.09it/s]Compiling configurations:  77%|███████▋  | 37/48 [00:49<00:05,  2.02it/s]Compiling configurations:  79%|███████▉  | 38/48 [00:49<00:04,  2.07it/s]Compiling configurations:  81%|████████▏ | 39/48 [00:50<00:04,  1.99it/s]Compiling configurations:  83%|████████▎ | 40/48 [00:50<00:04,  1.95it/s]Compiling configurations:  85%|████████▌ | 41/48 [00:51<00:03,  1.92it/s]Compiling configurations:  88%|████████▊ | 42/48 [00:51<00:03,  1.89it/s]Compiling configurations:  90%|████████▉ | 43/48 [00:52<00:02,  1.88it/s]Compiling configurations:  92%|█████████▏| 44/48 [00:53<00:02,  1.84it/s]Compiling configurations:  94%|█████████▍| 45/48 [00:53<00:01,  1.77it/s]Compiling configurations:  96%|█████████▌| 46/48 [00:54<00:01,  1.71it/s]Compiling configurations:  98%|█████████▊| 47/48 [00:55<00:00,  1.63it/s]Compiling configurations: 100%|██████████| 48/48 [00:55<00:00,  1.67it/s]Compiling configurations: 100%|██████████| 48/48 [00:55<00:00,  1.16s/it]
Bench configurations:   0%|          | 0/44 [00:00<?, ?it/s]Bench configurations:   0%|          | 0/44 [00:00<?, ?it/s, best_latency=0.107]                                                                                Bench configurations:   0%|          | 0/44 [00:00<?, ?it/s, best_latency=0.107]Bench configurations:   2%|▏         | 1/44 [00:00<00:14,  2.91it/s, best_latency=0.107]Bench configurations:   2%|▏         | 1/44 [00:00<00:14,  2.91it/s, best_latency=0.107]                                                                                        Bench configurations:   2%|▏         | 1/44 [00:00<00:14,  2.91it/s, best_latency=0.107]Bench configurations:   5%|▍         | 2/44 [00:00<00:14,  2.95it/s, best_latency=0.107]Bench configurations:   5%|▍         | 2/44 [00:01<00:14,  2.95it/s, best_latency=0.107]                                                                                        Bench configurations:   5%|▍         | 2/44 [00:01<00:14,  2.95it/s, best_latency=0.107]Bench configurations:   7%|▋         | 3/44 [00:01<00:13,  2.97it/s, best_latency=0.107]Bench configurations:   7%|▋         | 3/44 [00:01<00:13,  2.97it/s, best_latency=0.107]                                                                                        Bench configurations:   7%|▋         | 3/44 [00:01<00:13,  2.97it/s, best_latency=0.107]Bench configurations:   9%|▉         | 4/44 [00:01<00:14,  2.81it/s, best_latency=0.107]Bench configurations:   9%|▉         | 4/44 [00:01<00:14,  2.81it/s, best_latency=0.0912]                                                                                         Bench configurations:   9%|▉         | 4/44 [00:01<00:14,  2.81it/s, best_latency=0.0912]Bench configurations:  11%|█▏        | 5/44 [00:01<00:13,  2.80it/s, best_latency=0.0912]Bench configurations:  11%|█▏        | 5/44 [00:02<00:13,  2.80it/s, best_latency=0.0912]                                                                                         Bench configurations:  11%|█▏        | 5/44 [00:02<00:13,  2.80it/s, best_latency=0.0912]Bench configurations:  14%|█▎        | 6/44 [00:02<00:13,  2.79it/s, best_latency=0.0912]Bench configurations:  14%|█▎        | 6/44 [00:02<00:13,  2.79it/s, best_latency=0.0912]                                                                                         Bench configurations:  14%|█▎        | 6/44 [00:02<00:13,  2.79it/s, best_latency=0.0912]Bench configurations:  16%|█▌        | 7/44 [00:02<00:13,  2.78it/s, best_latency=0.0912]Bench configurations:  16%|█▌        | 7/44 [00:02<00:13,  2.78it/s, best_latency=0.0912]                                                                                         Bench configurations:  16%|█▌        | 7/44 [00:02<00:13,  2.78it/s, best_latency=0.0912]Bench configurations:  18%|█▊        | 8/44 [00:02<00:12,  2.82it/s, best_latency=0.0912]Bench configurations:  18%|█▊        | 8/44 [00:03<00:12,  2.82it/s, best_latency=0.0912]                                                                                         Bench configurations:  18%|█▊        | 8/44 [00:03<00:12,  2.82it/s, best_latency=0.0912]Bench configurations:  20%|██        | 9/44 [00:03<00:12,  2.80it/s, best_latency=0.0912]Bench configurations:  20%|██        | 9/44 [00:03<00:12,  2.80it/s, best_latency=0.0912]                                                                                         Bench configurations:  20%|██        | 9/44 [00:03<00:12,  2.80it/s, best_latency=0.0912]Bench configurations:  23%|██▎       | 10/44 [00:03<00:12,  2.74it/s, best_latency=0.0912]Bench configurations:  23%|██▎       | 10/44 [00:04<00:12,  2.74it/s, best_latency=0.0912]                                                                                          Bench configurations:  23%|██▎       | 10/44 [00:04<00:12,  2.74it/s, best_latency=0.0912]Bench configurations:  25%|██▌       | 11/44 [00:04<00:12,  2.57it/s, best_latency=0.0912]Bench configurations:  25%|██▌       | 11/44 [00:04<00:12,  2.57it/s, best_latency=0.0912]                                                                                          Bench configurations:  25%|██▌       | 11/44 [00:04<00:12,  2.57it/s, best_latency=0.0912]Bench configurations:  27%|██▋       | 12/44 [00:04<00:12,  2.48it/s, best_latency=0.0912]Bench configurations:  27%|██▋       | 12/44 [00:04<00:12,  2.48it/s, best_latency=0.0123]                                                                                          Bench configurations:  27%|██▋       | 12/44 [00:04<00:12,  2.48it/s, best_latency=0.0123]Bench configurations:  30%|██▉       | 13/44 [00:04<00:13,  2.35it/s, best_latency=0.0123]Bench configurations:  30%|██▉       | 13/44 [00:05<00:13,  2.35it/s, best_latency=0.0123]                                                                                          Bench configurations:  30%|██▉       | 13/44 [00:05<00:13,  2.35it/s, best_latency=0.0123]Bench configurations:  32%|███▏      | 14/44 [00:05<00:13,  2.20it/s, best_latency=0.0123]Bench configurations:  32%|███▏      | 14/44 [00:05<00:13,  2.20it/s, best_latency=0.0123]                                                                                          Bench configurations:  32%|███▏      | 14/44 [00:05<00:13,  2.20it/s, best_latency=0.0123]Bench configurations:  34%|███▍      | 15/44 [00:05<00:13,  2.12it/s, best_latency=0.0123]Bench configurations:  34%|███▍      | 15/44 [00:06<00:13,  2.12it/s, best_latency=0.0123]                                                                                          Bench configurations:  34%|███▍      | 15/44 [00:06<00:13,  2.12it/s, best_latency=0.0123]Bench configurations:  36%|███▋      | 16/44 [00:06<00:13,  2.15it/s, best_latency=0.0123]Bench configurations:  36%|███▋      | 16/44 [00:06<00:13,  2.15it/s, best_latency=0.0123]                                                                                          Bench configurations:  36%|███▋      | 16/44 [00:06<00:13,  2.15it/s, best_latency=0.0123]Bench configurations:  39%|███▊      | 17/44 [00:06<00:12,  2.16it/s, best_latency=0.0123]Bench configurations:  39%|███▊      | 17/44 [00:07<00:12,  2.16it/s, best_latency=0.0123]                                                                                          Bench configurations:  39%|███▊      | 17/44 [00:07<00:12,  2.16it/s, best_latency=0.0123]Bench configurations:  41%|████      | 18/44 [00:07<00:12,  2.11it/s, best_latency=0.0123]Bench configurations:  41%|████      | 18/44 [00:07<00:12,  2.11it/s, best_latency=0.0123]                                                                                          Bench configurations:  41%|████      | 18/44 [00:07<00:12,  2.11it/s, best_latency=0.0123]Bench configurations:  43%|████▎     | 19/44 [00:07<00:12,  2.01it/s, best_latency=0.0123]Bench configurations:  43%|████▎     | 19/44 [00:08<00:12,  2.01it/s, best_latency=0.0123]                                                                                          Bench configurations:  43%|████▎     | 19/44 [00:08<00:12,  2.01it/s, best_latency=0.0123]Bench configurations:  45%|████▌     | 20/44 [00:08<00:11,  2.01it/s, best_latency=0.0123]Bench configurations:  45%|████▌     | 20/44 [00:08<00:11,  2.01it/s, best_latency=0.0123]                                                                                          Bench configurations:  45%|████▌     | 20/44 [00:08<00:11,  2.01it/s, best_latency=0.0123]Bench configurations:  48%|████▊     | 21/44 [00:08<00:11,  2.04it/s, best_latency=0.0123]Bench configurations:  48%|████▊     | 21/44 [00:09<00:11,  2.04it/s, best_latency=0.0123]                                                                                          Bench configurations:  48%|████▊     | 21/44 [00:09<00:11,  2.04it/s, best_latency=0.0123]Bench configurations:  50%|█████     | 22/44 [00:09<00:10,  2.02it/s, best_latency=0.0123]Bench configurations:  50%|█████     | 22/44 [00:09<00:10,  2.02it/s, best_latency=0.0123]                                                                                          Bench configurations:  50%|█████     | 22/44 [00:09<00:10,  2.02it/s, best_latency=0.0123]Bench configurations:  52%|█████▏    | 23/44 [00:09<00:10,  2.04it/s, best_latency=0.0123]Bench configurations:  52%|█████▏    | 23/44 [00:10<00:10,  2.04it/s, best_latency=0.0123]                                                                                          Bench configurations:  52%|█████▏    | 23/44 [00:10<00:10,  2.04it/s, best_latency=0.0123]Bench configurations:  55%|█████▍    | 24/44 [00:10<00:09,  2.06it/s, best_latency=0.0123]Bench configurations:  55%|█████▍    | 24/44 [00:10<00:09,  2.06it/s, best_latency=0.0123]                                                                                          Bench configurations:  55%|█████▍    | 24/44 [00:10<00:09,  2.06it/s, best_latency=0.0123]Bench configurations:  57%|█████▋    | 25/44 [00:10<00:09,  2.10it/s, best_latency=0.0123]Bench configurations:  57%|█████▋    | 25/44 [00:11<00:09,  2.10it/s, best_latency=0.0123]                                                                                          Bench configurations:  57%|█████▋    | 25/44 [00:11<00:09,  2.10it/s, best_latency=0.0123]Bench configurations:  59%|█████▉    | 26/44 [00:11<00:08,  2.14it/s, best_latency=0.0123]Bench configurations:  59%|█████▉    | 26/44 [00:11<00:08,  2.14it/s, best_latency=0.0123]                                                                                          Bench configurations:  59%|█████▉    | 26/44 [00:11<00:08,  2.14it/s, best_latency=0.0123]Bench configurations:  61%|██████▏   | 27/44 [00:11<00:08,  2.04it/s, best_latency=0.0123]Bench configurations:  61%|██████▏   | 27/44 [00:12<00:08,  2.04it/s, best_latency=0.0121]                                                                                          Bench configurations:  61%|██████▏   | 27/44 [00:12<00:08,  2.04it/s, best_latency=0.0121]Bench configurations:  64%|██████▎   | 28/44 [00:12<00:07,  2.07it/s, best_latency=0.0121]Bench configurations:  64%|██████▎   | 28/44 [00:12<00:07,  2.07it/s, best_latency=0.0121]                                                                                          Bench configurations:  64%|██████▎   | 28/44 [00:12<00:07,  2.07it/s, best_latency=0.0121]Bench configurations:  66%|██████▌   | 29/44 [00:12<00:07,  2.04it/s, best_latency=0.0121]Bench configurations:  66%|██████▌   | 29/44 [00:13<00:07,  2.04it/s, best_latency=0.0121]                                                                                          Bench configurations:  66%|██████▌   | 29/44 [00:13<00:07,  2.04it/s, best_latency=0.0121]Bench configurations:  68%|██████▊   | 30/44 [00:13<00:06,  2.07it/s, best_latency=0.0121]Bench configurations:  68%|██████▊   | 30/44 [00:13<00:06,  2.07it/s, best_latency=0.0121]                                                                                          Bench configurations:  68%|██████▊   | 30/44 [00:13<00:06,  2.07it/s, best_latency=0.0121]Bench configurations:  70%|███████   | 31/44 [00:13<00:06,  2.09it/s, best_latency=0.0121]Bench configurations:  70%|███████   | 31/44 [00:14<00:06,  2.09it/s, best_latency=0.0121]                                                                                          Bench configurations:  70%|███████   | 31/44 [00:14<00:06,  2.09it/s, best_latency=0.0121]Bench configurations:  73%|███████▎  | 32/44 [00:14<00:05,  2.07it/s, best_latency=0.0121]Bench configurations:  73%|███████▎  | 32/44 [00:14<00:05,  2.07it/s, best_latency=0.0121]                                                                                          Bench configurations:  73%|███████▎  | 32/44 [00:14<00:05,  2.07it/s, best_latency=0.0121]Bench configurations:  75%|███████▌  | 33/44 [00:14<00:05,  1.99it/s, best_latency=0.0121]Bench configurations:  75%|███████▌  | 33/44 [00:15<00:05,  1.99it/s, best_latency=0.0121]                                                                                          Bench configurations:  75%|███████▌  | 33/44 [00:15<00:05,  1.99it/s, best_latency=0.0121]Bench configurations:  77%|███████▋  | 34/44 [00:15<00:04,  2.06it/s, best_latency=0.0121]Bench configurations:  77%|███████▋  | 34/44 [00:15<00:04,  2.06it/s, best_latency=0.0121]                                                                                          Bench configurations:  77%|███████▋  | 34/44 [00:15<00:04,  2.06it/s, best_latency=0.0121]Bench configurations:  80%|███████▉  | 35/44 [00:15<00:04,  1.97it/s, best_latency=0.0121]Bench configurations:  80%|███████▉  | 35/44 [00:16<00:04,  1.97it/s, best_latency=0.0121]                                                                                          Bench configurations:  80%|███████▉  | 35/44 [00:16<00:04,  1.97it/s, best_latency=0.0121]Bench configurations:  82%|████████▏ | 36/44 [00:16<00:04,  1.93it/s, best_latency=0.0121]Bench configurations:  82%|████████▏ | 36/44 [00:16<00:04,  1.93it/s, best_latency=0.0121]                                                                                          Bench configurations:  82%|████████▏ | 36/44 [00:16<00:04,  1.93it/s, best_latency=0.0121]Bench configurations:  84%|████████▍ | 37/44 [00:16<00:03,  1.90it/s, best_latency=0.0121]Bench configurations:  84%|████████▍ | 37/44 [00:17<00:03,  1.90it/s, best_latency=0.0121]                                                                                          Bench configurations:  84%|████████▍ | 37/44 [00:17<00:03,  1.90it/s, best_latency=0.0121]Bench configurations:  86%|████████▋ | 38/44 [00:17<00:03,  1.88it/s, best_latency=0.0121]Bench configurations:  86%|████████▋ | 38/44 [00:17<00:03,  1.88it/s, best_latency=0.0121]                                                                                          Bench configurations:  86%|████████▋ | 38/44 [00:17<00:03,  1.88it/s, best_latency=0.0121]Bench configurations:  89%|████████▊ | 39/44 [00:17<00:02,  1.86it/s, best_latency=0.0121]Bench configurations:  89%|████████▊ | 39/44 [00:18<00:02,  1.86it/s, best_latency=0.0121]                                                                                          Bench configurations:  89%|████████▊ | 39/44 [00:18<00:02,  1.86it/s, best_latency=0.0121]Bench configurations:  91%|█████████ | 40/44 [00:18<00:02,  1.96it/s, best_latency=0.0121]Bench configurations:  91%|█████████ | 40/44 [00:18<00:02,  1.96it/s, best_latency=0.0121]                                                                                          Bench configurations:  91%|█████████ | 40/44 [00:18<00:02,  1.96it/s, best_latency=0.0121]Bench configurations:  93%|█████████▎| 41/44 [00:18<00:01,  2.02it/s, best_latency=0.0121]Bench configurations:  93%|█████████▎| 41/44 [00:19<00:01,  2.02it/s, best_latency=0.0121]                                                                                          Bench configurations:  93%|█████████▎| 41/44 [00:19<00:01,  2.02it/s, best_latency=0.0121]Bench configurations:  95%|█████████▌| 42/44 [00:19<00:00,  2.08it/s, best_latency=0.0121]Bench configurations:  95%|█████████▌| 42/44 [00:19<00:00,  2.08it/s, best_latency=0.0121]                                                                                          Bench configurations:  95%|█████████▌| 42/44 [00:19<00:00,  2.08it/s, best_latency=0.0121]Bench configurations:  98%|█████████▊| 43/44 [00:19<00:00,  2.05it/s, best_latency=0.0121]Bench configurations:  98%|█████████▊| 43/44 [00:20<00:00,  2.05it/s, best_latency=0.0121]                                                                                          Bench configurations:  98%|█████████▊| 43/44 [00:20<00:00,  2.05it/s, best_latency=0.0121]Bench configurations: 100%|██████████| 44/44 [00:20<00:00,  2.10it/s, best_latency=0.0121]Bench configurations: 100%|██████████| 44/44 [00:20<00:00,  2.17it/s, best_latency=0.0121]
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
benchmarks/benchmark.py:140: in profile
    latency = do_bench(bench_fn, warmup=warmup, rep=rep, backend='cupti')
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
../../_tool/tileops_benchmarks_venv_db4a9af2f48857c5/lib/python3.11/site-packages/tilelang/profiler/bench.py:100: in do_bench
    fn()
benchmarks/benchmark.py:133: in bench_fn
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
tileops/kernels/flash_decode/mha_decode.py:446: in forward
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
tileops/kernels/flash_decode/mha_decode.py:334: in _mha_decode_wrapped_kernel
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
2026-03-19 10:16:04,114 INFO:Auto-tuning with 0.9 CPU utilizations, 240 CPUs available, 216 CPUs will be used
2026-03-19 10:16:13  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_no_split` with `out_idx=[4]`
2026-03-19 10:16:13  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-19 10:16:13  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_no_split` with `out_idx=[4]`
2026-03-19 10:16:13  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_no_split` with `out_idx=[4]`
2026-03-19 10:16:13  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-19 10:16:13  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_no_split` with `out_idx=[4]`
2026-03-19 10:16:13  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-19 10:16:13  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_no_split` with `out_idx=[4]`
2026-03-19 10:16:13  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_no_split` with `out_idx=[4]`
2026-03-19 10:16:13  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-19 10:16:13  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_no_split` with `out_idx=[4]`
2026-03-19 10:16:13  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-19 10:16:13  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-19 10:16:13  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-19 10:16:13  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_no_split` with `out_idx=[4]`
2026-03-19 10:16:13  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-19 10:16:13  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_no_split` with `out_idx=[4]`
2026-03-19 10:16:13  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_no_split` with `out_idx=[4]`
2026-03-19 10:16:13  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-19 10:16:13  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_no_split` with `out_idx=[4]`
2026-03-19 10:16:13  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_no_split` with `out_idx=[4]`
2026-03-19 10:16:13  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-19 10:16:13  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-19 10:16:13  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-19 10:16:13  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-19 10:16:13  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-19 10:16:13  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_no_split` with `out_idx=[4]`
2026-03-19 10:16:13  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-19 10:16:13  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-19 10:16:13  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-19 10:16:13  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-19 10:16:13  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-19 10:16:13  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-19 10:16:13  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-19 10:16:13  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_no_split` with `out_idx=[4]`
2026-03-19 10:16:13  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-19 10:16:13  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-19 10:16:13  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-19 10:16:13  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-19 10:16:13  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-19 10:16:14  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-19 10:16:14  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-19 10:16:14  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-19 10:16:14  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-19 10:16:14  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-19 10:16:14  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_no_split` with `out_idx=[4]`
2026-03-19 10:16:14  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[4]`
2026-03-19 10:16:26  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_no_split`
2026-03-19 10:16:26  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_no_split`
2026-03-19 10:16:26  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_no_split`
2026-03-19 10:16:26  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_no_split`
2026-03-19 10:16:26  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_no_split`
2026-03-19 10:16:26  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_no_split`
2026-03-19 10:16:26  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_no_split`
2026-03-19 10:16:26  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_no_split`
2026-03-19 10:16:27  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_no_split`
2026-03-19 10:16:27  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_no_split`
2026-03-19 10:16:27  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_no_split`
2026-03-19 10:16:45  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-19 10:16:46  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-19 10:16:46  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-19 10:16:47  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-19 10:16:47  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-19 10:16:47  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-19 10:16:47  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-19 10:16:47  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-19 10:16:47  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-19 10:16:47  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-19 10:16:47  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-19 10:16:47  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-19 10:16:47  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-19 10:16:47  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-19 10:16:47  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-19 10:16:47  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-19 10:16:47  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-19 10:16:47  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-19 10:16:47  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-19 10:16:47  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-19 10:16:47  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-19 10:16:47  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-19 10:16:47  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-19 10:16:48  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-19 10:16:48  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-19 10:16:48  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-19 10:16:48  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-19 10:16:48  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-19 10:16:48  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-19 10:16:48  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-19 10:16:48  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-19 10:16:49  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
Tuned Latency 0.0912725031375885 with config {'block_M': 64, 'block_N': 128, 'num_split': 1, 'num_stages': 3, 'threads': 128} at index 0
Tuned Latency 0.09107150137424469 with config {'block_M': 64, 'block_N': 128, 'num_split': 1, 'num_stages': 2, 'threads': 128} at index 1
Tuned Latency 0.15674880146980286 with config {'block_M': 128, 'block_N': 64, 'num_split': 1, 'num_stages': 3, 'threads': 256} at index 2
Tuned Latency 0.10735240578651428 with config {'block_M': 64, 'block_N': 64, 'num_split': 1, 'num_stages': 3, 'threads': 128} at index 3
Tuned Latency 0.12239763140678406 with config {'block_M': 128, 'block_N': 128, 'num_split': 1, 'num_stages': 2, 'threads': 256} at index 4
Tuned Latency 0.12618686258792877 with config {'block_M': 128, 'block_N': 128, 'num_split': 1, 'num_stages': 3, 'threads': 256} at index 5
Tuned Latency 0.1784452646970749 with config {'block_M': 128, 'block_N': 64, 'num_split': 1, 'num_stages': 3, 'threads': 128} at index 6
Tuned Latency 0.109685517847538 with config {'block_M': 64, 'block_N': 64, 'num_split': 1, 'num_stages': 2, 'threads': 128} at index 7
Tuned Latency 0.14477534592151642 with config {'block_M': 128, 'block_N': 64, 'num_split': 1, 'num_stages': 2, 'threads': 256} at index 8
Tuned Latency 0.17736585438251495 with config {'block_M': 128, 'block_N': 64, 'num_split': 1, 'num_stages': 2, 'threads': 128} at index 9
Tuned Latency 0.28734853863716125 with config {'block_M': 128, 'block_N': 128, 'num_split': 1, 'num_stages': 2, 'threads': 128} at index 10
Tuned Latency 0.2917463183403015 with config {'block_M': 128, 'block_N': 128, 'num_split': 1, 'num_stages': 3, 'threads': 128} at index 11
Tuned Latency 0.012557274661958218 with config {'block_M': 64, 'block_N': 64, 'num_split': 2, 'num_stages': 3, 'threads': 256} at index 12
Tuned Latency 0.017904508858919144 with config {'block_M': 128, 'block_N': 64, 'num_split': 4, 'num_stages': 2, 'threads': 128} at index 13
Tuned Latency 0.013599757105112076 with config {'block_M': 128, 'block_N': 64, 'num_split': 2, 'num_stages': 3, 'threads': 256} at index 14
Tuned Latency 0.01714971475303173 with config {'block_M': 64, 'block_N': 128, 'num_split': 4, 'num_stages': 3, 'threads': 128} at index 15
Tuned Latency 0.015556536614894867 with config {'block_M': 64, 'block_N': 128, 'num_split': 2, 'num_stages': 2, 'threads': 128} at index 16
Tuned Latency 0.012257239781320095 with config {'block_M': 64, 'block_N': 128, 'num_split': 2, 'num_stages': 2, 'threads': 256} at index 17
Tuned Latency 0.01761740632355213 with config {'block_M': 64, 'block_N': 64, 'num_split': 4, 'num_stages': 3, 'threads': 256} at index 18
Tuned Latency 0.019096635282039642 with config {'block_M': 64, 'block_N': 64, 'num_split': 4, 'num_stages': 2, 'threads': 128} at index 19
Tuned Latency 0.01750425435602665 with config {'block_M': 64, 'block_N': 128, 'num_split': 4, 'num_stages': 2, 'threads': 256} at index 20
Tuned Latency 0.019965900108218193 with config {'block_M': 64, 'block_N': 128, 'num_split': 4, 'num_stages': 3, 'threads': 256} at index 21
Tuned Latency 0.013597785495221615 with config {'block_M': 128, 'block_N': 64, 'num_split': 2, 'num_stages': 2, 'threads': 128} at index 22
Tuned Latency 0.02246507629752159 with config {'block_M': 64, 'block_N': 128, 'num_split': 4, 'num_stages': 2, 'threads': 128} at index 23
Tuned Latency 0.018141968175768852 with config {'block_M': 128, 'block_N': 64, 'num_split': 4, 'num_stages': 2, 'threads': 256} at index 24
Tuned Latency 0.01752498373389244 with config {'block_M': 128, 'block_N': 64, 'num_split': 4, 'num_stages': 3, 'threads': 128} at index 25
Tuned Latency 0.019398707896471024 with config {'block_M': 64, 'block_N': 64, 'num_split': 4, 'num_stages': 3, 'threads': 128} at index 26
Tuned Latency 0.012009412050247192 with config {'block_M': 64, 'block_N': 128, 'num_split': 2, 'num_stages': 3, 'threads': 256} at index 27
Tuned Latency 0.01634778082370758 with config {'block_M': 128, 'block_N': 64, 'num_split': 2, 'num_stages': 3, 'threads': 128} at index 28
Tuned Latency 0.012056864798069 with config {'block_M': 64, 'block_N': 128, 'num_split': 2, 'num_stages': 3, 'threads': 128} at index 29
Tuned Latency 0.0173573549836874 with config {'block_M': 64, 'block_N': 64, 'num_split': 4, 'num_stages': 2, 'threads': 256} at index 30
Tuned Latency 0.018191058188676834 with config {'block_M': 128, 'block_N': 64, 'num_split': 4, 'num_stages': 3, 'threads': 256} at index 31
Tuned Latency 0.014437374658882618 with config {'block_M': 64, 'block_N': 64, 'num_split': 2, 'num_stages': 3, 'threads': 128} at index 32
Tuned Latency 0.021344812586903572 with config {'block_M': 128, 'block_N': 128, 'num_split': 2, 'num_stages': 2, 'threads': 128} at index 33
Tuned Latency 0.014451446942985058 with config {'block_M': 64, 'block_N': 64, 'num_split': 2, 'num_stages': 2, 'threads': 256} at index 34
Tuned Latency 0.01614745706319809 with config {'block_M': 128, 'block_N': 64, 'num_split': 2, 'num_stages': 2, 'threads': 256} at index 35
Tuned Latency 0.02311326377093792 with config {'block_M': 128, 'block_N': 128, 'num_split': 4, 'num_stages': 2, 'threads': 128} at index 36
Tuned Latency 0.02127755992114544 with config {'block_M': 128, 'block_N': 128, 'num_split': 2, 'num_stages': 3, 'threads': 128} at index 37
Tuned Latency 0.021472269669175148 with config {'block_M': 128, 'block_N': 128, 'num_split': 2, 'num_stages': 2, 'threads': 256} at index 38
Tuned Latency 0.02422116883099079 with config {'block_M': 128, 'block_N': 128, 'num_split': 4, 'num_stages': 3, 'threads': 256} at index 39
Tuned Latency 0.024238159880042076 with config {'block_M': 128, 'block_N': 128, 'num_split': 4, 'num_stages': 2, 'threads': 256} at index 40
Tuned Latency 0.023133981972932816 with config {'block_M': 128, 'block_N': 128, 'num_split': 4, 'num_stages': 3, 'threads': 128} at index 41
Tuned Latency 0.01255433727055788 with config {'block_M': 64, 'block_N': 64, 'num_split': 2, 'num_stages': 2, 'threads': 128} at index 42
Tuned Latency 0.01430793758481741 with config {'block_M': 128, 'block_N': 128, 'num_split': 2, 'num_stages': 3, 'threads': 256} at index 43
Best config: {'block_M': 64, 'block_N': 128, 'num_split': 2, 'num_stages': 3, 'threads': 256}
mha_decode_kernel initialized with config: {'block_M': 64, 'block_N': 128, 'num_split': 2, 'num_stages': 3, 'threads': 256}
----------------------------- Captured stderr call -----------------------------
Compiling configurations:   0%|          | 0/48 [00:00<?, ?it/s]Compiling configurations:   2%|▏         | 1/48 [00:06<05:05,  6.51s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<05:05,  6.51s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<05:05,  6.51s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<05:05,  6.51s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<05:05,  6.51s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<05:05,  6.51s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<05:05,  6.51s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<05:05,  6.51s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<05:05,  6.51s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<05:05,  6.51s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<05:05,  6.51s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<05:05,  6.51s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<05:05,  6.51s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<05:05,  6.51s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<05:05,  6.51s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<05:05,  6.51s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<05:05,  6.51s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<05:05,  6.51s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<05:05,  6.51s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<05:05,  6.51s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<05:05,  6.51s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<05:05,  6.51s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<05:05,  6.51s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<05:05,  6.51s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<05:05,  6.51s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<05:05,  6.51s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<05:05,  6.51s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<05:05,  6.51s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<05:05,  6.51s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<05:05,  6.51s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<05:05,  6.51s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<05:05,  6.51s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<05:05,  6.51s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<05:05,  6.51s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<05:05,  6.51s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<05:05,  6.51s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<05:05,  6.51s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<05:05,  6.51s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<05:05,  6.51s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<05:05,  6.51s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<05:05,  6.51s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<05:05,  6.51s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<05:05,  6.51s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<05:05,  6.51s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<05:05,  6.51s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<05:05,  6.51s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<05:05,  6.51s/it]                                                                        Compiling configurations:   2%|▏         | 1/48 [00:06<05:05,  6.51s/it]Compiling configurations:   4%|▍         | 2/48 [00:06<02:16,  2.97s/it]Compiling configurations:   8%|▊         | 4/48 [00:07<00:50,  1.16s/it]                                                                        Compiling configurations:   8%|▊         | 4/48 [00:18<00:50,  1.16s/it]                                                                        Compiling configurations:   8%|▊         | 4/48 [00:19<00:50,  1.16s/it]                                                                        Compiling configurations:   8%|▊         | 4/48 [00:19<00:50,  1.16s/it]Compiling configurations:  12%|█▎        | 6/48 [00:19<02:26,  3.49s/it]                                                                        Compiling configurations:  12%|█▎        | 6/48 [00:19<02:26,  3.49s/it]                                                                        Compiling configurations:  12%|█▎        | 6/48 [00:19<02:26,  3.49s/it]                                                                        Compiling configurations:  12%|█▎        | 6/48 [00:19<02:26,  3.49s/it]                                                                        Compiling configurations:  12%|█▎        | 6/48 [00:19<02:26,  3.49s/it]                                                                        Compiling configurations:  12%|█▎        | 6/48 [00:19<02:26,  3.49s/it]                                                                        Compiling configurations:  12%|█▎        | 6/48 [00:19<02:26,  3.49s/it]Compiling configurations:  15%|█▍        | 7/48 [00:19<01:50,  2.70s/it]Compiling configurations:  17%|█▋        | 8/48 [00:20<01:22,  2.07s/it]                                                                        Compiling configurations:  17%|█▋        | 8/48 [00:20<01:22,  2.07s/it]                                                                        Compiling configurations:  17%|█▋        | 8/48 [00:20<01:22,  2.07s/it]Compiling configurations:  19%|█▉        | 9/48 [00:20<01:02,  1.59s/it]Compiling configurations:  21%|██        | 10/48 [00:20<00:47,  1.24s/it]Compiling configurations:  23%|██▎       | 11/48 [00:21<00:36,  1.00it/s]Compiling configurations:  25%|██▌       | 12/48 [00:21<00:28,  1.25it/s]Compiling configurations:  27%|██▋       | 13/48 [00:21<00:23,  1.52it/s]Compiling configurations:  29%|██▉       | 14/48 [00:22<00:19,  1.75it/s]Compiling configurations:  31%|███▏      | 15/48 [00:22<00:17,  1.92it/s]Compiling configurations:  33%|███▎      | 16/48 [00:22<00:16,  1.99it/s]                                                                         Compiling configurations:  33%|███▎      | 16/48 [00:38<00:16,  1.99it/s]Compiling configurations:  35%|███▌      | 17/48 [00:38<02:37,  5.09s/it]                                                                         Compiling configurations:  35%|███▌      | 17/48 [00:39<02:37,  5.09s/it]                                                                         Compiling configurations:  35%|███▌      | 17/48 [00:39<02:37,  5.09s/it]                                                                         Compiling configurations:  35%|███▌      | 17/48 [00:39<02:37,  5.09s/it]                                                                         Compiling configurations:  35%|███▌      | 17/48 [00:39<02:37,  5.09s/it]                                                                         Compiling configurations:  35%|███▌      | 17/48 [00:39<02:37,  5.09s/it]                                                                         Compiling configurations:  35%|███▌      | 17/48 [00:39<02:37,  5.09s/it]                                                                         Compiling configurations:  35%|███▌      | 17/48 [00:39<02:37,  5.09s/it]Compiling configurations:  38%|███▊      | 18/48 [00:39<01:57,  3.91s/it]                                                                         Compiling configurations:  38%|███▊      | 18/48 [00:39<01:57,  3.91s/it]                                                                         Compiling configurations:  38%|███▊      | 18/48 [00:40<01:57,  3.91s/it]                                                                         Compiling configurations:  38%|███▊      | 18/48 [00:40<01:57,  3.91s/it]                                                                         Compiling configurations:  38%|███▊      | 18/48 [00:40<01:57,  3.91s/it]                                                                         Compiling configurations:  38%|███▊      | 18/48 [00:40<01:57,  3.91s/it]                                                                         Compiling configurations:  38%|███▊      | 18/48 [00:40<01:57,  3.91s/it]                                                                         Compiling configurations:  38%|███▊      | 18/48 [00:40<01:57,  3.91s/it]                                                                         Compiling configurations:  38%|███▊      | 18/48 [00:40<01:57,  3.91s/it]                                                                         Compiling configurations:  38%|███▊      | 18/48 [00:40<01:57,  3.91s/it]Compiling configurations:  40%|███▉      | 19/48 [00:40<01:23,  2.89s/it]                                                                         Compiling configurations:  40%|███▉      | 19/48 [00:40<01:23,  2.89s/it]                                                                         Compiling configurations:  40%|███▉      | 19/48 [00:40<01:23,  2.89s/it]                                                                         Compiling configurations:  40%|███▉      | 19/48 [00:40<01:23,  2.89s/it]                                                                         Compiling configurations:  40%|███▉      | 19/48 [00:40<01:23,  2.89s/it]                                                                         Compiling configurations:  40%|███▉      | 19/48 [00:40<01:23,  2.89s/it]                                                                         Compiling configurations:  40%|███▉      | 19/48 [00:40<01:23,  2.89s/it]                                                                         Compiling configurations:  40%|███▉      | 19/48 [00:40<01:23,  2.89s/it]                                                                         Compiling configurations:  40%|███▉      | 19/48 [00:40<01:23,  2.89s/it]                                                                         Compiling configurations:  40%|███▉      | 19/48 [00:40<01:23,  2.89s/it]                                                                         Compiling configurations:  40%|███▉      | 19/48 [00:40<01:23,  2.89s/it]Compiling configurations:  42%|████▏     | 20/48 [00:40<01:00,  2.16s/it]                                                                         Compiling configurations:  42%|████▏     | 20/48 [00:40<01:00,  2.16s/it]                                                                         Compiling configurations:  42%|████▏     | 20/48 [00:40<01:00,  2.16s/it]                                                                         Compiling configurations:  42%|████▏     | 20/48 [00:41<01:00,  2.16s/it]                                                                         Compiling configurations:  42%|████▏     | 20/48 [00:41<01:00,  2.16s/it]Compiling configurations:  44%|████▍     | 21/48 [00:41<00:44,  1.64s/it]Compiling configurations:  46%|████▌     | 22/48 [00:41<00:33,  1.28s/it]Compiling configurations:  48%|████▊     | 23/48 [00:42<00:25,  1.02s/it]                                                                         Compiling configurations:  48%|████▊     | 23/48 [00:42<00:25,  1.02s/it]Compiling configurations:  50%|█████     | 24/48 [00:42<00:20,  1.18it/s]Compiling configurations:  52%|█████▏    | 25/48 [00:43<00:16,  1.38it/s]Compiling configurations:  54%|█████▍    | 26/48 [00:43<00:14,  1.56it/s]Compiling configurations:  56%|█████▋    | 27/48 [00:44<00:12,  1.69it/s]Compiling configurations:  58%|█████▊    | 28/48 [00:44<00:11,  1.82it/s]Compiling configurations:  60%|██████    | 29/48 [00:44<00:09,  1.91it/s]Compiling configurations:  62%|██████▎   | 30/48 [00:45<00:09,  1.97it/s]Compiling configurations:  65%|██████▍   | 31/48 [00:45<00:08,  2.06it/s]Compiling configurations:  67%|██████▋   | 32/48 [00:46<00:07,  2.11it/s]Compiling configurations:  69%|██████▉   | 33/48 [00:46<00:07,  2.11it/s]Compiling configurations:  71%|███████   | 34/48 [00:47<00:06,  2.14it/s]Compiling configurations:  73%|███████▎  | 35/48 [00:47<00:05,  2.19it/s]Compiling configurations:  75%|███████▌  | 36/48 [00:48<00:05,  2.16it/s]Compiling configurations:  77%|███████▋  | 37/48 [00:48<00:04,  2.21it/s]Compiling configurations:  79%|███████▉  | 38/48 [00:49<00:04,  2.10it/s]Compiling configurations:  81%|████████▏ | 39/48 [00:49<00:04,  2.16it/s]Compiling configurations:  83%|████████▎ | 40/48 [00:49<00:03,  2.14it/s]Compiling configurations:  85%|████████▌ | 41/48 [00:50<00:03,  2.06it/s]Compiling configurations:  88%|████████▊ | 42/48 [00:51<00:02,  2.01it/s]Compiling configurations:  90%|████████▉ | 43/48 [00:51<00:02,  1.99it/s]Compiling configurations:  92%|█████████▏| 44/48 [00:52<00:02,  1.97it/s]Compiling configurations:  94%|█████████▍| 45/48 [00:52<00:01,  1.94it/s]Compiling configurations:  96%|█████████▌| 46/48 [00:53<00:01,  1.93it/s]Compiling configurations:  98%|█████████▊| 47/48 [00:53<00:00,  1.98it/s]Compiling configurations: 100%|██████████| 48/48 [00:54<00:00,  1.90it/s]Compiling configurations: 100%|██████████| 48/48 [00:54<00:00,  1.13s/it]
Bench configurations:   0%|          | 0/44 [00:00<?, ?it/s]Bench configurations:   0%|          | 0/44 [00:00<?, ?it/s, best_latency=0.0913]                                                                                 Bench configurations:   0%|          | 0/44 [00:00<?, ?it/s, best_latency=0.0913]Bench configurations:   0%|          | 0/44 [00:00<?, ?it/s, best_latency=0.0911]                                                                                 Bench configurations:   0%|          | 0/44 [00:00<?, ?it/s, best_latency=0.0911]Bench configurations:   5%|▍         | 2/44 [00:00<00:08,  5.22it/s, best_latency=0.0911]Bench configurations:   5%|▍         | 2/44 [00:00<00:08,  5.22it/s, best_latency=0.0911]                                                                                         Bench configurations:   5%|▍         | 2/44 [00:00<00:08,  5.22it/s, best_latency=0.0911]Bench configurations:   7%|▋         | 3/44 [00:00<00:10,  4.00it/s, best_latency=0.0911]Bench configurations:   7%|▋         | 3/44 [00:01<00:10,  4.00it/s, best_latency=0.0911]                                                                                         Bench configurations:   7%|▋         | 3/44 [00:01<00:10,  4.00it/s, best_latency=0.0911]Bench configurations:   9%|▉         | 4/44 [00:01<00:11,  3.59it/s, best_latency=0.0911]Bench configurations:   9%|▉         | 4/44 [00:01<00:11,  3.59it/s, best_latency=0.0911]                                                                                         Bench configurations:   9%|▉         | 4/44 [00:01<00:11,  3.59it/s, best_latency=0.0911]Bench configurations:  11%|█▏        | 5/44 [00:01<00:11,  3.29it/s, best_latency=0.0911]Bench configurations:  11%|█▏        | 5/44 [00:01<00:11,  3.29it/s, best_latency=0.0911]                                                                                         Bench configurations:  11%|█▏        | 5/44 [00:01<00:11,  3.29it/s, best_latency=0.0911]Bench configurations:  14%|█▎        | 6/44 [00:01<00:12,  3.12it/s, best_latency=0.0911]Bench configurations:  14%|█▎        | 6/44 [00:02<00:12,  3.12it/s, best_latency=0.0911]                                                                                         Bench configurations:  14%|█▎        | 6/44 [00:02<00:12,  3.12it/s, best_latency=0.0911]Bench configurations:  16%|█▌        | 7/44 [00:02<00:12,  2.96it/s, best_latency=0.0911]Bench configurations:  16%|█▌        | 7/44 [00:02<00:12,  2.96it/s, best_latency=0.0911]                                                                                         Bench configurations:  16%|█▌        | 7/44 [00:02<00:12,  2.96it/s, best_latency=0.0911]Bench configurations:  18%|█▊        | 8/44 [00:02<00:12,  2.96it/s, best_latency=0.0911]Bench configurations:  18%|█▊        | 8/44 [00:02<00:12,  2.96it/s, best_latency=0.0911]                                                                                         Bench configurations:  18%|█▊        | 8/44 [00:02<00:12,  2.96it/s, best_latency=0.0911]Bench configurations:  20%|██        | 9/44 [00:02<00:11,  3.00it/s, best_latency=0.0911]Bench configurations:  20%|██        | 9/44 [00:03<00:11,  3.00it/s, best_latency=0.0911]                                                                                         Bench configurations:  20%|██        | 9/44 [00:03<00:11,  3.00it/s, best_latency=0.0911]Bench configurations:  23%|██▎       | 10/44 [00:03<00:11,  2.88it/s, best_latency=0.0911]Bench configurations:  23%|██▎       | 10/44 [00:03<00:11,  2.88it/s, best_latency=0.0911]                                                                                          Bench configurations:  23%|██▎       | 10/44 [00:03<00:11,  2.88it/s, best_latency=0.0911]Bench configurations:  25%|██▌       | 11/44 [00:03<00:12,  2.71it/s, best_latency=0.0911]Bench configurations:  25%|██▌       | 11/44 [00:04<00:12,  2.71it/s, best_latency=0.0911]                                                                                          Bench configurations:  25%|██▌       | 11/44 [00:04<00:12,  2.71it/s, best_latency=0.0911]Bench configurations:  27%|██▋       | 12/44 [00:04<00:12,  2.58it/s, best_latency=0.0911]Bench configurations:  27%|██▋       | 12/44 [00:04<00:12,  2.58it/s, best_latency=0.0126]                                                                                          Bench configurations:  27%|██▋       | 12/44 [00:04<00:12,  2.58it/s, best_latency=0.0126]Bench configurations:  30%|██▉       | 13/44 [00:04<00:12,  2.47it/s, best_latency=0.0126]Bench configurations:  30%|██▉       | 13/44 [00:04<00:12,  2.47it/s, best_latency=0.0126]                                                                                          Bench configurations:  30%|██▉       | 13/44 [00:04<00:12,  2.47it/s, best_latency=0.0126]Bench configurations:  32%|███▏      | 14/44 [00:04<00:12,  2.33it/s, best_latency=0.0126]Bench configurations:  32%|███▏      | 14/44 [00:05<00:12,  2.33it/s, best_latency=0.0126]                                                                                          Bench configurations:  32%|███▏      | 14/44 [00:05<00:12,  2.33it/s, best_latency=0.0126]Bench configurations:  34%|███▍      | 15/44 [00:05<00:12,  2.25it/s, best_latency=0.0126]Bench configurations:  34%|███▍      | 15/44 [00:05<00:12,  2.25it/s, best_latency=0.0126]                                                                                          Bench configurations:  34%|███▍      | 15/44 [00:05<00:12,  2.25it/s, best_latency=0.0126]Bench configurations:  36%|███▋      | 16/44 [00:05<00:12,  2.20it/s, best_latency=0.0126]Bench configurations:  36%|███▋      | 16/44 [00:06<00:12,  2.20it/s, best_latency=0.0126]                                                                                          Bench configurations:  36%|███▋      | 16/44 [00:06<00:12,  2.20it/s, best_latency=0.0126]Bench configurations:  39%|███▊      | 17/44 [00:06<00:12,  2.20it/s, best_latency=0.0126]Bench configurations:  39%|███▊      | 17/44 [00:06<00:12,  2.20it/s, best_latency=0.0123]                                                                                          Bench configurations:  39%|███▊      | 17/44 [00:06<00:12,  2.20it/s, best_latency=0.0123]Bench configurations:  41%|████      | 18/44 [00:06<00:11,  2.19it/s, best_latency=0.0123]Bench configurations:  41%|████      | 18/44 [00:07<00:11,  2.19it/s, best_latency=0.0123]                                                                                          Bench configurations:  41%|████      | 18/44 [00:07<00:11,  2.19it/s, best_latency=0.0123]Bench configurations:  43%|████▎     | 19/44 [00:07<00:11,  2.22it/s, best_latency=0.0123]Bench configurations:  43%|████▎     | 19/44 [00:07<00:11,  2.22it/s, best_latency=0.0123]                                                                                          Bench configurations:  43%|████▎     | 19/44 [00:07<00:11,  2.22it/s, best_latency=0.0123]Bench configurations:  45%|████▌     | 20/44 [00:07<00:10,  2.21it/s, best_latency=0.0123]Bench configurations:  45%|████▌     | 20/44 [00:08<00:10,  2.21it/s, best_latency=0.0123]                                                                                          Bench configurations:  45%|████▌     | 20/44 [00:08<00:10,  2.21it/s, best_latency=0.0123]Bench configurations:  48%|████▊     | 21/44 [00:08<00:10,  2.21it/s, best_latency=0.0123]Bench configurations:  48%|████▊     | 21/44 [00:08<00:10,  2.21it/s, best_latency=0.0123]                                                                                          Bench configurations:  48%|████▊     | 21/44 [00:08<00:10,  2.21it/s, best_latency=0.0123]Bench configurations:  50%|█████     | 22/44 [00:08<00:09,  2.20it/s, best_latency=0.0123]Bench configurations:  50%|█████     | 22/44 [00:09<00:09,  2.20it/s, best_latency=0.0123]                                                                                          Bench configurations:  50%|█████     | 22/44 [00:09<00:09,  2.20it/s, best_latency=0.0123]Bench configurations:  52%|█████▏    | 23/44 [00:09<00:09,  2.14it/s, best_latency=0.0123]Bench configurations:  52%|█████▏    | 23/44 [00:09<00:09,  2.14it/s, best_latency=0.0123]                                                                                          Bench configurations:  52%|█████▏    | 23/44 [00:09<00:09,  2.14it/s, best_latency=0.0123]Bench configurations:  55%|█████▍    | 24/44 [00:09<00:09,  2.15it/s, best_latency=0.0123]Bench configurations:  55%|█████▍    | 24/44 [00:10<00:09,  2.15it/s, best_latency=0.0123]                                                                                          Bench configurations:  55%|█████▍    | 24/44 [00:10<00:09,  2.15it/s, best_latency=0.0123]Bench configurations:  57%|█████▋    | 25/44 [00:10<00:08,  2.13it/s, best_latency=0.0123]Bench configurations:  57%|█████▋    | 25/44 [00:10<00:08,  2.13it/s, best_latency=0.0123]                                                                                          Bench configurations:  57%|█████▋    | 25/44 [00:10<00:08,  2.13it/s, best_latency=0.0123]Bench configurations:  59%|█████▉    | 26/44 [00:10<00:08,  2.11it/s, best_latency=0.0123]Bench configurations:  59%|█████▉    | 26/44 [00:10<00:08,  2.11it/s, best_latency=0.0123]                                                                                          Bench configurations:  59%|█████▉    | 26/44 [00:10<00:08,  2.11it/s, best_latency=0.0123]Bench configurations:  61%|██████▏   | 27/44 [00:10<00:07,  2.15it/s, best_latency=0.0123]Bench configurations:  61%|██████▏   | 27/44 [00:11<00:07,  2.15it/s, best_latency=0.012]                                                                                          Bench configurations:  61%|██████▏   | 27/44 [00:11<00:07,  2.15it/s, best_latency=0.012]Bench configurations:  64%|██████▎   | 28/44 [00:11<00:07,  2.16it/s, best_latency=0.012]Bench configurations:  64%|██████▎   | 28/44 [00:11<00:07,  2.16it/s, best_latency=0.012]                                                                                         Bench configurations:  64%|██████▎   | 28/44 [00:11<00:07,  2.16it/s, best_latency=0.012]Bench configurations:  66%|██████▌   | 29/44 [00:11<00:07,  2.13it/s, best_latency=0.012]Bench configurations:  66%|██████▌   | 29/44 [00:12<00:07,  2.13it/s, best_latency=0.012]                                                                                         Bench configurations:  66%|██████▌   | 29/44 [00:12<00:07,  2.13it/s, best_latency=0.012]Bench configurations:  68%|██████▊   | 30/44 [00:12<00:06,  2.16it/s, best_latency=0.012]Bench configurations:  68%|██████▊   | 30/44 [00:12<00:06,  2.16it/s, best_latency=0.012]                                                                                         Bench configurations:  68%|██████▊   | 30/44 [00:12<00:06,  2.16it/s, best_latency=0.012]Bench configurations:  70%|███████   | 31/44 [00:12<00:05,  2.19it/s, best_latency=0.012]Bench configurations:  70%|███████   | 31/44 [00:13<00:05,  2.19it/s, best_latency=0.012]                                                                                         Bench configurations:  70%|███████   | 31/44 [00:13<00:05,  2.19it/s, best_latency=0.012]Bench configurations:  73%|███████▎  | 32/44 [00:13<00:05,  2.15it/s, best_latency=0.012]Bench configurations:  73%|███████▎  | 32/44 [00:13<00:05,  2.15it/s, best_latency=0.012]                                                                                         Bench configurations:  73%|███████▎  | 32/44 [00:13<00:05,  2.15it/s, best_latency=0.012]Bench configurations:  75%|███████▌  | 33/44 [00:13<00:05,  2.18it/s, best_latency=0.012]Bench configurations:  75%|███████▌  | 33/44 [00:14<00:05,  2.18it/s, best_latency=0.012]                                                                                         Bench configurations:  75%|███████▌  | 33/44 [00:14<00:05,  2.18it/s, best_latency=0.012]Bench configurations:  77%|███████▋  | 34/44 [00:14<00:04,  2.07it/s, best_latency=0.012]Bench configurations:  77%|███████▋  | 34/44 [00:14<00:04,  2.07it/s, best_latency=0.012]                                                                                         Bench configurations:  77%|███████▋  | 34/44 [00:14<00:04,  2.07it/s, best_latency=0.012]Bench configurations:  80%|███████▉  | 35/44 [00:14<00:04,  2.05it/s, best_latency=0.012]Bench configurations:  80%|███████▉  | 35/44 [00:15<00:04,  2.05it/s, best_latency=0.012]                                                                                         Bench configurations:  80%|███████▉  | 35/44 [00:15<00:04,  2.05it/s, best_latency=0.012]Bench configurations:  82%|████████▏ | 36/44 [00:15<00:03,  2.06it/s, best_latency=0.012]Bench configurations:  82%|████████▏ | 36/44 [00:15<00:03,  2.06it/s, best_latency=0.012]                                                                                         Bench configurations:  82%|████████▏ | 36/44 [00:15<00:03,  2.06it/s, best_latency=0.012]Bench configurations:  84%|████████▍ | 37/44 [00:15<00:03,  2.00it/s, best_latency=0.012]Bench configurations:  84%|████████▍ | 37/44 [00:16<00:03,  2.00it/s, best_latency=0.012]                                                                                         Bench configurations:  84%|████████▍ | 37/44 [00:16<00:03,  2.00it/s, best_latency=0.012]Bench configurations:  86%|████████▋ | 38/44 [00:16<00:03,  1.95it/s, best_latency=0.012]Bench configurations:  86%|████████▋ | 38/44 [00:16<00:03,  1.95it/s, best_latency=0.012]                                                                                         Bench configurations:  86%|████████▋ | 38/44 [00:16<00:03,  1.95it/s, best_latency=0.012]Bench configurations:  89%|████████▊ | 39/44 [00:16<00:02,  1.92it/s, best_latency=0.012]Bench configurations:  89%|████████▊ | 39/44 [00:17<00:02,  1.92it/s, best_latency=0.012]                                                                                         Bench configurations:  89%|████████▊ | 39/44 [00:17<00:02,  1.92it/s, best_latency=0.012]Bench configurations:  91%|█████████ | 40/44 [00:17<00:02,  1.91it/s, best_latency=0.012]Bench configurations:  91%|█████████ | 40/44 [00:17<00:02,  1.91it/s, best_latency=0.012]                                                                                         Bench configurations:  91%|█████████ | 40/44 [00:17<00:02,  1.91it/s, best_latency=0.012]Bench configurations:  93%|█████████▎| 41/44 [00:17<00:01,  1.88it/s, best_latency=0.012]Bench configurations:  93%|█████████▎| 41/44 [00:18<00:01,  1.88it/s, best_latency=0.012]                                                                                         Bench configurations:  93%|█████████▎| 41/44 [00:18<00:01,  1.88it/s, best_latency=0.012]Bench configurations:  95%|█████████▌| 42/44 [00:18<00:01,  1.85it/s, best_latency=0.012]Bench configurations:  95%|█████████▌| 42/44 [00:18<00:01,  1.85it/s, best_latency=0.012]                                                                                         Bench configurations:  95%|█████████▌| 42/44 [00:18<00:01,  1.85it/s, best_latency=0.012]Bench configurations:  98%|█████████▊| 43/44 [00:18<00:00,  1.97it/s, best_latency=0.012]Bench configurations:  98%|█████████▊| 43/44 [00:19<00:00,  1.97it/s, best_latency=0.012]                                                                                         Bench configurations:  98%|█████████▊| 43/44 [00:19<00:00,  1.97it/s, best_latency=0.012]Bench configurations: 100%|██████████| 44/44 [00:19<00:00,  1.94it/s, best_latency=0.012]Bench configurations: 100%|██████████| 44/44 [00:19<00:00,  2.26it/s, best_latency=0.012]
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
benchmarks/ops/bench_fp8_lighting_indexer.py: 4 warnings
benchmarks/ops/bench_fp8_quant.py: 8 warnings
benchmarks/ops/bench_fused_add_layer_norm.py: 8 warnings
benchmarks/ops/bench_fused_add_rmsnorm.py: 8 warnings
benchmarks/ops/bench_gated_deltanet_chunkwise.py: 32 warnings
benchmarks/ops/bench_gated_deltanet_recurrence.py: 13 warnings
benchmarks/ops/bench_gemm.py: 6 warnings
benchmarks/ops/bench_gla_chunkwise.py: 24 warnings
benchmarks/ops/bench_gla_recurrence.py: 16 warnings
benchmarks/ops/bench_gqa.py: 6 warnings
benchmarks/ops/bench_gqa_decode.py: 3 warnings
benchmarks/ops/bench_gqa_decode_paged.py: 4 warnings
benchmarks/ops/bench_gqa_sliding_window_fwd.py: 5 warnings
benchmarks/ops/bench_gqa_sliding_window_varlen_fwd.py: 5 warnings
benchmarks/ops/bench_group_norm.py: 8 warnings
benchmarks/ops/bench_grouped_gemm.py: 16 warnings
benchmarks/ops/bench_independent_elementwise.py: 132 warnings
benchmarks/ops/bench_instance_norm.py: 8 warnings
benchmarks/ops/bench_layer_norm.py: 8 warnings
benchmarks/ops/bench_logical_reduce.py: 30 warnings
benchmarks/ops/bench_mean_pooling.py: 8 warnings
benchmarks/ops/bench_mha.py: 6 warnings
benchmarks/ops/bench_mha_decode.py: 1 warning
benchmarks/ops/bench_mha_decode_paged.py: 4 warnings
benchmarks/ops/bench_mhc_post.py: 6 warnings
benchmarks/ops/bench_mhc_pre.py: 6 warnings
benchmarks/ops/bench_moe_permute_align.py: 10 warnings
benchmarks/ops/bench_reduce.py: 20 warnings
benchmarks/ops/bench_rms_norm.py: 8 warnings
benchmarks/ops/bench_rope.py: 9 warnings
benchmarks/ops/bench_softmax.py: 24 warnings
benchmarks/ops/bench_topk_selector.py: 8 warnings
benchmarks/ops/bench_unary_elementwise.py: 21 warnings
benchmarks/ops/bench_unary_strategy.py: 27 warnings
benchmarks/ops/bench_vector_norm.py: 24 warnings
  /home/ci-runner/workdir/_tool/tileops_benchmarks_venv_db4a9af2f48857c5/lib/python3.11/site-packages/torch/profiler/profiler.py:509: UserWarning: Profiler won't be using warmup, this can skew profiler results
    warn("Profiler won't be using warmup, this can skew profiler results")

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED benchmarks/ops/bench_mha_decode.py::test_mha_decode_bench[fp16-long-cache] - RuntimeError: kernel mha_decode_split input Output_partial ndim expected 5, but got 4
FAILED benchmarks/ops/bench_mha_decode.py::test_mha_decode_bench[bf16-long-cache] - RuntimeError: kernel mha_decode_split input Output_partial ndim expected 5, but got 4
2 failed, 803 passed, 967 warnings in 3762.91s (1:02:42)

===== profile_run.log summary =====
# TileOPs Benchmark Report
Generated: 2026-03-19 10:27:50

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
| 1000 | 24.12 | 0.00 | 0.00 | 0.00 |
| 2000 | 25.51 | 0.00 | 0.00 | 0.00 |
| 4000 | 22.43 | 0.00 | 0.00 | 0.01 |
| 8000 | 22.83 | 0.00 | 0.00 | 0.02 |
| 16000 | 22.66 | 0.00 | 0.01 | 0.03 |
| 32000 | 22.35 | 0.00 | 0.02 | 0.06 |
| 64000 | 25.66 | 0.00 | 0.03 | 0.12 |
| 128000 | 22.64 | 0.00 | 0.06 | 0.24 |
| 256000 | 444.41 | 0.00 | 0.11 | 0.45 |
| 512000 | 25.09 | 0.00 | 0.20 | 0.78 |

## r4_strategy_unary

### relu_direct

| n_total | size_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | torch.float16 | direct | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.bfloat16 | direct | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.float32 | direct | 0.00 | 0.00 | 0.02 |
| 4194304 | 4M | torch.float16 | direct | 0.01 | 0.29 | 1.15 |
| 4194304 | 4M | torch.bfloat16 | direct | 0.01 | 0.29 | 1.15 |
| 4194304 | 4M | torch.float32 | direct | 0.02 | 0.19 | 1.50 |
| 16777216 | 16M | torch.float16 | direct | 0.05 | 0.32 | 1.29 |
| 16777216 | 16M | torch.bfloat16 | direct | 0.05 | 0.32 | 1.29 |
| 16777216 | 16M | torch.float32 | direct | 0.06 | 0.29 | 2.34 |

### relu_explicit_parallel

| n_total | size_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | torch.float16 | explicit_parallel | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.bfloat16 | explicit_parallel | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.float32 | explicit_parallel | 0.00 | 0.00 | 0.02 |
| 4194304 | 4M | torch.float16 | explicit_parallel | 0.02 | 0.22 | 0.86 |
| 4194304 | 4M | torch.bfloat16 | explicit_parallel | 0.02 | 0.22 | 0.86 |
| 4194304 | 4M | torch.float32 | explicit_parallel | 0.01 | 0.38 | 3.07 |
| 16777216 | 16M | torch.float16 | explicit_parallel | 0.07 | 0.25 | 0.98 |
| 16777216 | 16M | torch.bfloat16 | explicit_parallel | 0.07 | 0.25 | 0.98 |
| 16777216 | 16M | torch.float32 | explicit_parallel | 0.03 | 0.48 | 3.84 |

### relu_register_copy

| n_total | size_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | torch.float16 | register_copy | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.bfloat16 | register_copy | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.float32 | register_copy | 0.00 | 0.00 | 0.02 |
| 4194304 | 4M | torch.float16 | register_copy | 0.01 | 0.66 | 2.64 |
| 4194304 | 4M | torch.bfloat16 | register_copy | 0.01 | 0.44 | 1.76 |
| 4194304 | 4M | torch.float32 | register_copy | 0.01 | 0.38 | 3.08 |
| 16777216 | 16M | torch.float16 | register_copy | 0.02 | 0.90 | 3.61 |
| 16777216 | 16M | torch.bfloat16 | register_copy | 0.02 | 0.90 | 3.61 |
| 16777216 | 16M | torch.float32 | register_copy | 0.03 | 0.49 | 3.92 |

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
| 4194304 | 4M | torch.float16 | explicit_parallel | 0.01 | 0.46 | 1.84 |
| 4194304 | 4M | torch.bfloat16 | explicit_parallel | 0.01 | 0.46 | 1.83 |
| 4194304 | 4M | torch.float32 | explicit_parallel | 0.01 | 0.38 | 3.04 |
| 16777216 | 16M | torch.float16 | explicit_parallel | 0.03 | 0.57 | 2.26 |
| 16777216 | 16M | torch.bfloat16 | explicit_parallel | 0.03 | 0.56 | 2.23 |
| 16777216 | 16M | torch.float32 | explicit_parallel | 0.04 | 0.46 | 3.68 |

### gelu_register_copy

| n_total | size_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | torch.float16 | register_copy | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.bfloat16 | register_copy | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.float32 | register_copy | 0.00 | 0.00 | 0.02 |
| 4194304 | 4M | torch.float16 | register_copy | 0.01 | 0.47 | 1.87 |
| 4194304 | 4M | torch.bfloat16 | register_copy | 0.01 | 0.46 | 1.83 |
| 4194304 | 4M | torch.float32 | register_copy | 0.01 | 0.39 | 3.10 |
| 16777216 | 16M | torch.float16 | register_copy | 0.03 | 0.60 | 2.41 |
| 16777216 | 16M | torch.bfloat16 | register_copy | 0.03 | 0.58 | 2.32 |
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
| 4194304 | 4M | relu | 128 | 0.02 | 0.21 | 0.84 |
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
| 4194304 | 4M | erf | 128 | 0.01 | 0.49 | 1.95 |
| 16777216 | 16M | erf | 128 | 0.03 | 0.60 | 2.40 |

### erf_t256

| n_total | size_label | op_name | threads | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | erf | 256 | 0.00 | 0.00 | 0.01 |
| 4194304 | 4M | erf | 256 | 0.01 | 0.48 | 1.93 |
| 16777216 | 16M | erf | 256 | 0.03 | 0.60 | 2.39 |

### mish_t128

| n_total | size_label | op_name | threads | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | mish | 128 | 0.00 | 0.00 | 0.01 |
| 4194304 | 4M | mish | 128 | 0.01 | 0.36 | 1.43 |
| 16777216 | 16M | mish | 128 | 0.04 | 0.42 | 1.69 |

### mish_t256

| n_total | size_label | op_name | threads | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | mish | 256 | 0.00 | 0.00 | 0.01 |
| 4194304 | 4M | mish | 256 | 0.01 | 0.35 | 1.41 |
| 16777216 | 16M | mish | 256 | 0.04 | 0.42 | 1.67 |

## r7_dtype_npt

### relu_fp32_npt4

| dtype_label | num_per_thread | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| fp32 | 4 | 0.00 | 0.22 | 1.78 |

### relu_fp32_npt8

| dtype_label | num_per_thread | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| fp32 | 8 | 0.00 | 0.22 | 1.75 |

### relu_fp16_npt4

| dtype_label | num_per_thread | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| fp16 | 4 | 0.00 | 0.27 | 1.10 |

### relu_fp16_npt8

| dtype_label | num_per_thread | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| fp16 | 8 | 0.00 | 0.22 | 0.87 |

## relu

### tileops

| n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 4194304 | torch.float16 | 0.01 | 0.66 | 2.65 |
| 4194304 | torch.bfloat16 | 0.01 | 0.66 | 2.65 |
| 4194304 | torch.float32 | 0.01 | 0.39 | 3.13 |

### baseline

| n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 4194304 | torch.float16 | 0.01 | 0.63 | 2.53 |
| 4194304 | torch.bfloat16 | 0.01 | 0.65 | 2.60 |
| 4194304 | torch.float32 | 0.01 | 0.39 | 3.15 |

## ada_layer_norm

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.02 | 1.39 | 2.22 |
| 4096 | 4096 | torch.bfloat16 | 0.05 | 1.61 | 2.58 |
| 1024 | 3000 | torch.float16 | 0.04 | 0.38 | 0.61 |
| 1025 | 4096 | torch.float16 | 0.01 | 1.40 | 2.25 |

### baseline

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.02 | 0.85 | 1.35 |
| 4096 | 4096 | torch.bfloat16 | 0.08 | 1.04 | 1.67 |
| 1024 | 3000 | torch.float16 | 0.02 | 0.75 | 1.19 |
| 1025 | 4096 | torch.float16 | 0.03 | 0.83 | 1.33 |

## ada_layer_norm_zero

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.02 | 1.35 | 2.25 |
| 4096 | 4096 | torch.bfloat16 | 0.06 | 1.58 | 2.63 |
| 1024 | 3000 | torch.float16 | 0.05 | 0.35 | 0.59 |
| 1025 | 4096 | torch.float16 | 0.02 | 1.35 | 2.25 |

### baseline

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.03 | 0.80 | 1.33 |
| 4096 | 4096 | torch.bfloat16 | 0.11 | 0.96 | 1.59 |
| 1024 | 3000 | torch.float16 | 0.03 | 0.72 | 1.20 |
| 1025 | 4096 | torch.float16 | 0.03 | 0.80 | 1.34 |

## argreduce

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | argmax | 3.06 | 0.00 | 0.00 |
| 1024 | 4096 | torch.bfloat16 | argmax | 3.07 | 0.00 | 0.00 |
| 4096 | 4096 | torch.float16 | argmax | 10.71 | 0.00 | 0.00 |
| 1024 | 4096 | torch.float16 | argmin | 3.46 | 0.00 | 0.00 |
| 1024 | 4096 | torch.bfloat16 | argmin | 5.17 | 0.00 | 0.00 |
| 4096 | 4096 | torch.float16 | argmin | 12.11 | 0.00 | 0.00 |

### baseline

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | argmax | 0.02 | 0.19 | 0.38 |
| 1024 | 4096 | torch.bfloat16 | argmax | 0.02 | 0.19 | 0.37 |
| 4096 | 4096 | torch.float16 | argmax | 0.03 | 0.59 | 1.18 |
| 1024 | 4096 | torch.float16 | argmin | 0.02 | 0.18 | 0.36 |
| 1024 | 4096 | torch.bfloat16 | argmin | 0.02 | 0.19 | 0.37 |
| 4096 | 4096 | torch.float16 | argmin | 0.03 | 0.59 | 1.18 |

## batch_norm_fwd

### tileops

| N | C | spatial | dtype | training | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 64 | () | torch.float16 | True | 0.01 | 0.00 | 0.00 |
| 8 | 64 | (32, 32) | torch.float16 | True | 0.01 | 0.43 | 0.17 |
| 4 | 128 | (32, 32) | torch.float16 | True | 0.01 | 0.43 | 0.17 |
| 4 | 256 | (28, 28) | torch.float16 | True | 0.02 | 0.50 | 0.20 |
| 4 | 128 | (1024, 1024) | torch.float16 | True | 3.94 | 1.36 | 0.55 |
| 4 | 256 | (1024, 1024) | torch.float16 | True | 7.34 | 1.46 | 0.59 |

### torch_cudnn

| N | C | spatial | dtype | training | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 64 | () | torch.float16 | True | 0.02 | 0.00 | 0.00 |
| 8 | 64 | (32, 32) | torch.float16 | True | 0.01 | 0.38 | 0.15 |
| 4 | 128 | (32, 32) | torch.float16 | True | 0.01 | 0.40 | 0.16 |
| 4 | 256 | (28, 28) | torch.float16 | True | 0.01 | 0.58 | 0.23 |
| 4 | 128 | (1024, 1024) | torch.float16 | True | 4.12 | 1.30 | 0.52 |
| 4 | 256 | (1024, 1024) | torch.float16 | True | 7.29 | 1.47 | 0.59 |

## batch_norm_bwd

### tileops

| N | C | spatial | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 64 | () | torch.float16 | 0.01 | 0.00 | 0.00 |
| 8 | 64 | (32, 32) | torch.float16 | 0.02 | 0.26 | 0.19 |
| 4 | 128 | (32, 32) | torch.float16 | 0.02 | 0.26 | 0.19 |
| 4 | 256 | (28, 28) | torch.float16 | 0.02 | 0.32 | 0.24 |
| 4 | 128 | (1024, 1024) | torch.float16 | 6.21 | 0.69 | 0.52 |
| 4 | 256 | (1024, 1024) | torch.float16 | 11.29 | 0.76 | 0.57 |

### torch_autograd

| N | C | spatial | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 64 | () | torch.float16 | 0.02 | 0.00 | 0.00 |
| 8 | 64 | (32, 32) | torch.float16 | 0.03 | 0.16 | 0.12 |
| 4 | 128 | (32, 32) | torch.float16 | 0.02 | 0.19 | 0.14 |
| 4 | 256 | (28, 28) | torch.float16 | 0.02 | 0.27 | 0.20 |
| 4 | 128 | (1024, 1024) | torch.float16 | 12.08 | 0.36 | 0.27 |
| 4 | 256 | (1024, 1024) | torch.float16 | 18.14 | 0.47 | 0.36 |

## r1_vectorization

### add_same_shape_1d

| pattern_name | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| same_shape_1d | 1000000 | 0.00 | 0.24 | 1.42 |

### baseline_same_shape_1d

| pattern_name | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| same_shape_1d | 1000000 | 0.00 | 0.24 | 1.46 |

### add_bias_add_2d

| pattern_name | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| bias_add_2d | 1000000 | 0.00 | 0.27 | 1.07 |

### baseline_bias_add_2d

| pattern_name | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| bias_add_2d | 1000000 | 0.01 | 0.17 | 0.66 |

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
| broadcast_3d | 4096 | 0.00 | 0.00 | 0.00 |

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
| 1M | same_shape | direct | 1048576 | 0.01 | 0.16 | 1.93 |
| 16M | same_shape | direct | 16777216 | 0.06 | 0.29 | 1.74 |
| 16M | same_shape | direct | 16777216 | 0.06 | 0.29 | 1.74 |
| 16M | same_shape | direct | 16777216 | 0.07 | 0.25 | 3.00 |

### add_direct_bias_add_2d

| size_label | pattern_name | strategy | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4K | bias_add_2d | direct | 4096 | 0.00 | 0.00 | 0.01 |
| 4K | bias_add_2d | direct | 4096 | 0.00 | 0.00 | 0.01 |
| 4K | bias_add_2d | direct | 4096 | 0.00 | 0.00 | 0.02 |
| 1M | bias_add_2d | direct | 1048576 | 0.01 | 0.20 | 0.79 |
| 1M | bias_add_2d | direct | 1048576 | 0.01 | 0.20 | 0.79 |
| 1M | bias_add_2d | direct | 1048576 | 0.01 | 0.18 | 1.46 |
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
| 1M | interleaved_3d | direct | 1048576 | 0.00 | 0.22 | 0.89 |
| 16M | interleaved_3d | direct | 16777216 | 0.04 | 0.40 | 0.80 |
| 16M | interleaved_3d | direct | 16777216 | 0.04 | 0.40 | 0.80 |
| 16M | interleaved_3d | direct | 16777216 | 0.04 | 0.39 | 1.57 |

### add_explicit_parallel_same_shape

| size_label | pattern_name | strategy | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4K | same_shape | explicit_parallel | 4096 | 0.00 | 0.00 | 0.01 |
| 4K | same_shape | explicit_parallel | 4096 | 0.00 | 0.00 | 0.01 |
| 4K | same_shape | explicit_parallel | 4096 | 0.00 | 0.00 | 0.03 |
| 1M | same_shape | explicit_parallel | 1048576 | 0.00 | 0.26 | 1.58 |
| 1M | same_shape | explicit_parallel | 1048576 | 0.00 | 0.26 | 1.58 |
| 1M | same_shape | explicit_parallel | 1048576 | 0.01 | 0.19 | 2.30 |
| 16M | same_shape | explicit_parallel | 16777216 | 0.03 | 0.61 | 3.68 |
| 16M | same_shape | explicit_parallel | 16777216 | 0.03 | 0.55 | 3.28 |
| 16M | same_shape | explicit_parallel | 16777216 | 0.05 | 0.33 | 4.00 |

### add_explicit_parallel_bias_add_2d

| size_label | pattern_name | strategy | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4K | bias_add_2d | explicit_parallel | 4096 | 0.00 | 0.00 | 0.01 |
| 4K | bias_add_2d | explicit_parallel | 4096 | 0.00 | 0.00 | 0.01 |
| 4K | bias_add_2d | explicit_parallel | 4096 | 0.00 | 0.00 | 0.02 |
| 1M | bias_add_2d | explicit_parallel | 1048576 | 0.00 | 0.31 | 1.23 |
| 1M | bias_add_2d | explicit_parallel | 1048576 | 0.00 | 0.31 | 1.23 |
| 1M | bias_add_2d | explicit_parallel | 1048576 | 0.00 | 0.24 | 1.95 |
| 16M | bias_add_2d | explicit_parallel | 16777216 | 0.02 | 0.90 | 3.61 |
| 16M | bias_add_2d | explicit_parallel | 16777216 | 0.02 | 0.90 | 3.60 |
| 16M | bias_add_2d | explicit_parallel | 16777216 | 0.03 | 0.49 | 3.92 |

### add_explicit_parallel_interleaved_3d

| size_label | pattern_name | strategy | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4K | interleaved_3d | explicit_parallel | 4096 | 0.00 | 0.00 | 0.00 |
| 4K | interleaved_3d | explicit_parallel | 4096 | 0.00 | 0.00 | 0.00 |
| 4K | interleaved_3d | explicit_parallel | 4096 | 0.00 | 0.00 | 0.01 |
| 1M | interleaved_3d | explicit_parallel | 1048576 | 0.00 | 0.47 | 0.94 |
| 1M | interleaved_3d | explicit_parallel | 1048576 | 0.00 | 0.47 | 0.93 |
| 1M | interleaved_3d | explicit_parallel | 1048576 | 0.00 | 0.38 | 1.54 |
| 16M | interleaved_3d | explicit_parallel | 16777216 | 0.01 | 1.71 | 3.43 |
| 16M | interleaved_3d | explicit_parallel | 16777216 | 0.01 | 1.64 | 3.27 |
| 16M | interleaved_3d | explicit_parallel | 16777216 | 0.02 | 0.96 | 3.84 |

## r4_where

### tileops_where

| n_total | size_label | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 4096 | 4K | 0.00 | 0.00 | 0.01 |
| 1048576 | 1M | 0.00 | 0.23 | 1.64 |
| 16777216 | 16M | 0.03 | 0.53 | 3.68 |

### baseline_where

| n_total | size_label | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 4096 | 4K | 0.00 | 0.00 | 0.01 |
| 1048576 | 1M | 0.00 | 0.23 | 1.61 |
| 16777216 | 16M | 0.03 | 0.53 | 3.72 |

## add

### tileops

| n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 4194304 | torch.float16 | 0.01 | 0.46 | 2.77 |
| 4194304 | torch.bfloat16 | 0.01 | 0.46 | 2.77 |
| 4194304 | torch.float32 | 0.02 | 0.27 | 3.25 |

### baseline

| n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 4194304 | torch.float16 | 0.01 | 0.42 | 2.51 |
| 4194304 | torch.bfloat16 | 0.01 | 0.45 | 2.70 |
| 4194304 | torch.float32 | 0.02 | 0.27 | 3.30 |

## sub

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| sub | torch.float16 | torch.float16 | 0.01 | 0.46 | 2.73 |
| sub | torch.float16 | torch.float16 | 0.02 | 0.57 | 3.41 |
| sub | torch.float16 | torch.float16 | 0.03 | 0.63 | 3.80 |

### baseline

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| sub | torch.float16 | torch.float16 | 0.01 | 0.44 | 2.62 |
| sub | torch.float16 | torch.float16 | 0.02 | 0.57 | 3.41 |
| sub | torch.float16 | torch.float16 | 0.03 | 0.63 | 3.80 |

## mul

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| mul | torch.float16 | torch.float16 | 0.01 | 0.46 | 2.78 |
| mul | torch.float16 | torch.float16 | 0.02 | 0.56 | 3.38 |
| mul | torch.float16 | torch.float16 | 0.04 | 0.51 | 3.06 |

### baseline

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| mul | torch.float16 | torch.float16 | 0.01 | 0.46 | 2.74 |
| mul | torch.float16 | torch.float16 | 0.02 | 0.54 | 3.27 |
| mul | torch.float16 | torch.float16 | 0.03 | 0.63 | 3.80 |

## div

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| div | torch.float16 | torch.float16 | 0.01 | 0.46 | 2.78 |
| div | torch.float16 | torch.float16 | 0.02 | 0.56 | 3.39 |
| div | torch.float16 | torch.float16 | 0.03 | 0.63 | 3.79 |

### baseline

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| div | torch.float16 | torch.float16 | 0.01 | 0.45 | 2.71 |
| div | torch.float16 | torch.float16 | 0.02 | 0.56 | 3.34 |
| div | torch.float16 | torch.float16 | 0.03 | 0.63 | 3.76 |

## remainder

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| remainder | torch.float16 | torch.float16 | 0.01 | 0.46 | 2.73 |
| remainder | torch.float16 | torch.float16 | 0.02 | 0.56 | 3.37 |

### baseline

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| remainder | torch.float16 | torch.float16 | 0.01 | 0.40 | 2.42 |
| remainder | torch.float16 | torch.float16 | 0.02 | 0.51 | 3.06 |

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
| pow | torch.float16 | torch.float16 | 0.04 | 0.24 | 1.43 |

## floor_divide

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| floor_divide | torch.float16 | torch.float16 | 0.01 | 0.46 | 2.73 |
| floor_divide | torch.float16 | torch.float16 | 0.02 | 0.56 | 3.38 |

### baseline

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| floor_divide | torch.float16 | torch.float16 | 0.02 | 0.18 | 1.11 |
| floor_divide | torch.float16 | torch.float16 | 0.05 | 0.21 | 1.24 |

## lerp

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| lerp | torch.float16 | torch.float16 | 0.01 | 0.46 | 2.78 |
| lerp | torch.float16 | torch.float16 | 0.02 | 0.57 | 3.41 |

### baseline

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| lerp | torch.float16 | torch.float16 | 0.01 | 0.45 | 2.73 |
| lerp | torch.float16 | torch.float16 | 0.02 | 0.45 | 2.69 |

## maximum

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| maximum | torch.float16 | torch.float16 | 0.01 | 0.45 | 2.70 |
| maximum | torch.float16 | torch.float16 | 0.02 | 0.56 | 3.37 |
| maximum | torch.float16 | torch.float16 | 0.03 | 0.63 | 3.78 |

### baseline

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| maximum | torch.float16 | torch.float16 | 0.01 | 0.43 | 2.61 |
| maximum | torch.float16 | torch.float16 | 0.02 | 0.56 | 3.37 |
| maximum | torch.float16 | torch.float16 | 0.03 | 0.63 | 3.78 |

## minimum

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| minimum | torch.float16 | torch.float16 | 0.01 | 0.45 | 2.70 |
| minimum | torch.float16 | torch.float16 | 0.04 | 0.28 | 1.70 |
| minimum | torch.float16 | torch.float16 | 0.03 | 0.63 | 3.78 |

### baseline

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| minimum | torch.float16 | torch.float16 | 0.01 | 0.45 | 2.68 |
| minimum | torch.float16 | torch.float16 | 0.02 | 0.56 | 3.37 |
| minimum | torch.float16 | torch.float16 | 0.03 | 0.63 | 3.77 |

## cmp_eq

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| eq | torch.float16 | 0.02 | 0.22 | 1.10 |
| eq | torch.float16 | 0.04 | 0.26 | 1.31 |

### baseline

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| eq | torch.float16 | 0.01 | 0.51 | 2.57 |
| eq | torch.float16 | 0.02 | 0.63 | 3.17 |

## cmp_ne

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| ne | torch.float16 | 0.02 | 0.22 | 1.09 |

### baseline

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| ne | torch.float16 | 0.01 | 0.51 | 2.53 |

## cmp_gt

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| gt | torch.float16 | 0.02 | 0.18 | 0.88 |

### baseline

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| gt | torch.float16 | 0.01 | 0.51 | 2.55 |

## cmp_lt

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| lt | torch.float16 | 0.02 | 0.22 | 1.10 |

### baseline

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| lt | torch.float16 | 0.01 | 0.51 | 2.54 |

## cmp_ge

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| ge | torch.float16 | 0.02 | 0.22 | 1.10 |

### baseline

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| ge | torch.float16 | 0.01 | 0.51 | 2.55 |

## cmp_le

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| le | torch.float16 | 0.02 | 0.22 | 1.10 |

### baseline

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| le | torch.float16 | 0.02 | 0.24 | 1.21 |

## logical_and

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| logical_and | torch.float16 | 0.02 | 0.18 | 0.91 |
| logical_and | torch.float16 | 0.06 | 0.18 | 0.91 |

### baseline

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| logical_and | torch.float16 | 0.02 | 0.24 | 1.22 |
| logical_and | torch.float16 | 0.01 | 0.94 | 4.70 |

## logical_or

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| logical_or | torch.float16 | 0.02 | 0.22 | 1.10 |
| logical_or | torch.float16 | 0.04 | 0.25 | 1.25 |

### baseline

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| logical_or | torch.float16 | 0.01 | 0.71 | 3.55 |
| logical_or | torch.float16 | 0.01 | 0.92 | 4.62 |

## bitwise_and

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| bitwise_and | torch.int32 | 0.02 | 0.27 | 3.23 |
| bitwise_and | torch.int32 | 0.08 | 0.14 | 1.66 |

### baseline

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| bitwise_and | torch.int32 | 0.04 | 0.12 | 1.41 |
| bitwise_and | torch.int32 | 0.08 | 0.14 | 1.68 |

## bitwise_or

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| bitwise_or | torch.int32 | 0.03 | 0.12 | 1.45 |

### baseline

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| bitwise_or | torch.int32 | 0.03 | 0.12 | 1.46 |

## bitwise_xor

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| bitwise_xor | torch.int32 | 0.04 | 0.12 | 1.40 |

### baseline

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| bitwise_xor | torch.int32 | 0.03 | 0.12 | 1.45 |

## gelu_and_mul

### tileops

| op_name | M | N | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| gelu_and_mul | 1024 | 4096 | torch.float16 | 0.02 | 0.38 | 1.15 |
| gelu_and_mul | 1024 | 10240 | torch.float16 | 0.05 | 0.43 | 1.29 |
| gelu_and_mul | 1024 | 20480 | torch.float16 | 0.04 | 1.04 | 3.12 |

### baseline

| op_name | M | N | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| gelu_and_mul | 1024 | 4096 | torch.float16 | 0.07 | 0.12 | 0.36 |
| gelu_and_mul | 1024 | 10240 | torch.float16 | 0.15 | 0.14 | 0.43 |
| gelu_and_mul | 1024 | 20480 | torch.float16 | 0.13 | 0.32 | 0.96 |

## gelu_tanh_and_mul

### tileops

| op_name | M | N | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| gelu_tanh_and_mul | 1024 | 4096 | torch.float16 | 0.01 | 0.93 | 2.80 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.float16 | 0.02 | 1.13 | 3.40 |
| gelu_tanh_and_mul | 1024 | 20480 | torch.float16 | 0.03 | 1.24 | 3.71 |

### baseline

| op_name | M | N | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| gelu_tanh_and_mul | 1024 | 4096 | torch.float16 | 0.03 | 0.28 | 0.83 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.float16 | 0.07 | 0.31 | 0.94 |
| gelu_tanh_and_mul | 1024 | 20480 | torch.float16 | 0.13 | 0.32 | 0.96 |

## silu_and_mul_strategy

### tileops_direct

| op_name | M | N | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| silu_and_mul | 1024 | 4096 | torch.float16 | direct | 0.02 | 0.47 | 1.41 |
| silu_and_mul | 1024 | 4096 | torch.bfloat16 | direct | 0.04 | 0.20 | 0.61 |
| silu_and_mul | 1024 | 4096 | torch.float32 | direct | 0.04 | 0.21 | 1.23 |
| silu_and_mul | 1024 | 10240 | torch.float16 | direct | 0.08 | 0.25 | 0.75 |
| silu_and_mul | 1024 | 10240 | torch.bfloat16 | direct | 0.08 | 0.27 | 0.82 |
| silu_and_mul | 1024 | 10240 | torch.float32 | direct | 0.09 | 0.23 | 1.38 |
| silu_and_mul | 4096 | 4096 | torch.float16 | direct | 0.14 | 0.24 | 0.71 |
| silu_and_mul | 4096 | 4096 | torch.bfloat16 | direct | 0.14 | 0.24 | 0.71 |
| silu_and_mul | 4096 | 4096 | torch.float32 | direct | 0.16 | 0.20 | 1.22 |

### tileops_explicit_parallel

| op_name | M | N | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| silu_and_mul | 1024 | 4096 | torch.float16 | explicit_parallel | 0.02 | 0.46 | 1.37 |
| silu_and_mul | 1024 | 4096 | torch.bfloat16 | explicit_parallel | 0.02 | 0.41 | 1.22 |
| silu_and_mul | 1024 | 4096 | torch.float32 | explicit_parallel | 0.04 | 0.23 | 1.36 |
| silu_and_mul | 1024 | 10240 | torch.float16 | explicit_parallel | 0.05 | 0.44 | 1.31 |
| silu_and_mul | 1024 | 10240 | torch.bfloat16 | explicit_parallel | 0.04 | 0.47 | 1.42 |
| silu_and_mul | 1024 | 10240 | torch.float32 | explicit_parallel | 0.08 | 0.26 | 1.57 |
| silu_and_mul | 4096 | 4096 | torch.float16 | explicit_parallel | 0.07 | 0.46 | 1.38 |
| silu_and_mul | 4096 | 4096 | torch.bfloat16 | explicit_parallel | 0.06 | 0.60 | 1.81 |
| silu_and_mul | 4096 | 4096 | torch.float32 | explicit_parallel | 0.10 | 0.32 | 1.94 |

## gelu_and_mul_strategy

### tileops_direct

| op_name | M | N | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gelu_and_mul | 1024 | 4096 | torch.float16 | direct | 0.04 | 0.21 | 0.62 |
| gelu_and_mul | 1024 | 4096 | torch.bfloat16 | direct | 0.02 | 0.35 | 1.06 |
| gelu_and_mul | 1024 | 4096 | torch.float32 | direct | 0.05 | 0.18 | 1.10 |
| gelu_and_mul | 1024 | 10240 | torch.float16 | direct | 0.08 | 0.25 | 0.74 |
| gelu_and_mul | 1024 | 10240 | torch.bfloat16 | direct | 0.08 | 0.25 | 0.74 |
| gelu_and_mul | 1024 | 10240 | torch.float32 | direct | 0.09 | 0.24 | 1.41 |
| gelu_and_mul | 4096 | 4096 | torch.float16 | direct | 0.18 | 0.19 | 0.57 |
| gelu_and_mul | 4096 | 4096 | torch.bfloat16 | direct | 0.16 | 0.21 | 0.62 |
| gelu_and_mul | 4096 | 4096 | torch.float32 | direct | 0.16 | 0.21 | 1.28 |

### tileops_explicit_parallel

| op_name | M | N | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gelu_and_mul | 1024 | 4096 | torch.float16 | explicit_parallel | 0.03 | 0.31 | 0.92 |
| gelu_and_mul | 1024 | 4096 | torch.bfloat16 | explicit_parallel | 0.02 | 0.42 | 1.27 |
| gelu_and_mul | 1024 | 4096 | torch.float32 | explicit_parallel | 0.04 | 0.21 | 1.24 |
| gelu_and_mul | 1024 | 10240 | torch.float16 | explicit_parallel | 0.04 | 0.49 | 1.47 |
| gelu_and_mul | 1024 | 10240 | torch.bfloat16 | explicit_parallel | 0.05 | 0.43 | 1.29 |
| gelu_and_mul | 1024 | 10240 | torch.float32 | explicit_parallel | 0.08 | 0.26 | 1.55 |
| gelu_and_mul | 4096 | 4096 | torch.float16 | explicit_parallel | 0.08 | 0.44 | 1.31 |
| gelu_and_mul | 4096 | 4096 | torch.bfloat16 | explicit_parallel | 0.07 | 0.51 | 1.53 |
| gelu_and_mul | 4096 | 4096 | torch.float32 | explicit_parallel | 0.11 | 0.31 | 1.85 |

## gelu_tanh_and_mul_strategy

### tileops_direct

| op_name | M | N | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gelu_tanh_and_mul | 1024 | 4096 | torch.float16 | direct | 0.04 | 0.20 | 0.59 |
| gelu_tanh_and_mul | 1024 | 4096 | torch.bfloat16 | direct | 0.04 | 0.19 | 0.56 |
| gelu_tanh_and_mul | 1024 | 4096 | torch.float32 | direct | 0.04 | 0.20 | 1.19 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.float16 | direct | 0.09 | 0.23 | 0.70 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.bfloat16 | direct | 0.05 | 0.40 | 1.19 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.float32 | direct | 0.04 | 0.48 | 2.87 |
| gelu_tanh_and_mul | 4096 | 4096 | torch.float16 | direct | 0.06 | 0.55 | 1.66 |
| gelu_tanh_and_mul | 4096 | 4096 | torch.bfloat16 | direct | 0.06 | 0.55 | 1.66 |
| gelu_tanh_and_mul | 4096 | 4096 | torch.float32 | direct | 0.07 | 0.50 | 2.99 |

### tileops_explicit_parallel

| op_name | M | N | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gelu_tanh_and_mul | 1024 | 4096 | torch.float16 | explicit_parallel | 0.02 | 0.44 | 1.32 |
| gelu_tanh_and_mul | 1024 | 4096 | torch.bfloat16 | explicit_parallel | 0.02 | 0.40 | 1.20 |
| gelu_tanh_and_mul | 1024 | 4096 | torch.float32 | explicit_parallel | 0.04 | 0.22 | 1.34 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.float16 | explicit_parallel | 0.05 | 0.43 | 1.30 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.bfloat16 | explicit_parallel | 0.02 | 1.12 | 3.35 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.float32 | explicit_parallel | 0.03 | 0.64 | 3.83 |
| gelu_tanh_and_mul | 4096 | 4096 | torch.float16 | explicit_parallel | 0.03 | 1.20 | 3.59 |
| gelu_tanh_and_mul | 4096 | 4096 | torch.bfloat16 | explicit_parallel | 0.03 | 1.18 | 3.54 |
| gelu_tanh_and_mul | 4096 | 4096 | torch.float32 | explicit_parallel | 0.05 | 0.67 | 4.01 |

## sub_bcast

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| sub | torch.float16 | 0.01 | 0.66 | 2.66 |
| sub | torch.float16 | 0.01 | 0.82 | 3.26 |
| sub | torch.float16 | 0.02 | 0.93 | 3.71 |

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
| mul | torch.float16 | 0.01 | 0.66 | 2.66 |
| mul | torch.float16 | 0.01 | 0.82 | 3.27 |
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
| div | torch.float16 | 0.01 | 0.62 | 2.47 |
| div | torch.float16 | 0.01 | 0.76 | 3.06 |
| div | torch.float16 | 0.02 | 0.84 | 3.36 |

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
| 4194304 | 1024x4096 | torch.float16 | direct | 0.02 | 0.26 | 1.54 |
| 4194304 | 1024x4096 | torch.bfloat16 | direct | 0.02 | 0.26 | 1.54 |
| 4194304 | 1024x4096 | torch.float32 | direct | 0.02 | 0.22 | 2.59 |
| 10485760 | 1024x10240 | torch.float16 | direct | 0.04 | 0.28 | 1.68 |
| 10485760 | 1024x10240 | torch.bfloat16 | direct | 0.04 | 0.28 | 1.68 |
| 10485760 | 1024x10240 | torch.float32 | direct | 0.04 | 0.24 | 2.90 |
| 20971520 | 1024x20480 | torch.float16 | direct | 0.07 | 0.29 | 1.76 |
| 20971520 | 1024x20480 | torch.bfloat16 | direct | 0.07 | 0.30 | 1.77 |
| 20971520 | 1024x20480 | torch.float32 | direct | 0.08 | 0.26 | 3.08 |

### add_explicit_parallel

| n_total | shape_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4194304 | 1024x4096 | torch.float16 | explicit_parallel | 0.01 | 0.46 | 2.78 |
| 4194304 | 1024x4096 | torch.bfloat16 | explicit_parallel | 0.01 | 0.46 | 2.78 |
| 4194304 | 1024x4096 | torch.float32 | explicit_parallel | 0.02 | 0.27 | 3.25 |
| 10485760 | 1024x10240 | torch.float16 | explicit_parallel | 0.02 | 0.57 | 3.42 |
| 10485760 | 1024x10240 | torch.bfloat16 | explicit_parallel | 0.02 | 0.57 | 3.42 |
| 10485760 | 1024x10240 | torch.float32 | explicit_parallel | 0.03 | 0.32 | 3.80 |
| 20971520 | 1024x20480 | torch.float16 | explicit_parallel | 0.03 | 0.63 | 3.80 |
| 20971520 | 1024x20480 | torch.bfloat16 | explicit_parallel | 0.03 | 0.63 | 3.80 |
| 20971520 | 1024x20480 | torch.float32 | explicit_parallel | 0.06 | 0.34 | 4.04 |

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
| 1024 | 4096 | torch.bfloat16 | cumsum | 0.11 | 0.04 | 0.16 |
| 4096 | 4096 | torch.float16 | cumsum | 0.26 | 0.07 | 0.26 |
| 1024 | 4096 | torch.float16 | cumprod | 0.10 | 0.04 | 0.17 |
| 1024 | 4096 | torch.bfloat16 | cumprod | 0.10 | 0.04 | 0.17 |
| 4096 | 4096 | torch.float16 | cumprod | 0.26 | 0.07 | 0.26 |

## dsa_decode

### tileops

| batch | heads | seq_len_q | seq_len_kv | dim | dim_tail | topk | stride_kv | heads_kv | q_start_index_s | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 128 | 1024 | 2048 | 512 | 64 | 2048 | 1 | 1 | 1024 | torch.float16 | 1.14 | 513.02 | 0.26 |
| 1 | 128 | 512 | 4096 | 512 | 64 | 1024 | 1 | 1 | 512 | torch.float16 | 0.32 | 462.26 | 0.47 |

### baseline

| batch | heads | seq_len_q | seq_len_kv | dim | dim_tail | topk | stride_kv | heads_kv | q_start_index_s | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 128 | 1024 | 2048 | 512 | 64 | 2048 | 1 | 1 | 1024 | torch.float16 | 23.74 | 24.60 | 0.01 |
| 1 | 128 | 512 | 4096 | 512 | 64 | 1024 | 1 | 1 | 512 | torch.float16 | 16.72 | 8.74 | 0.01 |

## mla_decode

### tileops

| batch | heads | heads_kv | seq_len_kv | dim | dim_pe | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 128 | 1 | 8192 | 512 | 64 | torch.float16 | 0.17 | 440.48 | 1.85 |
| 16 | 128 | 1 | 4096 | 512 | 64 | torch.float16 | 0.05 | 379.52 | 1.62 |

### baseline

| batch | heads | heads_kv | seq_len_kv | dim | dim_pe | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 128 | 1 | 8192 | 512 | 64 | torch.float16 | 0.72 | 101.32 | 0.43 |
| 16 | 128 | 1 | 4096 | 512 | 64 | torch.float16 | 0.19 | 95.58 | 0.41 |

## nsa_cmp_fwd

### tileops

| seq_num | c_seq_len | heads | dim_k | dim_v | group | scale | bc | bs | bk | bv | dtype | accum_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 9 | 8192 | 32 | 128 | 128 | 16 | 0.08838834764831845 | 32 | 32 | 128 | 128 | torch.float16 | torch.float32 | 164.87 | 0.10 | 0.01 |
| 16 | 16384 | 32 | 128 | 128 | 16 | 0.08838834764831845 | 32 | 32 | 128 | 128 | torch.float16 | torch.float32 | 162.00 | 0.42 | 0.05 |

### baseline

| seq_num | c_seq_len | heads | dim_k | dim_v | group | scale | bc | bs | bk | bv | dtype | accum_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 9 | 8192 | 32 | 128 | 128 | 16 | 0.08838834764831845 | 32 | 32 | 128 | 128 | torch.float16 | torch.float32 | 307.05 | 0.06 | 0.01 |
| 16 | 16384 | 32 | 128 | 128 | 16 | 0.08838834764831845 | 32 | 32 | 128 | 128 | torch.float16 | torch.float32 | 591.94 | 0.12 | 0.01 |

## nsa_fwd

### tileops

| batch | heads | c_seq_len | dim | is_causal | scale | block_size | groups | selected_blocks | dtype | accum_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 16 | 1024 | 64 | True | 0.1 | 32 | 16 | 1 | torch.float16 | torch.float32 | 150.90 | 0.00 | 0.00 |
| 4 | 16 | 8192 | 64 | True | 0.1 | 32 | 16 | 1 | torch.float16 | torch.float32 | 0.04 | 28.56 | 0.95 |
| 2 | 16 | 8192 | 64 | True | 0.1 | 32 | 16 | 4 | torch.float16 | torch.float32 | 165.11 | 0.03 | 0.00 |

### baseline

| batch | heads | c_seq_len | dim | is_causal | scale | block_size | groups | selected_blocks | dtype | accum_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 16 | 1024 | 64 | True | 0.1 | 32 | 16 | 1 | torch.float16 | torch.float32 | 65.25 | 0.00 | 0.00 |
| 4 | 16 | 8192 | 64 | True | 0.1 | 32 | 16 | 1 | torch.float16 | torch.float32 | 522.50 | 0.00 | 0.00 |
| 2 | 16 | 8192 | 64 | True | 0.1 | 32 | 16 | 4 | torch.float16 | torch.float32 | 482.75 | 0.01 | 0.00 |

## nsa_topk

### tileops

| seq_num | c_seq_len | heads | dim | group | scale | selected_block_num | bc | bs | bk | dtype | accum_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 1024 | 32 | 128 | 16 | 1 | 16 | 32 | 32 | 128 | torch.float16 | torch.float32 | 282.87 | 0.00 | 0.00 |
| 3 | 512 | 32 | 128 | 16 | 1 | 16 | 32 | 32 | 128 | torch.float16 | torch.float32 | 260.54 | 0.00 | 0.00 |
| 9 | 8192 | 32 | 128 | 16 | 1 | 16 | 32 | 32 | 128 | torch.float16 | torch.float32 | 271.24 | 0.06 | 0.00 |

### baseline

| seq_num | c_seq_len | heads | dim | group | scale | selected_block_num | bc | bs | bk | dtype | accum_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 1024 | 32 | 128 | 16 | 1 | 16 | 32 | 32 | 128 | torch.float16 | torch.float32 | 244.15 | 0.00 | 0.00 |
| 3 | 512 | 32 | 128 | 16 | 1 | 16 | 32 | 32 | 128 | torch.float16 | torch.float32 | 120.76 | 0.00 | 0.00 |
| 9 | 8192 | 32 | 128 | 16 | 1 | 16 | 32 | 32 | 128 | torch.float16 | torch.float32 | 2523.03 | 0.01 | 0.00 |

## dropout

### tileops

| dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| torch.float16 | 4194304 | 0.01 | 0.66 | 2.65 |
| torch.bfloat16 | 4194304 | 0.01 | 0.66 | 2.65 |
| torch.float32 | 4194304 | 0.01 | 0.40 | 3.18 |
| torch.float16 | 10485760 | 0.01 | 0.84 | 3.36 |
| torch.bfloat16 | 10485760 | 0.01 | 0.84 | 3.36 |
| torch.float32 | 10485760 | 0.02 | 0.48 | 3.81 |
| torch.float16 | 20971520 | 0.02 | 0.94 | 3.78 |
| torch.bfloat16 | 20971520 | 0.02 | 0.94 | 3.78 |
| torch.float32 | 20971520 | 0.04 | 0.51 | 4.10 |

### baseline

| dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| torch.float16 | 4194304 | 0.01 | 0.39 | 1.54 |
| torch.bfloat16 | 4194304 | 0.01 | 0.38 | 1.51 |
| torch.float32 | 4194304 | 0.02 | 0.28 | 2.23 |
| torch.float16 | 10485760 | 0.02 | 0.50 | 2.01 |
| torch.bfloat16 | 10485760 | 0.02 | 0.50 | 1.98 |
| torch.float32 | 10485760 | 0.03 | 0.33 | 2.61 |
| torch.float16 | 20971520 | 0.04 | 0.55 | 2.20 |
| torch.bfloat16 | 20971520 | 0.04 | 0.52 | 2.09 |
| torch.float32 | 20971520 | 0.06 | 0.35 | 2.76 |

## relu_fp8

### tileops

| op_name | n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| relu_fp8 | 8388608 | torch.float8_e4m3fn | 0.01 | 0.57 | 1.15 |
| relu_fp8 | 8388608 | torch.float8_e5m2 | 0.02 | 0.46 | 0.92 |
| relu_fp8 | 67108864 | torch.float8_e4m3fn | 0.09 | 0.72 | 1.44 |
| relu_fp8 | 67108864 | torch.float8_e5m2 | 0.29 | 0.23 | 0.46 |
| relu_fp8 | 134217728 | torch.float8_e4m3fn | 0.18 | 0.74 | 1.49 |
| relu_fp8 | 134217728 | torch.float8_e5m2 | 0.24 | 0.57 | 1.14 |

### baseline

| op_name | n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| relu_fp8 | 8388608 | torch.float8_e4m3fn | 0.05 | 0.19 | 0.37 |
| relu_fp8 | 8388608 | torch.float8_e5m2 | 0.04 | 0.19 | 0.39 |
| relu_fp8 | 67108864 | torch.float8_e4m3fn | 0.51 | 0.13 | 0.26 |
| relu_fp8 | 67108864 | torch.float8_e5m2 | 0.52 | 0.13 | 0.26 |
| relu_fp8 | 134217728 | torch.float8_e4m3fn | 0.64 | 0.21 | 0.42 |
| relu_fp8 | 134217728 | torch.float8_e5m2 | 0.62 | 0.22 | 0.43 |

## exp_fp8

### tileops

| op_name | n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| exp_fp8 | 8388608 | torch.float8_e4m3fn | 0.01 | 0.98 | 1.96 |
| exp_fp8 | 8388608 | torch.float8_e5m2 | 0.02 | 0.46 | 0.91 |
| exp_fp8 | 67108864 | torch.float8_e4m3fn | 0.05 | 1.32 | 2.65 |
| exp_fp8 | 67108864 | torch.float8_e5m2 | 0.12 | 0.54 | 1.09 |
| exp_fp8 | 134217728 | torch.float8_e4m3fn | 0.10 | 1.39 | 2.79 |
| exp_fp8 | 134217728 | torch.float8_e5m2 | 0.24 | 0.57 | 1.14 |

### baseline

| op_name | n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| exp_fp8 | 8388608 | torch.float8_e4m3fn | 0.05 | 0.19 | 0.37 |
| exp_fp8 | 8388608 | torch.float8_e5m2 | 0.04 | 0.19 | 0.39 |
| exp_fp8 | 67108864 | torch.float8_e4m3fn | 0.33 | 0.20 | 0.41 |
| exp_fp8 | 67108864 | torch.float8_e5m2 | 0.32 | 0.21 | 0.42 |
| exp_fp8 | 134217728 | torch.float8_e4m3fn | 0.67 | 0.20 | 0.40 |
| exp_fp8 | 134217728 | torch.float8_e5m2 | 0.63 | 0.21 | 0.43 |

## add_fp8

### tileops

| op_name | n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| add_fp8 | 8388608 | torch.float8_e4m3fn | 0.01 | 0.86 | 2.58 |
| add_fp8 | 8388608 | torch.float8_e5m2 | 0.02 | 0.42 | 1.25 |
| add_fp8 | 67108864 | torch.float8_e4m3fn | 0.06 | 1.15 | 3.46 |
| add_fp8 | 67108864 | torch.float8_e5m2 | 0.13 | 0.51 | 1.52 |
| add_fp8 | 134217728 | torch.float8_e4m3fn | 0.11 | 1.19 | 3.58 |
| add_fp8 | 134217728 | torch.float8_e5m2 | 0.26 | 0.52 | 1.55 |

### baseline

| op_name | n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| add_fp8 | 8388608 | torch.float8_e4m3fn | 0.08 | 0.11 | 0.33 |
| add_fp8 | 8388608 | torch.float8_e5m2 | 0.07 | 0.11 | 0.34 |
| add_fp8 | 67108864 | torch.float8_e4m3fn | 0.58 | 0.12 | 0.35 |
| add_fp8 | 67108864 | torch.float8_e5m2 | 0.54 | 0.13 | 0.38 |
| add_fp8 | 134217728 | torch.float8_e4m3fn | 1.04 | 0.13 | 0.39 |
| add_fp8 | 134217728 | torch.float8_e5m2 | 1.01 | 0.13 | 0.40 |

## silu_and_mul_fp8

### tileops

| op_name | M | N | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| silu_and_mul_fp8 | 2048 | 11008 | torch.float8_e4m3fn | 0.04 | 2.54 | 1.52 |
| silu_and_mul_fp8 | 2048 | 11008 | torch.float8_e5m2 | 0.06 | 1.76 | 1.06 |
| silu_and_mul_fp8 | 16384 | 11008 | torch.float8_e4m3fn | 0.30 | 2.96 | 1.78 |
| silu_and_mul_fp8 | 16384 | 11008 | torch.float8_e5m2 | 0.47 | 1.91 | 1.15 |
| silu_and_mul_fp8 | 16384 | 28672 | torch.float8_e4m3fn | 0.77 | 3.03 | 1.82 |
| silu_and_mul_fp8 | 16384 | 28672 | torch.float8_e5m2 | 1.08 | 2.18 | 1.31 |

### baseline

| op_name | M | N | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| silu_and_mul_fp8 | 2048 | 11008 | torch.float8_e4m3fn | 0.29 | 0.38 | 0.23 |
| silu_and_mul_fp8 | 2048 | 11008 | torch.float8_e5m2 | 0.29 | 0.39 | 0.24 |
| silu_and_mul_fp8 | 16384 | 11008 | torch.float8_e4m3fn | 1.90 | 0.47 | 0.28 |
| silu_and_mul_fp8 | 16384 | 11008 | torch.float8_e5m2 | 1.96 | 0.46 | 0.28 |
| silu_and_mul_fp8 | 16384 | 28672 | torch.float8_e4m3fn | 4.25 | 0.55 | 0.33 |
| silu_and_mul_fp8 | 16384 | 28672 | torch.float8_e5m2 | 3.84 | 0.61 | 0.37 |

## engram_gate_conv_bwd

### tileops

| M | seq_len | d | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 256 | torch.float16 | 431.43 | 0.00 | 0.00 |
| 2 | 64 | 512 | torch.float16 | 310.88 | 0.00 | 0.00 |
| 1 | 128 | 256 | torch.bfloat16 | 428.37 | 0.00 | 0.00 |
| 2 | 16 | 256 | torch.bfloat16 | 309.73 | 0.00 | 0.00 |

## engram_decode

### tileops

| batch | d_mem | d | max_conv_len | conv_kernel_size | dilation | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 512 | 256 | 12 | 4 | 3 | torch.float16 | 208.85 | 0.00 | 0.00 |
| 4 | 1024 | 512 | 20 | 4 | 5 | torch.float16 | 0.07 | 0.12 | 0.03 |
| 8 | 512 | 256 | 18 | 4 | 3 | torch.bfloat16 | 209.62 | 0.00 | 0.00 |

### baseline

| batch | d_mem | d | max_conv_len | conv_kernel_size | dilation | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 512 | 256 | 12 | 4 | 3 | torch.float16 | 0.10 | 0.01 | 0.01 |
| 4 | 1024 | 512 | 20 | 4 | 5 | torch.float16 | 0.13 | 0.07 | 0.02 |
| 8 | 512 | 256 | 18 | 4 | 3 | torch.bfloat16 | 0.12 | 0.04 | 0.01 |

## engram_gate_conv_fwd

### tileops

| M | seq_len | d | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 256 | torch.float16 | 204.89 | 0.00 | 0.00 |
| 2 | 64 | 512 | torch.float16 | 0.01 | 0.30 | 0.13 |
| 1 | 128 | 256 | torch.bfloat16 | 207.66 | 0.00 | 0.00 |
| 2 | 16 | 256 | torch.bfloat16 | 207.39 | 0.00 | 0.00 |

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
| 64 | torch.complex64 | 213.22 | 0.00 | 0.00 |
| 128 | torch.complex64 | 187.75 | 0.00 | 0.00 |
| 256 | torch.complex64 | 194.91 | 0.00 | 0.00 |
| 512 | torch.complex64 | 186.17 | 0.00 | 0.00 |
| 1024 | torch.complex64 | 229.94 | 0.00 | 0.00 |

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
| 64 | torch.complex64 | 848.33 | 0.00 | 0.00 |
| 128 | torch.complex64 | 901.92 | 0.00 | 0.00 |
| 256 | torch.complex64 | 838.73 | 0.00 | 0.00 |
| 512 | torch.complex64 | 979.38 | 0.00 | 0.00 |
| 1024 | torch.complex64 | 949.52 | 0.00 | 0.00 |
| 4096 | torch.complex64 | 854.03 | 0.00 | 0.00 |
| 16384 | torch.complex64 | 0.01 | 0.09 | 0.02 |

### tileops-base

| n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 64 | torch.complex64 | 176.72 | 0.00 | 0.00 |
| 128 | torch.complex64 | 202.86 | 0.00 | 0.00 |
| 256 | torch.complex64 | 219.03 | 0.00 | 0.00 |
| 512 | torch.complex64 | 185.49 | 0.00 | 0.00 |
| 1024 | torch.complex64 | 209.31 | 0.00 | 0.00 |
| 4096 | torch.complex64 | 202.51 | 0.00 | 0.00 |
| 16384 | torch.complex64 | 172.61 | 0.00 | 0.00 |

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

## fp8_lighting_indexer

### tileops

| batch | seq_len | heads | index_dim | seq_len_kv | kv_group | clean_logits | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 4096 | 32 | 64 | 8192 | 1 | True | 108.26 | N/A | 0.00 |
| 1 | 2048 | 16 | 64 | 4096 | 1 | True | 117.27 | N/A | 0.00 |

### baseline

| batch | seq_len | heads | index_dim | seq_len_kv | kv_group | clean_logits | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 4096 | 32 | 64 | 8192 | 1 | True | 9.64 | N/A | 0.01 |
| 1 | 2048 | 16 | 64 | 4096 | 1 | True | 1.52 | N/A | 0.02 |

## fp8_quant

### tileops

| batch | seq_len_kv | kv_group | index_dim | in_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8192 | 1 | 64 | torch.float16 | 60.60 | 0.00 | 0.00 |
| 1 | 8192 | 1 | 64 | torch.bfloat16 | 60.00 | 0.00 | 0.00 |
| 1 | 4096 | 1 | 128 | torch.float32 | 60.06 | 0.00 | 0.00 |
| 1 | 16384 | 1 | 32 | torch.float32 | 59.65 | 0.00 | 0.00 |

### baseline

| batch | seq_len_kv | kv_group | index_dim | in_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8192 | 1 | 64 | torch.float16 | 0.02 | 0.18 | 0.06 |
| 1 | 8192 | 1 | 64 | torch.bfloat16 | 0.02 | 0.17 | 0.06 |
| 1 | 4096 | 1 | 128 | torch.float32 | 0.02 | 0.18 | 0.12 |
| 1 | 16384 | 1 | 32 | torch.float32 | 0.02 | 0.15 | 0.10 |

## fused_add_layer_norm

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 102.59 | 0.00 | 0.00 |
| 4096 | 4096 | torch.bfloat16 | 103.24 | 0.00 | 0.00 |
| 1024 | 3000 | torch.float16 | 103.20 | 0.00 | 0.00 |
| 1025 | 4096 | torch.float16 | 102.17 | 0.00 | 0.00 |

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
| 1024 | 4096 | torch.float16 | 71.04 | 0.00 | 0.00 |
| 4096 | 4096 | torch.bfloat16 | 70.80 | 0.00 | 0.00 |
| 2048 | 5120 | torch.float16 | 72.06 | 0.00 | 0.00 |
| 1025 | 4096 | torch.float16 | 72.82 | 0.00 | 0.00 |

### baseline

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.13 | 0.17 | 0.27 |
| 4096 | 4096 | torch.bfloat16 | 0.48 | 0.17 | 0.28 |
| 2048 | 5120 | torch.float16 | 0.32 | 0.17 | 0.26 |
| 1025 | 4096 | torch.float16 | 0.13 | 0.17 | 0.27 |

## gated_deltanet_fwd

### tileops

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 0.19 | 1.41 | 0.09 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 0.20 | 1.36 | 0.09 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 0.08 | 1.67 | 0.10 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 0.14 | 1.94 | 0.12 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 0.28 | 1.91 | 0.12 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 0.53 | 2.01 | 0.13 |
| 2 | 32768 | 4 | 64 | 64 | 64 | torch.float16 | 1.00 | 2.15 | 0.14 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.08 | 1.66 | 0.10 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.14 | 1.95 | 0.12 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.28 | 1.93 | 0.12 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.53 | 2.04 | 0.13 |
| 2 | 32768 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.99 | 2.18 | 0.14 |

### baseline

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 130.11 | 0.00 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 131.33 | 0.00 | 0.00 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 44.95 | 0.00 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 90.21 | 0.00 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 182.55 | 0.00 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 364.88 | 0.00 | 0.00 |
| 2 | 32768 | 4 | 64 | 64 | 64 | torch.float16 | 724.31 | 0.00 | 0.00 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 45.11 | 0.00 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 90.89 | 0.00 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 181.97 | 0.00 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 364.84 | 0.00 | 0.00 |
| 2 | 32768 | 4 | 64 | 64 | 64 | torch.bfloat16 | 728.85 | 0.00 | 0.00 |

## gated_deltanet_bwd

### tileops

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | H | S | DK | DV | BC | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.42 | 1.28 | 0.07 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.42 | 1.28 | 0.07 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.19 | 1.43 | 0.08 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.35 | 1.55 | 0.09 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.62 | 1.72 | 0.09 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 16384 | 64 | 64 | 64 | 1.18 | 1.81 | 0.10 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.19 | 1.42 | 0.08 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.35 | 1.54 | 0.09 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.64 | 1.68 | 0.09 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 16384 | 64 | 64 | 64 | 1.20 | 1.79 | 0.10 |

### baseline

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | H | S | DK | DV | BC | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 32 | 50.33 | 0.01 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 32 | 50.19 | 0.01 | 0.00 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 2048 | 64 | 64 | 64 | 19.12 | 0.01 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 64 | 39.61 | 0.01 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 8192 | 64 | 64 | 64 | 82.41 | 0.01 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 16384 | 64 | 64 | 64 | 175.71 | 0.01 | 0.00 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 2048 | 64 | 64 | 64 | 19.11 | 0.01 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 64 | 39.50 | 0.01 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 8192 | 64 | 64 | 64 | 82.76 | 0.01 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 16384 | 64 | 64 | 64 | 175.59 | 0.01 | 0.00 |

## gated_deltanet_fwdbwd

### tileops

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | H | S | DK | DV | BC | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.60 | 1.33 | 0.08 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.60 | 1.34 | 0.08 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.26 | 1.55 | 0.09 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.47 | 1.70 | 0.10 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.90 | 1.80 | 0.10 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 16384 | 64 | 64 | 64 | 1.64 | 1.96 | 0.11 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.26 | 1.53 | 0.09 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.47 | 1.70 | 0.10 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.89 | 1.82 | 0.10 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 16384 | 64 | 64 | 64 | 1.65 | 1.95 | 0.11 |

### baseline

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | H | S | DK | DV | BC | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 32 | 49.92 | 0.02 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 32 | 50.04 | 0.02 | 0.00 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 2048 | 64 | 64 | 64 | 19.10 | 0.02 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 64 | 39.45 | 0.02 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 8192 | 64 | 64 | 64 | 82.48 | 0.02 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 16384 | 64 | 64 | 64 | 175.40 | 0.02 | 0.00 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 2048 | 64 | 64 | 64 | 19.43 | 0.02 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 64 | 39.70 | 0.02 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 8192 | 64 | 64 | 64 | 82.74 | 0.02 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 16384 | 64 | 64 | 64 | 175.93 | 0.02 | 0.00 |

## gated_deltanet_decode

### tileops

| batch | heads | dim_k | dim_v | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 128 | 128 | torch.float32 | 0.01 | 0.30 | 0.41 |
| 1 | 32 | 128 | 128 | torch.bfloat16 | 0.01 | 0.32 | 0.21 |
| 8 | 32 | 128 | 128 | torch.float32 | 0.02 | 1.20 | 1.62 |
| 8 | 32 | 128 | 128 | torch.bfloat16 | 0.01 | 2.12 | 1.43 |
| 16 | 32 | 128 | 128 | torch.float32 | 0.04 | 1.28 | 1.72 |
| 16 | 32 | 128 | 128 | torch.bfloat16 | 0.02 | 3.02 | 2.04 |
| 32 | 32 | 128 | 128 | torch.float32 | 0.08 | 1.34 | 1.82 |
| 32 | 32 | 128 | 128 | torch.bfloat16 | 0.03 | 3.35 | 2.26 |
| 1 | 32 | 64 | 64 | torch.float32 | 0.01 | 0.14 | 0.19 |
| 8 | 32 | 64 | 64 | torch.float32 | 0.01 | 0.59 | 0.81 |
| 32 | 32 | 64 | 64 | torch.float32 | 0.04 | 0.71 | 0.98 |
| 64 | 32 | 128 | 128 | torch.float32 | 0.14 | 1.41 | 1.90 |
| 64 | 32 | 128 | 128 | torch.bfloat16 | 0.05 | 3.67 | 2.48 |

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
| 64 | 32 | 128 | 128 | torch.bfloat16 | 0.55 | 0.36 | 0.25 |

## gemm

### tileops

| m | n | k | dtype | trans_a | trans_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1024 | 1024 | 1024 | torch.float16 | False | False | 42.97 | 0.05 | 0.00 |
| 1 | 7168 | 16384 | torch.float16 | False | True | 0.06 | 3.62 | 3.62 |
| 1 | 18432 | 7168 | torch.bfloat16 | False | True | 0.07 | 3.77 | 3.77 |
| 7168 | 1 | 16384 | torch.float16 | False | False | 0.06 | 3.63 | 3.63 |
| 18432 | 1 | 7168 | torch.bfloat16 | False | False | 0.07 | 3.85 | 3.85 |

### baseline

| m | n | k | dtype | trans_a | trans_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1024 | 1024 | 1024 | torch.float16 | False | False | 0.01 | 322.60 | 0.95 |
| 1 | 7168 | 16384 | torch.float16 | False | True | 0.07 | 3.44 | 3.44 |
| 1 | 18432 | 7168 | torch.bfloat16 | False | True | 0.07 | 3.64 | 3.64 |
| 7168 | 1 | 16384 | torch.float16 | False | False | 0.07 | 3.45 | 3.45 |
| 18432 | 1 | 7168 | torch.bfloat16 | False | False | 0.07 | 3.59 | 3.60 |

## gla_fwd

### tileops

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.08 | 1.73 | 0.11 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.13 | 1.99 | 0.12 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.26 | 2.05 | 0.13 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.50 | 2.16 | 0.14 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.08 | 1.67 | 0.10 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.13 | 1.99 | 0.12 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.26 | 2.07 | 0.13 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.49 | 2.18 | 0.14 |

### baseline

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 1.99 | 0.07 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 4.04 | 0.07 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 8.09 | 0.07 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 16.18 | 0.07 | 0.00 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 1.99 | 0.07 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 4.04 | 0.07 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 8.11 | 0.07 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 16.15 | 0.07 | 0.00 |

## gla_bwd

### tileops

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | T | H | K | V | BC | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 2048 | 4 | 64 | 64 | 64 | 0.125 | 0.15 | 1.75 | 0.10 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4096 | 4 | 64 | 64 | 64 | 0.125 | 0.30 | 1.82 | 0.10 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 8192 | 4 | 64 | 64 | 64 | 0.125 | 0.55 | 1.94 | 0.11 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 16384 | 4 | 64 | 64 | 64 | 0.125 | 0.99 | 2.17 | 0.12 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 2048 | 4 | 64 | 64 | 64 | 0.125 | 0.15 | 1.76 | 0.10 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4096 | 4 | 64 | 64 | 64 | 0.125 | 0.30 | 1.81 | 0.10 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 8192 | 4 | 64 | 64 | 64 | 0.125 | 0.56 | 1.91 | 0.10 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 16384 | 4 | 64 | 64 | 64 | 0.125 | 1.13 | 1.90 | 0.10 |

### baseline

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | T | H | K | V | BC | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 2048 | 4 | 64 | 64 | 64 | 0.125 | 5.95 | 0.05 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4096 | 4 | 64 | 64 | 64 | 0.125 | 12.50 | 0.04 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 8192 | 4 | 64 | 64 | 64 | 0.125 | 28.55 | 0.04 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 16384 | 4 | 64 | 64 | 64 | 0.125 | 70.24 | 0.03 | 0.00 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 2048 | 4 | 64 | 64 | 64 | 0.125 | 5.93 | 0.05 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4096 | 4 | 64 | 64 | 64 | 0.125 | 12.48 | 0.04 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 8192 | 4 | 64 | 64 | 64 | 0.125 | 28.53 | 0.04 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 16384 | 4 | 64 | 64 | 64 | 0.125 | 70.17 | 0.03 | 0.00 |

## gla_fwdbwd

### tileops

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | T | B | BC | H | K | V | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2048 | 2 | 64 | 4 | 64 | 64 | 0.125 | 0.22 | 1.80 | 0.10 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 4096 | 2 | 64 | 4 | 64 | 64 | 0.125 | 0.42 | 1.91 | 0.11 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 8192 | 2 | 64 | 4 | 64 | 64 | 0.125 | 0.78 | 2.06 | 0.12 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 16384 | 2 | 64 | 4 | 64 | 64 | 0.125 | 1.36 | 2.37 | 0.14 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2048 | 2 | 64 | 4 | 64 | 64 | 0.125 | 0.23 | 1.78 | 0.10 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 4096 | 2 | 64 | 4 | 64 | 64 | 0.125 | 0.42 | 1.90 | 0.11 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 8192 | 2 | 64 | 4 | 64 | 64 | 0.125 | 0.79 | 2.04 | 0.12 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 16384 | 2 | 64 | 4 | 64 | 64 | 0.125 | 1.39 | 2.32 | 0.13 |

### baseline

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | T | B | BC | H | K | V | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2048 | 2 | 64 | 4 | 64 | 64 | 0.125 | 5.96 | 0.07 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 4096 | 2 | 64 | 4 | 64 | 64 | 0.125 | 12.47 | 0.06 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 8192 | 2 | 64 | 4 | 64 | 64 | 0.125 | 28.55 | 0.06 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 16384 | 2 | 64 | 4 | 64 | 64 | 0.125 | 70.32 | 0.05 | 0.00 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2048 | 2 | 64 | 4 | 64 | 64 | 0.125 | 5.96 | 0.07 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 4096 | 2 | 64 | 4 | 64 | 64 | 0.125 | 12.47 | 0.06 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 8192 | 2 | 64 | 4 | 64 | 64 | 0.125 | 28.55 | 0.06 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 16384 | 2 | 64 | 4 | 64 | 64 | 0.125 | 70.25 | 0.05 | 0.00 |

## gla_decode

### tileops

| batch | heads | dim_k | dim_v | dtype | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 64 | 64 | torch.float32 | 0.125 | 0.01 | 0.07 | 0.14 |
| 1 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.02 | 0.13 | 0.26 |
| 1 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.01 | 0.17 | 0.17 |
| 1 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.01 | 0.17 | 0.17 |
| 8 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.03 | 0.50 | 1.01 |
| 8 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.01 | 1.21 | 1.22 |
| 8 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.01 | 1.19 | 1.21 |
| 16 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.06 | 0.52 | 1.06 |
| 16 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.02 | 1.86 | 1.89 |
| 16 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.02 | 1.84 | 1.87 |
| 32 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.12 | 0.55 | 1.11 |
| 32 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.03 | 1.97 | 2.00 |
| 32 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.03 | 1.95 | 1.98 |
| 64 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.23 | 0.58 | 1.17 |
| 64 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.06 | 2.19 | 2.22 |
| 64 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.06 | 2.16 | 2.20 |

### baseline

| batch | heads | dim_k | dim_v | dtype | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 64 | 64 | torch.float32 | 0.125 | 0.01 | 0.04 | 0.08 |
| 1 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.02 | 0.13 | 0.25 |
| 1 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.04 | 0.06 | 0.06 |
| 1 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.04 | 0.06 | 0.06 |
| 8 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.05 | 0.33 | 0.66 |
| 8 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.09 | 0.20 | 0.20 |
| 8 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.09 | 0.20 | 0.20 |
| 16 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.09 | 0.37 | 0.74 |
| 16 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.15 | 0.23 | 0.23 |
| 16 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.15 | 0.23 | 0.23 |
| 32 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.17 | 0.39 | 0.78 |
| 32 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.27 | 0.25 | 0.26 |
| 32 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.27 | 0.25 | 0.25 |
| 64 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.33 | 0.41 | 0.83 |
| 64 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.49 | 0.27 | 0.28 |
| 64 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.50 | 0.27 | 0.28 |

## gqa_fwd

### tileops

| batch | seq_len | heads | heads_kv | dim | causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 8 | 4 | 64 | False | torch.float16 | 0.01 | 211.06 | 0.31 |
| 4 | 2048 | 64 | 4 | 128 | False | torch.float16 | 0.73 | 748.10 | 0.39 |
| 4 | 2048 | 64 | 4 | 128 | False | torch.bfloat16 | 0.72 | 767.96 | 0.40 |

## gqa_bwd

### tileops

| batch | seq_len | heads | heads_kv | dim | causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 8 | 4 | 64 | False | torch.float16 | 0.05 | 102.34 | 0.10 |
| 4 | 2048 | 64 | 4 | 128 | False | torch.float16 | 3.20 | 430.02 | 0.14 |
| 4 | 2048 | 64 | 4 | 128 | False | torch.bfloat16 | 3.19 | 430.60 | 0.14 |

## gqa_decode

### tileops

| batch | heads | heads_kv | seq_len_kv | dim | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 8 | 8192 | 128 | torch.float16 | 0.03 | 3.87 | 0.97 |
| 4 | 32 | 4 | 4096 | 128 | torch.bfloat16 | 0.04 | 6.86 | 0.86 |
| 8 | 64 | 16 | 8192 | 128 | torch.float16 | 0.16 | 13.08 | 3.27 |

### baseline

| batch | heads | heads_kv | seq_len_kv | dim | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 8 | 8192 | 128 | torch.float16 | 0.43 | 0.31 | 0.08 |
| 4 | 32 | 4 | 4096 | 128 | torch.bfloat16 | 0.81 | 0.33 | 0.04 |
| 8 | 64 | 16 | 8192 | 128 | torch.float16 | 5.63 | 0.38 | 0.10 |

## gqa_decode_paged

### tileops

| batch | heads | heads_kv | seqlen_kv | dim | page_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 16 | 8 | 512 | 128 | 128 | torch.float16 | 0.02 | 0.21 | 0.10 |
| 2 | 8 | 4 | 1024 | 64 | 256 | torch.float16 | 0.02 | 0.20 | 0.05 |
| 1 | 16 | 4 | 2048 | 128 | 512 | torch.float16 | 0.02 | 0.77 | 0.19 |
| 1 | 32 | 16 | 512 | 64 | 128 | torch.float16 | 0.02 | 0.21 | 0.11 |

### baseline

| batch | heads | heads_kv | seqlen_kv | dim | page_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 16 | 8 | 512 | 128 | 128 | torch.float16 | 0.07 | 0.06 | 0.03 |
| 2 | 8 | 4 | 1024 | 64 | 256 | torch.float16 | 0.13 | 0.03 | 0.01 |
| 1 | 16 | 4 | 2048 | 128 | 512 | torch.float16 | 0.07 | 0.25 | 0.06 |
| 1 | 32 | 16 | 512 | 64 | 128 | torch.float16 | 0.07 | 0.06 | 0.03 |

## gqa_sliding_window_fwd

### tileops

| batch | seq | heads | heads_kv | dim | is_causal | wl | wr | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 512 | 8 | 2 | 64 | True | -1 | -1 | torch.float16 | 222.69 | 0.00 | 0.00 |
| 2 | 512 | 8 | 2 | 64 | True | 128 | -1 | torch.float16 | 0.01 | 41.10 | 0.46 |
| 2 | 768 | 8 | 2 | 64 | False | 256 | -1 | torch.float16 | 199.41 | 0.01 | 0.00 |
| 2 | 512 | 8 | 2 | 64 | False | 64 | 64 | torch.bfloat16 | 200.68 | 0.00 | 0.00 |
| 1 | 2048 | 8 | 2 | 64 | True | 512 | -1 | torch.float16 | 199.70 | 0.01 | 0.00 |

## gqa_sliding_window_varlen_fwd

### tileops

| batch | heads | heads_kv | dim | is_causal | wl | wr | dtype | max_seqlen_q | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 8 | 128 | True | -1 | -1 | torch.float16 | 3000 | 0.22 | 341.74 | 0.28 |
| 2 | 32 | 8 | 128 | True | 2000 | -1 | torch.float16 | 3073 | 0.34 | 253.57 | 0.27 |
| 4 | 32 | 8 | 128 | False | 1500 | 1500 | torch.float16 | 3500 | 0.93 | 370.13 | 0.22 |
| 2 | 32 | 8 | 128 | True | -1 | -1 | torch.float16 | 1024 | 0.39 | 412.42 | 0.19 |
| 2 | 32 | 8 | 128 | True | -1 | -1 | torch.float16 | 8193 | 1.92 | 393.02 | 0.14 |

## group_norm

### tileops

| n | c | g | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 128 | 32 | torch.float16 | 90.33 | 0.00 | 0.00 |
| 8 | 128 | 32 | torch.bfloat16 | 88.89 | 0.00 | 0.00 |
| 4 | 256 | 32 | torch.float16 | 89.52 | 0.00 | 0.00 |
| 4 | 128 | 16 | torch.float16 | 90.03 | 0.00 | 0.00 |

### baseline

| n | c | g | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 128 | 32 | torch.float16 | 0.02 | 0.35 | 0.28 |
| 8 | 128 | 32 | torch.bfloat16 | 0.01 | 0.35 | 0.28 |
| 4 | 256 | 32 | torch.float16 | 0.02 | 0.25 | 0.20 |
| 4 | 128 | 16 | torch.float16 | 0.02 | 0.15 | 0.12 |

## grouped_gemm_nt

### tileops

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_nt | 16384 | 4 | 4864 | 4096 | torch.float16 | False | True | 125.19 | 5.21 | 0.00 |

### baseline

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_nt | 16384 | 4 | 4864 | 4096 | torch.float16 | False | True | 1.54 | 423.28 | 0.29 |

## grouped_gemm_nn

### tileops

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_nn | 16384 | 4 | 4864 | 4096 | torch.float16 | False | False | 135.96 | 4.80 | 0.00 |

### baseline

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_nn | 16384 | 4 | 4864 | 4096 | torch.float16 | False | False | 1.05 | 619.60 | 0.43 |

## grouped_gemm_tn

### tileops

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_tn | 16384 | 4 | 4864 | 4096 | torch.float16 | True | False | 126.99 | 5.14 | 0.00 |

### baseline

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_tn | 16384 | 4 | 4864 | 4096 | torch.float16 | True | False | 1.02 | 642.35 | 0.45 |

## grouped_gemm_tt

### tileops

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_tt | 16384 | 4 | 4864 | 4096 | torch.float16 | True | True | 126.42 | 5.16 | 0.00 |

### baseline

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_tt | 16384 | 4 | 4864 | 4096 | torch.float16 | True | True | 1.00 | 651.99 | 0.45 |

## grouped_gemm_complete

### tileops

| batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16384 | 4 | 4864 | 4096 | torch.float16 | True | False | 496.79 | 3.94 | N/A |

### baseline

| batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16384 | 4 | 4864 | 4096 | torch.float16 | True | False | 4.58 | 427.88 | N/A |

## leaky_relu

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| leaky_relu | torch.float16 | 4194304 | 0.01 | 0.72 | 2.87 |
| leaky_relu | torch.bfloat16 | 4194304 | 0.01 | 0.72 | 2.87 |
| leaky_relu | torch.float32 | 4194304 | 0.01 | 0.40 | 3.17 |
| leaky_relu | torch.float16 | 10485760 | 0.01 | 0.82 | 3.29 |
| leaky_relu | torch.bfloat16 | 10485760 | 0.01 | 0.83 | 3.31 |
| leaky_relu | torch.float32 | 10485760 | 0.02 | 0.47 | 3.77 |
| leaky_relu | torch.float16 | 20971520 | 0.02 | 0.92 | 3.67 |
| leaky_relu | torch.bfloat16 | 20971520 | 0.02 | 0.91 | 3.65 |
| leaky_relu | torch.float32 | 20971520 | 0.04 | 0.51 | 4.11 |

### baseline

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| leaky_relu | torch.float16 | 4194304 | 0.01 | 0.67 | 2.67 |
| leaky_relu | torch.bfloat16 | 4194304 | 0.01 | 0.67 | 2.67 |
| leaky_relu | torch.float32 | 4194304 | 0.01 | 0.40 | 3.22 |
| leaky_relu | torch.float16 | 10485760 | 0.01 | 0.83 | 3.34 |
| leaky_relu | torch.bfloat16 | 10485760 | 0.01 | 0.83 | 3.33 |
| leaky_relu | torch.float32 | 10485760 | 0.02 | 0.47 | 3.75 |
| leaky_relu | torch.float16 | 20971520 | 0.02 | 0.94 | 3.77 |
| leaky_relu | torch.bfloat16 | 20971520 | 0.02 | 0.94 | 3.76 |
| leaky_relu | torch.float32 | 20971520 | 0.04 | 0.51 | 4.08 |

## elu

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| elu | torch.float16 | 4194304 | 0.01 | 0.37 | 1.49 |
| elu | torch.bfloat16 | 4194304 | 0.01 | 0.38 | 1.52 |
| elu | torch.float32 | 4194304 | 0.01 | 0.31 | 2.50 |
| elu | torch.float16 | 10485760 | 0.02 | 0.45 | 1.80 |
| elu | torch.bfloat16 | 10485760 | 0.02 | 0.45 | 1.81 |
| elu | torch.float32 | 10485760 | 0.03 | 0.36 | 2.90 |
| elu | torch.float16 | 20971520 | 0.04 | 0.49 | 1.95 |
| elu | torch.bfloat16 | 20971520 | 0.04 | 0.49 | 1.95 |
| elu | torch.float32 | 20971520 | 0.06 | 0.38 | 3.04 |

### baseline

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| elu | torch.float16 | 4194304 | 0.01 | 0.42 | 1.70 |
| elu | torch.bfloat16 | 4194304 | 0.01 | 0.42 | 1.70 |
| elu | torch.float32 | 4194304 | 0.01 | 0.38 | 3.00 |
| elu | torch.float16 | 10485760 | 0.02 | 0.52 | 2.06 |
| elu | torch.bfloat16 | 10485760 | 0.02 | 0.51 | 2.04 |
| elu | torch.float32 | 10485760 | 0.02 | 0.45 | 3.62 |
| elu | torch.float16 | 20971520 | 0.04 | 0.57 | 2.29 |
| elu | torch.bfloat16 | 20971520 | 0.04 | 0.56 | 2.24 |
| elu | torch.float32 | 20971520 | 0.04 | 0.50 | 4.01 |

## hardtanh

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| hardtanh | torch.float16 | 4194304 | 0.01 | 0.67 | 2.69 |
| hardtanh | torch.bfloat16 | 4194304 | 0.01 | 0.67 | 2.67 |
| hardtanh | torch.float32 | 4194304 | 0.01 | 0.40 | 3.18 |
| hardtanh | torch.float16 | 10485760 | 0.01 | 0.82 | 3.30 |
| hardtanh | torch.bfloat16 | 10485760 | 0.01 | 0.83 | 3.33 |
| hardtanh | torch.float32 | 10485760 | 0.02 | 0.47 | 3.76 |
| hardtanh | torch.float16 | 20971520 | 0.02 | 0.94 | 3.74 |
| hardtanh | torch.bfloat16 | 20971520 | 0.02 | 0.94 | 3.77 |
| hardtanh | torch.float32 | 20971520 | 0.04 | 0.51 | 4.08 |

### baseline

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| hardtanh | torch.float16 | 4194304 | 0.01 | 0.63 | 2.51 |
| hardtanh | torch.bfloat16 | 4194304 | 0.01 | 0.65 | 2.61 |
| hardtanh | torch.float32 | 4194304 | 0.01 | 0.40 | 3.21 |
| hardtanh | torch.float16 | 10485760 | 0.01 | 0.79 | 3.14 |
| hardtanh | torch.bfloat16 | 10485760 | 0.01 | 0.83 | 3.31 |
| hardtanh | torch.float32 | 10485760 | 0.02 | 0.47 | 3.75 |
| hardtanh | torch.float16 | 20971520 | 0.02 | 0.92 | 3.70 |
| hardtanh | torch.bfloat16 | 20971520 | 0.02 | 0.93 | 3.73 |
| hardtanh | torch.float32 | 20971520 | 0.04 | 0.51 | 4.05 |

## softplus

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| softplus | torch.float16 | 4194304 | 0.01 | 0.32 | 1.26 |
| softplus | torch.bfloat16 | 4194304 | 0.01 | 0.31 | 1.26 |
| softplus | torch.float32 | 4194304 | 0.01 | 0.37 | 2.98 |
| softplus | torch.float16 | 10485760 | 0.03 | 0.38 | 1.50 |
| softplus | torch.bfloat16 | 10485760 | 0.03 | 0.37 | 1.49 |
| softplus | torch.float32 | 10485760 | 0.02 | 0.43 | 3.45 |
| softplus | torch.float16 | 20971520 | 0.05 | 0.40 | 1.61 |
| softplus | torch.bfloat16 | 20971520 | 0.05 | 0.40 | 1.61 |
| softplus | torch.float32 | 20971520 | 0.05 | 0.46 | 3.70 |

### baseline

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| softplus | torch.float16 | 4194304 | 0.01 | 0.36 | 1.44 |
| softplus | torch.bfloat16 | 4194304 | 0.01 | 0.36 | 1.43 |
| softplus | torch.float32 | 4194304 | 0.01 | 0.33 | 2.66 |
| softplus | torch.float16 | 10485760 | 0.02 | 0.43 | 1.71 |
| softplus | torch.bfloat16 | 10485760 | 0.02 | 0.42 | 1.70 |
| softplus | torch.float32 | 10485760 | 0.03 | 0.41 | 3.29 |
| softplus | torch.float16 | 20971520 | 0.05 | 0.47 | 1.86 |
| softplus | torch.bfloat16 | 20971520 | 0.05 | 0.46 | 1.84 |
| softplus | torch.float32 | 20971520 | 0.05 | 0.45 | 3.60 |

## clamp

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| clamp | torch.float16 | 4194304 | 0.01 | 0.67 | 2.70 |
| clamp | torch.bfloat16 | 4194304 | 0.01 | 0.67 | 2.69 |
| clamp | torch.float32 | 4194304 | 0.01 | 0.40 | 3.19 |
| clamp | torch.float16 | 10485760 | 0.01 | 0.83 | 3.31 |
| clamp | torch.bfloat16 | 10485760 | 0.01 | 0.83 | 3.33 |
| clamp | torch.float32 | 10485760 | 0.02 | 0.47 | 3.75 |
| clamp | torch.float16 | 20971520 | 0.02 | 0.94 | 3.77 |
| clamp | torch.bfloat16 | 20971520 | 0.02 | 0.94 | 3.74 |
| clamp | torch.float32 | 20971520 | 0.04 | 0.51 | 4.07 |

### baseline

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| clamp | torch.float16 | 4194304 | 0.01 | 0.63 | 2.52 |
| clamp | torch.bfloat16 | 4194304 | 0.01 | 0.66 | 2.63 |
| clamp | torch.float32 | 4194304 | 0.01 | 0.40 | 3.21 |
| clamp | torch.float16 | 10485760 | 0.01 | 0.79 | 3.14 |
| clamp | torch.bfloat16 | 10485760 | 0.01 | 0.83 | 3.31 |
| clamp | torch.float32 | 10485760 | 0.02 | 0.47 | 3.75 |
| clamp | torch.float16 | 20971520 | 0.02 | 0.88 | 3.51 |
| clamp | torch.bfloat16 | 20971520 | 0.02 | 0.93 | 3.70 |
| clamp | torch.float32 | 20971520 | 0.04 | 0.51 | 4.05 |

## nan_to_num

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| nan_to_num | torch.float16 | 4194304 | 0.01 | 0.67 | 2.69 |
| nan_to_num | torch.bfloat16 | 4194304 | 0.01 | 0.67 | 2.69 |
| nan_to_num | torch.float32 | 4194304 | 0.01 | 0.40 | 3.19 |
| nan_to_num | torch.float16 | 10485760 | 0.01 | 0.80 | 3.18 |
| nan_to_num | torch.bfloat16 | 10485760 | 0.01 | 0.80 | 3.18 |
| nan_to_num | torch.float32 | 10485760 | 0.02 | 0.47 | 3.73 |
| nan_to_num | torch.float16 | 20971520 | 0.02 | 0.90 | 3.58 |
| nan_to_num | torch.bfloat16 | 20971520 | 0.02 | 0.89 | 3.57 |
| nan_to_num | torch.float32 | 20971520 | 0.04 | 0.51 | 4.07 |

### baseline

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| nan_to_num | torch.float16 | 4194304 | 0.01 | 0.65 | 2.58 |
| nan_to_num | torch.bfloat16 | 4194304 | 0.01 | 0.65 | 2.58 |
| nan_to_num | torch.float32 | 4194304 | 0.01 | 0.40 | 3.19 |
| nan_to_num | torch.float16 | 10485760 | 0.01 | 0.81 | 3.25 |
| nan_to_num | torch.bfloat16 | 10485760 | 0.01 | 0.82 | 3.28 |
| nan_to_num | torch.float32 | 10485760 | 0.02 | 0.47 | 3.72 |
| nan_to_num | torch.float16 | 20971520 | 0.02 | 0.93 | 3.70 |
| nan_to_num | torch.bfloat16 | 20971520 | 0.02 | 0.93 | 3.70 |
| nan_to_num | torch.float32 | 20971520 | 0.04 | 0.51 | 4.07 |

## prelu

### tileops

| num_channels | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 128 | torch.float16 | 131072 | 0.00 | 0.06 | 0.25 |
| 128 | torch.bfloat16 | 131072 | 0.00 | 0.06 | 0.24 |
| 128 | torch.float32 | 131072 | 0.00 | 0.06 | 0.46 |
| 4096 | torch.float16 | 4194304 | 0.01 | 0.70 | 2.82 |
| 4096 | torch.bfloat16 | 4194304 | 0.01 | 0.71 | 2.84 |
| 4096 | torch.float32 | 4194304 | 0.01 | 0.40 | 3.21 |
| 10240 | torch.float16 | 10485760 | 0.01 | 0.82 | 3.27 |
| 10240 | torch.bfloat16 | 10485760 | 0.01 | 0.81 | 3.26 |
| 10240 | torch.float32 | 10485760 | 0.02 | 0.47 | 3.74 |
| 20480 | torch.float16 | 20971520 | 0.02 | 0.91 | 3.64 |
| 20480 | torch.bfloat16 | 20971520 | 0.02 | 0.91 | 3.65 |
| 20480 | torch.float32 | 20971520 | 0.04 | 0.50 | 4.04 |

### baseline

| num_channels | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 128 | torch.float16 | 131072 | 0.00 | 0.03 | 0.12 |
| 128 | torch.bfloat16 | 131072 | 0.00 | 0.03 | 0.12 |
| 128 | torch.float32 | 131072 | 0.00 | 0.04 | 0.29 |
| 4096 | torch.float16 | 4194304 | 0.01 | 0.29 | 1.15 |
| 4096 | torch.bfloat16 | 4194304 | 0.01 | 0.29 | 1.14 |
| 4096 | torch.float32 | 4194304 | 0.02 | 0.25 | 2.03 |
| 10240 | torch.float16 | 10485760 | 0.03 | 0.33 | 1.34 |
| 10240 | torch.bfloat16 | 10485760 | 0.03 | 0.33 | 1.34 |
| 10240 | torch.float32 | 10485760 | 0.04 | 0.29 | 2.32 |
| 20480 | torch.float16 | 20971520 | 0.06 | 0.37 | 1.46 |
| 20480 | torch.bfloat16 | 20971520 | 0.06 | 0.36 | 1.46 |
| 20480 | torch.float32 | 20971520 | 0.07 | 0.31 | 2.48 |

## where

### tileops

| dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| torch.float16 | 4194304 | 0.01 | 0.41 | 2.84 |
| torch.bfloat16 | 4194304 | 0.01 | 0.41 | 2.84 |
| torch.float32 | 4194304 | 0.02 | 0.26 | 3.34 |
| torch.float16 | 10485760 | 0.02 | 0.50 | 3.53 |
| torch.bfloat16 | 10485760 | 0.02 | 0.50 | 3.53 |
| torch.float32 | 10485760 | 0.03 | 0.30 | 3.90 |
| torch.float16 | 20971520 | 0.04 | 0.56 | 3.94 |
| torch.bfloat16 | 20971520 | 0.04 | 0.56 | 3.94 |
| torch.float32 | 20971520 | 0.06 | 0.33 | 4.24 |

### baseline

| dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| torch.float16 | 4194304 | 0.01 | 0.41 | 2.84 |
| torch.bfloat16 | 4194304 | 0.01 | 0.41 | 2.85 |
| torch.float32 | 4194304 | 0.02 | 0.26 | 3.42 |
| torch.float16 | 10485760 | 0.02 | 0.51 | 3.57 |
| torch.bfloat16 | 10485760 | 0.02 | 0.51 | 3.57 |
| torch.float32 | 10485760 | 0.03 | 0.30 | 3.92 |
| torch.float16 | 20971520 | 0.04 | 0.57 | 3.96 |
| torch.bfloat16 | 20971520 | 0.04 | 0.57 | 3.96 |
| torch.float32 | 20971520 | 0.06 | 0.33 | 4.23 |

## masked_fill

### tileops

| dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| torch.float16 | 4194304 | 0.01 | 0.54 | 2.72 |
| torch.bfloat16 | 4194304 | 0.01 | 0.55 | 2.73 |
| torch.float32 | 4194304 | 0.01 | 0.35 | 3.19 |
| torch.float16 | 10485760 | 0.02 | 0.69 | 3.44 |
| torch.bfloat16 | 10485760 | 0.02 | 0.69 | 3.45 |
| torch.float32 | 10485760 | 0.02 | 0.42 | 3.78 |
| torch.float16 | 20971520 | 0.03 | 0.77 | 3.83 |
| torch.bfloat16 | 20971520 | 0.03 | 0.76 | 3.81 |
| torch.float32 | 20971520 | 0.05 | 0.45 | 4.09 |

### baseline

| dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| torch.float16 | 4194304 | 0.01 | 0.35 | 1.77 |
| torch.bfloat16 | 4194304 | 0.01 | 0.36 | 1.79 |
| torch.float32 | 4194304 | 0.02 | 0.23 | 2.03 |
| torch.float16 | 10485760 | 0.02 | 0.44 | 2.21 |
| torch.bfloat16 | 10485760 | 0.02 | 0.44 | 2.21 |
| torch.float32 | 10485760 | 0.05 | 0.23 | 2.06 |
| torch.float16 | 20971520 | 0.05 | 0.43 | 2.16 |
| torch.bfloat16 | 20971520 | 0.05 | 0.43 | 2.16 |
| torch.float32 | 20971520 | 0.09 | 0.24 | 2.18 |

## alibi

### tileops

| op_name | seq_len | dim | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| alibi | 512 | 64 | torch.float16 | 0.03 | 0.52 | 1.05 |
| alibi | 512 | 64 | torch.bfloat16 | 0.03 | 0.53 | 1.05 |
| alibi | 512 | 64 | torch.float32 | 0.02 | 0.90 | 3.59 |
| alibi | 2048 | 64 | torch.float16 | 0.26 | 1.02 | 2.04 |
| alibi | 2048 | 64 | torch.bfloat16 | 0.27 | 1.01 | 2.02 |
| alibi | 2048 | 64 | torch.float32 | 0.25 | 1.09 | 4.36 |
| alibi | 4096 | 128 | torch.float16 | 0.82 | 2.63 | 5.26 |
| alibi | 4096 | 128 | torch.bfloat16 | 0.99 | 2.17 | 4.34 |
| alibi | 4096 | 128 | torch.float32 | 1.10 | 1.95 | 7.80 |

### baseline

| op_name | seq_len | dim | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| alibi | 512 | 64 | torch.float16 | 0.08 | 0.21 | 0.41 |
| alibi | 512 | 64 | torch.bfloat16 | 0.08 | 0.21 | 0.41 |
| alibi | 512 | 64 | torch.float32 | 0.06 | 0.30 | 1.22 |
| alibi | 2048 | 64 | torch.float16 | 1.00 | 0.27 | 0.54 |
| alibi | 2048 | 64 | torch.bfloat16 | 1.01 | 0.27 | 0.53 |
| alibi | 2048 | 64 | torch.float32 | 0.65 | 0.41 | 1.64 |
| alibi | 4096 | 128 | torch.float16 | 7.41 | 0.29 | 0.58 |
| alibi | 4096 | 128 | torch.bfloat16 | 7.93 | 0.27 | 0.54 |
| alibi | 4096 | 128 | torch.float32 | 5.53 | 0.39 | 1.55 |

## sinusoidal

### tileops

| op_name | seq_len | dim | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| sinusoidal | 512 | 256 | torch.float16 | 0.00 | 0.04 | 0.09 |
| sinusoidal | 512 | 256 | torch.bfloat16 | 0.00 | 0.05 | 0.09 |
| sinusoidal | 512 | 256 | torch.float32 | 0.00 | 0.05 | 0.21 |
| sinusoidal | 2048 | 300 | torch.float16 | 0.00 | 0.15 | 0.30 |
| sinusoidal | 2048 | 300 | torch.bfloat16 | 0.00 | 0.15 | 0.30 |
| sinusoidal | 2048 | 300 | torch.float32 | 0.00 | 0.13 | 0.53 |
| sinusoidal | 4096 | 512 | torch.float16 | 0.01 | 0.31 | 0.62 |
| sinusoidal | 4096 | 512 | torch.bfloat16 | 0.01 | 0.31 | 0.62 |
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
| sinusoidal | 4096 | 512 | torch.float32 | 0.03 | 0.07 | 0.30 |

## leaky_relu_fp8

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| leaky_relu | torch.float8_e4m3fn | 4194304 | 0.01 | 0.49 | 0.99 |
| leaky_relu | torch.float8_e5m2 | 4194304 | 0.03 | 0.16 | 0.32 |
| leaky_relu | torch.float8_e4m3fn | 10485760 | 0.02 | 0.57 | 1.14 |
| leaky_relu | torch.float8_e5m2 | 10485760 | 0.05 | 0.20 | 0.41 |
| leaky_relu | torch.float8_e4m3fn | 20971520 | 0.03 | 0.62 | 1.25 |
| leaky_relu | torch.float8_e5m2 | 20971520 | 0.10 | 0.22 | 0.44 |

### baseline

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| leaky_relu | torch.float8_e4m3fn | 4194304 | 0.03 | 0.16 | 0.33 |
| leaky_relu | torch.float8_e5m2 | 4194304 | 0.02 | 0.17 | 0.34 |
| leaky_relu | torch.float8_e4m3fn | 10485760 | 0.05 | 0.19 | 0.38 |
| leaky_relu | torch.float8_e5m2 | 10485760 | 0.05 | 0.20 | 0.40 |
| leaky_relu | torch.float8_e4m3fn | 20971520 | 0.11 | 0.19 | 0.38 |
| leaky_relu | torch.float8_e5m2 | 20971520 | 0.11 | 0.20 | 0.39 |

## elu_fp8

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| elu | torch.float8_e4m3fn | 4194304 | 0.01 | 0.67 | 1.33 |
| elu | torch.float8_e5m2 | 4194304 | 0.02 | 0.26 | 0.52 |
| elu | torch.float8_e4m3fn | 10485760 | 0.01 | 0.85 | 1.71 |
| elu | torch.float8_e5m2 | 10485760 | 0.03 | 0.31 | 0.62 |
| elu | torch.float8_e4m3fn | 20971520 | 0.02 | 0.97 | 1.93 |
| elu | torch.float8_e5m2 | 20971520 | 0.07 | 0.32 | 0.64 |

### baseline

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| elu | torch.float8_e4m3fn | 4194304 | 0.03 | 0.14 | 0.28 |
| elu | torch.float8_e5m2 | 4194304 | 0.03 | 0.15 | 0.29 |
| elu | torch.float8_e4m3fn | 10485760 | 0.06 | 0.16 | 0.33 |
| elu | torch.float8_e5m2 | 10485760 | 0.06 | 0.17 | 0.34 |
| elu | torch.float8_e4m3fn | 20971520 | 0.12 | 0.17 | 0.34 |
| elu | torch.float8_e5m2 | 20971520 | 0.12 | 0.17 | 0.34 |

## clamp_fp8

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| clamp | torch.float8_e4m3fn | 4194304 | 0.00 | 0.99 | 1.98 |
| clamp | torch.float8_e5m2 | 4194304 | 0.01 | 0.40 | 0.80 |
| clamp | torch.float8_e4m3fn | 10485760 | 0.01 | 1.36 | 2.71 |
| clamp | torch.float8_e5m2 | 10485760 | 0.02 | 0.50 | 1.00 |
| clamp | torch.float8_e4m3fn | 20971520 | 0.01 | 1.57 | 3.14 |
| clamp | torch.float8_e5m2 | 20971520 | 0.04 | 0.51 | 1.03 |

### baseline

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| clamp | torch.float8_e4m3fn | 4194304 | 0.03 | 0.16 | 0.32 |
| clamp | torch.float8_e5m2 | 4194304 | 0.03 | 0.17 | 0.33 |
| clamp | torch.float8_e4m3fn | 10485760 | 0.06 | 0.19 | 0.38 |
| clamp | torch.float8_e5m2 | 10485760 | 0.05 | 0.19 | 0.39 |
| clamp | torch.float8_e4m3fn | 20971520 | 0.11 | 0.19 | 0.37 |
| clamp | torch.float8_e5m2 | 20971520 | 0.11 | 0.19 | 0.38 |

## where_fp8

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| where | torch.float8_e4m3fn | 4194304 | 0.01 | 0.52 | 2.08 |
| where | torch.float8_e5m2 | 4194304 | 0.01 | 0.52 | 2.08 |
| where | torch.float8_e4m3fn | 10485760 | 0.02 | 0.58 | 2.30 |
| where | torch.float8_e5m2 | 10485760 | 0.02 | 0.58 | 2.30 |
| where | torch.float8_e4m3fn | 20971520 | 0.03 | 0.65 | 2.61 |
| where | torch.float8_e5m2 | 20971520 | 0.03 | 0.66 | 2.62 |

### baseline

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| where | torch.float8_e4m3fn | 4194304 | 0.01 | 0.58 | 2.33 |
| where | torch.float8_e5m2 | 4194304 | 0.01 | 0.58 | 2.33 |
| where | torch.float8_e4m3fn | 10485760 | 0.01 | 0.76 | 3.03 |
| where | torch.float8_e5m2 | 10485760 | 0.01 | 0.75 | 3.02 |
| where | torch.float8_e4m3fn | 20971520 | 0.02 | 0.85 | 3.39 |
| where | torch.float8_e5m2 | 20971520 | 0.02 | 0.85 | 3.42 |

## masked_fill_fp8

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| masked_fill | torch.float8_e4m3fn | 4194304 | 0.01 | 0.58 | 1.73 |
| masked_fill | torch.float8_e5m2 | 4194304 | 0.01 | 0.32 | 0.97 |
| masked_fill | torch.float8_e4m3fn | 10485760 | 0.01 | 0.75 | 2.25 |
| masked_fill | torch.float8_e5m2 | 10485760 | 0.03 | 0.40 | 1.21 |
| masked_fill | torch.float8_e4m3fn | 20971520 | 0.03 | 0.83 | 2.49 |
| masked_fill | torch.float8_e5m2 | 20971520 | 0.05 | 0.41 | 1.24 |

### baseline

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| masked_fill | torch.float8_e4m3fn | 4194304 | 0.03 | 0.14 | 0.41 |
| masked_fill | torch.float8_e5m2 | 4194304 | 0.03 | 0.14 | 0.42 |
| masked_fill | torch.float8_e4m3fn | 10485760 | 0.07 | 0.16 | 0.48 |
| masked_fill | torch.float8_e5m2 | 10485760 | 0.06 | 0.17 | 0.50 |
| masked_fill | torch.float8_e4m3fn | 20971520 | 0.14 | 0.15 | 0.46 |
| masked_fill | torch.float8_e5m2 | 20971520 | 0.13 | 0.16 | 0.48 |

## instance_norm

### tileops

| n | c | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 8 | 128 | torch.float16 | 88.85 | 0.00 | 0.00 |
| 8 | 128 | torch.bfloat16 | 89.50 | 0.00 | 0.00 |
| 4 | 256 | torch.float16 | 88.17 | 0.00 | 0.00 |
| 4 | 64 | torch.float16 | 89.83 | 0.00 | 0.00 |

### baseline

| n | c | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 8 | 128 | torch.float16 | 0.02 | 0.25 | 0.20 |
| 8 | 128 | torch.bfloat16 | 0.02 | 0.25 | 0.20 |
| 4 | 256 | torch.float16 | 0.02 | 0.19 | 0.15 |
| 4 | 64 | torch.float16 | 0.01 | 0.09 | 0.07 |

## layer_norm

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 88.05 | 0.00 | 0.00 |
| 4096 | 4096 | torch.bfloat16 | 88.22 | 0.00 | 0.00 |
| 2048 | 5120 | torch.float16 | 88.49 | 0.00 | 0.00 |
| 1025 | 4096 | torch.float16 | 88.60 | 0.00 | 0.00 |

### baseline

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.01 | 1.66 | 1.33 |
| 4096 | 4096 | torch.bfloat16 | 0.03 | 2.59 | 2.07 |
| 2048 | 5120 | torch.float16 | 0.02 | 2.35 | 1.88 |
| 1025 | 4096 | torch.float16 | 0.01 | 1.66 | 1.33 |

## logical_reduce

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | any | 57.08 | 0.00 | 0.00 |
| 1024 | 4096 | torch.bfloat16 | any | 56.72 | 0.00 | 0.00 |
| 1024 | 4096 | torch.float32 | any | 56.85 | 0.00 | 0.00 |
| 1024 | 4096 | torch.int32 | any | 56.82 | 0.00 | 0.00 |
| 4096 | 4096 | torch.float16 | any | 56.72 | 0.00 | 0.00 |
| 1024 | 4096 | torch.float16 | all | 56.85 | 0.00 | 0.00 |
| 1024 | 4096 | torch.bfloat16 | all | 57.29 | 0.00 | 0.00 |
| 1024 | 4096 | torch.float32 | all | 56.51 | 0.00 | 0.00 |
| 1024 | 4096 | torch.int32 | all | 56.91 | 0.00 | 0.00 |
| 4096 | 4096 | torch.float16 | all | 57.31 | 0.00 | 0.00 |
| 1024 | 4096 | torch.float16 | count_nonzero | 54.24 | 0.00 | 0.00 |
| 1024 | 4096 | torch.bfloat16 | count_nonzero | 54.26 | 0.00 | 0.00 |
| 1024 | 4096 | torch.float32 | count_nonzero | 55.33 | 0.00 | 0.00 |
| 1024 | 4096 | torch.int32 | count_nonzero | 53.74 | 0.00 | 0.00 |
| 4096 | 4096 | torch.float16 | count_nonzero | 53.57 | 0.00 | 0.00 |

### baseline

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | any | 0.03 | 0.16 | 0.32 |
| 1024 | 4096 | torch.bfloat16 | any | 0.03 | 0.16 | 0.32 |
| 1024 | 4096 | torch.float32 | any | 0.03 | 0.16 | 0.62 |
| 1024 | 4096 | torch.int32 | any | 0.03 | 0.16 | 0.63 |
| 4096 | 4096 | torch.float16 | any | 0.07 | 0.25 | 0.51 |
| 1024 | 4096 | torch.float16 | all | 0.03 | 0.16 | 0.33 |
| 1024 | 4096 | torch.bfloat16 | all | 0.03 | 0.16 | 0.33 |
| 1024 | 4096 | torch.float32 | all | 0.03 | 0.16 | 0.64 |
| 1024 | 4096 | torch.int32 | all | 0.03 | 0.16 | 0.64 |
| 4096 | 4096 | torch.float16 | all | 0.07 | 0.26 | 0.51 |
| 1024 | 4096 | torch.float16 | count_nonzero | 0.03 | 0.12 | 0.24 |
| 1024 | 4096 | torch.bfloat16 | count_nonzero | 0.03 | 0.12 | 0.24 |
| 1024 | 4096 | torch.float32 | count_nonzero | 0.04 | 0.11 | 0.45 |
| 1024 | 4096 | torch.int32 | count_nonzero | 0.04 | 0.11 | 0.45 |
| 4096 | 4096 | torch.float16 | count_nonzero | 0.11 | 0.15 | 0.31 |

## mean_pooling

### tileops

| batch_size | seq_len | heads | dim | chunk_size | dtype | accum_dtype | chunks_per_bacth | seq_num | use_offsets | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8192 | 64 | 128 | 64 | torch.float16 | torch.float32 | 128 | 1 | 0 | 61.42 | 0.00 | 0.00 |
| 2 | 2048 | 64 | 128 | 64 | torch.float16 | torch.float32 | 32 | 2 | 0 | 60.85 | 0.00 | 0.00 |
| 1 | 8192 | 64 | 128 | 64 | torch.float16 | torch.float32 | 128 | 4 | 1 | 61.64 | 0.00 | 0.00 |
| 1 | 1000 | 64 | 128 | 32 | torch.float16 | torch.float32 | 34 | 4 | 1 | 61.67 | 0.00 | 0.00 |

### baseline

| batch_size | seq_len | heads | dim | chunk_size | dtype | accum_dtype | chunks_per_bacth | seq_num | use_offsets | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8192 | 64 | 128 | 64 | torch.float16 | torch.float32 | 128 | 1 | 0 | 0.76 | 0.09 | 0.18 |
| 2 | 2048 | 64 | 128 | 64 | torch.float16 | torch.float32 | 32 | 2 | 0 | 0.27 | 0.12 | 0.25 |
| 1 | 8192 | 64 | 128 | 64 | torch.float16 | torch.float32 | 128 | 4 | 1 | 0.79 | 0.08 | 0.17 |
| 1 | 1000 | 64 | 128 | 32 | torch.float16 | torch.float32 | 34 | 4 | 1 | 0.27 | 0.03 | 0.06 |

## mha_fwd

### tileops

| batch | seq_len | heads | dim | causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 8 | 64 | False | torch.float16 | 0.01 | 202.26 | 0.40 |
| 16 | 2048 | 16 | 128 | False | torch.float16 | 0.74 | 744.89 | 0.73 |
| 4 | 4096 | 16 | 128 | False | torch.bfloat16 | 0.69 | 792.76 | 0.39 |

## mha_bwd

### tileops

| batch | seq_len | heads | dim | causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 8 | 64 | False | torch.float16 | 0.04 | 124.35 | 0.17 |
| 16 | 2048 | 16 | 128 | False | torch.float16 | 2.19 | 627.15 | 0.43 |
| 4 | 4096 | 16 | 128 | False | torch.bfloat16 | 2.13 | 645.71 | 0.22 |

## mha_decode

### tileops

| b | h | s_q | s_kv | d | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 128 | 5 | 128 | torch.float16 | 0.01 | 1.00 | 0.21 |

### baseline

| b | h | s_q | s_kv | d | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 128 | 5 | 128 | torch.float16 | 0.01 | 1.27 | 0.26 |

## mha_decode_paged

### tileops

| batch | heads | seqlen_q | seqlen_kv | dim | page_size | is_causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 16 | 1 | 512 | 128 | 128 | False | torch.float16 | 0.02 | 0.18 | 0.18 |
| 2 | 8 | 1 | 1024 | 64 | 256 | False | torch.float16 | 0.02 | 0.18 | 0.09 |
| 1 | 8 | 1 | 1024 | 64 | 256 | False | torch.float16 | 0.02 | 0.10 | 0.10 |
| 1 | 8 | 1 | 512 | 64 | 256 | False | torch.float16 | 0.02 | 0.05 | 0.05 |

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
| 1 | 4 | 1280 | torch.bfloat16 | 42.50 | 0.00 | 0.00 |
| 2 | 4 | 1920 | torch.bfloat16 | 42.22 | 0.01 | 0.00 |
| 4 | 4 | 2560 | torch.bfloat16 | 42.50 | 0.02 | 0.00 |

### baseline

| batch | n_expand | c_x | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | 1280 | torch.bfloat16 | 0.01 | 3.92 | 0.00 |
| 2 | 4 | 1920 | torch.bfloat16 | 0.01 | 16.70 | 0.00 |
| 4 | 4 | 2560 | torch.bfloat16 | 0.01 | 56.10 | 0.00 |

## mhc_pre

### tileops

| batch | n_expand | c_x | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | 1280 | torch.bfloat16 | 325.26 | 0.00 | 0.00 |
| 2 | 4 | 1920 | torch.bfloat16 | 330.83 | 0.02 | 0.00 |
| 4 | 4 | 2560 | torch.bfloat16 | 325.79 | 0.06 | 0.00 |

### baseline

| batch | n_expand | c_x | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | 1280 | torch.bfloat16 | 0.27 | 4.66 | 0.00 |
| 2 | 4 | 1920 | torch.bfloat16 | 0.30 | 19.11 | 0.00 |
| 4 | 4 | 2560 | torch.bfloat16 | 0.34 | 58.60 | 0.00 |

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
| 4096 | 6 | 64 | 128 | 24576 | 257 | 32831 | 0.08 | N/A | 0.00 |
| 8192 | 6 | 64 | 128 | 49152 | 449 | 57407 | 0.16 | N/A | 0.00 |
| 2048 | 6 | 256 | 128 | 12288 | 351 | 44927 | 0.05 | N/A | 0.00 |
| 8192 | 6 | 256 | 128 | 49152 | 639 | 81791 | 0.08 | N/A | 0.01 |

## reduce

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | sum | 64.80 | 0.00 | 0.00 |
| 1024 | 4096 | torch.bfloat16 | sum | 64.77 | 0.00 | 0.00 |
| 4096 | 4096 | torch.float16 | sum | 65.67 | 0.00 | 0.00 |
| 1024 | 4096 | torch.float16 | mean | 65.50 | 0.00 | 0.00 |
| 1024 | 4096 | torch.float16 | amax | 65.37 | 0.00 | 0.00 |
| 1024 | 4096 | torch.float16 | amin | 65.27 | 0.00 | 0.00 |
| 1024 | 4096 | torch.float16 | prod | 80.69 | 0.00 | 0.00 |
| 1024 | 4096 | torch.float16 | std | 113.92 | 0.00 | 0.00 |
| 1024 | 4096 | torch.float16 | var | 114.31 | 0.00 | 0.00 |
| 1024 | 4096 | torch.float16 | var_mean | 113.86 | 0.00 | 0.00 |

### baseline

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | sum | 0.03 | 0.16 | 0.32 |
| 1024 | 4096 | torch.bfloat16 | sum | 0.03 | 0.16 | 0.32 |
| 4096 | 4096 | torch.float16 | sum | 0.08 | 0.21 | 0.43 |
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
| 1024 | 4096 | torch.float16 | 56.46 | 0.00 | 0.00 |
| 4096 | 4096 | torch.bfloat16 | 56.38 | 0.00 | 0.00 |
| 2048 | 5120 | torch.float16 | 57.50 | 0.00 | 0.00 |
| 1025 | 4096 | torch.float16 | 56.62 | 0.00 | 0.00 |

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
| 2048 | 64 | torch.float32 | 0.00 | 0.21 | 0.64 |
| 2048 | 128 | torch.float16 | 0.00 | 0.43 | 0.64 |
| 2048 | 128 | torch.bfloat16 | 0.00 | 0.43 | 0.64 |
| 2048 | 128 | torch.float32 | 0.00 | 0.34 | 1.02 |
| 4096 | 128 | torch.float16 | 0.00 | 0.69 | 1.03 |
| 4096 | 128 | torch.bfloat16 | 0.00 | 0.68 | 1.02 |
| 4096 | 128 | torch.float32 | 0.00 | 0.55 | 1.65 |

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

## softmax

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 81.70 | 0.00 | 0.00 |
| 4096 | 4096 | torch.bfloat16 | 81.13 | 0.00 | 0.00 |
| 1024 | 3000 | torch.float16 | 80.72 | 0.00 | 0.00 |
| 1025 | 4096 | torch.float16 | 80.78 | 0.00 | 0.00 |

### baseline

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.02 | 0.99 | 0.99 |
| 4096 | 4096 | torch.bfloat16 | 0.06 | 1.15 | 1.15 |
| 1024 | 3000 | torch.float16 | 0.01 | 0.84 | 0.84 |
| 1025 | 4096 | torch.float16 | 0.02 | 0.99 | 0.99 |

## log_softmax

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 81.46 | 0.00 | 0.00 |
| 4096 | 4096 | torch.bfloat16 | 80.95 | 0.00 | 0.00 |
| 1024 | 3000 | torch.float16 | 81.55 | 0.00 | 0.00 |
| 1025 | 4096 | torch.float16 | 81.17 | 0.00 | 0.00 |

### baseline

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.02 | 1.39 | 1.11 |
| 4096 | 4096 | torch.bfloat16 | 0.05 | 1.63 | 1.30 |
| 1024 | 3000 | torch.float16 | 0.01 | 1.17 | 0.93 |
| 1025 | 4096 | torch.float16 | 0.02 | 1.39 | 1.11 |

## logsumexp

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 74.67 | 0.00 | 0.00 |
| 4096 | 4096 | torch.bfloat16 | 75.03 | 0.00 | 0.00 |
| 1024 | 3000 | torch.float16 | 75.28 | 0.00 | 0.00 |
| 1025 | 4096 | torch.float16 | 74.49 | 0.00 | 0.00 |

### baseline

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.05 | 0.25 | 0.17 |
| 4096 | 4096 | torch.bfloat16 | 0.10 | 0.50 | 0.33 |
| 1024 | 3000 | torch.float16 | 0.04 | 0.21 | 0.14 |
| 1025 | 4096 | torch.float16 | 0.05 | 0.24 | 0.16 |

## topk_selector

### tileops

| batch | seq_len | seq_len_kv | kv_group | topk | in_dtype_str | out_dtype_str | in_dtype | out_dtype | tune_used | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32768 | 65536 | 1 | 1024 | float32 | int32 | torch.float32 | torch.int32 | False | 326.36 | N/A | 0.03 |
| 1 | 32768 | 65536 | 1 | 2048 | float32 | int32 | torch.float32 | torch.int32 | False | 188.37 | N/A | 0.05 |
| 1 | 65535 | 131072 | 1 | 1024 | float32 | int32 | torch.float32 | torch.int32 | False | 365.46 | N/A | 0.09 |
| 1 | 65535 | 131072 | 1 | 2048 | float32 | int32 | torch.float32 | torch.int32 | False | 175.78 | N/A | 0.20 |

### baseline

| batch | seq_len | seq_len_kv | kv_group | topk | in_dtype_str | out_dtype_str | in_dtype | out_dtype | tune_used | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32768 | 65536 | 1 | 1024 | float32 | int32 | torch.float32 | torch.int32 | False | 5.75 | N/A | 1.52 |
| 1 | 32768 | 65536 | 1 | 2048 | float32 | int32 | torch.float32 | torch.int32 | False | 4.97 | N/A | 1.78 |
| 1 | 65535 | 131072 | 1 | 1024 | float32 | int32 | torch.float32 | torch.int32 | False | 109.46 | N/A | 0.32 |
| 1 | 65535 | 131072 | 1 | 2048 | float32 | int32 | torch.float32 | torch.int32 | False | 112.14 | N/A | 0.31 |

## exp

### tileops

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| exp | 262144 | torch.float16 | torch.float16 | 0.00 | 0.11 | 0.45 |
| exp | 1048576 | torch.float16 | torch.float16 | 0.00 | 0.32 | 1.27 |
| exp | 4000000 | torch.float16 | torch.float16 | 0.01 | 0.64 | 2.57 |
| exp | 262144 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.11 | 0.46 |
| exp | 1048576 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.32 | 1.30 |
| exp | 4000000 | torch.bfloat16 | torch.bfloat16 | 0.01 | 0.64 | 2.56 |

### baseline

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| exp | 262144 | torch.float16 | torch.float16 | 0.00 | 0.11 | 0.44 |
| exp | 1048576 | torch.float16 | torch.float16 | 0.00 | 0.31 | 1.26 |
| exp | 4000000 | torch.float16 | torch.float16 | 0.01 | 0.65 | 2.59 |
| exp | 262144 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.11 | 0.44 |
| exp | 1048576 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.32 | 1.26 |
| exp | 4000000 | torch.bfloat16 | torch.bfloat16 | 0.01 | 0.65 | 2.58 |

## gelu

### tileops

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| gelu | 262144 | torch.float16 | torch.float16 | 0.00 | 0.11 | 0.44 |
| gelu | 1048576 | torch.float16 | torch.float16 | 0.00 | 0.30 | 1.20 |
| gelu | 4000000 | torch.float16 | torch.float16 | 0.01 | 0.48 | 1.93 |
| gelu | 262144 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.11 | 0.44 |
| gelu | 1048576 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.29 | 1.18 |
| gelu | 4000000 | torch.bfloat16 | torch.bfloat16 | 0.01 | 0.46 | 1.86 |

### baseline

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| gelu | 262144 | torch.float16 | torch.float16 | 0.00 | 0.10 | 0.42 |
| gelu | 1048576 | torch.float16 | torch.float16 | 0.00 | 0.29 | 1.15 |
| gelu | 4000000 | torch.float16 | torch.float16 | 0.01 | 0.52 | 2.08 |
| gelu | 262144 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.10 | 0.41 |
| gelu | 1048576 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.28 | 1.14 |
| gelu | 4000000 | torch.bfloat16 | torch.bfloat16 | 0.01 | 0.51 | 2.02 |

## logical_not

### tileops

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| logical_not | 262144 | torch.float16 | torch.bool | 0.00 | 0.11 | 0.33 |
| logical_not | 1048576 | torch.float16 | torch.bool | 0.00 | 0.22 | 0.65 |
| logical_not | 4000000 | torch.float16 | torch.bool | 0.01 | 0.31 | 0.92 |

### baseline

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| logical_not | 262144 | torch.float16 | torch.bool | 0.00 | 0.11 | 0.34 |
| logical_not | 1048576 | torch.float16 | torch.bool | 0.00 | 0.34 | 1.02 |
| logical_not | 4000000 | torch.float16 | torch.bool | 0.01 | 0.73 | 2.19 |

## bitwise_not

### tileops

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| bitwise_not | 262144 | torch.int32 | torch.int32 | 0.00 | 0.09 | 0.76 |
| bitwise_not | 1048576 | torch.int32 | torch.int32 | 0.01 | 0.19 | 1.52 |
| bitwise_not | 4000000 | torch.int32 | torch.int32 | 0.02 | 0.27 | 2.13 |

### baseline

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| bitwise_not | 262144 | torch.int32 | torch.int32 | 0.00 | 0.10 | 0.80 |
| bitwise_not | 1048576 | torch.int32 | torch.int32 | 0.00 | 0.25 | 2.02 |
| bitwise_not | 4000000 | torch.int32 | torch.int32 | 0.01 | 0.40 | 3.22 |

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
| isnan | 1048576 | torch.float16 | torch.bool | 0.00 | 0.34 | 1.02 |
| isnan | 4000000 | torch.float16 | torch.bool | 0.01 | 0.78 | 2.35 |

## unary_strategy

### relu_direct

| n_total | shape_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4194304 | 1024x4096 | torch.float16 | direct | 0.01 | 0.29 | 1.17 |
| 4194304 | 1024x4096 | torch.bfloat16 | direct | 0.01 | 0.29 | 1.18 |
| 4194304 | 1024x4096 | torch.float32 | direct | 0.02 | 0.27 | 2.14 |
| 10485760 | 1024x10240 | torch.float16 | direct | 0.03 | 0.32 | 1.29 |
| 10485760 | 1024x10240 | torch.bfloat16 | direct | 0.03 | 0.32 | 1.29 |
| 10485760 | 1024x10240 | torch.float32 | direct | 0.04 | 0.30 | 2.36 |
| 20971520 | 1024x20480 | torch.float16 | direct | 0.06 | 0.34 | 1.36 |
| 20971520 | 1024x20480 | torch.bfloat16 | direct | 0.06 | 0.34 | 1.37 |
| 20971520 | 1024x20480 | torch.float32 | direct | 0.07 | 0.30 | 2.43 |

### relu_explicit_parallel

| n_total | shape_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4194304 | 1024x4096 | torch.float16 | explicit_parallel | 0.02 | 0.22 | 0.89 |
| 4194304 | 1024x4096 | torch.bfloat16 | explicit_parallel | 0.02 | 0.22 | 0.89 |
| 4194304 | 1024x4096 | torch.float32 | explicit_parallel | 0.01 | 0.39 | 3.16 |
| 10485760 | 1024x10240 | torch.float16 | explicit_parallel | 0.04 | 0.25 | 0.99 |
| 10485760 | 1024x10240 | torch.bfloat16 | explicit_parallel | 0.04 | 0.25 | 0.98 |
| 10485760 | 1024x10240 | torch.float32 | explicit_parallel | 0.02 | 0.46 | 3.72 |
| 20971520 | 1024x20480 | torch.float16 | explicit_parallel | 0.08 | 0.26 | 1.03 |
| 20971520 | 1024x20480 | torch.bfloat16 | explicit_parallel | 0.08 | 0.26 | 1.04 |
| 20971520 | 1024x20480 | torch.float32 | explicit_parallel | 0.04 | 0.51 | 4.07 |

### relu_register_copy

| n_total | shape_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4194304 | 1024x4096 | torch.float16 | register_copy | 0.01 | 0.67 | 2.69 |
| 4194304 | 1024x4096 | torch.bfloat16 | register_copy | 0.01 | 0.68 | 2.70 |
| 4194304 | 1024x4096 | torch.float32 | register_copy | 0.01 | 0.40 | 3.20 |
| 10485760 | 1024x10240 | torch.float16 | register_copy | 0.01 | 0.84 | 3.35 |
| 10485760 | 1024x10240 | torch.bfloat16 | register_copy | 0.01 | 0.83 | 3.33 |
| 10485760 | 1024x10240 | torch.float32 | register_copy | 0.02 | 0.47 | 3.78 |
| 20971520 | 1024x20480 | torch.float16 | register_copy | 0.02 | 0.94 | 3.75 |
| 20971520 | 1024x20480 | torch.bfloat16 | register_copy | 0.02 | 0.95 | 3.79 |
| 20971520 | 1024x20480 | torch.float32 | register_copy | 0.04 | 0.51 | 4.09 |

## vector_norm

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | l1 | 68.88 | 0.00 | 0.00 |
| 1024 | 4096 | torch.bfloat16 | l1 | 67.95 | 0.00 | 0.00 |
| 1024 | 4096 | torch.float32 | l1 | 67.48 | 0.00 | 0.00 |
| 4096 | 4096 | torch.float16 | l1 | 68.86 | 0.00 | 0.00 |
| 1024 | 4096 | torch.float16 | l2 | 67.68 | 0.00 | 0.00 |
| 1024 | 4096 | torch.bfloat16 | l2 | 67.35 | 0.00 | 0.00 |
| 1024 | 4096 | torch.float32 | l2 | 66.65 | 0.00 | 0.00 |
| 4096 | 4096 | torch.float16 | l2 | 67.06 | 0.00 | 0.00 |
| 1024 | 4096 | torch.float16 | inf | 67.24 | 0.00 | 0.00 |
| 1024 | 4096 | torch.bfloat16 | inf | 67.51 | 0.00 | 0.00 |
| 1024 | 4096 | torch.float32 | inf | 66.99 | 0.00 | 0.00 |
| 4096 | 4096 | torch.float16 | inf | 67.71 | 0.00 | 0.00 |

### baseline

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | l1 | 0.02 | 0.25 | 0.50 |
| 1024 | 4096 | torch.bfloat16 | l1 | 0.02 | 0.25 | 0.51 |
| 1024 | 4096 | torch.float32 | l1 | 0.02 | 0.22 | 0.89 |
| 4096 | 4096 | torch.float16 | l1 | 0.02 | 0.78 | 1.56 |
| 1024 | 4096 | torch.float16 | l2 | 0.02 | 0.25 | 0.51 |
| 1024 | 4096 | torch.bfloat16 | l2 | 0.02 | 0.25 | 0.51 |
| 1024 | 4096 | torch.float32 | l2 | 0.02 | 0.22 | 0.88 |
| 4096 | 4096 | torch.float16 | l2 | 0.02 | 0.78 | 1.56 |
| 1024 | 4096 | torch.float16 | inf | 0.02 | 0.25 | 0.49 |
| 1024 | 4096 | torch.bfloat16 | inf | 0.02 | 0.25 | 0.49 |
| 1024 | 4096 | torch.float32 | inf | 0.02 | 0.22 | 0.88 |
| 4096 | 4096 | torch.float16 | inf | 0.02 | 0.77 | 1.53 |

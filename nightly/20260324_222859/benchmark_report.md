........................................................................ [  8%]
........................................................................ [ 16%]
........................................................................ [ 24%]
........................................................................ [ 32%]
........................................................................ [ 40%]
........................................................................ [ 48%]
........................................................................ [ 56%]
........................................................................ [ 64%]
........................................................................ [ 72%]
........................................................................ [ 80%]
...........F............................................................ [ 88%]
........................................................................ [ 96%]
...................................                                      [100%]Benchmark report saved to profile_run.log

=================================== FAILURES ===================================
_____________________ test_mha_decode_bench[short-kv-tail] _____________________

b = 1, h = 32, s_q = 128, s_kv = 5, d = 128, dtype = torch.float16, tune = True

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
tileops/kernels/flash_decode/mha_decode.py:476: in forward
    return _mha_decode_no_split_op(self.batch, self.heads, self.seqlen_q, self.seqlen_kv,
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
tileops/kernels/flash_decode/mha_decode.py:337: in _mha_decode_no_split_op
    return _mha_decode_no_split_kernel(batch, heads, seqlen_q, seqlen_kv, dim, is_causal,
../../_tool/tileops_benchmarks_venv_db4a9af2f48857c5/lib/python3.11/site-packages/tilelang/jit/__init__.py:440: in __call__
    kernel = self.compile(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
../../_tool/tileops_benchmarks_venv_db4a9af2f48857c5/lib/python3.11/site-packages/tilelang/jit/__init__.py:375: in compile
    kernel_result = compile(
../../_tool/tileops_benchmarks_venv_db4a9af2f48857c5/lib/python3.11/site-packages/tilelang/jit/__init__.py:98: in compile
    return cached(
../../_tool/tileops_benchmarks_venv_db4a9af2f48857c5/lib/python3.11/site-packages/tilelang/cache/__init__.py:74: in cached
    return _dispatch_map[execution_backend].cached(
../../_tool/tileops_benchmarks_venv_db4a9af2f48857c5/lib/python3.11/site-packages/tilelang/cache/kernel_cache.py:264: in cached
    kernel = JITKernel(
../../_tool/tileops_benchmarks_venv_db4a9af2f48857c5/lib/python3.11/site-packages/tilelang/jit/kernel.py:137: in __init__
    adapter = self._compile_and_create_adapter(func, out_idx)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
../../_tool/tileops_benchmarks_venv_db4a9af2f48857c5/lib/python3.11/site-packages/tilelang/jit/kernel.py:241: in _compile_and_create_adapter
    artifact = tilelang.lower(
../../_tool/tileops_benchmarks_venv_db4a9af2f48857c5/lib/python3.11/site-packages/tilelang/engine/lower.py:249: in lower
    mod = LowerAndLegalize(mod, target)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
../../_tool/tileops_benchmarks_venv_db4a9af2f48857c5/lib/python3.11/site-packages/tilelang/engine/phase.py:178: in LowerAndLegalize
    mod = tilelang.transform.LayoutInference()(mod)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
../../_tool/tileops_benchmarks_venv_db4a9af2f48857c5/lib/python3.11/site-packages/tilelang/3rdparty/tvm/python/tvm/ir/transform.py:167: in __call__
    return _ffi_transform_api.RunPass(self, mod)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

>   ???
E   tvm.error.InternalError: Layout infer conflict between acc_s and acc_s_cast in T.Parallel loop:
E       loop Fragment([64, 64] -> [32], replicate: 2, thread: 256, forward_thread: _rep * 128 + _i // 16 * 32 + _i % 8 * 4 + _j % 8 // 2, forward_index: [_j // 8 * 4 + _i % 16 // 8 * 2 + _j % 2], thread_range: I.Range(0, 256))
E       fragment Fragment([64, 64] -> [16], replicate: 1, thread: 256, forward_thread: _j // 32 * 128 + _i // 16 * 32 + _i % 8 * 4 + _j % 8 // 2, forward_index: [_j % 32 // 8 * 4 + _i % 16 // 8 * 2 + _j % 2], thread_range: I.Range(0, 256))

python/tvm_ffi/cython/function.pxi:929: InternalError
----------------------------- Captured stdout call -----------------------------
Start autotuning mha_decode_kernel...
2026-03-24 20:08:26,680 INFO:Auto-tuning with 0.9 CPU utilizations, 240 CPUs available, 216 CPUs will be used
2026-03-24 20:08:31  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[7]`
2026-03-24 20:08:31  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[7]`
2026-03-24 20:08:31  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[7]`
2026-03-24 20:08:31  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[7]`
2026-03-24 20:08:31  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[7]`
2026-03-24 20:08:31  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[7]`
2026-03-24 20:08:31  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[7]`
2026-03-24 20:08:31  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[7]`
2026-03-24 20:08:31  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[7]`
2026-03-24 20:08:31  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[7]`
2026-03-24 20:08:31  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[7]`
2026-03-24 20:08:31  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[7]`
2026-03-24 20:08:31  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[7]`
2026-03-24 20:08:31  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[7]`
2026-03-24 20:08:31  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[7]`
2026-03-24 20:08:31  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[7]`
2026-03-24 20:08:31  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[7]`
2026-03-24 20:08:31  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[7]`
2026-03-24 20:08:31  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[7]`
2026-03-24 20:08:31  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[7]`
2026-03-24 20:08:31  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[7]`
2026-03-24 20:08:31  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[7]`
2026-03-24 20:08:31  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[7]`
2026-03-24 20:08:31  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[7]`
2026-03-24 20:08:31  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[7]`
2026-03-24 20:08:31  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[7]`
2026-03-24 20:08:31  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[7]`
2026-03-24 20:08:31  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[7]`
2026-03-24 20:08:31  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[7]`
2026-03-24 20:08:31  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[7]`
2026-03-24 20:08:31  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_split` with `out_idx=[7]`
2026-03-24 20:08:58  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-24 20:08:59  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-24 20:08:59  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-24 20:08:59  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-24 20:08:59  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-24 20:08:59  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-24 20:08:59  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-24 20:09:00  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-24 20:09:00  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-24 20:09:00  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-24 20:09:00  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-24 20:09:00  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-24 20:09:01  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-24 20:09:01  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-24 20:09:01  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-24 20:09:01  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-24 20:09:01  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-24 20:09:02  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-24 20:09:02  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-24 20:09:02  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-24 20:09:02  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-24 20:09:02  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-24 20:09:02  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-24 20:09:02  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-24 20:09:02  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-24 20:09:02  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-24 20:09:02  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-24 20:09:02  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-24 20:09:02  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-24 20:09:03  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
2026-03-24 20:09:03  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mha_decode_split`
Tuned Latency 0.020407484844326973 with config {'block_M': 64, 'block_N': 64, 'num_split': 4, 'num_stages': 2, 'threads': 128} at index 0
Tuned Latency 0.015331581234931946 with config {'block_M': 64, 'block_N': 64, 'num_split': 2, 'num_stages': 3, 'threads': 256} at index 1
Tuned Latency 0.019464999437332153 with config {'block_M': 128, 'block_N': 64, 'num_split': 4, 'num_stages': 3, 'threads': 256} at index 2
Tuned Latency 0.024861065670847893 with config {'block_M': 128, 'block_N': 128, 'num_split': 4, 'num_stages': 3, 'threads': 256} at index 3
Tuned Latency 0.016709493473172188 with config {'block_M': 64, 'block_N': 128, 'num_split': 2, 'num_stages': 2, 'threads': 256} at index 4
Tuned Latency 0.021676287055015564 with config {'block_M': 64, 'block_N': 128, 'num_split': 4, 'num_stages': 3, 'threads': 256} at index 5
Tuned Latency 0.015853749588131905 with config {'block_M': 64, 'block_N': 64, 'num_split': 2, 'num_stages': 2, 'threads': 128} at index 6
Tuned Latency 0.016042429953813553 with config {'block_M': 64, 'block_N': 128, 'num_split': 2, 'num_stages': 3, 'threads': 128} at index 7
Tuned Latency 0.01965908519923687 with config {'block_M': 128, 'block_N': 64, 'num_split': 4, 'num_stages': 2, 'threads': 128} at index 8
Tuned Latency 0.020034000277519226 with config {'block_M': 64, 'block_N': 64, 'num_split': 4, 'num_stages': 3, 'threads': 128} at index 9
Tuned Latency 0.016678255051374435 with config {'block_M': 64, 'block_N': 128, 'num_split': 2, 'num_stages': 2, 'threads': 128} at index 10
Tuned Latency 0.024849867448210716 with config {'block_M': 128, 'block_N': 128, 'num_split': 4, 'num_stages': 3, 'threads': 128} at index 11
Tuned Latency 0.022970017045736313 with config {'block_M': 128, 'block_N': 128, 'num_split': 2, 'num_stages': 2, 'threads': 128} at index 12
Tuned Latency 0.020063741132616997 with config {'block_M': 64, 'block_N': 64, 'num_split': 4, 'num_stages': 2, 'threads': 256} at index 13
Tuned Latency 0.025219764560461044 with config {'block_M': 128, 'block_N': 128, 'num_split': 4, 'num_stages': 2, 'threads': 256} at index 14
Tuned Latency 0.01762448437511921 with config {'block_M': 128, 'block_N': 64, 'num_split': 2, 'num_stages': 2, 'threads': 128} at index 15
Tuned Latency 0.016533806920051575 with config {'block_M': 64, 'block_N': 128, 'num_split': 2, 'num_stages': 3, 'threads': 256} at index 16
Tuned Latency 0.015310589224100113 with config {'block_M': 64, 'block_N': 64, 'num_split': 2, 'num_stages': 2, 'threads': 256} at index 17
Tuned Latency 0.020314456894993782 with config {'block_M': 64, 'block_N': 64, 'num_split': 4, 'num_stages': 3, 'threads': 256} at index 18
Tuned Latency 0.021991999819874763 with config {'block_M': 64, 'block_N': 128, 'num_split': 4, 'num_stages': 3, 'threads': 128} at index 19
Tuned Latency 0.02299395203590393 with config {'block_M': 128, 'block_N': 128, 'num_split': 2, 'num_stages': 3, 'threads': 256} at index 20
Tuned Latency 0.01765398308634758 with config {'block_M': 128, 'block_N': 64, 'num_split': 2, 'num_stages': 2, 'threads': 256} at index 21
Tuned Latency 0.01574992574751377 with config {'block_M': 64, 'block_N': 64, 'num_split': 2, 'num_stages': 3, 'threads': 128} at index 22
Tuned Latency 0.019236711785197258 with config {'block_M': 128, 'block_N': 64, 'num_split': 4, 'num_stages': 2, 'threads': 256} at index 23
Tuned Latency 0.02241535112261772 with config {'block_M': 64, 'block_N': 128, 'num_split': 4, 'num_stages': 2, 'threads': 128} at index 24
Tuned Latency 0.017581643536686897 with config {'block_M': 128, 'block_N': 64, 'num_split': 2, 'num_stages': 3, 'threads': 256} at index 25
Tuned Latency 0.02218090370297432 with config {'block_M': 64, 'block_N': 128, 'num_split': 4, 'num_stages': 2, 'threads': 256} at index 26
Tuned Latency 0.019343381747603416 with config {'block_M': 128, 'block_N': 64, 'num_split': 4, 'num_stages': 3, 'threads': 128} at index 27
Tuned Latency 0.018032122403383255 with config {'block_M': 128, 'block_N': 64, 'num_split': 2, 'num_stages': 3, 'threads': 128} at index 28
Tuned Latency 0.022807413712143898 with config {'block_M': 128, 'block_N': 128, 'num_split': 2, 'num_stages': 3, 'threads': 128} at index 29
Tuned Latency 0.02505600079894066 with config {'block_M': 128, 'block_N': 128, 'num_split': 4, 'num_stages': 2, 'threads': 128} at index 30
Tuned Latency 0.02243071049451828 with config {'block_M': 128, 'block_N': 128, 'num_split': 2, 'num_stages': 2, 'threads': 256} at index 31
Best config: {'block_M': 64, 'block_N': 64, 'num_split': 2, 'num_stages': 2, 'threads': 256}
mha_decode_kernel initialized with config: {'block_M': 64, 'block_N': 64, 'num_split': 2, 'num_stages': 2, 'threads': 256}
2026-03-24 20:09:29  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mha_decode_no_split` with `out_idx=[-1]`
----------------------------- Captured stderr call -----------------------------
Compiling configurations:   0%|          | 0/32 [00:00<?, ?it/s]Compiling configurations:   3%|▎         | 1/32 [00:02<01:22,  2.66s/it]                                                                        Compiling configurations:   3%|▎         | 1/32 [00:02<01:22,  2.66s/it]                                                                        Compiling configurations:   3%|▎         | 1/32 [00:02<01:22,  2.66s/it]                                                                        Compiling configurations:   3%|▎         | 1/32 [00:02<01:22,  2.66s/it]                                                                        Compiling configurations:   3%|▎         | 1/32 [00:02<01:22,  2.66s/it]                                                                        Compiling configurations:   3%|▎         | 1/32 [00:02<01:22,  2.66s/it]                                                                        Compiling configurations:   3%|▎         | 1/32 [00:02<01:22,  2.66s/it]                                                                        Compiling configurations:   3%|▎         | 1/32 [00:02<01:22,  2.66s/it]                                                                        Compiling configurations:   3%|▎         | 1/32 [00:02<01:22,  2.66s/it]                                                                        Compiling configurations:   3%|▎         | 1/32 [00:02<01:22,  2.66s/it]                                                                        Compiling configurations:   3%|▎         | 1/32 [00:02<01:22,  2.66s/it]                                                                        Compiling configurations:   3%|▎         | 1/32 [00:02<01:22,  2.66s/it]                                                                        Compiling configurations:   3%|▎         | 1/32 [00:02<01:22,  2.66s/it]                                                                        Compiling configurations:   3%|▎         | 1/32 [00:02<01:22,  2.66s/it]                                                                        Compiling configurations:   3%|▎         | 1/32 [00:02<01:22,  2.66s/it]                                                                        Compiling configurations:   3%|▎         | 1/32 [00:02<01:22,  2.66s/it]                                                                        Compiling configurations:   3%|▎         | 1/32 [00:02<01:22,  2.66s/it]                                                                        Compiling configurations:   3%|▎         | 1/32 [00:02<01:22,  2.66s/it]                                                                        Compiling configurations:   3%|▎         | 1/32 [00:02<01:22,  2.66s/it]                                                                        Compiling configurations:   3%|▎         | 1/32 [00:02<01:22,  2.66s/it]                                                                        Compiling configurations:   3%|▎         | 1/32 [00:02<01:22,  2.66s/it]                                                                        Compiling configurations:   3%|▎         | 1/32 [00:02<01:22,  2.66s/it]                                                                        Compiling configurations:   3%|▎         | 1/32 [00:02<01:22,  2.66s/it]                                                                        Compiling configurations:   3%|▎         | 1/32 [00:02<01:22,  2.66s/it]                                                                        Compiling configurations:   3%|▎         | 1/32 [00:02<01:22,  2.66s/it]                                                                        Compiling configurations:   3%|▎         | 1/32 [00:02<01:22,  2.66s/it]                                                                        Compiling configurations:   3%|▎         | 1/32 [00:02<01:22,  2.66s/it]                                                                        Compiling configurations:   3%|▎         | 1/32 [00:02<01:22,  2.66s/it]                                                                        Compiling configurations:   3%|▎         | 1/32 [00:02<01:22,  2.66s/it]                                                                        Compiling configurations:   3%|▎         | 1/32 [00:02<01:22,  2.66s/it]                                                                        Compiling configurations:   3%|▎         | 1/32 [00:02<01:22,  2.66s/it]                                                                        Compiling configurations:   3%|▎         | 1/32 [00:02<01:22,  2.66s/it]                                                                        Compiling configurations:   3%|▎         | 1/32 [00:30<01:22,  2.66s/it]Compiling configurations:   6%|▋         | 2/32 [00:30<08:43, 17.46s/it]                                                                        Compiling configurations:   6%|▋         | 2/32 [00:30<08:43, 17.46s/it]                                                                        Compiling configurations:   6%|▋         | 2/32 [00:30<08:43, 17.46s/it]                                                                        Compiling configurations:   6%|▋         | 2/32 [00:30<08:43, 17.46s/it]                                                                        Compiling configurations:   6%|▋         | 2/32 [00:31<08:43, 17.46s/it]Compiling configurations:   9%|▉         | 3/32 [00:31<04:45,  9.84s/it]                                                                        Compiling configurations:   9%|▉         | 3/32 [00:31<04:45,  9.84s/it]                                                                        Compiling configurations:   9%|▉         | 3/32 [00:31<04:45,  9.84s/it]                                                                        Compiling configurations:   9%|▉         | 3/32 [00:31<04:45,  9.84s/it]Compiling configurations:  12%|█▎        | 4/32 [00:31<02:53,  6.18s/it]                                                                        Compiling configurations:  12%|█▎        | 4/32 [00:32<02:53,  6.18s/it]                                                                        Compiling configurations:  12%|█▎        | 4/32 [00:32<02:53,  6.18s/it]Compiling configurations:  16%|█▌        | 5/32 [00:32<01:51,  4.12s/it]                                                                        Compiling configurations:  16%|█▌        | 5/32 [00:32<01:51,  4.12s/it]                                                                        Compiling configurations:  16%|█▌        | 5/32 [00:32<01:51,  4.12s/it]Compiling configurations:  19%|█▉        | 6/32 [00:32<01:14,  2.88s/it]                                                                        Compiling configurations:  19%|█▉        | 6/32 [00:32<01:14,  2.88s/it]                                                                        Compiling configurations:  19%|█▉        | 6/32 [00:32<01:14,  2.88s/it]                                                                        Compiling configurations:  19%|█▉        | 6/32 [00:33<01:14,  2.88s/it]                                                                        Compiling configurations:  19%|█▉        | 6/32 [00:33<01:14,  2.88s/it]                                                                        Compiling configurations:  19%|█▉        | 6/32 [00:33<01:14,  2.88s/it]Compiling configurations:  22%|██▏       | 7/32 [00:33<00:54,  2.18s/it]                                                                        Compiling configurations:  22%|██▏       | 7/32 [00:33<00:54,  2.18s/it]                                                                        Compiling configurations:  22%|██▏       | 7/32 [00:33<00:54,  2.18s/it]                                                                        Compiling configurations:  22%|██▏       | 7/32 [00:33<00:54,  2.18s/it]                                                                        Compiling configurations:  22%|██▏       | 7/32 [00:33<00:54,  2.18s/it]Compiling configurations:  25%|██▌       | 8/32 [00:33<00:39,  1.63s/it]                                                                        Compiling configurations:  25%|██▌       | 8/32 [00:33<00:39,  1.63s/it]                                                                        Compiling configurations:  25%|██▌       | 8/32 [00:34<00:39,  1.63s/it]                                                                        Compiling configurations:  25%|██▌       | 8/32 [00:34<00:39,  1.63s/it]                                                                        Compiling configurations:  25%|██▌       | 8/32 [00:34<00:39,  1.63s/it]                                                                        Compiling configurations:  25%|██▌       | 8/32 [00:34<00:39,  1.63s/it]                                                                        Compiling configurations:  25%|██▌       | 8/32 [00:34<00:39,  1.63s/it]                                                                        Compiling configurations:  25%|██▌       | 8/32 [00:34<00:39,  1.63s/it]Compiling configurations:  28%|██▊       | 9/32 [00:34<00:29,  1.28s/it]                                                                        Compiling configurations:  28%|██▊       | 9/32 [00:34<00:29,  1.28s/it]                                                                        Compiling configurations:  28%|██▊       | 9/32 [00:34<00:29,  1.28s/it]                                                                        Compiling configurations:  28%|██▊       | 9/32 [00:34<00:29,  1.28s/it]Compiling configurations:  31%|███▏      | 10/32 [00:34<00:22,  1.02s/it]Compiling configurations:  34%|███▍      | 11/32 [00:35<00:17,  1.18it/s]Compiling configurations:  38%|███▊      | 12/32 [00:35<00:14,  1.33it/s]Compiling configurations:  41%|████      | 13/32 [00:36<00:12,  1.46it/s]Compiling configurations:  44%|████▍     | 14/32 [00:36<00:10,  1.65it/s]Compiling configurations:  47%|████▋     | 15/32 [00:37<00:09,  1.72it/s]Compiling configurations:  50%|█████     | 16/32 [00:37<00:08,  1.81it/s]Compiling configurations:  53%|█████▎    | 17/32 [00:38<00:07,  1.92it/s]Compiling configurations:  56%|█████▋    | 18/32 [00:38<00:06,  2.03it/s]Compiling configurations:  59%|█████▉    | 19/32 [00:39<00:06,  2.09it/s]Compiling configurations:  62%|██████▎   | 20/32 [00:39<00:05,  2.11it/s]Compiling configurations:  66%|██████▌   | 21/32 [00:40<00:05,  2.04it/s]Compiling configurations:  69%|██████▉   | 22/32 [00:40<00:04,  2.06it/s]Compiling configurations:  72%|███████▏  | 23/32 [00:41<00:04,  2.14it/s]Compiling configurations:  75%|███████▌  | 24/32 [00:41<00:03,  2.12it/s]Compiling configurations:  78%|███████▊  | 25/32 [00:41<00:03,  2.15it/s]Compiling configurations:  81%|████████▏ | 26/32 [00:42<00:02,  2.13it/s]Compiling configurations:  84%|████████▍ | 27/32 [00:42<00:02,  2.15it/s]Compiling configurations:  88%|████████▊ | 28/32 [00:43<00:01,  2.13it/s]Compiling configurations:  91%|█████████ | 29/32 [00:43<00:01,  2.11it/s]Compiling configurations:  94%|█████████▍| 30/32 [00:44<00:00,  2.03it/s]Compiling configurations:  97%|█████████▋| 31/32 [00:44<00:00,  1.98it/s]Compiling configurations: 100%|██████████| 32/32 [00:45<00:00,  1.94it/s]Compiling configurations: 100%|██████████| 32/32 [00:45<00:00,  1.42s/it]
Bench configurations:   0%|          | 0/32 [00:00<?, ?it/s]Bench configurations:   0%|          | 0/32 [00:00<?, ?it/s, best_latency=0.0204]                                                                                 Bench configurations:   0%|          | 0/32 [00:00<?, ?it/s, best_latency=0.0204]Bench configurations:   0%|          | 0/32 [00:00<?, ?it/s, best_latency=0.0153]                                                                                 Bench configurations:   0%|          | 0/32 [00:00<?, ?it/s, best_latency=0.0153]Bench configurations:   6%|▋         | 2/32 [00:00<00:06,  4.33it/s, best_latency=0.0153]Bench configurations:   6%|▋         | 2/32 [00:00<00:06,  4.33it/s, best_latency=0.0153]                                                                                         Bench configurations:   6%|▋         | 2/32 [00:00<00:06,  4.33it/s, best_latency=0.0153]Bench configurations:   9%|▉         | 3/32 [00:00<00:09,  2.94it/s, best_latency=0.0153]Bench configurations:   9%|▉         | 3/32 [00:01<00:09,  2.94it/s, best_latency=0.0153]                                                                                         Bench configurations:   9%|▉         | 3/32 [00:01<00:09,  2.94it/s, best_latency=0.0153]Bench configurations:  12%|█▎        | 4/32 [00:01<00:11,  2.40it/s, best_latency=0.0153]Bench configurations:  12%|█▎        | 4/32 [00:01<00:11,  2.40it/s, best_latency=0.0153]                                                                                         Bench configurations:  12%|█▎        | 4/32 [00:01<00:11,  2.40it/s, best_latency=0.0153]Bench configurations:  16%|█▌        | 5/32 [00:01<00:11,  2.32it/s, best_latency=0.0153]Bench configurations:  16%|█▌        | 5/32 [00:02<00:11,  2.32it/s, best_latency=0.0153]                                                                                         Bench configurations:  16%|█▌        | 5/32 [00:02<00:11,  2.32it/s, best_latency=0.0153]Bench configurations:  19%|█▉        | 6/32 [00:02<00:11,  2.26it/s, best_latency=0.0153]Bench configurations:  19%|█▉        | 6/32 [00:02<00:11,  2.26it/s, best_latency=0.0153]                                                                                         Bench configurations:  19%|█▉        | 6/32 [00:02<00:11,  2.26it/s, best_latency=0.0153]Bench configurations:  22%|██▏       | 7/32 [00:02<00:10,  2.28it/s, best_latency=0.0153]Bench configurations:  22%|██▏       | 7/32 [00:03<00:10,  2.28it/s, best_latency=0.0153]                                                                                         Bench configurations:  22%|██▏       | 7/32 [00:03<00:10,  2.28it/s, best_latency=0.0153]Bench configurations:  25%|██▌       | 8/32 [00:03<00:10,  2.25it/s, best_latency=0.0153]Bench configurations:  25%|██▌       | 8/32 [00:03<00:10,  2.25it/s, best_latency=0.0153]                                                                                         Bench configurations:  25%|██▌       | 8/32 [00:03<00:10,  2.25it/s, best_latency=0.0153]Bench configurations:  28%|██▊       | 9/32 [00:03<00:10,  2.18it/s, best_latency=0.0153]Bench configurations:  28%|██▊       | 9/32 [00:04<00:10,  2.18it/s, best_latency=0.0153]                                                                                         Bench configurations:  28%|██▊       | 9/32 [00:04<00:10,  2.18it/s, best_latency=0.0153]Bench configurations:  31%|███▏      | 10/32 [00:04<00:10,  2.18it/s, best_latency=0.0153]Bench configurations:  31%|███▏      | 10/32 [00:04<00:10,  2.18it/s, best_latency=0.0153]                                                                                          Bench configurations:  31%|███▏      | 10/32 [00:04<00:10,  2.18it/s, best_latency=0.0153]Bench configurations:  34%|███▍      | 11/32 [00:04<00:09,  2.17it/s, best_latency=0.0153]Bench configurations:  34%|███▍      | 11/32 [00:05<00:09,  2.17it/s, best_latency=0.0153]                                                                                          Bench configurations:  34%|███▍      | 11/32 [00:05<00:09,  2.17it/s, best_latency=0.0153]Bench configurations:  38%|███▊      | 12/32 [00:05<00:09,  2.06it/s, best_latency=0.0153]Bench configurations:  38%|███▊      | 12/32 [00:05<00:09,  2.06it/s, best_latency=0.0153]                                                                                          Bench configurations:  38%|███▊      | 12/32 [00:05<00:09,  2.06it/s, best_latency=0.0153]Bench configurations:  41%|████      | 13/32 [00:05<00:09,  1.99it/s, best_latency=0.0153]Bench configurations:  41%|████      | 13/32 [00:06<00:09,  1.99it/s, best_latency=0.0153]                                                                                          Bench configurations:  41%|████      | 13/32 [00:06<00:09,  1.99it/s, best_latency=0.0153]Bench configurations:  44%|████▍     | 14/32 [00:06<00:08,  2.07it/s, best_latency=0.0153]Bench configurations:  44%|████▍     | 14/32 [00:06<00:08,  2.07it/s, best_latency=0.0153]                                                                                          Bench configurations:  44%|████▍     | 14/32 [00:06<00:08,  2.07it/s, best_latency=0.0153]Bench configurations:  47%|████▋     | 15/32 [00:06<00:08,  2.00it/s, best_latency=0.0153]Bench configurations:  47%|████▋     | 15/32 [00:07<00:08,  2.00it/s, best_latency=0.0153]                                                                                          Bench configurations:  47%|████▋     | 15/32 [00:07<00:08,  2.00it/s, best_latency=0.0153]Bench configurations:  50%|█████     | 16/32 [00:07<00:07,  2.01it/s, best_latency=0.0153]Bench configurations:  50%|█████     | 16/32 [00:07<00:07,  2.01it/s, best_latency=0.0153]                                                                                          Bench configurations:  50%|█████     | 16/32 [00:07<00:07,  2.01it/s, best_latency=0.0153]Bench configurations:  53%|█████▎    | 17/32 [00:07<00:07,  2.06it/s, best_latency=0.0153]Bench configurations:  53%|█████▎    | 17/32 [00:08<00:07,  2.06it/s, best_latency=0.0153]                                                                                          Bench configurations:  53%|█████▎    | 17/32 [00:08<00:07,  2.06it/s, best_latency=0.0153]Bench configurations:  56%|█████▋    | 18/32 [00:08<00:06,  2.13it/s, best_latency=0.0153]Bench configurations:  56%|█████▋    | 18/32 [00:08<00:06,  2.13it/s, best_latency=0.0153]                                                                                          Bench configurations:  56%|█████▋    | 18/32 [00:08<00:06,  2.13it/s, best_latency=0.0153]Bench configurations:  59%|█████▉    | 19/32 [00:08<00:05,  2.17it/s, best_latency=0.0153]Bench configurations:  59%|█████▉    | 19/32 [00:09<00:05,  2.17it/s, best_latency=0.0153]                                                                                          Bench configurations:  59%|█████▉    | 19/32 [00:09<00:05,  2.17it/s, best_latency=0.0153]Bench configurations:  62%|██████▎   | 20/32 [00:09<00:05,  2.17it/s, best_latency=0.0153]Bench configurations:  62%|██████▎   | 20/32 [00:09<00:05,  2.17it/s, best_latency=0.0153]                                                                                          Bench configurations:  62%|██████▎   | 20/32 [00:09<00:05,  2.17it/s, best_latency=0.0153]Bench configurations:  66%|██████▌   | 21/32 [00:09<00:05,  2.06it/s, best_latency=0.0153]Bench configurations:  66%|██████▌   | 21/32 [00:10<00:05,  2.06it/s, best_latency=0.0153]                                                                                          Bench configurations:  66%|██████▌   | 21/32 [00:10<00:05,  2.06it/s, best_latency=0.0153]Bench configurations:  69%|██████▉   | 22/32 [00:10<00:04,  2.05it/s, best_latency=0.0153]Bench configurations:  69%|██████▉   | 22/32 [00:10<00:04,  2.05it/s, best_latency=0.0153]                                                                                          Bench configurations:  69%|██████▉   | 22/32 [00:10<00:04,  2.05it/s, best_latency=0.0153]Bench configurations:  72%|███████▏  | 23/32 [00:10<00:04,  2.12it/s, best_latency=0.0153]Bench configurations:  72%|███████▏  | 23/32 [00:11<00:04,  2.12it/s, best_latency=0.0153]                                                                                          Bench configurations:  72%|███████▏  | 23/32 [00:11<00:04,  2.12it/s, best_latency=0.0153]Bench configurations:  75%|███████▌  | 24/32 [00:11<00:03,  2.10it/s, best_latency=0.0153]Bench configurations:  75%|███████▌  | 24/32 [00:11<00:03,  2.10it/s, best_latency=0.0153]                                                                                          Bench configurations:  75%|███████▌  | 24/32 [00:11<00:03,  2.10it/s, best_latency=0.0153]Bench configurations:  78%|███████▊  | 25/32 [00:11<00:03,  2.12it/s, best_latency=0.0153]Bench configurations:  78%|███████▊  | 25/32 [00:11<00:03,  2.12it/s, best_latency=0.0153]                                                                                          Bench configurations:  78%|███████▊  | 25/32 [00:11<00:03,  2.12it/s, best_latency=0.0153]Bench configurations:  81%|████████▏ | 26/32 [00:11<00:02,  2.10it/s, best_latency=0.0153]Bench configurations:  81%|████████▏ | 26/32 [00:12<00:02,  2.10it/s, best_latency=0.0153]                                                                                          Bench configurations:  81%|████████▏ | 26/32 [00:12<00:02,  2.10it/s, best_latency=0.0153]Bench configurations:  84%|████████▍ | 27/32 [00:12<00:02,  2.12it/s, best_latency=0.0153]Bench configurations:  84%|████████▍ | 27/32 [00:12<00:02,  2.12it/s, best_latency=0.0153]                                                                                          Bench configurations:  84%|████████▍ | 27/32 [00:12<00:02,  2.12it/s, best_latency=0.0153]Bench configurations:  88%|████████▊ | 28/32 [00:12<00:01,  2.10it/s, best_latency=0.0153]Bench configurations:  88%|████████▊ | 28/32 [00:13<00:01,  2.10it/s, best_latency=0.0153]                                                                                          Bench configurations:  88%|████████▊ | 28/32 [00:13<00:01,  2.10it/s, best_latency=0.0153]Bench configurations:  91%|█████████ | 29/32 [00:13<00:01,  2.09it/s, best_latency=0.0153]Bench configurations:  91%|█████████ | 29/32 [00:13<00:01,  2.09it/s, best_latency=0.0153]                                                                                          Bench configurations:  91%|█████████ | 29/32 [00:13<00:01,  2.09it/s, best_latency=0.0153]Bench configurations:  94%|█████████▍| 30/32 [00:13<00:00,  2.02it/s, best_latency=0.0153]Bench configurations:  94%|█████████▍| 30/32 [00:14<00:00,  2.02it/s, best_latency=0.0153]                                                                                          Bench configurations:  94%|█████████▍| 30/32 [00:14<00:00,  2.02it/s, best_latency=0.0153]Bench configurations:  97%|█████████▋| 31/32 [00:14<00:00,  1.96it/s, best_latency=0.0153]Bench configurations:  97%|█████████▋| 31/32 [00:15<00:00,  1.96it/s, best_latency=0.0153]                                                                                          Bench configurations:  97%|█████████▋| 31/32 [00:15<00:00,  1.96it/s, best_latency=0.0153]Bench configurations: 100%|██████████| 32/32 [00:15<00:00,  1.93it/s, best_latency=0.0153]Bench configurations: 100%|██████████| 32/32 [00:15<00:00,  2.13it/s, best_latency=0.0153]
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
benchmarks/ops/bench_deltanet_chunkwise.py: 32 warnings
benchmarks/ops/bench_deltanet_recurrence.py: 13 warnings
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
benchmarks/ops/bench_mha_decode.py: 2 warnings
benchmarks/ops/bench_mha_decode_paged.py: 4 warnings
benchmarks/ops/bench_mhc_post.py: 3 warnings
benchmarks/ops/bench_mhc_pre.py: 3 warnings
benchmarks/ops/bench_moe_fused_topk.py: 9 warnings
benchmarks/ops/bench_moe_permute.py: 9 warnings
benchmarks/ops/bench_moe_permute_align.py: 10 warnings
benchmarks/ops/bench_moe_qwen3_moe.py: 9 warnings
benchmarks/ops/bench_moe_unpermute.py: 9 warnings
benchmarks/ops/bench_reduce.py: 10 warnings
benchmarks/ops/bench_rms_norm.py: 4 warnings
benchmarks/ops/bench_rope.py: 9 warnings
benchmarks/ops/bench_softmax.py: 12 warnings
benchmarks/ops/bench_ssd_chunk_scan_fwd.py: 8 warnings
benchmarks/ops/bench_ssd_chunk_state_fwd.py: 10 warnings
benchmarks/ops/bench_ssd_state_passing_fwd.py: 8 warnings
benchmarks/ops/bench_topk_selector.py: 4 warnings
benchmarks/ops/bench_unary_elementwise.py: 21 warnings
benchmarks/ops/bench_unary_strategy.py: 27 warnings
benchmarks/ops/bench_vector_norm.py: 12 warnings
  /home/ci-runner/workdir/_tool/tileops_benchmarks_venv_db4a9af2f48857c5/lib/python3.11/site-packages/torch/profiler/profiler.py:509: UserWarning: Profiler won't be using warmup, this can skew profiler results
    warn("Profiler won't be using warmup, this can skew profiler results")

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED benchmarks/ops/bench_mha_decode.py::test_mha_decode_bench[short-kv-tail] - tvm.error.InternalError: Layout infer conflict between acc_s and acc_s_cast in T.Parallel loop:
    loop Fragment([64, 64] -> [32], replicate: 2, thread: 256, forward_thread: _rep * 128 + _i // 16 * 32 + _i % 8 * 4 + _j % 8 // 2, forward_index: [_j // 8 * 4 + _i % 16 // 8 * 2 + _j % 2], thread_range: I.Range(0, 256))
    fragment Fragment([64, 64] -> [16], replicate: 1, thread: 256, forward_thread: _j // 32 * 128 + _i // 16 * 32 + _i % 8 * 4 + _j % 8 // 2, forward_index: [_j % 32 // 8 * 4 + _i % 16 // 8 * 2 + _j % 2], thread_range: I.Range(0, 256))
1 failed, 898 passed, 922 warnings in 3385.21s (0:56:25)

===== profile_run.log summary =====
# TileOPs Benchmark Report
Generated: 2026-03-24 20:20:03

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
| 4194304 | torch.bfloat16 | 0.01 | 0.64 | 2.55 |
| 4194304 | torch.float32 | 0.01 | 0.39 | 3.12 |

## r3_jit_cost

### relu_jit

| n_total | cold_ms | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 1000 | 22.85 | 0.00 | 0.00 | 0.00 |
| 2000 | 23.39 | 0.00 | 0.00 | 0.00 |
| 4000 | 23.39 | 0.00 | 0.00 | 0.01 |
| 8000 | 20.79 | 0.00 | 0.00 | 0.02 |
| 16000 | 20.92 | 0.00 | 0.01 | 0.03 |
| 32000 | 20.71 | 0.00 | 0.02 | 0.06 |
| 64000 | 23.0 | 0.00 | 0.03 | 0.12 |
| 128000 | 22.03 | 0.00 | 0.06 | 0.23 |
| 256000 | 20.06 | 0.00 | 0.11 | 0.45 |
| 512000 | 20.46 | 0.00 | 0.19 | 0.78 |

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
| 16777216 | 16M | torch.float32 | explicit_parallel | 0.03 | 0.48 | 3.84 |

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
| 16777216 | 16M | torch.bfloat16 | register_copy | 0.02 | 0.91 | 3.63 |
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
| 16777216 | 16M | torch.float32 | explicit_parallel | 0.04 | 0.46 | 3.68 |

### gelu_register_copy

| n_total | size_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 4K | torch.float16 | register_copy | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.bfloat16 | register_copy | 0.00 | 0.00 | 0.01 |
| 4096 | 4K | torch.float32 | register_copy | 0.00 | 0.00 | 0.02 |
| 4194304 | 4M | torch.float16 | register_copy | 0.01 | 0.48 | 1.90 |
| 4194304 | 4M | torch.bfloat16 | register_copy | 0.01 | 0.46 | 1.86 |
| 4194304 | 4M | torch.float32 | register_copy | 0.01 | 0.39 | 3.09 |
| 16777216 | 16M | torch.float16 | register_copy | 0.03 | 0.60 | 2.41 |
| 16777216 | 16M | torch.bfloat16 | register_copy | 0.03 | 0.58 | 2.32 |
| 16777216 | 16M | torch.float32 | register_copy | 0.04 | 0.47 | 3.77 |

## r5_boundary

### relu_aligned

| n_total | align_label | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 2048000 | aligned | 0.00 | 0.45 | 1.79 |

### relu_unaligned_plus_1

| n_total | align_label | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 2048001 | unaligned_plus_1 | 0.01 | 0.29 | 1.15 |

### relu_unaligned_plus_127

| n_total | align_label | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| 2048127 | unaligned_plus_127 | 0.01 | 0.29 | 1.15 |

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
| 4194304 | 4M | relu | 256 | 0.02 | 0.21 | 0.86 |
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
| 4194304 | 4M | mish | 128 | 0.01 | 0.36 | 1.43 |
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
| fp32 | 4 | 0.00 | 0.22 | 1.74 |

### relu_fp32_npt8

| dtype_label | num_per_thread | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| fp32 | 8 | 0.00 | 0.22 | 1.74 |

### relu_fp16_npt4

| dtype_label | num_per_thread | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| fp16 | 4 | 0.00 | 0.28 | 1.12 |

### relu_fp16_npt8

| dtype_label | num_per_thread | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| fp16 | 8 | 0.00 | 0.22 | 0.88 |

## AdaLayerNormOp

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.02 | 1.34 | 2.14 |
| 4096 | 4096 | torch.bfloat16 | 0.05 | 1.60 | 2.56 |
| 1024 | 3000 | torch.float16 | 0.04 | 0.38 | 0.60 |
| 1025 | 4096 | torch.float16 | 0.02 | 1.34 | 2.14 |

### baseline

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.02 | 0.85 | 1.36 |
| 4096 | 4096 | torch.bfloat16 | 0.08 | 1.05 | 1.68 |
| 1024 | 3000 | torch.float16 | 0.02 | 0.75 | 1.21 |
| 1025 | 4096 | torch.float16 | 0.03 | 0.83 | 1.33 |

## AdaLayerNormZeroOp

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.02 | 1.29 | 2.15 |
| 4096 | 4096 | torch.bfloat16 | 0.06 | 1.58 | 2.63 |
| 1024 | 3000 | torch.float16 | 0.05 | 0.35 | 0.58 |
| 1025 | 4096 | torch.float16 | 0.02 | 1.29 | 2.15 |

### baseline

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.03 | 0.79 | 1.32 |
| 4096 | 4096 | torch.bfloat16 | 0.11 | 0.96 | 1.59 |
| 1024 | 3000 | torch.float16 | 0.03 | 0.72 | 1.20 |
| 1025 | 4096 | torch.float16 | 0.03 | 0.80 | 1.33 |

## ArgmaxOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | argmax | 3.07 | 0.00 | 0.00 |
| 1024 | 4096 | torch.bfloat16 | argmax | 3.07 | 0.00 | 0.00 |
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
| 8 | 64 | (32, 32) | torch.float16 | True | 0.01 | 0.43 | 0.17 |
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
| 4 | 128 | (1024, 1024) | torch.float16 | 6.22 | 0.69 | 0.52 |
| 4 | 256 | (1024, 1024) | torch.float16 | 11.27 | 0.76 | 0.57 |

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
| bias_add_2d | 1000000 | 0.00 | 0.26 | 1.06 |

### baseline_bias_add_2d

| pattern_name | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| bias_add_2d | 1000000 | 0.01 | 0.17 | 0.67 |

### add_interleaved_3d

| pattern_name | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| interleaved_3d | 1048576 | 0.00 | 0.42 | 0.84 |

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
| 1M | bias_add_2d | direct | 1048576 | 0.01 | 0.19 | 1.50 |
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
| 1M | interleaved_3d | direct | 1048576 | 0.00 | 0.23 | 0.46 |
| 1M | interleaved_3d | direct | 1048576 | 0.00 | 0.23 | 0.91 |
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
| 1M | same_shape | explicit_parallel | 1048576 | 0.00 | 0.26 | 1.56 |
| 1M | same_shape | explicit_parallel | 1048576 | 0.01 | 0.19 | 2.24 |
| 16M | same_shape | explicit_parallel | 16777216 | 0.03 | 0.59 | 3.56 |
| 16M | same_shape | explicit_parallel | 16777216 | 0.03 | 0.59 | 3.56 |
| 16M | same_shape | explicit_parallel | 16777216 | 0.05 | 0.33 | 4.01 |

### add_explicit_parallel_bias_add_2d

| size_label | pattern_name | strategy | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4K | bias_add_2d | explicit_parallel | 4096 | 0.00 | 0.00 | 0.01 |
| 4K | bias_add_2d | explicit_parallel | 4096 | 0.00 | 0.00 | 0.01 |
| 4K | bias_add_2d | explicit_parallel | 4096 | 0.00 | 0.00 | 0.02 |
| 1M | bias_add_2d | explicit_parallel | 1048576 | 0.00 | 0.31 | 1.26 |
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
| 1M | interleaved_3d | explicit_parallel | 1048576 | 0.00 | 0.42 | 0.84 |
| 1M | interleaved_3d | explicit_parallel | 1048576 | 0.00 | 0.42 | 0.84 |
| 1M | interleaved_3d | explicit_parallel | 1048576 | 0.00 | 0.38 | 1.53 |
| 16M | interleaved_3d | explicit_parallel | 16777216 | 0.01 | 1.17 | 2.35 |
| 16M | interleaved_3d | explicit_parallel | 16777216 | 0.01 | 1.17 | 2.35 |
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
| 1048576 | 1M | 0.00 | 0.23 | 1.64 |
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
| mul | torch.float16 | torch.float16 | 0.01 | 0.47 | 2.82 |
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
| remainder | torch.float16 | torch.float16 | 0.01 | 0.40 | 2.42 |
| remainder | torch.float16 | torch.float16 | 0.02 | 0.51 | 3.05 |

## pow

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| pow | torch.float16 | torch.float16 | 0.02 | 0.24 | 1.45 |
| pow | torch.float16 | torch.float16 | 0.04 | 0.28 | 1.66 |

### baseline

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| pow | torch.float16 | torch.float16 | 0.02 | 0.24 | 1.44 |
| pow | torch.float16 | torch.float16 | 0.04 | 0.28 | 1.66 |

## floor_divide

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| floor_divide | torch.float16 | torch.float16 | 0.01 | 0.45 | 2.67 |
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
| maximum | torch.float16 | torch.float16 | 0.02 | 0.56 | 3.38 |
| maximum | torch.float16 | torch.float16 | 0.03 | 0.63 | 3.81 |

## minimum

### tileops

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| minimum | torch.float16 | torch.float16 | 0.01 | 0.46 | 2.77 |
| minimum | torch.float16 | torch.float16 | 0.02 | 0.56 | 3.38 |
| minimum | torch.float16 | torch.float16 | 0.03 | 0.64 | 3.81 |

### baseline

| op_name | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| minimum | torch.float16 | torch.float16 | 0.01 | 0.46 | 2.75 |
| minimum | torch.float16 | torch.float16 | 0.02 | 0.56 | 3.38 |
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
| eq | torch.float16 | 0.01 | 0.52 | 2.58 |
| eq | torch.float16 | 0.02 | 0.64 | 3.21 |

## cmp_ne

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| ne | torch.float16 | 0.02 | 0.22 | 1.11 |

### baseline

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| ne | torch.float16 | 0.01 | 0.52 | 2.58 |

## cmp_gt

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| gt | torch.float16 | 0.02 | 0.22 | 1.10 |

### baseline

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| gt | torch.float16 | 0.01 | 0.51 | 2.56 |

## cmp_lt

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| lt | torch.float16 | 0.02 | 0.22 | 1.10 |

### baseline

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| lt | torch.float16 | 0.01 | 0.51 | 2.55 |

## cmp_ge

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| ge | torch.float16 | 0.02 | 0.22 | 1.10 |

### baseline

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| ge | torch.float16 | 0.01 | 0.51 | 2.56 |

## cmp_le

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| le | torch.float16 | 0.02 | 0.22 | 1.10 |

### baseline

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| le | torch.float16 | 0.01 | 0.51 | 2.55 |

## logical_and

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| logical_and | torch.float16 | 0.02 | 0.22 | 1.10 |
| logical_and | torch.float16 | 0.04 | 0.26 | 1.32 |

### baseline

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| logical_and | torch.float16 | 0.01 | 0.71 | 3.53 |
| logical_and | torch.float16 | 0.01 | 0.93 | 4.66 |

## logical_or

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| logical_or | torch.float16 | 0.02 | 0.22 | 1.10 |
| logical_or | torch.float16 | 0.04 | 0.26 | 1.32 |

### baseline

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| logical_or | torch.float16 | 0.01 | 0.71 | 3.55 |
| logical_or | torch.float16 | 0.01 | 0.94 | 4.69 |

## bitwise_and

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| bitwise_and | torch.int32 | 0.02 | 0.27 | 3.21 |
| bitwise_and | torch.int32 | 0.03 | 0.31 | 3.76 |

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
| gelu_tanh_and_mul | 1024 | 4096 | torch.float16 | 0.01 | 0.89 | 2.67 |
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
| silu_and_mul | 1024 | 4096 | torch.bfloat16 | direct | 0.02 | 0.49 | 1.48 |
| silu_and_mul | 1024 | 4096 | torch.float32 | direct | 0.02 | 0.43 | 2.55 |
| silu_and_mul | 1024 | 10240 | torch.float16 | direct | 0.04 | 0.53 | 1.60 |
| silu_and_mul | 1024 | 10240 | torch.bfloat16 | direct | 0.04 | 0.53 | 1.60 |
| silu_and_mul | 1024 | 10240 | torch.float32 | direct | 0.04 | 0.47 | 2.82 |
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
| silu_and_mul | 4096 | 4096 | torch.float16 | explicit_parallel | 0.03 | 1.13 | 3.39 |
| silu_and_mul | 4096 | 4096 | torch.bfloat16 | explicit_parallel | 0.03 | 1.11 | 3.33 |
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
| gelu_and_mul | 1024 | 4096 | torch.float16 | explicit_parallel | 0.01 | 0.77 | 2.30 |
| gelu_and_mul | 1024 | 4096 | torch.bfloat16 | explicit_parallel | 0.01 | 0.77 | 2.30 |
| gelu_and_mul | 1024 | 4096 | torch.float32 | explicit_parallel | 0.02 | 0.55 | 3.28 |
| gelu_and_mul | 1024 | 10240 | torch.float16 | explicit_parallel | 0.02 | 0.88 | 2.65 |
| gelu_and_mul | 1024 | 10240 | torch.bfloat16 | explicit_parallel | 0.02 | 0.87 | 2.62 |
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
| gelu_tanh_and_mul | 4096 | 4096 | torch.bfloat16 | direct | 0.06 | 0.56 | 1.67 |
| gelu_tanh_and_mul | 4096 | 4096 | torch.float32 | direct | 0.07 | 0.50 | 2.99 |

### tileops_explicit_parallel

| op_name | M | N | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gelu_tanh_and_mul | 1024 | 4096 | torch.float16 | explicit_parallel | 0.01 | 0.89 | 2.68 |
| gelu_tanh_and_mul | 1024 | 4096 | torch.bfloat16 | explicit_parallel | 0.01 | 0.89 | 2.67 |
| gelu_tanh_and_mul | 1024 | 4096 | torch.float32 | explicit_parallel | 0.02 | 0.55 | 3.30 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.float16 | explicit_parallel | 0.02 | 1.06 | 3.18 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.bfloat16 | explicit_parallel | 0.02 | 1.05 | 3.16 |
| gelu_tanh_and_mul | 1024 | 10240 | torch.float32 | explicit_parallel | 0.03 | 0.64 | 3.81 |
| gelu_tanh_and_mul | 4096 | 4096 | torch.float16 | explicit_parallel | 0.03 | 1.12 | 3.36 |
| gelu_tanh_and_mul | 4096 | 4096 | torch.bfloat16 | explicit_parallel | 0.03 | 1.12 | 3.35 |
| gelu_tanh_and_mul | 4096 | 4096 | torch.float32 | explicit_parallel | 0.05 | 0.67 | 4.01 |

## sub_bcast

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| sub | torch.float16 | 0.01 | 0.61 | 2.45 |
| sub | torch.float16 | 0.01 | 0.75 | 3.02 |
| sub | torch.float16 | 0.03 | 0.83 | 3.31 |

### baseline

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| sub | torch.float16 | 0.01 | 0.29 | 1.15 |
| sub | torch.float16 | 0.03 | 0.34 | 1.35 |
| sub | torch.float16 | 0.06 | 0.37 | 1.46 |

## mul_bcast

### tileops

| op_name | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| mul | torch.float16 | 0.01 | 0.61 | 2.45 |
| mul | torch.float16 | 0.01 | 0.75 | 3.02 |
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
| div | torch.float16 | 0.03 | 0.78 | 3.14 |

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
| 4194304 | 1024x4096 | torch.float16 | direct | 0.02 | 0.26 | 1.57 |
| 4194304 | 1024x4096 | torch.bfloat16 | direct | 0.02 | 0.26 | 1.56 |
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
| 4194304 | 1024x4096 | torch.float16 | explicit_parallel | 0.01 | 0.46 | 2.74 |
| 4194304 | 1024x4096 | torch.bfloat16 | explicit_parallel | 0.01 | 0.46 | 2.74 |
| 4194304 | 1024x4096 | torch.float32 | explicit_parallel | 0.02 | 0.27 | 3.28 |
| 10485760 | 1024x10240 | torch.float16 | explicit_parallel | 0.02 | 0.55 | 3.29 |
| 10485760 | 1024x10240 | torch.bfloat16 | explicit_parallel | 0.02 | 0.55 | 3.28 |
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
| 1 | 128 | 1024 | 2048 | 512 | 64 | 2048 | 1 | 1 | 1024 | torch.float16 | 1.15 | 507.53 | 0.26 |
| 1 | 128 | 512 | 4096 | 512 | 64 | 1024 | 1 | 1 | 512 | torch.float16 | 0.31 | 469.25 | 0.48 |

### baseline

| batch | heads | seq_len_q | seq_len_kv | dim | dim_tail | topk | stride_kv | heads_kv | q_start_index_s | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 128 | 1024 | 2048 | 512 | 64 | 2048 | 1 | 1 | 1024 | torch.float16 | 16.51 | 35.39 | 0.02 |
| 1 | 128 | 512 | 4096 | 512 | 64 | 1024 | 1 | 1 | 512 | torch.float16 | 16.72 | 8.73 | 0.01 |

## MultiHeadLatentAttentionDecodeWithKVCacheOp

### tileops

| batch | heads | heads_kv | seq_len_kv | dim | dim_pe | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 128 | 1 | 8192 | 512 | 64 | torch.float16 | 0.17 | 441.87 | 1.86 |
| 16 | 128 | 1 | 4096 | 512 | 64 | torch.float16 | 0.05 | 375.70 | 1.60 |

### baseline

| batch | heads | heads_kv | seq_len_kv | dim | dim_pe | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 128 | 1 | 8192 | 512 | 64 | torch.float16 | 0.72 | 101.13 | 0.42 |
| 16 | 128 | 1 | 4096 | 512 | 64 | torch.float16 | 0.19 | 95.76 | 0.41 |

## NSACmpFwdVarlenOp

### tileops

| seq_num | c_seq_len | heads | dim_k | dim_v | group | scale | bc | bs | bk | bv | dtype | accum_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 9 | 8192 | 32 | 128 | 128 | 16 | 0.08838834764831845 | 32 | 32 | 128 | 128 | torch.float16 | torch.float32 | 0.32 | 54.30 | 7.00 |
| 16 | 16384 | 32 | 128 | 128 | 16 | 0.08838834764831845 | 32 | 32 | 128 | 128 | torch.float16 | torch.float32 | 0.35 | 198.10 | 25.15 |

### baseline

| seq_num | c_seq_len | heads | dim_k | dim_v | group | scale | bc | bs | bk | bv | dtype | accum_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 9 | 8192 | 32 | 128 | 128 | 16 | 0.08838834764831845 | 32 | 32 | 128 | 128 | torch.float16 | torch.float32 | 304.94 | 0.06 | 0.01 |
| 16 | 16384 | 32 | 128 | 128 | 16 | 0.08838834764831845 | 32 | 32 | 128 | 128 | torch.float16 | torch.float32 | 592.27 | 0.12 | 0.01 |

## NSAFwdVarlenOp

### tileops

| batch | heads | c_seq_len | dim | is_causal | scale | block_size | groups | selected_blocks | dtype | accum_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 16 | 1024 | 64 | True | 0.1 | 32 | 16 | 1 | torch.float16 | torch.float32 | 0.01 | 15.01 | 0.50 |
| 4 | 16 | 8192 | 64 | True | 0.1 | 32 | 16 | 1 | torch.float16 | torch.float32 | 0.04 | 28.17 | 0.94 |
| 2 | 16 | 8192 | 64 | True | 0.1 | 32 | 16 | 4 | torch.float16 | torch.float32 | 0.06 | 77.70 | 0.65 |

### baseline

| batch | heads | c_seq_len | dim | is_causal | scale | block_size | groups | selected_blocks | dtype | accum_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 16 | 1024 | 64 | True | 0.1 | 32 | 16 | 1 | torch.float16 | torch.float32 | 64.82 | 0.00 | 0.00 |
| 4 | 16 | 8192 | 64 | True | 0.1 | 32 | 16 | 1 | torch.float16 | torch.float32 | 518.63 | 0.00 | 0.00 |
| 2 | 16 | 8192 | 64 | True | 0.1 | 32 | 16 | 4 | torch.float16 | torch.float32 | 478.64 | 0.01 | 0.00 |

## NSATopkVarlenOp

### tileops

| seq_num | c_seq_len | heads | dim | group | scale | selected_block_num | bc | bs | bk | dtype | accum_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 1024 | 32 | 128 | 16 | 1 | 16 | 32 | 32 | 128 | torch.float16 | torch.float32 | 0.04 | 6.89 | 0.65 |
| 3 | 512 | 32 | 128 | 16 | 1 | 16 | 32 | 32 | 128 | torch.float16 | torch.float32 | 0.02 | 2.74 | 0.35 |
| 9 | 8192 | 32 | 128 | 16 | 1 | 16 | 32 | 32 | 128 | torch.float16 | torch.float32 | 0.37 | 46.60 | 3.10 |

### baseline

| seq_num | c_seq_len | heads | dim | group | scale | selected_block_num | bc | bs | bk | dtype | accum_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 1024 | 32 | 128 | 16 | 1 | 16 | 32 | 32 | 128 | torch.float16 | torch.float32 | 240.49 | 0.00 | 0.00 |
| 3 | 512 | 32 | 128 | 16 | 1 | 16 | 32 | 32 | 128 | torch.float16 | torch.float32 | 119.62 | 0.00 | 0.00 |
| 9 | 8192 | 32 | 128 | 16 | 1 | 16 | 32 | 32 | 128 | torch.float16 | torch.float32 | 2495.50 | 0.01 | 0.00 |

## DeltaNetFwdOp

### tileops

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 0.10 | 2.79 | 0.17 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 0.16 | 1.73 | 0.11 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 0.17 | 1.59 | 0.10 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 0.05 | 2.54 | 0.16 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 0.21 | 2.53 | 0.16 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 0.42 | 2.59 | 0.16 |
| 2 | 32768 | 4 | 64 | 64 | 64 | torch.float16 | 0.80 | 2.70 | 0.17 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.05 | 2.54 | 0.16 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.10 | 2.76 | 0.17 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.21 | 2.53 | 0.16 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.42 | 2.58 | 0.16 |
| 2 | 32768 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.84 | 2.56 | 0.16 |

### torch

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 85.03 | 0.00 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 121.91 | 0.00 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 121.56 | 0.00 | 0.00 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 43.10 | 0.00 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 171.63 | 0.00 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 345.00 | 0.00 | 0.00 |
| 2 | 32768 | 4 | 64 | 64 | 64 | torch.float16 | 693.15 | 0.00 | 0.00 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 43.39 | 0.00 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 85.12 | 0.00 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 173.29 | 0.00 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 346.47 | 0.00 | 0.00 |
| 2 | 32768 | 4 | 64 | 64 | 64 | torch.bfloat16 | 692.99 | 0.00 | 0.00 |

## DeltaNetBwdOp

### tileops

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | H | S | DK | DV | BC | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.22 | 2.39 | 0.13 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.26 | 2.10 | 0.12 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.26 | 2.10 | 0.12 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.11 | 2.35 | 0.13 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.42 | 2.55 | 0.14 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 16384 | 64 | 64 | 64 | 0.79 | 2.70 | 0.15 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.11 | 2.37 | 0.13 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.22 | 2.43 | 0.13 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.42 | 2.56 | 0.14 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 16384 | 64 | 64 | 64 | 0.79 | 2.70 | 0.15 |

### torch

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | H | S | DK | DV | BC | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 64 | 32.63 | 0.02 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 32 | 36.27 | 0.01 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 32 | 36.29 | 0.01 | 0.00 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 2048 | 64 | 64 | 64 | 15.89 | 0.02 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 8192 | 64 | 64 | 64 | 68.21 | 0.02 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 16384 | 64 | 64 | 64 | 146.66 | 0.01 | 0.00 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 2048 | 64 | 64 | 64 | 15.88 | 0.02 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 64 | 32.59 | 0.02 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 8192 | 64 | 64 | 64 | 68.30 | 0.02 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 16384 | 64 | 64 | 64 | 146.99 | 0.01 | 0.00 |

## DeltaNetOp

### tileops

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | H | S | DK | DV | BC | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.31 | 2.60 | 0.15 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.40 | 2.01 | 0.12 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.41 | 1.95 | 0.11 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.16 | 2.52 | 0.15 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.62 | 2.61 | 0.15 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 16384 | 64 | 64 | 64 | 1.16 | 2.79 | 0.16 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.16 | 2.53 | 0.15 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.31 | 2.61 | 0.15 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.62 | 2.62 | 0.15 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 16384 | 64 | 64 | 64 | 1.15 | 2.79 | 0.16 |

### torch

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | H | S | DK | DV | BC | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 64 | 32.63 | 0.02 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 32 | 36.17 | 0.02 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 32 | 36.16 | 0.02 | 0.00 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 2048 | 64 | 64 | 64 | 15.90 | 0.03 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 8192 | 64 | 64 | 64 | 68.44 | 0.02 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 16384 | 64 | 64 | 64 | 146.60 | 0.02 | 0.00 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 2048 | 64 | 64 | 64 | 15.87 | 0.03 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 64 | 32.61 | 0.02 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 8192 | 64 | 64 | 64 | 68.20 | 0.02 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 16384 | 64 | 64 | 64 | 146.83 | 0.02 | 0.00 |

## DeltaNetDecodeOp

### tileops

| batch | heads | dim_k | dim_v | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 128 | 128 | torch.float32 | 0.01 | 0.31 | 0.42 |
| 1 | 32 | 128 | 128 | torch.bfloat16 | 0.01 | 0.33 | 0.23 |
| 8 | 32 | 128 | 128 | torch.float32 | 0.02 | 1.19 | 1.61 |
| 8 | 32 | 128 | 128 | torch.bfloat16 | 0.01 | 2.18 | 1.47 |
| 16 | 32 | 128 | 128 | torch.float32 | 0.04 | 1.28 | 1.73 |
| 16 | 32 | 128 | 128 | torch.bfloat16 | 0.02 | 3.09 | 2.09 |
| 32 | 32 | 128 | 128 | torch.float32 | 0.08 | 1.33 | 1.80 |
| 32 | 32 | 128 | 128 | torch.bfloat16 | 0.03 | 3.43 | 2.31 |
| 1 | 32 | 64 | 64 | torch.float32 | 0.01 | 0.15 | 0.20 |
| 8 | 32 | 64 | 64 | torch.float32 | 0.01 | 0.62 | 0.84 |
| 32 | 32 | 64 | 64 | torch.float32 | 0.04 | 0.72 | 0.99 |
| 64 | 32 | 128 | 128 | torch.float32 | 0.15 | 1.38 | 1.87 |
| 64 | 32 | 128 | 128 | torch.bfloat16 | 0.05 | 3.75 | 2.53 |

### torch

| batch | heads | dim_k | dim_v | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 128 | 128 | torch.float32 | 0.02 | 0.14 | 0.19 |
| 1 | 32 | 128 | 128 | torch.bfloat16 | 0.04 | 0.08 | 0.05 |
| 8 | 32 | 128 | 128 | torch.float32 | 0.06 | 0.44 | 0.59 |
| 8 | 32 | 128 | 128 | torch.bfloat16 | 0.09 | 0.29 | 0.20 |
| 16 | 32 | 128 | 128 | torch.float32 | 0.08 | 0.60 | 0.80 |
| 16 | 32 | 128 | 128 | torch.bfloat16 | 0.14 | 0.37 | 0.25 |
| 32 | 32 | 128 | 128 | torch.float32 | 0.15 | 0.67 | 0.90 |
| 32 | 32 | 128 | 128 | torch.bfloat16 | 0.24 | 0.42 | 0.28 |
| 1 | 32 | 64 | 64 | torch.float32 | 0.02 | 0.04 | 0.06 |
| 8 | 32 | 64 | 64 | torch.float32 | 0.03 | 0.23 | 0.32 |
| 32 | 32 | 64 | 64 | torch.float32 | 0.05 | 0.46 | 0.63 |
| 64 | 32 | 128 | 128 | torch.float32 | 0.28 | 0.73 | 0.99 |
| 64 | 32 | 128 | 128 | torch.bfloat16 | 0.44 | 0.46 | 0.31 |

## DropoutOp

### tileops

| dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| torch.float16 | 4194304 | 0.01 | 0.65 | 2.60 |
| torch.bfloat16 | 4194304 | 0.01 | 0.65 | 2.61 |
| torch.float32 | 4194304 | 0.01 | 0.40 | 3.17 |
| torch.float16 | 10485760 | 0.01 | 0.85 | 3.38 |
| torch.bfloat16 | 10485760 | 0.01 | 0.84 | 3.37 |
| torch.float32 | 10485760 | 0.02 | 0.48 | 3.80 |
| torch.float16 | 20971520 | 0.02 | 0.95 | 3.80 |
| torch.bfloat16 | 20971520 | 0.02 | 0.95 | 3.80 |
| torch.float32 | 20971520 | 0.04 | 0.51 | 4.08 |

### baseline

| dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| torch.float16 | 4194304 | 0.01 | 0.38 | 1.51 |
| torch.bfloat16 | 4194304 | 0.01 | 0.38 | 1.50 |
| torch.float32 | 4194304 | 0.01 | 0.28 | 2.24 |
| torch.float16 | 10485760 | 0.02 | 0.50 | 2.01 |
| torch.bfloat16 | 10485760 | 0.02 | 0.49 | 1.98 |
| torch.float32 | 10485760 | 0.03 | 0.33 | 2.60 |
| torch.float16 | 20971520 | 0.04 | 0.55 | 2.20 |
| torch.bfloat16 | 20971520 | 0.04 | 0.54 | 2.16 |
| torch.float32 | 20971520 | 0.06 | 0.35 | 2.79 |

## relu_fp8

### tileops

| op_name | n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| relu_fp8 | 8388608 | torch.float8_e4m3fn | 0.01 | 0.58 | 1.16 |
| relu_fp8 | 8388608 | torch.float8_e5m2 | 0.02 | 0.46 | 0.91 |
| relu_fp8 | 67108864 | torch.float8_e4m3fn | 0.09 | 0.71 | 1.43 |
| relu_fp8 | 67108864 | torch.float8_e5m2 | 0.12 | 0.55 | 1.09 |
| relu_fp8 | 134217728 | torch.float8_e4m3fn | 0.18 | 0.74 | 1.48 |
| relu_fp8 | 134217728 | torch.float8_e5m2 | 0.24 | 0.56 | 1.13 |

### baseline

| op_name | n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| relu_fp8 | 8388608 | torch.float8_e4m3fn | 0.05 | 0.19 | 0.37 |
| relu_fp8 | 8388608 | torch.float8_e5m2 | 0.04 | 0.19 | 0.38 |
| relu_fp8 | 67108864 | torch.float8_e4m3fn | 0.34 | 0.20 | 0.40 |
| relu_fp8 | 67108864 | torch.float8_e5m2 | 0.33 | 0.20 | 0.40 |
| relu_fp8 | 134217728 | torch.float8_e4m3fn | 0.65 | 0.21 | 0.41 |
| relu_fp8 | 134217728 | torch.float8_e5m2 | 0.64 | 0.21 | 0.42 |

## exp_fp8

### tileops

| op_name | n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| exp_fp8 | 8388608 | torch.float8_e4m3fn | 0.01 | 0.99 | 1.98 |
| exp_fp8 | 8388608 | torch.float8_e5m2 | 0.02 | 0.45 | 0.91 |
| exp_fp8 | 67108864 | torch.float8_e4m3fn | 0.05 | 1.32 | 2.65 |
| exp_fp8 | 67108864 | torch.float8_e5m2 | 0.12 | 0.54 | 1.09 |
| exp_fp8 | 134217728 | torch.float8_e4m3fn | 0.10 | 1.38 | 2.77 |
| exp_fp8 | 134217728 | torch.float8_e5m2 | 0.24 | 0.56 | 1.11 |

### baseline

| op_name | n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| exp_fp8 | 8388608 | torch.float8_e4m3fn | 0.05 | 0.19 | 0.37 |
| exp_fp8 | 8388608 | torch.float8_e5m2 | 0.04 | 0.19 | 0.39 |
| exp_fp8 | 67108864 | torch.float8_e4m3fn | 0.33 | 0.20 | 0.40 |
| exp_fp8 | 67108864 | torch.float8_e5m2 | 0.32 | 0.21 | 0.42 |
| exp_fp8 | 134217728 | torch.float8_e4m3fn | 0.64 | 0.21 | 0.42 |
| exp_fp8 | 134217728 | torch.float8_e5m2 | 0.63 | 0.21 | 0.43 |

## add_fp8

### tileops

| op_name | n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| add_fp8 | 8388608 | torch.float8_e4m3fn | 0.01 | 0.85 | 2.55 |
| add_fp8 | 8388608 | torch.float8_e5m2 | 0.02 | 0.42 | 1.26 |
| add_fp8 | 67108864 | torch.float8_e4m3fn | 0.06 | 1.15 | 3.46 |
| add_fp8 | 67108864 | torch.float8_e5m2 | 0.13 | 0.50 | 1.50 |
| add_fp8 | 134217728 | torch.float8_e4m3fn | 0.11 | 1.20 | 3.60 |
| add_fp8 | 134217728 | torch.float8_e5m2 | 0.26 | 0.52 | 1.56 |

### baseline

| op_name | n_total | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| add_fp8 | 8388608 | torch.float8_e4m3fn | 0.08 | 0.11 | 0.33 |
| add_fp8 | 8388608 | torch.float8_e5m2 | 0.07 | 0.11 | 0.34 |
| add_fp8 | 67108864 | torch.float8_e4m3fn | 0.57 | 0.12 | 0.35 |
| add_fp8 | 67108864 | torch.float8_e5m2 | 0.54 | 0.13 | 0.38 |
| add_fp8 | 134217728 | torch.float8_e4m3fn | 1.11 | 0.12 | 0.36 |
| add_fp8 | 134217728 | torch.float8_e5m2 | 1.02 | 0.13 | 0.39 |

## silu_and_mul_fp8

### tileops

| op_name | M | N | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| silu_and_mul_fp8 | 2048 | 11008 | torch.float8_e4m3fn | 0.04 | 2.57 | 1.54 |
| silu_and_mul_fp8 | 2048 | 11008 | torch.float8_e5m2 | 0.06 | 1.77 | 1.06 |
| silu_and_mul_fp8 | 16384 | 11008 | torch.float8_e4m3fn | 0.31 | 2.93 | 1.76 |
| silu_and_mul_fp8 | 16384 | 11008 | torch.float8_e5m2 | 0.45 | 2.02 | 1.21 |
| silu_and_mul_fp8 | 16384 | 28672 | torch.float8_e4m3fn | 0.77 | 3.04 | 1.83 |
| silu_and_mul_fp8 | 16384 | 28672 | torch.float8_e5m2 | 1.12 | 2.09 | 1.26 |

### baseline

| op_name | M | N | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| silu_and_mul_fp8 | 2048 | 11008 | torch.float8_e4m3fn | 0.29 | 0.38 | 0.23 |
| silu_and_mul_fp8 | 2048 | 11008 | torch.float8_e5m2 | 0.28 | 0.40 | 0.24 |
| silu_and_mul_fp8 | 16384 | 11008 | torch.float8_e4m3fn | 1.96 | 0.46 | 0.28 |
| silu_and_mul_fp8 | 16384 | 11008 | torch.float8_e5m2 | 1.93 | 0.47 | 0.28 |
| silu_and_mul_fp8 | 16384 | 28672 | torch.float8_e4m3fn | 3.73 | 0.63 | 0.38 |
| silu_and_mul_fp8 | 16384 | 28672 | torch.float8_e5m2 | 3.96 | 0.59 | 0.36 |

## EngramGateConvBwdOp

### tileops

| M | seq_len | d | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 256 | torch.float16 | 0.01 | 0.07 | 0.03 |
| 2 | 64 | 512 | torch.float16 | 0.01 | 0.33 | 0.11 |
| 1 | 128 | 256 | torch.bfloat16 | 0.01 | 0.18 | 0.06 |
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
| 1 | 512 | 256 | 12 | 4 | 3 | torch.float16 | 0.06 | 0.01 | 0.01 |
| 4 | 1024 | 512 | 20 | 4 | 5 | torch.float16 | 0.07 | 0.12 | 0.03 |
| 8 | 512 | 256 | 18 | 4 | 3 | torch.bfloat16 | 0.05 | 0.09 | 0.01 |

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
| 1 | 128 | 256 | torch.bfloat16 | 0.01 | 0.15 | 0.06 |
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
| 1 | 4096 | 32 | 64 | 8192 | 1 | True | 9.66 | N/A | 0.01 |
| 1 | 2048 | 16 | 64 | 4096 | 1 | True | 1.52 | N/A | 0.02 |

## Fp8QuantOp

### tileops

| batch | seq_len_kv | kv_group | index_dim | in_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8192 | 1 | 64 | torch.float16 | 0.00 | 1.04 | 0.35 |
| 1 | 8192 | 1 | 64 | torch.bfloat16 | 0.00 | 1.04 | 0.35 |
| 1 | 4096 | 1 | 128 | torch.float32 | 0.00 | 0.78 | 0.52 |
| 1 | 16384 | 1 | 32 | torch.float32 | 0.00 | 0.91 | 0.60 |

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
| 1024 | 4096 | torch.float16 | 0.02 | 1.25 | 1.67 |
| 4096 | 4096 | torch.bfloat16 | 0.07 | 1.53 | 2.04 |
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
| 1024 | 4096 | torch.float16 | 0.02 | 1.14 | 1.82 |
| 4096 | 4096 | torch.bfloat16 | 0.06 | 1.43 | 2.29 |
| 2048 | 5120 | torch.float16 | 0.04 | 1.36 | 2.17 |
| 1025 | 4096 | torch.float16 | 0.02 | 1.13 | 1.80 |

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
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 0.19 | 1.38 | 0.09 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 0.08 | 1.66 | 0.10 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 0.14 | 1.95 | 0.12 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 0.28 | 1.91 | 0.12 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 0.53 | 2.01 | 0.13 |
| 2 | 32768 | 4 | 64 | 64 | 64 | torch.float16 | 1.05 | 2.05 | 0.13 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.08 | 1.67 | 0.11 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.14 | 1.95 | 0.12 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.28 | 1.89 | 0.12 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.53 | 2.03 | 0.13 |
| 2 | 32768 | 4 | 64 | 64 | 64 | torch.bfloat16 | 1.01 | 2.13 | 0.13 |

### baseline

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 130.86 | 0.00 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 131.29 | 0.00 | 0.00 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 44.96 | 0.00 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 90.49 | 0.00 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 182.47 | 0.00 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 363.92 | 0.00 | 0.00 |
| 2 | 32768 | 4 | 64 | 64 | 64 | torch.float16 | 727.08 | 0.00 | 0.00 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 45.29 | 0.00 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 90.57 | 0.00 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 182.51 | 0.00 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 364.77 | 0.00 | 0.00 |
| 2 | 32768 | 4 | 64 | 64 | 64 | torch.bfloat16 | 726.88 | 0.00 | 0.00 |

## GatedDeltaNetBwdOp

### tileops

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | H | S | DK | DV | BC | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.42 | 1.27 | 0.07 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.42 | 1.29 | 0.07 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.19 | 1.43 | 0.08 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.34 | 1.56 | 0.09 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.63 | 1.70 | 0.09 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 16384 | 64 | 64 | 64 | 1.21 | 1.78 | 0.10 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.19 | 1.41 | 0.08 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.35 | 1.55 | 0.09 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.64 | 1.69 | 0.09 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 16384 | 64 | 64 | 64 | 1.20 | 1.79 | 0.10 |

### baseline

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | H | S | DK | DV | BC | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 32 | 49.88 | 0.01 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 32 | 49.95 | 0.01 | 0.00 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 2048 | 64 | 64 | 64 | 19.38 | 0.01 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 64 | 39.74 | 0.01 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 8192 | 64 | 64 | 64 | 82.64 | 0.01 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 16384 | 64 | 64 | 64 | 175.74 | 0.01 | 0.00 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 2048 | 64 | 64 | 64 | 19.47 | 0.01 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 64 | 39.78 | 0.01 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 8192 | 64 | 64 | 64 | 82.68 | 0.01 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 16384 | 64 | 64 | 64 | 175.62 | 0.01 | 0.00 |

## GatedDeltaNetOp

### tileops

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | H | S | DK | DV | BC | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.61 | 1.32 | 0.08 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 32 | 0.60 | 1.34 | 0.08 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.26 | 1.54 | 0.09 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.48 | 1.69 | 0.10 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.90 | 1.78 | 0.10 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 16384 | 64 | 64 | 64 | 1.70 | 1.89 | 0.11 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 2048 | 64 | 64 | 64 | 0.26 | 1.52 | 0.09 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 64 | 0.48 | 1.69 | 0.10 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 8192 | 64 | 64 | 64 | 0.91 | 1.77 | 0.10 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 16384 | 64 | 64 | 64 | 1.71 | 1.88 | 0.11 |

### baseline

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | H | S | DK | DV | BC | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 32 | 50.03 | 0.02 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 32 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 32 | 50.06 | 0.02 | 0.00 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 2048 | 64 | 64 | 64 | 19.45 | 0.02 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 4096 | 64 | 64 | 64 | 39.86 | 0.02 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 8192 | 64 | 64 | 64 | 82.71 | 0.02 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4 | 16384 | 64 | 64 | 64 | 175.75 | 0.02 | 0.00 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 2048 | 64 | 64 | 64 | 19.47 | 0.02 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 4096 | 64 | 64 | 64 | 39.81 | 0.02 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 8192 | 64 | 64 | 64 | 82.63 | 0.02 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4 | 16384 | 64 | 64 | 64 | 175.70 | 0.02 | 0.00 |

## GatedDeltaNetDecodeOp

### tileops

| batch | heads | dim_k | dim_v | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 128 | 128 | torch.float32 | 0.01 | 0.30 | 0.41 |
| 1 | 32 | 128 | 128 | torch.bfloat16 | 0.01 | 0.31 | 0.21 |
| 8 | 32 | 128 | 128 | torch.float32 | 0.02 | 1.17 | 1.58 |
| 8 | 32 | 128 | 128 | torch.bfloat16 | 0.01 | 2.09 | 1.41 |
| 16 | 32 | 128 | 128 | torch.float32 | 0.04 | 1.27 | 1.72 |
| 16 | 32 | 128 | 128 | torch.bfloat16 | 0.02 | 3.03 | 2.05 |
| 32 | 32 | 128 | 128 | torch.float32 | 0.08 | 1.33 | 1.80 |
| 32 | 32 | 128 | 128 | torch.bfloat16 | 0.03 | 3.28 | 2.21 |
| 1 | 32 | 64 | 64 | torch.float32 | 0.01 | 0.14 | 0.19 |
| 8 | 32 | 64 | 64 | torch.float32 | 0.01 | 0.59 | 0.81 |
| 32 | 32 | 64 | 64 | torch.float32 | 0.04 | 0.71 | 0.97 |
| 64 | 32 | 128 | 128 | torch.float32 | 0.15 | 1.38 | 1.86 |
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
| 64 | 32 | 128 | 128 | torch.float32 | 0.39 | 0.51 | 0.70 |
| 64 | 32 | 128 | 128 | torch.bfloat16 | 0.56 | 0.36 | 0.24 |

## GemmOp

### tileops

| m | n | k | dtype | trans_a | trans_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1024 | 1024 | 1024 | torch.float16 | False | False | 0.01 | 239.86 | 0.70 |
| 1 | 7168 | 16384 | torch.float16 | False | True | 0.06 | 3.63 | 3.63 |
| 1 | 18432 | 7168 | torch.bfloat16 | False | True | 0.07 | 3.79 | 3.79 |
| 7168 | 1 | 16384 | torch.float16 | False | False | 0.06 | 3.64 | 3.64 |
| 18432 | 1 | 7168 | torch.bfloat16 | False | False | 0.07 | 3.83 | 3.83 |

### baseline

| m | n | k | dtype | trans_a | trans_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1024 | 1024 | 1024 | torch.float16 | False | False | 0.01 | 324.62 | 0.95 |
| 1 | 7168 | 16384 | torch.float16 | False | True | 0.07 | 3.38 | 3.38 |
| 1 | 18432 | 7168 | torch.bfloat16 | False | True | 0.08 | 3.45 | 3.45 |
| 7168 | 1 | 16384 | torch.float16 | False | False | 0.07 | 3.38 | 3.38 |
| 18432 | 1 | 7168 | torch.bfloat16 | False | False | 0.08 | 3.46 | 3.46 |

## GLAFwdOp

### tileops

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.08 | 1.73 | 0.11 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.14 | 1.98 | 0.12 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.26 | 2.04 | 0.13 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.50 | 2.15 | 0.13 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.08 | 1.67 | 0.10 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.13 | 2.00 | 0.12 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.26 | 2.06 | 0.13 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.50 | 2.14 | 0.13 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.23 | 1.79 | 0.10 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.43 | 1.85 | 0.11 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 0.83 | 1.94 | 0.11 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 1.48 | 2.18 | 0.13 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.23 | 1.79 | 0.10 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.43 | 1.85 | 0.11 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 0.82 | 1.97 | 0.11 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 1.49 | 2.17 | 0.12 |

### baseline

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 1.99 | 0.07 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 4.05 | 0.07 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 8.11 | 0.07 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 16.19 | 0.07 | 0.00 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 1.99 | 0.07 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 4.04 | 0.07 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 8.10 | 0.07 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 16.14 | 0.07 | 0.00 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 5.84 | 0.07 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 12.60 | 0.06 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 28.56 | 0.06 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 0.125 | 70.31 | 0.05 | 0.00 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 5.85 | 0.07 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 12.56 | 0.06 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 28.56 | 0.06 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 0.125 | 70.29 | 0.05 | 0.00 |

## GLABwdOp

### tileops

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | T | H | K | V | BC | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 2048 | 4 | 64 | 64 | 64 | 0.125 | 0.15 | 1.75 | 0.10 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4096 | 4 | 64 | 64 | 64 | 0.125 | 0.30 | 1.81 | 0.10 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 8192 | 4 | 64 | 64 | 64 | 0.125 | 0.56 | 1.92 | 0.10 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 16384 | 4 | 64 | 64 | 64 | 0.125 | 1.12 | 1.92 | 0.10 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 2048 | 4 | 64 | 64 | 64 | 0.125 | 0.15 | 1.74 | 0.09 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4096 | 4 | 64 | 64 | 64 | 0.125 | 0.30 | 1.77 | 0.10 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 8192 | 4 | 64 | 64 | 64 | 0.125 | 0.60 | 1.79 | 0.10 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 16384 | 4 | 64 | 64 | 64 | 0.125 | 1.09 | 1.98 | 0.11 |

### baseline

| batch | seq_len | heads | dim_k | dim_v | chunk_size | dtype | B | T | H | K | V | BC | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 2048 | 4 | 64 | 64 | 64 | 0.125 | 5.91 | 0.05 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 4096 | 4 | 64 | 64 | 64 | 0.125 | 12.57 | 0.04 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 8192 | 4 | 64 | 64 | 64 | 0.125 | 28.62 | 0.04 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.float16 | 2 | 16384 | 4 | 64 | 64 | 64 | 0.125 | 70.31 | 0.03 | 0.00 |
| 2 | 2048 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 2048 | 4 | 64 | 64 | 64 | 0.125 | 5.93 | 0.05 | 0.00 |
| 2 | 4096 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 4096 | 4 | 64 | 64 | 64 | 0.125 | 12.58 | 0.04 | 0.00 |
| 2 | 8192 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 8192 | 4 | 64 | 64 | 64 | 0.125 | 28.61 | 0.04 | 0.00 |
| 2 | 16384 | 4 | 64 | 64 | 64 | torch.bfloat16 | 2 | 16384 | 4 | 64 | 64 | 64 | 0.125 | 70.32 | 0.03 | 0.00 |

## GLADecodeOp

### tileops

| batch | heads | dim_k | dim_v | dtype | scale | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 64 | 64 | torch.float32 | 0.125 | 0.01 | 0.07 | 0.14 |
| 1 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.02 | 0.13 | 0.26 |
| 1 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.01 | 0.17 | 0.17 |
| 1 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.01 | 0.17 | 0.17 |
| 8 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.03 | 0.50 | 1.01 |
| 8 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.01 | 1.20 | 1.22 |
| 8 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.01 | 1.18 | 1.20 |
| 16 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.06 | 0.52 | 1.05 |
| 16 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.02 | 1.86 | 1.89 |
| 16 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.02 | 1.83 | 1.86 |
| 32 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.12 | 0.54 | 1.10 |
| 32 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.03 | 1.94 | 1.97 |
| 32 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.04 | 1.92 | 1.95 |
| 64 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.24 | 0.56 | 1.14 |
| 64 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.06 | 2.18 | 2.21 |
| 64 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.06 | 2.15 | 2.19 |

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
| 16 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.15 | 0.22 | 0.23 |
| 32 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.18 | 0.38 | 0.78 |
| 32 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.27 | 0.25 | 0.25 |
| 32 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.27 | 0.25 | 0.25 |
| 64 | 32 | 128 | 128 | torch.float32 | 0.08838834764831845 | 0.33 | 0.41 | 0.82 |
| 64 | 32 | 128 | 128 | torch.float16 | 0.08838834764831845 | 0.50 | 0.27 | 0.28 |
| 64 | 32 | 128 | 128 | torch.bfloat16 | 0.08838834764831845 | 0.50 | 0.27 | 0.27 |

## GroupQueryAttentionFwdOp

### tileops

| batch | seq_len | heads | heads_kv | dim | causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 8 | 4 | 64 | False | torch.float16 | 0.01 | 208.81 | 0.31 |
| 4 | 2048 | 64 | 4 | 128 | False | torch.float16 | 0.79 | 692.44 | 0.36 |
| 4 | 2048 | 64 | 4 | 128 | False | torch.bfloat16 | 0.83 | 665.49 | 0.35 |

### torch

| batch | seq_len | heads | heads_kv | dim | causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 8 | 4 | 64 | False | torch.float16 | 0.02 | 117.48 | 0.17 |
| 4 | 2048 | 64 | 4 | 128 | False | torch.float16 | 1.25 | 439.86 | 0.23 |
| 4 | 2048 | 64 | 4 | 128 | False | torch.bfloat16 | 1.15 | 477.85 | 0.25 |

## GroupQueryAttentionBwdOp

### tileops

| batch | seq_len | heads | heads_kv | dim | causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 8 | 4 | 64 | False | torch.float16 | 0.05 | 102.48 | 0.10 |
| 4 | 2048 | 64 | 4 | 128 | False | torch.float16 | 3.45 | 398.70 | 0.13 |
| 4 | 2048 | 64 | 4 | 128 | False | torch.bfloat16 | 3.36 | 409.35 | 0.13 |

### torch

| batch | seq_len | heads | heads_kv | dim | causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 8 | 4 | 64 | False | torch.float16 | 0.08 | 69.30 | 0.07 |
| 4 | 2048 | 64 | 4 | 128 | False | torch.float16 | 4.91 | 279.86 | 0.09 |
| 4 | 2048 | 64 | 4 | 128 | False | torch.bfloat16 | 4.14 | 332.29 | 0.11 |

## GroupQueryAttentionDecodeWithKVCacheOp

### tileops

| batch | heads | heads_kv | seq_len_kv | dim | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 8 | 8192 | 128 | torch.float16 | 0.02 | 7.44 | 1.86 |
| 4 | 32 | 4 | 4096 | 128 | torch.bfloat16 | 0.02 | 11.61 | 1.45 |
| 8 | 64 | 16 | 8192 | 128 | torch.float16 | 0.15 | 14.54 | 3.64 |

### baseline

| batch | heads | heads_kv | seq_len_kv | dim | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 8 | 8192 | 128 | torch.float16 | 0.43 | 0.31 | 0.08 |
| 4 | 32 | 4 | 4096 | 128 | torch.bfloat16 | 0.82 | 0.33 | 0.04 |
| 8 | 64 | 16 | 8192 | 128 | torch.float16 | 6.41 | 0.33 | 0.08 |

## GroupQueryAttentionDecodePagedWithKVCacheOp

### tileops

| batch | heads | heads_kv | seqlen_kv | dim | page_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 16 | 8 | 512 | 128 | 128 | torch.float16 | 0.02 | 0.22 | 0.11 |
| 2 | 8 | 4 | 1024 | 64 | 256 | torch.float16 | 0.02 | 0.21 | 0.05 |
| 1 | 16 | 4 | 2048 | 128 | 512 | torch.float16 | 0.02 | 0.82 | 0.21 |
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
| 2 | 512 | 8 | 2 | 64 | True | -1 | -1 | torch.float16 | 0.01 | 71.07 | 0.35 |
| 2 | 512 | 8 | 2 | 64 | True | 128 | -1 | torch.float16 | 0.01 | 41.88 | 0.46 |
| 2 | 768 | 8 | 2 | 64 | False | 256 | -1 | torch.float16 | 0.01 | 150.21 | 0.31 |
| 2 | 512 | 8 | 2 | 64 | False | 64 | 64 | torch.bfloat16 | 0.01 | 43.87 | 0.45 |
| 1 | 2048 | 8 | 2 | 64 | True | 512 | -1 | torch.float16 | 0.01 | 152.56 | 0.42 |

### torch

| batch | seq | heads | heads_kv | dim | is_causal | wl | wr | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 512 | 8 | 2 | 64 | True | -1 | -1 | torch.float16 | 0.15 | 3.60 | 0.02 |
| 2 | 512 | 8 | 2 | 64 | True | 128 | -1 | torch.float16 | 0.16 | 1.50 | 0.02 |
| 2 | 768 | 8 | 2 | 64 | False | 256 | -1 | torch.float16 | 0.28 | 6.83 | 0.01 |
| 2 | 512 | 8 | 2 | 64 | False | 64 | 64 | torch.bfloat16 | 0.16 | 1.59 | 0.02 |
| 1 | 2048 | 8 | 2 | 64 | True | 512 | -1 | torch.float16 | 0.81 | 2.31 | 0.01 |

## GqaSlidingWindowVarlenFwdOp

### tileops

| batch | heads | heads_kv | dim | is_causal | wl | wr | dtype | max_seqlen_q | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 8 | 128 | True | -1 | -1 | torch.float16 | 3000 | 0.22 | 339.60 | 0.28 |
| 2 | 32 | 8 | 128 | True | 2000 | -1 | torch.float16 | 3073 | 0.34 | 253.56 | 0.27 |
| 4 | 32 | 8 | 128 | False | 1500 | 1500 | torch.float16 | 3500 | 0.91 | 379.92 | 0.23 |
| 2 | 32 | 8 | 128 | True | -1 | -1 | torch.float16 | 1024 | 0.40 | 405.42 | 0.19 |
| 2 | 32 | 8 | 128 | True | -1 | -1 | torch.float16 | 8193 | 1.95 | 387.06 | 0.14 |

### torch

| batch | heads | heads_kv | dim | is_causal | wl | wr | dtype | max_seqlen_q | max_seqlen_k | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 8 | 128 | True | -1 | -1 | torch.float16 | 3000 | 3000 | 7.65 | 9.64 | 0.01 |
| 2 | 32 | 8 | 128 | True | 2000 | -1 | torch.float16 | 3073 | 3073 | 17.05 | 5.12 | 0.01 |
| 4 | 32 | 8 | 128 | False | 1500 | 1500 | torch.float16 | 3500 | 3500 | 44.44 | 7.74 | 0.00 |
| 2 | 32 | 8 | 128 | True | -1 | -1 | torch.float16 | 1024 | 8192 | 14.96 | 10.77 | 0.01 |
| 2 | 32 | 8 | 128 | True | -1 | -1 | torch.float16 | 8193 | 8193 | 43.28 | 17.44 | 0.01 |

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
| 8 | 128 | 32 | torch.bfloat16 | 0.01 | 0.36 | 0.29 |
| 4 | 256 | 32 | torch.float16 | 0.02 | 0.25 | 0.20 |
| 4 | 128 | 16 | torch.float16 | 0.02 | 0.14 | 0.12 |

## grouped_gemm_nt

### tileops

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_nt | 16384 | 4 | 4864 | 4096 | torch.float16 | False | True | 1.29 | 507.76 | 0.35 |

### baseline

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_nt | 16384 | 4 | 4864 | 4096 | torch.float16 | False | True | 1.63 | 400.82 | 0.28 |

## grouped_gemm_nn

### tileops

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_nn | 16384 | 4 | 4864 | 4096 | torch.float16 | False | False | 1.05 | 622.32 | 0.43 |

### baseline

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_nn | 16384 | 4 | 4864 | 4096 | torch.float16 | False | False | 1.10 | 595.08 | 0.41 |

## grouped_gemm_tn

### tileops

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_tn | 16384 | 4 | 4864 | 4096 | torch.float16 | True | False | 9.26 | 70.51 | 0.05 |

### baseline

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_tn | 16384 | 4 | 4864 | 4096 | torch.float16 | True | False | 1.02 | 637.56 | 0.44 |

## grouped_gemm_tt

### tileops

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_tt | 16384 | 4 | 4864 | 4096 | torch.float16 | True | True | 10.35 | 63.10 | 0.04 |

### baseline

| name | batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_gemm_tt | 16384 | 4 | 4864 | 4096 | torch.float16 | True | True | 1.03 | 632.35 | 0.44 |

## GroupedGemmOp

### tileops

| batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16384 | 4 | 4864 | 4096 | torch.float16 | True | False | 14.78 | 132.49 | N/A |

### baseline

| batch_sum | batch_count | N | K | dtype | transpose_a | transpose_b | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16384 | 4 | 4864 | 4096 | torch.float16 | True | False | 4.93 | 397.41 | N/A |

## leaky_relu

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| leaky_relu | torch.float16 | 4194304 | 0.01 | 0.66 | 2.64 |
| leaky_relu | torch.bfloat16 | 4194304 | 0.01 | 0.66 | 2.64 |
| leaky_relu | torch.float32 | 4194304 | 0.01 | 0.39 | 3.16 |
| leaky_relu | torch.float16 | 10485760 | 0.01 | 0.84 | 3.36 |
| leaky_relu | torch.bfloat16 | 10485760 | 0.01 | 0.84 | 3.37 |
| leaky_relu | torch.float32 | 10485760 | 0.02 | 0.48 | 3.80 |
| leaky_relu | torch.float16 | 20971520 | 0.02 | 0.95 | 3.81 |
| leaky_relu | torch.bfloat16 | 20971520 | 0.02 | 0.95 | 3.81 |
| leaky_relu | torch.float32 | 20971520 | 0.04 | 0.51 | 4.07 |

### baseline

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| leaky_relu | torch.float16 | 4194304 | 0.01 | 0.65 | 2.60 |
| leaky_relu | torch.bfloat16 | 4194304 | 0.01 | 0.65 | 2.60 |
| leaky_relu | torch.float32 | 4194304 | 0.01 | 0.40 | 3.17 |
| leaky_relu | torch.float16 | 10485760 | 0.01 | 0.83 | 3.34 |
| leaky_relu | torch.bfloat16 | 10485760 | 0.01 | 0.84 | 3.34 |
| leaky_relu | torch.float32 | 10485760 | 0.02 | 0.47 | 3.76 |
| leaky_relu | torch.float16 | 20971520 | 0.02 | 0.95 | 3.79 |
| leaky_relu | torch.bfloat16 | 20971520 | 0.02 | 0.95 | 3.79 |
| leaky_relu | torch.float32 | 20971520 | 0.04 | 0.51 | 4.05 |

## elu

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| elu | torch.float16 | 4194304 | 0.01 | 0.61 | 2.44 |
| elu | torch.bfloat16 | 4194304 | 0.01 | 0.61 | 2.45 |
| elu | torch.float32 | 4194304 | 0.01 | 0.39 | 3.10 |
| elu | torch.float16 | 10485760 | 0.01 | 0.76 | 3.04 |
| elu | torch.bfloat16 | 10485760 | 0.01 | 0.77 | 3.07 |
| elu | torch.float32 | 10485760 | 0.02 | 0.46 | 3.70 |
| elu | torch.float16 | 20971520 | 0.02 | 0.86 | 3.43 |
| elu | torch.bfloat16 | 20971520 | 0.02 | 0.86 | 3.44 |
| elu | torch.float32 | 20971520 | 0.04 | 0.50 | 3.97 |

### baseline

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| elu | torch.float16 | 4194304 | 0.01 | 0.43 | 1.70 |
| elu | torch.bfloat16 | 4194304 | 0.01 | 0.42 | 1.68 |
| elu | torch.float32 | 4194304 | 0.01 | 0.37 | 2.98 |
| elu | torch.float16 | 10485760 | 0.02 | 0.52 | 2.07 |
| elu | torch.bfloat16 | 10485760 | 0.02 | 0.51 | 2.04 |
| elu | torch.float32 | 10485760 | 0.02 | 0.45 | 3.60 |
| elu | torch.float16 | 20971520 | 0.04 | 0.57 | 2.27 |
| elu | torch.bfloat16 | 20971520 | 0.04 | 0.56 | 2.24 |
| elu | torch.float32 | 20971520 | 0.04 | 0.50 | 3.97 |

## hardtanh

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| hardtanh | torch.float16 | 4194304 | 0.01 | 0.65 | 2.62 |
| hardtanh | torch.bfloat16 | 4194304 | 0.01 | 0.65 | 2.61 |
| hardtanh | torch.float32 | 4194304 | 0.01 | 0.40 | 3.16 |
| hardtanh | torch.float16 | 10485760 | 0.01 | 0.84 | 3.35 |
| hardtanh | torch.bfloat16 | 10485760 | 0.01 | 0.84 | 3.35 |
| hardtanh | torch.float32 | 10485760 | 0.02 | 0.47 | 3.78 |
| hardtanh | torch.float16 | 20971520 | 0.02 | 0.95 | 3.79 |
| hardtanh | torch.bfloat16 | 20971520 | 0.02 | 0.95 | 3.79 |
| hardtanh | torch.float32 | 20971520 | 0.04 | 0.51 | 4.06 |

### baseline

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| hardtanh | torch.float16 | 4194304 | 0.01 | 0.62 | 2.49 |
| hardtanh | torch.bfloat16 | 4194304 | 0.01 | 0.64 | 2.56 |
| hardtanh | torch.float32 | 4194304 | 0.01 | 0.39 | 3.15 |
| hardtanh | torch.float16 | 10485760 | 0.01 | 0.78 | 3.13 |
| hardtanh | torch.bfloat16 | 10485760 | 0.01 | 0.83 | 3.31 |
| hardtanh | torch.float32 | 10485760 | 0.02 | 0.47 | 3.72 |
| hardtanh | torch.float16 | 20971520 | 0.02 | 0.87 | 3.49 |
| hardtanh | torch.bfloat16 | 20971520 | 0.02 | 0.94 | 3.75 |
| hardtanh | torch.float32 | 20971520 | 0.04 | 0.50 | 3.96 |

## softplus

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| softplus | torch.float16 | 4194304 | 0.01 | 0.42 | 1.69 |
| softplus | torch.bfloat16 | 4194304 | 0.01 | 0.42 | 1.66 |
| softplus | torch.float32 | 4194304 | 0.01 | 0.37 | 2.99 |
| softplus | torch.float16 | 10485760 | 0.02 | 0.51 | 2.06 |
| softplus | torch.bfloat16 | 10485760 | 0.02 | 0.51 | 2.02 |
| softplus | torch.float32 | 10485760 | 0.02 | 0.43 | 3.44 |
| softplus | torch.float16 | 20971520 | 0.04 | 0.56 | 2.25 |
| softplus | torch.bfloat16 | 20971520 | 0.04 | 0.55 | 2.20 |
| softplus | torch.float32 | 20971520 | 0.05 | 0.46 | 3.67 |

### baseline

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| softplus | torch.float16 | 4194304 | 0.01 | 0.36 | 1.43 |
| softplus | torch.bfloat16 | 4194304 | 0.01 | 0.36 | 1.42 |
| softplus | torch.float32 | 4194304 | 0.01 | 0.33 | 2.67 |
| softplus | torch.float16 | 10485760 | 0.02 | 0.43 | 1.70 |
| softplus | torch.bfloat16 | 10485760 | 0.02 | 0.42 | 1.69 |
| softplus | torch.float32 | 10485760 | 0.03 | 0.41 | 3.30 |
| softplus | torch.float16 | 20971520 | 0.05 | 0.46 | 1.84 |
| softplus | torch.bfloat16 | 20971520 | 0.05 | 0.45 | 1.82 |
| softplus | torch.float32 | 20971520 | 0.05 | 0.45 | 3.60 |

## clamp

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| clamp | torch.float16 | 4194304 | 0.01 | 0.65 | 2.61 |
| clamp | torch.bfloat16 | 4194304 | 0.01 | 0.65 | 2.62 |
| clamp | torch.float32 | 4194304 | 0.01 | 0.40 | 3.16 |
| clamp | torch.float16 | 10485760 | 0.01 | 0.84 | 3.36 |
| clamp | torch.bfloat16 | 10485760 | 0.01 | 0.84 | 3.35 |
| clamp | torch.float32 | 10485760 | 0.02 | 0.47 | 3.78 |
| clamp | torch.float16 | 20971520 | 0.02 | 0.95 | 3.79 |
| clamp | torch.bfloat16 | 20971520 | 0.02 | 0.95 | 3.80 |
| clamp | torch.float32 | 20971520 | 0.04 | 0.51 | 4.06 |

### baseline

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| clamp | torch.float16 | 4194304 | 0.01 | 0.62 | 2.48 |
| clamp | torch.bfloat16 | 4194304 | 0.01 | 0.64 | 2.57 |
| clamp | torch.float32 | 4194304 | 0.01 | 0.39 | 3.12 |
| clamp | torch.float16 | 10485760 | 0.01 | 0.78 | 3.12 |
| clamp | torch.bfloat16 | 10485760 | 0.01 | 0.82 | 3.30 |
| clamp | torch.float32 | 10485760 | 0.02 | 0.47 | 3.73 |
| clamp | torch.float16 | 20971520 | 0.02 | 0.87 | 3.49 |
| clamp | torch.bfloat16 | 20971520 | 0.02 | 0.94 | 3.75 |
| clamp | torch.float32 | 20971520 | 0.04 | 0.50 | 4.03 |

## nan_to_num

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| nan_to_num | torch.float16 | 4194304 | 0.01 | 0.64 | 2.56 |
| nan_to_num | torch.bfloat16 | 4194304 | 0.01 | 0.64 | 2.56 |
| nan_to_num | torch.float32 | 4194304 | 0.01 | 0.39 | 3.15 |
| nan_to_num | torch.float16 | 10485760 | 0.01 | 0.82 | 3.28 |
| nan_to_num | torch.bfloat16 | 10485760 | 0.01 | 0.82 | 3.28 |
| nan_to_num | torch.float32 | 10485760 | 0.02 | 0.47 | 3.78 |
| nan_to_num | torch.float16 | 20971520 | 0.02 | 0.92 | 3.68 |
| nan_to_num | torch.bfloat16 | 20971520 | 0.02 | 0.93 | 3.71 |
| nan_to_num | torch.float32 | 20971520 | 0.04 | 0.50 | 4.04 |

### baseline

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| nan_to_num | torch.float16 | 4194304 | 0.01 | 0.63 | 2.52 |
| nan_to_num | torch.bfloat16 | 4194304 | 0.01 | 0.63 | 2.53 |
| nan_to_num | torch.float32 | 4194304 | 0.01 | 0.40 | 3.16 |
| nan_to_num | torch.float16 | 10485760 | 0.01 | 0.82 | 3.27 |
| nan_to_num | torch.bfloat16 | 10485760 | 0.01 | 0.82 | 3.27 |
| nan_to_num | torch.float32 | 10485760 | 0.02 | 0.47 | 3.73 |
| nan_to_num | torch.float16 | 20971520 | 0.02 | 0.93 | 3.71 |
| nan_to_num | torch.bfloat16 | 20971520 | 0.02 | 0.93 | 3.70 |
| nan_to_num | torch.float32 | 20971520 | 0.04 | 0.50 | 4.02 |

## PreluOp

### tileops

| num_channels | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 128 | torch.float16 | 131072 | 0.00 | 0.06 | 0.25 |
| 128 | torch.bfloat16 | 131072 | 0.00 | 0.06 | 0.25 |
| 128 | torch.float32 | 131072 | 0.00 | 0.06 | 0.46 |
| 4096 | torch.float16 | 4194304 | 0.01 | 0.66 | 2.63 |
| 4096 | torch.bfloat16 | 4194304 | 0.01 | 0.66 | 2.63 |
| 4096 | torch.float32 | 4194304 | 0.01 | 0.40 | 3.17 |
| 10240 | torch.float16 | 10485760 | 0.01 | 0.84 | 3.36 |
| 10240 | torch.bfloat16 | 10485760 | 0.01 | 0.84 | 3.35 |
| 10240 | torch.float32 | 10485760 | 0.02 | 0.47 | 3.76 |
| 20480 | torch.float16 | 20971520 | 0.02 | 0.94 | 3.76 |
| 20480 | torch.bfloat16 | 20971520 | 0.02 | 0.94 | 3.76 |
| 20480 | torch.float32 | 20971520 | 0.04 | 0.50 | 4.04 |

### baseline

| num_channels | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 128 | torch.float16 | 131072 | 0.00 | 0.03 | 0.11 |
| 128 | torch.bfloat16 | 131072 | 0.00 | 0.03 | 0.11 |
| 128 | torch.float32 | 131072 | 0.00 | 0.04 | 0.29 |
| 4096 | torch.float16 | 4194304 | 0.01 | 0.29 | 1.14 |
| 4096 | torch.bfloat16 | 4194304 | 0.01 | 0.28 | 1.13 |
| 4096 | torch.float32 | 4194304 | 0.02 | 0.25 | 2.02 |
| 10240 | torch.float16 | 10485760 | 0.03 | 0.34 | 1.34 |
| 10240 | torch.bfloat16 | 10485760 | 0.03 | 0.33 | 1.33 |
| 10240 | torch.float32 | 10485760 | 0.04 | 0.29 | 2.31 |
| 20480 | torch.float16 | 20971520 | 0.06 | 0.36 | 1.45 |
| 20480 | torch.bfloat16 | 20971520 | 0.06 | 0.36 | 1.44 |
| 20480 | torch.float32 | 20971520 | 0.07 | 0.31 | 2.45 |

## WhereOp

### tileops

| dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| torch.float16 | 4194304 | 0.01 | 0.40 | 2.83 |
| torch.bfloat16 | 4194304 | 0.01 | 0.40 | 2.83 |
| torch.float32 | 4194304 | 0.02 | 0.25 | 3.30 |
| torch.float16 | 10485760 | 0.02 | 0.49 | 3.43 |
| torch.bfloat16 | 10485760 | 0.02 | 0.49 | 3.43 |
| torch.float32 | 10485760 | 0.04 | 0.29 | 3.79 |
| torch.float16 | 20971520 | 0.04 | 0.55 | 3.86 |
| torch.bfloat16 | 20971520 | 0.04 | 0.55 | 3.83 |
| torch.float32 | 20971520 | 0.07 | 0.32 | 4.14 |
| torch.float8_e4m3fn | 4194304 | 0.01 | 0.50 | 2.01 |
| torch.float8_e5m2 | 4194304 | 0.01 | 0.50 | 2.02 |
| torch.float8_e4m3fn | 10485760 | 0.02 | 0.59 | 2.37 |
| torch.float8_e5m2 | 10485760 | 0.02 | 0.59 | 2.34 |
| torch.float8_e4m3fn | 20971520 | 0.03 | 0.65 | 2.61 |
| torch.float8_e5m2 | 20971520 | 0.03 | 0.65 | 2.61 |

### baseline

| dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| torch.float16 | 4194304 | 0.01 | 0.41 | 2.87 |
| torch.bfloat16 | 4194304 | 0.01 | 0.41 | 2.88 |
| torch.float32 | 4194304 | 0.02 | 0.26 | 3.34 |
| torch.float16 | 10485760 | 0.02 | 0.50 | 3.49 |
| torch.bfloat16 | 10485760 | 0.02 | 0.50 | 3.48 |
| torch.float32 | 10485760 | 0.04 | 0.29 | 3.83 |
| torch.float16 | 20971520 | 0.04 | 0.55 | 3.87 |
| torch.bfloat16 | 20971520 | 0.04 | 0.56 | 3.91 |
| torch.float32 | 20971520 | 0.07 | 0.32 | 4.11 |
| torch.float8_e4m3fn | 4194304 | 0.01 | 0.59 | 2.35 |
| torch.float8_e5m2 | 4194304 | 0.01 | 0.59 | 2.36 |
| torch.float8_e4m3fn | 10485760 | 0.01 | 0.73 | 2.91 |
| torch.float8_e5m2 | 10485760 | 0.01 | 0.72 | 2.89 |
| torch.float8_e4m3fn | 20971520 | 0.03 | 0.84 | 3.34 |
| torch.float8_e5m2 | 20971520 | 0.03 | 0.83 | 3.34 |

## MaskedFillOp

### tileops

| dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| torch.float16 | 4194304 | 0.01 | 0.54 | 2.70 |
| torch.bfloat16 | 4194304 | 0.01 | 0.54 | 2.71 |
| torch.float32 | 4194304 | 0.01 | 0.35 | 3.11 |
| torch.float16 | 10485760 | 0.02 | 0.67 | 3.34 |
| torch.bfloat16 | 10485760 | 0.02 | 0.67 | 3.33 |
| torch.float32 | 10485760 | 0.02 | 0.42 | 3.78 |
| torch.float16 | 20971520 | 0.03 | 0.76 | 3.78 |
| torch.bfloat16 | 20971520 | 0.03 | 0.75 | 3.77 |
| torch.float32 | 20971520 | 0.05 | 0.45 | 4.06 |
| torch.float8_e4m3fn | 4194304 | 0.01 | 0.58 | 1.73 |
| torch.float8_e5m2 | 4194304 | 0.01 | 0.32 | 0.96 |
| torch.float8_e4m3fn | 10485760 | 0.01 | 0.75 | 2.24 |
| torch.float8_e5m2 | 10485760 | 0.03 | 0.40 | 1.20 |
| torch.float8_e4m3fn | 20971520 | 0.03 | 0.82 | 2.46 |
| torch.float8_e5m2 | 20971520 | 0.05 | 0.41 | 1.24 |

### baseline

| dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- |
| torch.float16 | 4194304 | 0.01 | 0.35 | 1.77 |
| torch.bfloat16 | 4194304 | 0.01 | 0.36 | 1.78 |
| torch.float32 | 4194304 | 0.02 | 0.22 | 2.02 |
| torch.float16 | 10485760 | 0.02 | 0.44 | 2.22 |
| torch.bfloat16 | 10485760 | 0.02 | 0.44 | 2.22 |
| torch.float32 | 10485760 | 0.05 | 0.23 | 2.06 |
| torch.float16 | 20971520 | 0.05 | 0.43 | 2.14 |
| torch.bfloat16 | 20971520 | 0.05 | 0.43 | 2.16 |
| torch.float32 | 20971520 | 0.09 | 0.24 | 2.18 |
| torch.float8_e4m3fn | 4194304 | 0.03 | 0.14 | 0.41 |
| torch.float8_e5m2 | 4194304 | 0.03 | 0.14 | 0.42 |
| torch.float8_e4m3fn | 10485760 | 0.07 | 0.16 | 0.48 |
| torch.float8_e5m2 | 10485760 | 0.06 | 0.16 | 0.49 |
| torch.float8_e4m3fn | 20971520 | 0.14 | 0.15 | 0.46 |
| torch.float8_e5m2 | 20971520 | 0.13 | 0.16 | 0.47 |

## alibi

### tileops

| op_name | seq_len | dim | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| alibi | 512 | 64 | torch.float16 | 0.03 | 0.52 | 1.04 |
| alibi | 512 | 64 | torch.bfloat16 | 0.03 | 0.52 | 1.04 |
| alibi | 512 | 64 | torch.float32 | 0.02 | 0.89 | 3.55 |
| alibi | 2048 | 64 | torch.float16 | 0.28 | 0.95 | 1.90 |
| alibi | 2048 | 64 | torch.bfloat16 | 0.28 | 0.97 | 1.95 |
| alibi | 2048 | 64 | torch.float32 | 0.26 | 1.03 | 4.11 |
| alibi | 4096 | 128 | torch.float16 | 0.97 | 2.21 | 4.42 |
| alibi | 4096 | 128 | torch.bfloat16 | 0.95 | 2.25 | 4.51 |
| alibi | 4096 | 128 | torch.float32 | 1.56 | 1.38 | 5.52 |

### baseline

| op_name | seq_len | dim | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| alibi | 512 | 64 | torch.float16 | 0.08 | 0.21 | 0.41 |
| alibi | 512 | 64 | torch.bfloat16 | 0.08 | 0.20 | 0.41 |
| alibi | 512 | 64 | torch.float32 | 0.06 | 0.30 | 1.21 |
| alibi | 2048 | 64 | torch.float16 | 1.03 | 0.26 | 0.52 |
| alibi | 2048 | 64 | torch.bfloat16 | 1.02 | 0.26 | 0.53 |
| alibi | 2048 | 64 | torch.float32 | 0.66 | 0.40 | 1.62 |
| alibi | 4096 | 128 | torch.float16 | 9.45 | 0.23 | 0.45 |
| alibi | 4096 | 128 | torch.bfloat16 | 7.71 | 0.28 | 0.56 |
| alibi | 4096 | 128 | torch.float32 | 5.96 | 0.36 | 1.44 |

## sinusoidal

### tileops

| op_name | seq_len | dim | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| sinusoidal | 512 | 256 | torch.float16 | 0.00 | 0.04 | 0.09 |
| sinusoidal | 512 | 256 | torch.bfloat16 | 0.00 | 0.04 | 0.09 |
| sinusoidal | 512 | 256 | torch.float32 | 0.00 | 0.05 | 0.21 |
| sinusoidal | 2048 | 300 | torch.float16 | 0.00 | 0.15 | 0.29 |
| sinusoidal | 2048 | 300 | torch.bfloat16 | 0.00 | 0.15 | 0.29 |
| sinusoidal | 2048 | 300 | torch.float32 | 0.00 | 0.13 | 0.51 |
| sinusoidal | 4096 | 512 | torch.float16 | 0.01 | 0.30 | 0.60 |
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
| sinusoidal | 4096 | 512 | torch.float32 | 0.03 | 0.07 | 0.30 |

## leaky_relu_fp8

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| leaky_relu | torch.float8_e4m3fn | 4194304 | 0.01 | 0.49 | 0.97 |
| leaky_relu | torch.float8_e5m2 | 4194304 | 0.03 | 0.16 | 0.32 |
| leaky_relu | torch.float8_e4m3fn | 10485760 | 0.02 | 0.56 | 1.13 |
| leaky_relu | torch.float8_e5m2 | 10485760 | 0.05 | 0.20 | 0.40 |
| leaky_relu | torch.float8_e4m3fn | 20971520 | 0.03 | 0.61 | 1.23 |
| leaky_relu | torch.float8_e5m2 | 20971520 | 0.10 | 0.21 | 0.43 |

### baseline

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| leaky_relu | torch.float8_e4m3fn | 4194304 | 0.03 | 0.16 | 0.33 |
| leaky_relu | torch.float8_e5m2 | 4194304 | 0.02 | 0.17 | 0.34 |
| leaky_relu | torch.float8_e4m3fn | 10485760 | 0.05 | 0.19 | 0.38 |
| leaky_relu | torch.float8_e5m2 | 10485760 | 0.05 | 0.20 | 0.39 |
| leaky_relu | torch.float8_e4m3fn | 20971520 | 0.11 | 0.19 | 0.38 |
| leaky_relu | torch.float8_e5m2 | 20971520 | 0.11 | 0.19 | 0.39 |

## elu_fp8

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| elu | torch.float8_e4m3fn | 4194304 | 0.01 | 0.66 | 1.32 |
| elu | torch.float8_e5m2 | 4194304 | 0.02 | 0.26 | 0.52 |
| elu | torch.float8_e4m3fn | 10485760 | 0.01 | 0.85 | 1.70 |
| elu | torch.float8_e5m2 | 10485760 | 0.03 | 0.31 | 0.62 |
| elu | torch.float8_e4m3fn | 20971520 | 0.02 | 0.95 | 1.90 |
| elu | torch.float8_e5m2 | 20971520 | 0.07 | 0.32 | 0.64 |

### baseline

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| elu | torch.float8_e4m3fn | 4194304 | 0.03 | 0.14 | 0.28 |
| elu | torch.float8_e5m2 | 4194304 | 0.03 | 0.14 | 0.29 |
| elu | torch.float8_e4m3fn | 10485760 | 0.06 | 0.16 | 0.32 |
| elu | torch.float8_e5m2 | 10485760 | 0.06 | 0.17 | 0.33 |
| elu | torch.float8_e4m3fn | 20971520 | 0.13 | 0.17 | 0.33 |
| elu | torch.float8_e5m2 | 20971520 | 0.12 | 0.17 | 0.34 |

## clamp_fp8

### tileops

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| clamp | torch.float8_e4m3fn | 4194304 | 0.00 | 0.98 | 1.96 |
| clamp | torch.float8_e5m2 | 4194304 | 0.01 | 0.39 | 0.78 |
| clamp | torch.float8_e4m3fn | 10485760 | 0.01 | 1.32 | 2.63 |
| clamp | torch.float8_e5m2 | 10485760 | 0.02 | 0.49 | 0.99 |
| clamp | torch.float8_e4m3fn | 20971520 | 0.01 | 1.55 | 3.11 |
| clamp | torch.float8_e5m2 | 20971520 | 0.04 | 0.51 | 1.02 |

### baseline

| op_name | dtype | n_total | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| clamp | torch.float8_e4m3fn | 4194304 | 0.03 | 0.16 | 0.32 |
| clamp | torch.float8_e5m2 | 4194304 | 0.03 | 0.16 | 0.33 |
| clamp | torch.float8_e4m3fn | 10485760 | 0.06 | 0.19 | 0.37 |
| clamp | torch.float8_e5m2 | 10485760 | 0.05 | 0.19 | 0.39 |
| clamp | torch.float8_e4m3fn | 20971520 | 0.11 | 0.19 | 0.37 |
| clamp | torch.float8_e5m2 | 20971520 | 0.11 | 0.19 | 0.38 |

## InstanceNormOp

### tileops

| n | c | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 8 | 128 | torch.float16 | 0.02 | 0.33 | 0.26 |
| 8 | 128 | torch.bfloat16 | 0.02 | 0.34 | 0.28 |
| 4 | 256 | torch.float16 | 0.02 | 0.22 | 0.18 |
| 4 | 64 | torch.float16 | 0.01 | 0.08 | 0.07 |

### baseline

| n | c | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 8 | 128 | torch.float16 | 0.02 | 0.25 | 0.20 |
| 8 | 128 | torch.bfloat16 | 0.02 | 0.25 | 0.20 |
| 4 | 256 | torch.float16 | 0.02 | 0.19 | 0.16 |
| 4 | 64 | torch.float16 | 0.01 | 0.09 | 0.07 |

## LayerNormOp

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.01 | 1.92 | 1.53 |
| 4096 | 4096 | torch.bfloat16 | 0.04 | 2.25 | 1.80 |
| 2048 | 5120 | torch.float16 | 0.03 | 2.03 | 1.62 |
| 1025 | 4096 | torch.float16 | 0.01 | 1.91 | 1.53 |

### baseline

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.01 | 1.64 | 1.31 |
| 4096 | 4096 | torch.bfloat16 | 0.03 | 2.56 | 2.04 |
| 2048 | 5120 | torch.float16 | 0.02 | 2.34 | 1.87 |
| 1025 | 4096 | torch.float16 | 0.01 | 1.62 | 1.30 |

## AnyOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | any | 0.01 | 0.40 | 0.80 |
| 1024 | 4096 | torch.bfloat16 | any | 0.01 | 0.40 | 0.79 |
| 1024 | 4096 | torch.float32 | any | 0.01 | 0.29 | 1.14 |
| 1024 | 4096 | torch.int32 | any | 0.03 | 0.16 | 0.62 |
| 4096 | 4096 | torch.float16 | any | 0.03 | 0.62 | 1.24 |

### baseline

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | any | 0.03 | 0.16 | 0.32 |
| 1024 | 4096 | torch.bfloat16 | any | 0.03 | 0.16 | 0.32 |
| 1024 | 4096 | torch.float32 | any | 0.03 | 0.15 | 0.62 |
| 1024 | 4096 | torch.int32 | any | 0.03 | 0.15 | 0.62 |
| 4096 | 4096 | torch.float16 | any | 0.07 | 0.25 | 0.50 |

## AllOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | all | 0.01 | 0.40 | 0.80 |
| 1024 | 4096 | torch.bfloat16 | all | 0.01 | 0.40 | 0.79 |
| 1024 | 4096 | torch.float32 | all | 0.01 | 0.29 | 1.14 |
| 1024 | 4096 | torch.int32 | all | 0.03 | 0.16 | 0.62 |
| 4096 | 4096 | torch.float16 | all | 0.03 | 0.62 | 1.24 |

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
| 1024 | 4096 | torch.float16 | count_nonzero | 0.01 | 0.53 | 1.05 |
| 1024 | 4096 | torch.bfloat16 | count_nonzero | 0.01 | 0.53 | 1.06 |
| 1024 | 4096 | torch.float32 | count_nonzero | 0.01 | 0.34 | 1.37 |
| 1024 | 4096 | torch.int32 | count_nonzero | 0.02 | 0.17 | 0.69 |
| 4096 | 4096 | torch.float16 | count_nonzero | 0.02 | 0.69 | 1.39 |

### baseline

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | count_nonzero | 0.04 | 0.12 | 0.24 |
| 1024 | 4096 | torch.bfloat16 | count_nonzero | 0.04 | 0.12 | 0.24 |
| 1024 | 4096 | torch.float32 | count_nonzero | 0.04 | 0.11 | 0.44 |
| 1024 | 4096 | torch.int32 | count_nonzero | 0.04 | 0.11 | 0.45 |
| 4096 | 4096 | torch.float16 | count_nonzero | 0.11 | 0.15 | 0.31 |

## MeanPoolingForwardOp

### tileops

| batch_size | seq_len | heads | dim | chunk_size | dtype | accum_dtype | chunks_per_bacth | seq_num | use_offsets | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8192 | 64 | 128 | 64 | torch.float16 | torch.float32 | 128 | 1 | 0 | 0.13 | 0.52 | 1.05 |
| 2 | 2048 | 64 | 128 | 64 | torch.float16 | torch.float32 | 32 | 2 | 0 | 0.07 | 0.47 | 0.95 |
| 1 | 8192 | 64 | 128 | 64 | torch.float16 | torch.float32 | 128 | 4 | 1 | 0.19 | 0.36 | 0.73 |
| 1 | 1000 | 64 | 128 | 32 | torch.float16 | torch.float32 | 34 | 4 | 1 | 0.02 | 0.39 | 0.74 |

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
| 1 | 1024 | 8 | 64 | False | torch.float16 | 0.01 | 205.77 | 0.40 |
| 16 | 2048 | 16 | 128 | False | torch.float16 | 0.85 | 646.67 | 0.63 |
| 4 | 4096 | 16 | 128 | False | torch.bfloat16 | 0.75 | 729.56 | 0.36 |

### torch

| batch | seq_len | heads | dim | causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 8 | 64 | False | torch.float16 | 0.02 | 116.64 | 0.23 |
| 16 | 2048 | 16 | 128 | False | torch.float16 | 1.43 | 383.85 | 0.37 |
| 4 | 4096 | 16 | 128 | False | torch.bfloat16 | 1.57 | 350.90 | 0.17 |

## MultiHeadAttentionBwdOp

### tileops

| batch | seq_len | heads | dim | causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 8 | 64 | False | torch.float16 | 0.04 | 125.90 | 0.17 |
| 16 | 2048 | 16 | 128 | False | torch.float16 | 2.70 | 509.06 | 0.35 |
| 4 | 4096 | 16 | 128 | False | torch.bfloat16 | 3.05 | 450.93 | 0.15 |

### torch

| batch | seq_len | heads | dim | causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 8 | 64 | False | torch.float16 | 0.07 | 79.98 | 0.11 |
| 16 | 2048 | 16 | 128 | False | torch.float16 | 4.88 | 281.83 | 0.19 |
| 4 | 4096 | 16 | 128 | False | torch.bfloat16 | 4.00 | 343.74 | 0.12 |

## MultiHeadAttentionDecodeWithKVCacheOp

### tileops

| b | h | s_q | s_kv | d | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 128 | 8192 | 128 | torch.float16 | 0.05 | 318.78 | 2.53 |
| 1 | 32 | 128 | 8192 | 128 | torch.bfloat16 | 0.05 | 320.58 | 2.54 |

### torch

| b | h | s_q | s_kv | d | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 128 | 8192 | 128 | torch.float16 | 0.07 | 256.71 | 2.04 |
| 1 | 32 | 128 | 8192 | 128 | torch.bfloat16 | 0.07 | 255.80 | 2.03 |

## MultiHeadAttentionDecodePagedWithKVCacheOp

### tileops

| batch | heads | seqlen_q | seqlen_kv | dim | page_size | is_causal | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 16 | 1 | 512 | 128 | 128 | False | torch.float16 | 0.02 | 0.18 | 0.18 |
| 2 | 8 | 1 | 1024 | 64 | 256 | False | torch.float16 | 0.02 | 0.19 | 0.10 |
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
| 1 | 4 | 1280 | torch.bfloat16 | 0.00 | 28.74 | 0.01 |
| 2 | 4 | 1920 | torch.bfloat16 | 0.00 | 122.76 | 0.01 |
| 4 | 4 | 2560 | torch.bfloat16 | 0.00 | 415.25 | 0.01 |

### baseline

| batch | n_expand | c_x | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | 1280 | torch.bfloat16 | 0.01 | 3.96 | 0.00 |
| 2 | 4 | 1920 | torch.bfloat16 | 0.01 | 16.63 | 0.00 |
| 4 | 4 | 2560 | torch.bfloat16 | 0.02 | 55.77 | 0.00 |

## ManifoldConstrainedHyperConnectionPreOp

### tileops

| batch | n_expand | c_x | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | 1280 | torch.bfloat16 | 0.04 | 30.18 | 0.00 |
| 2 | 4 | 1920 | torch.bfloat16 | 0.05 | 104.00 | 0.00 |
| 4 | 4 | 2560 | torch.bfloat16 | 0.07 | 291.05 | 0.00 |

### baseline

| batch | n_expand | c_x | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | 1280 | torch.bfloat16 | 0.28 | 4.57 | 0.00 |
| 2 | 4 | 1920 | torch.bfloat16 | 0.30 | 18.68 | 0.00 |
| 4 | 4 | 2560 | torch.bfloat16 | 0.34 | 59.47 | 0.00 |

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
| 512 | 2 | 8 | 4096 | torch.bfloat16 | 0.01 | N/A | 1.53 |
| 2048 | 2 | 8 | 4096 | torch.bfloat16 | 0.02 | N/A | 2.17 |
| 4096 | 2 | 8 | 4096 | torch.bfloat16 | 0.05 | N/A | 2.23 |
| 512 | 8 | 256 | 7168 | torch.bfloat16 | 0.05 | N/A | 1.46 |
| 2048 | 8 | 256 | 7168 | torch.bfloat16 | 0.17 | N/A | 1.57 |
| 4096 | 8 | 256 | 7168 | torch.bfloat16 | 0.34 | N/A | 1.55 |
| 512 | 8 | 128 | 2048 | torch.bfloat16 | 0.02 | N/A | 0.81 |
| 2048 | 8 | 128 | 2048 | torch.bfloat16 | 0.07 | N/A | 1.08 |
| 4096 | 8 | 128 | 2048 | torch.bfloat16 | 0.14 | N/A | 1.09 |

### pytorch-ref

| total_tokens | top_k | num_experts | hidden_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 2 | 8 | 4096 | torch.bfloat16 | 1.29 | N/A | 0.01 |
| 2048 | 2 | 8 | 4096 | torch.bfloat16 | 5.32 | N/A | 0.01 |
| 4096 | 2 | 8 | 4096 | torch.bfloat16 | 11.00 | N/A | 0.01 |
| 512 | 8 | 256 | 7168 | torch.bfloat16 | 5.19 | N/A | 0.01 |
| 2048 | 8 | 256 | 7168 | torch.bfloat16 | 22.85 | N/A | 0.01 |
| 4096 | 8 | 256 | 7168 | torch.bfloat16 | 47.87 | N/A | 0.01 |
| 512 | 8 | 128 | 2048 | torch.bfloat16 | 4.57 | N/A | 0.00 |
| 2048 | 8 | 128 | 2048 | torch.bfloat16 | 18.52 | N/A | 0.00 |
| 4096 | 8 | 128 | 2048 | torch.bfloat16 | 38.55 | N/A | 0.00 |

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

## Qwen3MoEOp

### tileops

| num_tokens | num_experts | top_k | hidden_size | ffn_size | scoring_func | renormalize | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 128 | 8 | 2048 | 1024 | softmax | False | torch.bfloat16 | 0.60 | 85.54 | 2.68 |
| 2048 | 128 | 8 | 2048 | 1024 | softmax | False | torch.bfloat16 | 0.95 | 215.98 | 1.70 |
| 4096 | 128 | 8 | 2048 | 1024 | softmax | False | torch.bfloat16 | 1.46 | 282.53 | 1.13 |
| 512 | 256 | 8 | 2048 | 1024 | softmax | True | torch.bfloat16 | 1.20 | 42.87 | 2.68 |
| 2048 | 256 | 8 | 2048 | 1024 | softmax | True | torch.bfloat16 | 1.47 | 140.17 | 2.20 |
| 4096 | 256 | 8 | 2048 | 1024 | softmax | True | torch.bfloat16 | 1.96 | 209.96 | 1.66 |
| 512 | 256 | 8 | 2048 | 1024 | sigmoid | True | torch.bfloat16 | 1.13 | 45.53 | 2.85 |
| 2048 | 256 | 8 | 2048 | 1024 | sigmoid | True | torch.bfloat16 | 1.43 | 144.66 | 2.27 |
| 4096 | 256 | 8 | 2048 | 1024 | sigmoid | True | torch.bfloat16 | 1.95 | 211.98 | 1.67 |

### torch

| num_tokens | num_experts | top_k | hidden_size | ffn_size | scoring_func | renormalize | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 128 | 8 | 2048 | 1024 | softmax | False | torch.bfloat16 | 12.29 | 4.19 | 0.13 |
| 2048 | 128 | 8 | 2048 | 1024 | softmax | False | torch.bfloat16 | 17.96 | 11.48 | 0.09 |
| 4096 | 128 | 8 | 2048 | 1024 | softmax | False | torch.bfloat16 | 23.93 | 17.23 | 0.07 |
| 512 | 256 | 8 | 2048 | 1024 | softmax | True | torch.bfloat16 | 24.53 | 2.10 | 0.13 |
| 2048 | 256 | 8 | 2048 | 1024 | softmax | True | torch.bfloat16 | 29.30 | 7.04 | 0.11 |
| 4096 | 256 | 8 | 2048 | 1024 | softmax | True | torch.bfloat16 | 36.03 | 11.44 | 0.09 |
| 512 | 256 | 8 | 2048 | 1024 | sigmoid | True | torch.bfloat16 | 24.57 | 2.10 | 0.13 |
| 2048 | 256 | 8 | 2048 | 1024 | sigmoid | True | torch.bfloat16 | 29.36 | 7.02 | 0.11 |
| 4096 | 256 | 8 | 2048 | 1024 | sigmoid | True | torch.bfloat16 | 35.79 | 11.52 | 0.09 |

## moe_unpermute

### tileops

| total_tokens | top_k | hidden_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 512 | 2 | 4096 | torch.bfloat16 | 0.01 | 1.38 | 2.07 |
| 2048 | 2 | 4096 | torch.bfloat16 | 0.02 | 2.03 | 3.04 |
| 4096 | 2 | 4096 | torch.bfloat16 | 0.03 | 2.30 | 3.45 |
| 512 | 8 | 7168 | torch.bfloat16 | 0.03 | 2.34 | 2.63 |
| 2048 | 8 | 7168 | torch.bfloat16 | 0.07 | 3.21 | 3.62 |
| 4096 | 8 | 7168 | torch.bfloat16 | 0.13 | 3.66 | 4.12 |
| 512 | 8 | 2048 | torch.bfloat16 | 0.01 | 2.05 | 2.31 |
| 2048 | 8 | 2048 | torch.bfloat16 | 0.02 | 2.81 | 3.16 |
| 4096 | 8 | 2048 | torch.bfloat16 | 0.04 | 3.17 | 3.57 |

### pytorch-vec

| total_tokens | top_k | hidden_size | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 512 | 2 | 4096 | torch.bfloat16 | 0.06 | 0.15 | 0.22 |
| 2048 | 2 | 4096 | torch.bfloat16 | 0.20 | 0.17 | 0.25 |
| 4096 | 2 | 4096 | torch.bfloat16 | 0.38 | 0.18 | 0.26 |
| 512 | 8 | 7168 | torch.bfloat16 | 0.32 | 0.19 | 0.21 |
| 2048 | 8 | 7168 | torch.bfloat16 | 1.17 | 0.20 | 0.23 |
| 4096 | 8 | 7168 | torch.bfloat16 | 2.31 | 0.20 | 0.23 |
| 512 | 8 | 2048 | torch.bfloat16 | 0.10 | 0.17 | 0.19 |
| 2048 | 8 | 2048 | torch.bfloat16 | 0.36 | 0.19 | 0.21 |
| 4096 | 8 | 2048 | torch.bfloat16 | 0.69 | 0.20 | 0.22 |

## SumOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | sum | 0.01 | 0.53 | 1.06 |
| 1024 | 4096 | torch.bfloat16 | sum | 0.01 | 0.53 | 1.06 |
| 4096 | 4096 | torch.float16 | sum | 0.02 | 0.70 | 1.40 |

### baseline

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | sum | 0.03 | 0.16 | 0.32 |
| 1024 | 4096 | torch.bfloat16 | sum | 0.03 | 0.16 | 0.32 |
| 4096 | 4096 | torch.float16 | sum | 0.08 | 0.21 | 0.42 |

## MeanOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | mean | 0.01 | 0.52 | 1.05 |

### baseline

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | mean | 0.03 | 0.16 | 0.31 |

## AmaxOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | amax | 0.01 | 0.53 | 1.06 |

### baseline

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | amax | 0.03 | 0.16 | 0.31 |

## AminOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | amin | 0.01 | 0.53 | 1.06 |

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
| 1024 | 4096 | torch.float16 | std | 0.01 | 1.38 | 0.92 |

### baseline

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | std | 0.04 | 0.32 | 0.22 |

## VarOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | var | 0.01 | 1.40 | 0.93 |

### baseline

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | var | 0.04 | 0.33 | 0.22 |

## VarMeanOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | var_mean | 0.01 | 1.50 | 1.00 |

### baseline

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | var_mean | 0.05 | 0.26 | 0.17 |

## RmsNormOp

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.01 | 1.72 | 1.72 |
| 4096 | 4096 | torch.bfloat16 | 0.03 | 2.05 | 2.05 |
| 2048 | 5120 | torch.float16 | 0.02 | 1.97 | 1.97 |
| 1025 | 4096 | torch.float16 | 0.01 | 1.70 | 1.70 |

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
| 2048 | 128 | torch.float16 | 0.00 | 0.41 | 0.62 |
| 2048 | 128 | torch.bfloat16 | 0.00 | 0.41 | 0.61 |
| 2048 | 128 | torch.float32 | 0.00 | 0.33 | 1.00 |
| 4096 | 128 | torch.float16 | 0.00 | 0.67 | 1.01 |
| 4096 | 128 | torch.bfloat16 | 0.00 | 0.67 | 1.01 |
| 4096 | 128 | torch.float32 | 0.00 | 0.53 | 1.60 |

### baseline

| seq_len | head_dim | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 2048 | 64 | torch.float16 | 0.01 | 0.04 | 0.06 |
| 2048 | 64 | torch.bfloat16 | 0.01 | 0.04 | 0.06 |
| 2048 | 64 | torch.float32 | 0.01 | 0.04 | 0.12 |
| 2048 | 128 | torch.float16 | 0.01 | 0.07 | 0.11 |
| 2048 | 128 | torch.bfloat16 | 0.01 | 0.07 | 0.11 |
| 2048 | 128 | torch.float32 | 0.01 | 0.07 | 0.22 |
| 4096 | 128 | torch.float16 | 0.02 | 0.13 | 0.19 |
| 4096 | 128 | torch.bfloat16 | 0.02 | 0.13 | 0.20 |
| 4096 | 128 | torch.float32 | 0.02 | 0.12 | 0.37 |

## SoftmaxOp

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.02 | 1.04 | 1.04 |
| 4096 | 4096 | torch.bfloat16 | 0.06 | 1.16 | 1.16 |
| 1024 | 3000 | torch.float16 | 0.02 | 0.49 | 0.49 |
| 1025 | 4096 | torch.float16 | 0.02 | 1.04 | 1.04 |

### baseline

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.02 | 0.96 | 0.96 |
| 4096 | 4096 | torch.bfloat16 | 0.06 | 1.13 | 1.13 |
| 1024 | 3000 | torch.float16 | 0.01 | 0.83 | 0.83 |
| 1025 | 4096 | torch.float16 | 0.02 | 0.97 | 0.97 |

## LogSoftmaxOp

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.02 | 1.00 | 0.80 |
| 4096 | 4096 | torch.bfloat16 | 0.08 | 1.08 | 0.86 |
| 1024 | 3000 | torch.float16 | 0.03 | 0.52 | 0.42 |
| 1025 | 4096 | torch.float16 | 0.02 | 1.01 | 0.80 |

### baseline

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.02 | 1.36 | 1.09 |
| 4096 | 4096 | torch.bfloat16 | 0.05 | 1.58 | 1.27 |
| 1024 | 3000 | torch.float16 | 0.01 | 1.15 | 0.92 |
| 1025 | 4096 | torch.float16 | 0.02 | 1.34 | 1.07 |

## LogSumExpOp

### tileops

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.01 | 1.28 | 0.86 |
| 4096 | 4096 | torch.bfloat16 | 0.03 | 1.57 | 1.04 |
| 1024 | 3000 | torch.float16 | 0.02 | 0.56 | 0.38 |
| 1025 | 4096 | torch.float16 | 0.01 | 1.27 | 0.85 |

### baseline

| m | n | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | 0.05 | 0.25 | 0.16 |
| 4096 | 4096 | torch.bfloat16 | 0.10 | 0.50 | 0.33 |
| 1024 | 3000 | torch.float16 | 0.04 | 0.21 | 0.14 |
| 1025 | 4096 | torch.float16 | 0.05 | 0.24 | 0.16 |

## SsdChunkScanFwdOp

### tileops

| batch | num_chunks | chunk_len | n_heads | d_head | d_state | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 64 | 4 | 64 | 32 | torch.float16 | 110.75 | 0.00 | 0.00 |
| 2 | 4 | 64 | 8 | 64 | 64 | torch.float16 | 111.52 | 0.00 | 0.00 |
| 1 | 2 | 128 | 4 | 128 | 32 | torch.bfloat16 | 113.60 | 0.00 | 0.00 |
| 2 | 2 | 64 | 4 | 64 | 32 | torch.bfloat16 | 111.73 | 0.00 | 0.00 |

## ssd_chunk_scan_fwd

### baseline

| batch | num_chunks | chunk_len | n_heads | d_head | d_state | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 64 | 4 | 64 | 32 | torch.float16 | 0.05 | 0.08 | 0.01 |
| 2 | 4 | 64 | 8 | 64 | 64 | torch.float16 | 0.06 | 0.82 | 0.05 |
| 1 | 2 | 128 | 4 | 128 | 32 | torch.bfloat16 | 0.06 | 0.42 | 0.02 |
| 2 | 2 | 64 | 4 | 64 | 32 | torch.bfloat16 | 0.06 | 0.15 | 0.01 |

## SsdChunkStateFwdOp

### tileops

| batch | num_chunks | chunk_len | n_heads | d_head | d_state | n_groups | dtype | has_seq_idx | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 64 | 4 | 64 | 32 | 1 | torch.float16 | False | 95.27 | 0.00 | 0.00 |
| 2 | 4 | 64 | 8 | 64 | 64 | 2 | torch.float16 | False | 95.18 | 0.00 | 0.00 |
| 1 | 2 | 128 | 4 | 128 | 32 | 1 | torch.bfloat16 | False | 94.56 | 0.00 | 0.00 |
| 2 | 2 | 64 | 4 | 64 | 32 | 2 | torch.bfloat16 | False | 95.37 | 0.00 | 0.00 |
| 2 | 4 | 64 | 8 | 64 | 64 | 2 | torch.float16 | True | 95.56 | 0.00 | 0.00 |

## ssd_chunk_state_fwd

### baseline

| batch | num_chunks | chunk_len | n_heads | d_head | d_state | n_groups | dtype | has_seq_idx | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 64 | 4 | 64 | 32 | 1 | torch.float16 | False | 0.03 | 0.06 | 0.00 |
| 2 | 4 | 64 | 8 | 64 | 64 | 2 | torch.float16 | False | 0.11 | 0.29 | 0.02 |
| 1 | 2 | 128 | 4 | 128 | 32 | 1 | torch.bfloat16 | False | 0.05 | 0.18 | 0.01 |
| 2 | 2 | 64 | 4 | 64 | 32 | 2 | torch.bfloat16 | False | 0.04 | 0.10 | 0.01 |
| 2 | 4 | 64 | 8 | 64 | 64 | 2 | torch.float16 | True | 0.12 | 0.28 | 0.01 |

## SsdStatePassingFwdOp

### tileops

| batch | num_chunks | n_heads | d_state | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 4 | 32 | torch.float16 | 57.78 | 0.00 | 0.00 |
| 2 | 4 | 8 | 64 | torch.float16 | 56.80 | 0.00 | 0.00 |
| 1 | 2 | 4 | 32 | torch.bfloat16 | 58.43 | 0.00 | 0.00 |
| 2 | 4 | 8 | 64 | torch.bfloat16 | 56.86 | 0.00 | 0.00 |

## ssd_state_passing_fwd

### baseline

| batch | num_chunks | n_heads | d_state | dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 4 | 32 | torch.float16 | 0.02 | 0.00 | 0.00 |
| 2 | 4 | 8 | 64 | torch.float16 | 0.04 | 0.00 | 0.00 |
| 1 | 2 | 4 | 32 | torch.bfloat16 | 0.02 | 0.00 | 0.00 |
| 2 | 4 | 8 | 64 | torch.bfloat16 | 0.04 | 0.00 | 0.00 |

## TopkSelectorOp

### tileops

| batch | seq_len | seq_len_kv | kv_group | topk | in_dtype | out_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32768 | 65536 | 1 | 1024 | torch.float32 | torch.int32 | 13.53 | N/A | 0.64 |
| 1 | 32768 | 65536 | 1 | 2048 | torch.float32 | torch.int32 | 14.00 | N/A | 0.63 |
| 1 | 65535 | 131072 | 1 | 1024 | torch.float32 | torch.int32 | 44.56 | N/A | 0.78 |
| 1 | 65535 | 131072 | 1 | 2048 | torch.float32 | torch.int32 | 44.96 | N/A | 0.78 |

### baseline

| batch | seq_len | seq_len_kv | kv_group | topk | in_dtype | out_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32768 | 65536 | 1 | 1024 | torch.float32 | torch.int32 | 20.36 | N/A | 0.43 |
| 1 | 32768 | 65536 | 1 | 2048 | torch.float32 | torch.int32 | 20.16 | N/A | 0.44 |
| 1 | 65535 | 131072 | 1 | 1024 | torch.float32 | torch.int32 | 109.39 | N/A | 0.32 |
| 1 | 65535 | 131072 | 1 | 2048 | torch.float32 | torch.int32 | 112.15 | N/A | 0.31 |

## exp

### tileops

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| exp | 262144 | torch.float16 | torch.float16 | 0.00 | 0.11 | 0.45 |
| exp | 1048576 | torch.float16 | torch.float16 | 0.00 | 0.33 | 1.32 |
| exp | 4000000 | torch.float16 | torch.float16 | 0.01 | 0.62 | 2.50 |
| exp | 262144 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.11 | 0.45 |
| exp | 1048576 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.32 | 1.29 |
| exp | 4000000 | torch.bfloat16 | torch.bfloat16 | 0.01 | 0.63 | 2.52 |

### baseline

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| exp | 262144 | torch.float16 | torch.float16 | 0.00 | 0.11 | 0.44 |
| exp | 1048576 | torch.float16 | torch.float16 | 0.00 | 0.32 | 1.27 |
| exp | 4000000 | torch.float16 | torch.float16 | 0.01 | 0.63 | 2.52 |
| exp | 262144 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.11 | 0.44 |
| exp | 1048576 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.32 | 1.27 |
| exp | 4000000 | torch.bfloat16 | torch.bfloat16 | 0.01 | 0.63 | 2.52 |

## gelu

### tileops

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| gelu | 262144 | torch.float16 | torch.float16 | 0.00 | 0.11 | 0.43 |
| gelu | 1048576 | torch.float16 | torch.float16 | 0.00 | 0.29 | 1.17 |
| gelu | 4000000 | torch.float16 | torch.float16 | 0.01 | 0.47 | 1.89 |
| gelu | 262144 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.11 | 0.43 |
| gelu | 1048576 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.29 | 1.15 |
| gelu | 4000000 | torch.bfloat16 | torch.bfloat16 | 0.01 | 0.46 | 1.83 |

### baseline

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| gelu | 262144 | torch.float16 | torch.float16 | 0.00 | 0.10 | 0.41 |
| gelu | 1048576 | torch.float16 | torch.float16 | 0.00 | 0.28 | 1.14 |
| gelu | 4000000 | torch.float16 | torch.float16 | 0.01 | 0.51 | 2.05 |
| gelu | 262144 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.10 | 0.41 |
| gelu | 1048576 | torch.bfloat16 | torch.bfloat16 | 0.00 | 0.28 | 1.12 |
| gelu | 4000000 | torch.bfloat16 | torch.bfloat16 | 0.01 | 0.49 | 1.97 |

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
| logical_not | 4000000 | torch.float16 | torch.bool | 0.01 | 0.72 | 2.15 |

## bitwise_not

### tileops

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| bitwise_not | 262144 | torch.int32 | torch.int32 | 0.00 | 0.10 | 0.76 |
| bitwise_not | 1048576 | torch.int32 | torch.int32 | 0.01 | 0.19 | 1.54 |
| bitwise_not | 4000000 | torch.int32 | torch.int32 | 0.02 | 0.27 | 2.13 |

### baseline

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| bitwise_not | 262144 | torch.int32 | torch.int32 | 0.00 | 0.10 | 0.82 |
| bitwise_not | 1048576 | torch.int32 | torch.int32 | 0.00 | 0.24 | 1.94 |
| bitwise_not | 4000000 | torch.int32 | torch.int32 | 0.01 | 0.40 | 3.17 |

## isnan

### tileops

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| isnan | 262144 | torch.float16 | torch.bool | 0.00 | 0.11 | 0.32 |
| isnan | 1048576 | torch.float16 | torch.bool | 0.00 | 0.21 | 0.64 |
| isnan | 4000000 | torch.float16 | torch.bool | 0.01 | 0.30 | 0.90 |

### baseline

| op_name | n_total | dtype | output_dtype | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| isnan | 262144 | torch.float16 | torch.bool | 0.00 | 0.11 | 0.34 |
| isnan | 1048576 | torch.float16 | torch.bool | 0.00 | 0.33 | 1.00 |
| isnan | 4000000 | torch.float16 | torch.bool | 0.01 | 0.73 | 2.18 |

## unary_strategy

### relu_direct

| n_total | shape_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4194304 | 1024x4096 | torch.float16 | direct | 0.01 | 0.29 | 1.16 |
| 4194304 | 1024x4096 | torch.bfloat16 | direct | 0.01 | 0.29 | 1.16 |
| 4194304 | 1024x4096 | torch.float32 | direct | 0.02 | 0.26 | 2.10 |
| 10485760 | 1024x10240 | torch.float16 | direct | 0.03 | 0.31 | 1.26 |
| 10485760 | 1024x10240 | torch.bfloat16 | direct | 0.03 | 0.32 | 1.26 |
| 10485760 | 1024x10240 | torch.float32 | direct | 0.04 | 0.29 | 2.29 |
| 20971520 | 1024x20480 | torch.float16 | direct | 0.06 | 0.33 | 1.33 |
| 20971520 | 1024x20480 | torch.bfloat16 | direct | 0.06 | 0.34 | 1.35 |
| 20971520 | 1024x20480 | torch.float32 | direct | 0.07 | 0.30 | 2.41 |

### relu_explicit_parallel

| n_total | shape_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4194304 | 1024x4096 | torch.float16 | explicit_parallel | 0.01 | 0.61 | 2.43 |
| 4194304 | 1024x4096 | torch.bfloat16 | explicit_parallel | 0.01 | 0.61 | 2.43 |
| 4194304 | 1024x4096 | torch.float32 | explicit_parallel | 0.01 | 0.38 | 3.06 |
| 10485760 | 1024x10240 | torch.float16 | explicit_parallel | 0.01 | 0.75 | 3.01 |
| 10485760 | 1024x10240 | torch.bfloat16 | explicit_parallel | 0.01 | 0.75 | 3.01 |
| 10485760 | 1024x10240 | torch.float32 | explicit_parallel | 0.02 | 0.46 | 3.69 |
| 20971520 | 1024x20480 | torch.float16 | explicit_parallel | 0.03 | 0.83 | 3.31 |
| 20971520 | 1024x20480 | torch.bfloat16 | explicit_parallel | 0.03 | 0.82 | 3.30 |
| 20971520 | 1024x20480 | torch.float32 | explicit_parallel | 0.04 | 0.50 | 3.99 |

### relu_register_copy

| n_total | shape_label | dtype | strategy | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 4194304 | 1024x4096 | torch.float16 | register_copy | 0.01 | 0.67 | 2.66 |
| 4194304 | 1024x4096 | torch.bfloat16 | register_copy | 0.01 | 0.66 | 2.64 |
| 4194304 | 1024x4096 | torch.float32 | register_copy | 0.01 | 0.39 | 3.15 |
| 10485760 | 1024x10240 | torch.float16 | register_copy | 0.01 | 0.84 | 3.36 |
| 10485760 | 1024x10240 | torch.bfloat16 | register_copy | 0.01 | 0.84 | 3.36 |
| 10485760 | 1024x10240 | torch.float32 | register_copy | 0.02 | 0.47 | 3.79 |
| 20971520 | 1024x20480 | torch.float16 | register_copy | 0.02 | 0.95 | 3.79 |
| 20971520 | 1024x20480 | torch.bfloat16 | register_copy | 0.02 | 0.95 | 3.79 |
| 20971520 | 1024x20480 | torch.float32 | register_copy | 0.04 | 0.51 | 4.11 |

## L1NormOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | l1 | 0.01 | 0.52 | 1.05 |
| 1024 | 4096 | torch.bfloat16 | l1 | 0.01 | 0.53 | 1.06 |
| 1024 | 4096 | torch.float32 | l1 | 0.01 | 0.34 | 1.37 |
| 4096 | 4096 | torch.float16 | l1 | 0.02 | 0.70 | 1.40 |

### baseline

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | l1 | 0.02 | 0.25 | 0.50 |
| 1024 | 4096 | torch.bfloat16 | l1 | 0.02 | 0.25 | 0.49 |
| 1024 | 4096 | torch.float32 | l1 | 0.02 | 0.22 | 0.88 |
| 4096 | 4096 | torch.float16 | l1 | 0.02 | 0.76 | 1.51 |

## L2NormOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | l2 | 0.01 | 0.52 | 1.04 |
| 1024 | 4096 | torch.bfloat16 | l2 | 0.01 | 0.52 | 1.05 |
| 1024 | 4096 | torch.float32 | l2 | 0.01 | 0.34 | 1.35 |
| 4096 | 4096 | torch.float16 | l2 | 0.02 | 0.68 | 1.37 |

### baseline

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | l2 | 0.02 | 0.25 | 0.49 |
| 1024 | 4096 | torch.bfloat16 | l2 | 0.02 | 0.25 | 0.50 |
| 1024 | 4096 | torch.float32 | l2 | 0.02 | 0.22 | 0.87 |
| 4096 | 4096 | torch.float16 | l2 | 0.02 | 0.76 | 1.53 |

## InfNormOp

### tileops

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | inf | 0.03 | 0.15 | 0.30 |
| 1024 | 4096 | torch.bfloat16 | inf | 0.03 | 0.15 | 0.29 |
| 1024 | 4096 | torch.float32 | inf | 0.03 | 0.12 | 0.50 |
| 4096 | 4096 | torch.float16 | inf | 0.06 | 0.30 | 0.59 |

### baseline

| m | n | dtype | op_kind | latency_ms | tflops | bandwidth_tbs |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4096 | torch.float16 | inf | 0.02 | 0.24 | 0.49 |
| 1024 | 4096 | torch.bfloat16 | inf | 0.02 | 0.24 | 0.49 |
| 1024 | 4096 | torch.float32 | inf | 0.02 | 0.22 | 0.86 |
| 4096 | 4096 | torch.float16 | inf | 0.02 | 0.75 | 1.49 |

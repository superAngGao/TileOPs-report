........................................................................ [  7%]
........................................................................ [ 14%]
........................................................................ [ 21%]
........................................................................ [ 28%]
........................................................................ [ 35%]
........................................................................ [ 43%]
.......................................xxx.............................. [ 50%]
........................................................................ [ 57%]
.............FFFsss..................................................... [ 64%]
........................................................................ [ 71%]
........................................................................ [ 78%]
..FFF................................................................... [ 86%]
........................................................................ [ 93%]
....................................................................     [100%]Benchmark report saved to profile_run.log

=================================== FAILURES ===================================
_________________ test_gqa_decode_paged_bench[serving-8b-p256] _________________

batch = 32, heads = 32, heads_kv = 8, seqlen_kv = 4096, dim = 128
page_size = 256, dtype = torch.float16, tune = True

    @pytest.mark.parametrize(
        "batch, heads, heads_kv, seqlen_kv, dim, page_size, dtype, tune",
        _GQA_DECODE_PAGED_BENCH_PARAMS,
    )
    def test_gqa_decode_paged_bench(batch: int, heads: int, heads_kv: int, seqlen_kv: int, dim: int,
                                    page_size: int, dtype: torch.dtype, tune: bool) -> None:
        test = _GqaDecodePagedTestBaseline(batch, heads, heads_kv, seqlen_kv, dim, page_size, dtype)
        bm = GqaDecodePagedBenchmark(test)
        inputs = test.gen_inputs()
        q, k, v, real_seqlen_kv, block_table = inputs
    
        op = GroupQueryAttentionDecodePagedWithKVCacheOp(
            batch, heads, heads_kv, seqlen_kv, dim, page_size, dtype, tune=tune)
        result = bm.profile(op, *inputs)
        BenchmarkReport.record(op, locals(), result, tag="tileops")
    
        fa3_fn = _fa3_gqa_decode_paged(test, k, v)
        if fa3_fn is not None:
>           result_fa3 = bm.profile(fa3_fn, *inputs)
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^

benchmarks/ops/bench_gqa_decode_paged.py:202: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
benchmarks/benchmark.py:239: in profile
    latency = bench_kernel(functor, args=inputs)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
benchmarks/benchmark.py:125: in bench_kernel
    _run(i % n_repeat)
benchmarks/benchmark.py:112: in _run
    return fn(*arg_pool[i % _N_CLONES])
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

q = tensor([[[-0.8574, -1.0078, -1.0186,  ...,  1.2695,  1.6689,  0.9092],
         [ 0.5884,  0.5854, -1.3262,  ...,  0.1...,
         [-0.5088, -2.0293, -0.4111,  ...,  0.7905,  0.2430,  1.2207]]],
       device='cuda:0', dtype=torch.float16)
k = tensor([[[-0.0262, -1.0488, -0.5830,  ...,  0.2300,  0.1849,  0.6099],
         [ 0.3738, -0.2444,  0.0745,  ...,  0.8...,
         [ 1.4932,  0.2634, -1.2920,  ...,  0.4453, -0.6030, -1.2969]]],
       device='cuda:0', dtype=torch.float16)
v = tensor([[[-5.3125e-01, -7.9895e-02,  4.7852e-01,  ..., -1.0713e+00,
          -1.2217e+00, -8.2812e-01],
         [-1.....1387e+00,  1.2299e-01,  ..., -1.8750e+00,
          -3.4619e-01, -1.1127e-01]]], device='cuda:0', dtype=torch.float16)
real_seqlen_kv = tensor([1280, 2304, 3072, 3840, 1280, 1536, 1024, 1536, 1280, 1024,  256,  768,
         768,  768, 2816, 2816,  512, ..., 3328, 2304, 3072,
        3584, 2048, 3840, 2048, 2560,  768, 2304, 3584], device='cuda:0',
       dtype=torch.int32)
block_table = tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
        [ 0,  1,  2,  3,  4,  5,  6,  7,  8,...,
        [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15]],
       device='cuda:0', dtype=torch.int32)

    def baseline_fn(q, k, v, real_seqlen_kv, block_table):
        # Q is (batch, heads, dim) — add seq dim for flash_attn
>       out = flash_attn_with_kvcache(
            q.unsqueeze(1), k_paged, v_paged,
            cache_seqlens=real_seqlen_kv.int(),
            block_table=block_table.int())
E       TypeError: flash_attn_with_kvcache() got an unexpected keyword argument 'block_table'

benchmarks/ops/bench_gqa_decode_paged.py:85: TypeError
----------------------------- Captured stdout call -----------------------------
Start autotuning gqa_decode_paged_kernel...
Best config: {'block_H': 64, 'block_N': 64, 'num_split': 2, 'num_stages': 2, 'threads': 128}
gqa_decode_paged_kernel initialized with config: {'block_H': 64, 'block_N': 64, 'num_split': 2, 'num_stages': 2, 'threads': 128}
2026-04-05 19:31:39  [TileLang:tilelang.language.eager.builder:WARNING]: Immutable value `k_global` is re-bound. If you want to modify its value, please use T.alloc_var to make it a variable!
Stack (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/usr/local/lib/python3.11/dist-packages/pytest/__main__.py", line 9, in <module>
    raise SystemExit(pytest.console_main())
  File "/usr/local/lib/python3.11/dist-packages/_pytest/config/__init__.py", line 223, in console_main
    code = main()
  File "/usr/local/lib/python3.11/dist-packages/_pytest/config/__init__.py", line 199, in main
    ret: ExitCode | int = config.hook.pytest_cmdline_main(config=config)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_hooks.py", line 512, in __call__
    return self._hookexec(self.name, self._hookimpls.copy(), kwargs, firstresult)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_manager.py", line 120, in _hookexec
    return self._inner_hookexec(hook_name, methods, kwargs, firstresult)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_callers.py", line 121, in _multicall
    res = hook_impl.function(*args)
  File "/usr/local/lib/python3.11/dist-packages/_pytest/main.py", line 365, in pytest_cmdline_main
    return wrap_session(config, _main)
  File "/usr/local/lib/python3.11/dist-packages/_pytest/main.py", line 318, in wrap_session
    session.exitstatus = doit(config, session) or 0
  File "/usr/local/lib/python3.11/dist-packages/_pytest/main.py", line 372, in _main
    config.hook.pytest_runtestloop(session=session)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_hooks.py", line 512, in __call__
    return self._hookexec(self.name, self._hookimpls.copy(), kwargs, firstresult)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_manager.py", line 120, in _hookexec
    return self._inner_hookexec(hook_name, methods, kwargs, firstresult)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_callers.py", line 121, in _multicall
    res = hook_impl.function(*args)
  File "/usr/local/lib/python3.11/dist-packages/_pytest/main.py", line 396, in pytest_runtestloop
    item.config.hook.pytest_runtest_protocol(item=item, nextitem=nextitem)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_hooks.py", line 512, in __call__
    return self._hookexec(self.name, self._hookimpls.copy(), kwargs, firstresult)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_manager.py", line 120, in _hookexec
    return self._inner_hookexec(hook_name, methods, kwargs, firstresult)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_callers.py", line 121, in _multicall
    res = hook_impl.function(*args)
  File "/usr/local/lib/python3.11/dist-packages/_pytest/runner.py", line 118, in pytest_runtest_protocol
    runtestprotocol(item, nextitem=nextitem)
  File "/usr/local/lib/python3.11/dist-packages/_pytest/runner.py", line 137, in runtestprotocol
    reports.append(call_and_report(item, "call", log))
  File "/usr/local/lib/python3.11/dist-packages/_pytest/runner.py", line 244, in call_and_report
    call = CallInfo.from_call(
  File "/usr/local/lib/python3.11/dist-packages/_pytest/runner.py", line 353, in from_call
    result: TResult | None = func()
  File "/usr/local/lib/python3.11/dist-packages/_pytest/runner.py", line 245, in <lambda>
    lambda: runtest_hook(item=item, **kwds),
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_hooks.py", line 512, in __call__
    return self._hookexec(self.name, self._hookimpls.copy(), kwargs, firstresult)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_manager.py", line 120, in _hookexec
    return self._inner_hookexec(hook_name, methods, kwargs, firstresult)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_callers.py", line 121, in _multicall
    res = hook_impl.function(*args)
  File "/usr/local/lib/python3.11/dist-packages/_pytest/runner.py", line 179, in pytest_runtest_call
    item.runtest()
  File "/usr/local/lib/python3.11/dist-packages/_pytest/python.py", line 1720, in runtest
    self.ihook.pytest_pyfunc_call(pyfuncitem=self)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_hooks.py", line 512, in __call__
    return self._hookexec(self.name, self._hookimpls.copy(), kwargs, firstresult)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_manager.py", line 120, in _hookexec
    return self._inner_hookexec(hook_name, methods, kwargs, firstresult)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_callers.py", line 121, in _multicall
    res = hook_impl.function(*args)
  File "/usr/local/lib/python3.11/dist-packages/_pytest/python.py", line 166, in pytest_pyfunc_call
    result = testfunction(**testargs)
  File "/workspace/benchmarks/ops/bench_gqa_decode_paged.py", line 197, in test_gqa_decode_paged_bench
    result = bm.profile(op, *inputs)
  File "/workspace/benchmarks/benchmark.py", line 239, in profile
    latency = bench_kernel(functor, args=inputs)
  File "/workspace/benchmarks/benchmark.py", line 125, in bench_kernel
    _run(i % n_repeat)
  File "/workspace/benchmarks/benchmark.py", line 112, in _run
    return fn(*arg_pool[i % _N_CLONES])
  File "/workspace/tileops/ops/op.py", line 79, in __call__
    return self.forward(*args, **kwargs)
  File "/workspace/tileops/ops/gqa_decode_paged.py", line 46, in forward
    return self.kernel(q, k, v, real_seqlen_kv, block_table)
  File "/workspace/tileops/kernels/kernel.py", line 64, in __call__
    return self.forward(*args, **kwargs)
  File "/workspace/tileops/kernels/flash_decode/gqa_decode_paged.py", line 494, in forward
    return _gqa_decode_paged_split_op(
  File "/usr/local/lib/python3.11/dist-packages/torch/_library/custom_ops.py", line 698, in __call__
    return self._opoverload(*args, **kwargs)
  File "/usr/local/lib/python3.11/dist-packages/torch/_ops.py", line 819, in __call__
    return self._op(*args, **kwargs)
  File "/usr/local/lib/python3.11/dist-packages/torch/_library/autograd.py", line 112, in autograd_impl
    result = forward_no_grad(*args, Metadata(keyset, keyword_only_args))
  File "/usr/local/lib/python3.11/dist-packages/torch/_library/autograd.py", line 41, in forward_no_grad
    result = op.redispatch(keyset & _C._after_autograd_keyset, *args, **kwargs)
  File "/usr/local/lib/python3.11/dist-packages/torch/_ops.py", line 826, in redispatch
    return self._handle.redispatch_boxed(keyset, *args, **kwargs)  # type: ignore[return-value]
  File "/usr/local/lib/python3.11/dist-packages/torch/_library/custom_ops.py", line 347, in backend_impl
    result = self._backend_fns[device_type](*args, **kwargs)
  File "/usr/local/lib/python3.11/dist-packages/torch/_compile.py", line 54, in inner
    return disable_fn(*args, **kwargs)
  File "/usr/local/lib/python3.11/dist-packages/torch/_dynamo/eval_frame.py", line 1181, in _fn
    return fn(*args, **kwargs)
  File "/usr/local/lib/python3.11/dist-packages/torch/_library/custom_ops.py", line 382, in wrapped_fn
    return fn(*args, **kwargs)
  File "/workspace/tileops/kernels/flash_decode/gqa_decode_paged.py", line 340, in _gqa_decode_paged_split_op
    return _gqa_decode_split_paged_kernel(batch, heads, groups, seqlen_kv, dim, page_size,
  File "/usr/local/lib/python3.11/dist-packages/tilelang/jit/__init__.py", line 440, in __call__
    kernel = self.compile(*args, **kwargs)
  File "/usr/local/lib/python3.11/dist-packages/tilelang/jit/__init__.py", line 374, in compile
    prim_func = self.get_tir(*args, **kwargs)
  File "/usr/local/lib/python3.11/dist-packages/tilelang/jit/__init__.py", line 298, in get_tir
    tir = self.func(*args, **kwargs)
  File "/usr/local/lib/python3.11/dist-packages/tilelang/language/eager/builder.py", line 1142, in __call__
    return self.get_tir(*args, **kwargs)
  File "/usr/local/lib/python3.11/dist-packages/tilelang/language/eager/builder.py", line 1138, in get_tir
    self.p1_cache[p1_key] = self._build_tir_template(**kwargs)
  File "/usr/local/lib/python3.11/dist-packages/tilelang/language/eager/builder.py", line 1106, in _build_tir_template
    return TirTemplate.from_lazy_style(self.orig_func.__name__, self.orig_func(*args, **kwargs))
  File "/workspace/tileops/kernels/flash_decode/gqa_decode_paged.py", line 287, in _func
    @T.prim_func
  File "/usr/local/lib/python3.11/dist-packages/tilelang/language/eager/builder.py", line 1224, in prim_func
    return impl(func) if func is not None else impl
  File "/usr/local/lib/python3.11/dist-packages/tilelang/language/eager/builder.py", line 1214, in impl
    ir_gen.gen(builder)(**annot)
  File "/workspace/tileops/kernels/flash_decode/gqa_decode_paged.py", line 299, in gqa_decode_split
    _gqa_decode_split(Q, K, V, real_seqlen_kv, block_table, glse, Output_partial,
  File "/usr/local/lib/python3.11/dist-packages/tilelang/language/eager/builder.py", line 775, in __call__
    res = self.ir_gen.gen(builder)(*args, **kwargs)
  File "/workspace/tileops/kernels/flash_decode/gqa_decode_paged.py", line 212, in _gqa_decode_split
    k_global += offset
________________ test_gqa_decode_paged_bench[serving-70b-p256] _________________

batch = 8, heads = 64, heads_kv = 8, seqlen_kv = 4096, dim = 128
page_size = 256, dtype = torch.float16, tune = True

    @pytest.mark.parametrize(
        "batch, heads, heads_kv, seqlen_kv, dim, page_size, dtype, tune",
        _GQA_DECODE_PAGED_BENCH_PARAMS,
    )
    def test_gqa_decode_paged_bench(batch: int, heads: int, heads_kv: int, seqlen_kv: int, dim: int,
                                    page_size: int, dtype: torch.dtype, tune: bool) -> None:
        test = _GqaDecodePagedTestBaseline(batch, heads, heads_kv, seqlen_kv, dim, page_size, dtype)
        bm = GqaDecodePagedBenchmark(test)
        inputs = test.gen_inputs()
        q, k, v, real_seqlen_kv, block_table = inputs
    
        op = GroupQueryAttentionDecodePagedWithKVCacheOp(
            batch, heads, heads_kv, seqlen_kv, dim, page_size, dtype, tune=tune)
        result = bm.profile(op, *inputs)
        BenchmarkReport.record(op, locals(), result, tag="tileops")
    
        fa3_fn = _fa3_gqa_decode_paged(test, k, v)
        if fa3_fn is not None:
>           result_fa3 = bm.profile(fa3_fn, *inputs)
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^

benchmarks/ops/bench_gqa_decode_paged.py:202: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
benchmarks/benchmark.py:239: in profile
    latency = bench_kernel(functor, args=inputs)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
benchmarks/benchmark.py:125: in bench_kernel
    _run(i % n_repeat)
benchmarks/benchmark.py:112: in _run
    return fn(*arg_pool[i % _N_CLONES])
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

q = tensor([[[-8.5742e-01, -1.0078e+00, -1.0186e+00,  ...,  1.2695e+00,
           1.6689e+00,  9.0918e-01],
         [ 5.....9426e-02,  2.9688e-01,  ...,  5.6854e-02,
          -1.0205e+00, -1.5391e+00]]], device='cuda:0', dtype=torch.float16)
k = tensor([[[-0.0262, -1.0488, -0.5830,  ...,  0.2300,  0.1849,  0.6099],
         [ 0.3738, -0.2444,  0.0745,  ...,  0.8...,
         [ 1.4932,  0.2634, -1.2920,  ...,  0.4453, -0.6030, -1.2969]]],
       device='cuda:0', dtype=torch.float16)
v = tensor([[[-5.3125e-01, -7.9895e-02,  4.7852e-01,  ..., -1.0713e+00,
          -1.2217e+00, -8.2812e-01],
         [-1.....1387e+00,  1.2299e-01,  ..., -1.8750e+00,
          -3.4619e-01, -1.1127e-01]]], device='cuda:0', dtype=torch.float16)
real_seqlen_kv = tensor([1280, 2304, 3072, 3840, 1280, 1536, 1024, 1536], device='cuda:0',
       dtype=torch.int32)
block_table = tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
        [ 0,  1,  2,  3,  4,  5,  6,  7,  8,...,
        [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15]],
       device='cuda:0', dtype=torch.int32)

    def baseline_fn(q, k, v, real_seqlen_kv, block_table):
        # Q is (batch, heads, dim) — add seq dim for flash_attn
>       out = flash_attn_with_kvcache(
            q.unsqueeze(1), k_paged, v_paged,
            cache_seqlens=real_seqlen_kv.int(),
            block_table=block_table.int())
E       TypeError: flash_attn_with_kvcache() got an unexpected keyword argument 'block_table'

benchmarks/ops/bench_gqa_decode_paged.py:85: TypeError
----------------------------- Captured stdout call -----------------------------
Start autotuning gqa_decode_paged_kernel...
Best config: {'block_H': 64, 'block_N': 64, 'num_split': 4, 'num_stages': 2, 'threads': 128}
gqa_decode_paged_kernel initialized with config: {'block_H': 64, 'block_N': 64, 'num_split': 4, 'num_stages': 2, 'threads': 128}
2026-04-05 19:31:39  [TileLang:tilelang.language.eager.builder:WARNING]: Immutable value `k_global` is re-bound. If you want to modify its value, please use T.alloc_var to make it a variable!
Stack (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/usr/local/lib/python3.11/dist-packages/pytest/__main__.py", line 9, in <module>
    raise SystemExit(pytest.console_main())
  File "/usr/local/lib/python3.11/dist-packages/_pytest/config/__init__.py", line 223, in console_main
    code = main()
  File "/usr/local/lib/python3.11/dist-packages/_pytest/config/__init__.py", line 199, in main
    ret: ExitCode | int = config.hook.pytest_cmdline_main(config=config)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_hooks.py", line 512, in __call__
    return self._hookexec(self.name, self._hookimpls.copy(), kwargs, firstresult)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_manager.py", line 120, in _hookexec
    return self._inner_hookexec(hook_name, methods, kwargs, firstresult)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_callers.py", line 121, in _multicall
    res = hook_impl.function(*args)
  File "/usr/local/lib/python3.11/dist-packages/_pytest/main.py", line 365, in pytest_cmdline_main
    return wrap_session(config, _main)
  File "/usr/local/lib/python3.11/dist-packages/_pytest/main.py", line 318, in wrap_session
    session.exitstatus = doit(config, session) or 0
  File "/usr/local/lib/python3.11/dist-packages/_pytest/main.py", line 372, in _main
    config.hook.pytest_runtestloop(session=session)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_hooks.py", line 512, in __call__
    return self._hookexec(self.name, self._hookimpls.copy(), kwargs, firstresult)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_manager.py", line 120, in _hookexec
    return self._inner_hookexec(hook_name, methods, kwargs, firstresult)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_callers.py", line 121, in _multicall
    res = hook_impl.function(*args)
  File "/usr/local/lib/python3.11/dist-packages/_pytest/main.py", line 396, in pytest_runtestloop
    item.config.hook.pytest_runtest_protocol(item=item, nextitem=nextitem)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_hooks.py", line 512, in __call__
    return self._hookexec(self.name, self._hookimpls.copy(), kwargs, firstresult)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_manager.py", line 120, in _hookexec
    return self._inner_hookexec(hook_name, methods, kwargs, firstresult)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_callers.py", line 121, in _multicall
    res = hook_impl.function(*args)
  File "/usr/local/lib/python3.11/dist-packages/_pytest/runner.py", line 118, in pytest_runtest_protocol
    runtestprotocol(item, nextitem=nextitem)
  File "/usr/local/lib/python3.11/dist-packages/_pytest/runner.py", line 137, in runtestprotocol
    reports.append(call_and_report(item, "call", log))
  File "/usr/local/lib/python3.11/dist-packages/_pytest/runner.py", line 244, in call_and_report
    call = CallInfo.from_call(
  File "/usr/local/lib/python3.11/dist-packages/_pytest/runner.py", line 353, in from_call
    result: TResult | None = func()
  File "/usr/local/lib/python3.11/dist-packages/_pytest/runner.py", line 245, in <lambda>
    lambda: runtest_hook(item=item, **kwds),
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_hooks.py", line 512, in __call__
    return self._hookexec(self.name, self._hookimpls.copy(), kwargs, firstresult)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_manager.py", line 120, in _hookexec
    return self._inner_hookexec(hook_name, methods, kwargs, firstresult)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_callers.py", line 121, in _multicall
    res = hook_impl.function(*args)
  File "/usr/local/lib/python3.11/dist-packages/_pytest/runner.py", line 179, in pytest_runtest_call
    item.runtest()
  File "/usr/local/lib/python3.11/dist-packages/_pytest/python.py", line 1720, in runtest
    self.ihook.pytest_pyfunc_call(pyfuncitem=self)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_hooks.py", line 512, in __call__
    return self._hookexec(self.name, self._hookimpls.copy(), kwargs, firstresult)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_manager.py", line 120, in _hookexec
    return self._inner_hookexec(hook_name, methods, kwargs, firstresult)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_callers.py", line 121, in _multicall
    res = hook_impl.function(*args)
  File "/usr/local/lib/python3.11/dist-packages/_pytest/python.py", line 166, in pytest_pyfunc_call
    result = testfunction(**testargs)
  File "/workspace/benchmarks/ops/bench_gqa_decode_paged.py", line 197, in test_gqa_decode_paged_bench
    result = bm.profile(op, *inputs)
  File "/workspace/benchmarks/benchmark.py", line 239, in profile
    latency = bench_kernel(functor, args=inputs)
  File "/workspace/benchmarks/benchmark.py", line 125, in bench_kernel
    _run(i % n_repeat)
  File "/workspace/benchmarks/benchmark.py", line 112, in _run
    return fn(*arg_pool[i % _N_CLONES])
  File "/workspace/tileops/ops/op.py", line 79, in __call__
    return self.forward(*args, **kwargs)
  File "/workspace/tileops/ops/gqa_decode_paged.py", line 46, in forward
    return self.kernel(q, k, v, real_seqlen_kv, block_table)
  File "/workspace/tileops/kernels/kernel.py", line 64, in __call__
    return self.forward(*args, **kwargs)
  File "/workspace/tileops/kernels/flash_decode/gqa_decode_paged.py", line 494, in forward
    return _gqa_decode_paged_split_op(
  File "/usr/local/lib/python3.11/dist-packages/torch/_library/custom_ops.py", line 698, in __call__
    return self._opoverload(*args, **kwargs)
  File "/usr/local/lib/python3.11/dist-packages/torch/_ops.py", line 819, in __call__
    return self._op(*args, **kwargs)
  File "/usr/local/lib/python3.11/dist-packages/torch/_library/autograd.py", line 112, in autograd_impl
    result = forward_no_grad(*args, Metadata(keyset, keyword_only_args))
  File "/usr/local/lib/python3.11/dist-packages/torch/_library/autograd.py", line 41, in forward_no_grad
    result = op.redispatch(keyset & _C._after_autograd_keyset, *args, **kwargs)
  File "/usr/local/lib/python3.11/dist-packages/torch/_ops.py", line 826, in redispatch
    return self._handle.redispatch_boxed(keyset, *args, **kwargs)  # type: ignore[return-value]
  File "/usr/local/lib/python3.11/dist-packages/torch/_library/custom_ops.py", line 347, in backend_impl
    result = self._backend_fns[device_type](*args, **kwargs)
  File "/usr/local/lib/python3.11/dist-packages/torch/_compile.py", line 54, in inner
    return disable_fn(*args, **kwargs)
  File "/usr/local/lib/python3.11/dist-packages/torch/_dynamo/eval_frame.py", line 1181, in _fn
    return fn(*args, **kwargs)
  File "/usr/local/lib/python3.11/dist-packages/torch/_library/custom_ops.py", line 382, in wrapped_fn
    return fn(*args, **kwargs)
  File "/workspace/tileops/kernels/flash_decode/gqa_decode_paged.py", line 340, in _gqa_decode_paged_split_op
    return _gqa_decode_split_paged_kernel(batch, heads, groups, seqlen_kv, dim, page_size,
  File "/usr/local/lib/python3.11/dist-packages/tilelang/jit/__init__.py", line 440, in __call__
    kernel = self.compile(*args, **kwargs)
  File "/usr/local/lib/python3.11/dist-packages/tilelang/jit/__init__.py", line 374, in compile
    prim_func = self.get_tir(*args, **kwargs)
  File "/usr/local/lib/python3.11/dist-packages/tilelang/jit/__init__.py", line 298, in get_tir
    tir = self.func(*args, **kwargs)
  File "/usr/local/lib/python3.11/dist-packages/tilelang/language/eager/builder.py", line 1142, in __call__
    return self.get_tir(*args, **kwargs)
  File "/usr/local/lib/python3.11/dist-packages/tilelang/language/eager/builder.py", line 1138, in get_tir
    self.p1_cache[p1_key] = self._build_tir_template(**kwargs)
  File "/usr/local/lib/python3.11/dist-packages/tilelang/language/eager/builder.py", line 1106, in _build_tir_template
    return TirTemplate.from_lazy_style(self.orig_func.__name__, self.orig_func(*args, **kwargs))
  File "/workspace/tileops/kernels/flash_decode/gqa_decode_paged.py", line 287, in _func
    @T.prim_func
  File "/usr/local/lib/python3.11/dist-packages/tilelang/language/eager/builder.py", line 1224, in prim_func
    return impl(func) if func is not None else impl
  File "/usr/local/lib/python3.11/dist-packages/tilelang/language/eager/builder.py", line 1214, in impl
    ir_gen.gen(builder)(**annot)
  File "/workspace/tileops/kernels/flash_decode/gqa_decode_paged.py", line 299, in gqa_decode_split
    _gqa_decode_split(Q, K, V, real_seqlen_kv, block_table, glse, Output_partial,
  File "/usr/local/lib/python3.11/dist-packages/tilelang/language/eager/builder.py", line 775, in __call__
    res = self.ir_gen.gen(builder)(*args, **kwargs)
  File "/workspace/tileops/kernels/flash_decode/gqa_decode_paged.py", line 212, in _gqa_decode_split
    k_global += offset
________________ test_gqa_decode_paged_bench[serving-405b-p256] ________________

batch = 4, heads = 128, heads_kv = 8, seqlen_kv = 4096, dim = 128
page_size = 256, dtype = torch.float16, tune = True

    @pytest.mark.parametrize(
        "batch, heads, heads_kv, seqlen_kv, dim, page_size, dtype, tune",
        _GQA_DECODE_PAGED_BENCH_PARAMS,
    )
    def test_gqa_decode_paged_bench(batch: int, heads: int, heads_kv: int, seqlen_kv: int, dim: int,
                                    page_size: int, dtype: torch.dtype, tune: bool) -> None:
        test = _GqaDecodePagedTestBaseline(batch, heads, heads_kv, seqlen_kv, dim, page_size, dtype)
        bm = GqaDecodePagedBenchmark(test)
        inputs = test.gen_inputs()
        q, k, v, real_seqlen_kv, block_table = inputs
    
        op = GroupQueryAttentionDecodePagedWithKVCacheOp(
            batch, heads, heads_kv, seqlen_kv, dim, page_size, dtype, tune=tune)
        result = bm.profile(op, *inputs)
        BenchmarkReport.record(op, locals(), result, tag="tileops")
    
        fa3_fn = _fa3_gqa_decode_paged(test, k, v)
        if fa3_fn is not None:
>           result_fa3 = bm.profile(fa3_fn, *inputs)
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^

benchmarks/ops/bench_gqa_decode_paged.py:202: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
benchmarks/benchmark.py:239: in profile
    latency = bench_kernel(functor, args=inputs)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
benchmarks/benchmark.py:125: in bench_kernel
    _run(i % n_repeat)
benchmarks/benchmark.py:112: in _run
    return fn(*arg_pool[i % _N_CLONES])
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

q = tensor([[[-8.5742e-01, -1.0078e+00, -1.0186e+00,  ...,  1.2695e+00,
           1.6689e+00,  9.0918e-01],
         [ 5.....9426e-02,  2.9688e-01,  ...,  5.6854e-02,
          -1.0205e+00, -1.5391e+00]]], device='cuda:0', dtype=torch.float16)
k = tensor([[[-0.0262, -1.0488, -0.5830,  ...,  0.2300,  0.1849,  0.6099],
         [ 0.3738, -0.2444,  0.0745,  ...,  0.8...,
         [ 1.4932,  0.2634, -1.2920,  ...,  0.4453, -0.6030, -1.2969]]],
       device='cuda:0', dtype=torch.float16)
v = tensor([[[-5.3125e-01, -7.9895e-02,  4.7852e-01,  ..., -1.0713e+00,
          -1.2217e+00, -8.2812e-01],
         [-1.....1387e+00,  1.2299e-01,  ..., -1.8750e+00,
          -3.4619e-01, -1.1127e-01]]], device='cuda:0', dtype=torch.float16)
real_seqlen_kv = tensor([1280, 2304, 3072, 3840], device='cuda:0', dtype=torch.int32)
block_table = tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
        [ 0,  1,  2,  3,  4,  5,  6,  7,  8,...,
        [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15]],
       device='cuda:0', dtype=torch.int32)

    def baseline_fn(q, k, v, real_seqlen_kv, block_table):
        # Q is (batch, heads, dim) — add seq dim for flash_attn
>       out = flash_attn_with_kvcache(
            q.unsqueeze(1), k_paged, v_paged,
            cache_seqlens=real_seqlen_kv.int(),
            block_table=block_table.int())
E       TypeError: flash_attn_with_kvcache() got an unexpected keyword argument 'block_table'

benchmarks/ops/bench_gqa_decode_paged.py:85: TypeError
----------------------------- Captured stdout call -----------------------------
Start autotuning gqa_decode_paged_kernel...
Best config: {'block_H': 64, 'block_N': 64, 'num_split': 8, 'num_stages': 2, 'threads': 128}
gqa_decode_paged_kernel initialized with config: {'block_H': 64, 'block_N': 64, 'num_split': 8, 'num_stages': 2, 'threads': 128}
2026-04-05 19:31:40  [TileLang:tilelang.language.eager.builder:WARNING]: Immutable value `k_global` is re-bound. If you want to modify its value, please use T.alloc_var to make it a variable!
Stack (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/usr/local/lib/python3.11/dist-packages/pytest/__main__.py", line 9, in <module>
    raise SystemExit(pytest.console_main())
  File "/usr/local/lib/python3.11/dist-packages/_pytest/config/__init__.py", line 223, in console_main
    code = main()
  File "/usr/local/lib/python3.11/dist-packages/_pytest/config/__init__.py", line 199, in main
    ret: ExitCode | int = config.hook.pytest_cmdline_main(config=config)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_hooks.py", line 512, in __call__
    return self._hookexec(self.name, self._hookimpls.copy(), kwargs, firstresult)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_manager.py", line 120, in _hookexec
    return self._inner_hookexec(hook_name, methods, kwargs, firstresult)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_callers.py", line 121, in _multicall
    res = hook_impl.function(*args)
  File "/usr/local/lib/python3.11/dist-packages/_pytest/main.py", line 365, in pytest_cmdline_main
    return wrap_session(config, _main)
  File "/usr/local/lib/python3.11/dist-packages/_pytest/main.py", line 318, in wrap_session
    session.exitstatus = doit(config, session) or 0
  File "/usr/local/lib/python3.11/dist-packages/_pytest/main.py", line 372, in _main
    config.hook.pytest_runtestloop(session=session)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_hooks.py", line 512, in __call__
    return self._hookexec(self.name, self._hookimpls.copy(), kwargs, firstresult)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_manager.py", line 120, in _hookexec
    return self._inner_hookexec(hook_name, methods, kwargs, firstresult)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_callers.py", line 121, in _multicall
    res = hook_impl.function(*args)
  File "/usr/local/lib/python3.11/dist-packages/_pytest/main.py", line 396, in pytest_runtestloop
    item.config.hook.pytest_runtest_protocol(item=item, nextitem=nextitem)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_hooks.py", line 512, in __call__
    return self._hookexec(self.name, self._hookimpls.copy(), kwargs, firstresult)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_manager.py", line 120, in _hookexec
    return self._inner_hookexec(hook_name, methods, kwargs, firstresult)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_callers.py", line 121, in _multicall
    res = hook_impl.function(*args)
  File "/usr/local/lib/python3.11/dist-packages/_pytest/runner.py", line 118, in pytest_runtest_protocol
    runtestprotocol(item, nextitem=nextitem)
  File "/usr/local/lib/python3.11/dist-packages/_pytest/runner.py", line 137, in runtestprotocol
    reports.append(call_and_report(item, "call", log))
  File "/usr/local/lib/python3.11/dist-packages/_pytest/runner.py", line 244, in call_and_report
    call = CallInfo.from_call(
  File "/usr/local/lib/python3.11/dist-packages/_pytest/runner.py", line 353, in from_call
    result: TResult | None = func()
  File "/usr/local/lib/python3.11/dist-packages/_pytest/runner.py", line 245, in <lambda>
    lambda: runtest_hook(item=item, **kwds),
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_hooks.py", line 512, in __call__
    return self._hookexec(self.name, self._hookimpls.copy(), kwargs, firstresult)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_manager.py", line 120, in _hookexec
    return self._inner_hookexec(hook_name, methods, kwargs, firstresult)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_callers.py", line 121, in _multicall
    res = hook_impl.function(*args)
  File "/usr/local/lib/python3.11/dist-packages/_pytest/runner.py", line 179, in pytest_runtest_call
    item.runtest()
  File "/usr/local/lib/python3.11/dist-packages/_pytest/python.py", line 1720, in runtest
    self.ihook.pytest_pyfunc_call(pyfuncitem=self)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_hooks.py", line 512, in __call__
    return self._hookexec(self.name, self._hookimpls.copy(), kwargs, firstresult)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_manager.py", line 120, in _hookexec
    return self._inner_hookexec(hook_name, methods, kwargs, firstresult)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_callers.py", line 121, in _multicall
    res = hook_impl.function(*args)
  File "/usr/local/lib/python3.11/dist-packages/_pytest/python.py", line 166, in pytest_pyfunc_call
    result = testfunction(**testargs)
  File "/workspace/benchmarks/ops/bench_gqa_decode_paged.py", line 197, in test_gqa_decode_paged_bench
    result = bm.profile(op, *inputs)
  File "/workspace/benchmarks/benchmark.py", line 239, in profile
    latency = bench_kernel(functor, args=inputs)
  File "/workspace/benchmarks/benchmark.py", line 125, in bench_kernel
    _run(i % n_repeat)
  File "/workspace/benchmarks/benchmark.py", line 112, in _run
    return fn(*arg_pool[i % _N_CLONES])
  File "/workspace/tileops/ops/op.py", line 79, in __call__
    return self.forward(*args, **kwargs)
  File "/workspace/tileops/ops/gqa_decode_paged.py", line 46, in forward
    return self.kernel(q, k, v, real_seqlen_kv, block_table)
  File "/workspace/tileops/kernels/kernel.py", line 64, in __call__
    return self.forward(*args, **kwargs)
  File "/workspace/tileops/kernels/flash_decode/gqa_decode_paged.py", line 494, in forward
    return _gqa_decode_paged_split_op(
  File "/usr/local/lib/python3.11/dist-packages/torch/_library/custom_ops.py", line 698, in __call__
    return self._opoverload(*args, **kwargs)
  File "/usr/local/lib/python3.11/dist-packages/torch/_ops.py", line 819, in __call__
    return self._op(*args, **kwargs)
  File "/usr/local/lib/python3.11/dist-packages/torch/_library/autograd.py", line 112, in autograd_impl
    result = forward_no_grad(*args, Metadata(keyset, keyword_only_args))
  File "/usr/local/lib/python3.11/dist-packages/torch/_library/autograd.py", line 41, in forward_no_grad
    result = op.redispatch(keyset & _C._after_autograd_keyset, *args, **kwargs)
  File "/usr/local/lib/python3.11/dist-packages/torch/_ops.py", line 826, in redispatch
    return self._handle.redispatch_boxed(keyset, *args, **kwargs)  # type: ignore[return-value]
  File "/usr/local/lib/python3.11/dist-packages/torch/_library/custom_ops.py", line 347, in backend_impl
    result = self._backend_fns[device_type](*args, **kwargs)
  File "/usr/local/lib/python3.11/dist-packages/torch/_compile.py", line 54, in inner
    return disable_fn(*args, **kwargs)
  File "/usr/local/lib/python3.11/dist-packages/torch/_dynamo/eval_frame.py", line 1181, in _fn
    return fn(*args, **kwargs)
  File "/usr/local/lib/python3.11/dist-packages/torch/_library/custom_ops.py", line 382, in wrapped_fn
    return fn(*args, **kwargs)
  File "/workspace/tileops/kernels/flash_decode/gqa_decode_paged.py", line 340, in _gqa_decode_paged_split_op
    return _gqa_decode_split_paged_kernel(batch, heads, groups, seqlen_kv, dim, page_size,
  File "/usr/local/lib/python3.11/dist-packages/tilelang/jit/__init__.py", line 440, in __call__
    kernel = self.compile(*args, **kwargs)
  File "/usr/local/lib/python3.11/dist-packages/tilelang/jit/__init__.py", line 374, in compile
    prim_func = self.get_tir(*args, **kwargs)
  File "/usr/local/lib/python3.11/dist-packages/tilelang/jit/__init__.py", line 298, in get_tir
    tir = self.func(*args, **kwargs)
  File "/usr/local/lib/python3.11/dist-packages/tilelang/language/eager/builder.py", line 1142, in __call__
    return self.get_tir(*args, **kwargs)
  File "/usr/local/lib/python3.11/dist-packages/tilelang/language/eager/builder.py", line 1138, in get_tir
    self.p1_cache[p1_key] = self._build_tir_template(**kwargs)
  File "/usr/local/lib/python3.11/dist-packages/tilelang/language/eager/builder.py", line 1106, in _build_tir_template
    return TirTemplate.from_lazy_style(self.orig_func.__name__, self.orig_func(*args, **kwargs))
  File "/workspace/tileops/kernels/flash_decode/gqa_decode_paged.py", line 287, in _func
    @T.prim_func
  File "/usr/local/lib/python3.11/dist-packages/tilelang/language/eager/builder.py", line 1224, in prim_func
    return impl(func) if func is not None else impl
  File "/usr/local/lib/python3.11/dist-packages/tilelang/language/eager/builder.py", line 1214, in impl
    ir_gen.gen(builder)(**annot)
  File "/workspace/tileops/kernels/flash_decode/gqa_decode_paged.py", line 299, in gqa_decode_split
    _gqa_decode_split(Q, K, V, real_seqlen_kv, block_table, glse, Output_partial,
  File "/usr/local/lib/python3.11/dist-packages/tilelang/language/eager/builder.py", line 775, in __call__
    res = self.ir_gen.gen(builder)(*args, **kwargs)
  File "/workspace/tileops/kernels/flash_decode/gqa_decode_paged.py", line 212, in _gqa_decode_split
    k_global += offset
_________________ test_mha_decode_paged_bench[batch2-page256] __________________

batch = 2, heads = 8, seqlen_q = 1, seqlen_kv = 1024, dim = 64, page_size = 256
is_causal = False, dtype = torch.float16, tune = True

    @pytest.mark.parametrize(
        "batch, heads, seqlen_q, seqlen_kv, dim, page_size, is_causal, dtype, tune",
        _MHA_DECODE_PAGED_BENCH_PARAMS,
    )
    def test_mha_decode_paged_bench(batch: int, heads: int, seqlen_q: int, seqlen_kv: int, dim: int,
                                    page_size: int, is_causal: bool, dtype: torch.dtype,
                                    tune: bool) -> None:
        test = _MhaDecodePagedTestBaseline(batch, heads, seqlen_q, seqlen_kv, dim, page_size, is_causal, dtype)
        bm = MhaDecodePagedBenchmark(test)
        inputs = test.gen_inputs()
        q, k, v, real_seqlen_kv, block_table = inputs
    
        op = MultiHeadAttentionDecodePagedWithKVCacheOp(
            batch, heads, seqlen_q, seqlen_kv, dim, page_size, is_causal, dtype, tune=tune)
        result = bm.profile(op, *inputs)
        BenchmarkReport.record(op, locals(), result, tag="tileops")
    
        fa3_fn = _fa3_mha_decode_paged(test, k, v)
        if fa3_fn is not None:
>           result_fa3 = bm.profile(fa3_fn, *inputs)
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^

benchmarks/ops/bench_mha_decode_paged.py:162: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
benchmarks/benchmark.py:239: in profile
    latency = bench_kernel(functor, args=inputs)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
benchmarks/benchmark.py:125: in bench_kernel
    _run(i % n_repeat)
benchmarks/benchmark.py:112: in _run
    return fn(*arg_pool[i % _N_CLONES])
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

q = tensor([[[[ 0.1224,  1.3906, -0.6587,  ..., -0.0832, -0.6914, -1.1475],
          [-0.6328, -1.7529,  0.0854,  ...,  0...          [-0.8594, -0.2272,  0.0122,  ...,  2.0605,  0.1791, -1.0342]]]],
       device='cuda:0', dtype=torch.float16)
k = tensor([[[-0.8574, -1.0078, -1.0186,  ...,  2.6348, -0.3135,  0.1152],
         [-0.4451,  0.7578, -0.0101,  ...,  1.2...,
         [ 1.8154,  0.3037, -0.2505,  ...,  0.6899, -2.0352, -0.1475]]],
       device='cuda:0', dtype=torch.float16)
v = tensor([[[-0.0262, -1.0488, -0.5830,  ...,  0.3291, -1.9971,  0.6655],
         [-1.0225,  1.4580,  1.2725,  ...,  0.2...,
         [-0.6460, -1.3691,  1.0576,  ..., -0.0720, -0.9175,  0.6562]]],
       device='cuda:0', dtype=torch.float16)
real_seqlen_kv = tensor([1024, 1024], device='cuda:0', dtype=torch.int32)
block_table = tensor([[0, 1, 2, 3],
        [0, 1, 2, 3]], device='cuda:0', dtype=torch.int32)

    def baseline_fn(q, k, v, real_seqlen_kv, block_table):
>       out = flash_attn_with_kvcache(
            q, k_paged, v_paged,
            cache_seqlens=real_seqlen_kv.int(),
            block_table=block_table.int())
E       TypeError: flash_attn_with_kvcache() got an unexpected keyword argument 'block_table'

benchmarks/ops/bench_mha_decode_paged.py:82: TypeError
----------------------------- Captured stdout call -----------------------------
Start autotuning mha_decode_paged_kernel...
Best config: {'block_M': 64, 'block_N': 64, 'num_split': 4, 'num_stages': 3, 'threads': 256}
mha_decode_paged_kernel initialized with config: {'block_M': 64, 'block_N': 64, 'num_split': 4, 'num_stages': 3, 'threads': 256}
2026-04-05 19:36:57  [TileLang:tilelang.language.eager.builder:WARNING]: Immutable value `k_global` is re-bound. If you want to modify its value, please use T.alloc_var to make it a variable!
Stack (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/usr/local/lib/python3.11/dist-packages/pytest/__main__.py", line 9, in <module>
    raise SystemExit(pytest.console_main())
  File "/usr/local/lib/python3.11/dist-packages/_pytest/config/__init__.py", line 223, in console_main
    code = main()
  File "/usr/local/lib/python3.11/dist-packages/_pytest/config/__init__.py", line 199, in main
    ret: ExitCode | int = config.hook.pytest_cmdline_main(config=config)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_hooks.py", line 512, in __call__
    return self._hookexec(self.name, self._hookimpls.copy(), kwargs, firstresult)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_manager.py", line 120, in _hookexec
    return self._inner_hookexec(hook_name, methods, kwargs, firstresult)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_callers.py", line 121, in _multicall
    res = hook_impl.function(*args)
  File "/usr/local/lib/python3.11/dist-packages/_pytest/main.py", line 365, in pytest_cmdline_main
    return wrap_session(config, _main)
  File "/usr/local/lib/python3.11/dist-packages/_pytest/main.py", line 318, in wrap_session
    session.exitstatus = doit(config, session) or 0
  File "/usr/local/lib/python3.11/dist-packages/_pytest/main.py", line 372, in _main
    config.hook.pytest_runtestloop(session=session)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_hooks.py", line 512, in __call__
    return self._hookexec(self.name, self._hookimpls.copy(), kwargs, firstresult)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_manager.py", line 120, in _hookexec
    return self._inner_hookexec(hook_name, methods, kwargs, firstresult)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_callers.py", line 121, in _multicall
    res = hook_impl.function(*args)
  File "/usr/local/lib/python3.11/dist-packages/_pytest/main.py", line 396, in pytest_runtestloop
    item.config.hook.pytest_runtest_protocol(item=item, nextitem=nextitem)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_hooks.py", line 512, in __call__
    return self._hookexec(self.name, self._hookimpls.copy(), kwargs, firstresult)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_manager.py", line 120, in _hookexec
    return self._inner_hookexec(hook_name, methods, kwargs, firstresult)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_callers.py", line 121, in _multicall
    res = hook_impl.function(*args)
  File "/usr/local/lib/python3.11/dist-packages/_pytest/runner.py", line 118, in pytest_runtest_protocol
    runtestprotocol(item, nextitem=nextitem)
  File "/usr/local/lib/python3.11/dist-packages/_pytest/runner.py", line 137, in runtestprotocol
    reports.append(call_and_report(item, "call", log))
  File "/usr/local/lib/python3.11/dist-packages/_pytest/runner.py", line 244, in call_and_report
    call = CallInfo.from_call(
  File "/usr/local/lib/python3.11/dist-packages/_pytest/runner.py", line 353, in from_call
    result: TResult | None = func()
  File "/usr/local/lib/python3.11/dist-packages/_pytest/runner.py", line 245, in <lambda>
    lambda: runtest_hook(item=item, **kwds),
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_hooks.py", line 512, in __call__
    return self._hookexec(self.name, self._hookimpls.copy(), kwargs, firstresult)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_manager.py", line 120, in _hookexec
    return self._inner_hookexec(hook_name, methods, kwargs, firstresult)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_callers.py", line 121, in _multicall
    res = hook_impl.function(*args)
  File "/usr/local/lib/python3.11/dist-packages/_pytest/runner.py", line 179, in pytest_runtest_call
    item.runtest()
  File "/usr/local/lib/python3.11/dist-packages/_pytest/python.py", line 1720, in runtest
    self.ihook.pytest_pyfunc_call(pyfuncitem=self)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_hooks.py", line 512, in __call__
    return self._hookexec(self.name, self._hookimpls.copy(), kwargs, firstresult)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_manager.py", line 120, in _hookexec
    return self._inner_hookexec(hook_name, methods, kwargs, firstresult)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_callers.py", line 121, in _multicall
    res = hook_impl.function(*args)
  File "/usr/local/lib/python3.11/dist-packages/_pytest/python.py", line 166, in pytest_pyfunc_call
    result = testfunction(**testargs)
  File "/workspace/benchmarks/ops/bench_mha_decode_paged.py", line 157, in test_mha_decode_paged_bench
    result = bm.profile(op, *inputs)
  File "/workspace/benchmarks/benchmark.py", line 239, in profile
    latency = bench_kernel(functor, args=inputs)
  File "/workspace/benchmarks/benchmark.py", line 125, in bench_kernel
    _run(i % n_repeat)
  File "/workspace/benchmarks/benchmark.py", line 112, in _run
    return fn(*arg_pool[i % _N_CLONES])
  File "/workspace/tileops/ops/op.py", line 79, in __call__
    return self.forward(*args, **kwargs)
  File "/workspace/tileops/ops/mha_decode_paged.py", line 48, in forward
    return self.kernel(q, k, v, real_seqlen_kv, block_table)
  File "/workspace/tileops/kernels/kernel.py", line 64, in __call__
    return self.forward(*args, **kwargs)
  File "/workspace/tileops/kernels/flash_decode/mha_decode_paged.py", line 582, in forward
    return _mha_decode_paged_split_op(
  File "/usr/local/lib/python3.11/dist-packages/torch/_library/custom_ops.py", line 698, in __call__
    return self._opoverload(*args, **kwargs)
  File "/usr/local/lib/python3.11/dist-packages/torch/_ops.py", line 819, in __call__
    return self._op(*args, **kwargs)
  File "/usr/local/lib/python3.11/dist-packages/torch/_library/autograd.py", line 112, in autograd_impl
    result = forward_no_grad(*args, Metadata(keyset, keyword_only_args))
  File "/usr/local/lib/python3.11/dist-packages/torch/_library/autograd.py", line 41, in forward_no_grad
    result = op.redispatch(keyset & _C._after_autograd_keyset, *args, **kwargs)
  File "/usr/local/lib/python3.11/dist-packages/torch/_ops.py", line 826, in redispatch
    return self._handle.redispatch_boxed(keyset, *args, **kwargs)  # type: ignore[return-value]
  File "/usr/local/lib/python3.11/dist-packages/torch/_library/custom_ops.py", line 347, in backend_impl
    result = self._backend_fns[device_type](*args, **kwargs)
  File "/usr/local/lib/python3.11/dist-packages/torch/_compile.py", line 54, in inner
    return disable_fn(*args, **kwargs)
  File "/usr/local/lib/python3.11/dist-packages/torch/_dynamo/eval_frame.py", line 1181, in _fn
    return fn(*args, **kwargs)
  File "/usr/local/lib/python3.11/dist-packages/torch/_library/custom_ops.py", line 382, in wrapped_fn
    return fn(*args, **kwargs)
  File "/workspace/tileops/kernels/flash_decode/mha_decode_paged.py", line 421, in _mha_decode_paged_split_op
    return _mha_decode_split_kernel(batch, heads, seqlen_q, seqlen_kv, dim, page_size, is_causal,
  File "/usr/local/lib/python3.11/dist-packages/tilelang/jit/__init__.py", line 440, in __call__
    kernel = self.compile(*args, **kwargs)
  File "/usr/local/lib/python3.11/dist-packages/tilelang/jit/__init__.py", line 374, in compile
    prim_func = self.get_tir(*args, **kwargs)
  File "/usr/local/lib/python3.11/dist-packages/tilelang/jit/__init__.py", line 298, in get_tir
    tir = self.func(*args, **kwargs)
  File "/usr/local/lib/python3.11/dist-packages/tilelang/language/eager/builder.py", line 1142, in __call__
    return self.get_tir(*args, **kwargs)
  File "/usr/local/lib/python3.11/dist-packages/tilelang/language/eager/builder.py", line 1138, in get_tir
    self.p1_cache[p1_key] = self._build_tir_template(**kwargs)
  File "/usr/local/lib/python3.11/dist-packages/tilelang/language/eager/builder.py", line 1106, in _build_tir_template
    return TirTemplate.from_lazy_style(self.orig_func.__name__, self.orig_func(*args, **kwargs))
  File "/workspace/tileops/kernels/flash_decode/mha_decode_paged.py", line 362, in _func
    @T.prim_func
  File "/usr/local/lib/python3.11/dist-packages/tilelang/language/eager/builder.py", line 1224, in prim_func
    return impl(func) if func is not None else impl
  File "/usr/local/lib/python3.11/dist-packages/tilelang/language/eager/builder.py", line 1214, in impl
    ir_gen.gen(builder)(**annot)
  File "/workspace/tileops/kernels/flash_decode/mha_decode_paged.py", line 376, in mha_decode_split
    _mha_decode_split(Q, K, V, real_seqlen_kv, block_table, glse, Output_partial,
  File "/usr/local/lib/python3.11/dist-packages/tilelang/language/eager/builder.py", line 775, in __call__
    res = self.ir_gen.gen(builder)(*args, **kwargs)
  File "/workspace/tileops/kernels/flash_decode/mha_decode_paged.py", line 278, in _mha_decode_split
    k_global += offset
__________________ test_mha_decode_paged_bench[longer-cache] ___________________

batch = 1, heads = 8, seqlen_q = 1, seqlen_kv = 1024, dim = 64, page_size = 256
is_causal = False, dtype = torch.float16, tune = True

    @pytest.mark.parametrize(
        "batch, heads, seqlen_q, seqlen_kv, dim, page_size, is_causal, dtype, tune",
        _MHA_DECODE_PAGED_BENCH_PARAMS,
    )
    def test_mha_decode_paged_bench(batch: int, heads: int, seqlen_q: int, seqlen_kv: int, dim: int,
                                    page_size: int, is_causal: bool, dtype: torch.dtype,
                                    tune: bool) -> None:
        test = _MhaDecodePagedTestBaseline(batch, heads, seqlen_q, seqlen_kv, dim, page_size, is_causal, dtype)
        bm = MhaDecodePagedBenchmark(test)
        inputs = test.gen_inputs()
        q, k, v, real_seqlen_kv, block_table = inputs
    
        op = MultiHeadAttentionDecodePagedWithKVCacheOp(
            batch, heads, seqlen_q, seqlen_kv, dim, page_size, is_causal, dtype, tune=tune)
        result = bm.profile(op, *inputs)
        BenchmarkReport.record(op, locals(), result, tag="tileops")
    
        fa3_fn = _fa3_mha_decode_paged(test, k, v)
        if fa3_fn is not None:
>           result_fa3 = bm.profile(fa3_fn, *inputs)
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^

benchmarks/ops/bench_mha_decode_paged.py:162: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
benchmarks/benchmark.py:239: in profile
    latency = bench_kernel(functor, args=inputs)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
benchmarks/benchmark.py:125: in bench_kernel
    _run(i % n_repeat)
benchmarks/benchmark.py:112: in _run
    return fn(*arg_pool[i % _N_CLONES])
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

q = tensor([[[[ 1.2244e-01,  1.3906e+00, -6.5869e-01,  1.8262e+00,  4.2090e-01,
            1.8896e+00,  5.3613e-01,  2.72...26e-01,
            7.7100e-01,  6.4014e-01, -1.1211e+00,  8.0078e-01]]]],
       device='cuda:0', dtype=torch.float16)
k = tensor([[[-0.8574, -1.0078, -1.0186,  ...,  2.6348, -0.3135,  0.1152],
         [-0.4451,  0.7578, -0.0101,  ...,  1.2...,
         [ 1.8154,  0.3037, -0.2505,  ...,  0.6899, -2.0352, -0.1475]]],
       device='cuda:0', dtype=torch.float16)
v = tensor([[[-0.0262, -1.0488, -0.5830,  ...,  0.3291, -1.9971,  0.6655],
         [-1.0225,  1.4580,  1.2725,  ...,  0.2...,
         [-0.6460, -1.3691,  1.0576,  ..., -0.0720, -0.9175,  0.6562]]],
       device='cuda:0', dtype=torch.float16)
real_seqlen_kv = tensor([1024], device='cuda:0', dtype=torch.int32)
block_table = tensor([[0, 1, 2, 3]], device='cuda:0', dtype=torch.int32)

    def baseline_fn(q, k, v, real_seqlen_kv, block_table):
>       out = flash_attn_with_kvcache(
            q, k_paged, v_paged,
            cache_seqlens=real_seqlen_kv.int(),
            block_table=block_table.int())
E       TypeError: flash_attn_with_kvcache() got an unexpected keyword argument 'block_table'

benchmarks/ops/bench_mha_decode_paged.py:82: TypeError
----------------------------- Captured stdout call -----------------------------
Start autotuning mha_decode_paged_kernel...
Best config: {'block_M': 64, 'block_N': 64, 'num_split': 4, 'num_stages': 2, 'threads': 256}
mha_decode_paged_kernel initialized with config: {'block_M': 64, 'block_N': 64, 'num_split': 4, 'num_stages': 2, 'threads': 256}
2026-04-05 19:36:58  [TileLang:tilelang.language.eager.builder:WARNING]: Immutable value `k_global` is re-bound. If you want to modify its value, please use T.alloc_var to make it a variable!
Stack (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/usr/local/lib/python3.11/dist-packages/pytest/__main__.py", line 9, in <module>
    raise SystemExit(pytest.console_main())
  File "/usr/local/lib/python3.11/dist-packages/_pytest/config/__init__.py", line 223, in console_main
    code = main()
  File "/usr/local/lib/python3.11/dist-packages/_pytest/config/__init__.py", line 199, in main
    ret: ExitCode | int = config.hook.pytest_cmdline_main(config=config)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_hooks.py", line 512, in __call__
    return self._hookexec(self.name, self._hookimpls.copy(), kwargs, firstresult)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_manager.py", line 120, in _hookexec
    return self._inner_hookexec(hook_name, methods, kwargs, firstresult)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_callers.py", line 121, in _multicall
    res = hook_impl.function(*args)
  File "/usr/local/lib/python3.11/dist-packages/_pytest/main.py", line 365, in pytest_cmdline_main
    return wrap_session(config, _main)
  File "/usr/local/lib/python3.11/dist-packages/_pytest/main.py", line 318, in wrap_session
    session.exitstatus = doit(config, session) or 0
  File "/usr/local/lib/python3.11/dist-packages/_pytest/main.py", line 372, in _main
    config.hook.pytest_runtestloop(session=session)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_hooks.py", line 512, in __call__
    return self._hookexec(self.name, self._hookimpls.copy(), kwargs, firstresult)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_manager.py", line 120, in _hookexec
    return self._inner_hookexec(hook_name, methods, kwargs, firstresult)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_callers.py", line 121, in _multicall
    res = hook_impl.function(*args)
  File "/usr/local/lib/python3.11/dist-packages/_pytest/main.py", line 396, in pytest_runtestloop
    item.config.hook.pytest_runtest_protocol(item=item, nextitem=nextitem)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_hooks.py", line 512, in __call__
    return self._hookexec(self.name, self._hookimpls.copy(), kwargs, firstresult)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_manager.py", line 120, in _hookexec
    return self._inner_hookexec(hook_name, methods, kwargs, firstresult)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_callers.py", line 121, in _multicall
    res = hook_impl.function(*args)
  File "/usr/local/lib/python3.11/dist-packages/_pytest/runner.py", line 118, in pytest_runtest_protocol
    runtestprotocol(item, nextitem=nextitem)
  File "/usr/local/lib/python3.11/dist-packages/_pytest/runner.py", line 137, in runtestprotocol
    reports.append(call_and_report(item, "call", log))
  File "/usr/local/lib/python3.11/dist-packages/_pytest/runner.py", line 244, in call_and_report
    call = CallInfo.from_call(
  File "/usr/local/lib/python3.11/dist-packages/_pytest/runner.py", line 353, in from_call
    result: TResult | None = func()
  File "/usr/local/lib/python3.11/dist-packages/_pytest/runner.py", line 245, in <lambda>
    lambda: runtest_hook(item=item, **kwds),
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_hooks.py", line 512, in __call__
    return self._hookexec(self.name, self._hookimpls.copy(), kwargs, firstresult)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_manager.py", line 120, in _hookexec
    return self._inner_hookexec(hook_name, methods, kwargs, firstresult)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_callers.py", line 121, in _multicall
    res = hook_impl.function(*args)
  File "/usr/local/lib/python3.11/dist-packages/_pytest/runner.py", line 179, in pytest_runtest_call
    item.runtest()
  File "/usr/local/lib/python3.11/dist-packages/_pytest/python.py", line 1720, in runtest
    self.ihook.pytest_pyfunc_call(pyfuncitem=self)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_hooks.py", line 512, in __call__
    return self._hookexec(self.name, self._hookimpls.copy(), kwargs, firstresult)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_manager.py", line 120, in _hookexec
    return self._inner_hookexec(hook_name, methods, kwargs, firstresult)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_callers.py", line 121, in _multicall
    res = hook_impl.function(*args)
  File "/usr/local/lib/python3.11/dist-packages/_pytest/python.py", line 166, in pytest_pyfunc_call
    result = testfunction(**testargs)
  File "/workspace/benchmarks/ops/bench_mha_decode_paged.py", line 157, in test_mha_decode_paged_bench
    result = bm.profile(op, *inputs)
  File "/workspace/benchmarks/benchmark.py", line 239, in profile
    latency = bench_kernel(functor, args=inputs)
  File "/workspace/benchmarks/benchmark.py", line 125, in bench_kernel
    _run(i % n_repeat)
  File "/workspace/benchmarks/benchmark.py", line 112, in _run
    return fn(*arg_pool[i % _N_CLONES])
  File "/workspace/tileops/ops/op.py", line 79, in __call__
    return self.forward(*args, **kwargs)
  File "/workspace/tileops/ops/mha_decode_paged.py", line 48, in forward
    return self.kernel(q, k, v, real_seqlen_kv, block_table)
  File "/workspace/tileops/kernels/kernel.py", line 64, in __call__
    return self.forward(*args, **kwargs)
  File "/workspace/tileops/kernels/flash_decode/mha_decode_paged.py", line 582, in forward
    return _mha_decode_paged_split_op(
  File "/usr/local/lib/python3.11/dist-packages/torch/_library/custom_ops.py", line 698, in __call__
    return self._opoverload(*args, **kwargs)
  File "/usr/local/lib/python3.11/dist-packages/torch/_ops.py", line 819, in __call__
    return self._op(*args, **kwargs)
  File "/usr/local/lib/python3.11/dist-packages/torch/_library/autograd.py", line 112, in autograd_impl
    result = forward_no_grad(*args, Metadata(keyset, keyword_only_args))
  File "/usr/local/lib/python3.11/dist-packages/torch/_library/autograd.py", line 41, in forward_no_grad
    result = op.redispatch(keyset & _C._after_autograd_keyset, *args, **kwargs)
  File "/usr/local/lib/python3.11/dist-packages/torch/_ops.py", line 826, in redispatch
    return self._handle.redispatch_boxed(keyset, *args, **kwargs)  # type: ignore[return-value]
  File "/usr/local/lib/python3.11/dist-packages/torch/_library/custom_ops.py", line 347, in backend_impl
    result = self._backend_fns[device_type](*args, **kwargs)
  File "/usr/local/lib/python3.11/dist-packages/torch/_compile.py", line 54, in inner
    return disable_fn(*args, **kwargs)
  File "/usr/local/lib/python3.11/dist-packages/torch/_dynamo/eval_frame.py", line 1181, in _fn
    return fn(*args, **kwargs)
  File "/usr/local/lib/python3.11/dist-packages/torch/_library/custom_ops.py", line 382, in wrapped_fn
    return fn(*args, **kwargs)
  File "/workspace/tileops/kernels/flash_decode/mha_decode_paged.py", line 421, in _mha_decode_paged_split_op
    return _mha_decode_split_kernel(batch, heads, seqlen_q, seqlen_kv, dim, page_size, is_causal,
  File "/usr/local/lib/python3.11/dist-packages/tilelang/jit/__init__.py", line 440, in __call__
    kernel = self.compile(*args, **kwargs)
  File "/usr/local/lib/python3.11/dist-packages/tilelang/jit/__init__.py", line 374, in compile
    prim_func = self.get_tir(*args, **kwargs)
  File "/usr/local/lib/python3.11/dist-packages/tilelang/jit/__init__.py", line 298, in get_tir
    tir = self.func(*args, **kwargs)
  File "/usr/local/lib/python3.11/dist-packages/tilelang/language/eager/builder.py", line 1142, in __call__
    return self.get_tir(*args, **kwargs)
  File "/usr/local/lib/python3.11/dist-packages/tilelang/language/eager/builder.py", line 1138, in get_tir
    self.p1_cache[p1_key] = self._build_tir_template(**kwargs)
  File "/usr/local/lib/python3.11/dist-packages/tilelang/language/eager/builder.py", line 1106, in _build_tir_template
    return TirTemplate.from_lazy_style(self.orig_func.__name__, self.orig_func(*args, **kwargs))
  File "/workspace/tileops/kernels/flash_decode/mha_decode_paged.py", line 362, in _func
    @T.prim_func
  File "/usr/local/lib/python3.11/dist-packages/tilelang/language/eager/builder.py", line 1224, in prim_func
    return impl(func) if func is not None else impl
  File "/usr/local/lib/python3.11/dist-packages/tilelang/language/eager/builder.py", line 1214, in impl
    ir_gen.gen(builder)(**annot)
  File "/workspace/tileops/kernels/flash_decode/mha_decode_paged.py", line 376, in mha_decode_split
    _mha_decode_split(Q, K, V, real_seqlen_kv, block_table, glse, Output_partial,
  File "/usr/local/lib/python3.11/dist-packages/tilelang/language/eager/builder.py", line 775, in __call__
    res = self.ir_gen.gen(builder)(*args, **kwargs)
  File "/workspace/tileops/kernels/flash_decode/mha_decode_paged.py", line 278, in _mha_decode_split
    k_global += offset
__________________ test_mha_decode_paged_bench[shorter-cache] __________________

batch = 1, heads = 8, seqlen_q = 1, seqlen_kv = 512, dim = 64, page_size = 256
is_causal = False, dtype = torch.float16, tune = True

    @pytest.mark.parametrize(
        "batch, heads, seqlen_q, seqlen_kv, dim, page_size, is_causal, dtype, tune",
        _MHA_DECODE_PAGED_BENCH_PARAMS,
    )
    def test_mha_decode_paged_bench(batch: int, heads: int, seqlen_q: int, seqlen_kv: int, dim: int,
                                    page_size: int, is_causal: bool, dtype: torch.dtype,
                                    tune: bool) -> None:
        test = _MhaDecodePagedTestBaseline(batch, heads, seqlen_q, seqlen_kv, dim, page_size, is_causal, dtype)
        bm = MhaDecodePagedBenchmark(test)
        inputs = test.gen_inputs()
        q, k, v, real_seqlen_kv, block_table = inputs
    
        op = MultiHeadAttentionDecodePagedWithKVCacheOp(
            batch, heads, seqlen_q, seqlen_kv, dim, page_size, is_causal, dtype, tune=tune)
        result = bm.profile(op, *inputs)
        BenchmarkReport.record(op, locals(), result, tag="tileops")
    
        fa3_fn = _fa3_mha_decode_paged(test, k, v)
        if fa3_fn is not None:
>           result_fa3 = bm.profile(fa3_fn, *inputs)
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^

benchmarks/ops/bench_mha_decode_paged.py:162: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
benchmarks/benchmark.py:239: in profile
    latency = bench_kernel(functor, args=inputs)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
benchmarks/benchmark.py:125: in bench_kernel
    _run(i % n_repeat)
benchmarks/benchmark.py:112: in _run
    return fn(*arg_pool[i % _N_CLONES])
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

q = tensor([[[[ 1.2244e-01,  1.3906e+00, -6.5869e-01,  1.8262e+00,  4.2090e-01,
            1.8896e+00,  5.3613e-01,  2.72...26e-01,
            7.7100e-01,  6.4014e-01, -1.1211e+00,  8.0078e-01]]]],
       device='cuda:0', dtype=torch.float16)
k = tensor([[[-8.5742e-01, -1.0078e+00, -1.0186e+00,  ...,  2.6348e+00,
          -3.1348e-01,  1.1517e-01],
         [-4.....3711e-03, -4.1211e-01,  ..., -7.1143e-01,
           2.0654e-01,  1.8008e+00]]], device='cuda:0', dtype=torch.float16)
v = tensor([[[-0.0262, -1.0488, -0.5830,  ...,  0.3291, -1.9971,  0.6655],
         [-1.0225,  1.4580,  1.2725,  ...,  0.2...,
         [ 0.7754,  1.5244,  0.6660,  ..., -1.6934, -1.3438, -1.6357]]],
       device='cuda:0', dtype=torch.float16)
real_seqlen_kv = tensor([512], device='cuda:0', dtype=torch.int32)
block_table = tensor([[0, 1]], device='cuda:0', dtype=torch.int32)

    def baseline_fn(q, k, v, real_seqlen_kv, block_table):
>       out = flash_attn_with_kvcache(
            q, k_paged, v_paged,
            cache_seqlens=real_seqlen_kv.int(),
            block_table=block_table.int())
E       TypeError: flash_attn_with_kvcache() got an unexpected keyword argument 'block_table'

benchmarks/ops/bench_mha_decode_paged.py:82: TypeError
----------------------------- Captured stdout call -----------------------------
Start autotuning mha_decode_paged_kernel...
Best config: {'block_M': 64, 'block_N': 64, 'num_split': 2, 'num_stages': 2, 'threads': 128}
mha_decode_paged_kernel initialized with config: {'block_M': 64, 'block_N': 64, 'num_split': 2, 'num_stages': 2, 'threads': 128}
2026-04-05 19:36:58  [TileLang:tilelang.language.eager.builder:WARNING]: Immutable value `k_global` is re-bound. If you want to modify its value, please use T.alloc_var to make it a variable!
Stack (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/usr/local/lib/python3.11/dist-packages/pytest/__main__.py", line 9, in <module>
    raise SystemExit(pytest.console_main())
  File "/usr/local/lib/python3.11/dist-packages/_pytest/config/__init__.py", line 223, in console_main
    code = main()
  File "/usr/local/lib/python3.11/dist-packages/_pytest/config/__init__.py", line 199, in main
    ret: ExitCode | int = config.hook.pytest_cmdline_main(config=config)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_hooks.py", line 512, in __call__
    return self._hookexec(self.name, self._hookimpls.copy(), kwargs, firstresult)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_manager.py", line 120, in _hookexec
    return self._inner_hookexec(hook_name, methods, kwargs, firstresult)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_callers.py", line 121, in _multicall
    res = hook_impl.function(*args)
  File "/usr/local/lib/python3.11/dist-packages/_pytest/main.py", line 365, in pytest_cmdline_main
    return wrap_session(config, _main)
  File "/usr/local/lib/python3.11/dist-packages/_pytest/main.py", line 318, in wrap_session
    session.exitstatus = doit(config, session) or 0
  File "/usr/local/lib/python3.11/dist-packages/_pytest/main.py", line 372, in _main
    config.hook.pytest_runtestloop(session=session)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_hooks.py", line 512, in __call__
    return self._hookexec(self.name, self._hookimpls.copy(), kwargs, firstresult)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_manager.py", line 120, in _hookexec
    return self._inner_hookexec(hook_name, methods, kwargs, firstresult)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_callers.py", line 121, in _multicall
    res = hook_impl.function(*args)
  File "/usr/local/lib/python3.11/dist-packages/_pytest/main.py", line 396, in pytest_runtestloop
    item.config.hook.pytest_runtest_protocol(item=item, nextitem=nextitem)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_hooks.py", line 512, in __call__
    return self._hookexec(self.name, self._hookimpls.copy(), kwargs, firstresult)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_manager.py", line 120, in _hookexec
    return self._inner_hookexec(hook_name, methods, kwargs, firstresult)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_callers.py", line 121, in _multicall
    res = hook_impl.function(*args)
  File "/usr/local/lib/python3.11/dist-packages/_pytest/runner.py", line 118, in pytest_runtest_protocol
    runtestprotocol(item, nextitem=nextitem)
  File "/usr/local/lib/python3.11/dist-packages/_pytest/runner.py", line 137, in runtestprotocol
    reports.append(call_and_report(item, "call", log))
  File "/usr/local/lib/python3.11/dist-packages/_pytest/runner.py", line 244, in call_and_report
    call = CallInfo.from_call(
  File "/usr/local/lib/python3.11/dist-packages/_pytest/runner.py", line 353, in from_call
    result: TResult | None = func()
  File "/usr/local/lib/python3.11/dist-packages/_pytest/runner.py", line 245, in <lambda>
    lambda: runtest_hook(item=item, **kwds),
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_hooks.py", line 512, in __call__
    return self._hookexec(self.name, self._hookimpls.copy(), kwargs, firstresult)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_manager.py", line 120, in _hookexec
    return self._inner_hookexec(hook_name, methods, kwargs, firstresult)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_callers.py", line 121, in _multicall
    res = hook_impl.function(*args)
  File "/usr/local/lib/python3.11/dist-packages/_pytest/runner.py", line 179, in pytest_runtest_call
    item.runtest()
  File "/usr/local/lib/python3.11/dist-packages/_pytest/python.py", line 1720, in runtest
    self.ihook.pytest_pyfunc_call(pyfuncitem=self)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_hooks.py", line 512, in __call__
    return self._hookexec(self.name, self._hookimpls.copy(), kwargs, firstresult)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_manager.py", line 120, in _hookexec
    return self._inner_hookexec(hook_name, methods, kwargs, firstresult)
  File "/usr/local/lib/python3.11/dist-packages/pluggy/_callers.py", line 121, in _multicall
    res = hook_impl.function(*args)
  File "/usr/local/lib/python3.11/dist-packages/_pytest/python.py", line 166, in pytest_pyfunc_call
    result = testfunction(**testargs)
  File "/workspace/benchmarks/ops/bench_mha_decode_paged.py", line 157, in test_mha_decode_paged_bench
    result = bm.profile(op, *inputs)
  File "/workspace/benchmarks/benchmark.py", line 239, in profile
    latency = bench_kernel(functor, args=inputs)
  File "/workspace/benchmarks/benchmark.py", line 125, in bench_kernel
    _run(i % n_repeat)
  File "/workspace/benchmarks/benchmark.py", line 112, in _run
    return fn(*arg_pool[i % _N_CLONES])
  File "/workspace/tileops/ops/op.py", line 79, in __call__
    return self.forward(*args, **kwargs)
  File "/workspace/tileops/ops/mha_decode_paged.py", line 48, in forward
    return self.kernel(q, k, v, real_seqlen_kv, block_table)
  File "/workspace/tileops/kernels/kernel.py", line 64, in __call__
    return self.forward(*args, **kwargs)
  File "/workspace/tileops/kernels/flash_decode/mha_decode_paged.py", line 582, in forward
    return _mha_decode_paged_split_op(
  File "/usr/local/lib/python3.11/dist-packages/torch/_library/custom_ops.py", line 698, in __call__
    return self._opoverload(*args, **kwargs)
  File "/usr/local/lib/python3.11/dist-packages/torch/_ops.py", line 819, in __call__
    return self._op(*args, **kwargs)
  File "/usr/local/lib/python3.11/dist-packages/torch/_library/autograd.py", line 112, in autograd_impl
    result = forward_no_grad(*args, Metadata(keyset, keyword_only_args))
  File "/usr/local/lib/python3.11/dist-packages/torch/_library/autograd.py", line 41, in forward_no_grad
    result = op.redispatch(keyset & _C._after_autograd_keyset, *args, **kwargs)
  File "/usr/local/lib/python3.11/dist-packages/torch/_ops.py", line 826, in redispatch
    return self._handle.redispatch_boxed(keyset, *args, **kwargs)  # type: ignore[return-value]
  File "/usr/local/lib/python3.11/dist-packages/torch/_library/custom_ops.py", line 347, in backend_impl
    result = self._backend_fns[device_type](*args, **kwargs)
  File "/usr/local/lib/python3.11/dist-packages/torch/_compile.py", line 54, in inner
    return disable_fn(*args, **kwargs)
  File "/usr/local/lib/python3.11/dist-packages/torch/_dynamo/eval_frame.py", line 1181, in _fn
    return fn(*args, **kwargs)
  File "/usr/local/lib/python3.11/dist-packages/torch/_library/custom_ops.py", line 382, in wrapped_fn
    return fn(*args, **kwargs)
  File "/workspace/tileops/kernels/flash_decode/mha_decode_paged.py", line 421, in _mha_decode_paged_split_op
    return _mha_decode_split_kernel(batch, heads, seqlen_q, seqlen_kv, dim, page_size, is_causal,
  File "/usr/local/lib/python3.11/dist-packages/tilelang/jit/__init__.py", line 440, in __call__
    kernel = self.compile(*args, **kwargs)
  File "/usr/local/lib/python3.11/dist-packages/tilelang/jit/__init__.py", line 374, in compile
    prim_func = self.get_tir(*args, **kwargs)
  File "/usr/local/lib/python3.11/dist-packages/tilelang/jit/__init__.py", line 298, in get_tir
    tir = self.func(*args, **kwargs)
  File "/usr/local/lib/python3.11/dist-packages/tilelang/language/eager/builder.py", line 1142, in __call__
    return self.get_tir(*args, **kwargs)
  File "/usr/local/lib/python3.11/dist-packages/tilelang/language/eager/builder.py", line 1138, in get_tir
    self.p1_cache[p1_key] = self._build_tir_template(**kwargs)
  File "/usr/local/lib/python3.11/dist-packages/tilelang/language/eager/builder.py", line 1106, in _build_tir_template
    return TirTemplate.from_lazy_style(self.orig_func.__name__, self.orig_func(*args, **kwargs))
  File "/workspace/tileops/kernels/flash_decode/mha_decode_paged.py", line 362, in _func
    @T.prim_func
  File "/usr/local/lib/python3.11/dist-packages/tilelang/language/eager/builder.py", line 1224, in prim_func
    return impl(func) if func is not None else impl
  File "/usr/local/lib/python3.11/dist-packages/tilelang/language/eager/builder.py", line 1214, in impl
    ir_gen.gen(builder)(**annot)
  File "/workspace/tileops/kernels/flash_decode/mha_decode_paged.py", line 376, in mha_decode_split
    _mha_decode_split(Q, K, V, real_seqlen_kv, block_table, glse, Output_partial,
  File "/usr/local/lib/python3.11/dist-packages/tilelang/language/eager/builder.py", line 775, in __call__
    res = self.ir_gen.gen(builder)(*args, **kwargs)
  File "/workspace/tileops/kernels/flash_decode/mha_decode_paged.py", line 278, in _mha_decode_split
    k_global += offset
=============================== warnings summary ===============================
../usr/local/lib/python3.11/dist-packages/torch/jit/_script.py:362: 14 warnings
  /usr/local/lib/python3.11/dist-packages/torch/jit/_script.py:362: DeprecationWarning: `torch.jit.script_method` is deprecated. Please switch to `torch.compile` or `torch.export`.
    warnings.warn(

<frozen importlib._bootstrap>:241
  <frozen importlib._bootstrap>:241: DeprecationWarning: builtin type SwigPyPacked has no __module__ attribute

<frozen importlib._bootstrap>:241
  <frozen importlib._bootstrap>:241: DeprecationWarning: builtin type SwigPyObject has no __module__ attribute

benchmarks/ops/bench_activation.py::test_r2_small_tensor_unary[4096-dtype0]
  /usr/local/lib/python3.11/dist-packages/torch/profiler/profiler.py:217: UserWarning: Warning: Profiler clears events at the end of each cycle.Only events from the current cycle will be reported.To keep events across cycles, set acc_events=True.
    _warn_once(

benchmarks/ops/bench_grouped_gemm_block_m.py: 20 warnings
  /usr/local/lib/python3.11/dist-packages/tilelang/profiler/bench.py:182: UserWarning: Profiler won't be using warmup, this can skew profiler results
    schedule = torch.profiler.schedule(wait=1, warmup=0, active=1, repeat=1)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED benchmarks/ops/bench_gqa_decode_paged.py::test_gqa_decode_paged_bench[serving-8b-p256]
FAILED benchmarks/ops/bench_gqa_decode_paged.py::test_gqa_decode_paged_bench[serving-70b-p256]
FAILED benchmarks/ops/bench_gqa_decode_paged.py::test_gqa_decode_paged_bench[serving-405b-p256]
FAILED benchmarks/ops/bench_mha_decode_paged.py::test_mha_decode_paged_bench[batch2-page256]
FAILED benchmarks/ops/bench_mha_decode_paged.py::test_mha_decode_paged_bench[longer-cache]
FAILED benchmarks/ops/bench_mha_decode_paged.py::test_mha_decode_paged_bench[shorter-cache]
6 failed, 992 passed, 3 skipped, 3 xfailed, 37 warnings in 1530.90s (0:25:30)

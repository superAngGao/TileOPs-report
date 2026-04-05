........................................................................ [  7%]
........................................................................ [ 15%]
........................................................................ [ 23%]
........................................................................ [ 30%]
........................................................................ [ 38%]
........................................................................ [ 46%]
..........................FFF........................................... [ 53%]
........................................................................ [ 61%]
........................................................................ [ 69%]
........................................................................ [ 77%]
........................................................................ [ 84%]
........................................................................ [ 92%]
......................................................................   [100%]Benchmark report saved to profile_run.log

=================================== FAILURES ===================================
_________ test_fused_add_rmsnorm_bench[llama-3.1-405b-prefill-float16] _________

m = 2048, n = 16384, dtype = torch.float16, tune = True

    @pytest.mark.parametrize("m, n, dtype, tune", _manifest_params())
    def test_fused_add_rmsnorm_bench(m: int, n: int, dtype: torch.dtype, tune: bool) -> None:
        test = FusedAddRmsNormTest(m, n, dtype)
        bm = FusedAddRmsNormBenchmark(test)
        inputs = test.gen_inputs()
    
>       op = FusedAddRmsNormOp(M=m, N=n, dtype=dtype, tune=tune)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

benchmarks/ops/bench_fused_add_rmsnorm.py:51: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
tileops/ops/norm/fused_add_rmsnorm.py:68: in __init__
    self.kernel = self.kernel_map["fused_add_rms_norm"](
tileops/kernels/norm/fused_add_norm/fwd.py:356: in __init__
    self.init_config(config, tune)
tileops/kernels/kernel.py:33: in init_config
    self.autotune()
tileops/kernels/kernel.py:94: in autotune
    tuned_kernel = autotuned_kernel_fn()
                   ^^^^^^^^^^^^^^^^^^^^^
/usr/local/lib/python3.11/dist-packages/tilelang/autotuner/tuner.py:695: in __call__
    artifact = autotuner.run()
               ^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <tilelang.autotuner.tuner.AutoTuner object at 0x7f0e013d8b10>
warmup = 10, rep = 10, timeout = 100

    def run(self, warmup: int = 25, rep: int = 100, timeout: int = 30):
        """Run the auto-tuning process.
    
        Args:
            warmup: Number of warmup iterations.
            rep: Number of repetitions for timing.
            timeout: Maximum time per configuration.
    
        Returns:
            AutotuneResult: Results of the auto-tuning process.
        """
        _init_logger_handlers()
    
        sig = inspect.signature(self.fn)
        parameters = sig.parameters
    
        # NOTE(chaofan):  We need to extract some parameters from the closure.
        # Consider the case:
        #   def gemm(M, N, K):
        #       def kernel(...)
        # If we only extract source, M/N/K will be symbolic and there will be cache problem.
        extra_parameters: dict[str, Any] = {}
        cells = self.fn.__closure__
        var_names = self.fn.__code__.co_freevars
        if cells is not None:
            assert len(var_names) == len(cells), "Number of free variables does not match"
            for var_name, cell in zip(var_names, cells):
                if var_name in parameters:
                    continue
                # Cell content must be serializable
                assert isinstance(cell.cell_contents, (int, float, str, bool, type(None))), (
                    f"Cell contents {cell.cell_contents} is not serializable: {type(cell.cell_contents)}"
                )
                extra_parameters[var_name] = cell.cell_contents
    
        if isinstance(self.configs, Callable):
            self.configs = self.configs(*self._kernel_parameters)
    
        key = self.generate_cache_key(parameters, extra_parameters)
    
        with self._lock:
            if env.is_cache_enabled() and not env.is_autotune_cache_disabled():
                # First check in-memory cache
                if key in self._memory_cache:
                    # Include PrimFunc name when hitting autotuner memory cache
                    cached_result = self._memory_cache[key]
                    prim = getattr(cached_result, "func", None)
                    kernel_name = get_prim_func_name(prim, "<unknown>")
                    logger.warning(
                        "Found kernel '%s' in memory cache. For better performance, consider using `@tilelang.autotune` instead of direct AutoTuner.from_kernel.",
                        kernel_name,
                    )
                    return cached_result
    
                # Then check disk cache
                result = self._load_result_from_disk(key)
                if result is not None:
                    # Populate memory cache with disk result
                    self._memory_cache[key] = result
                    return result
    
        best_latency: float = 1e8
        best_config: dict[str, Any] | None = None
        best_kernel: tilelang.JITKernel | None = None
    
        def _compile(**config_arg) -> tilelang.JITKernel:
            compile_args = self.compile_args
            return compile_args.compile_program(self.fn(**config_arg))
    
        if self.jit_compile is None:
            self.jit_compile = _compile
    
        def target_fn(jit_kernel: tilelang.JITKernel):
            # Unpack the context
            profile_args = self.profile_args
            supply_type = profile_args.supply_type
            skip_check = profile_args.skip_check
            manual_check_prog = profile_args.manual_check_prog
            cache_input_tensors = profile_args.cache_input_tensors
            ref_prog = profile_args.ref_prog
            supply_prog = profile_args.supply_prog
            rtol = profile_args.rtol
            atol = profile_args.atol
            max_mismatched_ratio = profile_args.max_mismatched_ratio
            backend = profile_args.backend
    
            profiler = jit_kernel.get_profiler(tensor_supply_type=supply_type)
    
            # Factory functions for generating input tensors.
            # This encapsulates the logic of using either a custom supply program (`supply_prog`)
            # or the default profiler input generation (`profiler._get_inputs`).
            def get_input_tensors_supply(with_output: bool):
                def func():
                    if supply_prog is not None:
                        return supply_prog(profiler._get_params(with_output=with_output))
                    else:
                        return profiler._get_inputs(with_output=with_output)
    
                return func
    
            jit_input_tensors_supply = get_input_tensors_supply(with_output=False)
            ref_input_tensors_supply = get_input_tensors_supply(with_output=False)
    
            if cache_input_tensors:
                params = profiler._get_params(with_output=False)
                if self.jit_input_tensors is None:
                    self.jit_input_tensors = jit_input_tensors_supply()
                else:
                    # check if the cached tensors are compatible with the current configuration
                    assert len(params) == len(self.jit_input_tensors), "len(params) != len(self.jit_input_tensors)"
                    for p, c in zip(params, self.jit_input_tensors):
                        if not isinstance(c, torch.Tensor):
                            # skip non-tensor inputs checking
                            continue
    
                        # Check tensor compatibility using generator expression
                        def shape_equal(a, b):
                            return all(
                                a_dim == b_dim or isinstance(a_dim, Var) or isinstance(b_dim, Var) for a_dim, b_dim in zip(a.shape, b.shape)
                            )
    
                        if p.dtype != c.dtype or not shape_equal(p, c):
                            logger.warning(
                                "\nIncompatible input tensor properties detected between cached tensors and "
                                "tensors regenerated for the current configuration trial. "
                                "This can happen if different tuning configurations require different input shapes/dtypes "
                                "and input tensor caching is enabled.\n"
                                "To ensure fresh, compatible inputs are generated for every trial "
                                "you can disable caching by setting:\n"
                                "  `cache_input_tensors=False`\n"
                                "within your `.set_compile_args(...)` call.\n"
                            )
                            # otherwise, regenerate the input tensors for safety
                            self.jit_input_tensors = jit_input_tensors_supply()
                            break
            else:
                self.jit_input_tensors = jit_input_tensors_supply()
    
            if (not skip_check) and (ref_prog is not None):
                if manual_check_prog is not None:
                    profiler.manual_assert_close(ref_prog, input_tensors=self.jit_input_tensors, manual_check_prog=manual_check_prog)
                else:
                    profiler.assert_allclose(
                        ref_prog, input_tensors=self.jit_input_tensors, rtol=rtol, atol=atol, max_mismatched_ratio=max_mismatched_ratio
                    )
            latency = profiler.do_bench(warmup=warmup, rep=rep, input_tensors=self.jit_input_tensors, backend=backend)
    
            if self.ref_latency_cache is None and ref_prog is not None:
                self.ref_input_tensors = ref_input_tensors_supply()
                self.ref_latency_cache = profiler.do_bench(
                    ref_prog,
                    n_warmup=warmup,
                    n_repeat=rep,
                    input_tensors=self.ref_input_tensors,
                    backend=backend,
                )
    
            return latency, self.ref_latency_cache
    
        config_args = []
        for config in self.configs:
            new_kwargs = {}
            keys = config.keys()
            for name, _ in parameters.items():
                if name in config:
                    new_kwargs[name] = config[name]
            unused_keys = set(keys) - set(new_kwargs.keys())
            if len(unused_keys) > 0:
                raise ValueError(f"Unused keys in config: {unused_keys}")
            config_args.append(new_kwargs)
    
        if len(config_args) == 0:
>           raise ValueError("No configurations to tune, please check your `@autotune` decorator")
E           ValueError: No configurations to tune, please check your `@autotune` decorator

/usr/local/lib/python3.11/dist-packages/tilelang/autotuner/tuner.py:481: ValueError
----------------------------- Captured stdout call -----------------------------
Start autotuning FusedAddRmsNormKernel...
________ test_fused_add_rmsnorm_bench[llama-3.1-405b-prefill-bfloat16] _________

m = 2048, n = 16384, dtype = torch.bfloat16, tune = True

    @pytest.mark.parametrize("m, n, dtype, tune", _manifest_params())
    def test_fused_add_rmsnorm_bench(m: int, n: int, dtype: torch.dtype, tune: bool) -> None:
        test = FusedAddRmsNormTest(m, n, dtype)
        bm = FusedAddRmsNormBenchmark(test)
        inputs = test.gen_inputs()
    
>       op = FusedAddRmsNormOp(M=m, N=n, dtype=dtype, tune=tune)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

benchmarks/ops/bench_fused_add_rmsnorm.py:51: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
tileops/ops/norm/fused_add_rmsnorm.py:68: in __init__
    self.kernel = self.kernel_map["fused_add_rms_norm"](
tileops/kernels/norm/fused_add_norm/fwd.py:356: in __init__
    self.init_config(config, tune)
tileops/kernels/kernel.py:33: in init_config
    self.autotune()
tileops/kernels/kernel.py:94: in autotune
    tuned_kernel = autotuned_kernel_fn()
                   ^^^^^^^^^^^^^^^^^^^^^
/usr/local/lib/python3.11/dist-packages/tilelang/autotuner/tuner.py:695: in __call__
    artifact = autotuner.run()
               ^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <tilelang.autotuner.tuner.AutoTuner object at 0x7f0e3ce2d110>
warmup = 10, rep = 10, timeout = 100

    def run(self, warmup: int = 25, rep: int = 100, timeout: int = 30):
        """Run the auto-tuning process.
    
        Args:
            warmup: Number of warmup iterations.
            rep: Number of repetitions for timing.
            timeout: Maximum time per configuration.
    
        Returns:
            AutotuneResult: Results of the auto-tuning process.
        """
        _init_logger_handlers()
    
        sig = inspect.signature(self.fn)
        parameters = sig.parameters
    
        # NOTE(chaofan):  We need to extract some parameters from the closure.
        # Consider the case:
        #   def gemm(M, N, K):
        #       def kernel(...)
        # If we only extract source, M/N/K will be symbolic and there will be cache problem.
        extra_parameters: dict[str, Any] = {}
        cells = self.fn.__closure__
        var_names = self.fn.__code__.co_freevars
        if cells is not None:
            assert len(var_names) == len(cells), "Number of free variables does not match"
            for var_name, cell in zip(var_names, cells):
                if var_name in parameters:
                    continue
                # Cell content must be serializable
                assert isinstance(cell.cell_contents, (int, float, str, bool, type(None))), (
                    f"Cell contents {cell.cell_contents} is not serializable: {type(cell.cell_contents)}"
                )
                extra_parameters[var_name] = cell.cell_contents
    
        if isinstance(self.configs, Callable):
            self.configs = self.configs(*self._kernel_parameters)
    
        key = self.generate_cache_key(parameters, extra_parameters)
    
        with self._lock:
            if env.is_cache_enabled() and not env.is_autotune_cache_disabled():
                # First check in-memory cache
                if key in self._memory_cache:
                    # Include PrimFunc name when hitting autotuner memory cache
                    cached_result = self._memory_cache[key]
                    prim = getattr(cached_result, "func", None)
                    kernel_name = get_prim_func_name(prim, "<unknown>")
                    logger.warning(
                        "Found kernel '%s' in memory cache. For better performance, consider using `@tilelang.autotune` instead of direct AutoTuner.from_kernel.",
                        kernel_name,
                    )
                    return cached_result
    
                # Then check disk cache
                result = self._load_result_from_disk(key)
                if result is not None:
                    # Populate memory cache with disk result
                    self._memory_cache[key] = result
                    return result
    
        best_latency: float = 1e8
        best_config: dict[str, Any] | None = None
        best_kernel: tilelang.JITKernel | None = None
    
        def _compile(**config_arg) -> tilelang.JITKernel:
            compile_args = self.compile_args
            return compile_args.compile_program(self.fn(**config_arg))
    
        if self.jit_compile is None:
            self.jit_compile = _compile
    
        def target_fn(jit_kernel: tilelang.JITKernel):
            # Unpack the context
            profile_args = self.profile_args
            supply_type = profile_args.supply_type
            skip_check = profile_args.skip_check
            manual_check_prog = profile_args.manual_check_prog
            cache_input_tensors = profile_args.cache_input_tensors
            ref_prog = profile_args.ref_prog
            supply_prog = profile_args.supply_prog
            rtol = profile_args.rtol
            atol = profile_args.atol
            max_mismatched_ratio = profile_args.max_mismatched_ratio
            backend = profile_args.backend
    
            profiler = jit_kernel.get_profiler(tensor_supply_type=supply_type)
    
            # Factory functions for generating input tensors.
            # This encapsulates the logic of using either a custom supply program (`supply_prog`)
            # or the default profiler input generation (`profiler._get_inputs`).
            def get_input_tensors_supply(with_output: bool):
                def func():
                    if supply_prog is not None:
                        return supply_prog(profiler._get_params(with_output=with_output))
                    else:
                        return profiler._get_inputs(with_output=with_output)
    
                return func
    
            jit_input_tensors_supply = get_input_tensors_supply(with_output=False)
            ref_input_tensors_supply = get_input_tensors_supply(with_output=False)
    
            if cache_input_tensors:
                params = profiler._get_params(with_output=False)
                if self.jit_input_tensors is None:
                    self.jit_input_tensors = jit_input_tensors_supply()
                else:
                    # check if the cached tensors are compatible with the current configuration
                    assert len(params) == len(self.jit_input_tensors), "len(params) != len(self.jit_input_tensors)"
                    for p, c in zip(params, self.jit_input_tensors):
                        if not isinstance(c, torch.Tensor):
                            # skip non-tensor inputs checking
                            continue
    
                        # Check tensor compatibility using generator expression
                        def shape_equal(a, b):
                            return all(
                                a_dim == b_dim or isinstance(a_dim, Var) or isinstance(b_dim, Var) for a_dim, b_dim in zip(a.shape, b.shape)
                            )
    
                        if p.dtype != c.dtype or not shape_equal(p, c):
                            logger.warning(
                                "\nIncompatible input tensor properties detected between cached tensors and "
                                "tensors regenerated for the current configuration trial. "
                                "This can happen if different tuning configurations require different input shapes/dtypes "
                                "and input tensor caching is enabled.\n"
                                "To ensure fresh, compatible inputs are generated for every trial "
                                "you can disable caching by setting:\n"
                                "  `cache_input_tensors=False`\n"
                                "within your `.set_compile_args(...)` call.\n"
                            )
                            # otherwise, regenerate the input tensors for safety
                            self.jit_input_tensors = jit_input_tensors_supply()
                            break
            else:
                self.jit_input_tensors = jit_input_tensors_supply()
    
            if (not skip_check) and (ref_prog is not None):
                if manual_check_prog is not None:
                    profiler.manual_assert_close(ref_prog, input_tensors=self.jit_input_tensors, manual_check_prog=manual_check_prog)
                else:
                    profiler.assert_allclose(
                        ref_prog, input_tensors=self.jit_input_tensors, rtol=rtol, atol=atol, max_mismatched_ratio=max_mismatched_ratio
                    )
            latency = profiler.do_bench(warmup=warmup, rep=rep, input_tensors=self.jit_input_tensors, backend=backend)
    
            if self.ref_latency_cache is None and ref_prog is not None:
                self.ref_input_tensors = ref_input_tensors_supply()
                self.ref_latency_cache = profiler.do_bench(
                    ref_prog,
                    n_warmup=warmup,
                    n_repeat=rep,
                    input_tensors=self.ref_input_tensors,
                    backend=backend,
                )
    
            return latency, self.ref_latency_cache
    
        config_args = []
        for config in self.configs:
            new_kwargs = {}
            keys = config.keys()
            for name, _ in parameters.items():
                if name in config:
                    new_kwargs[name] = config[name]
            unused_keys = set(keys) - set(new_kwargs.keys())
            if len(unused_keys) > 0:
                raise ValueError(f"Unused keys in config: {unused_keys}")
            config_args.append(new_kwargs)
    
        if len(config_args) == 0:
>           raise ValueError("No configurations to tune, please check your `@autotune` decorator")
E           ValueError: No configurations to tune, please check your `@autotune` decorator

/usr/local/lib/python3.11/dist-packages/tilelang/autotuner/tuner.py:481: ValueError
----------------------------- Captured stdout call -----------------------------
Start autotuning FusedAddRmsNormKernel...
_________ test_fused_add_rmsnorm_bench[llama-3.1-405b-decode-bfloat16] _________

m = 1, n = 16384, dtype = torch.bfloat16, tune = True

    @pytest.mark.parametrize("m, n, dtype, tune", _manifest_params())
    def test_fused_add_rmsnorm_bench(m: int, n: int, dtype: torch.dtype, tune: bool) -> None:
        test = FusedAddRmsNormTest(m, n, dtype)
        bm = FusedAddRmsNormBenchmark(test)
        inputs = test.gen_inputs()
    
>       op = FusedAddRmsNormOp(M=m, N=n, dtype=dtype, tune=tune)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

benchmarks/ops/bench_fused_add_rmsnorm.py:51: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
tileops/ops/norm/fused_add_rmsnorm.py:68: in __init__
    self.kernel = self.kernel_map["fused_add_rms_norm"](
tileops/kernels/norm/fused_add_norm/fwd.py:356: in __init__
    self.init_config(config, tune)
tileops/kernels/kernel.py:33: in init_config
    self.autotune()
tileops/kernels/kernel.py:94: in autotune
    tuned_kernel = autotuned_kernel_fn()
                   ^^^^^^^^^^^^^^^^^^^^^
/usr/local/lib/python3.11/dist-packages/tilelang/autotuner/tuner.py:695: in __call__
    artifact = autotuner.run()
               ^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <tilelang.autotuner.tuner.AutoTuner object at 0x7f0e3ffdb010>
warmup = 10, rep = 10, timeout = 100

    def run(self, warmup: int = 25, rep: int = 100, timeout: int = 30):
        """Run the auto-tuning process.
    
        Args:
            warmup: Number of warmup iterations.
            rep: Number of repetitions for timing.
            timeout: Maximum time per configuration.
    
        Returns:
            AutotuneResult: Results of the auto-tuning process.
        """
        _init_logger_handlers()
    
        sig = inspect.signature(self.fn)
        parameters = sig.parameters
    
        # NOTE(chaofan):  We need to extract some parameters from the closure.
        # Consider the case:
        #   def gemm(M, N, K):
        #       def kernel(...)
        # If we only extract source, M/N/K will be symbolic and there will be cache problem.
        extra_parameters: dict[str, Any] = {}
        cells = self.fn.__closure__
        var_names = self.fn.__code__.co_freevars
        if cells is not None:
            assert len(var_names) == len(cells), "Number of free variables does not match"
            for var_name, cell in zip(var_names, cells):
                if var_name in parameters:
                    continue
                # Cell content must be serializable
                assert isinstance(cell.cell_contents, (int, float, str, bool, type(None))), (
                    f"Cell contents {cell.cell_contents} is not serializable: {type(cell.cell_contents)}"
                )
                extra_parameters[var_name] = cell.cell_contents
    
        if isinstance(self.configs, Callable):
            self.configs = self.configs(*self._kernel_parameters)
    
        key = self.generate_cache_key(parameters, extra_parameters)
    
        with self._lock:
            if env.is_cache_enabled() and not env.is_autotune_cache_disabled():
                # First check in-memory cache
                if key in self._memory_cache:
                    # Include PrimFunc name when hitting autotuner memory cache
                    cached_result = self._memory_cache[key]
                    prim = getattr(cached_result, "func", None)
                    kernel_name = get_prim_func_name(prim, "<unknown>")
                    logger.warning(
                        "Found kernel '%s' in memory cache. For better performance, consider using `@tilelang.autotune` instead of direct AutoTuner.from_kernel.",
                        kernel_name,
                    )
                    return cached_result
    
                # Then check disk cache
                result = self._load_result_from_disk(key)
                if result is not None:
                    # Populate memory cache with disk result
                    self._memory_cache[key] = result
                    return result
    
        best_latency: float = 1e8
        best_config: dict[str, Any] | None = None
        best_kernel: tilelang.JITKernel | None = None
    
        def _compile(**config_arg) -> tilelang.JITKernel:
            compile_args = self.compile_args
            return compile_args.compile_program(self.fn(**config_arg))
    
        if self.jit_compile is None:
            self.jit_compile = _compile
    
        def target_fn(jit_kernel: tilelang.JITKernel):
            # Unpack the context
            profile_args = self.profile_args
            supply_type = profile_args.supply_type
            skip_check = profile_args.skip_check
            manual_check_prog = profile_args.manual_check_prog
            cache_input_tensors = profile_args.cache_input_tensors
            ref_prog = profile_args.ref_prog
            supply_prog = profile_args.supply_prog
            rtol = profile_args.rtol
            atol = profile_args.atol
            max_mismatched_ratio = profile_args.max_mismatched_ratio
            backend = profile_args.backend
    
            profiler = jit_kernel.get_profiler(tensor_supply_type=supply_type)
    
            # Factory functions for generating input tensors.
            # This encapsulates the logic of using either a custom supply program (`supply_prog`)
            # or the default profiler input generation (`profiler._get_inputs`).
            def get_input_tensors_supply(with_output: bool):
                def func():
                    if supply_prog is not None:
                        return supply_prog(profiler._get_params(with_output=with_output))
                    else:
                        return profiler._get_inputs(with_output=with_output)
    
                return func
    
            jit_input_tensors_supply = get_input_tensors_supply(with_output=False)
            ref_input_tensors_supply = get_input_tensors_supply(with_output=False)
    
            if cache_input_tensors:
                params = profiler._get_params(with_output=False)
                if self.jit_input_tensors is None:
                    self.jit_input_tensors = jit_input_tensors_supply()
                else:
                    # check if the cached tensors are compatible with the current configuration
                    assert len(params) == len(self.jit_input_tensors), "len(params) != len(self.jit_input_tensors)"
                    for p, c in zip(params, self.jit_input_tensors):
                        if not isinstance(c, torch.Tensor):
                            # skip non-tensor inputs checking
                            continue
    
                        # Check tensor compatibility using generator expression
                        def shape_equal(a, b):
                            return all(
                                a_dim == b_dim or isinstance(a_dim, Var) or isinstance(b_dim, Var) for a_dim, b_dim in zip(a.shape, b.shape)
                            )
    
                        if p.dtype != c.dtype or not shape_equal(p, c):
                            logger.warning(
                                "\nIncompatible input tensor properties detected between cached tensors and "
                                "tensors regenerated for the current configuration trial. "
                                "This can happen if different tuning configurations require different input shapes/dtypes "
                                "and input tensor caching is enabled.\n"
                                "To ensure fresh, compatible inputs are generated for every trial "
                                "you can disable caching by setting:\n"
                                "  `cache_input_tensors=False`\n"
                                "within your `.set_compile_args(...)` call.\n"
                            )
                            # otherwise, regenerate the input tensors for safety
                            self.jit_input_tensors = jit_input_tensors_supply()
                            break
            else:
                self.jit_input_tensors = jit_input_tensors_supply()
    
            if (not skip_check) and (ref_prog is not None):
                if manual_check_prog is not None:
                    profiler.manual_assert_close(ref_prog, input_tensors=self.jit_input_tensors, manual_check_prog=manual_check_prog)
                else:
                    profiler.assert_allclose(
                        ref_prog, input_tensors=self.jit_input_tensors, rtol=rtol, atol=atol, max_mismatched_ratio=max_mismatched_ratio
                    )
            latency = profiler.do_bench(warmup=warmup, rep=rep, input_tensors=self.jit_input_tensors, backend=backend)
    
            if self.ref_latency_cache is None and ref_prog is not None:
                self.ref_input_tensors = ref_input_tensors_supply()
                self.ref_latency_cache = profiler.do_bench(
                    ref_prog,
                    n_warmup=warmup,
                    n_repeat=rep,
                    input_tensors=self.ref_input_tensors,
                    backend=backend,
                )
    
            return latency, self.ref_latency_cache
    
        config_args = []
        for config in self.configs:
            new_kwargs = {}
            keys = config.keys()
            for name, _ in parameters.items():
                if name in config:
                    new_kwargs[name] = config[name]
            unused_keys = set(keys) - set(new_kwargs.keys())
            if len(unused_keys) > 0:
                raise ValueError(f"Unused keys in config: {unused_keys}")
            config_args.append(new_kwargs)
    
        if len(config_args) == 0:
>           raise ValueError("No configurations to tune, please check your `@autotune` decorator")
E           ValueError: No configurations to tune, please check your `@autotune` decorator

/usr/local/lib/python3.11/dist-packages/tilelang/autotuner/tuner.py:481: ValueError
----------------------------- Captured stdout call -----------------------------
Start autotuning FusedAddRmsNormKernel...
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
FAILED benchmarks/ops/bench_fused_add_rmsnorm.py::test_fused_add_rmsnorm_bench[llama-3.1-405b-prefill-float16]
FAILED benchmarks/ops/bench_fused_add_rmsnorm.py::test_fused_add_rmsnorm_bench[llama-3.1-405b-prefill-bfloat16]
FAILED benchmarks/ops/bench_fused_add_rmsnorm.py::test_fused_add_rmsnorm_bench[llama-3.1-405b-decode-bfloat16]
3 failed, 931 passed, 37 warnings in 1545.59s (0:25:45)

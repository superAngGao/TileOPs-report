
==================================== ERRORS ====================================
___________ ERROR collecting benchmarks/ops/bench_gated_deltanet.py ____________
ImportError while importing test module '/home/ci-runner/workdir/TileOPs/TileOPs/benchmarks/ops/bench_gated_deltanet.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
../../_tool/Python/3.11.15/x64/lib/python3.11/importlib/__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
benchmarks/ops/bench_gated_deltanet.py:7: in <module>
    from tests.ops.test_gated_deltanet_bwd import GatedDeltaNetBwdTest
E   ImportError: cannot import name 'GatedDeltaNetBwdTest' from 'tests.ops.test_gated_deltanet_bwd' (/home/ci-runner/workdir/TileOPs/TileOPs/tests/ops/test_gated_deltanet_bwd.py)
____ ERROR collecting benchmarks/ops/bench_gated_deltanet_fla_validation.py ____
ImportError while importing test module '/home/ci-runner/workdir/TileOPs/TileOPs/benchmarks/ops/bench_gated_deltanet_fla_validation.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
../../_tool/Python/3.11.15/x64/lib/python3.11/importlib/__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
benchmarks/ops/bench_gated_deltanet_fla_validation.py:19: in <module>
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
E   ModuleNotFoundError: No module named 'fla'
________ ERROR collecting benchmarks/ops/bench_gated_deltanet_vs_fla.py ________
ImportError while importing test module '/home/ci-runner/workdir/TileOPs/TileOPs/benchmarks/ops/bench_gated_deltanet_vs_fla.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
../../_tool/Python/3.11.15/x64/lib/python3.11/importlib/__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
benchmarks/ops/bench_gated_deltanet_vs_fla.py:16: in <module>
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
E   ModuleNotFoundError: No module named 'fla'
=========================== short test summary info ============================
ERROR benchmarks/ops/bench_gated_deltanet.py
ERROR benchmarks/ops/bench_gated_deltanet_fla_validation.py
ERROR benchmarks/ops/bench_gated_deltanet_vs_fla.py
!!!!!!!!!!!!!!!!!!! Interrupted: 3 errors during collection !!!!!!!!!!!!!!!!!!!!
3 errors in 3.09s
::warning::profile_run.log not found; benchmark may have failed partially

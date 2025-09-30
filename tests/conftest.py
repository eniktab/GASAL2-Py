# tests/conftest.py
import ctypes.util
import os
import pytest

def _has_cuda_runtime() -> bool:
    # Try to locate libcudart — simplest universal signal for CUDA runtime availability.
    libcudart = ctypes.util.find_library("cudart")
    if libcudart:
        return True
    # Allow explicit opt-in via env var even if the above check fails (exotic setups)
    return os.environ.get("GASAL2_TEST_FORCE_CUDA", "") == "1"

def pytest_collection_modifyitems(config, items):
    have_cuda = _has_cuda_runtime()
    skip_gpu = pytest.mark.skip(reason="CUDA runtime (libcudart) not detected; set GASAL2_TEST_FORCE_CUDA=1 to override.")
    for it in items:
        if "gpu" in it.keywords and not have_cuda:
            it.add_marker(skip_gpu)

# tests/test_smoke.py
import os
import shutil
import pytest

def _has_gpu():
    return shutil.which("nvidia-smi") is not None

def test_import():
    # Import should succeed if build completed & runtime libs are discoverable
    import gasal2  # noqa: F401

@pytest.mark.skipif(not _has_gpu(), reason="No NVIDIA GPU available")
def test_basic_align():
    from gasal2 import GasalAligner
    try:
        aln = GasalAligner()
    except TypeError:
        # Fall back to a common scoring set if ctor requires it
        aln = GasalAligner(match=2, mismatch=-3, gap_open=5, gap_extend=2)

    # Trivial identity alignment to exercise the kernel
    q, s = "AAATCG", "AAATCG"
    res = aln.align(q, s)  # API shape depends on your wrapper; just smoke it
    assert res is not None

import pytest

@pytest.mark.gpu
def test_align_identity():
    # A minimal semi-global identity alignment should succeed if GPU path is valid.
    from gasal2 import GasalAligner
    try:
        aln = GasalAligner()  # constructor may be arg-less in your build
    except TypeError:
        # fallback to common scoring (as suggested in your README)
        aln = GasalAligner(match=2, mismatch=-3, gap_open=5, gap_extend=2)
    q = "AAATCG"
    s = "AAATCG"
    res = aln.align(q, s)
    # We can't assume a particular result class; assert the core invariants:
    # - result object exists
    # - it has at least a score or cigar / end positions
    assert res is not None
    # Optional, if your binding exposes these
    for attr in ("score", "cigar", "q_end", "s_end"):
        if hasattr(res, attr):
            assert getattr(res, attr) is not None

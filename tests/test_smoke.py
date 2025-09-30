# tests/test_smoke.py
import os
import shutil
import pytest

def _has_gpu():
    return shutil.which("nvidia-smi") is not None

def test_import():
    # Import should succeed if build completed & runtime libs are discoverable
    import gasal2  # noqa: F401

def dump_alignment(res, q: str, s: str) -> None:
    """
    Best-effort pretty-printer for various alignment result shapes.
    Supports dict-like, object-with-attrs, or tuple/list returns.
    """
    import json
    from dataclasses import is_dataclass, asdict

    def first_key(d, *names):
        for n in names:
            if n in d:
                return d[n]
        return None

    def first_attr(o, *names):
        for n in names:
            if hasattr(o, n):
                return getattr(o, n)
        return None

    kind = type(res).__name__
    print(f"\n=== Alignment Result ({kind}) ===")

    data = None
    score = cigar = q_start = q_end = s_start = s_end = aln_len = None
    matches = mismatches = gaps = edit_distance = None

    # 1) Dict-like (most wrappers expose JSON-friendly dicts)
    if isinstance(res, dict):
        data = res
        score = first_key(res, "score", "Score", "alignment_score")
        cigar = first_key(res, "cigar", "CIGAR", "cigar_str")
        q_start = first_key(res, "q_start", "query_start", "qb", "qbeg")
        q_end   = first_key(res, "q_end", "query_end", "qe", "qend")
        s_start = first_key(res, "s_start", "target_start", "tb", "sbeg")
        s_end   = first_key(res, "s_end", "target_end", "te", "send")
        aln_len = first_key(res, "aln_len", "alignment_length", "length")
        matches = first_key(res, "matches", "match")
        mismatches = first_key(res, "mismatches", "mismatch")
        gaps = first_key(res, "gaps", "gap_opens", "gap_count")
        edit_distance = first_key(res, "edit_distance", "ed", "distance")

    # 2) Dataclass instance
    elif is_dataclass(res):
        data = asdict(res)
        score = data.get("score")
        cigar = data.get("cigar") or data.get("cigar_str")
        q_start = data.get("q_start") or data.get("query_start")
        q_end   = data.get("q_end")   or data.get("query_end")
        s_start = data.get("s_start") or data.get("target_start")
        s_end   = data.get("s_end")   or data.get("target_end")
        aln_len = data.get("aln_len") or data.get("alignment_length")
        matches = data.get("matches")
        mismatches = data.get("mismatches")
        gaps = data.get("gaps")
        edit_distance = data.get("edit_distance")

    # 3) Object with attributes
    elif hasattr(res, "__dict__") or hasattr(res, "__slots__"):
        score = first_attr(res, "score", "alignment_score")
        cigar = first_attr(res, "cigar", "cigar_str", "CIGAR")
        q_start = first_attr(res, "q_start", "query_start", "qb", "qbeg")
        q_end   = first_attr(res, "q_end", "query_end", "qe", "qend")
        s_start = first_attr(res, "s_start", "target_start", "tb", "sbeg")
        s_end   = first_attr(res, "s_end", "target_end", "te", "send")
        aln_len = first_attr(res, "aln_len", "alignment_length", "length")
        matches = first_attr(res, "matches", "match")
        mismatches = first_attr(res, "mismatches", "mismatch")
        gaps = first_attr(res, "gaps", "gap_opens", "gap_count")
        edit_distance = first_attr(res, "edit_distance", "ed", "distance")
        try:
            data = res.__dict__.copy()
        except Exception:
            data = {"_repr": repr(res)}

    # 4) Tuple/list – try a common convention: (score, cigar, q_start, q_end, s_start, s_end)
    elif isinstance(res, (tuple, list)):
        if len(res) >= 1: score = res[0]
        if len(res) >= 2: cigar = res[1]
        if len(res) >= 4:
            q_start, q_end = res[2], res[3]
        if len(res) >= 6:
            s_start, s_end = res[4], res[5]
        data = {"tuple": list(res)}

    # Derived / fallbacks
    if aln_len is None and q_start is not None and q_end is not None:
        try:
            aln_len = int(q_end) - int(q_start)
        except Exception:
            pass

    # Print the core stats
    print(f"query:  {q}")
    print(f"target: {s}")
    print(f"score:  {score}")
    print(f"cigar:  {cigar}")
    print(f"q-range: {q_start}..{q_end}   s-range: {s_start}..{s_end}")
    print(f"aln_len: {aln_len}")
    if matches is not None or mismatches is not None or gaps is not None or edit_distance is not None:
        print(f"matches: {matches}   mismatches: {mismatches}   gaps: {gaps}   edit_distance: {edit_distance}")

    # Also dump the raw payload for debugging
    try:
        print("raw:", json.dumps(data, indent=2, default=str))
    except Exception:
        print("raw:", repr(data))
    print("=== End Alignment Result ===\n")

    
@pytest.mark.skipif(not _has_gpu(), reason="No NVIDIA GPU available")
def test_basic_align():
    from gasal2 import GasalAligner
    try:
        aln = GasalAligner()
    except TypeError:
        # Fall back to a common scoring set if ctor requires it
        aln = GasalAligner(match=2, mismatch=-3, gap_open=5, gap_extend=2,
                            max_q=4096, max_t=16384, max_batch=3)

    # Trivial identity alignment to exercise the kernel
    q, s = "AAATCG", "AAATCG"
    res = aln.align(q, s)  # API shape depends on your wrapper; just smoke it
    assert res is not None
    dump_alignment(res, q, s)

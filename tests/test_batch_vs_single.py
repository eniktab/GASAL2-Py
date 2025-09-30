# tests/test_batch_vs_single.py
import random
import time
from typing import List, Tuple

import pytest
from gasal2 import GasalAligner, PAlign

# ---- markers ----
pytestmark = [pytest.mark.gpu, pytest.mark.slow]

# ---- helpers ----
_ALPH = "ACGT"
_OP2C = {0: "M", 1: "X", 2: "D", 3: "I"}

def rand_seq(L: int, seed: int | None = None) -> str:
    rnd = random.Random(seed) if seed is not None else random
    return "".join(rnd.choice(_ALPH) for _ in range(L))

def mutate(s: str, p_sub: float = 0.05, p_ins: float = 0.02, p_del: float = 0.02, seed: int | None = None) -> str:
    rnd = random.Random(seed) if seed is not None else random
    out = []
    i = 0
    while i < len(s):
        r = rnd.random()
        if r < p_sub:
            out.append(rnd.choice([b for b in _ALPH if b != s[i]]))
            i += 1
        elif r < p_sub + p_ins:
            out.append(rnd.choice(_ALPH))  # insert
        elif r < p_sub + p_ins + p_del:
            i += 1  # delete base in target
        else:
            out.append(s[i])
            i += 1
    return "".join(out)

def cigar_runs_to_sam(ops: List[int], lens: List[int]) -> str:
    if not ops or not lens or len(ops) != len(lens):
        return ""
    return "".join(f"{ln}{_OP2C.get(op, 'M')}" for op, ln in zip(ops, lens))

def same_result(a: PAlign, b: PAlign) -> Tuple[bool, str]:
    if (a.score, a.q_beg, a.q_end, a.s_beg, a.s_end) != (b.score, b.q_beg, b.q_end, b.s_beg, b.s_end):
        return False, "score/coords differ"
    if list(a.ops) != list(b.ops) or list(a.lens) != list(b.lens):
        return False, f"CIGAR differ: {cigar_runs_to_sam(a.ops, a.lens)} vs {cigar_runs_to_sam(b.ops, b.lens)}"
    return True, ""

def build_pairs() -> List[Tuple[str, str]]:
    pairs = []
    # 1) small, equal length
    q1 = rand_seq(120, seed=1)
    s1 = mutate(q1, p_sub=0.03, p_ins=0.01, p_del=0.01, seed=2)
    pairs.append((q1, s1))
    # 2) medium, target longer
    q2 = rand_seq(500, seed=3)
    s2 = mutate(q2 + rand_seq(30, seed=4), p_sub=0.05, p_ins=0.02, p_del=0.02, seed=5)
    pairs.append((q2, s2))
    # 3) medium, target shorter
    q3 = rand_seq(600, seed=6)
    s3 = mutate(q3, p_sub=0.05, p_ins=0.0, p_del=0.05, seed=7)
    pairs.append((q3, s3))
    # 4) different GC content
    q4 = ("G" * 150) + rand_seq(200, seed=8) + ("C" * 100)
    s4 = mutate(q4, p_sub=0.04, p_ins=0.02, p_del=0.02, seed=9)
    pairs.append((q4, s4))
    # 5) varied sizes to trigger multiple mini-batches when max_batch=3
    for k in range(10):
        L = 200 + 25 * k  # 200..425
        q = rand_seq(L, seed=10 + k)
        s = mutate(q, p_sub=0.03, p_ins=0.01, p_del=0.01, seed=20 + k)
        pairs.append((q, s))
    return pairs

@pytest.fixture(scope="module")
def aligner() -> GasalAligner:
    # Scoring: match=2, mismatch=-3, gap_open=5, gap_extend=2
    # Force small max_batch to exercise internal mini-batches.
    try:
        return GasalAligner(match=2, mismatch=-3, gap_open=5, gap_extend=2,
                            max_q=4096, max_t=16384, max_batch=3)
    except TypeError:
        # Older constructor signature fallback
        return GasalAligner()

def _compare_pairs(aligner: GasalAligner, pairs: list[tuple[str,str]]):
    queries  = [q for q, _ in pairs]
    targets  = [s for _, s in pairs]

    # Single
    t0 = time.perf_counter()
    singles = [aligner.align(q, s) for (q, s) in pairs]
    t1 = time.perf_counter()

    # Batched
    t2 = time.perf_counter()
    batched = aligner.align_batch(queries, targets)
    t3 = time.perf_counter()

    assert len(singles) == len(batched), "length mismatch singles vs batched"

    mismatches = []
    for i, (ra, rb) in enumerate(zip(singles, batched)):
        ok, why = same_result(ra, rb)
        if not ok:
            qa, sa = pairs[i]
            mismatches.append(
                f"pair #{i}: {why} (|q|={len(qa)}, |s|={len(sa)})\n"
                f"  single CIGAR: {cigar_runs_to_sam(ra.ops, ra.lens)}\n"
                f"  batched CIGAR: {cigar_runs_to_sam(rb.ops, rb.lens)}"
            )

    if mismatches:
        pytest.fail("MISMATCHES detected:\n" + "\n".join(mismatches))

    # Provide timing info as test output (non-failing)
    n = len(pairs)
    single_ms = (t1 - t0) / n * 1e3
    batch_ms  = (t3 - t2) / n * 1e3
    speedup   = ((t1 - t0) / (t3 - t2)) if (t3 - t2) > 0 else float("inf")
    print(f"OK — {n} pairs")
    print(f"Single mode:  {t1 - t0:.4f} s  ({single_ms:.2f} ms/pair)")
    print(f"Batched mode: {t3 - t2:.4f} s  ({batch_ms:.2f} ms/pair)")
    print(f"Speedup (single / batched): x{speedup:.2f}")
    # Do not assert on speedup—CI variance is large.

def test_batch_matches_single(aligner: GasalAligner):
    pairs = build_pairs()
    _compare_pairs(aligner, pairs)

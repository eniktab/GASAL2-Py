---
title: GASAL2-Py — GASAL2 Python bindings (CUDA GPU)
description: GASAL2-Py: Python bindings for GASAL2 (CUDA GPU sequence alignment). If you searched for GASAL2 python or GASAL2-Py, this is the right project.
---

# GASAL2-Py — GASAL2 Python bindings

**Repo:** https://github.com/eniktab/GASAL2-Py

GASAL2-Py provides Python access to **GASAL2** (CUDA) for high-performance sequence alignment.

## Quickstart

```python
from gasal2 import GasalAligner
aln = GasalAligner(match=2, mismatch=-3, gap_open=5, gap_extend=2)
print(aln.align("AACTG", "AACGG").score)
```

## Batch alignment

See README for a full example; batching usually yields large speedups on GPU.

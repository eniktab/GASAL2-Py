# GASAL2-Py: Python bindings for GPU alignment

Python bindings for **GASAL2** (CUDA).  

- A high-performance Pybind11 wrapper (gasal_py.cpp) with:
- double-buffered (ping-pong) CUDA streams,
- correct 8-byte alignment for ASCII H2D buffers,
- OpenMP parallel post-processing for CIGAR coalescing (optional).

> CUDA is supported on Linux and Windows. macOS is not supported by NVIDIA CUDA.

## 0) Requirements

- NVIDIA GPU + compatible driver
- **CUDA Toolkit** installed (e.g., `/usr/local/cuda-12.x`)
- Python ≥ 3.9
- Build tools:
  - **pip** ≥ 23, **cmake** ≥ 3.27, **ninja** ≥ 1.11
  - A C++17 compiler

Optional:
- OpenMP (for parallel host-side post-processing, if enabled in your wrapper)

## 1) Install — the easy way (pip)

### A) Regular install (recommended)
Builds a wheel once; no import-time rebuilds.

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -U pip "cmake>=3.27" "ninja>=1.11"
pip install .
```

#### CUDA detection knobs (if needed)


Supported cache variables (plumbed via CMake):
- `GASAL2_CUDA_HOME` — path to your CUDA toolkit (bin/include/lib64)
- `GASAL2_GPU_SM_ARCH` — e.g., `sm_80`, `sm_86`, `sm_89`, `sm_90`
- `GASAL2_MAX_QUERY_LEN` — compile-time bound for query length (default 2048)
- `GASAL2_N_CODE` — ASCII code used for ambiguous base `'N'` (default `0x4E`)
- `GASAL2_N_PENALTY` — optional define to penalize `'N'` matches

If CMake cannot find CUDA automatically, set:

```bash
export CMAKE_ARGS="-DGASAL2_CUDA_HOME=/usr/local/cuda-12.4 -DGASAL2_GPU_SM_ARCH=sm_86"
# then:
pip install .        # or: pip install -e .
```

## 2) Install — CMake/Ninja quick start (out-of-tree)

If you prefer a pure CMake build/install flow:

```bash
# from repo root
mkdir -p build && cd build
cmake -S .. -B . -G "Ninja" -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
cmake --install .
```

Notes:
- This produces and installs the `_gasal2` extension into your active environment/site-packages (or CMAKE_INSTALL_PREFIX if configured).
- Pass the same CUDA knobs via `-D`:
  ```bash
  cmake -S .. -B . -G "Ninja"     -DCMAKE_BUILD_TYPE=Release     -DGASAL2_CUDA_HOME=/usr/local/cuda-12.4     -DGASAL2_GPU_SM_ARCH=sm_86
  ```

## 3) Verifying the install

### Import-only smoke test

```bash
python - <<'PY'
from gasal2 import GasalAligner, PAlign
print("OK:", GasalAligner, PAlign)
PY
```

### Minimal functional check

```bash
python - <<'PY'
from gasal2 import GasalAligner

try:
    aln = GasalAligner()  # if your build expects ctor args, this will raise
except TypeError:
    # common fallback scoring: match=2, mismatch=-3, gap_open=5, gap_extend=2
    aln = GasalAligner(match=2, mismatch=-3, gap_open=5, gap_extend=2)

print("Methods:", [m for m in dir(aln) if not m.startswith("_")])

if hasattr(aln, "align"):
    q, s = "AAATCG", "AAATCG"
    res = aln.align(q, s)
    print("align() result:", res)
else:
    print("No align() method exposed; inspect API above.")
PY
```

## 4) Choosing the right GPU SM architecture

Find your SM:

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
# Map 8.0->sm_80, 8.6->sm_86, 8.9->sm_89, 9.0->sm_90, etc.
```

Then set `-DGASAL2_GPU_SM_ARCH=sm_XX` during build (pip or CMake).

## 5) Common issues

- **`/tmp/.../cmake: not found` during `import gasal2` (editable installs)**  
  Ensure `cmake`/`ninja` are installed **inside your venv** and on PATH:
  ```bash
  python -m pip install -U "cmake>=3.27" "ninja>=1.11"
  ```
  Or use a non-editable `pip install .`.

- **CUDA not found**  
  Provide `-DGASAL2_CUDA_HOME=/path/to/cuda` and ensure `bin` is on `PATH` and `lib64` on `LD_LIBRARY_PATH` as needed.

- **Wrong SM arch**  
  Rebuild with the correct `GASAL2_GPU_SM_ARCH` for your GPU.

- **Link/runtime errors for `libcudart.so`**  
  Make sure your runtime can find CUDA libraries (system paths, `LD_LIBRARY_PATH`, or rpaths supplied by the build).

## 6) API surface (high-level)

The `gasal2` package re-exports symbols from the compiled `_gasal2` extension:

- `GasalAligner(...)` — construct an aligner (scoring, capacity options)
- `PAlign` — alignment result / POD type (fields depend on binding)

Refer to docstrings and `dir()`/`help()` on the objects after import.

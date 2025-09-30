# GASAL2‑Py — Fast GPU Semi‑Global Alignment with Python Bindings

[![DOI](https://zenodo.org/badge/1066128365.svg)](https://doi.org/10.5281/zenodo.17231489)


> **Description**  
> This project provides Python bindings for **GASAL2** (GPU-accelerated sequence alignment).  
> If the PyPI wheel does not work on your system, clone and **build from source** here: https://github.com/eniktab/GASAL2-Py

> **Note on installation:** Because GASAL2 relies on NVIDIA CUDA libraries, **many users will need to build from source**. Prebuilt wheels on PyPI may not work on all systems (e.g., due to CUDA version/driver/toolkit mismatches or unsupported platform tags). Ensure you have a compatible CUDA Toolkit/driver installed and accessible on your build machine.



**GASAL2‑Py** provides Python bindings and build helpers for **[GASAL2]**, a CUDA‑accelerated pairwise aligner.
This repo includes a minimal reproducible GASAL2 build (static + shared) **and** a high‑performance Pybind11
extension with double‑buffered CUDA streams and optional OpenMP post‑processing.

> The provided code is include **asynchronous GPU
> pipelining (ping‑pong buffering)**, correct **8‑byte ASCII buffer alignment** for H2D copies, and **OpenMP
> parallelization** for host‑side CIGAR coalescing.

---

> CUDA is supported on Linux and Windows. macOS is not supported by NVIDIA CUDA.

---

## Requirements

- **CUDA Toolkit** matching your NVIDIA driver (12.x recommended)
- **Python ≥ 3.8**, `pip`
- One of:
  - **CMake ≥ 3.20** (3.27+ recommended) and a build tool (e.g., Ninja), or
  - System **g++/clang++** and **pybind11** for the manual build path
- Optional: **OpenMP** (GCC/Clang: `-fopenmp`; MSVC: `/openmp`)
- Optional: **pytest** for tests

Quick sanity check:
```bash
nvcc --version
nvidia-smi
```

---

## Quick Start (pip / CMake build)

If wheels are not available for your platform, `pip` will build from source using CMake.

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -U pip setuptools wheel "cmake>=3.27" "ninja>=1.11"
pip install .
```

Verify import:
```bash
python -c "import gasal2; print('GASAL2-Py OK:', gasal2.__version__)"
```

### Run tests (optional)

A) **pytest directly** (after editable install):
```bash
python -m pip install -U pytest
pip install -e .
pytest -q -s
```

B) **CTest without installing**:
```bash
mkdir -p build && cd build
cmake -S .. -B . -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DGASAL2_ENABLE_TESTS=ON
cmake --build . -j
ctest --output-on-failure -C Release
```

C) **Post‑build tests** on `check` target:
```bash
cmake -S .. -B build -G "Ninja" -DCMAKE_BUILD_TYPE=Release       -DGASAL2_ENABLE_TESTS=ON -DGASAL2_TEST_AFTER_BUILD=ON
cmake --build build --target check -j
```

---

## Manual Build (Makefile + Pybind11)

This repo also provides a minimal **GASAL2** build and a **Pybind11** extension for low‑level users.

### 1) Set CUDA

```bash
export CUDA_HOME=/apps/cuda/12.9.0   # adjust to your toolkit path
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

If present, configure helpers:
```bash
make clean || true
./configure.sh "$CUDA_HOME"
```

### 2) Choose GPU SM architecture

Use one of: `sm_70, sm_75, sm_80, sm_86, sm_89, sm_90` (e.g., V100/T4/A100/RTX40/H100).

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
# map 8.0 -> sm_80, 8.9 -> sm_89, 9.0 -> sm_90, etc.
```

### 3) Build GASAL2

Common knobs:
- `GPU_SM_ARCH`: e.g., `sm_80`
- `MAX_QUERY_LEN`: compile‑time bound for buffers (e.g., `4096`)
- `N_CODE`: ASCII code for ambiguous base `'N'` (`0x4E` for uppercase `N`)
- optional `N_PENALTY` define if you penalize matches involving `N`

```bash
make clean || true
./configure.sh "$CUDA_HOME"
make GPU_SM_ARCH=sm_80 MAX_QUERY_LEN=4096 N_CODE=0x4E
# artifacts: ./lib/libgasal.{a,so}, headers in ./include/
```

### 4) Build the Pybind11 module

```bash
python -m pip install pybind11
c++ -O3 -std=c++17 -shared -fPIC gasal_py.cpp   -I./include $(python -m pybind11 --includes)   -L./lib -lgasal -lcudart   -Wl,-rpath,'$ORIGIN/lib'   -fopenmp   -o gasalwrap$(python -c "import sysconfig;print(sysconfig.get_config_var('EXT_SUFFIX'))")
```

- Keep `-Wl,-rpath,'$ORIGIN/lib'` so `libgasal.so` is found at runtime.
- Drop `-fopenmp` if you do not want OpenMP CIGAR coalescing.

Verify:
```bash
python -c "import gasalwrap; print('ok:', gasalwrap)"
```

---

## Minimal Example

```python
# After either build path:
# match=+2, mismatch=-3, gap_open=-5, gap_extend=-1
import gasalwrap

aln = gasalwrap.GasalAligner(2, -3, -5, -1, max_q=2048, max_t=8192, max_batch=1024)

q = "AAACTGNNNTTT"
s = "AAACTGTTTTTT"

res = aln.align(q, s)
print("score:", res.score)
print("q:", res.q_beg, res.q_end, "s:", res.s_beg, res.s_end)
print("ops:", list(res.ops))
print("lens:", list(res.lens))
```

If you see plausible coordinates and nonempty `ops/lens`, the CUDA pipeline and host‑side post‑processing are working.

---

## How to cite

If you use **GASAL2‑Py** in academic work, please cite the Zenodo record and the upstream GASAL2 project.

**DOI:** [![DOI](https://zenodo.org/badge/1066128365.svg)](https://doi.org/10.5281/zenodo.17231489)

### BibTeX (Zenodo)
```bibtex
@software{gasal2py-zenodo,
  title        = {GASAL2-Py: Python bindings for GASAL2},
  author       = {Niktab, Eli and contributors},
  year         = {2025},
  publisher    = {Zenodo},
  version      = {v0.1.0},
  doi          = {10.5281/zenodo.17231489},
  url          = {https://doi.org/10.5281/zenodo.17231489}
}
```

Also cite the **GASAL2** repository (and paper, if applicable) corresponding to the version you use:
- GASAL2 (upstream): https://github.com/nahmedraja/GASAL2

## Troubleshooting

- **`nvcc: command not found`** — set `CUDA_HOME` and update `PATH`.
- **`undefined reference to cudart`** — ensure `LD_LIBRARY_PATH=$CUDA_HOME/lib64` or use the rpath as shown.
- **`actual_target_batch_bytes=… not a multiple of 8`** — use the provided wrapper (it pads H2D sizes to 8‑byte boundaries).
- **Wrong `GPU_SM_ARCH`** — rebuild `libgasal` for your exact GPU.
- **No OpenMP** — remove `-fopenmp` (GCC/Clang) or `/openmp` (MSVC) or install a compiler with OpenMP.

---

## Configuration Reference (CMake)

- `-DGASAL2_ENABLE_TESTS=ON` — enable CTest targets to run Python tests
- `-DGASAL2_TEST_AFTER_BUILD=ON` — adds a `check` target that runs tests post‑build
- `-DCMAKE_CUDA_ARCHITECTURES=<archs>` — e.g., `70;75;80`
- `-DCMAKE_BUILD_TYPE=Release|RelWithDebInfo|Debug`
- Toolchain overrides: `-DCMAKE_C_COMPILER`, `-DCMAKE_CXX_COMPILER`

---

## Versioning & Compatibility

- Wrapper assumes **semi‑global alignment with traceback** is enabled in GASAL2.
- CUDA 12.x generally requires an R545+ NVIDIA driver (check your distro).
- When upgrading CUDA, **rebuild** both GASAL2 and the Python extension.

---

## Upstream repository & licensing

This Python wrapper builds and links against **GASAL2**. The current upstream repository is:

- **Main GASAL2 repo:** https://github.com/nahmedraja/GASAL2

**Important:** Please **check the GASAL2 repository's LICENSE** and any third‑party dependency licenses **before redistributing binaries or wheels** built from this project. Your usage and redistribution must comply with the upstream license(s).

## License

See `LICENSE` in this repository.

[GASAL2]: https://github.com/ixxi-dante/gasal2

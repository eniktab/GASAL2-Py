# GASAL2 + Pybind11 Wrapper (GPU Semi-global Alignment)

This repo provides:

- A minimal, reproducible **GASAL2** build (static + shared libs) using a simplified Makefile.
- A high-performance **Pybind11** wrapper (`gasal_py.cpp`) with:
  - double-buffered (ping-pong) CUDA streams,
  - correct 8-byte alignment for ASCII H2D buffers,
  - OpenMP parallel post-processing for CIGAR coalescing (optional).

Tested on Linux + CUDA 12.x with modern NVIDIA GPUs.

> **Note**: CUDA is supported on Linux and Windows. macOS is not supported by NVIDIA CUDA.

---

## 0) Prerequisites

- **NVIDIA driver** compatible with your CUDA (e.g., CUDA 12.9 usually needs R555+).
- **CUDA Toolkit** (we use `/apps/cuda/12.9.0` below—replace with your path):
  - Provides `nvcc`, `libcudart.so`, headers.
- **Build tools**: `g++` (or `clang++`), `make`, `sed`, `ar`.
- **Python 3.8+**, **pip**, and **Pybind11** for the binding:
  ```bash
  python -m pip install --upgrade pip
  python -m pip install pybind11
  ```
- (Optional) **OpenMP** for parallel CIGAR coalescing:
  - GCC/Clang on Linux: already available (`-fopenmp`).
  - MSVC: `/openmp`.

> If you use Conda/Mamba:
> ```bash
> mamba create -n gasal python=3.11 pybind11 -y
> mamba activate gasal
> ```

---

## 1) Get GASAL2 Source

Choose one:

### A. Clone official GASAL2
```bash
git clone https://github.com/ixxi-dante/gasal2.git
cd gasal2
```

### B. Use the included `src/` from this repo
If this repo already contains `src/`, `include/`, and the custom `Makefile`, you can skip cloning.

---

## 2) Discover Your GPU Architecture (`GPU_SM_ARCH`)

GASAL must be compiled for your GPU’s **SM** (streaming multiprocessor) code, e.g. `sm_80`, `sm_86`, `sm_90`.

### Option 1: `nvidia-smi` + quick mapping
```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
# Example output:
# NVIDIA A100-SXM4-40GB, 8.0
# NVIDIA RTX 4090, 8.9
# NVIDIA H100-PCIE-80GB, 9.0
```
Map **compute capability** → **SM**:
- 7.0 → `sm_70` (V100)
- 7.5 → `sm_75` (T4 / RTX 20)
- 8.0 → `sm_80` (A100)
- 8.6 → `sm_86` (RTX 30)
- 8.9 → `sm_89` (RTX 40)
- 9.0 → `sm_90` (H100/H200/GB200)

### Option 2: CUDA sample (`deviceQuery`)
```bash
$CUDA_HOME/extras/demo_suite/deviceQuery | grep -i 'CUDA Capability'
# CUDA Capability Major/Minor version number: 8.0  -> sm_80
```

### Option 3: Python snippet
```python
import ctypes, os, sys
try:
    import torch
    cc = torch.cuda.get_device_capability(0)
    print(f"sm_{cc[0]}{cc[1]}")
except Exception:
    print("Install PyTorch or use nvidia-smi/deviceQuery to get SM.")
```

> **Set** `GPU_SM_ARCH` accordingly (e.g., `sm_80`).

---

## 3) Configure CUDA Path (once)

If your CUDA toolkit is at `/apps/cuda/12.9.0`:
```bash
export CUDA_HOME=/apps/cuda/12.9.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

If your tree includes a helper script:
```bash
make clean || true
./configure.sh /apps/cuda/12.9.0
```

---

## 4) Build GASAL with the Custom Makefile

Replace the repository’s original Makefile with the one below. It builds both `libgasal.a` and `libgasal.so` with PIC and the right GPU code.

### Build command (example)
- `GPU_SM_ARCH`: from step 2 (e.g., `sm_80`).
- `MAX_QUERY_LEN`: maximum query length GASAL should expect (e.g., `4096`).
- `N_CODE`: ASCII code used for ambiguous base `'N'` (uppercase N = `0x4E`; lowercase n = `0x6E`).
- (Optional) `N_PENALTY`: define if your scoring penalizes `N`.

```bash
make clean || true

# Configure CUDA prefix if needed
./configure.sh /apps/cuda/12.9.0

# Build (example: A100, max query 4096, ASCII 'N')
make GPU_SM_ARCH=sm_80 MAX_QUERY_LEN=4096 N_CODE=0x4E
```

Artifacts will appear in:
```
./lib/libgasal.a
./lib/libgasal.so
./include/  (headers copied from src/)
```

> **Tip**: If you get unresolved `-lcudart` at link time, ensure `LD_LIBRARY_PATH` includes your CUDA `lib64` OR set an rpath in the final binding (see below).

---

## 5) Build the Pybind11 Module

The binding is `gasal_py.cpp` (C++17). It assumes `./include` and `./lib` from the GASAL build are present in the current directory.

```bash
python -m pip install pybind11

c++ -O3 -std=c++17 -shared -fPIC gasal_py.cpp \
  -I./include $(python -m pybind11 --includes) \
  -L./lib -lgasal -lcudart \
  -Wl,-rpath,'$ORIGIN/lib' \
  -fopenmp \
  -o gasalwrap$(python -c "import sysconfig;print(sysconfig.get_config_var('EXT_SUFFIX'))")
```

- `-Wl,-rpath,'$ORIGIN/lib'` ensures `libgasal.so` is found at runtime next to your module.
- Remove `-fopenmp` if you do not want OpenMP parallel host-side CIGAR coalescing.

### Verify import
```bash
python -c "import gasalwrap; print('ok:', gasalwrap)"
```

---

## 6) Quick Smoke Test

```python
import gasalwrap

# match=+2, mismatch=-3, gap_open=-5, gap_extend=-1
aln = gasalwrap.GasalAligner(2, -3, -5, -1, max_q=2048, max_t=8192, max_batch=1024)

q = "AAACTGNNNTTT"
s = "AAACTGTTTTTT"

res = aln.align(q, s)
print("score:", res.score)
print("q:", res.q_beg, res.q_end, "s:", res.s_beg, res.s_end)
print("ops:", list(res.ops))
print("lens:", list(res.lens))
```

If you see a result with plausible coords and nonempty `ops/lens`, the pipeline is working.

---

## 7) Common Issues & Fixes

- **`nvcc: command not found`**  
  Set `CUDA_HOME` and update `PATH`:  
  `export CUDA_HOME=/path/to/cuda && export PATH=$CUDA_HOME/bin:$PATH`

- **`undefined reference to 'cudart'` / cannot find `libcudart.so`**  
  Add `LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH` **or** keep `-Wl,-rpath,'$ORIGIN/lib'` and ensure your final `.so` sits next to `./lib/libgasal.so`.

- **`actual_target_batch_bytes=... is not a multiple of 8`**  
  The wrapper ensures 8-byte padded H2D sizes; ensure you’re using the provided `gasal_py.cpp`.

- **Wrong `GPU_SM_ARCH`**  
  Rebuild `libgasal` with the correct `GPU_SM_ARCH` for your GPU. Using `sm_86` on a `sm_80` GPU will fail.

- **No OpenMP**  
  Remove `-fopenmp` (GCC/Clang) or `/openmp` (MSVC), or install a compiler with OpenMP.

---

## 8) Notes on Parameters

- `MAX_QUERY_LEN`: Compile-time limit to bound GPU buffers. Safe defaults: 2048–4096 for short-read seeds or moderate reads. Increase if needed (with corresponding GPU memory).
- `N_CODE`: GASAL encodes input chars to internal codes. Using `0x4E` (ASCII `'N'`) is standard for ambiguity; set consistently across your pipeline.
- `N_PENALTY`: If defined, GASAL kernels can treat `'N'` differently (e.g., penalize ambiguous matches). Omit to use default behavior.

---

## 9) Repro Build (all in one)

```bash
# 0) Prereqs (one time)
python -m pip install --upgrade pip
python -m pip install pybind11

# 1) CUDA env (adjust path)
export CUDA_HOME=/apps/cuda/12.9.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 2) Clean & configure
make clean || true
./configure.sh $CUDA_HOME

# 3) Build GASAL (example for A100: sm_80)
make GPU_SM_ARCH=sm_80 MAX_QUERY_LEN=4096 N_CODE=0x4E

# 4) Build the Python module
c++ -O3 -std=c++17 -shared -fPIC gasal_py.cpp \
  -I./include $(python -m pybind11 --includes) \
  -L./lib -lgasal -lcudart \
  -Wl,-rpath,'$ORIGIN/lib' \
  -fopenmp \
  -o gasalwrap$(python -c "import sysconfig;print(sysconfig.get_config_var('EXT_SUFFIX'))")

# 5) Test
python -c "import gasalwrap; print(gasalwrap)"
```

---

## 10) Versioning & Compatibility

- The wrapper assumes **semi-global alignment with traceback** (`WITH_TB`) in GASAL.
- For CUDA 12.x you need a recent NVIDIA driver (R545+).
- If you upgrade CUDA, rebuild both GASAL and the Pybind11 module.

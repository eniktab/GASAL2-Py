# GASAL2‑Py

Python bindings and helper utilities for [GASAL2], a GPU‑accelerated pairwise alignment library.

This README covers install/build options and **opt‑in testing** (pytest via CTest) you can trigger after or during the CMake build.

---

## Requirements

- **CUDA Toolkit** (matching your NVIDIA driver)
- **CMake ≥ 3.20** (3.27+ recommended) and a build tool (e.g., Ninja)
- **Python ≥ 3.8** with `pip`
- **gcc/g++** compatible with your CUDA Toolkit
- Optional for tests: `pytest`

> Tip: On Linux, ensure `nvcc --version` and `nvidia-smi` both work before building.

---

## Quick start (pip)

If wheels are not available for your platform, `pip` will build from source.

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -U pip setuptools wheel "cmake>=3.27" "ninja>=1.11"
pip install .
```

Verify import:

```bash
python -c "import gasal2; print('GASAL2-Py OK:', gasal2.__version__)"
```

---

## Build from source with CMake

If you prefer an explicit CMake build (e.g., for CI or local dev):

```bash
# From repo root
mkdir -p build && cd build
cmake -S .. -B . -G "Ninja" -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
```

Install (optional):

```bash
cmake --install .
```

> You can pass CUDA/toolchain hints, e.g. `-DCMAKE_CUDA_ARCHITECTURES=80` or `-DCMAKE_C_COMPILER=gcc-12 -DCMAKE_CXX_COMPILER=g++-12` if needed.

---

## Running tests

You can run the Python test suite with **pytest directly**, or via **CMake/CTest** without installing the package.

### A) Run tests after a pip build

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -U pip "cmake>=3.27" "ninja>=1.11" pytest
pip install -e .           # or: pip install .
pytest -q                  # discovers and runs tests/ with pytest
```

### B) Run tests with CMake/CTest (no install needed)

Enable tests at configure time and run them via `ctest`:

```bash
mkdir -p build && cd build
cmake -S .. -B . -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DGASAL2_ENABLE_TESTS=ON
cmake --build . -j
ctest --output-on-failure -C Release
```

This adds the build tree to `PYTHONPATH` so the compiled extension can be imported by the tests without `cmake --install`.

### C) Run tests automatically after the build (opt‑in)

If you want a **post‑build** test step (useful for CI or local dev), also pass:

```bash
cmake -S .. -B build -G "Ninja" -DCMAKE_BUILD_TYPE=Release \
      -DGASAL2_ENABLE_TESTS=ON -DGASAL2_TEST_AFTER_BUILD=ON
cmake --build build --target check -j
```

This defines a `check` target that invokes `ctest --output-on-failure` after building.

---

## Configuration options (CMake)

- `-DGASAL2_ENABLE_TESTS=ON` – enable CTest targets to run Python tests
- `-DGASAL2_TEST_AFTER_BUILD=ON` – add a `check` target that runs tests post‑build
- `-DCMAKE_CUDA_ARCHITECTURES=<archs>` – e.g., `70;75;80`
- `-DCMAKE_BUILD_TYPE=Release|RelWithDebInfo|Debug`
- Toolchain overrides: `-DCMAKE_C_COMPILER`, `-DCMAKE_CXX_COMPILER`

---

## Troubleshooting

- **CUDA toolkit/driver mismatch**: align your `nvcc` version with the installed driver. Reconfigure the build after fixing your environment.
- **Can’t import in tests**: when using CTest, we inject `PYTHONPATH=${CMAKE_BINARY_DIR}`; if you changed target names or layout, update that path in `CMakeLists.txt`.
- **Link errors to CUDA libs**: ensure `LD_LIBRARY_PATH` (Linux) includes your CUDA lib directory or that rpaths are set correctly.
- **Multiple Python versions**: the build uses the interpreter selected by `pip`/CMake; activate the intended virtualenv first.

---

## License

See `LICENSE` in this repository.


from pathlib import Path
import os, shutil, subprocess, sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sysconfig

# Allow users/CI to pass GPU arch & build knobs via env
GPU_SM_ARCH   = os.environ.get("GASAL_GPU_SM_ARCH", "sm_80")
MAX_QUERY_LEN = os.environ.get("GASAL_MAX_QUERY_LEN", "4096")
N_CODE        = os.environ.get("GASAL_N_CODE", "0x4E")
CUDA_HOME     = os.environ.get("CUDA_HOME", os.environ.get("CUDA_PATH", ""))

here = Path(__file__).parent.resolve()
pkg_lib_dir = here / "src" / "gasalwrap" / "lib"
pkg_lib_dir.mkdir(parents=True, exist_ok=True)

class build_ext_with_make(build_ext):
    def run(self):
        # 1) Build GASAL via your Makefile
        env = os.environ.copy()
        if CUDA_HOME:
            env["CUDA_HOME"] = CUDA_HOME
            env["PATH"] = f"{CUDA_HOME}/bin:" + env.get("PATH", "")
            env["LD_LIBRARY_PATH"] = f"{CUDA_HOME}/lib64:" + env.get("LD_LIBRARY_PATH","")

        # Clean & build
        subprocess.check_call(["make", "clean"], cwd=here, env=env)
        subprocess.check_call([
            "make",
            f"GPU_SM_ARCH={GPU_SM_ARCH}",
            f"MAX_QUERY_LEN={MAX_QUERY_LEN}",
            f"N_CODE={N_CODE}",
        ], cwd=here, env=env)

        # 2) Copy libgasal.so into the package (to be included in wheel)
        built_so = here / "lib" / "libgasal.so"
        if not built_so.exists():
            raise RuntimeError("lib/libgasal.so not found after make")
        shutil.copy2(built_so, pkg_lib_dir / "libgasal.so")

        # 3) Proceed with normal extension build
        super().run()

def ext():
    # Compiler/linker flags
    cxxflags = ["-O3", "-std=c++17", "-fPIC"]
    # Optional OpenMP: add -fopenmp if your code uses it (README mentions it for coalescing)
    if os.environ.get("GASAL_OPENMP", "1") != "0":
        cxxflags.append("-fopenmp")

    # RPATH so the extension finds libgasal.so shipped inside the wheel
    # $ORIGIN expands to the dir of the extension (which lives in gasalwrap/)
    link_args = ["-Wl,-rpath,$ORIGIN/lib"]

    # Include dirs: pybind11 includes + your local include/
    try:
        import pybind11
        py_inc = pybind11.get_include()
    except Exception:
        py_inc = sysconfig.get_paths()["include"]

    return Extension(
        name="gasalwrap.gasalwrap",  # places binary under package
        sources=["gasal_py.cpp"],
        include_dirs=[str(here / "include"), py_inc],
        library_dirs=[str(here / "lib")],
        libraries=["gasal", "cudart"],   # -lgasal -lcudart
        extra_compile_args=cxxflags,
        extra_link_args=link_args,
        # If you need to hint at runtime rpath for CUDA too, prefer system CUDA via LD_LIBRARY_PATH
    )

setup(
    packages=["gasalwrap"],
    package_dir={"": "src"},
    package_data={"gasalwrap": ["lib/libgasal.so"]},
    ext_modules=[ext()],
    cmdclass={"build_ext": build_ext_with_make},
)

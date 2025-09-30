cmake_minimum_required(VERSION 3.23)
project(gasalwrap LANGUAGES CXX)

# ---- Disable LTO globally to prevent toolchain defaults ----
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-lto")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fno-lto")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fno-lto")

# ---- Basics ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

include(FetchContent)
include(ExternalProject)

# Make our local cmake/ folder available and load CUDA env helpers
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
# DetectCudaEnv should define: cuda_detect_sm(), cuda_apply_env_to_target(), etc.
include(DetectCudaEnv)

# --------------------------
# User-configurable options
# --------------------------
set(GASAL2_CUDA_HOME "${GASAL2_CUDA_HOME}" CACHE PATH "Path to CUDA installation (for GASAL2 configure.sh)")
set(GASAL2_GPU_SM_ARCH "${GASAL2_GPU_SM_ARCH}" CACHE STRING "GPU SM arch (e.g., sm_80, sm_86, sm_89)")
set(GASAL2_MAX_QUERY_LEN "2048" CACHE STRING "MAX_QUERY_LEN for GASAL2 make")
set(GASAL2_N_CODE "0x4E" CACHE STRING "N_CODE for GASAL2 (ASCII 'N' = 0x4E)")
set(GASAL2_N_PENALTY "" CACHE STRING "Optional N_PENALTY for GASAL2. Leave empty to omit.")
set(GASAL2_GIT_REPOSITORY "https://github.com/nahmedraja/GASAL2" CACHE STRING "GASAL2 Git repository")
set(GASAL2_GIT_TAG "master" CACHE STRING "GASAL2 Git tag/branch/commit")

# --------------------------
# Python + pybind11 + CUDA
# --------------------------
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

find_package(CUDAToolkit)
if (CUDAToolkit_FOUND)
  message(STATUS "Found CUDAToolkit: ${CUDAToolkit_VERSION}")
elseif(GASAL2_CUDA_HOME)
  message(STATUS "CUDAToolkit package not found; will use GASAL2_CUDA_HOME=${GASAL2_CUDA_HOME} for headers/libs")
else()
  message(WARNING "Neither CUDAToolkit found nor GASAL2_CUDA_HOME set; cudart linking may fail.")
endif()

# Auto-detect SM arch if not provided
if(NOT GASAL2_GPU_SM_ARCH OR GASAL2_GPU_SM_ARCH STREQUAL "")
  cuda_detect_sm(_auto_arch)
  set(GASAL2_GPU_SM_ARCH "${_auto_arch}" CACHE STRING "GPU SM architecture (sm_XX)" FORCE)
endif()
message(STATUS "GASAL2_GPU_SM_ARCH = ${GASAL2_GPU_SM_ARCH}")

# -------------
# GASAL2 paths
# -------------
set(GASAL2_PREFIX        "${CMAKE_BINARY_DIR}/_deps/gasal2")
set(GASAL2_SRC_DIR       "${GASAL2_PREFIX}/src")
set(GASAL2_STAMP_DIR     "${GASAL2_PREFIX}/stamp")
set(GASAL2_DOWNLOAD_DIR  "${GASAL2_PREFIX}/download")
set(GASAL2_INCLUDE_DIR   "${GASAL2_SRC_DIR}/include")
set(GASAL2_LIB_DIR       "${GASAL2_SRC_DIR}/lib")
set(GASAL2_LIBRARY       "${GASAL2_LIB_DIR}/libgasal.a")

# Optional N penalty forwarding
set(_N_PENALTY_ARG "")
if (GASAL2_N_PENALTY)
  set(_N_PENALTY_ARG "N_PENALTY=${GASAL2_N_PENALTY}")
endif()

# -------------
# GASAL2 build (force PIC for static lib used in shared module)
# -------------
ExternalProject_Add(gasal2_ep
  GIT_REPOSITORY        ${GASAL2_GIT_REPOSITORY}
  GIT_TAG               ${GASAL2_GIT_TAG}
  GIT_SHALLOW           1
  GIT_PROGRESS          1
  UPDATE_DISCONNECTED   1

  PREFIX                ${GASAL2_PREFIX}
  DOWNLOAD_DIR          ${GASAL2_DOWNLOAD_DIR}
  STAMP_DIR             ${GASAL2_STAMP_DIR}
  SOURCE_DIR            ${GASAL2_SRC_DIR}
  BUILD_IN_SOURCE       1

  # Run configure, then patch the pattern rules to add -fPIC
  CONFIGURE_COMMAND
    ${CMAKE_COMMAND} -E env PATH=$ENV{PATH} bash -c "
      ./configure.sh ${GASAL2_CUDA_HOME} &&
      # Patch the %.cuo pattern rules to add -Xcompiler -fPIC to NVCC
      sed -i 's|\\$(NVCC) -c|\\$(NVCC) -c -Xcompiler -fPIC|g' Makefile &&
      # Patch the %.cppo pattern rules to add -fPIC to CC (g++)
      sed -i 's|\\$(CC) -c|\\$(CC) -c -fPIC|g' Makefile
    "

  # Build with explicit flags in environment AND make variables
  BUILD_COMMAND
    ${CMAKE_COMMAND} -E env 
      CFLAGS=-fPIC
      CXXFLAGS=-fPIC
      "NVCCFLAGS=-Xcompiler -fPIC"
      PATH=$ENV{PATH}
      bash -lc
        "make GPU_SM_ARCH=${GASAL2_GPU_SM_ARCH} MAX_QUERY_LEN=${GASAL2_MAX_QUERY_LEN} N_CODE=${GASAL2_N_CODE} ${_N_PENALTY_ARG} CFLAGS=-fPIC CXXFLAGS=-fPIC 'NVCCFLAGS=-Xcompiler -fPIC'"

  INSTALL_COMMAND       ""

  LOG_DOWNLOAD          1
  LOG_CONFIGURE         1
  LOG_BUILD             1
)

# ---- Tell Ninja how the library appears ----
add_custom_command(
  OUTPUT "${GASAL2_LIBRARY}"
  COMMAND ${CMAKE_COMMAND} -E env bash -lc "test -f '${GASAL2_LIBRARY}' || { echo 'ERROR: ${GASAL2_LIBRARY} missing after gasal2_ep build'; exit 1; }"
  DEPENDS gasal2_ep
  COMMENT "Waiting for GASAL2 static library: ${GASAL2_LIBRARY}"
  VERBATIM
)
add_custom_target(gasal2_ready DEPENDS "${GASAL2_LIBRARY}")

# -------------------------
# pybind11 extension
# -------------------------
pybind11_add_module(_gasal2 src/gasal2/gasal_py.cpp)

add_dependencies(_gasal2 gasal2_ready)
target_include_directories(_gasal2 PRIVATE "${GASAL2_INCLUDE_DIR}")
target_link_libraries(_gasal2 PRIVATE "${GASAL2_LIBRARY}")

# cudart, via package or manual path
if (CUDAToolkit_FOUND)
  target_link_libraries(_gasal2 PRIVATE CUDA::cudart)
elseif (GASAL2_CUDA_HOME)
  target_link_directories(_gasal2 PRIVATE "${GASAL2_CUDA_HOME}/lib64" "${GASAL2_CUDA_HOME}/lib")
  target_link_libraries(_gasal2 PRIVATE cudart)
endif()

# Optional OpenMP, to mirror your manual -fopenmp
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
  target_link_libraries(_gasal2 PRIVATE OpenMP::OpenMP_CXX)
endif()

# Align CUDA header API version + add CUDA include dirs
cuda_apply_env_to_target(_gasal2)

# --- Disable LTO completely on this target ---
set_property(TARGET _gasal2 PROPERTY INTERPROCEDURAL_OPTIMIZATION OFF)
target_compile_options(_gasal2 PRIVATE -fno-lto)
target_link_options(_gasal2 PRIVATE -fno-lto)

# Ensure PIC on the module
target_compile_options(_gasal2 PRIVATE -fPIC)

# RPATH similar to your manual -Wl,-rpath,'$ORIGIN/lib'
set_target_properties(_gasal2 PROPERTIES
  BUILD_RPATH "\$ORIGIN;\$ORIGIN/lib"
  INSTALL_RPATH "\$ORIGIN;\$ORIGIN/lib"
)

# Install the extension into the package
install(TARGETS _gasal2 DESTINATION gasal2)

option(GASAL2_ENABLE_TESTS "Enable CTest targets to run Python tests" OFF)
option(GASAL2_TEST_AFTER_BUILD "Add 'check' target to run tests after build" OFF)

if (GASAL2_ENABLE_TESTS)
  include(CTest)
  enable_testing()

  # Find Python to drive pytest
  find_package(Python3 COMPONENTS Interpreter REQUIRED)

  # Expose build outputs to Python without install
  set(_PY_ENV "PYTHONPATH=${CMAKE_BINARY_DIR}:$ENV{PYTHONPATH}")

  # Single ctest entry that runs all pytest tests in ./tests
  add_test(
    NAME python_pytest
    COMMAND ${Python3_EXECUTABLE} -m pytest -q
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  )
  set_tests_properties(python_pytest PROPERTIES ENVIRONMENT "${_PY_ENV}")

  # Optional post-build 'check' target (handy for CI or local dev)
  if (GASAL2_TEST_AFTER_BUILD)
    add_custom_target(check
      COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -C $<CONFIG>
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      COMMENT "Running tests (ctest) after build"
    )
    add_dependencies(check ALL)
  endif()
endif()


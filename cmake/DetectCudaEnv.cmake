# cmake/DetectCudaEnv.cmake
include_guard(GLOBAL)

# --- CUDA home detection ---
if(NOT DEFINED GASAL2_CUDA_HOME AND DEFINED ENV{CUDA_HOME})
  set(GASAL2_CUDA_HOME "$ENV{CUDA_HOME}" CACHE PATH "CUDA installation directory")
endif()
if(NOT GASAL2_CUDA_HOME)
  if(EXISTS "/usr/local/cuda")
    set(GASAL2_CUDA_HOME "/usr/local/cuda" CACHE PATH "CUDA installation directory")
  endif()
endif()

# --- Compute capability -> "sm_XX" detection (pick highest if multi-GPU) ---
function(cuda_detect_sm out_var)
  set(_arch "")
  find_program(NVIDIA_SMI nvidia-smi)
  if(NVIDIA_SMI)
    execute_process(
      COMMAND ${NVIDIA_SMI} --query-gpu=compute_cap --format=csv,noheader
      OUTPUT_VARIABLE cap_raw
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET)
    if(cap_raw)
      string(REPLACE "\r\n" "\n" caps "${cap_raw}")
      string(REPLACE "\n" ";" caps "${caps}")
      set(best 0)
      foreach(c IN LISTS caps)
        string(STRIP "${c}" c)
        if(c MATCHES "^[0-9]+(\\.[0-9]+)?$")
          string(REPLACE "." "" cnd "${c}")         # e.g., 8.9 -> 89
          if(NOT cnd STREQUAL "")
            math(EXPR ci "${cnd}")
            if(ci GREATER best)
              set(best ${ci})
            endif()
          endif()
        endif()
      endforeach()
      if(best GREATER 0)
        set(_arch "sm_${best}")
      endif()
    endif()
  endif()
  if(NOT _arch)
    set(_arch "sm_80") # sensible default
  endif()
  set(${out_var} "${_arch}" PARENT_SCOPE)
endfunction()

# --- Read CUDART_VERSION from headers and set CUDA_API_VERSION accordingly ---
function(cuda_detect_api_version out_var)
  # Prefer headers from GASAL2_CUDA_HOME
  set(_candidates
    "${GASAL2_CUDA_HOME}/targets/x86_64-linux/include"
    "${GASAL2_CUDA_HOME}/include")

  # Also consider toolkit discovered by CMake, if available
  if(DEFINED CUDAToolkit_INCLUDE_DIRS)
    list(APPEND _candidates ${CUDAToolkit_INCLUDE_DIRS})
  endif()

  # Find cuda_runtime_api.h
  set(_found "")
  foreach(dir IN LISTS _candidates)
    if(EXISTS "${dir}/cuda_runtime_api.h")
      set(_found "${dir}")
      break()
    endif()
  endforeach()

  set(_api 12000)  # safe default for CUDA 12.0+
  if(_found)
    file(READ "${_found}/cuda_runtime_api.h" _rt)
    string(REGEX MATCH "#[ \t]*define[ \t]+CUDART_VERSION[ \t]+([0-9]+)" _m "${_rt}")
    if(CMAKE_MATCH_1)
      set(_api "${CMAKE_MATCH_1}")
    endif()
  endif()
  set(${out_var} "${_api}" PARENT_SCOPE)
endfunction()

# --- Public helper: apply both API version + includes to a target ---
function(cuda_apply_env_to_target tgt)
  if(NOT TARGET ${tgt})
    message(FATAL_ERROR "cuda_apply_env_to_target: target '${tgt}' not found")
  endif()
  # API version
  cuda_detect_api_version(_api)
  target_compile_definitions(${tgt} PRIVATE CUDA_FORCE_API_VERSION=${_api})
  # Include path (so <cuda_runtime.h> resolves consistently)
  if(GASAL2_CUDA_HOME)
    target_include_directories(${tgt} PRIVATE
      "${GASAL2_CUDA_HOME}/targets/x86_64-linux/include"
      "${GASAL2_CUDA_HOME}/include")
  endif()
endfunction()

# --- Cache SM arch (sm_XX) so users can override with -DGASAL2_GPU_SM_ARCH=sm_89 ---
if(NOT DEFINED GASAL2_GPU_SM_ARCH OR GASAL2_GPU_SM_ARCH STREQUAL "")
  cuda_detect_sm(_sm)
  set(GASAL2_GPU_SM_ARCH "${_sm}" CACHE STRING "CUDA SM architecture (e.g., sm_80, sm_86, sm_89)" FORCE)
endif()
message(STATUS "CUDA_HOME=${GASAL2_CUDA_HOME}  SM=${GASAL2_GPU_SM_ARCH}")


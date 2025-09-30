// gasal_py.cpp
// GASAL2 Pybind11 wrapper (Semi-global WITH_TB, batching) — fixed coords, zero-copy, RAII, GIL release.
//
// Build (Linux/macOS clang/gcc):
//   c++ -O3 -std=c++17 -fPIC -shared gasal_py.cpp \
//     -I./include $(python -m pybind11 --includes) \
//     -L./lib -lgasal -lcudart \
//     -fopenmp \
//     -o gasalwrap$(python -c "import sysconfig;print(sysconfig.get_config_var('EXT_SUFFIX'))")
//
// Build (MSVC):
//   cl /O2 /std:c++17 /LD gasal_py.cpp /I include (pybind includes) /openmp /link gasal.lib cudart.lib

#ifndef CUDA_FORCE_API_VERSION
#define CUDA_FORCE_API_VERSION 12000
#endif
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cstdlib>
#include <thread>
#include <type_traits>
#include <utility>
#include <memory>
#include <iostream>

#ifdef _OPENMP
  #include <omp.h>
#endif

#include "gasal_header.h"  // declares gasal_* APIs, gasal_subst_scores, etc.

namespace py = pybind11;

// ----------------------------------------------------------------------------------
// Scoring holder (converted to GASAL type when copying to device)
// ----------------------------------------------------------------------------------
struct SubstScores {
  int match;
  int mismatch;
  int gap_open;
  int gap_extend;
};

// Set to 1 to recompute score from ops/lens in the wrapper (determinism / testing)
#ifndef GASALWRAP_RECOMPUTE_SCORE
#define GASALWRAP_RECOMPUTE_SCORE 1
#endif

// ----------------------------- CUDA error checking --------------------------------
#define CHECK_CUDA(call) do { \
  cudaError_t err = (call); \
  if (err != cudaSuccess) { \
    throw std::runtime_error(std::string("CUDA error @ ") + __FILE__ + ":" + std::to_string(__LINE__) + " - " + cudaGetErrorString(err)); \
  } \
} while (0)

static inline void check_cuda_last_error() { CHECK_CUDA(cudaGetLastError()); }
static inline int  round_up_8(int bytes)   { return (bytes + 7) & ~7; }

// helper to clamp negatives to 0 then cast to u32
template <typename T>
static inline uint32_t nonneg_to_u32(T v) {
  if constexpr (std::is_signed_v<T>) {
    return v > 0 ? static_cast<uint32_t>(v) : 0u;
  } else {
    return static_cast<uint32_t>(v);
  }
}

// ---------- Result object exposed to Python ----------
struct PAlign {
  int score;
  int q_beg, q_end;
  int s_beg, s_end;
  std::vector<uint8_t>  ops;   // 0=M,1=X,2=D,3=I (coalesced)
  std::vector<uint32_t> lens;  // run-lengths, same size as ops
};

// ------------------------------ Aligner class -------------------------------------
class GasalAligner {
public:
  GasalAligner(int match, int mismatch, int gap_open, int gap_extend,
               int max_q = 2048, int max_t = 8192, int max_batch = 1024)
  : max_q_(max_q), max_t_(max_t), init_cap_(std::max(1, max_batch)), cur_cap_(0)
  {
    int ndev = 0; CHECK_CUDA(cudaGetDeviceCount(&ndev));
    if (ndev == 0) throw std::runtime_error("No CUDA device detected.");

    subst_.match      = match;
    subst_.mismatch   = mismatch;   // typically negative
    subst_.gap_open   = gap_open;   // typically negative
    subst_.gap_extend = gap_extend; // typically negative

    gasal_subst_scores gsubst{};
    gsubst.match      = subst_.match;
    gsubst.mismatch   = subst_.mismatch;
    gsubst.gap_open   = subst_.gap_open;
    gsubst.gap_extend = subst_.gap_extend;
    gasal_copy_subst_scores(&gsubst);
    check_cuda_last_error();

    // RAII for Parameters
    args_ = std::make_unique<Parameters>(0, nullptr);
    args_->algo = SEMI_GLOBAL;
    args_->start_pos = WITH_TB;
    args_->semiglobal_skipping_head = QUERY;
    args_->semiglobal_skipping_tail = QUERY;

    // Two streams for ping-pong pipeline
    stor_v_ = gasal_init_gpu_storage_v(2);
    check_cuda_last_error();

    ensure_capacity(init_cap_);
  }

  ~GasalAligner() {
    if (cur_cap_ > 0) {
      cudaError_t err = cudaDeviceSynchronize();
      if (err != cudaSuccess) {
        std::cerr << "[GASALWRAP WARN] Device sync failed in destructor: "
                  << cudaGetErrorString(err) << std::endl;
      }
    }
    gasal_destroy_streams(&stor_v_, args_.get());
    gasal_destroy_gpu_storage_v(&stor_v_);
  }

  PAlign align(std::string_view q, std::string_view s) {
    if ((int)q.size() > max_q_ || (int)s.size() > max_t_)
      throw std::runtime_error("Sequence length exceeds configured max_q/max_t");
    ensure_capacity(1);

    auto& stor = stor_v_.a[0];
    stor.current_n_alns = 0;

    auto [q_end_bytes, t_end_bytes] = fill_sequence(stor, 0, q, s);
    stor.current_n_alns = 1;

    finalize_batch(stor, 1, /*stream_idx=*/0);

    const int qb = round_up_8(q_end_bytes);
    const int tb = round_up_8(t_end_bytes);

    gasal_aln_async(&stor, qb, tb, 1, args_.get());
    check_cuda_last_error();
    wait_for_stream(stor);

    const auto& R = *stor.host_res;
    const uint32_t nops0 = (R.n_cigar_ops ? nonneg_to_u32(R.n_cigar_ops[0]) : 0u);
    const uint32_t total_ops = nops0;
    return fetch_one(stor, 0, /*prefix_off=*/0u, total_ops);
  }

  // Zero-copy batch interface
  std::vector<PAlign> align_batch(const std::vector<std::string_view>& queries,
                                  const std::vector<std::string_view>& refs)
  {
    const int n = (int)queries.size();
    if (n != (int)refs.size()) throw std::invalid_argument("queries/refs length mismatch");
    if (n == 0) return {};

    // Pre-validate BEFORE any GPU work
    for (int i = 0; i < n; ++i) {
      if ((int)queries[i].size() > max_q_ || (int)refs[i].size() > max_t_) {
        throw std::runtime_error("Sequence length exceeds max_q/max_t at pair " + std::to_string(i));
      }
    }

    std::vector<PAlign> out(n);

    // Ping-pong state
    bool pending[2] = {false, false};
    int  pending_chunk[2] = {0, 0};
    int  pending_base[2]  = {0, 0};

    int base = 0;
    int stream = 0;

    auto launch_chunk = [&](int s, int b) -> int {
      const int chunk = std::min(init_cap_, n - b);
      if (chunk <= 0) return 0;

      ensure_capacity(chunk);

      auto& stor = stor_v_.a[s];
      stor.current_n_alns = 0;

      int max_q_end = 0;
      int max_t_end = 0;

      for (int i = 0; i < chunk; ++i) {
        const auto& q = queries[b + i];
        const auto& r = refs[b + i];

        auto [q_end_bytes, t_end_bytes] = fill_sequence(stor, i, q, r);
        if (q_end_bytes > max_q_end) max_q_end = q_end_bytes;
        if (t_end_bytes > max_t_end) max_t_end = t_end_bytes;

        stor.current_n_alns++;
      }

      finalize_batch(stor, (uint32_t)chunk, s);

      const int qb = round_up_8(max_q_end);
      const int tb = round_up_8(max_t_end);

      gasal_aln_async(&stor, qb, tb, chunk, args_.get());
      check_cuda_last_error();

      pending[s] = true;
      pending_chunk[s] = chunk;
      pending_base[s]  = b;
      return chunk;
    };

    auto fetch_results = [&](int s) {
      if (!pending[s]) return;
      auto& stor = stor_v_.a[s];
      wait_for_stream(stor);

      const auto& R = *stor.host_res;
      const int chunk = pending_chunk[s];
      const int base_idx = pending_base[s];

      std::vector<uint32_t> prefix_sums(chunk, 0u);
      uint32_t total_ops = 0u;

      if (R.n_cigar_ops && R.cigar) {
        for (int i = 0; i < chunk; ++i) {
          prefix_sums[i] = total_ops;
          total_ops += nonneg_to_u32(R.n_cigar_ops[i]);
        }
      }

#ifdef _OPENMP
      #pragma omp parallel for schedule(dynamic, 16) if (chunk >= 64)
#endif
      for (int i = 0; i < chunk; ++i) {
        out[base_idx + i] = fetch_one(stor, i, prefix_sums[i], total_ops);
      }

      pending[s] = false;
      pending_chunk[s] = 0;
      pending_base[s]  = 0;
    };

    if (base < n) {
      int launched = launch_chunk(0, base);
      base += launched;
      stream = 1;
    }
    while (base < n) {
      const int s = stream;
      const int other = s ^ 1;

      int launched = launch_chunk(s, base);
      base += launched;
      fetch_results(other);
      stream ^= 1;
    }
    fetch_results(0);
    fetch_results(1);

    return out;
  }

private:
  void ensure_capacity(int /*need*/) {
    if (cur_cap_ > 0) return; // already allocated up to init_cap_
    gasal_init_streams(&stor_v_, max_q_, max_t_, init_cap_, args_.get());
    check_cuda_last_error();
    cur_cap_ = init_cap_;
    opbuf_[0].assign((size_t)init_cap_, 0u);
    opbuf_[1].assign((size_t)init_cap_, 0u);
  }

  // Fill one pair; returns end offsets in ASCII bytes for query and target
  static inline std::pair<int,int> fill_sequence(gasal_gpu_storage_t& stor, int i,
                                  std::string_view q, std::string_view s)
  {
    const uint32_t qoff = gasal_host_batch_fill(&stor, (uint32_t)i, q.data(), (uint32_t)q.size(), QUERY);
    const uint32_t toff = gasal_host_batch_fill(&stor, (uint32_t)i, s.data(), (uint32_t)s.size(), TARGET);

    stor.host_query_batch_offsets[i]  = qoff;
    stor.host_target_batch_offsets[i] = toff;
    stor.host_query_batch_lens[i]     = (uint32_t)q.size();
    stor.host_target_batch_lens[i]    = (uint32_t)s.size();

    return { (int)qoff + (int)q.size(), (int)toff + (int)s.size() };
  }

  inline void finalize_batch(gasal_gpu_storage_t& stor, uint32_t nseqs, int stream_idx) {
    if (nseqs == 0) {
      gasal_op_fill(&stor, nullptr, 0, QUERY);
      gasal_op_fill(&stor, nullptr, 0, TARGET);
    } else {
      gasal_op_fill(&stor, opbuf_[stream_idx].data(), nseqs, QUERY);
      gasal_op_fill(&stor, opbuf_[stream_idx].data(), nseqs, TARGET);
    }
  }

  static inline void wait_for_stream(gasal_gpu_storage_t& stor) {
    while (gasal_is_aln_async_done(&stor) == -1) std::this_thread::yield();
    check_cuda_last_error();
  }

  static inline bool valid_slice(const gasal_gpu_storage_t& stor,
                                 uint32_t start, uint32_t nops, uint32_t total_ops)
  {
    if (nops == 0) return true;
    if (!stor.host_res || !stor.host_res->cigar) return false;
    if (start > total_ops) return false;
    if (total_ops - start < nops) return false;
    return true;
  }

  // Safe getter for possibly-null arrays (report 0 if missing)
  static inline int get_coord(const int32_t* arr, int i) {
    if (!arr) return 0;
    return std::max(0, arr[i]);
  }

  // CIGAR extraction and coordinate handling
  inline PAlign fetch_one(gasal_gpu_storage_t& stor, int i,
                          uint32_t prefix_off, uint32_t total_ops)
  {
    const auto& R = *stor.host_res;

    const int score_in = (R.aln_score ? R.aln_score[i] : 0);
    const uint32_t nops_i = (R.n_cigar_ops ? nonneg_to_u32(R.n_cigar_ops[i]) : 0u);

    // Start/end reported by GASAL are relative to the ORIGINAL inputs (required behavior)
    int q_beg = get_coord(R.query_batch_start, i);
    int s_beg = get_coord(R.target_batch_start, i);
    int q_end = get_coord(R.query_batch_end,   i);
    int s_end = get_coord(R.target_batch_end,  i);

    if (q_end < q_beg) q_end = q_beg;
    if (s_end < s_beg) s_end = s_beg;

    // If no CIGAR or invalid slice: return empty traceback and reported coords
    if (nops_i == 0 || !R.cigar || !valid_slice(stor, prefix_off, nops_i, total_ops)) {
      return PAlign{score_in, q_beg, q_end, s_beg, s_end, {}, {}};
    }

    // Coalesce contiguous identical ops; sanitize >3 -> M(0)
    const uint8_t* ops_raw = R.cigar + prefix_off;
    std::vector<uint8_t>  ops;
    std::vector<uint32_t> lens;
    ops.reserve(nops_i / 2 + 4);
    lens.reserve(nops_i / 2 + 4);

    uint8_t prev = ops_raw[0]; if (prev > 3) prev = 0;
    uint32_t run = 1;
    for (uint32_t t = 1; t < nops_i; ++t) {
      uint8_t o = ops_raw[t]; if (o > 3) o = 0;
      if (o == prev) { ++run; }
      else { ops.push_back(prev); lens.push_back(run); prev = o; run = 1; }
    }
    ops.push_back(prev);
    lens.push_back(run);

    // Compute spans from ops
    uint32_t q_span = 0, s_span = 0;
    uint64_t m_count = 0, x_count = 0;
    for (size_t k = 0; k < ops.size(); ++k) {
      const uint8_t o = ops[k];
      const uint32_t L = lens[k];
      if (o == 0) { q_span += L; s_span += L; m_count += L; }      // M
      else if (o == 1) { q_span += L; s_span += L; x_count += L; } // X
      else if (o == 2) { s_span += L; }                            // D
      else if (o == 3) { q_span += L; }                            // I
    }

    // If CIGAR spans disagree with reported (semi-global) lengths, trust CIGAR
    if ((int)q_span != (q_end - q_beg)) q_end = q_beg + (int)q_span;
    if ((int)s_span != (s_end - s_beg)) s_end = s_beg + (int)s_span;

    int score_out = score_in;
#if GASALWRAP_RECOMPUTE_SCORE
    {
      const int match      = subst_.match;
      const int mismatch   = subst_.mismatch;
      const int gap_open   = subst_.gap_open;
      const int gap_extend = subst_.gap_extend;

      auto gap_cost = [&](uint32_t L) -> int {
        if (L == 0) return 0;
        return gap_open + (int)(L - 1) * gap_extend;
      };

      int acc = 0;
      acc += (int)m_count * match;
      acc += (int)x_count * mismatch;
      for (size_t k = 0; k < ops.size(); ++k) {
        if (ops[k] == 2 || ops[k] == 3) acc += gap_cost(lens[k]);
      }
      score_out = acc;
    }
#endif

    return PAlign{score_out, q_beg, q_end, s_beg, s_end, std::move(ops), std::move(lens)};
  }

private:
  int max_q_, max_t_;
  int init_cap_;
  int cur_cap_;
  SubstScores subst_{};
  gasal_gpu_storage_v stor_v_{};
  std::vector<uint8_t> opbuf_[2];
  std::unique_ptr<Parameters> args_;
};

// ------------------------------- Pybind11 module ----------------------------------
PYBIND11_MODULE(_gasal2, m) {
  py::class_<PAlign>(m, "PAlign")
      .def_readonly("score", &PAlign::score)
      .def_readonly("q_beg", &PAlign::q_beg)
      .def_readonly("q_end", &PAlign::q_end)
      .def_readonly("s_beg", &PAlign::s_beg)
      .def_readonly("s_end", &PAlign::s_end)
      .def_readonly("ops",   &PAlign::ops)
      .def_readonly("lens",  &PAlign::lens);

  py::class_<GasalAligner>(m, "GasalAligner")
      .def(py::init<int,int,int,int,int,int,int>(),
           py::arg("match"), py::arg("mismatch"),
           py::arg("gap_open"), py::arg("gap_extend"),
           py::arg("max_q")=2048, py::arg("max_t")=8192, py::arg("max_batch")=1024)
      .def("align",       &GasalAligner::align,       py::call_guard<py::gil_scoped_release>())
      .def("align_batch", &GasalAligner::align_batch, py::call_guard<py::gil_scoped_release>());
}


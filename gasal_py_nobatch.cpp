// gasal_py.cpp
// Minimal pybind11 wrapper for one-pair semi-global alignment (WITH_TB) in GASAL2.
// Build:
//   python -m pip install pybind11
//   c++ -O3 -std=c++17 -shared -fPIC gasal_py.cpp -I./include $(python -m pybind11 --includes) \
//       -L./lib -lgasal -lcudart -o gasalwrap$(python -c "import sysconfig;print(sysconfig.get_config_var('EXT_SUFFIX'))")

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <string>
#include <vector>
#include <stdexcept>

#include "gasal_header.h"

namespace py = pybind11;

struct PAlign {
  int score;
  int q_beg, q_end;
  int s_beg, s_end;
  std::vector<uint8_t> cigar; // raw GASAL2 CIGAR bytes
};

class GasalAligner {
public:
  GasalAligner(int match, int mismatch, int gap_open, int gap_extend,
               int max_q = 2048, int max_t = 8192)
  : max_q_(max_q), max_t_(max_t) {
    subst_.match      = match;
    subst_.mismatch   = mismatch;
    subst_.gap_open   = gap_open;
    subst_.gap_extend = gap_extend;
    gasal_copy_subst_scores(&subst_);

    args_ = new Parameters(0, nullptr);
    args_->algo = SEMI_GLOBAL;
    args_->start_pos = WITH_TB;
    // typical semi-global: query ends are free; adjust to your exact semantics
    args_->semiglobal_skipping_head = QUERY;
    args_->semiglobal_skipping_tail = QUERY;

    stor_v_ = gasal_init_gpu_storage_v(1);
    gasal_init_streams(&stor_v_, max_q_, max_t_, /*max_n_alns*/1, args_);
  }

  ~GasalAligner() {
    gasal_destroy_streams(&stor_v_, args_);
    gasal_destroy_gpu_storage_v(&stor_v_);
    delete args_;
  }

PAlign align(const std::string& q_in, const std::string& s_in) {
  // 1) sanitize to uppercase A/C/G/T/N
  auto sanitize = [](const std::string& x) {
    std::string y; y.reserve(x.size());
    for (unsigned char c : x) {
      char u = (char)std::toupper(c);
      y.push_back((u=='A'||u=='C'||u=='G'||u=='T'||u=='N') ? u : 'N');
    }
    return y;
  };
  std::string q = sanitize(q_in);
  std::string s = sanitize(s_in);

  if ((int)q.size() > max_q_ || (int)s.size() > max_t_)
    throw std::runtime_error("Sequence length exceeds configured max_q/max_t");

  auto& stor = stor_v_.a[0];
  stor.current_n_alns = 0;

  // 2) batch-fill, capture offsets, set true lengths
  uint32_t q_off = gasal_host_batch_fill(&stor, 0, q.c_str(), (uint32_t)q.size(), QUERY);
  uint32_t t_off = gasal_host_batch_fill(&stor, 0, s.c_str(), (uint32_t)s.size(), TARGET);
  stor.host_query_batch_offsets[0]  = q_off;
  stor.host_target_batch_offsets[0] = t_off;
  stor.host_query_batch_lens[0]  = (uint32_t)q.size();
  stor.host_target_batch_lens[0] = (uint32_t)s.size();
  stor.current_n_alns = 1;

  gasal_op_fill(&stor, 0, 0, QUERY);
  gasal_op_fill(&stor, 0, 0, TARGET);

  // 3) compute packed bytes and round UP to 8-byte boundary
  auto packed_bytes_rounded = [](uint32_t L) -> int {
    // 4 bases per byte → ceil(L/4) bytes, then round bytes to multiple of 8
    uint32_t bytes = (L + 3u) / 4u;
    return int(((bytes + 7u) / 8u) * 8u);
  };
  const int q_bytes = packed_bytes_rounded(stor.host_query_batch_lens[0]);
  const int t_bytes = packed_bytes_rounded(stor.host_target_batch_lens[0]);

  gasal_aln_async(&stor, q_bytes, t_bytes, /*n_alns*/1, args_);
  while (gasal_is_aln_async_done(&stor) == -1) {}

  const auto& R = *stor.host_res;
  const int idx = 0;

  const int score = R.aln_score[idx];
  const int q_beg = R.query_batch_start[idx];
  const int q_end = R.query_batch_end[idx];
  const int s_beg = R.target_batch_start[idx];
  const int s_end = R.target_batch_end[idx];

  const uint32_t c0  = stor.host_query_batch_offsets[idx];
  const uint32_t n   = R.n_cigar_ops[idx];
  const uint8_t* cb  = R.cigar + c0;
  const uint8_t* ce  = cb + n;

  return PAlign{score, q_beg, q_end, s_beg, s_end, std::vector<uint8_t>(cb, ce)};
}

private:
  gasal_subst_scores subst_{};
  Parameters* args_{nullptr};
  gasal_gpu_storage_v stor_v_{};
  int max_q_{0}, max_t_{0};
};

PYBIND11_MODULE(gasalwrap, m) {
  py::class_<PAlign>(m, "PAlign")
      .def_readonly("score", &PAlign::score)
      .def_readonly("q_beg", &PAlign::q_beg)
      .def_readonly("q_end", &PAlign::q_end)
      .def_readonly("s_beg", &PAlign::s_beg)
      .def_readonly("s_end", &PAlign::s_end)
      .def_readonly("cigar", &PAlign::cigar);

  py::class_<GasalAligner>(m, "GasalAligner")
      .def(py::init<int,int,int,int,int,int>(),
           py::arg("match"), py::arg("mismatch"),
           py::arg("gap_open"), py::arg("gap_extend"),
           py::arg("max_q") = 2048, py::arg("max_t") = 8192)
      .def("align", &GasalAligner::align);
}


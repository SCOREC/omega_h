#include "Omega_h_library.hpp"
#include "Omega_h_array_ops.hpp"
#include "Omega_h_sort.hpp"
#include "Omega_h_file.hpp"
#include "Omega_h_atomics.hpp"
#include "Omega_h_for.hpp"
#include <fstream>

struct CompareKeySets {
  Omega_h::Write<Omega_h::LO> const* keys_;
  int N;
  CompareKeySets(Omega_h::Write<Omega_h::LO> const* keys, int n) {
    keys_ = keys;
    N = n;
}
  OMEGA_H_INLINE bool operator()(const Omega_h::LO& a, const Omega_h::LO& b) const {
    for (int i = 0; i < N; ++i) {
      Omega_h::LO x = (*keys_)[a * N + i];
      Omega_h::LO y = (*keys_)[b * N + i];
      if (x != y) return x < y;
    }
    return false;
  }
};

int main(int argc, char** argv) {
  using namespace Omega_h;
  auto lib = Library(&argc, &argv);
  auto world = lib.world();
  {
    LOs a({0, 2, 0, 1});
    LOs perm = sort_by_keys(a,1);
    LOs gold({0, 2, 3, 1});
    auto perm_hr = HostRead<LO>(perm);
    auto gold_hr = HostRead<LO>(gold);
    for(int j=0; j<perm_hr.size(); j++) {
      fprintf(stderr, "%d %d %d\n", j, perm_hr[j], gold_hr[j]);
    }
    OMEGA_H_CHECK(perm == gold);
  }
  {
    LOs a({0, 2, 0, 1});
    LOs perm = sort_by_keys(a, 2);
    OMEGA_H_CHECK(perm == LOs({1, 0}));
  }
  {
    LOs a({0, 2, 1, 1});
    LOs perm = sort_by_keys(a, 2);
    OMEGA_H_CHECK(perm == LOs({0, 1}));
  }
  {
    LOs a({1, 2, 3, 1, 2, 2, 3, 0, 0});
    LOs perm = sort_by_keys(a, 3);
    OMEGA_H_CHECK(perm == LOs({1, 0, 2}));
  }
  {
    for(int i=0; i<3; i++) {
      fprintf(stderr, "large test %d\n", i);
      Write<LO> random_keys();
      auto n = 1;     
      //auto n = divide_no_remainder(random_keys.size(), i);
      Write<LO> gold_perm(n, 0, 1);
      LO* begin = gold_perm.data();
      LO* end = gold_perm.data() + n;
      std::stable_sort(begin, end, CompareKeySets(&random_keys, i));  
    }
  }
  return 0;
}

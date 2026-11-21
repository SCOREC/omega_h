#ifndef OMEGA_H_EMBEDDED_FIXTURE_TEST_HPP
#define OMEGA_H_EMBEDDED_FIXTURE_TEST_HPP

#include "Omega_h_build.hpp"
#include "Omega_h_class.hpp"
#include "Omega_h_mark.hpp"
#include <cmath>

namespace embedded_test {
using namespace Omega_h;

inline void check(bool value, char const* message) {
  if (!value) Omega_h_fail("embedded test: %s\n", message);
}

inline Reals embed(CoordsView<> input, Int sdim) {
  auto src_storage = copy_coords_to_host(input);
  auto src = make_coords_view<>(src_storage, input.extent(1));
  auto tdim = input.extent(1);
  HostWrite<Real> dst(input.extent(0) * sdim);
  // Rx(30), Ry(45), Rz(60), followed by a translation.
  auto const a = OMEGA_H_PI / 6;
  auto const b = OMEGA_H_PI / 4;
  auto const c = OMEGA_H_PI / 3;
  for (LO v = 0; v < input.extent(0); ++v) {
    Real x = src(v, 0);
    Real y = tdim == 2 ? src(v, 1) : 0;
    if (sdim == 2) {
      dst[v * 2] = std::cos(c) * x + 0.3;
      dst[v * 2 + 1] = std::sin(c) * x - 1.2;
    } else {
      Real yy = std::cos(a) * y;
      Real z = std::sin(a) * y;
      Real xx = std::cos(b) * x + std::sin(b) * z;
      dst[v * 3] = std::cos(c) * xx - std::sin(c) * yy + 0.3;
      dst[v * 3 + 1] = std::sin(c) * xx + std::cos(c) * yy - 1.2;
      dst[v * 3 + 2] = -std::sin(b) * x + std::cos(b) * z + 2.5;
    }
  }
  return dst.write();
}

inline Mesh plane(Library& lib, LO n = 8) {
  auto base = build_box(lib.self(), OMEGA_H_SIMPLEX, 1, 1, 0, n, n, 0);
  Mesh result(&lib);
  build_from_elems_and_coords(&result, OMEGA_H_SIMPLEX, 2,
      base.ask_verts_of(2), embed(base.coords_view(), 3), 3);
  return result;
}

inline void classify(Mesh& mesh) {
  classify_elements(&mesh);
  classify_sides_by_exposure(&mesh, mark_exposed_sides(&mesh));
  finalize_classification(&mesh);
}
}  // namespace embedded_test
#endif

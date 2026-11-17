#include <sstream>
#include <string>

#include "Omega_h_array_ops.hpp"
#include "Omega_h_build.hpp"
#include "Omega_h_compare.hpp"
#include "Omega_h_file.hpp"
#include "Omega_h_mdspan.hpp"

using namespace Omega_h;

static void require(bool condition) {
  if (!condition) Omega_h_fail("embedded mesh test failed\n");
}

#ifdef OMEGA_H_THROW
template <class F> static void rejects(F f) {
  bool caught = false;
  try { f(); } catch (Omega_h::exception const&) { caught = true; }
  require(caught);
}
#endif

int main(int argc, char** argv) {
  Library lib(&argc, &argv);
  Real storage[6] = {};
  mdspan<Real, extents<LO, 2, 3>> span(storage);
  span(1, 2) = 7;
  require(storage[5] == 7);
  for (Int dim = 1; dim <= 3; ++dim) {
    auto ordinary = build_box(lib.self(), OMEGA_H_SIMPLEX, 1, 1, 1,
        2, dim >= 2 ? 2 : 0, dim == 3 ? 2 : 0);
    require(ordinary.tdim() == dim && ordinary.space_dim() == dim);
  }
  Mesh embedded(&lib);
  auto coords = Reals({0.3, -1.2, 2.5, 1.3, -1.2, 2.5, 0.3, -0.2, 3.5});
  build_from_elems_and_coords(&embedded, OMEGA_H_SIMPLEX, 2,
      LOs({0, 1, 2}), coords, 3);
  require(embedded.space_dim() == 3);
  auto copied = embedded.copy_meta();
  require(copied.space_dim() == 3);
  copied.set_space_dim(2);
  require(compare_meshes(&embedded, &copied, MeshCompareOpts(), false, false) == OMEGA_H_DIFF);
  std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
  binary::write(stream, &embedded);
  Mesh loaded(&lib);
  loaded.set_comm(lib.self());
  binary::read(stream, &loaded, binary::latest_version);
  require(loaded.space_dim() == 3 && loaded.coords() == coords);
  auto old = binary::read(argv[1], lib.self());
  require(old.space_dim() == old.dim());
#ifdef OMEGA_H_THROW
  rejects([&] { embedded.set_space_dim(1); });
  rejects([&] { embedded.set_space_dim(4); });
  rejects([&] { embedded.set_space_dim(2); });
  rejects([&] {
    Mesh bad(&lib);
    build_from_elems_and_coords(&bad, OMEGA_H_SIMPLEX, 2, LOs({0,1,2}), coords, 0);
  });
  rejects([&] {
    Mesh bad(&lib);
    build_from_elems_and_coords(&bad, OMEGA_H_SIMPLEX, 2, LOs({0,1,2}), Reals({1,2,3,4}), 3);
  });
#endif
}

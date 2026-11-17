#include <sstream>
#include <string>

#include "Omega_h_array_ops.hpp"
#include "Omega_h_build.hpp"
#include "Omega_h_compare.hpp"
#include "Omega_h_file.hpp"
#include "Omega_h_mdspan.hpp"
#include "Omega_h_for.hpp"

using namespace Omega_h;
#include "embedded_gates_test.hpp"

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

OMEGA_H_ALLOW_DEPRECATED_BEGIN
static void check_deprecated(Mesh& ordinary, Mesh& embedded) {
  require(ordinary.coords() == ordinary.coords_view().raw());
  ordinary.set_coords(ordinary.coords_view().raw());
#ifdef OMEGA_H_THROW
  rejects([&] { (void)embedded.coords(); });
  rejects([&] { embedded.set_coords(embedded.coords_view().raw()); });
#else
  (void)embedded;
#endif
}
OMEGA_H_ALLOW_DEPRECATED_END

int main(int argc, char** argv) {
  Library lib(&argc, &argv);
  if (argc > 2 && std::string(argv[2]) == "bad_static") {
    CoordsView<2> invalid(Reals({1, 2, 3}), 3);
    return 0;
  }
  if (argc > 2 && std::string(argv[2]) == "bad_index") {
    CoordsView<3> invalid(Reals({1, 2, 3}), 3);
    (void)invalid(0, 3);
    return 0;
  }
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
  auto ordinary = build_box(lib.self(), OMEGA_H_SIMPLEX, 1, 1, 0, 2, 2, 0);
  check_deprecated(ordinary, embedded);
#ifdef OMEGA_H_THROW
  check_embedded_gates(embedded, ordinary);
#endif
  auto fixed = embedded.coords_view<3>();
  CoordsView<> dynamic = fixed;
  auto roundtrip = dynamic.as<3>();
  require(dynamic.extent(0) == 3 && dynamic.extent(1) == 3);
  Write<Real> output(coords.size());
  auto check_indexing = OMEGA_H_LAMBDA(LO v) {
    auto gathered = gather_vectors<1>(fixed, Few<LO, 1>{v});
    for (Int d = 0; d < 3; ++d) {
      output[v * 3 + d] = gathered[0][d];
      OMEGA_H_CHECK(dynamic(v, d) == roundtrip(v, d));
    }
  };
  parallel_for(embedded.nverts(), check_indexing);
  require(Reals(output) == coords);
  CoordsWriteView<3> writable(output, 3);
  auto write_coords = OMEGA_H_LAMBDA(LO v) {
    for (Int d = 0; d < 3; ++d) writable(v, d) = fixed(v, d);
  };
  parallel_for(embedded.nverts(), write_coords);
  embedded.set_coords(CoordsView<>(writable));
  require(embedded.coords_view().raw() == coords);

  auto copied = embedded.copy_meta();
  require(copied.space_dim() == 3);
  copied.set_space_dim(2);
  require(compare_meshes(&embedded, &copied, MeshCompareOpts(), false, false) == OMEGA_H_DIFF);
  std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
  binary::write(stream, &embedded);
  Mesh loaded(&lib);
  loaded.set_comm(lib.self());
  binary::read(stream, &loaded, binary::latest_version);
  require(loaded.space_dim() == 3 && loaded.coords_view().raw() == coords);
  auto old = binary::read(argv[1], lib.self());
  require(old.space_dim() == old.dim());
#ifdef OMEGA_H_THROW
  rejects([&] { embedded.coords_view<2>(); });
  rejects([&] { CoordsView<3> bad(Reals({1, 2, 3, 4}), 3); });
  rejects([&] { embedded.set_coords(CoordsView<>(Reals({1, 2, 3}), 3)); });
  rejects([&] { embedded.set_coords(CoordsView<>(Reals({1, 2, 3, 4, 5, 6}), 2)); });
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

#include "coords_test.hpp"
#include <sstream>
#include <string>
#include <iostream>

#include "Omega_h_array_ops.hpp"
#include "Omega_h_build.hpp"
#include "Omega_h_compare.hpp"
#include "Omega_h_file.hpp"
#include "Omega_h_mdspan.hpp"
#include "Omega_h_for.hpp"

using namespace Omega_h;
#include "embedded_gates_test.hpp"
#include "embedded_fixture_test.hpp"
#include "Omega_h_base64.hpp"
#include <vector>

static void require_impl(bool condition, int line) {
  if (!condition) Omega_h_fail("embedded mesh test failed at line %d\n", line);
}
#define require(condition) require_impl(condition, __LINE__)

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

static void check_geometry(Library& lib) {
  for (Int tdim = 1; tdim <= 2; ++tdim) {
    auto base = build_box(lib.self(), OMEGA_H_SIMPLEX, 1, 1, 0, 4, tdim == 2 ? 4 : 0, 0);
    for (Int sdim = tdim + 1; sdim <= 3; ++sdim) {
      Mesh embedded(&lib);
      build_from_elems_and_coords(&embedded, OMEGA_H_SIMPLEX, tdim,
          base.ask_verts_of(tdim), embedded_test::embed(base.coords_view(), sdim), sdim);
      embedded_test::classify(embedded);
      require(are_close(embedded.ask_sizes(), base.ask_sizes(), 1e-12));
      auto lengths = measure_edges_real(&embedded);
      HostCoordsView<> original(base.coords_view());
      HostRead<Real> actual(lengths);
      HostRead<LO> edges(embedded.ask_verts_of(EDGE));
      for (LO e = 0; e < embedded.nedges(); ++e) {
        Real squared = 0;
        for (Int d = 0; d < tdim; ++d) {
          auto delta = original(edges[e * 2], d) - original(edges[e * 2 + 1], d);
          squared += delta * delta;
        }
        require(std::abs(actual[e] - std::sqrt(squared)) < 1e-12);
      }
      require(are_close(embedded.ask_lengths(), lengths, 1e-12));
      require(std::abs(get_sum(embedded.ask_sizes()) - 1.0) < 1e-12);
    }
  }
  // Two unit right triangles folded along their shared edge: total area is 1.
  Mesh folded(&lib);
  build_from_elems_and_coords(&folded, OMEGA_H_SIMPLEX, 2,
      LOs({0, 1, 2, 1, 0, 3}),
      Reals({0.3, -1.2, 2.5, 1.3, -1.2, 2.5, 0.3, -0.2, 2.5, 0.3, -1.2, 3.5}), 3);
  require(std::abs(get_sum(folded.ask_sizes()) - 1.0) < 1e-12);
  std::ostringstream out;
  vtk::write_vtu(out, &folded, 2, TagSet{}, false);
  auto text = out.str();
  auto points = text.find("<Points>");
  auto begin = text.find('>', text.find("<DataArray", points)) + 1;
  auto end = text.find("</DataArray>", begin);
  require(points != std::string::npos && end != std::string::npos);
  std::istringstream encoded_stream(text.substr(begin, end - begin));
  encoded_stream >> std::ws;
  auto encoded = base64::read_encoded(encoded_stream);
  auto header_size = base64::encoded_size(sizeof(std::uint64_t));
  require(encoded.size() >= header_size);
  std::uint64_t bytes = 0;
  base64::decode(encoded.substr(0, header_size), &bytes, sizeof(bytes));
  if (!is_little_endian_cpu()) binary::swap_bytes(bytes);
  require(bytes == std::uint64_t(folded.nverts()) * 3 * sizeof(Real));
  require(encoded.size() == header_size + base64::encoded_size(bytes));
  std::vector<Real> decoded(folded.nverts() * 3);
  base64::decode(encoded.substr(header_size), decoded.data(), bytes);
  if (!is_little_endian_cpu()) for (auto& value : decoded) binary::swap_bytes(value);
  HostCoordsView<3> expected(folded.coords_view<3>());
  for (LO v = 0; v < expected.extent(0); ++v)
    for (Int d = 0; d < 3; ++d) require(decoded[v * 3 + d] == expected(v, d));
}

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
  auto mode = argc > 2 ? std::string(argv[2]) : std::string();
  if (mode == "bad_dimension") {
    Mesh invalid(&lib);
    build_from_elems_and_coords(&invalid, OMEGA_H_SIMPLEX, 2, LOs({0,1,2}), Reals({1,2,3}), 0);
    return 0;
  }
  if (mode == "bad_length") { CoordsView<3> invalid(Reals({1,2,3,4}), 3); return 0; }
  if (mode == "bad_space_dim") {
    Mesh invalid(&lib); invalid.set_dim(2); invalid.set_space_dim(1); return 0;
  }
  if (mode == "bad_set_extent" || mode == "bad_tag_width") {
    auto invalid = build_box(lib.self(), OMEGA_H_SIMPLEX, 1, 1, 0, 2, 2, 0);
    if (mode == "bad_tag_width") invalid.set_space_dim(3);
    else invalid.set_coords(CoordsView<>(Reals({1,2}), 2));
    return 0;
  }
  if (mode == "geometry") {
    check_geometry(lib);
    std::cout << "embedded geometry checks passed\n";
    return 0;
  }
  auto owned = [] {
    Write<Real> values(6, 7.0);
    CoordsView<3> fixed(Reals(values), 3);
    return CoordsView<>(fixed);
  }();
  HostCoordsView<> owned_host(owned);
  Real sum = 0;
  for (LO v = 0; v < owned_host.extent(0); ++v)
    for (Int d = 0; d < owned_host.extent(1); ++d) sum += owned_host(v, d);
  require(sum == 42.0);
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
  Write<Real> output(coords.size()), dynamic_output(coords.size()), span_output(coords.size());
  auto borrowed = roundtrip.span();
  auto check_indexing = OMEGA_H_LAMBDA(LO v) {
    auto gathered = gather_vectors<1>(fixed, Few<LO, 1>{v});
    for (Int d = 0; d < 3; ++d) {
      output[v * 3 + d] = gathered[0][d];
      dynamic_output[v * 3 + d] = dynamic(v, d);
      span_output[v * 3 + d] = borrowed(v, d);
    }
  };
  parallel_for(embedded.nverts(), check_indexing);
  require(Reals(output) == coords);
  require(Reals(dynamic_output) == coords && Reals(span_output) == coords);
  Write<Real> writable_data(coords.size(), 0.0);
  CoordsWriteView<3> writable(writable_data, 3);
  auto write_coords = OMEGA_H_LAMBDA(LO v) {
    for (Int d = 0; d < 3; ++d) writable(v, d) = fixed(v, d);
  };
  parallel_for(embedded.nverts(), write_coords);
  embedded.set_coords(CoordsView<>(writable));
  require(test::coords_equal(embedded.coords_view<3>(), CoordsView<3>(coords, 3)));

  auto copied = embedded.copy_meta();
  require(copied.space_dim() == 3);
  copied.set_space_dim(2);
  require(compare_meshes(&embedded, &copied, MeshCompareOpts(), false, false) == OMEGA_H_DIFF);
  std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
  binary::write(stream, &embedded);
  Mesh loaded(&lib);
  loaded.set_comm(lib.self());
  binary::read(stream, &loaded, binary::latest_version);
  require(loaded.space_dim() == 3 && test::coords_equal(loaded.coords_view<3>(), CoordsView<3>(coords, 3)));
  auto old = binary::read(argv[1], lib.self());
  require(old.space_dim() == old.dim());
#ifdef OMEGA_H_THROW
  rejects([&] { embedded.coords_view<2>(); });
  rejects([&] { dynamic.as<2>(); });
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
  std::cout << "embedded unit checks passed\n";
}

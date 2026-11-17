#ifndef OMEGA_H_COORDS_TEST_HPP
#define OMEGA_H_COORDS_TEST_HPP

#include "Omega_h_mesh.hpp"
#include "Omega_h_for.hpp"
#include "Omega_h_element.hpp"

namespace Omega_h {
namespace test {
template <std::size_t A, std::size_t B>
bool coords_equal(CoordsView<A> const& a, CoordsView<B> const& b) {
  if (a.extent(0) != b.extent(0) || a.extent(1) != b.extent(1)) return false;
  HostCoordsView<A> ha(a);
  HostCoordsView<B> hb(b);
  for (LO v = 0; v < a.extent(0); ++v)
    for (Int d = 0; d < a.extent(1); ++d)
      if (ha(v, d) != hb(v, d)) return false;
  return true;
}

template <Int sdim>
CoordsView<sdim> element_centroids(Mesh& mesh) {
  auto coords = mesh.coords_view<sdim>();
  auto vertices = mesh.ask_verts_of(mesh.dim());
  auto degree = element_degree(mesh.family(), mesh.dim(), VERT);
  Write<Real> result(mesh.nelems() * sdim);
  auto average = OMEGA_H_LAMBDA(LO e) {
    auto value = zero_vector<sdim>();
    for (Int i = 0; i < degree; ++i)
      value += get_vector<sdim>(coords, vertices[e * degree + i]);
    set_vector<sdim>(result, e, value / Real(degree));
  };
  parallel_for(mesh.nelems(), average);
  return CoordsView<sdim>(Reals(result), sdim);
}
}  // namespace test
}  // namespace Omega_h
#endif

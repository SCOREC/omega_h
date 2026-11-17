#include "Omega_h_shape.hpp"

#include "Omega_h_for.hpp"
#include "Omega_h_dispatch.hpp"

namespace Omega_h {

template <Int mesh_dim, Int metric_dim>
Reals measure_edges_metric_tmpl(Mesh* mesh, LOs a2e, Reals metrics) {
  MetricEdgeLengths<mesh_dim, metric_dim> measurer(mesh, metrics);
  auto ev2v = mesh->ask_verts_of(EDGE);
  auto na = a2e.size();
  Write<Real> lengths(na);
  auto f = OMEGA_H_LAMBDA(LO a) {
    auto e = a2e[a];
    auto v = gather_verts<2>(ev2v, e);
    lengths[a] = measurer.measure(v);
  };
  parallel_for(na, f, "measure_edges");
  return lengths;
}

Reals measure_edges_metric(Mesh* mesh, LOs a2e, Reals metrics) {
  ::Omega_h::detail::require_same_tdim_sdim(*mesh, __func__);
  if (a2e.size() == 0) return Reals({});
  auto metric_dim = get_metrics_dim(mesh->nverts(), metrics);
  if (mesh->dim() == 3 && metric_dim == 3) {
    return measure_edges_metric_tmpl<3, 3>(mesh, a2e, metrics);
  } else if (mesh->dim() == 2 && metric_dim == 2) {
    return measure_edges_metric_tmpl<2, 2>(mesh, a2e, metrics);
  } else if (mesh->dim() == 3 && metric_dim == 1) {
    return measure_edges_metric_tmpl<3, 1>(mesh, a2e, metrics);
  } else if (mesh->dim() == 2 && metric_dim == 1) {
    return measure_edges_metric_tmpl<2, 1>(mesh, a2e, metrics);
  } else if (mesh->dim() == 1 && metric_dim == 1) {
    return measure_edges_metric_tmpl<1, 1>(mesh, a2e, metrics);
  }
  OMEGA_H_NORETURN(Reals());
}

Reals measure_edges_metric(Mesh* mesh, LOs a2e) {
  ::Omega_h::detail::require_same_tdim_sdim(*mesh, __func__);
  return measure_edges_metric(mesh, a2e, mesh->get_array<Real>(VERT, "metric"));
}

Reals measure_edges_metric(Mesh* mesh, Reals metrics) {
  ::Omega_h::detail::require_same_tdim_sdim(*mesh, __func__);
  return measure_edges_metric(mesh, LOs(mesh->nedges(), 0, 1), metrics);
}

Reals measure_edges_metric(Mesh* mesh) {
  ::Omega_h::detail::require_same_tdim_sdim(*mesh, __func__);
  return measure_edges_metric(mesh, mesh->get_array<Real>(VERT, "metric"));
}

template <Int sdim, Int edim>
Reals measure_ents_real_tmpl(Mesh* mesh, LOs a2e, CoordsView<sdim> coords) {
  OMEGA_H_TIME_FUNCTION;

  auto ev2v = mesh->ask_verts_of(edim);
  auto na = a2e.size();
  Write<Real> sizes(na);
  auto f = OMEGA_H_LAMBDA(LO a) {
    auto e = a2e[a];
    auto v = gather_verts<edim + 1>(ev2v, e);
    auto p = gather_vectors<edim + 1>(coords, v);
    sizes[a] = real_simplex_size<sdim, edim>(p);
  };
  parallel_for(na, f, "measure_ents_real");
  return sizes;
}

static Reals measure_ents_real_view(Mesh* mesh, Int ent_dim, LOs a2e, CoordsView<> coords) {
  if (ent_dim < 1 || ent_dim > mesh->dim() || coords.extent(0) != mesh->nverts() ||
      coords.extent(1) != mesh->space_dim())
    Omega_h_fail("measure_ents_real: entity dimension or coordinate extents do not match mesh\n");
  return detail::dispatch_sdim_edim(mesh->space_dim(), ent_dim, [&](auto sdim, auto edim) {
    return measure_ents_real_tmpl<sdim.value, edim.value>(mesh, a2e, coords.as<sdim.value>());
  });
}

Reals measure_ents_real(Mesh* mesh, Int ent_dim, LOs a2e, Reals coords) {
  return measure_ents_real_view(mesh, ent_dim, a2e, CoordsView<>(coords, mesh->space_dim()));
}

Reals measure_elements_real(Mesh* mesh, LOs a2e) {
  return measure_ents_real_view(mesh, mesh->dim(), a2e, mesh->coords_view());
}

Reals measure_elements_real(Mesh* mesh) {
  return measure_elements_real(mesh, LOs(mesh->nelems(), 0, 1));
}

template <Int sdim>
static Reals measure_edges_real_tmpl(Mesh* mesh, LOs a2e) {
  RealEdgeLengths<sdim> measurer(mesh);
  auto ev2v = mesh->ask_verts_of(EDGE);
  Write<Real> lengths(a2e.size());
  auto f = OMEGA_H_LAMBDA(LO a) {
    lengths[a] = measurer.measure(gather_verts<2>(ev2v, a2e[a]));
  };
  parallel_for(a2e.size(), f, "measure_edges_real");
  return lengths;
}

Reals measure_edges_real(Mesh* mesh, LOs a2e) {
  return detail::dispatch_sdim_edim(mesh->space_dim(), EDGE, [&](auto sdim, auto) {
    return measure_edges_real_tmpl<sdim.value>(mesh, a2e);
  });
}

Reals measure_edges_real(Mesh* mesh) {
  return measure_edges_real(mesh, LOs(mesh->nedges(), 0, 1));
}

}  // end namespace Omega_h

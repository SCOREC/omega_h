#include "Omega_h_adapt.hpp"

#include "Omega_h_array_ops.hpp"
#include "Omega_h_metric.hpp"
#include "Omega_h_shape.hpp"
#include "Omega_h_for.hpp"

#include <iostream>

namespace Omega_h {

static void check_okay(Mesh* mesh, AdaptOpts const& opts) {
  auto minq = min_fixable_quality(mesh, opts);
  if (minq < opts.min_quality_allowed) {
    Omega_h_fail("minimum element quality %f < minimum allowed quality %f\n",
        minq, opts.min_quality_allowed);
  }
  auto maxl = mesh->max_length();
  if (maxl > opts.max_length_allowed) {
    Omega_h_fail("maximum edge length %f > maximum allowed length %f\n", maxl,
        opts.max_length_allowed);
  }
}

static bool okay(Mesh* mesh, AdaptOpts const& opts) {
  auto minq = min_fixable_quality(mesh, opts);
  if (minq < opts.min_quality_allowed) {
    return false;
  }
  auto maxl = mesh->max_length();
  if (maxl > opts.max_length_allowed) {
    return false;
  }
  return true;
}

bool warp_to_limit(
    Mesh* mesh, AdaptOpts const& opts, bool exit_on_stall, Int max_niters) {
  if (!mesh->has_tag(VERT, "warp")) return false;
  check_okay(mesh, opts);
  auto coords = mesh->coords();
  auto warp = mesh->get_array<Real>(VERT, "warp");
  mesh->set_coords(add_each(coords, warp));
  if (okay(mesh, opts)) {
    if (opts.verbosity >= EACH_REBUILD && can_print(mesh)) {
      std::cout << "warp_to_limit completed in one step\n";
    }
    mesh->remove_tag(VERT, "warp");
    return true;
  }
  auto remainder = Reals(warp.size(), 0.0);
  Int i = 0;
  Real factor = 1.0;
  do {
    ++i;
    if (i > max_niters) {
      if (exit_on_stall) {
        if (can_print(mesh)) {
          std::cout << "warp_to_limit stalled, dropping warp field and "
                       "continuing anyway\n";
        }
        mesh->remove_tag(VERT, "warp");
        return true;
      }
      Omega_h_fail(
          "warp step %d : Omega_h is probably unable to satisfy"
          " this warp under this size field\n"
          "min quality %.2e max length %.2e\n",
          i, min_fixable_quality(mesh, opts), mesh->max_length());
    }
    auto half_warp = divide_each_by(warp, 2.0);
    factor /= 2.0;
    warp = half_warp;
    remainder = add_each(remainder, half_warp);
    mesh->set_coords(add_each(coords, warp));
  } while (!okay(mesh, opts));
  if (opts.verbosity >= EACH_REBUILD && can_print(mesh)) {
    std::cout << "warp_to_limit moved by factor " << factor << '\n';
  }
  mesh->set_tag(VERT, "warp", remainder);
  return true;
}

template <Int dim>
bool is_metric_spd_dim(Reals metrics) {
  auto n = divide_no_remainder(metrics.size(), symm_ncomps(dim));
  auto isDegen = Write<LO>(n);
  auto f = OMEGA_H_LAMBDA(LO i) {
    auto m = get_symm<dim>(metrics, i);
    auto m_dc = decompose_eigen(m);
    Few<Int, dim> m_ew_is_degen;
    for (Int j = 0; j < dim; ++j) {
      // if EPSILON is 1e-10, this corresponds roughly to edges
      // of absolute length 1e5 or more being considered "infinite"
      m_ew_is_degen[j] = m_dc.l[j] < OMEGA_H_EPSILON;
    }
    auto nm_degen_ews = reduce(m_ew_is_degen, plus<Int>());
    isDegen[i] = (nm_degen_ews > 0);
  };
  parallel_for(n, f, "mark_degen_metrics");
  const int numDegen = get_sum(read(isDegen));
  return (numDegen == 0);
}

bool is_metric_spd(Mesh* mesh, Reals metrics) {
  if( mesh->dim() == 1)
    return is_metric_spd_dim<1>(metrics); // is this supported?
  else if( mesh->dim() == 2)
    return is_metric_spd_dim<2>(metrics);
  else if( mesh->dim() == 3)
    return is_metric_spd_dim<3>(metrics);
  else
    return false;
}

bool approach_metric(Mesh* mesh, AdaptOpts const& opts, Real min_step) {
  auto name = "metric";
  auto target_name = "target_metric";
  if (!mesh->has_tag(VERT, target_name)) return false;
  check_okay(mesh, opts);
  auto orig = mesh->get_array<Real>(VERT, name);
  auto target = mesh->get_array<Real>(VERT, target_name);
  mesh->set_tag(VERT, name, target);
  if (okay(mesh, opts)) {
    mesh->remove_tag(VERT, target_name);
    return true;
  }
  Real factor = 1.0;
  do {
    factor /= 2.0;
    if (factor < min_step) {
      auto minq = min_fixable_quality(mesh, opts);
      auto maxl = mesh->max_length();
      if (can_print(mesh)) {
        if (minq < opts.min_quality_allowed) {
          std::cerr << "Metric approach has stalled with minimum quality "
                    << minq << " < " << opts.min_quality_allowed << "\n";
          std::cerr << "Decreasing \"Min Quality Allowed\" may help, but "
                       "otherwise the metric is likely not satisfiable\n";
        }
        if (maxl > opts.max_length_allowed) {
          std::cerr << "Metric approach has stalled with maximum length "
                    << maxl << " > " << opts.max_length_allowed << "\n";
          std::cerr << "Increasing \"Max Length Allowed\" will probably fix "
                       "this, otherwise the metric is likely not satisfiable\n";
        }
      }
      Omega_h_fail("Metric approach has stalled at step size = %f < %f.\n",
          factor, min_step);
    }
    OMEGA_H_CHECK(is_metric_spd(mesh,orig));
    OMEGA_H_CHECK(is_metric_spd(mesh,target));
    auto current =
        interpolate_between_metrics(mesh->nverts(), orig, target, factor);
    mesh->set_tag(VERT, name, current);
  } while (!okay(mesh, opts));
  if (opts.verbosity >= EACH_REBUILD && can_print(mesh)) {
    std::cout << "approach_metric moved by factor " << factor << '\n';
  }
  return true;
}

}  // end namespace Omega_h

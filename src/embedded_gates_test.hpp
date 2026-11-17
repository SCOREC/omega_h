#ifndef OMEGA_H_EMBEDDED_GATES_TEST_HPP
#define OMEGA_H_EMBEDDED_GATES_TEST_HPP
#include "Omega_h_adapt.hpp"
#include "Omega_h_amr.hpp"
#include "Omega_h_coarsen.hpp"
#include "Omega_h_confined.hpp"
#include "Omega_h_conserve.hpp"
#include "Omega_h_metric.hpp"
#include "Omega_h_quality.hpp"
#include "Omega_h_recover.hpp"
#include "Omega_h_refine.hpp"
#include "Omega_h_surface.hpp"
#include "Omega_h_swap.hpp"
#include "Omega_h_class.hpp"
#include <string>
#ifdef OMEGA_H_THROW
template <class F>
static void expect_embedded_gate(F f, char const* name) {
  try { f(); }
  catch (Omega_h::exception const& e) {
    if (std::string(e.what()).find("space_dim (3) != dim (2)") == std::string::npos)
      Omega_h_fail("%s: unexpected failure: %s\n", name, e.what());
    return;
  }
  Omega_h_fail("%s: embedded mesh was accepted\n", name);
}

static void check_embedded_gates(Mesh& mesh, Mesh& ordinary) {
  AdaptOpts opts(2);
  expect_embedded_gate([&] { min_fixable_quality(&mesh, opts); }, "min_fixable_quality");
  expect_embedded_gate([&] { adapt(&mesh, opts); }, "adapt");
  expect_embedded_gate([&] { print_adapt_status(&mesh, opts); }, "print_adapt_status");
  expect_embedded_gate([&] { print_adapt_histograms(&mesh, opts); }, "print_adapt_histograms");
  expect_embedded_gate([&] { fix_momentum_velocity_verts(&mesh, std::vector<ClassPair>{}, Int{}); }, "fix_momentum_velocity_verts");
  expect_embedded_gate([&] { warp_to_limit(&mesh, opts, bool{}, Int{}); }, "warp_to_limit");
  expect_embedded_gate([&] { approach_metric(&mesh, opts, Real{}); }, "approach_metric");
  expect_embedded_gate([&] { generate_metrics(&mesh, MetricInput{}); }, "generate_metrics");
  expect_embedded_gate([&] { add_metric_tag(&mesh, Reals{}, std::string{}); }, "add_metric_tag");
  expect_embedded_gate([&] { generate_metric_tag(&mesh, MetricInput{}); }, "generate_metric_tag");
  expect_embedded_gate([&] { generate_target_metric_tag(&mesh, MetricInput{}); }, "generate_target_metric_tag");
  expect_embedded_gate([&] { add_implied_metric_tag(&mesh); }, "add_implied_metric_tag");
  expect_embedded_gate([&] { add_implied_isos_tag(&mesh); }, "add_implied_isos_tag");
  expect_embedded_gate([&] { add_implied_metric_based_on_target(&mesh); }, "add_implied_metric_based_on_target");
  expect_embedded_gate([&] { fix(&mesh, opts, Omega_h_Isotropy{}, bool{}); }, "fix");
  expect_embedded_gate([&] { fix_for_given_metric(&mesh, opts, bool{}); }, "fix_for_given_metric");
  expect_embedded_gate([&] { grade_fix_adapt(&mesh, opts, Reals{}, bool{}); }, "grade_fix_adapt");
  expect_embedded_gate([&] { refine_by_size(&mesh, opts); }, "refine_by_size");
  expect_embedded_gate([&] { mark_dead_ents(&mesh, LOs{}, Read<I8>{}); }, "mark_dead_ents");
  expect_embedded_gate([&] { find_coarsen_domains(&mesh, LOs{}, Int{}, Read<I8>{}); }, "find_coarsen_domains");
  expect_embedded_gate([&] { coarsen_qualities(&mesh, LOs{}, Read<I8>{}); }, "coarsen_qualities");
  expect_embedded_gate([&] { filter_coarsen_improve(&mesh, LOs{}, Read<I8>{}, Reals{}); }, "filter_coarsen_improve");
  expect_embedded_gate([&] { prevent_coarsen_overshoot(&mesh, Real{}, LOs{}, Read<I8>{}); }, "prevent_coarsen_overshoot");
  expect_embedded_gate([&] { prevent_coarsen_flip(&mesh, LOs{}, Bytes{}); }, "prevent_coarsen_flip");
  expect_embedded_gate([&] { choose_rails(&mesh, LOs{}, Read<I8>{}, Reals{}, nullptr, nullptr, nullptr); }, "choose_rails");
  expect_embedded_gate([&] { find_rails(&mesh, LOs{}, Read<GO>{}, nullptr, nullptr); }, "find_rails");
  expect_embedded_gate([&] { get_verts_onto(&mesh, LOs{}, Read<I8>{}); }, "get_verts_onto");
  expect_embedded_gate([&] { coarsen_topology(&mesh, LOs{}, Int{}, Adj{}, LOs{}); }, "coarsen_topology");
  expect_embedded_gate([&] { coarsen_by_size(&mesh, opts); }, "coarsen_by_size");
  expect_embedded_gate([&] { coarsen_slivers(&mesh, opts); }, "coarsen_slivers");
  expect_embedded_gate([&] { swap_part1(&mesh, opts); }, "swap_part1");
  expect_embedded_gate([&] { filter_swap_improve(&mesh, LOs{}, Reals{}); }, "filter_swap_improve");
  expect_embedded_gate([&] { swap_edges(&mesh, opts); }, "swap_edges");
  expect_embedded_gate([&] { get_metric_dim(&mesh); }, "get_metric_dim");
  expect_embedded_gate([&] { get_mident_metrics(&mesh, Int{}, LOs{}, Reals{}, bool{}); }, "get_mident_metrics");
  expect_embedded_gate([&] { get_mident_metrics(&mesh, Int{}, Reals{}, bool{}); }, "get_mident_metrics");
  expect_embedded_gate([&] { project_metrics(&mesh, Reals{}); }, "project_metrics");
  expect_embedded_gate([&] { get_pure_implied_isos(&mesh); }, "get_pure_implied_isos");
  expect_embedded_gate([&] { get_implied_isos(&mesh); }, "get_implied_isos");
  expect_embedded_gate([&] { get_element_implied_length_metrics(&mesh); }, "get_element_implied_length_metrics");
  expect_embedded_gate([&] { get_pure_implied_metrics(&mesh); }, "get_pure_implied_metrics");
  expect_embedded_gate([&] { get_implied_metrics(&mesh); }, "get_implied_metrics");
  expect_embedded_gate([&] { limit_metric_gradation(&mesh, Reals{}, Real{}, Real{}, bool{}); }, "limit_metric_gradation");
  expect_embedded_gate([&] { get_complexity_per_elem(&mesh, Reals{}); }, "get_complexity_per_elem");
  expect_embedded_gate([&] { get_nelems_per_elem(&mesh, Reals{}); }, "get_nelems_per_elem");
  expect_embedded_gate([&] { get_complexity(&mesh, Reals{}); }, "get_complexity");
  expect_embedded_gate([&] { get_expected_nelems(&mesh, Reals{}); }, "get_expected_nelems");
  expect_embedded_gate([&] { get_metric_scalar_for_nelems(&mesh, Reals{}, Real{}); }, "get_metric_scalar_for_nelems");
  expect_embedded_gate([&] { smooth_metric_once(&mesh, Reals{}, bool{}); }, "smooth_metric_once");
  expect_embedded_gate([&] { get_curvature_metrics(&mesh, Real{}); }, "get_curvature_metrics");
  expect_embedded_gate([&] { get_derivative_metrics(&mesh, std::string{}, Real{}); }, "get_derivative_metrics");
  expect_embedded_gate([&] { get_variation_metrics(&mesh, std::string{}, Real{}); }, "get_variation_metrics");
  expect_embedded_gate([&] { has_interior_verts(&mesh); }, "has_interior_verts");
  expect_embedded_gate([&] { project_by_fit(&mesh, Reals{}); }, "project_by_fit");
  expect_embedded_gate([&] { project_by_average(&mesh, Reals{}); }, "project_by_average");
  expect_embedded_gate([&] { derive_element_gradients(&mesh, Reals{}); }, "derive_element_gradients");
  expect_embedded_gate([&] { derive_element_hessians(&mesh, Reals{}); }, "derive_element_hessians");
  expect_embedded_gate([&] { recover_gradients(&mesh, Reals{}); }, "recover_gradients");
  expect_embedded_gate([&] { recover_hessians_from_gradients(&mesh, Reals{}); }, "recover_hessians_from_gradients");
  expect_embedded_gate([&] { recover_hessians(&mesh, Reals{}); }, "recover_hessians");
  expect_embedded_gate([&] { transfer_conserve_refine(&mesh, TransferOpts{}, &ordinary, LOs{}, LOs{}, LOs{}, LOs{}, LOs{}); }, "transfer_conserve_refine");
  expect_embedded_gate([&] { transfer_conserve_refine(&ordinary, TransferOpts{}, &mesh, LOs{}, LOs{}, LOs{}, LOs{}, LOs{}); }, "transfer_conserve_refine");
  expect_embedded_gate([&] { transfer_densities_and_conserve_swap(&mesh, TransferOpts{}, &ordinary, LOs{}, LOs{}, LOs{}, LOs{}, LOs{}); }, "transfer_densities_and_conserve_swap");
  expect_embedded_gate([&] { transfer_densities_and_conserve_swap(&ordinary, TransferOpts{}, &mesh, LOs{}, LOs{}, LOs{}, LOs{}, LOs{}); }, "transfer_densities_and_conserve_swap");
  expect_embedded_gate([&] { transfer_densities_and_conserve_coarsen(&mesh, TransferOpts{}, &ordinary, LOs{}, Adj{}, LOs{}, LOs{}, LOs{}); }, "transfer_densities_and_conserve_coarsen");
  expect_embedded_gate([&] { transfer_densities_and_conserve_coarsen(&ordinary, TransferOpts{}, &mesh, LOs{}, Adj{}, LOs{}, LOs{}, LOs{}); }, "transfer_densities_and_conserve_coarsen");
  expect_embedded_gate([&] { transfer_conserve_motion(&mesh, TransferOpts{}, &ordinary, LOs{}, Graph{}, LOs{}, LOs{}); }, "transfer_conserve_motion");
  expect_embedded_gate([&] { transfer_conserve_motion(&ordinary, TransferOpts{}, &mesh, LOs{}, Graph{}, LOs{}, LOs{}); }, "transfer_conserve_motion");
  expect_embedded_gate([&] { setup_conservation_tags(&mesh, opts); }, "setup_conservation_tags");
  expect_embedded_gate([&] { correct_integral_errors(&mesh, opts); }, "correct_integral_errors");
  expect_embedded_gate([&] { measure_qualities(&mesh, LOs{}, Reals{}); }, "measure_qualities");
  expect_embedded_gate([&] { measure_qualities(&mesh, LOs{}); }, "measure_qualities");
  expect_embedded_gate([&] { measure_qualities(&mesh); }, "measure_qualities");
  expect_embedded_gate([&] { measure_qualities(&mesh, Reals{}); }, "measure_qualities");
  expect_embedded_gate([&] { get_1d_cavity_qualities(&mesh, Int{}, LOs{}); }, "get_1d_cavity_qualities");
  expect_embedded_gate([&] { amr::remove_non_leaf_uses(&mesh); }, "amr::remove_non_leaf_uses");
  expect_embedded_gate([&] { amr::enforce_2to1_refine(&mesh, Int{}, Bytes{}); }, "amr::enforce_2to1_refine");
  expect_embedded_gate([&] { amr::refine(&mesh, Bytes{}, TransferOpts{}); }, "amr::refine");
  expect_embedded_gate([&] { amr::derefine(&mesh, Bytes{}, TransferOpts{}); }, "amr::derefine");
  expect_embedded_gate([&] { get_side_vectors(&mesh, LOs{}); }, "get_side_vectors");
  expect_embedded_gate([&] { get_curv_edge_tangents(&mesh, LOs{}); }, "get_curv_edge_tangents");
  expect_embedded_gate([&] { get_hinge_angles(&mesh, Reals{}, LOs{}, LOs{}); }, "get_hinge_angles");
  expect_embedded_gate([&] { get_side_vert_normals(&mesh, LOs{}, Reals{}, LOs{}); }, "get_side_vert_normals");
  expect_embedded_gate([&] { get_curv_vert_tangents(&mesh, LOs{}, Reals{}, LOs{}); }, "get_curv_vert_tangents");
  expect_embedded_gate([&] { get_surf_tri_IIs(&mesh, LOs{}, Reals{}, LOs{}, Reals{}); }, "get_surf_tri_IIs");
  expect_embedded_gate([&] { get_surf_vert_IIs(&mesh, LOs{}, Reals{}, Reals{}, LOs{}, Reals{}); }, "get_surf_vert_IIs");
  expect_embedded_gate([&] { get_curv_edge_curvatures(&mesh, LOs{}, Reals{}, LOs{}, Reals{}); }, "get_curv_edge_curvatures");
  expect_embedded_gate([&] { get_curv_vert_curvatures(&mesh, LOs{}, Reals{}, LOs{}); }, "get_curv_vert_curvatures");
  expect_embedded_gate([&] { get_surface_info(&mesh); }, "get_surface_info");
  expect_embedded_gate([&] { get_vert_curvatures(&mesh, SurfaceInfo{}); }, "get_vert_curvatures");
  expect_embedded_gate([&] { find_bridge_edges(&mesh); }, "find_bridge_edges");
  expect_embedded_gate([&] { get_pad_dists(&mesh, Int{}, Read<I8>{}); }, "get_pad_dists");
  expect_embedded_gate([&] { get_pinched_angles(&mesh, Int{}); }, "get_pinched_angles");
  expect_embedded_gate([&] { mesh.ask_qualities(); }, "ask_qualities");
  expect_embedded_gate([&] { mesh.balance(true); }, "balance predictive");
  expect_embedded_gate([&] { classify_by_angles(&mesh, 0.5); }, "classify_by_angles");
  expect_embedded_gate([&] { MetricElementQualities<2, 2> q(&mesh, Reals{}); }, "MetricElementQualities");
  expect_embedded_gate([&] { MetricEdgeLengths<2, 2> q(&mesh, Reals{}); }, "MetricEdgeLengths");
}
#endif
#endif

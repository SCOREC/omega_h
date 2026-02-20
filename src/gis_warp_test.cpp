#include <Omega_h_adapt.hpp>
#include <Omega_h_array_ops.hpp>
#include <Omega_h_cmdline.hpp>
#include <Omega_h_file.hpp>
#include <Omega_h_metric.hpp>
#include <Omega_h_profile.hpp>
#include <iostream>

int main(int argc, char** argv) {
  auto lib = Omega_h::Library(&argc, &argv);
  Omega_h::CmdLine cmdline;
  cmdline.add_arg<std::string>("input.osh");
  cmdline.add_arg<double>("desired-num-elements");
  cmdline.add_arg<std::string>("geomSimModel.smd");
  cmdline.add_arg<std::string>("inputSplines.oshb");
  cmdline.add_arg<std::string>("output.vtk");
  cmdline.add_arg<int>("enableSnap");
  cmdline.add_arg<int>("enableSmoothing");
  auto const world = lib.world();
  if (!cmdline.parse_final(world, &argc, argv)) {
    return -1;
  }
  Omega_h::ScopedTimer scoped_timer("main");
  auto const world_rank = world->rank();
  auto const inpath = cmdline.get<std::string>("input.osh");
  auto const desired_nelems = cmdline.get<double>("desired-num-elements");
  auto const smdPath = cmdline.get<std::string>("geomSimModel.smd");
  auto const splinesPath = cmdline.get<std::string>("inputSplines.oshb");
  auto const outVtkPrefix = cmdline.get<std::string>("outputVtkPrefix");
  auto const enableSnap = cmdline.get<int>("enableSnap");
  auto const enableSmoothing = cmdline.get<int>("enableSmoothing");
  auto model = Omega_h::BsplineModel2D(smdPath, splinesPath);
  Omega_h::Mesh mesh(&lib);
  mesh = Omega_h::binary::read(inpath, world, true);
  mesh.balance();
  Omega_h::AdaptOpts opts(&mesh);
#ifdef OMEGA_H_USE_KOKKOS //FIXME should not be a kokkos flag...
  if(enableSnap) {
    opts.bspline_model = &model;
    if(enableSmoothing) {
      opts.should_smooth_snap = true;
    } else {
      opts.should_smooth_snap = false;
    }
  }
#endif
  opts.min_quality_allowed = 0.010;
  auto nelems = mesh.nglobal_ents(mesh.dim());
  if (world_rank == 0)
    std::cout << "mesh has " << nelems << " total elements\n";
  if (double(nelems) >= desired_nelems) {
    if (world_rank == 0)
      std::cout << "element count " << nelems << " >= target "
        << desired_nelems << ", will not adapt\n";
  }

  while (double(nelems) < desired_nelems) {
    if (world_rank == 0)
      std::cout << "element count " << nelems << " < target "
        << desired_nelems << ", will adapt\n";
    if (!mesh.has_tag(0, "metric")) {
      if (world_rank == 0)
        std::cout
          << "mesh had no metric, adding implied and adapting to it\n";
      Omega_h::add_implied_metric_tag(&mesh);
      Omega_h::adapt(&mesh, opts);
      nelems = mesh.nglobal_ents(mesh.dim());
      if (world_rank == 0)
        std::cout << "mesh now has " << nelems << " total elements\n";
    }
    auto metrics = mesh.get_array<double>(0, "metric");
    metrics = Omega_h::multiply_each_by(metrics, 1.2);
    auto const metric_ncomps =
      Omega_h::divide_no_remainder(metrics.size(), mesh.nverts());
    mesh.add_tag(0, "metric", metric_ncomps, metrics);
    if (world_rank == 0) std::cout << "adapting to scaled metric\n";
    Omega_h::adapt(&mesh, opts);
    nelems = mesh.nglobal_ents(mesh.dim());
    if (world_rank == 0)
      std::cout << "mesh now has " << nelems << " total elements\n";
  }
  Omega_h::vtk::write_parallel(outVtkPrefix + "_sides.vtk", &mesh, mesh.dim()-1);
  Omega_h::vtk::write_parallel(outVtkPrefix + ".vtk", &mesh, mesh.dim());
  return 0;
}

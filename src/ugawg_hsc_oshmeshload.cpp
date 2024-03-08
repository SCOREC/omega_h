#include <Omega_h_adapt.hpp>
#include <Omega_h_cmdline.hpp>
#include <Omega_h_file.hpp>
#include <Omega_h_library.hpp>
#include <Omega_h_mesh.hpp>
#include <Omega_h_metric.hpp>
#include <Omega_h_vtk.hpp>

int main(int argc, char** argv) {
  auto lib = Omega_h::Library(&argc, &argv);
  Omega_h::CmdLine cmdline;
  cmdline.add_arg<std::string>("mesh_in.meshb");
  if (!cmdline.parse_final(lib.world(), &argc, argv)) return -1;
  auto mesh_path = cmdline.get<std::string>("mesh_in.meshb");
  Omega_h::Mesh mesh(&lib);
  Omega_h::binary::read(mesh_path.c_str(), lib.world(), &mesh);

  auto hasOrigMetric = mesh.has_tag(0,"original_metric");
  auto hasTgtMetric = mesh.has_tag(0,"target_metric");
  if (!hasOrigMetric || !hasTgtMetric) {
    fprintf(stderr, "%s must have both the \"original_metric\" (%s)"
        " and \"target_metric\" (%s) vertex tags\n",
        mesh_path.c_str(),
        hasOrigMetric ? "exists" : "missing",
        hasTgtMetric ? "exists" : "missing");
    exit(EXIT_FAILURE);
  }

  fprintf(stderr, "rank %d: nverts %d nelms %d\n",
      lib.world()->rank(), mesh.nverts(), mesh.nelems());

  mesh.set_parting(OMEGA_H_GHOSTED); //adds one layer of ghosts, used in
                                     //ugawg_parallel_adapt
  Omega_h::add_implied_metric_tag(&mesh);
  mesh.ask_qualities();
  auto opts = Omega_h::AdaptOpts(&mesh);
  opts.xfer_opts.type_map["mach"] = OMEGA_H_LINEAR_INTERP;
  opts.min_quality_allowed = 0.07;

  opts.verbosity = Omega_h::EXTRA_STATS;
  while (Omega_h::approach_metric(&mesh, opts)) {
    Omega_h::adapt(&mesh, opts);
  }

  Omega_h::vtk::write_parallel("deltaWing_adapted_omegah.vtk", &mesh);
  return 0;
}

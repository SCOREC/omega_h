#include <iostream>

#include "Omega_h_adapt.hpp"
#include "Omega_h_array_ops.hpp"
#include "Omega_h_build.hpp"
#include "Omega_h_class.hpp"
#include "Omega_h_compare.hpp"
#include "Omega_h_for.hpp"
#include "Omega_h_shape.hpp"
#include "Omega_h_timer.hpp"
#include <Omega_h_file.hpp> //Omega_h::binary

using namespace Omega_h;

static void check_total_mass(Mesh* mesh) {
  const Real iceDensity = 910; //kg/m^3
  auto sizes = mesh->ask_sizes();
  Reals masses = multiply_each_by(sizes, iceDensity);
  auto owned_masses = mesh->owned_array(mesh->dim(), masses, 1);
  auto mass = get_sum(mesh->comm(), owned_masses);
  if (!mesh->comm()->rank()) {
    std::cout << "mass " << mass << '\n';
  }
}

void printTriCount(Mesh* mesh) {
  const auto nTri = mesh->nglobal_ents(2);
  if (!mesh->comm()->rank())
    std::cout << "nTri: " << nTri << "\n";
}

int main(int argc, char** argv) {
  auto lib = Library(&argc, &argv);
  if( argc != 3 ) {
    fprintf(stderr, "Usage: %s inputMesh.osh outputMeshPrefix\n", argv[0]);
    exit(EXIT_FAILURE);
  }
  auto world = lib.world();
  Omega_h::Mesh mesh(&lib);
  Omega_h::binary::read(argv[1], world, &mesh);

  //set size field - TODO: use ice thickness
  mesh.set_parting(OMEGA_H_GHOSTED);
  {
    auto metrics = get_implied_isos(&mesh);
    auto scalar = metric_eigenvalue_from_length(0.5);
    metrics = multiply_each_by(metrics, scalar);
    mesh.add_tag(VERT, "metric", 1, metrics);
  }
  mesh.set_parting(OMEGA_H_ELEM_BASED);

  auto opts = AdaptOpts(&mesh);
  opts.xfer_opts.type_map["ice_thickness"] = OMEGA_H_CONSERVE;
  opts.xfer_opts.integral_map["ice_thickness"] = "mass";
  opts.xfer_opts.integral_diffuse_map["mass"] =
      VarCompareOpts{VarCompareOpts::RELATIVE, 0.9, 0.0};
  check_total_mass(&mesh);

  printTriCount(&mesh);  
  adapt(&mesh, opts);
  printTriCount(&mesh);  
  check_total_mass(&mesh);
  const std::string vtkFileName = std::string(argv[2]) + ".vtk";
  Omega_h::vtk::write_parallel(vtkFileName, &mesh, 2);
  return 0;
}

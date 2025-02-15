#include <iostream>

#include "Omega_h_cmdline.hpp"
#include "Omega_h_file.hpp"
#include "Omega_h_mesh.hpp"
#include "Omega_h_build.hpp"
#include "Omega_h_for.hpp"

void localToGlobal(Omega_h::Mesh* mesh, int entDim) {
  OMEGA_H_CHECK(entDim >= 0 && entDim <= mesh->dim());
  std::string kernelName = "local_to_global_" + entDim;
  Omega_h::Write<Omega_h::GO> global(mesh->nents(entDim));
  Omega_h::parallel_for(kernelName.c_str(), mesh->nents(entDim),
    OMEGA_H_LAMBDA(int i) { global[i] = static_cast<Omega_h::GO>(i); }
  );
  mesh->add_tag<Omega_h::GO>(entDim, "global", 1, Omega_h::read(global));
}


int main(int argc, char** argv) {
  auto lib = Omega_h::Library(&argc, &argv);
  auto comm = lib.world();
  Omega_h::CmdLine cmdline;
  cmdline.add_arg<std::string>("mesh-in");
  cmdline.add_arg<std::string>("model-in(geomSim)");
  cmdline.add_arg<std::string>("mesh-out");
  auto& numberingFlag = cmdline.add_flag(
      "-numbering", "Attach the vertex numbering from the specified Simmetrix .nex file");
  numberingFlag.add_arg<std::string>("numbering-in");

  if (!cmdline.parse_final(comm, &argc, argv)) return -1;
  auto mesh_in = cmdline.get<std::string>("mesh-in");
  auto model_in = cmdline.get<std::string>("model-in(geomSim)");
  auto mesh_out = cmdline.get<std::string>("mesh-out");
  std::string numbering_in;
  if (cmdline.parsed("-numbering")) {
    std::cout << "attaching numbering...\n";
    numbering_in = cmdline.get<std::string>("-numbering", "numbering-in");
  }
  auto isMixed = Omega_h::meshsim::isMixed(mesh_in, model_in);
  std::cerr << "isMixed " << isMixed << "\n";
  if( !isMixed ) {
    auto mesh = Omega_h::meshsim::read(mesh_in, model_in, numbering_in, comm);
    //convert the local numbering to global
    //needed for oshdiff, see github issue #40
    for(int entDim=0; entDim<=mesh.dim(); entDim++) {
      localToGlobal(&mesh,entDim);
    }
    Omega_h::binary::write(mesh_out, &mesh);
  } else {
    auto mesh = Omega_h::meshsim::readMixed(mesh_in, model_in, comm);
  }
  return 0;
}

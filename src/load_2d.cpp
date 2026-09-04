#include <Omega_h_file.hpp>
#include <Omega_h_library.hpp>
#include <Omega_h_mesh2d.hpp>

#include <cstdlib>

int main(int argc, char** argv) {
  auto lib = Omega_h::Library(&argc, &argv);
  if(argc != 2) {
    fprintf(stderr, "Usage: %s inputMesh.osh\n", argv[0]);
    exit(EXIT_FAILURE);
  }
  OMEGA_H_ALWAYS_CHECK(argc == 2);
  Omega_h::Mesh2D mesh(&lib);
  Omega_h::binary::read(argv[1], lib.world(), &mesh);
}

#include <Omega_h_file.hpp>

int main(int argc, char** argv) {
  auto lib = Omega_h::Library(&argc, &argv);
  OMEGA_H_CHECK(argc == 3);
  auto world = lib.world();
  auto mesh = Omega_h::gmsh::read(argv[1], world);
  require_same_tdim_sdim(mesh);
  Omega_h::binary::write(argv[2], &mesh);
}

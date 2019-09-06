#include <Omega_h_file.hpp>
#include <Omega_h_library.hpp>
#include <Omega_h_mesh.hpp>

#include <cstdlib>

int main(int argc, char** argv) {
  auto lib = Omega_h::Library(&argc, &argv);
  OMEGA_H_CHECK(argc == 2);
  const auto strict = true;
  const auto rank = lib.world()->rank();
  if(!rank)
    fprintf(stderr, "loading mesh %s\n", argv[1]);
  Omega_h::Mesh mesh = Omega_h::binary::read(argv[1], lib.self(), strict);
  if(!rank)
    fprintf(stderr, "mesh tri %d\n", mesh.nelems());
  OMEGA_H_CHECK(cudaSuccess == cudaDeviceSynchronize());
  mesh.ask_up(0,2);
  OMEGA_H_CHECK(cudaSuccess == cudaDeviceSynchronize());
  mesh.ask_up(1,2);
  OMEGA_H_CHECK(cudaSuccess == cudaDeviceSynchronize());
  lib.world()->barrier();
  if(!rank)
    fprintf(stderr, "done\n");
  return 0;
}

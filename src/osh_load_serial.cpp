#include <Omega_h_file.hpp>
#include <Omega_h_library.hpp>
#include <Omega_h_mesh.hpp>

#include <cstdlib>

int main(int argc, char** argv) {
  auto lib = Omega_h::Library(&argc, &argv);
  OMEGA_H_CHECK(argc == 2);
  const auto strict = true;
  Omega_h::Mesh mesh = Omega_h::binary::read(argv[1], lib.self(), strict);
  auto dim = mesh.dim();
  OMEGA_H_CHECK(cudaSuccess == cudaDeviceSynchronize());
  mesh.ask_up(0,2);
  OMEGA_H_CHECK(cudaSuccess == cudaDeviceSynchronize());
  mesh.ask_up(1,2);
  OMEGA_H_CHECK(cudaSuccess == cudaDeviceSynchronize());
  return 0;
}

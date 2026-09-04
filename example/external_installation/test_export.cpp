#include <Omega_h_build.hpp>
#include <Omega_h_file.hpp>
#include <Omega_h_library.hpp>
#include <Omega_h_mesh.hpp>
#include <iostream>

int main(int argc, char** argv) {
  // Initialize Omega_h library
  auto lib = Omega_h::Library(&argc, &argv);
  auto world = lib.world();

  // Create a simple 2D box mesh
  std::cout << "Creating 2D box mesh..." << std::endl;
  auto mesh = Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1.0, 1.0, 0.0, 4, 4, 0);

  std::cout << "Mesh created with:" << std::endl;
  std::cout << "  Dimension: " << mesh.dim() << std::endl;
  std::cout << "  Vertices: " << mesh.nverts() << std::endl;
  std::cout << "  Elements: " << mesh.nelems() << std::endl;

  std::cout << "Test completed successfully!" << std::endl;

  return 0;
}
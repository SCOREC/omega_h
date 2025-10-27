//
// Converts Omega_h Mesh (Serial Mesh Only) to Gmsh Mesh
//

#include "Omega_h_file.hpp"

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "Usage: osh2msh <input.osh> <output.msh>\n";
    return 1;
  }
  const char* input_filename = argv[1];
  const char* output_filename = argv[2];

  try {
    auto lib = Omega_h::Library(&argc, &argv);
    Omega_h::Mesh mesh = Omega_h::binary::read(input_filename, lib.world());
    Omega_h::gmsh::write(output_filename, &mesh);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}

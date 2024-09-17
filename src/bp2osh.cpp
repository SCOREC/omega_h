#include <Omega_h_file.hpp>
#include <Omega_h_library.hpp>
#include <Omega_h_mesh.hpp>
#include <Omega_h_adios2.hpp>
#include <cstdlib>

int main(int argc, char** argv) {
  auto lib = Omega_h::Library(&argc, &argv);
  if (lib.world()->size()>1)
  {
    if (!lib.world()->rank())
      fprintf(stderr, "ADIOS2 file I/O with partitioned mesh is not supported yet\n");
    exit(EXIT_FAILURE);
  }

  if( argc != 3) {
    fprintf(stderr, "Usage: %s inputMesh.bp outputMesh.osh\n", argv[0]);
    exit(EXIT_FAILURE);
  }
  OMEGA_H_CHECK(argc == 3);
  Omega_h::Mesh mesh = read_adios2(argv[1], &lib);
  Omega_h::binary::write(argv[2], &mesh);
  return 0;
}

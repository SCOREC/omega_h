#include <mpi.h>

#include <iostream>

#include "Omega_h_build.hpp"
#include "Omega_h_defines.hpp"
#include "Omega_h_library.hpp"
#include "Omega_h_mesh.hpp"

Omega_h::Mesh make_mesh_wrong_with_buildbox(int argc, char** argv) {
  Omega_h::Library lib(&argc, &argv, MPI_COMM_SELF);

  // Mesh is built with a pointer to a library that will go out of scope
  Omega_h::Mesh mesh =
      Omega_h::build_box(lib.world(), OMEGA_H_SIMPLEX, 1, 1, 1, 1, 1, 1);
  printf("Library world size inside function: %d.\n", lib.world()->size());

  return mesh;
}

Omega_h::Mesh make_mesh_wrong_with_constructor(int argc, char** argv) {
  Omega_h::Library lib(&argc, &argv, MPI_COMM_SELF);

  // Library pointer is being copied
  Omega_h::Mesh mesh(&lib);
  printf("Library world size inside function: %d.\n", lib.world()->size());

  // out of scope library after return
  return mesh;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  Omega_h::Mesh m_box = make_mesh_wrong_with_buildbox(argc, argv);
  Omega_h::Mesh m_ctor = make_mesh_wrong_with_constructor(argc, argv);

  // Problematic access to the library after it has been destroyed
  auto lib_box = m_box.library();
  auto lib_ctor = m_ctor.library();

  // Will fail here
  int world_size_box = lib_box->world()->size();
  int world_size_ctor = lib_ctor->world()->size();

  // code to makes sure the values are used to skip compiler optimizations
  printf("World size from box mesh: %d.\n", world_size_box);
  printf("World size from ctor mesh: %d.\n", world_size_ctor);

  MPI_Finalize();
}

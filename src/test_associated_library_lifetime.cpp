#include <iostream>

#include "Omega_h_build.hpp"
#include "Omega_h_defines.hpp"
#include "Omega_h_library.hpp"
#include "Omega_h_mesh.hpp"

Omega_h::Mesh make_mesh_wrong_with_buildbox(int argc, char** argv) {
  Omega_h::Library lib(&argc, &argv);

  // Mesh is built with a pointer to a library that will go out of scope
  Omega_h::Mesh mesh =
      Omega_h::build_box(lib.world(), OMEGA_H_SIMPLEX, 1, 1, 1, 1, 1, 1);
  printf("Library.silent_ %d.\n", lib.silent_);
  printf("Library.argv_.size() member %zu.\n", lib.argv_.size());
  printf("Library.self_send_threshold_ %d.\n", lib.self_send_threshold_);

  return mesh;
}

Omega_h::Mesh make_mesh_wrong_with_constructor(int argc, char** argv) {
  Omega_h::Library lib(&argc, &argv);

  // Library pointer is being copied
  Omega_h::Mesh mesh(&lib);
  printf("Library.silent_ %d.\n", lib.silent_);
  printf("Library.argv_.size() member %zu.\n", lib.argv_.size());

  // out of scope library after return
  return mesh;
}

int main(int argc, char** argv) {
  Omega_h::Mesh m_box = make_mesh_wrong_with_buildbox(argc, argv);
  Omega_h::Mesh m_ctor = make_mesh_wrong_with_constructor(argc, argv);

  // Problematic access to the library after it has been destroyed
  const Omega_h::Library* lib_box = m_box.library();
  const Omega_h::Library* lib_ctor = m_ctor.library();

  // Will fail here
  const bool silent_mbox = lib_box->silent_;
  const bool silent_mctor = lib_ctor->silent_;
  const size_t size_mbox = lib_box->argv_.size();
  const size_t size_mcotr = lib_ctor->argv_.size();
  const int self_send_threshold = lib_box->self_send_threshold_;
  const int self_send_threshold_ctor = lib_ctor->self_send_threshold_;

  printf(
      "Build_box Mesh: silent_ %d, argv_.size() %zu, self_send_threshold %d.\n",
      silent_mbox, size_mbox, self_send_threshold);
  printf(
      "Constructed Mesh: silent_ %d, argv_.size() %zu, self_send_threshold "
      "%d.\n",
      silent_mctor, size_mcotr, self_send_threshold_ctor);

  return 0;  // should not reach here
}

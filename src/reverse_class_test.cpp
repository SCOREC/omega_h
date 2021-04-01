#include <iostream>
#include <string>

#include "Omega_h_file.hpp"
#include "Omega_h_for.hpp"
#include "Omega_h_mesh.hpp"
#include "Omega_h_element.hpp"
#include "Omega_h_array_ops.hpp"
using namespace Omega_h;

void call_print(LOs a) {
  fprintf(stderr,"\n");
  auto a_w = Write<LO> (a.size());
  auto r2w = OMEGA_H_LAMBDA(LO i) {
    a_w[i] = a[i];
  };
  parallel_for(a.size(), r2w);
  auto a_host = HostWrite<LO>(a_w);
  for (int i=0; i<a_host.size(); ++i) {
    fprintf(stderr," %d,", a_host[i]);
  };
  fprintf(stderr,"\n");
  fprintf(stderr,"\n");
  return;
}

void test_2d(Library *lib) {

  auto mesh = Mesh(lib);
  binary::read ("/lore/joshia5/Meshes/oh-mfem/plate_6elem.osh",
                lib->world(), &mesh);

  // test rc API
  fprintf(stderr,"for face\n");
  OMEGA_H_CHECK (!mesh.has_revClass(2));
  auto face_rc = mesh.ask_revClass(2);
  auto face_rc_get = mesh.get_revClass(2);
  OMEGA_H_CHECK (mesh.has_revClass(2));
  OMEGA_H_CHECK (face_rc.ab2b == face_rc_get.ab2b);
  OMEGA_H_CHECK (face_rc.a2ab == face_rc_get.a2ab);
  OMEGA_H_CHECK (face_rc.ab2b == LOs({0, 1, 2, 3, 4, 5}));
  OMEGA_H_CHECK (face_rc.a2ab == LOs({0, 0, 0, 6}));
  fprintf(stderr,"a2ab = \n");
  call_print(face_rc.a2ab);
  fprintf(stderr,"ab2b = \n");
  call_print(face_rc.ab2b);

  fprintf(stderr,"for edg\n");
  OMEGA_H_CHECK (!mesh.has_revClass(1));
  auto edge_rc = mesh.ask_revClass(1);
  auto edge_rc_get = mesh.get_revClass(1);
  OMEGA_H_CHECK (mesh.has_revClass(1));
  OMEGA_H_CHECK (edge_rc.ab2b == edge_rc_get.ab2b);
  OMEGA_H_CHECK (edge_rc.a2ab == edge_rc_get.a2ab);
  fprintf(stderr,"a2ab = \n");
  call_print(edge_rc.a2ab);
  fprintf(stderr,"ab2b = \n");
  call_print(edge_rc.ab2b);

  fprintf(stderr,"for vtx\n");
  OMEGA_H_CHECK (!mesh.has_revClass(0));
  auto vert_rc = mesh.ask_revClass(0);
  auto vert_rc_get = mesh.get_revClass(0);
  OMEGA_H_CHECK (mesh.has_revClass(0));
  OMEGA_H_CHECK (vert_rc.ab2b == vert_rc_get.ab2b);
  OMEGA_H_CHECK (vert_rc.a2ab == vert_rc_get.a2ab);
  OMEGA_H_CHECK (vert_rc.ab2b == LOs({3, 2, 1, 0}));
  OMEGA_H_CHECK (vert_rc.a2ab == LOs({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2,
                                      2, 2, 2, 3, 3, 3, 3, 4, 4}));
  fprintf(stderr,"a2ab = \n");
  call_print(vert_rc.a2ab);
  fprintf(stderr,"ab2b = \n");
  call_print(vert_rc.ab2b);

  return;
}

void test_3d(Library *lib) {

  auto mesh = Mesh(lib);
  binary::read ("/lore/joshia5/Meshes/oh-mfem/unitbox_cutTriCube_1k.osh",
                lib->world(), &mesh);

  // test reverse class APIs
  OMEGA_H_CHECK (!mesh.has_revClass(3));
  auto reg_rc = mesh.ask_revClass(3);
  auto reg_rc_get = mesh.get_revClass(3);
  OMEGA_H_CHECK (mesh.has_revClass(3));
  OMEGA_H_CHECK (reg_rc.ab2b == reg_rc_get.ab2b);
  OMEGA_H_CHECK (reg_rc.a2ab == reg_rc_get.a2ab);

  OMEGA_H_CHECK (!mesh.has_revClass(2));
  auto face_rc = mesh.ask_revClass(2);
  auto face_rc_get = mesh.get_revClass(2);
  OMEGA_H_CHECK (mesh.has_revClass(2));
  OMEGA_H_CHECK (face_rc.ab2b == face_rc_get.ab2b);
  OMEGA_H_CHECK (face_rc.a2ab == face_rc_get.a2ab);

  OMEGA_H_CHECK (!mesh.has_revClass(1));
  auto edge_rc = mesh.ask_revClass(1);
  auto edge_rc_get = mesh.get_revClass(1);
  OMEGA_H_CHECK (mesh.has_revClass(1));
  OMEGA_H_CHECK (edge_rc.ab2b == edge_rc_get.ab2b);
  OMEGA_H_CHECK (edge_rc.a2ab == edge_rc_get.a2ab);

  OMEGA_H_CHECK (!mesh.has_revClass(0));
  auto vert_rc = mesh.ask_revClass(0);
  auto vert_rc_get = mesh.get_revClass(0);
  OMEGA_H_CHECK (mesh.has_revClass(0));
  OMEGA_H_CHECK (vert_rc.ab2b == vert_rc_get.ab2b);
  OMEGA_H_CHECK (vert_rc.a2ab == vert_rc_get.a2ab);

  return;
}

int main(int argc, char** argv) {

  auto temp = argc;
  auto tempv = argv;

  auto lib = Library();

  test_2d(&lib);
  test_3d(&lib);
  // using mfem adapt tests, it was confirmed that rc info is
  // destroyed during adapt

  return 0;
}
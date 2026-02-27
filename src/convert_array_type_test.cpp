#include <fstream>
#include <iostream>

#include "Omega_h_vtk.hpp"
#include "Omega_h_file.hpp"

#ifdef OMEGA_H_USE_ADIOS2
#include <Omega_h_adios2.hpp>
#endif

using namespace Omega_h;

void check_SymmetricSquareMatrix_tag(const Tag<Real>* tag) {
  OMEGA_H_ALWAYS_CHECK(tag->array_type() == ArrayType::SymmetricSquareMatrix);
  OMEGA_H_ALWAYS_CHECK(tag->ncomps() == 3);
}

void check_vtk(CommPtr world, const std::string& tag_name) {
  std::ifstream file_converted("fields_converted.vtu");
  auto mesh_vtk_converted = Omega_h::Mesh();
  if (file_converted.is_open()) {
      Omega_h::vtk::read_vtu(file_converted, world, &mesh_vtk_converted);
      file_converted.close();
  }
  auto tag_converted = mesh_vtk_converted.get_tag<Omega_h::Real>(0, tag_name);
  check_SymmetricSquareMatrix_tag(tag_converted);
}

void check_binary(CommPtr world, const std::string& tag_name) {
  auto mesh_binary_converted = Omega_h::binary::read("fields_converted.osh", world);
  auto tag_converted = mesh_binary_converted.get_tag<Omega_h::Real>(0, tag_name);
  check_SymmetricSquareMatrix_tag(tag_converted);
}

#ifdef OMEGA_H_USE_ADIOS2
void check_adios2(Library* lib, const std::string& tag_name) {
  auto mesh_adios2_converted = Omega_h::adios::read("fields_converted.bp", lib);
  auto tag_converted = mesh_adios2_converted.get_tag<Omega_h::Real>(0, tag_name);
  check_SymmetricSquareMatrix_tag(tag_converted);
}
#endif

#ifdef OMEGA_H_USE_LIBMESHB
void check_meshb(Library* lib, const std::string& tag_name) {
  auto mesh_meshb_converted = Omega_h::Mesh(lib);
  Omega_h::meshb::read(&mesh_meshb_converted, "fields.meshb");
  Omega_h::meshb::read_sol(&mesh_meshb_converted, "fields_sol_converted.meshb", tag_name);
  auto tag_converted = mesh_meshb_converted.get_tag<Omega_h::Real>(0, tag_name);
  check_SymmetricSquareMatrix_tag(tag_converted);
}
#endif

void check_vtk_parallel(CommPtr world, Library* lib, const std::string& tag_name){
  auto mesh_vtk_parallel_converted = Omega_h::Mesh(lib);
  Omega_h::vtk::read_parallel("fields_parallel_converted/pieces.pvtu", world, &mesh_vtk_parallel_converted);
  auto tag_converted = mesh_vtk_parallel_converted.get_tag<Omega_h::Real>(0, tag_name);
  check_SymmetricSquareMatrix_tag(tag_converted);
}


int main(int argc, char** argv) {
  auto lib = Library(&argc, &argv);
  auto world = lib.world();
  auto rank = world->rank();
  auto size = world->size();

  const std::string tag_name = "electric_field";
  if (size > 1) {
    check_vtk_parallel(world, &lib, tag_name);
  } else {
    check_vtk(world, tag_name);
    check_binary(world, tag_name);
#ifdef OMEGA_H_USE_ADIOS2
    check_adios2(&lib, tag_name);
#endif
#ifdef OMEGA_H_USE_LIBMESHB
    check_meshb(&lib, tag_name);
#endif
  }
  return 0;
}
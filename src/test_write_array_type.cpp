#include <fstream>
#include <iostream>
#include <numeric>

#include "Omega_h_array_ops.hpp"
#include "Omega_h_build.hpp"
#include "Omega_h_mesh.hpp"
#include "Omega_h_vtk.hpp"
#include "Omega_h_file.hpp"
#include "Omega_h_for.hpp"

#ifdef OMEGA_H_USE_ADIOS2
#include <Omega_h_adios2.hpp>
#endif

using namespace Omega_h;

void check_tag(const Tag<Real>* tag, int flag) {
  if (flag) {
    OMEGA_H_ALWAYS_CHECK(tag->array_type() == ArrayType::SymmetricSquareMatrix);
  } else {
    OMEGA_H_ALWAYS_CHECK(tag->array_type() == ArrayType::VectorND);
  }
  OMEGA_H_ALWAYS_CHECK(tag->ncomps() == 3);
}

void test_vtk(Mesh* mesh, CommPtr world, const std::string& tag_name, int flag) {
  printf("Testing VTK\n");
  Omega_h::vtk::write_vtu("fields.vtu", mesh, mesh->dim());
  std::ifstream file("fields.vtu");
  auto mesh_vtk = Omega_h::Mesh();
  if (file.is_open()) {
      Omega_h::vtk::read_vtu(file, world, &mesh_vtk);
      file.close();
  }
  auto tag = mesh_vtk.get_tag<Omega_h::Real>(0, tag_name);
  std::cout << "vtk Components: " << tag->ncomps() << std::endl;
  check_tag(tag, flag);
}

void test_binary(
  Mesh* mesh, CommPtr world, const std::string& tag_name, int flag) {
  printf("Testing Binary\n");
  Omega_h::binary::write("fields.osh", mesh);
  auto mesh_binary = Omega_h::binary::read("fields.osh", world);
  auto tag = mesh_binary.get_tag<Omega_h::Real>(0, tag_name);
  std::cout << "Binary Components: " << tag->ncomps() << std::endl;
  std::cout << "ArrayType: " << ArrayTypeNames.at(tag->array_type()) << std::endl;
  check_tag(tag, flag);
}

#ifdef OMEGA_H_USE_ADIOS2
void test_adios2(
  Mesh* mesh, Library* lib, const std::string& tag_name, int flag) {
  printf("Testing ADIOS2\n");
  Omega_h::adios::write("fields.bp", mesh);
  auto mesh_adios2 = Omega_h::adios::read("fields.bp", lib);
  auto tag = mesh_adios2.get_tag<Omega_h::Real>(0, tag_name);
  std::cout << "Adios2 Components: " << tag->ncomps() << std::endl;
  std::cout << "ArrayType: " << ArrayTypeNames.at(tag->array_type()) << std::endl;
  check_tag(tag, flag);
}
#endif

#ifdef OMEGA_H_USE_LIBMESHB
void test_meshb(
  Mesh* mesh, Library* lib, const std::string& tag_name, int flag) {
  printf("Testing Meshb\n");
  Omega_h::meshb::write(mesh, "fields.meshb", 4);
  Omega_h::meshb::write_sol(mesh, "fields_sol.meshb", tag_name, 4);
  auto mesh_meshb = Omega_h::Mesh(lib);
  Omega_h::meshb::read(&mesh_meshb, "fields.meshb");
  Omega_h::meshb::read_sol(&mesh_meshb, "fields_sol.meshb", tag_name);
  auto tag = mesh_meshb.get_tag<Omega_h::Real>(0, tag_name);
  std::cout << "Meshb Components: " << tag->ncomps() << std::endl;
  std::cout << "ArrayType: " << ArrayTypeNames.at(tag->array_type()) << std::endl;
  check_tag(tag, flag);
}
#endif

void test_vtk_parallel(Mesh* mesh, CommPtr world, Library* lib, const std::string& tag_name, int flag){
  Omega_h::vtk::write_parallel("fields_parallel", mesh, mesh->dim());
    auto mesh_vtk_parallel = Omega_h::Mesh(lib);
    Omega_h::vtk::read_parallel("fields_parallel/pieces.pvtu", world, &mesh_vtk_parallel);
    auto tag = mesh_vtk_parallel.get_tag<Omega_h::Real>(0, tag_name);
    std::cout << "Parallel VTK Components: " << tag->ncomps() << std::endl;
    std::cout << "ArrayType: " << ArrayTypeNames.at(tag->array_type()) << std::endl;
    check_tag(tag, flag);
}


int main(int argc, char** argv) {
  auto lib = Library(&argc, &argv);
  auto world = lib.world();
  auto rank = world->rank();
  auto size = world->size();
  auto nx = 10;
  auto mesh = build_box(world, OMEGA_H_SIMPLEX, 1., 1., 0., nx, nx, 0);
  mesh.balance();
  mesh.set_parting(OMEGA_H_GHOSTED, 0);

  int input_array_type_flag = atoi(argv[1]);
  auto array_type_flag = ArrayType::VectorND;
  if (input_array_type_flag) {
    array_type_flag = ArrayType::SymmetricSquareMatrix;
  }

  mesh.add_tag<Omega_h::Real>(0, "Piece", 2);
  Omega_h::Write<Omega_h::Real> piece_w(mesh.nverts() * 2, 0);
  const auto init_piece = OMEGA_H_LAMBDA(Omega_h::LO r) {
    piece_w[r * 2 + 0] = 1.0;
    piece_w[r * 2 + 1] = 2.0;
  };
  Omega_h::parallel_for(mesh.nverts(), init_piece);
  mesh.set_tag(0, "Piece", Omega_h::Reals(piece_w), false);

  mesh.add_tag<Omega_h::Real>(0, "electric_field", 3);
  const std::string tag_name = "electric_field";
  Omega_h::Write<Omega_h::Real> E_node(mesh.nverts() * 3, 0);

  Omega_h::parallel_for(mesh.nverts() * 3, OMEGA_H_LAMBDA(int i) {
        E_node[i] = 7.0;
      });

  mesh.set_tag(0, "electric_field", Omega_h::Reals(E_node), false, array_type_flag);
  auto field = mesh.get_tag<Omega_h::Real>(0, tag_name);
  std::cout << "Components: " << field->ncomps() << std::endl;
 
  if (size > 1) {
    test_vtk_parallel(&mesh, world, &lib, tag_name, input_array_type_flag);
  } else {
    test_vtk(&mesh, world, tag_name, input_array_type_flag);
    test_binary(&mesh, world, tag_name, input_array_type_flag);
#ifdef OMEGA_H_USE_ADIOS2
    test_adios2(&mesh, &lib, tag_name, input_array_type_flag);
#endif
#ifdef OMEGA_H_USE_LIBMESHB
    test_meshb(&mesh, &lib, tag_name, input_array_type_flag);
#endif
  }
}
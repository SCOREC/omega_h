#include <Omega_h_file.hpp>
#include <Omega_h_library.hpp>
#include <Omega_h_mesh.hpp>
#include <Omega_h_build.hpp> //build_box

using namespace Omega_h;

void test2x2(Omega_h::CommPtr comm) {
  OMEGA_H_ALWAYS_CHECK(comm->size() == 1);
  const auto x = 2.0;
  const auto y = 2.0;
  const auto z = 0.0;
  const auto nx = 2;
  const auto ny = 2;
  const auto nz = 0;
  const auto symmetric = false;
  auto mesh = Omega_h::build_box(comm, OMEGA_H_SIMPLEX, x, y, z, nx, ny, nz, symmetric);
  const auto minPatchSize = 3;
  auto patches = mesh.get_vtx_patches(minPatchSize);
  Graph expected(
   {0,4,7,11,17,20,24,27,30,34},
   {0,1,2,6,1,2,4,1,2,4,5,0,1,2,3,5,6,1,4,5,1,3,5,6,3,6,7,0,6,7,0,3,6,7});
  OMEGA_H_ALWAYS_CHECK(patches == expected);
}

void test1x5(Omega_h::CommPtr comm) {
  OMEGA_H_ALWAYS_CHECK(comm->size() == 1);
  const auto x = 1.0;
  const auto y = 5.0;
  const auto z = 0.0;
  const auto nx = 1;
  const auto ny = 5;
  const auto nz = 0;
  const auto symmetric = false;
  auto mesh = Omega_h::build_box(comm, OMEGA_H_SIMPLEX, x, y, z, nx, ny, nz, symmetric);
  {
    const auto minPatchSize = 3;
    auto patches = mesh.get_vtx_patches(minPatchSize);
    Graph expected(
      {0,3,6,9,12,15,18,21,24,27,30,33,36},
      {0,1,2,1,2,3,0,1,2,0,1,2,1,3,4,3,4,6,5,6,9,4,5,6,7,8,9,7,8,9,7,8,9,5,7,9});
    OMEGA_H_ALWAYS_CHECK(patches == expected);
  }
  {
    const auto minPatchSize = 4;
    auto patches = mesh.get_vtx_patches(minPatchSize);
    Graph expected(
       {0,4,9,13,17,22,27,32,37,41,45,49,54},
       {0,1,2,3,0,1,2,3,4,0,1,2,3,0,1,2,3,1,2,3,4,6,1,3,
        4,5,6,4,5,6,7,9,3,4,5,6,9,5,7,8,9,5,7,8,9,5,7,8,9,5,6,7,8,9});
    OMEGA_H_ALWAYS_CHECK(patches == expected);
  }
}

void test3D(Omega_h::CommPtr comm) {
  OMEGA_H_ALWAYS_CHECK(comm->size() == 1);
  const auto x = 1.0;
  const auto y = 1.0;
  const auto z = 1.0;
  const auto nx = 2;
  const auto ny = 2;
  const auto nz = 2;
  const auto symmetric = false;
  auto mesh = Omega_h::build_box(comm, OMEGA_H_SIMPLEX, x, y, z, nx, ny, nz, symmetric);
  const auto minPatchSize = 3;
  auto patches = mesh.get_vtx_patches(minPatchSize);
  OMEGA_H_ALWAYS_CHECK(patches.nnodes() != -1);
}

void testPar(Omega_h::CommPtr comm) {
  OMEGA_H_ALWAYS_CHECK(comm->size() == 4);
  const auto x = 1.0;
  const auto y = 1.0;
  const auto z = 0.0;
  const auto nx = 4;
  const auto ny = 4;
  const auto nz = 0;
  const auto symmetric = false;
  auto mesh = Omega_h::build_box(comm, OMEGA_H_SIMPLEX, x, y, z, nx, ny, nz, symmetric);
  mesh.set_parting(OMEGA_H_GHOSTED);
  {
    const auto min_elems = comm->allreduce(mesh.nelems(), OMEGA_H_MIN);
    const auto minPatchSize = min_elems+1;
    auto patches = mesh.get_vtx_patches(minPatchSize);
    if( mesh.nelems() < minPatchSize ) {
      OMEGA_H_ALWAYS_CHECK(patches.nnodes() == -1); //expected to fail
    } else {
      OMEGA_H_ALWAYS_CHECK(patches.nnodes() != -1); //expected to pass
    }
  }
  {
    const auto minPatchSize = 6;
    auto patches = mesh.get_vtx_patches(minPatchSize);
    OMEGA_H_ALWAYS_CHECK(patches.nnodes() != -1);
  }
}

int main(int argc, char** argv) {
  auto lib = Omega_h::Library(&argc, &argv);
  auto world = lib.world();
  OMEGA_H_ALWAYS_CHECK(argc == 1);
  if (world->rank() == 0) {
    test2x2(lib.self());
    test1x5(lib.self());
    test3D(lib.self());
  }
  if (world->size() == 4) {
    testPar(world);
  }
  return 0;
}

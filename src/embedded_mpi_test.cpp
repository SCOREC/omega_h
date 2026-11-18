#include "embedded_fixture_test.hpp"
#include "Omega_h_array_ops.hpp"
#include "Omega_h_file.hpp"
#include <array>
#include <map>
#include <iostream>
#include <string>

using namespace Omega_h;
using embedded_test::check;

int main(int argc, char** argv) {
  Library lib(&argc, &argv);
  auto comm = lib.world();
  check(argc == 2 && comm->size() == std::stoi(argv[1]), "MPI rank count differs from requested count");
  auto reference = embedded_test::plane(lib);
  HostRead<GO> ref_globals(reference.globals(VERT));
  auto ref_coords_storage = copy_coords_to_host(reference.coords_view<3>());
  auto ref_coords = make_coords_view<3>(ref_coords_storage);
  std::map<GO, std::array<Real, 3>> expected;
  for (LO v = 0; v < reference.nverts(); ++v)
    expected[ref_globals[v]] = {{ref_coords(v, 0), ref_coords(v, 1), ref_coords(v, 2)}};

  auto verify = [&](Mesh& mesh) {
    check(mesh.space_dim() == 3, "lost spatial dimension on an MPI rank");
    HostRead<GO> globals(mesh.globals(VERT));
    auto coords_storage = copy_coords_to_host(mesh.coords_view<3>());
    auto coords = make_coords_view<3>(coords_storage);
    for (LO v = 0; v < mesh.nverts(); ++v)
      for (Int d = 0; d < 3; ++d)
        check(coords(v, d) == expected.at(globals[v])[d], "migration changed coordinates");
    auto area = get_sum(comm, mesh.owned_array(mesh.dim(), mesh.ask_sizes(), 1));
    check(std::abs(area - 1.0) < 1e-12, "parallel area sum differs");
  };

  bool root = comm->rank() == 0;
  Mesh mesh(&lib);
  build_from_elems2verts(&mesh, comm, OMEGA_H_SIMPLEX, 2,
      root ? reference.ask_verts_of(2) : LOs({}),
      root ? reference.globals(VERT) : GOs({}));
  mesh.set_space_dim(3);
  mesh.add_coords(root ? reference.retain_coords_storage() : Reals({}));
  mesh.set_parting(OMEGA_H_GHOSTED);
  embedded_test::classify(mesh);
  mesh.balance(false);
  verify(mesh);
  mesh.set_parting(OMEGA_H_GHOSTED);
  verify(mesh);
  mesh.set_parting(OMEGA_H_ELEM_BASED);
  verify(mesh);
  mesh.balance(Reals(mesh.nelems(), 1.0));
  verify(mesh);

  auto suffix = std::to_string(comm->size()) + ".osh";
  auto serial_path = "embedded_serial_" + suffix;
  if (root) binary::write(serial_path, &reference);
  comm->barrier();
  // This expands from one input rank and exercises bcast_mesh.
  auto expanded = binary::read(serial_path, comm);
  verify(expanded);
  expanded.balance(false);
  verify(expanded);
  auto parallel_path = "embedded_parallel_" + suffix;
  binary::write(parallel_path, &mesh);
  auto loaded = binary::read(parallel_path, comm, true);
  verify(loaded);
  comm->barrier();
  if (root) std::cout << "embedded MPI checks passed\n";
}

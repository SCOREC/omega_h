#include <Omega_h_macros.h>

#include <Omega_h_adj.hpp>
#include <Omega_h_file.hpp>
#include <Omega_h_mesh.hpp>
#include <Omega_h_reduce.hpp>
#include <cstdio>

#include "Omega_h_fail.hpp"

bool is_ccw_oriented(Omega_h::Mesh& mesh) {
  OMEGA_H_CHECK_PRINTF(mesh.dim() == 2, "ERROR: Mesh is not 2D. Found Dim = %d\n", mesh.dim());
  const auto& face2nodes = mesh.ask_down(Omega_h::FACE, Omega_h::VERT).ab2b;
  const auto& coords = mesh.coords();

  Omega_h::LO ccw;
  Kokkos::Sum<Omega_h::LO> sum_reducer(ccw);
  auto find_ccw_face = KOKKOS_LAMBDA(const Omega_h::LO face, Omega_h::LO& ccw) {
    const auto faceVerts = Omega_h::gather_verts<3>(face2nodes, face);
    const auto faceCoords = Omega_h::gather_vectors<3, 2>(coords, faceVerts);

    /*
     * It calculates the orientation based on sign of the cross product of
     * vector AB and AC if the vertices are given as A, B, C.
     *                   C (x3, y3)
     *                   /\
     *                  /  \
     *                 /    \
     *                /      \
     *               ↗        \
     *              /          \
     *             /            \
     *            /              \
     *  A(x1,y1) /--------→-------\ B (x2, y2)
     *
     * Vector AB = (x2 - x1, y2 - y1)
     * Vector AC = (x3 - x1, y3 - y1)
     * Cross product = AB x AC
     *               = - (y2 - y1) * (x3 - x1) + (x2 - x1) * (y3 - y1)
     */

    const Omega_h::LO local_ccw =
        -(faceCoords[1][1] - faceCoords[0][1]) *
                (faceCoords[2][0] - faceCoords[1][0]) +
            (faceCoords[2][1] - faceCoords[1][1]) *
                (faceCoords[1][0] - faceCoords[0][0]) >
        0;
    sum_reducer.join(ccw, local_ccw);
  };
  Kokkos::parallel_reduce("find_ccw", mesh.nfaces(), find_ccw_face, sum_reducer);
  OMEGA_H_CHECK_PRINTF(ccw == mesh.nfaces() || ccw == 0,
      "Expected 0 or %d but got ccw = %d\n", mesh.nfaces(), ccw);

  return bool(ccw / mesh.nfaces());
}

int main(int argc, char** argv) {
  Omega_h::Library library(&argc, &argv);
  Omega_h::Mesh mesh(&library);
  mesh = Omega_h::gmsh::read(argv[1], library.self());
  bool ccw = is_ccw_oriented(mesh);

  if (!ccw) {
    std::fprintf(stderr, "ERROR: Mesh is not counter clockwise oriented\n");
    return 1;
  }

  return 0;
}
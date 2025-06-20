#include "Omega_h_model2d.hpp"
#include <iostream>

namespace Omega_h {

Model2D Model2D::MeshModel2D_load(Mesh2D& mesh) {
  return Model2D();
}

void Model2D::printInfo() {
  std::cout << "==Model2d Info==\n";
  std::cout << "model entity type, count\n";
  std::cout << "vertices, " << vtxIds.size() << "\n";
  std::cout << "edges, " << edgeIds.size() << "\n";
  std::cout << "faces, " << faceIds.size() << "\n";
  std::cout << "edge use, " << edgeUseIds.size() << "\n";
  std::cout << "loop use, " << loopUseIds.size() << "\n";
}

//returns the edges in a loop use
LOs Model2D::getEdgesinLoop(LO loopUse) const {
  OMEGA_H_CHECK(loopUse < loopUseToEdgeUse.size());
  LOs edgesUses = loopUseToEdgeUse[loopUse];
  LOs edges(edgesUses.size());

  Kokkos::parallel_for(
      "getEdgesinLoop", edgesUses.size(),
      KOKKOS_CLASS_LAMBDA(const LO index) {
        edges[i] = edgeUseToVtx[edgesUses[i]];
      });
  return edges;
}

}  // namespace Omega_h

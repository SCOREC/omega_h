#ifndef OMEGA_H_MODEL2D_HPP
#define OMEGA_H_MODEL2D_HPP

#include <Omega_h_array.hpp>
#include <Omega_h_graph.hpp>
#include <Omega_h_mesh2d.hpp>

namespace Omega_h {

class Model2D {
public:
  explicit Model2D(Mesh2D& mesh);

private:
  float vtxTol, edgeTol;
  LOs vtxIds, edgeIds, loopIds, faceIds;
  LOs looptoLoopUse;
  Graph edgeToEdgeUse;
  Graph faceToLoopUse;
  Graph loopUseToEdgeUse;
  LOs edgeUseToVtx;
  Graph vtxToEdgeUse;
  LOs edgeUseToLoopUse;
  LOs loopUseToFace;
  LOs vtxCoords, edgeUseOrientation, loopUseOrientation;
};

}

#endif //OMEGA_H_MODEL2D_HPP

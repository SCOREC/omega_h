#ifndef OMEGA_H_MODEL2D_HPP
#define OMEGA_H_MODEL2D_HPP

#include <Omega_h_array.hpp>
#include <Omega_h_graph.hpp>

namespace Omega_h {

class Model2D {
private:
  float vtxTol, edgeTol;
  Write<size_t> vtxIds, edgeIds, loopIds, faceIds;
  Write<size_t> looptoLoopUse;
  Graph edgeToEdgeUse;
  Graph faceToLoopUse;
  Graph loopUseToEdgeUse;
  Write<size_t> edgeUseToVtx;
  Graph vtxToEdgeUse;
  Write<size_t> edgeUseToLoopUse;
  Write<size_t> loopUseToFace;
  Write<size_t> vtxCoords, edgeUseOrientation, loopUseOrientation;
};

}

#endif //OMEGA_H_MODEL2D_HPP

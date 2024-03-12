#ifndef OMEGA_H_MODEL2D_HPP
#define OMEGA_H_MODEL2D_HPP

#include <Omega_h_array.hpp>
#include <Omega_h_graph.hpp>
#include <Omega_h_mesh2d.hpp>

namespace Omega_h {

class Model2D {
public:
#ifdef OMEGA_H_USE_SIMMODSUITE
  static Model2D SimModel2D_load(std::string const& filename);
#endif
  static Model2D MeshModel2D_load(Mesh& mesh);

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
private:
  Model2D();
};

}

#endif //OMEGA_H_MODEL2D_HPP

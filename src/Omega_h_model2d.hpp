#ifndef OMEGA_H_MODEL2D_HPP
#define OMEGA_H_MODEL2D_HPP

#include <Omega_h_array.hpp>
#include <Omega_h_graph.hpp>

namespace Omega_h {

//forward declare, avoid circular dependency
class Mesh2D;

class Model2D {
public:
  //constructors
#ifdef OMEGA_H_USE_SIMMODSUITE
  static Model2D SimModel2D_load(std::string const& filename);
#endif
  static Model2D MeshModel2D_load(Mesh2D& mesh);
  //ids
  LOs vtxIds /*x*/, edgeIds /*x*/, faceIds /*x*/;
  //equal order adjacencies
  Graph edgeToEdgeUse;
  //downward adjacencies
  Graph faceToLoopUse;
  Graph loopUseToEdgeUse;
  LOs edgeUseToVtx;
  //upward adjacencies
  Graph vtxToEdgeUse;
  LOs edgeUseToLoopUse;
  LOs loopUseToFace;
  //use orientation
  LOs edgeUseOrientation, loopUseOrientation;
  //geometry
  Real vtxTol, edgeTol;
  Reals vtxCoords; //x
private:
  Model2D() = default;
};

}

#endif //OMEGA_H_MODEL2D_HPP
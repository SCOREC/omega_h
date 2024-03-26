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
  void printInfo();
  //ids
  LOs vtxIds, edgeIds , faceIds;
  LOs edgeUseIds, loopUseIds;
  //equal order adjacencies
  Graph edgeToEdgeUse;
  //downward adjacencies
  Graph faceToLoopUse;
  Graph loopUseToEdgeUse;
  LOs edgeUseToVtx; //each edgeUse has exactly two adjacent vertices
  //upward adjacencies
  Graph vtxToEdgeUse;
  LOs edgeUseToLoopUse; //each edgeUse has one adjacent loop use
  LOs loopUseToFace; //each loopUse has one adjacent face use

  //For each edgeUse, indicates the direction of the edge use 
  //relative to its owning edge. 1: same dir, 0: opposite dir
  LOs edgeUseOrientation;

  //For each loopUse, indicates forward or backward traversal order
  //of the edgeUses belonging to the loop.  1: forward, 0: backward
  LOs loopUseOrientation;

  //geometry
  Real vtxTol, edgeTol;
  Reals vtxCoords;
private:
  Model2D() = default;
};

}

#endif //OMEGA_H_MODEL2D_HPP

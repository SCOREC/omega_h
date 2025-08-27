#ifndef OMEGA_H_MODEL2D_HPP
#define OMEGA_H_MODEL2D_HPP

#include <Omega_h_array.hpp>
#include <Omega_h_graph.hpp>
#include <splineInterpolation.h>

namespace Omega_h {

//forward declare, avoid circular dependency
class Mesh2D;

class Model2D {

private:
  //ids
  LOs vtxIds, edgeIds , faceIds;
  LOs edgeUseIds, loopUseIds;
  //equal order adjacencies
  //each edge will have at most two uses, edges bounding a single face will only have one
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
  //TODO: Implement a heuristic to determine these tolerances
  Real vtxTol, edgeTol;
  Reals vtxCoords;
  std::vector<SplineInterp::BSpline2d> splines;

  Model2D() = default;

  //Model load helper functions
  void setAdjInfo(Graph edgeToEdgeUse, LOs edgeUseToVtx, LOs loopUseToFace, LOs edgeUseToLoopUse);

  void setVertexInfo(LOs ids, Reals coords) {
    this->vtxIds = ids;
    this->vtxCoords = coords;
  }

  void setEdgeInfo(LOs ids, LOs useIds, LOs useOrientation) {
    this->edgeIds = ids;
    this->edgeUseIds = useIds;
    this->edgeUseOrientation = useOrientation;
  }

  void setFaceIds(LOs ids) {
    this->faceIds = ids;
  }

  void setLoopUseIdsAndDir(LOs ids, LOs useOrientation) {
    this->loopUseIds = ids;
    this->loopUseOrientation = useOrientation;
  }

public:
  //constructors
#ifdef OMEGA_H_USE_SIMMODSUITE
  static const Model2D SimModel2D_load(std::string const& filename);
#endif
  void printInfo() const;
  
  //accessors
  /** @brief Get the vertex ids of the model.
   *  
   *  @return Read<LO> of vertex ids.
   */
  OMEGA_H_INLINE LOs getVtxIds() const { return vtxIds; }
  /** @brief Get the edge ids of the model.
   *  
   *  @return Read<LO> of edge ids.
   */
  OMEGA_H_INLINE LOs getEdgeIds() const { return edgeIds; }
  /** @brief Get the face ids of the model.
   *  
   *  @return Read<LO> of face ids.
   */
  OMEGA_H_INLINE LOs getFaceIds() const { return faceIds; }
    /** @brief Get the edge use ids of the model.
   *  
   *  @return Read<LO> of edge use ids.
   */
  OMEGA_H_INLINE LOs getEdgeUseIds() const { return edgeUseIds; }
  /** @brief Get the loop use ids of the model.
   *  
   *  @return Read<LO> of loop use ids.
   */
  OMEGA_H_INLINE LOs getLoopUseIds() const { return loopUseIds; }
  /** @brief Get the edge to edge use adjacency graph.
   *  
   *  @return Graph of edge to edge use adjacencies.
   */
  OMEGA_H_INLINE Graph getEdgeToEdgeUse() const { return edgeToEdgeUse; }
  /** @brief Get the face to loop use adjacency graph.
   *  
   *  @return Graph of face to loop use adjacencies.
   */
  OMEGA_H_INLINE Graph getFaceToLoopUse() const { return faceToLoopUse; }
  /** @brief Get the loop use to edge use adjacency graph.
   *  
   *  @return Graph of loop use to edge use adjacencies.
   */
  OMEGA_H_INLINE Graph getLoopUseToEdgeUse() const { return loopUseToEdgeUse; }
  /** @brief Get the edge use to vertex adjacency array.
   *  
   *  @return Read<LO> of edge use to vertex adjacencies.
   */
  OMEGA_H_INLINE LOs getEdgeUseToVtx() const { return edgeUseToVtx; }
  /** @brief Get the vertex to edge use adjacency graph.
   *  
   *  @return Graph of vertex to edge use adjacencies.
   */
  OMEGA_H_INLINE Graph getVtxToEdgeUse() const { return vtxToEdgeUse; }
  /** @brief Get the edge use to loop use adjacency array.
   *  
   *  @return Read<LO> of edge use to loop use adjacencies.
   */
  OMEGA_H_INLINE LOs getEdgeUseToLoopUse() const { return edgeUseToLoopUse; }
  /** @brief Get the loop use to face adjacency array.
   *  
   *  @return Read<LO> of loop use to face adjacencies.
   */
  OMEGA_H_INLINE LOs getLoopUseToFace() const { return loopUseToFace; }
  /** @brief Get the edge use orientations.
   *  
   *  Returns a Read<LO> where each entry indicates the direction of the
   *  edge use relative to its owning edge.
   *  - 1: same direction.
   *  - 0: opposite direction.
   * 
   *  @return Read<LO> of edge use orientations.
   */
  OMEGA_H_INLINE LOs getEdgeUseOrientation() const { return edgeUseOrientation; }
    /** @brief Get the loop use orientations.
   *  
   *  Returns a Read<LO> where each entry indicates the direction of the
   *  loop use relative to its owning edge.
   *  - 1: same direction.
   *  - 0: opposite direction.
   * 
   *  @return Read<LO> of loop use orientations.
   */
  OMEGA_H_INLINE LOs getLoopUseOrientation() const { return loopUseOrientation; }
  /** @brief Get the vertex tolerance.
   *  
   *  @return Real value of vertex tolerance.
   */
  OMEGA_H_INLINE Real getVtxTol() const { return vtxTol; }
  /** @brief Get the edge tolerance.
   *  
   *  @return Real value of edge tolerance.
   */
  OMEGA_H_INLINE Real getEdgeTol() const { return edgeTol; }
  /** @brief Get the vertex coordinates.
   *  
   *  @return Read<Real> of vertex coordinates in the form {x_0,y_0,z_0, x_1,y_1,z_1...,x_n,y_n,z_n}.
   */
  OMEGA_H_INLINE Reals const& getVtxCoords() const { return vtxCoords; }
};

}

#endif //OMEGA_H_MODEL2D_HPP

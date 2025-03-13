#ifndef OMEGA_H_MESHSIM_HPP
#define OMEGA_H_MESHSIM_HPP

#include "Omega_h_file.hpp"
#include "Omega_h_build.hpp"
#include "Omega_h_class.hpp"
#include "Omega_h_map.hpp"
#include "Omega_h_vector.hpp"
#include "Omega_h_mesh.hpp"
#include "Omega_h_mixedMesh.hpp"
#include "Omega_h_adj.hpp"

#include "MeshSim.h" // required to load GeomSim models in '.smd' files
#include "SimUtil.h"
#ifdef OMEGA_H_USE_SIMDISCRETE
#include "SimDiscrete.h"  // required to load discrete models in '.smd' files
#endif

namespace Omega_h {

namespace meshsim {

struct SimMeshInfo {
  int count_tet;
  int count_hex;
  int count_wedge;
  int count_pyramid;
  int count_tri;
  int count_quad;
  bool is_simplex;
  bool is_hypercube;
};


struct SimMesh{
  std::vector <pVertex> mV; 
  std::vector <pEdge> mE; 
  std::vector <pFace> mF; 
  std::vector <pRegion> mR; 
  SimMesh(std::vector <pVertex> vertices, std::vector <pEdge> edges, 
          std::vector <pFace> faces, std::vector <pRegion> regions);
  SimMesh(pMesh m);
};

SimMeshInfo getSimMeshInfo(const SimMesh& mesh);

struct SimMeshEntInfo {
  int maxDim;
  bool hasNumbering;

  std::vector <pVertex> mV;  // mesh vertices
  std::vector <pEdge> mE;  // mesh edges
  std::vector <pFace> mF;  // mesh faces
  std::vector <pRegion> mR;  // mesh regions

  SimMeshEntInfo(const SimMesh& simMesh, bool hasNumbering_in);

  struct EntClass {
    HostWrite<LO> id;
    HostWrite<I8> dim;
    HostWrite<LO> verts;
  };

  struct VertexInfo {
    HostWrite<Real> coords;
    HostWrite<LO> id;
    HostWrite<I8> dim;
    HostWrite<LO> numbering;
  };

  VertexInfo readVerts(pMeshNex numbering);
  void setVtxIds(pPList listVerts, const int vtxPerEnt, const int entIdx, HostWrite<LO>& verts);
  EntClass readEdges();

  struct MixedFaceClass {
    EntClass tri;
    EntClass quad;
  };

  MixedFaceClass readMixedFaces(GO count_tri, GO count_quad);
  EntClass readMonoTopoFaces(GO numFaces, LO vtxPerFace);
  
  struct MixedRgnClass {
    EntClass tet;
    EntClass hex;
    EntClass wedge;
    EntClass pyramid;
  };

  MixedRgnClass readMixedTopoRegions(
      LO numTets, LO numHexs,
      LO numWedges, LO numPyramids);

  EntClass readMonoTopoRegions(GO numRgn, LO vtxPerRgn);

  private:
  SimMeshEntInfo();
  int getMaxDim(std::array<int,4> numEnts);
};

std::vector <pVertex> getMeshVertices(pMesh m);
std::vector <pEdge> getMeshEdges(pMesh m);
std::vector <pFace> getMeshFaces(pMesh m);
std::vector <pRegion> getMeshRegions(pMesh m);
void setEntToMesh(Mesh* mesh, SimMeshEntInfo simEnts, pMeshNex numbering, SimMeshInfo info);
void readMixed_internal(SimMesh simMesh, MixedMesh* mesh, SimMeshInfo info);
void read_internal(SimMesh simMesh, Mesh* mesh, pMeshNex numbering, SimMeshInfo info);

}  // namespace meshsim

}  // end namespace Omega_h

#endif


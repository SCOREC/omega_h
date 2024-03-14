#include <SimModel.h>
#include <SimUtil.h>
#include "Omega_h_model2d.hpp"
#include <map>

namespace Omega_h {

bool isModel2D(pGModel mdl) {
  return (GM_numRegions(mdl) == 0);
}

bool isValid(pGModel mdl, bool checkGeo = false) {
  const auto maxCheckLevel = 2; //may be expensive
  auto geoCheck = checkGeo ? maxCheckLevel : 0;
  pPList errList =  NULL;
  const auto valid = 1;
  return (GM_isValid(mdl,geoCheck,errList) == valid);
}

struct VtxIdsAndCoords {
  LOs ids;
  Reals coords;
};

struct EdgeInfo {
  LOs ids;
  std::map<int,int> idToIdx;
};

LOs getFaceIds(pGModel mdl) {
  auto numFaces = GM_numFaces(mdl);
  auto ids_h = HostWrite<LO>(numFaces);
  GFIter modelFaces = GM_faceIter(mdl);
  int idx = 0;
  pGFace modelFace;
  while(modelFace=GFIter_next(modelFaces)) {
    ids_h[idx++] = GEN_tag(modelFace);
  }
  GFIter_delete(modelFaces);
  return LOs(ids_h);
}

/*
 * retreive the entity-to-use adjacencies
 *
 * SimModSuite has a limited set of APIs for accessing
 * this info ... I've prepared some spaghetti below
 */
LOs getUses(pGModel mdl, const EdgeInfo& edgeInfo) {
  const auto numEdges = GM_numEdges(mdl);
  auto edgeToEdgeUseDegree = HostWrite<LO>(numEdges);
  for( int i=0; i<numEdges; i++ ) edgeToEdgeUseDegree[i] = 0; //better way?
  GFIter modelFaces = GM_faceIter(mdl);
  int idx = 0;
  pGFace modelFace;
  while(modelFace=GFIter_next(modelFaces)) {
    for(int side=0; side<2; side++) {
      auto faceUse = GF_use(modelFace,side);
      auto numLoops = GF_numLoops(modelFace);
      auto loopUses = GFU_loopIter(faceUse);
      pGLoopUse loopUse;
      while(loopUse=GLUIter_next(loopUses)) {
        auto edgeUses = GLU_edgeUseIter(loopUse);
        pGEdgeUse edgeUse;
        while(edgeUse=GEUIter_next(edgeUses)) {
          auto edge = GEU_edge(edgeUse);
          const auto edgeId = GEN_tag(edge);
          const auto edgeIdx = edgeInfo.idToIdx.at(edgeId);
          edgeToEdgeUseDegree[edgeIdx] = edgeToEdgeUseDegree[edgeIdx]+1;
          auto vtx0 = GE_vertex(edge,0);
          auto vtx1 = GE_vertex(edge,1);
        }
        GEUIter_delete(edgeUses);
      }
      GLUIter_delete(loopUses);
    } //sides
  }
  GFIter_delete(modelFaces);
  return LOs();
}

EdgeInfo getEdgeIds(pGModel mdl) {
  std::map<int,int> idToIdx;
  const auto numEdges = GM_numEdges(mdl);
  auto ids_h = HostWrite<LO>(numEdges);
  GEIter modelEdges = GM_edgeIter(mdl);
  int idx = 0;
  pGEdge modelEdge;
  while(modelEdge=GEIter_next(modelEdges)) {
    const auto id =GEN_tag(modelEdge);
    ids_h[idx] = id;
    idToIdx[id] = idx;
    idx++;
  }
  GEIter_delete(modelEdges);
  return EdgeInfo{LOs(ids_h),idToIdx};
}

VtxIdsAndCoords getVtxIdsAndCoords(pGModel mdl) {
  const auto numSpatialDims = 3;
  auto numVtx = GM_numVertices(mdl);
  auto vtxIds_h = HostWrite<LO>(numVtx);
  auto vtxCoords_h = HostWrite<Real>(numVtx*numSpatialDims);
  GVIter modelVertices = GM_vertexIter(mdl);
  int idx = 0;
  pGVertex modelVertex;
  double vpoint[3];
  while(modelVertex=GVIter_next(modelVertices)) {
    vtxIds_h[idx] = GEN_tag(modelVertex);
    GV_point(modelVertex, vpoint);
    for(int i=0; i<numSpatialDims; i++)
      vtxCoords_h[idx*numSpatialDims+i] = vpoint[i];
    idx++;
  }
  GVIter_delete(modelVertices);
  return VtxIdsAndCoords{LOs(vtxIds_h), Reals(vtxCoords_h)};
}

Model2D Model2D::SimModel2D_load(std::string const& filename) {
  pNativeModel nm = NULL;
  pProgress p = NULL;
  pGModel g = GM_load(filename.c_str(), nm, p);
  const char* msg2d = "Simmetrix GeomSim model is not 2D... exiting\n";
  OMEGA_H_CHECK_MSG(isModel2D(g), msg2d);
  const char* msgValid = "Simmetrix GeomSim model is not valid... exiting\n";
  OMEGA_H_CHECK_MSG(isValid(g), msgValid);
  auto mdl = Model2D();
  const auto vtxInfo = getVtxIdsAndCoords(g);
  mdl.vtxIds = vtxInfo.ids;
  mdl.vtxCoords = vtxInfo.coords;
  const auto edgeInfo = getEdgeIds(g);
  mdl.edgeIds = edgeInfo.ids;
  mdl.faceIds = getFaceIds(g);
  getUses(g,edgeInfo);
  GM_release(g);
  return mdl;
}
  
}//end namespace Omega_h

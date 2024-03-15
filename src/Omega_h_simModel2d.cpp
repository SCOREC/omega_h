#include <SimModel.h>
#include <SimUtil.h>
#include "Omega_h_model2d.hpp"
#include <map>
#include <algorithm> //std::fill
#include <numeric> //std::exclusive_scan

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

struct EntInfo {
  LOs ids;
  std::map<int,int> idToIdx;
};

struct VtxInfo : public EntInfo {
  Reals coords;
};

EntInfo getFaceIds(pGModel mdl) {
  std::map<int,int> idToIdx;
  auto numFaces = GM_numFaces(mdl);
  auto ids_h = HostWrite<LO>(numFaces);
  GFIter modelFaces = GM_faceIter(mdl);
  int idx = 0;
  pGFace modelFace;
  while(modelFace=GFIter_next(modelFaces)) {
    const auto id = GEN_tag(modelFace);
    ids_h[idx] = id;
    idToIdx[id] = idx;
    idx++;
  }
  GFIter_delete(modelFaces);
  return EntInfo{LOs(ids_h), idToIdx};
}

void incrementDegree(HostWrite<LO>& degree, std::map<int,int> idToIdx, pGEntity ent) {
  const auto id = GEN_tag(ent);
  const auto idx = idToIdx.at(id);
  degree[idx] = degree[idx]+1;
}

HostWrite<LO> createAndInitArray(size_t size, const LO init=0) {
  auto array = HostWrite<LO>(size);
  std::fill(array.data(), array.data()+array.size(), init);
  return array;
}

HostWrite<LO> createArray(size_t size) {
  return HostWrite<LO>(size);
}

struct CSR {
  CSR(int size) {
    offset = createAndInitArray(size+1);
    count = createAndInitArray(size);
    //values array is allocated once degree is populated
  }
  HostWrite<LO> offset;
  HostWrite<LO> count;
  HostWrite<LO> values;
  void degreeToOffset() {
    std::exclusive_scan(offset.data(), offset.data()+offset.size(), offset.data(), 0);
    values = createArray(offset[offset.size()-1]);
  };
  void setValue(int entIdx, LO value) {
    const auto adjIdx = offset[entIdx];
    const auto adjCount = count[entIdx];
    values[adjIdx+adjCount] = value;
    count[entIdx]++;
  }
  private:
  CSR();
};

template<typename UseType>
struct EntToAdjUse : public CSR {
  EntToAdjUse(int size, const EntInfo& entInfo_in)
    : entInfo(entInfo_in), CSR(size) {}
  const EntInfo& entInfo;
  template<int mode>
  void countOrSet(pGEntity ent, UseType use) {
    static_assert((mode == 0 || mode == 1), "countOrSet<mode> called with invalid mode");
    if constexpr (mode == 0 ) {
      incrementDegree(offset, entInfo.idToIdx, ent);
    } else {
      const auto entId = GEN_tag(ent);
      const auto entIdx = entInfo.idToIdx.at(entId);
      const auto useId = GEN_tag(use);
      const auto useIdx = 0; //FIXME
      setValue(entIdx, useIdx);
    }
  }
  private:
  EntToAdjUse();
};

/*
 * retrieve the entity-to-use adjacencies
 *
 * SimModSuite has a limited set of APIs for accessing
 * this info ... I've prepared some spaghetti below
 */
template<int mode>
LOs getUses(pGModel mdl,
    EntToAdjUse<pGEdgeUse>& v2eu,
    EntToAdjUse<pGEdgeUse>& e2eu,
    EntToAdjUse<pGLoopUse>& f2lu) {
  static_assert((mode == 0 || mode == 1), "getUses<mode> called with invalid mode");

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
        f2lu.countOrSet<mode>(modelFace,loopUse);
        auto edgeUses = GLU_edgeUseIter(loopUse);
        pGEdgeUse edgeUse;
        while(edgeUse=GEUIter_next(edgeUses)) {
          auto edge = GEU_edge(edgeUse);
          e2eu.countOrSet<mode>(edge,edgeUse);
          auto vtx0 = GE_vertex(edge,0);
          v2eu.countOrSet<mode>(vtx0,edgeUse);
          auto vtx1 = GE_vertex(edge,1);
          v2eu.countOrSet<mode>(vtx1,edgeUse);
        }
        GEUIter_delete(edgeUses);
      }
      GLUIter_delete(loopUses);
    } //sides
  }
  GFIter_delete(modelFaces);

  if constexpr (mode==0) {
    f2lu.degreeToOffset();
    e2eu.degreeToOffset();
    v2eu.degreeToOffset();
  }
  return LOs();
}

EntInfo getEdgeIds(pGModel mdl) {
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
  return EntInfo{LOs(ids_h),idToIdx};
}

VtxInfo getVtxInfo(pGModel mdl) {
  const auto numSpatialDims = 3;
  auto numVtx = GM_numVertices(mdl);
  auto vtxIds_h = HostWrite<LO>(numVtx);
  auto vtxCoords_h = HostWrite<Real>(numVtx*numSpatialDims);
  std::map<int,int> idToIdx;
  GVIter modelVertices = GM_vertexIter(mdl);
  int idx = 0;
  pGVertex modelVertex;
  double vpoint[3];
  while(modelVertex=GVIter_next(modelVertices)) {
    const auto id = GEN_tag(modelVertex);
    vtxIds_h[idx] = id;
    idToIdx[id] = idx;
    GV_point(modelVertex, vpoint);
    for(int i=0; i<numSpatialDims; i++)
      vtxCoords_h[idx*numSpatialDims+i] = vpoint[i];
    idx++;
  }
  GVIter_delete(modelVertices);
  return VtxInfo{LOs(vtxIds_h), idToIdx, Reals(vtxCoords_h)};
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
  const auto vtxInfo = getVtxInfo(g);
  mdl.vtxIds = vtxInfo.ids;
  mdl.vtxCoords = vtxInfo.coords;
  const auto edgeInfo = getEdgeIds(g);
  mdl.edgeIds = edgeInfo.ids;
  const auto faceInfo = getFaceIds(g);
  mdl.faceIds = faceInfo.ids;

  const auto numVtx = GM_numVertices(g);
  EntToAdjUse<pGEdgeUse> v2eu(numVtx, vtxInfo);

  const auto numEdges = GM_numEdges(g);
  EntToAdjUse<pGEdgeUse> e2eu(numEdges, edgeInfo);

  const auto numFaces = GM_numFaces(g);
  EntToAdjUse<pGLoopUse> f2lu(numFaces, faceInfo);

  getUses<0>(g,v2eu,e2eu,f2lu);
  getUses<1>(g,v2eu,e2eu,f2lu);
  GM_release(g);
  return mdl;
}

}//end namespace Omega_h

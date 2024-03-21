#include <SimModel.h>
#include <SimUtil.h>
#include "Omega_h_model2d.hpp"
#include "Omega_h_profile.hpp"
#include <map>
#include <algorithm> //std::fill
#include <numeric> //std::exclusive_scan
#include <variant> //std::visit, std::variant

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

struct UseInfo : public EntInfo {
  UseInfo() : idx(0) {}
  std::vector<LO> ids_h;
  int idx;
  void idsToDevice() {
    auto array = HostWrite<LO>(ids_h.size());
    std::copy(ids_h.begin(), ids_h.end(), array.data());
//    for(int i=0; i<ids_h.size(); i++) {
//      ids[i] = ids_h[i];
//    }
    ids = LOs(array);
  }
};

struct LoopUseInfo : public UseInfo {};
struct EdgeUseInfo : public UseInfo {};

VtxInfo getVtxInfo(pGModel mdl) {
  OMEGA_H_TIME_FUNCTION;
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

EntInfo getEdgeInfo(pGModel mdl) {
  OMEGA_H_TIME_FUNCTION;
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

EntInfo getFaceInfo(pGModel mdl) {
  OMEGA_H_TIME_FUNCTION;
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
    OMEGA_H_TIME_FUNCTION;
    offset = createAndInitArray(size+1);
    count = createAndInitArray(size);
    //values array is allocated once degree is populated
  }
  HostWrite<LO> offset;
  HostWrite<LO> count;
  HostWrite<LO> values;
  void degreeToOffset() {
    OMEGA_H_TIME_FUNCTION;
    std::exclusive_scan(offset.data(), offset.data()+offset.size(), offset.data(), 0);
    values = createArray(offset[offset.size()-1]);
  };
  void incrementDegree(const int entIdx) {
    offset[entIdx] = offset[entIdx]+1;
  }
  void setValue(int entIdx, LO value) {
    OMEGA_H_TIME_FUNCTION;
    const auto adjIdx = offset[entIdx];
    const auto adjCount = count[entIdx];
    values[adjIdx+adjCount] = value;
    count[entIdx]++;
  }
  private:
  CSR();
};

enum GetUsesMode {
  CountAdj = 1,
  SetAdj = 2
};

template<typename EntType, typename UseType>
struct EntToAdjUse : public CSR {
  EntToAdjUse(const EntInfo& entInfo_in)
    : entInfo(entInfo_in), useCount(0),
      CSR(entInfo_in.ids.size()) {}
  const EntInfo& entInfo;
  std::map<int, int> useIdToIdx;
  int useCount;
  template<GetUsesMode mode>
  void countOrSet(EntType ent, UseType use) {
    static_assert((mode == GetUsesMode::CountAdj || mode == GetUsesMode::SetAdj),
        "countOrSet<mode> called with invalid mode");
    OMEGA_H_TIME_FUNCTION;
    if constexpr (mode == GetUsesMode::CountAdj) {
      ScopedTimer timer("EntToAdjUse::count");
      const auto entId = GEN_tag(ent);
      const auto entIdx = entInfo.idToIdx.at(entId);
      incrementDegree(entIdx);
    } else {
      ScopedTimer timer("EntToAdjUse::set");
      const auto entId = GEN_tag(ent);
      const auto entIdx = entInfo.idToIdx.at(entId);
      const auto useId = GEN_tag(use);
      if( ! useIdToIdx.count(useId) ) {
        useIdToIdx[useId] = useCount++;
      }
      const auto useIdx = useIdToIdx[useId];
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
struct ITraverseUses {
  // abstract operators
  virtual void loopUseOperator(pGFace modelFace, pGLoopUse loopUse) = 0;
  virtual void edgeUseOperator(pGLoopUse loopUse, pGEdgeUse edgeUse) = 0;

  // algorithm that calls abstract operators
  virtual void run(pGModel mdl) final {
    OMEGA_H_TIME_FUNCTION;

    GFIter modelFaces = GM_faceIter(mdl);
    int idx = 0;
    pGFace modelFace;
    while(modelFace=GFIter_next(modelFaces)) {
      for(int side=0; side<2; side++) {
        auto faceUse = GF_use(modelFace,side);
        auto loopUses = GFU_loopIter(faceUse);
        pGLoopUse loopUse;
        while(loopUse=GLUIter_next(loopUses)) {
          loopUseOperator(modelFace, loopUse);
          auto edgeUses = GLU_edgeUseIter(loopUse);
          pGEdgeUse edgeUse;
          while(edgeUse=GEUIter_next(edgeUses)) {
            edgeUseOperator(loopUse, edgeUse);
          }
          GEUIter_delete(edgeUses);
        }
        GLUIter_delete(loopUses);
      } //sides
    }
    GFIter_delete(modelFaces);
  }
};

struct SetUseIds : public ITraverseUses {
  LoopUseInfo& loopUseInfo_;
  EdgeUseInfo& edgeUseInfo_;
  SetUseIds(LoopUseInfo& loopUseInfo, EdgeUseInfo& edgeUseInfo)
    : loopUseInfo_(loopUseInfo), edgeUseInfo_(edgeUseInfo) {}

  void loopUseOperator(pGFace modelFace, pGLoopUse loopUse) override {
    const auto id = GEN_tag(loopUse);
    loopUseInfo_.ids_h.push_back(id);
    loopUseInfo_.idToIdx[id] = loopUseInfo_.idx;
    loopUseInfo_.idx++;
  }

  virtual void edgeUseOperator(pGLoopUse loopUse, pGEdgeUse edgeUse) override {
    const auto id = GEN_tag(edgeUse);
    edgeUseInfo_.ids_h.push_back(id);
    edgeUseInfo_.idToIdx[id] = edgeUseInfo_.idx;
    edgeUseInfo_.idx++;
  }
};

struct Adjacencies {
  EntToAdjUse<pGVertex, pGEdgeUse>& v2eu;
  EntToAdjUse<pGEdge, pGEdgeUse>& e2eu;
  EntToAdjUse<pGFace, pGLoopUse>& f2lu;
  EntToAdjUse<pGLoopUse, pGEdgeUse>& lu2eu;
};

struct CountUses : public ITraverseUses {
  Adjacencies& adj_;
  CountUses(Adjacencies& adj) : adj_(adj) {}

  void loopUseOperator(pGFace modelFace, pGLoopUse loopUse) override {
    const auto Mode = GetUsesMode::CountAdj;
    adj_.f2lu.countOrSet<Mode>(modelFace, loopUse);
  }

  virtual void edgeUseOperator(pGLoopUse loopUse, pGEdgeUse edgeUse) override {
    const auto Mode = GetUsesMode::CountAdj;
    auto edge = GEU_edge(edgeUse);
    adj_.lu2eu.countOrSet<Mode>(loopUse,edgeUse);
    adj_.e2eu.countOrSet<Mode>(edge,edgeUse);
    auto vtx0 = GE_vertex(edge,0);
    adj_.v2eu.countOrSet<Mode>(vtx0,edgeUse);
    auto vtx1 = GE_vertex(edge,1);
    adj_.v2eu.countOrSet<Mode>(vtx1,edgeUse);
  }
};

struct SetUses : public ITraverseUses {
  Adjacencies& adj_;
  SetUses(Adjacencies& adj) : adj_(adj) {}

  void loopUseOperator(pGFace modelFace, pGLoopUse loopUse) override {
    const auto Mode = GetUsesMode::SetAdj;
    adj_.f2lu.countOrSet<Mode>(modelFace, loopUse);
  }

  virtual void edgeUseOperator(pGLoopUse loopUse, pGEdgeUse edgeUse) override {
    const auto Mode = GetUsesMode::SetAdj;
    auto edge = GEU_edge(edgeUse);
    adj_.lu2eu.countOrSet<Mode>(loopUse,edgeUse);
    adj_.e2eu.countOrSet<Mode>(edge,edgeUse);
    auto vtx0 = GE_vertex(edge,0);
    adj_.v2eu.countOrSet<Mode>(vtx0,edgeUse);
    auto vtx1 = GE_vertex(edge,1);
    adj_.v2eu.countOrSet<Mode>(vtx1,edgeUse);
  }
};

Model2D Model2D::SimModel2D_load(std::string const& filename) {
  OMEGA_H_TIME_FUNCTION;
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
  const auto edgeInfo = getEdgeInfo(g);
  mdl.edgeIds = edgeInfo.ids;
  const auto faceInfo = getFaceInfo(g);
  mdl.faceIds = faceInfo.ids;
  auto loopUseInfo = LoopUseInfo();
  auto edgeUseInfo = EdgeUseInfo();
  SetUseIds setUseIds(loopUseInfo, edgeUseInfo);
  setUseIds.run(g);
  loopUseInfo.idsToDevice();
  edgeUseInfo.idsToDevice();
  mdl.loopUseIds = loopUseInfo.ids;
  mdl.edgeUseIds = edgeUseInfo.ids;

  EntToAdjUse<pGVertex, pGEdgeUse> v2eu(vtxInfo);
  EntToAdjUse<pGEdge, pGEdgeUse> e2eu(edgeInfo);
  EntToAdjUse<pGFace, pGLoopUse> f2lu(faceInfo);
  EntToAdjUse<pGLoopUse, pGEdgeUse> lu2eu(loopUseInfo);
  Adjacencies adj{v2eu, e2eu, f2lu, lu2eu};

  CountUses countUses(adj);
  countUses.run(g);
  f2lu.degreeToOffset();
  e2eu.degreeToOffset();
  v2eu.degreeToOffset();
  lu2eu.degreeToOffset();
  SetUses setUses(adj);
  setUses.run(g);

  GM_release(g);
  return mdl;
}

}//end namespace Omega_h

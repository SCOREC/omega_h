#include <SimModel.h>
#include <SimUtil.h>
#include "Omega_h_model2d.hpp"
#include "Omega_h_profile.hpp"
#include "Omega_h_map.hpp"  // get_degrees
#include "Omega_h_adj.hpp"  // invert_adj
#include "Omega_h_array_ops.hpp" // operator==(LOs,LOs)
#include <map>
#include <algorithm> //std::fill
#include <numeric> //std::exclusive_scan

namespace Omega_h {

HostWrite<LO> createAndInitArray(size_t size, const LO init=0) {
  auto array = HostWrite<LO>(size);
  std::fill(array.data(), array.data()+array.size(), init);
  return array;
}

HostWrite<LO> createArray(size_t size) {
  return HostWrite<LO>(size);
}

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
  using EntType = pGVertex;
  Reals coords;
};

struct EdgeInfo : public EntInfo {};
struct FaceInfo : public EntInfo {};

struct UseInfoPrecursor {
  UseInfoPrecursor() : idx(0) {}
  std::map<int,int> idToIdx;
  std::vector<LO> ids_h;
  HostWrite<LO> dir_h;
  int idx;
  virtual void initDir() {
    assert(ids_h.size() > 0);
    dir_h = createArray(ids_h.size());
  }
};

struct LoopUseInfoPrecursor : public UseInfoPrecursor {
  static const int DirUndefined = -1;
  LoopUseInfoPrecursor() = default;
  void initDir() override {
    dir_h = createAndInitArray(ids_h.size(), DirUndefined);
  }
  void setDir(const int luIdx, const int dir) {
    if ( dir_h[luIdx] == DirUndefined ) {
      dir_h[luIdx] = dir;
    }
  }
};

using EdgeUseInfoPrecursor = UseInfoPrecursor;

struct UseInfo : public EntInfo {
  LOs dir;
  LOs toDevice(const std::vector<LO>& ids_h) {
    auto array = HostWrite<LO>(ids_h.size());
    std::copy(ids_h.begin(), ids_h.end(), array.data());
    return LOs(array);
  }
  UseInfo(UseInfoPrecursor& uip) {
    ids = toDevice(uip.ids_h);
    idToIdx = uip.idToIdx;
    dir = LOs(uip.dir_h);
  }
};

using LoopUseInfo = UseInfo;
using EdgeUseInfo = UseInfo;

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

EdgeInfo getEdgeInfo(pGModel mdl) {
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
  return EdgeInfo{LOs(ids_h),idToIdx};
}

FaceInfo getFaceInfo(pGModel mdl) {
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
  return FaceInfo{LOs(ids_h), idToIdx};
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

enum GetUsesMode { //FIXME - remove this
  CountAdj = 1,
  SetAdj = 2
};

//TODO determine EntType from EntInfo... they need to match
template<typename SrcEntType, typename DestEntType>
struct EntToAdjUse : public CSR {
  EntToAdjUse(const EntInfo& entInfo_in, const EntInfo& destEntInfo_in)
    : entInfo(entInfo_in),
      CSR(entInfo_in.ids.size()),
      destEntIdToIdx(destEntInfo_in.idToIdx) {}
  const EntInfo& entInfo;
  const std::map<int, int>& destEntIdToIdx;
  template<GetUsesMode mode> //FIXME - remove this
  void countOrSet(SrcEntType srcEnt, DestEntType destEnt) { //split into two functions: count and set
    static_assert((mode == GetUsesMode::CountAdj || mode == GetUsesMode::SetAdj),
        "countOrSet<mode> called with invalid mode");
    OMEGA_H_TIME_FUNCTION;
    if constexpr (mode == GetUsesMode::CountAdj) {
      ScopedTimer timer("EntToAdjUse::count");
      const auto srcEntId = GEN_tag(srcEnt);
      const auto srcEntIdx = entInfo.idToIdx.at(srcEntId);
      incrementDegree(srcEntIdx);
    } else {
      ScopedTimer timer("EntToAdjUse::set");
      const auto srcEntId = GEN_tag(srcEnt);
      const auto srcEntIdx = entInfo.idToIdx.at(srcEntId);
      const auto destEntId = GEN_tag(destEnt);
      const auto destEntIdx = destEntIdToIdx.at(destEntId);
      setValue(srcEntIdx, destEntIdx);
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
template <typename Operator>
void traverseUses(pGModel mdl, Operator& op) {
  static_assert( std::is_invocable_v<decltype(&Operator::loopUseOp),
                                     Operator&, pGFace, pGLoopUse> );
  static_assert( std::is_invocable_v<decltype(&Operator::edgeUseOp),
                                     Operator&, pGLoopUse, pGEdgeUse> );
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
        op.loopUseOp(modelFace, loopUse);
        auto edgeUses = GLU_edgeUseIter(loopUse);
        pGEdgeUse edgeUse;
        while(edgeUse=GEUIter_next(edgeUses)) {
          op.edgeUseOp(loopUse, edgeUse);
        }
        GEUIter_delete(edgeUses);
      }
      GLUIter_delete(loopUses);
    } //sides
  }
  GFIter_delete(modelFaces);
}

struct SetUseIds {
  LoopUseInfoPrecursor& loopUsePre_;
  EdgeUseInfoPrecursor& edgeUsePre_;
  SetUseIds(LoopUseInfoPrecursor& loopUsePre, EdgeUseInfoPrecursor& edgeUsePre)
    : loopUsePre_(loopUsePre), edgeUsePre_(edgeUsePre) {}

  void loopUseOp(pGFace modelFace, pGLoopUse loopUse) {
    const auto id = GEN_tag(loopUse);
    loopUsePre_.ids_h.push_back(id);
    loopUsePre_.idToIdx[id] = loopUsePre_.idx;
    loopUsePre_.idx++;
  }

  void edgeUseOp(pGLoopUse loopUse, pGEdgeUse edgeUse) {
    const auto id = GEN_tag(edgeUse);
    edgeUsePre_.ids_h.push_back(id);
    edgeUsePre_.idToIdx[id] = edgeUsePre_.idx;
    edgeUsePre_.idx++;
  }
};

struct SetUseDir {
  LoopUseInfoPrecursor& loopUsePre_;
  EdgeUseInfoPrecursor& edgeUsePre_;
  SetUseDir(LoopUseInfoPrecursor& loopUsePre, EdgeUseInfoPrecursor& edgeUsePre)
    : loopUsePre_(loopUsePre), edgeUsePre_(edgeUsePre) {}

  void loopUseOp(pGFace, pGLoopUse) {}

  void edgeUseOp(pGLoopUse loopUse, pGEdgeUse edgeUse) {
    const auto euId = GEN_tag(edgeUse);
    const auto euIdx = edgeUsePre_.idToIdx.at(euId);
    const int dir = GEU_dir(edgeUse);
    edgeUsePre_.dir_h[euIdx] = dir;

    const auto luId = GEN_tag(loopUse);
    const auto luIdx = loopUsePre_.idToIdx.at(luId);
    loopUsePre_.setDir(luIdx, dir);
  }
};


struct Adjacencies {
  EntToAdjUse<pGEdgeUse, pGVertex> eu2v;
  EntToAdjUse<pGEdge, pGEdgeUse> e2eu;
  EntToAdjUse<pGLoopUse, pGFace> lu2f;
  EntToAdjUse<pGEdgeUse, pGLoopUse> eu2lu;

  static const int e2euDegree = 2;
  static const int eu2vDegree = 2;
  static const int eu2luDegree = 1;
  static const int lu2fDegree = 1;

  struct CountUses {
    Adjacencies& adj_;
    CountUses(Adjacencies& adj) : adj_(adj) {}

    void loopUseOp(pGFace modelFace, pGLoopUse loopUse) {
      const auto Mode = GetUsesMode::CountAdj;
      adj_.lu2f.countOrSet<Mode>(loopUse, modelFace);
    }

    void edgeUseOp(pGLoopUse loopUse, pGEdgeUse edgeUse) {
      const auto Mode = GetUsesMode::CountAdj;
      auto edge = GEU_edge(edgeUse);
      adj_.eu2lu.countOrSet<Mode>(edgeUse, loopUse);
      adj_.e2eu.countOrSet<Mode>(edge,edgeUse);
      auto vtx0 = GE_vertex(edge,0);
      adj_.eu2v.countOrSet<Mode>(edgeUse,vtx0);
      auto vtx1 = GE_vertex(edge,1);
      adj_.eu2v.countOrSet<Mode>(edgeUse,vtx1);
    }
  };

  struct SetUses {
    Adjacencies& adj_;
    SetUses(Adjacencies& adj) : adj_(adj) {}

    void loopUseOp(pGFace modelFace, pGLoopUse loopUse) {
      const auto Mode = GetUsesMode::SetAdj;
      adj_.lu2f.countOrSet<Mode>(loopUse, modelFace);
    }

    void edgeUseOp(pGLoopUse loopUse, pGEdgeUse edgeUse) {
      const auto Mode = GetUsesMode::SetAdj;
      auto edge = GEU_edge(edgeUse);
      adj_.eu2lu.countOrSet<Mode>(edgeUse, loopUse);
      adj_.e2eu.countOrSet<Mode>(edge,edgeUse);
      auto vtx0 = GE_vertex(edge,0);
      adj_.eu2v.countOrSet<Mode>(edgeUse, vtx0);
      auto vtx1 = GE_vertex(edge,1);
      adj_.eu2v.countOrSet<Mode>(edgeUse, vtx1);
    }
  };

  void checkDegree(const HostWrite<LO>& offset, int degree) {
    const auto deg = get_degrees(LOs(offset));
    const auto exp_deg = LOs(deg.size(), degree);
    OMEGA_H_CHECK(deg == exp_deg);
  }

  Adjacencies(pGModel g, const VtxInfo& vi, const EdgeInfo& ei, const FaceInfo& fi,
              const EdgeUseInfo& eui, const LoopUseInfo& lui) :
      eu2v(eui, vi), e2eu(ei, eui), lu2f(lui, fi), eu2lu(eui, lui) {
    CountUses countUses(*this);
    traverseUses(g, countUses);
    lu2f.degreeToOffset();
    e2eu.degreeToOffset();
    eu2v.degreeToOffset();
    eu2lu.degreeToOffset();
    SetUses setUses(*this);
    traverseUses(g, setUses);
    //check degree
    checkDegree(lu2f.offset, lu2fDegree);
    checkDegree(eu2lu.offset, eu2luDegree);
    checkDegree(eu2v.offset, eu2vDegree);
    checkDegree(e2eu.offset, e2euDegree);
  }
};

void setVertexInfo(Model2D& mdl, const VtxInfo& vtxInfo) {
  mdl.vtxIds = vtxInfo.ids;
  mdl.vtxCoords = vtxInfo.coords;
}

void setEdgeIds(Model2D& mdl, const EdgeInfo& edgeInfo) {
  mdl.edgeIds = edgeInfo.ids;
}

void setFaceIds(Model2D& mdl, const FaceInfo& faceInfo) {
  mdl.faceIds = faceInfo.ids;
}

void setLoopUseIdsAndDir(Model2D& mdl, const LoopUseInfo& loopUseInfo) {
  mdl.loopUseIds = loopUseInfo.ids;
  mdl.loopUseOrientation = loopUseInfo.dir;
}

void setEdgeUseIdsAndDir(Model2D& mdl, const EdgeUseInfo& edgeUseInfo) {
  mdl.edgeUseIds = edgeUseInfo.ids;
  mdl.edgeUseOrientation = edgeUseInfo.dir;
}

void setAdjInfo(Model2D& mdl, Adjacencies& adj) {
  mdl.edgeToEdgeUse = Graph(LOs(adj.e2eu.offset), LOs(adj.e2eu.values));
  mdl.edgeUseToVtx = LOs(adj.eu2v.values);
  mdl.loopUseToFace = LOs(adj.lu2f.values);
  mdl.edgeUseToLoopUse = LOs(adj.eu2lu.values);

  //The last two arguments to 'invert_adj(...)' are 'high' and 'low' entity
  //dimensions and are used to define names for the graph arrays. They have no
  //impact on the graph inversion.
  mdl.vtxToEdgeUse = invert_adj(Adj(mdl.edgeUseToVtx),
                                Adjacencies::eu2vDegree,
                                mdl.vtxIds.size(), 1, 0);
  mdl.loopUseToEdgeUse = invert_adj(Adj(mdl.edgeUseToLoopUse),
                                    Adjacencies::eu2luDegree,
                                    mdl.loopUseIds.size(), 1, 1);
  mdl.faceToLoopUse = invert_adj(Adj(mdl.loopUseToFace), 
                                 Adjacencies::lu2fDegree, 
                                 mdl.faceIds.size(), 1, 2);
}

Model2D Model2D::SimModel2D_load(std::string const& filename) {
  OMEGA_H_TIME_FUNCTION;
  pNativeModel nm = NULL;
  pProgress p = NULL;
  pGModel g = GM_load(filename.c_str(), nm, p);
  const char* msg2d = "Simmetrix GeomSim model is not 2D... exiting\n";
  OMEGA_H_CHECK_MSG(isModel2D(g), msg2d);
  const char* msgValid = "Simmetrix GeomSim model is not valid... exiting\n";
  OMEGA_H_CHECK_MSG(isValid(g), msgValid);

  //collect per entity info
  const auto vtxInfo = getVtxInfo(g);
  const auto edgeInfo = getEdgeInfo(g);
  const auto faceInfo = getFaceInfo(g);

  // use info requires traverals
  auto loopUsePre = LoopUseInfoPrecursor();
  auto edgeUsePre = EdgeUseInfoPrecursor();
  SetUseIds setUseIds(loopUsePre, edgeUsePre);
  traverseUses(g, setUseIds);
  loopUsePre.initDir();
  edgeUsePre.initDir();
  SetUseDir setUseDir(loopUsePre, edgeUsePre);
  traverseUses(g, setUseDir);
  auto edgeUseInfo = EdgeUseInfo(edgeUsePre);
  auto loopUseInfo = LoopUseInfo(loopUsePre);

  //collect adjacency info
  Adjacencies adj(g, vtxInfo, edgeInfo, faceInfo, edgeUseInfo, loopUseInfo);

  //setup model
  auto mdl = Model2D();
  setVertexInfo(mdl, vtxInfo);
  setEdgeIds(mdl, edgeInfo);
  setFaceIds(mdl, faceInfo);
  setLoopUseIdsAndDir(mdl, loopUseInfo);
  setEdgeUseIdsAndDir(mdl, edgeUseInfo);
  setAdjInfo(mdl, adj);

  GM_release(g);
  return mdl;
}

}//end namespace Omega_h

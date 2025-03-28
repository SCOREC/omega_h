#include "Omega_h_meshsim.hpp"

namespace {
  int classId(pEntity e) {
    pGEntity g = EN_whatIn(e);
    assert(g);
    return GEN_tag(g);
  }

  int classType(pEntity e) {
    pGEntity g = EN_whatIn(e);
    assert(g);
    assert((0 <= GEN_type(g)) && (3 >= GEN_type(g)));
    return GEN_type(g);
  }

  int getNumber(pMeshNex nex, pEntity e, int order=0) {
    if(!MeshNex_hasNode(nex, e, order)) {
      fprintf(stderr, "ERROR: entity type %d id %d has no MeshNex node\n", EN_type(e), EN_id(e));
    }
    assert(MeshNex_hasNode(nex, e, order));
    return MeshNex_node(nex, e, order);
  }
}

namespace Omega_h {

namespace meshsim {


SimMesh::SimMesh(std::vector <pVertex> vertices, std::vector <pEdge> edges, 
          std::vector <pFace> faces, std::vector <pRegion> regions) {
  mV = vertices; 
  mE = edges;
  mF = faces;
  mR = regions;
}

SimMesh::SimMesh(pMesh m) {

  // Regions
  RIter regions = M_regionIter(m);
  pRegion rgn;
  while ((rgn = (pRegion) RIter_next(regions))) {
    mR.push_back(rgn);
  }
  RIter_delete(regions);

  // Faces
  FIter faces = M_faceIter(m);
  pFace face;
  while ((face = (pFace) FIter_next(faces))) {
    mF.push_back(face);
  }
  FIter_delete(faces);

  // Edges
  EIter edges = M_edgeIter(m);
  pEdge edge;
  while ((edge = (pEdge) EIter_next(edges))) {
    mE.push_back(edge);
  }
  EIter_delete(edges);
  
  // Vertices
  VIter vertices = M_vertexIter(m);
  pVertex vert;
  while ((vert = (pVertex) VIter_next(vertices))) {
    mV.push_back(vert);
  }
  VIter_delete(vertices);
}

SimMeshInfo getSimMeshInfo(const SimMesh& mesh) {
  SimMeshInfo info = {0,0,0,0,0,0,false,false};

  pRegion rgn;
  for (int i = 0; i < mesh.mR.size(); i++) {
    rgn = mesh.mR[i];
    if (R_topoType(rgn) == Rtet) {
      info.count_tet += 1;
    }
    else if (R_topoType(rgn) == Rhex) {
      info.count_hex += 1;
    }
    else if (R_topoType(rgn) == Rwedge) {
      info.count_wedge += 1;
    }
    else if (R_topoType(rgn) == Rpyramid) {
      info.count_pyramid += 1;
    }
    else {
      Omega_h_fail("Region is not tet, hex, wedge, or pyramid \n");
    }
  }

  pFace face;
  for (int i = 0; i < mesh.mF.size(); i++) {
    face = mesh.mF[i];
    if (F_numEdges(face) == 3) {
      info.count_tri += 1;
    }
    else if (F_numEdges(face) == 4) {
      info.count_quad += 1;
    }
    else {
      Omega_h_fail ("Face is neither tri nor quad \n");
    }
  }

  info.is_simplex = false;
  info.is_hypercube = false;
  if (info.count_hex == 0 && info.count_wedge == 0 && info.count_pyramid == 0) {
    if (info.count_tet == 0) {
      if (info.count_tri > 0) {
        info.is_simplex = true;
      }
    }
    else if (info.count_tet > 0) {
      info.is_simplex = true;
    }
    else {
      Omega_h_fail ("Invaild topology type\n");
    }
  }

  if (info.count_tet == 0 && info.count_wedge == 0 && info.count_pyramid == 0) {
    if (info.count_hex == 0) {
      if (info.count_quad > 0) {
        info.is_hypercube = true;
      }
    }
    else if (info.count_hex > 0) {
      info.is_hypercube = true;
    }
    else {
      Omega_h_fail ("Invaild topology type\n");
    }
  }

  fprintf(stderr, "tet=%d, hex=%d, wedge=%d, pyramid=%d\n",
         info.count_tet, info.count_hex, info.count_wedge, info.count_pyramid);
  fprintf(stderr, "tri=%d, quad=%d\n", info.count_tri, info.count_quad);
  return info;
}

SimMeshEntInfo::SimMeshEntInfo(const SimMesh& simMesh, bool hasNumbering_in) {
  hasNumbering = hasNumbering_in;
  mV = simMesh.mV;
  mE = simMesh.mE;
  mF = simMesh.mF;
  mR = simMesh.mR;
  int numV = mV.size();
  int numE = mE.size();
  int numF = mF.size();
  int numR = mR.size(); 
  std::array<int,4> numEnts = {numV, numE, numF, numR};
  maxDim = getMaxDim(numEnts);
}

SimMeshEntInfo::VertexInfo SimMeshEntInfo::readVerts(pMeshNex numbering) {
  const int numVtx = mV.size();
  VertexInfo vtxInfo;
  vtxInfo.coords = HostWrite<Real>(numVtx*maxDim);
  vtxInfo.id = HostWrite<LO>(numVtx);
  vtxInfo.dim = HostWrite<I8>(numVtx);
  if(hasNumbering)
    vtxInfo.numbering = HostWrite<LO>(numVtx);

  pVertex vtx;
  LO v = 0;

  // Check if transformation data is attached on mesh vertices.
  bool coordTransformationOn = false;
  pMeshDataId tData = MD_lookupMeshDataId("tCoord");
  if (EN_getDataPtr(mV[0], tData, NULL))
  {
    coordTransformationOn = true;
    std::cout << "Coordinate Transformation = ON\n";
  }

  for (int i = 0; i <numVtx; i++) {
    vtx = mV[i];
    double xyz[3];
    V_coord(vtx,xyz);
     
     // Convert coordinates from 3d cartesian to 2d cylindrical
     if(coordTransformationOn)
     {
       double* rzphi;  // R,Z,Phi
       EN_getDataPtr((pEntity)vtx, tData, (void**)&rzphi);
       xyz[0] = rzphi[0];
       xyz[1] = rzphi[1];   
       xyz[2] = 0.0;  
    }

    if( maxDim < 3 && xyz[2] != 0 )
      Omega_h_fail("The z coordinate must be zero for a 2d mesh!\n");
    for(int j=0; j<maxDim; j++) {
      vtxInfo.coords[v * maxDim + j] = xyz[j];
    }
    vtxInfo.id[v] = classId(vtx);
    vtxInfo.dim[v] = classType(vtx);
    if(hasNumbering) {
      vtxInfo.numbering[v] = getNumber(numbering,vtx);
    }
    ++v;
  }
  return vtxInfo;
}

void SimMeshEntInfo::setVtxIds(pPList listVerts, const int vtxPerEnt, const int entIdx, HostWrite<LO>& verts) {
  assert (PList_size(listVerts) == vtxPerEnt);
  void *iter = 0;
  int i = 0;
  pVertex vtx;
  while ((vtx = (pVertex) PList_next(listVerts, &iter))) {
    verts[entIdx*vtxPerEnt+i] = EN_id(vtx);
    i++;
  }
  PList_delete(listVerts);
}

SimMeshEntInfo::EntClass SimMeshEntInfo::readEdges() {
  const int numEdges = mE.size();
  EntClass edgeClass;
  edgeClass.verts = HostWrite<LO>(numEdges*2);
  edgeClass.id = HostWrite<LO>(numEdges);
  edgeClass.dim = HostWrite<I8>(numEdges);
  const int vtxPerEdge = 2;
  auto edgeIdx = 0;
    
  pEdge edge;
  for (int i = 0; i < numEdges; i++) {
    edge = mE[i];
    pPList verts = PList_new();
    //there is no E_vertices(edge) function that returns a list
    verts = PList_append(verts, E_vertex(edge,0));
    verts = PList_append(verts, E_vertex(edge,1));
    setVtxIds(verts, vtxPerEdge, edgeIdx, edgeClass.verts);
    edgeClass.id[edgeIdx] = classId(edge);
    edgeClass.dim[edgeIdx] = classType(edge);
    edgeIdx++;
  }
  return edgeClass;
}

SimMeshEntInfo::MixedFaceClass SimMeshEntInfo::readMixedFaces(GO count_tri, GO count_quad) {
  EntClass tri;
  tri.verts = HostWrite<LO>(count_tri*3);
  tri.id = HostWrite<LO>(count_tri);
  tri.dim = HostWrite<I8>(count_tri);
  const int edgePerTri = 3;
  const int vtxPerTri = 3;
  int triIdx = 0;

  EntClass quad;
  quad.verts = HostWrite<LO>(count_quad*4);
  quad.id = HostWrite<LO>(count_quad);
  quad.dim = HostWrite<I8>(count_quad);
  const int edgePerQuad = 4;
  const int vtxPerQuad = 4;
  int quadIdx = 0;

  const int numFaces = mF.size();
  pFace face;
  for (int i = 0; i < numFaces; i++) {
    face = mF[i];
    if (F_numEdges(face) == edgePerTri) {
      pPList verts = F_vertices(face,1);
      setVtxIds(verts, vtxPerTri, triIdx, tri.verts);
      tri.id[triIdx] = classId(face);
      tri.dim[triIdx] = classType(face);
      triIdx++;
    }
    else if (F_numEdges(face) == edgePerQuad) {
      pPList verts = F_vertices(face,1);
      setVtxIds(verts, vtxPerQuad, quadIdx, quad.verts);
      quad.id[quadIdx] = classId(face);
      quad.dim[quadIdx] = classType(face);
      quadIdx++;
    }
    else {
      Omega_h_fail ("Face is neither tri nor quad \n");
    }
  }
  return MixedFaceClass{tri,quad};
}
 
SimMeshEntInfo::EntClass SimMeshEntInfo::readMonoTopoFaces(GO numFaces, LO vtxPerFace) {
  EntClass ents;
  ents.verts = HostWrite<LO>(numFaces*vtxPerFace);
  ents.id = HostWrite<LO>(numFaces);
  ents.dim = HostWrite<I8>(numFaces);
  int faceIdx = 0;

  pFace face;
  for (int i = 0; i < mF.size(); i++) {
    face = mF[i];
    assert(F_numEdges(face) == vtxPerFace);
    pPList verts = F_vertices(face,1);
    setVtxIds(verts, vtxPerFace, faceIdx, ents.verts);
    ents.id[faceIdx] = classId(face);
    ents.dim[faceIdx] = classType(face);
    faceIdx++;
  }
  return ents;
}

SimMeshEntInfo::MixedRgnClass SimMeshEntInfo::readMixedTopoRegions(
      LO numTets, LO numHexs,
      LO numWedges, LO numPyramids) {

  EntClass tet;
  const auto vtxPerTet = 4;
  tet.verts = HostWrite<LO>(numTets*vtxPerTet);
  tet.id = HostWrite<LO>(numTets);
  tet.dim = HostWrite<I8>(numTets);
  auto tetIdx = 0;

  EntClass hex;
  const auto vtxPerHex = 8;
  hex.verts = HostWrite<LO>(numHexs*vtxPerHex);
  hex.id = HostWrite<LO>(numHexs);
  hex.dim = HostWrite<I8>(numHexs);
  auto hexIdx = 0;

  EntClass wedge;
  const auto vtxPerWedge = 6;
  wedge.verts = HostWrite<LO>(numWedges*vtxPerWedge);
  wedge.id = HostWrite<LO>(numWedges);
  wedge.dim = HostWrite<I8>(numWedges);
  auto wedgeIdx = 0;

  EntClass pyramid;
  const auto vtxPerPyramid = 5;
  pyramid.verts = HostWrite<LO>(numPyramids*vtxPerPyramid);
  pyramid.id = HostWrite<LO>(numPyramids);
  pyramid.dim = HostWrite<I8>(numPyramids);
  auto pyramidIdx = 0;

  int numRegions = mR.size();
  pRegion rgn;
  for (int i = 0; i < numRegions; i++) {
    rgn = mR[i];
    if (R_topoType(rgn) == Rtet) {
      pPList verts = R_vertices(rgn,1);
      setVtxIds(verts, vtxPerTet, tetIdx, tet.verts);
      tet.id[tetIdx] = classId(rgn);
      tet.dim[tetIdx] = classType(rgn);
      tetIdx++;
    }
    else if (R_topoType(rgn) == Rhex) {
      pPList verts = R_vertices(rgn,1);
      setVtxIds(verts, vtxPerHex, hexIdx, hex.verts);
      hex.id[hexIdx] = classId(rgn);
      hex.dim[hexIdx] = classType(rgn);
      hexIdx++;
    }
    else if (R_topoType(rgn) == Rwedge) {
      pPList verts = R_vertices(rgn,1);
      setVtxIds(verts, vtxPerWedge, wedgeIdx, wedge.verts);
      wedge.id[wedgeIdx] = classId(rgn);
      wedge.dim[wedgeIdx] = classType(rgn);
      wedgeIdx++;
    }
    else if (R_topoType(rgn) == Rpyramid) {
      pPList verts = R_vertices(rgn,1);
      setVtxIds(verts, vtxPerPyramid, pyramidIdx, pyramid.verts);
      pyramid.id[pyramidIdx] = classId(rgn);
      pyramid.dim[pyramidIdx] = classType(rgn);
      pyramidIdx++;
    }
    else {
      Omega_h_fail ("Region is not tet, hex, wedge, or pyramid \n");
    }
  }
  return MixedRgnClass({tet,hex,wedge,pyramid});
}

SimMeshEntInfo::EntClass SimMeshEntInfo::readMonoTopoRegions(GO numRgn, LO vtxPerRgn) {
  assert(vtxPerRgn == 4 || vtxPerRgn == 8);
  if(vtxPerRgn != 4 && vtxPerRgn != 8) {
    Omega_h_fail("Mono topology 3d mesh must be all tets or all hex\n");
  }

  EntClass rgnClass;
  rgnClass.verts = HostWrite<LO>(numRgn*vtxPerRgn);
  rgnClass.id = HostWrite<LO>(numRgn);
  rgnClass.dim = HostWrite<I8>(numRgn);

  const auto rgnType = ( vtxPerRgn == 4 ) ? Rtet : Rhex;
  pRegion rgn;
  int rgnIdx = 0;
  for (int i = 0; i < mR.size(); i++) {
    rgn = mR[i];
    assert(R_topoType(rgn) == rgnType);
    pPList verts = R_vertices(rgn,1);
    setVtxIds(verts, vtxPerRgn, rgnIdx, rgnClass.verts);
    rgnClass.id[rgnIdx] = classId(rgn);
    rgnClass.dim[rgnIdx] = classType(rgn);
    rgnIdx++;
  }
  return rgnClass;
}

int SimMeshEntInfo::getMaxDim(std::array<int,4> numEnts) {
  int max_dim;
  if (numEnts[3]) {
    max_dim = 3;
  } else if (numEnts[2]) {
    max_dim = 2;
  } else if (numEnts[1]) {
    max_dim = 1;
  } else {
    Omega_h_fail("There were no Elements of dimension higher than zero!\n");
  }
  return max_dim;
}

std::vector <pVertex> getMeshVertices(pMesh m) {
  std::vector <pVertex> v;
  
  VIter vertices = M_vertexIter(m);
  pVertex vtx;
  while ((vtx = (pVertex) VIter_next(vertices))) {
    v.push_back(vtx);
  }
  VIter_delete(vertices);
  
  return v;
}

std::vector <pEdge> getMeshEdges(pMesh m) {
  std::vector <pEdge> e;
  
  EIter edges = M_edgeIter(m);
  pEdge edge;
  while ((edge = (pEdge) EIter_next(edges))) {
    e.push_back(edge);
  }
  EIter_delete(edges);
  
  return e;
}

std::vector <pFace> getMeshFaces (pMesh m) {
  std::vector <pFace> f;
  
  FIter faces = M_faceIter(m);
  pFace face;
  while ((face = (pFace) FIter_next(faces))) {
    f.push_back(face);
  }
  FIter_delete(faces);

  return f;
}

std::vector <pRegion> getMeshRegions (pMesh m) {
  std::vector <pRegion> r;

  RIter regions = M_regionIter(m);
  pRegion rgn;
  while ((rgn = (pRegion) RIter_next(regions))) {
    r.push_back(rgn);
  }
  RIter_delete(regions);
 
  return r;
}

void setEntToMesh(Mesh* mesh, SimMeshEntInfo simEnts, pMeshNex numbering, SimMeshInfo info)
{
  int numVtx = simEnts.mV.size();
  mesh->set_dim(simEnts.maxDim);
  if (info.is_simplex) {
    mesh->set_family(OMEGA_H_SIMPLEX);
  } else if (info.is_hypercube){
    mesh->set_family(OMEGA_H_HYPERCUBE);
  }

  //process verts
  auto vtxInfo = simEnts.readVerts(numbering);
  mesh->set_verts(numVtx);
  mesh->add_coords(vtxInfo.coords.write());
  mesh->add_tag<ClassId>(0, "class_id", 1,
      Read<ClassId>(vtxInfo.id.write()));
  mesh->add_tag<I8>(0, "class_dim", 1,
      Read<I8>(vtxInfo.dim.write()));
  if(simEnts.hasNumbering) {
    mesh->add_tag<LO>(0, "simNumbering", 1,
        Read<LO>(vtxInfo.numbering.write()));
  }

  //process edges
  auto edges = simEnts.readEdges();
  auto ev2v = Read<LO>(edges.verts.write());
  mesh->set_ents(1, Adj(ev2v));
  mesh->add_tag<ClassId>(1, "class_id", 1,
                Read<ClassId>(edges.id.write()));
  mesh->add_tag<I8>(1, "class_dim", 1,
                    Read<I8>(edges.dim.write()));

  //process faces
  if(info.is_simplex) {
    const auto vtxPerTri = 3;
    auto entClass = simEnts.readMonoTopoFaces(info.count_tri, vtxPerTri);
    auto edge2vert = mesh->get_adj(1, 0);
    auto vert2edge = mesh->ask_up(0, 1);
    auto tri2verts = Read<LO>(entClass.verts.write());
    auto down = reflect_down(tri2verts, edge2vert.ab2b, vert2edge,
                             OMEGA_H_SIMPLEX, 2, 1);
    mesh->set_ents(2, down);
    mesh->add_tag<ClassId>(2, "class_id", 1,
                           Read<ClassId>(entClass.id.write()));
    mesh->add_tag<I8>(2, "class_dim", 1,
                      Read<I8>(entClass.dim.write()));
  } else { // hypercube
    const auto vtxPerQuad = 4;
    auto entClass = simEnts.readMonoTopoFaces(info.count_quad, vtxPerQuad);
    auto edge2vert = mesh->get_adj(1, 0);
    auto vert2edge = mesh->ask_up(0, 1);
    auto quad2verts = Read<LO>(entClass.verts.write());
    auto down = reflect_down(quad2verts, edge2vert.ab2b, vert2edge,
                        OMEGA_H_HYPERCUBE, 2, 1);
    mesh->set_ents(2, down);
    mesh->add_tag<ClassId>(2, "class_id", 1,
                           Read<ClassId>(entClass.id.write()));
    mesh->add_tag<I8>(2, "class_dim", 1,
                      Read<I8>(entClass.dim.write()));
  }

  //process regions
  if(simEnts.maxDim == 2)
    return; //there are no regions

  if(info.is_simplex) {
    const auto vtxPerTet = 4;
    auto entClass = simEnts.readMonoTopoRegions(info.count_tet, vtxPerTet);
    auto tri2vert = mesh->ask_down(2, 0);
    auto vert2tri = mesh->ask_up(0, 2);
    auto tet2verts = Read<LO>(entClass.verts.write());
    auto down = reflect_down(tet2verts, tri2vert.ab2b, vert2tri,
        OMEGA_H_SIMPLEX, 3, 2);
    mesh->set_ents(3, down);
    mesh->template add_tag<ClassId>(3, "class_id", 1,
        Read<ClassId>(entClass.id.write()));
    mesh->template add_tag<I8>(3, "class_dim", 1,
        Read<I8>(entClass.dim.write()));
  } else { //hypercube
    const auto vtxPerHex = 8;
    auto entClass = simEnts.readMonoTopoRegions(info.count_hex, vtxPerHex);
    auto quad2vert = mesh->ask_down(2, 0);
    auto vert2quad = mesh->ask_up(0, 2);
    auto hex2verts = Read<LO>(entClass.verts.write());
    auto down = reflect_down(hex2verts, quad2vert.ab2b, vert2quad,
        OMEGA_H_HYPERCUBE, 3, 2);
    mesh->set_ents(3, down);
    mesh->template add_tag<ClassId>(3, "class_id", 1,
        Read<ClassId>(entClass.id.write()));
    mesh->template add_tag<I8>(3, "class_dim", 1,
        Read<I8>(entClass.dim.write()));
  }
}

void readMixed_internal(SimMesh simMesh, MixedMesh* mesh, SimMeshInfo info) {
  assert(!info.is_simplex && !info.is_hypercube);
  if(info.is_simplex || info.is_hypercube) {
      Omega_h_fail("Attempting to use the mixed mesh reader for a mono topology"
                   "mesh (family = simplex|hypercube)!\n");
  }

  const bool hasNumbering = false;

  SimMeshEntInfo simEnts(simMesh,  hasNumbering);
  mesh->set_dim(simEnts.maxDim);
  const int numVtx = simEnts.mV.size();

  // process verts
  auto vtxInfo = simEnts.readVerts(nullptr);
  mesh->set_verts_type(numVtx);
  mesh->add_coords_mix(vtxInfo.coords.write());
  mesh->add_tag<ClassId>(Topo_type::vertex, "class_id", 1,
                Read<ClassId>(vtxInfo.id.write()));
  mesh->add_tag<I8>(Topo_type::vertex, "class_dim", 1,
                    Read<I8>(vtxInfo.dim.write()));

  // process edges
  auto edges = simEnts.readEdges();
  auto ev2v = Read<LO>(edges.verts.write());
  mesh->set_ents(Topo_type::edge, Topo_type::vertex, Adj(ev2v));
  mesh->add_tag<ClassId>(Topo_type::edge, "class_id", 1,
                         Read<ClassId>(edges.id.write()));
  mesh->add_tag<I8>(Topo_type::edge, "class_dim", 1,
                    Read<I8>(edges.dim.write()));

  //process faces
  auto mixedFaceClass = simEnts.readMixedFaces(info.count_tri, info.count_quad);
  auto edge2vert = mesh->get_adj(Topo_type::edge, Topo_type::vertex);
  auto vert2edge = mesh->ask_up(Topo_type::vertex, Topo_type::edge);

  //// tris
  auto tri2verts = Read<LO>(mixedFaceClass.tri.verts.write());
  auto down = reflect_down(tri2verts, edge2vert.ab2b, vert2edge,
      Topo_type::triangle, Topo_type::edge);
  mesh->set_ents(Topo_type::triangle, Topo_type::edge, down);
  mesh->add_tag<ClassId>(Topo_type::triangle, "class_id", 1,
      Read<ClassId>(mixedFaceClass.tri.id.write()));
  mesh->add_tag<I8>(Topo_type::triangle, "class_dim", 1,
      Read<I8>(mixedFaceClass.tri.dim.write()));

  //// quads
  auto quad2verts = Read<LO>(mixedFaceClass.quad.verts.write());
  down = reflect_down(quad2verts, edge2vert.ab2b, vert2edge,
      Topo_type::quadrilateral, Topo_type::edge);
  mesh->set_ents(Topo_type::quadrilateral, Topo_type::edge, down);
  mesh->add_tag<ClassId>(Topo_type::quadrilateral, "class_id", 1,
      Read<ClassId>(mixedFaceClass.quad.id.write()));
  mesh->add_tag<I8>(Topo_type::quadrilateral, "class_dim", 1,
      Read<I8>(mixedFaceClass.quad.dim.write()));

  //process regions
  if(simEnts.maxDim == 2)
    return; //there are no regions

  auto mixedRgnClass = simEnts.readMixedTopoRegions(
      info.count_tet, info.count_hex,
      info.count_wedge, info.count_pyramid);
  auto tri2vert = mesh->ask_down(Topo_type::triangle, Topo_type::vertex);
  auto vert2tri = mesh->ask_up(Topo_type::vertex, Topo_type::triangle);
  auto quad2vert = mesh->ask_down(Topo_type::quadrilateral, Topo_type::vertex);
  auto vert2quad = mesh->ask_up(Topo_type::vertex, Topo_type::quadrilateral);

  /// tets
  auto tet2verts = Read<LO>(mixedRgnClass.tet.verts.write());
  down = reflect_down(tet2verts, tri2vert.ab2b, vert2tri,
      Topo_type::tetrahedron, Topo_type::triangle);
  mesh->set_ents(Topo_type::tetrahedron, Topo_type::triangle, down);
  mesh->add_tag<ClassId>(Topo_type::tetrahedron, "class_id", 1,
      Read<ClassId>(mixedRgnClass.tet.id.write()));
  mesh->add_tag<I8>(Topo_type::tetrahedron, "class_dim", 1,
      Read<I8>(mixedRgnClass.tet.dim.write()));

  /// hexs
  auto hex2verts = Read<LO>(mixedRgnClass.hex.verts.write());
  down = reflect_down(hex2verts, quad2vert.ab2b, vert2quad,
      Topo_type::hexahedron, Topo_type::quadrilateral);
  mesh->set_ents(Topo_type::hexahedron, Topo_type::quadrilateral, down);
  mesh->add_tag<ClassId>(Topo_type::hexahedron, "class_id", 1,
      Read<ClassId>(mixedRgnClass.hex.id.write()));
  mesh->add_tag<I8>(Topo_type::hexahedron, "class_dim", 1,
      Read<I8>(mixedRgnClass.hex.dim.write()));

  /// wedges
  auto wedge2verts = Read<LO>(mixedRgnClass.wedge.verts.write());
  down = reflect_down(wedge2verts, quad2vert.ab2b, vert2quad,
      Topo_type::wedge, Topo_type::quadrilateral);
  mesh->set_ents(Topo_type::wedge, Topo_type::quadrilateral, down);
  down = reflect_down(wedge2verts, tri2vert.ab2b, vert2tri,
      Topo_type::wedge, Topo_type::triangle);
  mesh->set_ents(Topo_type::wedge, Topo_type::triangle, down);
  mesh->add_tag<ClassId>(Topo_type::wedge, "class_id", 1,
      Read<ClassId>(mixedRgnClass.wedge.id.write()));
  mesh->add_tag<I8>(Topo_type::wedge, "class_dim", 1,
      Read<I8>(mixedRgnClass.wedge.dim.write()));

  /// pyramids
  auto pyramid2verts = Read<LO>(mixedRgnClass.pyramid.verts.write());
  down = reflect_down(pyramid2verts, tri2vert.ab2b, vert2tri,
      Topo_type::pyramid, Topo_type::triangle);
  mesh->set_ents(Topo_type::pyramid, Topo_type::triangle, down);
  down = reflect_down(pyramid2verts, quad2vert.ab2b, vert2quad,
      Topo_type::pyramid, Topo_type::quadrilateral);
  mesh->set_ents(Topo_type::pyramid, Topo_type::quadrilateral, down);
  mesh->add_tag<ClassId>(Topo_type::pyramid, "class_id", 1,
      Read<ClassId>(mixedRgnClass.pyramid.id.write()));
  mesh->add_tag<I8>(Topo_type::pyramid, "class_dim", 1,
      Read<I8>(mixedRgnClass.pyramid.dim.write()));
}

void read_internal(SimMesh m, Mesh* mesh, pMeshNex numbering, SimMeshInfo info) {
  assert(info.is_simplex || info.is_hypercube);
  if(!info.is_simplex && !info.is_hypercube) {
      Omega_h_fail("Attempting to use the mono topology reader for a mixed"
                   "mesh (family = mixed)!\n");
  }

  const bool hasNumbering = (numbering != NULL);

  SimMeshEntInfo simEnts(m, hasNumbering);
  setEntToMesh(mesh, simEnts, numbering, info);
}
 
MixedMesh readMixedImpl(filesystem::path const& mesh_fname,
    filesystem::path const& mdl_fname,
    CommPtr comm) {
  MS_init();
  Sim_readLicenseFile(NULL);
#ifdef OMEGA_H_USE_SIMDISCRETE
  SimDiscrete_start(0);
#endif
  pNativeModel nm = NULL;
  pProgress p = NULL;
  pGModel g = GM_load(mdl_fname.c_str(), nm, p);
  pMesh m = M_load(mesh_fname.c_str(), g, p);
  SimMesh simMesh(m);

  auto simMeshInfo = getSimMeshInfo(simMesh);
  auto mesh = MixedMesh(comm->library());
  mesh.set_comm(comm);
  meshsim::readMixed_internal(simMesh, &mesh, simMeshInfo);
  M_release(m);
  GM_release(g);
#ifdef OMEGA_H_USE_SIMDISCRETE
  SimDiscrete_stop(0);
#endif
  MS_exit();
  return mesh;
}

Mesh readImpl(filesystem::path const& mesh_fname, filesystem::path const& mdl_fname,
    filesystem::path const& numbering_fname, CommPtr comm) {
  MS_init();
  Sim_readLicenseFile(NULL);
#ifdef OMEGA_H_USE_SIMDISCRETE
  SimDiscrete_start(0);
#endif
  pNativeModel nm = NULL;
  pProgress p = NULL;
  pGModel g = GM_load(mdl_fname.c_str(), nm, p);
  pMesh m = M_load(mesh_fname.c_str(), g, p);
  SimMesh simMesh(m);

  auto simMeshInfo = getSimMeshInfo(simMesh);
  const bool hasNumbering = (numbering_fname.native() != std::string(""));
  pMeshNex numbering = hasNumbering ? MeshNex_load(numbering_fname.c_str(), m) : nullptr;
  auto mesh = Mesh(comm->library());
  mesh.set_comm(comm);
  mesh.set_parting(OMEGA_H_ELEM_BASED);
  meshsim::read_internal(simMesh, &mesh, numbering, simMeshInfo);
  if(hasNumbering) MeshNex_delete(numbering);
  M_release(m);
  GM_release(g);
#ifdef OMEGA_H_USE_SIMDISCRETE
  SimDiscrete_stop(0);
#endif
  MS_exit();
  return mesh;
}

bool isMixed(filesystem::path const& mesh_fname, filesystem::path const& mdl_fname) {
  MS_init();
  Sim_readLicenseFile(NULL);
#ifdef OMEGA_H_USE_SIMDISCRETE
  SimDiscrete_start(0);
#endif
  pNativeModel nm = NULL;
  pProgress p = NULL;
  pGModel g = GM_load(mdl_fname.c_str(), nm, p);
  pMesh m = M_load(mesh_fname.c_str(), g, p);
  SimMesh simMesh(m);

  auto simMeshInfo = getSimMeshInfo(simMesh);
  M_release(m);
  GM_release(g);
#ifdef OMEGA_H_USE_SIMDISCRETE
  SimDiscrete_stop(0);
#endif
  MS_exit();
  bool isMixed = (!simMeshInfo.is_simplex && !simMeshInfo.is_hypercube);
  return isMixed;
}

Mesh read(filesystem::path const& mesh_fname, filesystem::path const& mdl_fname,
    filesystem::path const& numbering_fname, CommPtr comm) {
  return readImpl(mesh_fname, mdl_fname, numbering_fname, comm);
}

Mesh read(filesystem::path const& mesh_fname, filesystem::path const& mdl_fname,
    CommPtr comm) {
  return readImpl(mesh_fname, mdl_fname, std::string(""), comm);
}

MixedMesh readMixed(filesystem::path const& mesh_fname, filesystem::path const& mdl_fname,
    CommPtr comm) {
  return readMixedImpl(mesh_fname, mdl_fname, comm);
}

}  // namespace meshsim

}  // end namespace Omega_h

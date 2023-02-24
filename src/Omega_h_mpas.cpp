#include "Omega_h_file.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <cctype>
#include <sstream>
#include <unordered_map>

#include "Omega_h_build.hpp"
#include "Omega_h_class.hpp"
#include "Omega_h_element.hpp"
#include "Omega_h_map.hpp"
#include "Omega_h_vector.hpp"

#ifdef OMEGA_H_USE_MPAS
#include <netcdf.h>
#endif  // OMEGA_H_USE_MPAS

namespace {

//TODO read from file
std::vector<std::string> readVtxFieldList(Omega_h::filesystem::path const& filename) {
  return {
    "observedSurfaceVelocityX",
    "observedSurfaceVelocityY",
    "observedSurfaceVelocityUncertainty",
    "sfcMassBal",
    "surfaceAirTemperature",
    "basalHeatFlux",
    "observedThicknessTendency",
    "floatingBasalMassBal"
  };
}

} //end anonymous namespace

namespace Omega_h {

namespace mpas {

#ifndef OMEGA_H_USE_MPAS

Mesh read(filesystem::path const&, const std::vector<std::string>&, CommPtr) {
  Omega_h_fail("recompile with Omega_h_USE_MPAS enabled!\n");
  return Mesh(comm->library()); //silence warning
}

Mesh read(filesystem::path const&, filesystem::path const&, CommPtr) {
  Omega_h_fail("recompile with Omega_h_USE_MPAS enabled!\n");
  return Mesh(comm->library()); //silence warning
}

Mesh read(filesystem::path const&, filesystem::path const&, CommPtr comm) {
  Omega_h_fail("recompile with Omega_h_USE_MPAS enabled!\n");
  return Mesh(comm->library()); //silence warning
}

#else

inline void checkErr(int err) {
  if(err != NC_NOERR) {
    printf("Error: %d %s\n", err, nc_strerror(err));
  }
  OMEGA_H_CHECK(err==NC_NOERR);
}

size_t readDimension(int ncid, const std::string name) {
  int dimId;
  int err = nc_inq_dimid(ncid, name.c_str(), &dimId);
  checkErr(err);
  size_t dim;
  err = nc_inq_dimlen(ncid, dimId, &dim);
  checkErr(err);
  return dim;
}

template<class T>
HostWrite<T> readArray(int ncid, const std::string name, const size_t len) {
  int arrId;
  int err = nc_inq_varid(ncid, name.c_str(), &arrId);
  checkErr(err);
  auto arr = HostWrite<T>(len);
  err = nc_get_var(ncid, arrId, arr.data());
  return arr;
}

HostWrite<Real> createCoordinates(int ncid, bool useCartesianCoords,
    const size_t dim, const size_t nCells) {
  OMEGA_H_CHECK(dim==2); //TODO support 3d
  auto x = useCartesianCoords ? readArray<Real>(ncid, "xCell", nCells) :
                                      readArray<Real>(ncid, "latCell", nCells);
  auto y = useCartesianCoords ? readArray<Real>(ncid, "yCell", nCells) :
                                      readArray<Real>(ncid, "lonCell", nCells);
  HostWrite<Real> coords(nCells*dim);
  for(size_t i=0; i<nCells; i++) {
    coords[i*2]   = x[i];
    coords[i*2+1] = y[i];
  }
  return coords;
}

template <class T>
bool anyZeros(T& arr, const size_t start, const size_t end) {
  for(size_t i=start; i<end; i++) {
    if(arr[i] == 0) return true;
  }
  return false;
}

template <class T>
void copySubArray(T& src, T& dest,
    const size_t srcStart, const size_t destStart, const size_t len) {
  for(size_t i=0; i<len; i++) {
    dest[destStart+i] = src[srcStart+i];
  }
}

HostWrite<LO> createElemConnectivity(HostWrite<LO>& mpasConn, const size_t nPrimalVtx) {
  OMEGA_H_CHECK(nPrimalVtx*3 == mpasConn.size());
  //Primal vertices that don't bound three primal cells
  // don't form valid dual triangles.
  //This handles boundaries of the mesh.
  //See figure 5.3 of MPAS v1.0 spec
  int nTri=0;
  for(int i=0; i<nPrimalVtx; i++) {
    if( ! anyZeros(mpasConn, i*3, (i+1)*3) )
      nTri++;
  }
  printf("nTri %d\n", nTri);
  //create omegah connectivity array
  int triIdx=0;
  HostWrite<LO> elm2vtx(nTri*3);
  for(int i=0; i<nPrimalVtx; i++) {
    if( ! anyZeros(mpasConn, i*3, (i+1)*3) ) {
      copySubArray(mpasConn, elm2vtx, i*3, triIdx*3, 3);
      triIdx++;
    }
  }
  //decrement the indices - mpas uses 1-based indexing
  for(int i=0; i<nTri*3; i++) {
    --elm2vtx[i];
    if(elm2vtx[i] < 0) printf("foo %d %d\n", i, elm2vtx[i]);
    OMEGA_H_CHECK(elm2vtx[i]>=0);
  }
  return elm2vtx;
}

void read_internal(int ncid, bool useCartesianCoords,
    const std::vector<std::string>& vtxFieldNames, Mesh* mesh) {
  const size_t dim = 2;
  const size_t nCells = readDimension(ncid, "nCells");
  const size_t nVertices = readDimension(ncid, "nVertices");
  auto coords = createCoordinates(ncid, useCartesianCoords, dim, nCells);
  printf("primal cells verts: %u %u\n", nCells, nVertices);
  auto cellsOnVertices = readArray<LO>(ncid, "cellsOnVertex", nVertices*3);
  std::vector< HostWrite<Real> > fieldVals;
  for(auto name : vtxFieldNames) //primal (polygonal) cell = dual (triangle) vertex
    fieldVals.push_back( readArray<Real>(ncid, name, nCells) );
  auto elm2vtx = createElemConnectivity(cellsOnVertices, nVertices);
  Read<LO> elm2vtx_d(elm2vtx.write());
  Read<Real> coords_d(coords.write());
  build_from_elems_and_coords(mesh, OMEGA_H_SIMPLEX, dim, elm2vtx_d, coords_d);
  for(size_t i=0; i<vtxFieldNames.size(); i++) {
    mesh->add_tag<Real>(OMEGA_H_VERT, vtxFieldNames[i], 1, fieldVals[i].write());
  }
}

Mesh read(int ncid, const std::vector<std::string>& vtxFieldList, CommPtr comm) {
  auto mesh = Mesh(comm->library());
  bool useCartesianCoords = true;
  if (comm->rank() == 0) {
    read_internal(ncid, useCartesianCoords, vtxFieldList, &mesh);
  }
  fprintf(stderr, "done read\n");
  mesh.set_comm(comm);
  mesh.balance();
  return mesh;
}

Mesh read(filesystem::path const& filename, const std::vector<std::string>& vtxFieldList, CommPtr comm) {
  int retval;
  int ncid;
  retval = nc_open(filename.c_str(), NC_NOWRITE, &ncid);
  checkErr(retval);
  auto mesh = mpas::read(ncid, vtxFieldList, comm);
  retval = nc_close(ncid);
  checkErr(retval);
  return mesh;
}

Mesh read(filesystem::path const& filename, filesystem::path const& vtxFieldListFile, CommPtr comm) {
  const auto vtxFieldList = readVtxFieldList(vtxFieldListFile);
  return mpas::read(filename, vtxFieldList, comm);
}

Mesh read(filesystem::path const& filename, CommPtr comm) {
  const std::vector<std::string> vtxFieldList;
  return mpas::read(filename, vtxFieldList, comm);
}

#endif  // OMEGA_H_USE_MPAS

}  // namespace mpas

}  // end namespace Omega_h

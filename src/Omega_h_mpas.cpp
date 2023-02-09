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

namespace Omega_h {

namespace mpas {

#ifndef OMEGA_H_USE_MPAS

Mesh read(filesystem::path const&, CommPtr comm) {
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

template <class T>
bool anyZeros(T& arr, size_t start, size_t end) {
  for(size_t i=start; i<end; i++) {
    if(arr[i] == 0) return true;
  }
  return false;
}

void read_internal(int ncid, bool useCartesianCoords,
    std::vector<std::string>& vtxFieldNames, Mesh* mesh) {
  (void)mesh;
  const size_t nCells = readDimension(ncid, "nCells");
  const size_t nVertices = readDimension(ncid, "nVertices");
  printf("primal cells verts: %u %u\n", nCells, nVertices);
  auto coords0 = useCartesianCoords ? readArray<Real>(ncid, "xCell", nCells) :
                                      readArray<Real>(ncid, "latCell", nCells);
  auto cellsOnVertices = readArray<LO>(ncid, "cellsOnVertex", nVertices*3);
  std::vector< HostWrite<Real> > fieldVals;
  for(auto name : vtxFieldNames) //primal (polygonal) cell = dual (triangle) vertex
    fieldVals.push_back( readArray<Real>(ncid, name, nCells) );

  //Primal vertices that don't bound three primal cells
  // don't form valid dual triangles.
  //This handles boundaries of the mesh.
  //See figure 5.3 of MPAS v1.0 spec
  int nTri=0;
  for(int i=0; i<nVertices; i++) {
    if( ! anyZeros(cellsOnVertices, i*3, (i+1)*3) )
      nTri++;
  }
  printf("nTri %d\n", nTri);

}

Mesh read(int ncid, CommPtr comm) {
  auto mesh = Mesh(comm->library());
  bool useCartesianCoords = true;
  std::vector<std::string> vtxFieldNames = {"observedSurfaceVelocityX", "observedSurfaceVelocityY"};
  if (comm->rank() == 0) {
    read_internal(ncid, useCartesianCoords, vtxFieldNames, &mesh);
  }
  fprintf(stderr, "done read\n");
  mesh.set_comm(comm);
  mesh.balance();
  return mesh;
}

Mesh read(filesystem::path const& filename, CommPtr comm) {
  int retval;
  int ncid;
  retval = nc_open(filename.c_str(), NC_NOWRITE, &ncid);
  checkErr(retval);
  auto mesh = mpas::read(ncid, comm);
  retval = nc_close(ncid);
  checkErr(retval);
  return mesh;
}

#endif  // OMEGA_H_USE_MPAS

}  // namespace mpas

}  // end namespace Omega_h

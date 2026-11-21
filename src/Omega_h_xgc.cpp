#include "Omega_h_file.hpp"

#include <fstream>
#include <string>

#include "Omega_h_build.hpp"
#include "Omega_h_mark.hpp"

namespace Omega_h {

// XGC mesh file format
// based on https://xgc.pppl.gov/html/mesh_file_format.html
// accessed 5/27/2026

namespace xgc {

namespace {

void read_node_file(std::istream& stream, HostWrite<Real>& coords,
    HostWrite<I8>& boundary_markers, bool& has_boundary_markers) {
  Int nverts, dim, nattributes, nboundary_markers;
  stream >> nverts >> dim >> nattributes >> nboundary_markers;
  OMEGA_H_CHECK(nverts > 0);
  OMEGA_H_CHECK(dim == 2);
  OMEGA_H_CHECK(nattributes == 0);
  OMEGA_H_CHECK(nboundary_markers == 0 || nboundary_markers == 1);

  has_boundary_markers = (nboundary_markers == 1);
  coords = HostWrite<Real>(nverts * 2);
  if (has_boundary_markers) {
    boundary_markers = HostWrite<I8>(nverts);
  }

  for (Int i = 0; i < nverts; ++i) {
    Int index;
    Real r, z;
    stream >> index >> r >> z;
    OMEGA_H_CHECK(index == i + 1);
    coords[i * 2] = r;
    coords[i * 2 + 1] = z;
    if (has_boundary_markers) {
      Int bm;
      stream >> bm;
      boundary_markers[i] = static_cast<I8>(bm);
    }
  }
}

void read_ele_file(std::istream& stream, HostWrite<LO>& conn) {
  Int ntris, nodes_per_tri, nattributes;
  stream >> ntris >> nodes_per_tri >> nattributes;
  OMEGA_H_CHECK(ntris > 0);
  OMEGA_H_CHECK(nodes_per_tri == 3);
  OMEGA_H_CHECK(nattributes == 0);

  conn = HostWrite<LO>(ntris * 3);

  for (Int i = 0; i < ntris; ++i) {
    Int index;
    stream >> index;
    for (Int j = 0; j < 3; ++j) {
      Int node;
      stream >> node;
      conn[i * 3 + j] = node - 1;
    }
  }
}

}  // end anonymous namespace

Mesh read(filesystem::path const& basename, CommPtr comm) {
  auto node_path = basename.string() + ".node";
  auto ele_path = basename.string() + ".ele";

  auto mesh = Mesh(comm->library());
  if (comm->rank() == 0) {
    std::ifstream node_stream(node_path);
    if (!node_stream.is_open()) {
      Omega_h_fail("couldn't open \"%s\"\n", node_path.c_str());
    }
    std::ifstream ele_stream(ele_path);
    if (!ele_stream.is_open()) {
      Omega_h_fail("couldn't open \"%s\"\n", ele_path.c_str());
    }

    HostWrite<Real> host_coords;
    HostWrite<I8> host_bm;
    bool has_boundary_markers;
    read_node_file(node_stream, host_coords, host_bm, has_boundary_markers);

    HostWrite<LO> host_conn;
    read_ele_file(ele_stream, host_conn);

    auto ev2v_dev = LOs(host_conn);
    auto coords_dev = Reals(host_coords);
    build_from_elems_and_coords(&mesh, OMEGA_H_SIMPLEX, FACE, ev2v_dev, coords_dev);

    if (has_boundary_markers) {
      mesh.add_tag(VERT, "boundary_marker", 1, Read<I8>(host_bm));
    }
  }
  mesh.set_comm(comm);
  mesh.balance();
  return mesh;
}

void write(filesystem::path const& basename, Mesh* mesh) {
  require_same_tdim_sdim(*mesh);
  OMEGA_H_CHECK(mesh->comm()->size() == 1);
  OMEGA_H_CHECK(mesh->dim() == 2);

  auto node_path = basename.string() + ".node";
  auto ele_path = basename.string() + ".ele";

  auto nverts = mesh->nverts();
  auto nelems = mesh->nelems();
  auto dim = mesh->dim();
  auto coords = mesh->coords_view<2>();
  auto e2v = mesh->ask_elem_verts();

  auto exposed_sides = mark_exposed_sides(mesh);
  auto exposed_nodes = mark_down(mesh, dim - 1, 0, exposed_sides);

  auto h_coords_storage = HostRead<Real>(mesh->retain_coords_storage());

  auto h_coords = make_coords_view<2>(h_coords_storage);
  HostRead<LO> h_e2v(e2v);
  HostRead<I8> h_exposed(exposed_nodes);

  std::ofstream node_stream(node_path);
  if (!node_stream.is_open()) {
    Omega_h_fail("couldn't open \"%s\"\n", node_path.c_str());
  }
  node_stream << nverts << " 2 0 1\n";
  for (LO i = 0; i < nverts; ++i) {
    node_stream << i + 1 << " " << h_coords(i, 0) << " "
                << h_coords(i, 1) << " "
                << static_cast<int>(h_exposed[i]) << "\n";
  }
  node_stream << "\n";
  node_stream.close();

  std::ofstream ele_stream(ele_path);
  if (!ele_stream.is_open()) {
    Omega_h_fail("couldn't open \"%s\"\n", ele_path.c_str());
  }
  ele_stream << nelems << " 3 0\n";
  for (LO i = 0; i < nelems; ++i) {
    ele_stream << i + 1;
    for (LO j = 0; j < dim + 1; ++j) {
      ele_stream << " " << h_e2v[i * (dim + 1) + j] + 1;
    }
    ele_stream << "\n";
  }
  ele_stream << "\n";
  ele_stream.close();
}

}  // namespace xgc

}  // end namespace Omega_h

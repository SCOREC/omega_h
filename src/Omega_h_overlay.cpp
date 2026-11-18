#include <Omega_h_overlay.hpp>

#include <Omega_h_build.hpp>
#include <Omega_h_for.hpp>

namespace Omega_h {

Overlay::Overlay(
    int* argc, char*** argv, const Omega_h::Vector<3>& overlayGridCenter,
    double cellSize, size_t Nx, size_t Ny, size_t Nz
#ifdef OMEGA_H_USE_MPI
    ,
    MPI_Comm comm
#endif
    )
    : cell_size(cellSize),
      library(argc, argv
#ifdef OMEGA_H_USE_MPI
          ,
          comm
#endif
          ),
      mesh(build_box(library.world(), OMEGA_H_HYPERCUBE, Nx * cellSize,
          Ny * cellSize, Nz * cellSize, LO(Nx), LO(Ny), LO(Nz))) {
  silly_cells.reserve(size_t(mesh.nregions()));
  silly_faces.reserve(size_t(mesh.nfaces()));
  silly_edges.reserve(size_t(mesh.nedges()));
  silly_nodes.reserve(size_t(mesh.nverts()));
  for (int i = 0; i < mesh.nregions(); ++i) silly_cells.push_back(size_t(i));
  for (int i = 0; i < mesh.nfaces(); ++i) silly_faces.push_back(size_t(i));
  for (int i = 0; i < mesh.nedges(); ++i) silly_edges.push_back(size_t(i));
  for (int i = 0; i < mesh.nverts(); ++i) silly_nodes.push_back(size_t(i));
  auto coords_storage = mesh.retain_coords_storage();
  auto coords = make_coords_view<3>(coords_storage);
  auto new_coords = Write<double>(mesh.nverts() * 3);
  auto diff = overlayGridCenter - vector_3(Nx * cellSize / 2.0,
                                      Ny * cellSize / 2.0, Nz * cellSize / 2.0);
  auto translate = OMEGA_H_LAMBDA(LO i) {
    auto x = get_vector<3>(coords, i);
    x += diff;
    set_vector(new_coords, i, x);
  };
  parallel_for(mesh.nverts(), translate);
  mesh.set_coords<3>(new_coords);
  mesh.ask_down(FACE, VERT);
  mesh.ask_down(REGION, VERT);
  mesh.ask_down(REGION, FACE);
  mesh.ask_up(FACE, REGION);
}

std::array<size_t, 2> Overlay::get_face_cells(size_t face) const {
  std::array<size_t, 2> cells;
  auto faces2cells = mesh.get_adj(FACE, REGION);
  auto face_i = LO(face);
  auto begin_i = size_t(faces2cells.a2ab[face_i]);
  auto ncells = size_t(faces2cells.a2ab[face_i + 1]) - begin_i;
  for (size_t i = 0; i < ncells; ++i) {
    cells[i] = size_t(faces2cells.ab2b[LO(begin_i + i)]);
  }
  for (size_t i = ncells; i < 2; ++i) {
    cells[i] = get_invalid_cell_handle();
  }
  return cells;
}

const std::vector<size_t>& Overlay::get_cells() const { return silly_cells; }

const std::vector<size_t>& Overlay::get_edges() const { return silly_edges; }

const std::vector<size_t>& Overlay::get_faces() const { return silly_faces; }

const std::vector<size_t>& Overlay::get_nodes() const { return silly_nodes; }

std::array<size_t, 8> Overlay::get_cell_nodes(size_t cell) const {
  std::array<size_t, 8> nodes;
  auto cells2nodes = mesh.get_adj(REGION, VERT).ab2b;
  for (size_t i = 0; i < 8; ++i) {
    nodes[i] = size_t(cells2nodes[LO(cell * 8 + i)]);
  }
  return nodes;
}

std::array<size_t, 6> Overlay::get_cell_faces(size_t cell) const {
  std::array<size_t, 6> faces;
  auto cells2faces = mesh.get_adj(REGION, FACE).ab2b;
  for (size_t i = 0; i < 6; ++i) {
    faces[i] = size_t(cells2faces[LO(cell * 6 + i)]);
  }
  return faces;
}

std::array<size_t, 4> Overlay::get_face_nodes(size_t face) const {
  std::array<size_t, 4> nodes;
  auto faces2nodes = mesh.get_adj(FACE, VERT).ab2b;
  for (size_t i = 0; i < 4; ++i) {
    nodes[i] = size_t(faces2nodes[LO(face * 4 + i)]);
  }
  return nodes;
}

std::array<size_t, 2> Overlay::get_edge_nodes(size_t edge) const {
  std::array<size_t, 2> nodes;
  auto edges2nodes = mesh.get_adj(EDGE, VERT).ab2b;
  for (size_t i = 0; i < 2; ++i) {
    nodes[i] = size_t(edges2nodes[LO(edge * 2 + i)]);
  }
  return nodes;
}

Omega_h::Vector<3> Overlay::get_node_coordinates(size_t node) const {
  auto coords_storage = HostRead<Real>(mesh.retain_coords_storage());
  auto coords = make_coords_view<3>(coords_storage);
  auto v = LO(node);
  return vector_3(coords(v, 0), coords(v, 1), coords(v, 2));
}

Omega_h::Vector<3> Overlay::get_cell_center_location(size_t cell) const {
  HostRead<LO> cells2nodes(mesh.get_adj(REGION, VERT).ab2b);
  auto coords_storage = HostRead<Real>(mesh.retain_coords_storage());
  auto coords = make_coords_view<3>(coords_storage);
  auto center = zero_vector<3>();
  for (Int i = 0; i < 8; ++i) {
    auto v = cells2nodes[LO(cell) * 8 + i];
    for (Int d = 0; d < 3; ++d) center[d] += coords(v, d);
  }
  return center / 8.0;
}

size_t Overlay::get_invalid_cell_handle() const { return ~size_t(0); }

double Overlay::get_cell_size() const { return cell_size; }

}  // end namespace Omega_h

// Tests for ents_on_closure and nodes_on_closure.
//
// Mesh: a 2x1 simplex unit-square (4 triangles, 6 vertices).  The box class
// sets are populated automatically by build_box via get_box_class_sets(2).
//
// test_ents_on_closure covers:
//   - single set + single dimension (vertex count on bottom boundary)
//   - single set + higher dimension (edge count on bottom boundary)
//   - a different boundary set (left boundary vertex count)
//   - multi-set union where two boundary sets share a corner vertex,
//     verifying deduplication
//
// test_nodes_on_closure uses the 1st-order identity (nodes == vertices) as an
// oracle: when nodes2ents[d] = ask_up(VERT, d), the upward-propagation path
// in nodes_on_closure and the downward-propagation path in ents_on_closure
// must produce identical sorted local indices.  Only nodes2ents[mesh.dim()]
// and nodes2ents[EDGE] are populated; the named sets contain only model edges
// (class_dim=1), so the implementation skips the other dimensions.

#include <cstdio>

#include "Omega_h_build.hpp"
#include "Omega_h_library.hpp"
#include "Omega_h_mesh.hpp"

using namespace Omega_h;

// Build a 2D simplex unit-square mesh with a 2x1 grid of quads (4 triangles).
// Six vertices:  (0,0) (0.5,0) (1,0) (0,1) (0.5,1) (1,1)
// Box class sets populated by build_box (see get_box_class_sets(2)):
//   "y-"   -> {dim=1, id=1}   bottom boundary (y=0): 2 edges, 3 verts
//   "x-"   -> {dim=1, id=3}   left boundary   (x=0): 1 edge,  2 verts
//   "body" -> {dim=2, id=4}   interior region
//   "x+"   -> {dim=1, id=5}   right boundary  (x=1): 1 edge,  2 verts
//   "y+"   -> {dim=1, id=7}   top boundary    (y=1): 2 edges, 3 verts

static void test_ents_on_closure(Library* lib) {
  auto mesh = build_box(lib->world(), OMEGA_H_SIMPLEX, 1., 1., 0., 2, 1, 0);
  OMEGA_H_CHECK(mesh.dim() == 2);

  // bottom boundary: 3 vertices at y=0 (x in {0, 0.5, 1})
  auto bot_verts = ents_on_closure(&mesh, {"y-"}, VERT);
  OMEGA_H_CHECK(bot_verts.size() == 3);

  // bottom boundary: 2 edges along y=0
  auto bot_edges = ents_on_closure(&mesh, {"y-"}, EDGE);
  OMEGA_H_CHECK(bot_edges.size() == 2);

  // left boundary: 2 vertices at x=0 (y in {0, 1})
  auto left_verts = ents_on_closure(&mesh, {"x-"}, VERT);
  OMEGA_H_CHECK(left_verts.size() == 2);

  // union of bottom + left: 4 distinct vertices (corner (0,0) is shared)
  auto bot_left_verts = ents_on_closure(&mesh, {"y-", "x-"}, VERT);
  OMEGA_H_CHECK(bot_left_verts.size() == 4);
}

// For a 1st-order mesh, nodes == vertices.  Build nodes2ents from the upward
// vertex-to-entity adjacency; nodes_on_closure must then return identical
// sorted local indices as ents_on_closure(..., VERT).
static void test_nodes_on_closure(Library* lib) {
  auto mesh = build_box(lib->world(), OMEGA_H_SIMPLEX, 1., 1., 0., 2, 1, 0);

  // nodes2ents[d] maps each vertex (= node) to adjacent entities of dim d.
  // nodes2ents[mesh.dim()] is always required.
  // nodes2ents[EDGE] is required here because "y-" and "x-" are model edges
  // (class_dim=1), so the implementation consults nodes2ents[1].
  Graph nodes2ents[4];
  nodes2ents[mesh.dim()] = mesh.ask_up(VERT, mesh.dim());
  nodes2ents[EDGE]       = mesh.ask_up(VERT, EDGE);
  // nodes2ents[VERT] and nodes2ents[REGION] left default-constructed (empty).

  auto bot_nodes = nodes_on_closure(&mesh, {"y-"}, nodes2ents);
  auto bot_verts = ents_on_closure(&mesh, {"y-"}, VERT);
  OMEGA_H_CHECK(bot_nodes == bot_verts);

  auto left_nodes = nodes_on_closure(&mesh, {"x-"}, nodes2ents);
  auto left_verts = ents_on_closure(&mesh, {"x-"}, VERT);
  OMEGA_H_CHECK(left_nodes == left_verts);
}

int main(int argc, char** argv) {
  auto lib = Library(&argc, &argv);
  test_ents_on_closure(&lib);
  test_nodes_on_closure(&lib);
  fprintf(stderr, "done\n");
  return 0;
}

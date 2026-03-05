#include <Omega_h_class_parametric_transfer.hpp>
#include <Omega_h_for.hpp>
#include <Omega_h_map.hpp>
#include <Omega_h_mesh.hpp>

namespace Omega_h {

ClassParametricTransfer::ClassParametricTransfer(BsplineModel2D* model)
    : model_(model) {

  // Build lookup table mapping model edge ID to its endpoint vertices
  // This is done once in the constructor for efficiency
  auto edge_ids_host = HostRead<LO>(model_->getEdgeIds());
  auto edge2edgeUse = model_->getEdgeToEdgeUse();
  auto edge2edgeUse_ab_host = HostRead<LO>(edge2edgeUse.a2ab);
  auto edge2edgeUse_b_host = HostRead<LO>(edge2edgeUse.ab2b);
  auto edgeUse2vtx_host = HostRead<LO>(model_->getEdgeUseToVtx());

  auto nedges = edge_ids_host.size();

  // Find max edge ID to size the array
  LO max_edge_id = 0;
  for (LO i = 0; i < nedges; ++i) {
    if (edge_ids_host[i] > max_edge_id) max_edge_id = edge_ids_host[i];
  }

  // Create lookup table: edge_vtx_lookup[edge_id * 2 + {0,1}] = vertex_id
  Write<LO> edge_vtx_lookup_w((max_edge_id + 1) * 2, -1);
  auto edge_vtx_lookup_host = HostWrite<LO>(edge_vtx_lookup_w);

  for (LO i = 0; i < nedges; ++i) {
    LO model_edge_id = edge_ids_host[i];
    LO edge_use_begin = edge2edgeUse_ab_host[i];
    LO edge_use_end = edge2edgeUse_ab_host[i + 1];

    if (edge_use_end > edge_use_begin) {
      // Get first edge use
      LO first_edge_use_idx = edge2edgeUse_b_host[edge_use_begin];
      LO vtx0 = edgeUse2vtx_host[first_edge_use_idx * 2 + 0];
      LO vtx1 = edgeUse2vtx_host[first_edge_use_idx * 2 + 1];

      edge_vtx_lookup_host[model_edge_id * 2 + 0] = vtx0;
      edge_vtx_lookup_host[model_edge_id * 2 + 1] = vtx1;
    }
  }

  // Store as device-accessible array
  edge_vtx_lookup_ = LOs(edge_vtx_lookup_w);
}

void ClassParametricTransfer::out_of_line_virtual_method() {}

void ClassParametricTransfer::refine(
    Mesh& old_mesh, Mesh& new_mesh, LOs keys2edges, LOs keys2midverts,
    Int prod_dim, LOs keys2prods, LOs prods2new_ents, LOs same_ents2old_ents,
    LOs same_ents2new_ents) {

  // Only handle vertices
  if (prod_dim != VERT) return;

  // Check if tag exists on old mesh
  if (!old_mesh.has_tag(VERT, "class_parametric")) return;

  // Use pre-built lookup table from constructor
  auto edge_vtx_lookup = edge_vtx_lookup_;

  // Get old mesh data (READ FROM OLD_MESH - see CRITICAL ISSUE #6)
  auto old_params = old_mesh.get_array<Real>(VERT, "class_parametric");
  auto edge_class_dim = old_mesh.get_array<I8>(EDGE, "class_dim");
  auto edge_class_id = old_mesh.get_array<LO>(EDGE, "class_id");
  auto vert_class_dim = old_mesh.get_array<I8>(VERT, "class_dim");
  auto vert_class_id = old_mesh.get_array<LO>(VERT, "class_id");
  auto edge2verts = old_mesh.ask_verts_of(EDGE);

  // Allocate output
  auto nnew_verts = new_mesh.nverts();
  Write<Real> new_params_w(nnew_verts * 2);

  // Handle same_ents (GATHER-SCATTER pattern)
  auto same_params = read(unmap(same_ents2old_ents, old_params, 2));
  map_into(same_params, same_ents2new_ents, new_params_w, 2);

  // Compute new midpoint vertices (DEVICE PARALLEL)
  auto nkeys = keys2edges.size();

  auto f = OMEGA_H_LAMBDA(LO key) {
    auto edge = keys2edges[key];
    auto midvert = keys2midverts[key];

    // Only process edges classified on model edges
    auto e_class_dim = edge_class_dim[edge];
    if (e_class_dim != 1) return;

    auto e_class_id = edge_class_id[edge];

    // Get bounding vertices
    auto v0 = edge2verts[edge * 2 + 0];
    auto v1 = edge2verts[edge * 2 + 1];

    auto v0_class_dim = vert_class_dim[v0];
    auto v0_class_id = vert_class_id[v0];
    auto v1_class_dim = vert_class_dim[v1];
    auto v1_class_id = vert_class_id[v1];

    Real v0_param_on_edge;
    Real v1_param_on_edge;

    // Get v0's parametric coordinate on the model edge
    if (v0_class_dim == 1 && v0_class_id == e_class_id) {
      // Vertex on same model edge - use its parametric coord directly
      v0_param_on_edge = old_params[v0 * 2 + 0];
    } else if (v0_class_dim == 0) {
      // Vertex on model vertex - determine its position on model edge (0.0 or 1.0)
      LO model_vert_id = v0_class_id;
      LO edge_vtx0 = edge_vtx_lookup[e_class_id * 2 + 0];
      LO edge_vtx1 = edge_vtx_lookup[e_class_id * 2 + 1];

      if (model_vert_id == edge_vtx0) {
        v0_param_on_edge = 0.0;
      } else if (model_vert_id == edge_vtx1) {
        v0_param_on_edge = 1.0;
      } else {
        // Model vertex doesn't bound this specific edge
        // This can happen with complex topologies - skip this edge
        return;
      }
    } else {
      // Vertex classified on model face or higher dimension
      // Skip - can't determine parametric coordinate
      return;
    }

    // Get v1's parametric coordinate on the model edge
    if (v1_class_dim == 1 && v1_class_id == e_class_id) {
      // Vertex on same model edge - use its parametric coord directly
      v1_param_on_edge = old_params[v1 * 2 + 0];
    } else if (v1_class_dim == 0) {
      // Vertex on model vertex - determine its position on model edge (0.0 or 1.0)
      LO model_vert_id = v1_class_id;
      LO edge_vtx0 = edge_vtx_lookup[e_class_id * 2 + 0];
      LO edge_vtx1 = edge_vtx_lookup[e_class_id * 2 + 1];

      if (model_vert_id == edge_vtx0) {
        v1_param_on_edge = 0.0;
      } else if (model_vert_id == edge_vtx1) {
        v1_param_on_edge = 1.0;
      } else {
        // Model vertex doesn't bound this specific edge
        // This can happen with complex topologies - skip this edge
        return;
      }
    } else {
      // Vertex classified on model face or higher dimension
      // Skip - can't determine parametric coordinate
      return;
    }

    // Average the parametric coordinates
    Real new_param = (v0_param_on_edge + v1_param_on_edge) / 2.0;

    // Store result (2D parametric coords, but only first component used for edges)
    new_params_w[midvert * 2 + 0] = new_param;
    new_params_w[midvert * 2 + 1] = 0.0;
  };

  parallel_for(nkeys, f, "transfer_class_parametric");

  // Set tag on new mesh (internal=true to skip cache invalidation)
  new_mesh.add_tag(VERT, "class_parametric", 2, Read<Real>(new_params_w), true);
}

void ClassParametricTransfer::coarsen(
    Mesh& old_mesh, Mesh& new_mesh, LOs keys2verts, Adj keys2doms,
    Int prod_dim, LOs prods2new_ents, LOs same_ents2old_ents,
    LOs same_ents2new_ents) {

  // Only handle vertices
  if (prod_dim != VERT) return;

  // Check if tag exists on old mesh
  if (!old_mesh.has_tag(VERT, "class_parametric")) return;

  auto old_params = old_mesh.get_array<Real>(VERT, "class_parametric");
  auto nnew_verts = new_mesh.nverts();
  Write<Real> new_params_w(nnew_verts * 2);

  // Copy unchanged vertices using gather-scatter
  auto same_params = read(unmap(same_ents2old_ents, old_params, 2));
  map_into(same_params, same_ents2new_ents, new_params_w, 2);

  // Set tag on new mesh
  new_mesh.add_tag(VERT, "class_parametric", 2, Read<Real>(new_params_w), true);
}

void ClassParametricTransfer::swap(
    Mesh& old_mesh, Mesh& new_mesh, Int prod_dim, LOs keys2edges,
    LOs keys2prods, LOs prods2new_ents, LOs same_ents2old_ents,
    LOs same_ents2new_ents) {

  // Only handle vertices
  if (prod_dim != VERT) return;

  // Check if tag exists on old mesh
  if (!old_mesh.has_tag(VERT, "class_parametric")) return;

  auto old_params = old_mesh.get_array<Real>(VERT, "class_parametric");
  auto nnew_verts = new_mesh.nverts();
  Write<Real> new_params_w(nnew_verts * 2);

  // Vertices don't move during swap, just copy
  auto same_params = read(unmap(same_ents2old_ents, old_params, 2));
  map_into(same_params, same_ents2new_ents, new_params_w, 2);

  // Set tag on new mesh
  new_mesh.add_tag(VERT, "class_parametric", 2, Read<Real>(new_params_w), true);
}

void ClassParametricTransfer::swap_copy_verts(
    Mesh& old_mesh, Mesh& new_mesh) {

  // Check if tag exists on old mesh
  if (!old_mesh.has_tag(VERT, "class_parametric")) return;

  // This is called during swap operations to copy all vertex data
  // Since vertices maintain their indices during swap, we can do a direct copy
  auto old_params = old_mesh.get_array<Real>(VERT, "class_parametric");
  auto nverts = old_mesh.nverts();

  OMEGA_H_CHECK(nverts == new_mesh.nverts());

  // Direct copy since vertex indices don't change
  new_mesh.add_tag(VERT, "class_parametric", 2, old_params, true);
}

}  // end namespace Omega_h

#ifndef OMEGA_H_CLASS_PARAMETRIC_TRANSFER_HPP
#define OMEGA_H_CLASS_PARAMETRIC_TRANSFER_HPP

#include <Omega_h_adapt.hpp>
#include <Omega_h_bsplineModel2d.hpp>

namespace Omega_h {

/**
 * Custom transfer handler for the class_parametric tag during mesh adaptation.
 *
 * This class ensures that parametric coordinates are correctly computed for new
 * vertices created during refinement. When a mesh edge is split, the new midpoint
 * vertex needs parametric coordinates that properly interpolate between the
 * endpoint coordinates, accounting for cases where vertices are classified on
 * model vertices (and thus have implicit parametric coordinates of 0.0 or 1.0).
 */
class ClassParametricTransfer : public UserTransfer {
private:
  BsplineModel2D* model_;

  /**
   * Lookup table mapping model edge ID to its endpoint vertices.
   * Format: edge_vtx_lookup_[edge_id * 2 + {0,1}] = vertex_id
   * Built once in constructor for efficiency.
   */
  LOs edge_vtx_lookup_;

public:
  ClassParametricTransfer(BsplineModel2D* model);

  void out_of_line_virtual_method() override;

  /**
   * Handle class_parametric transfer during refinement.
   * Computes parametric coordinates for new midpoint vertices.
   */
  void refine(Mesh& old_mesh, Mesh& new_mesh,
              LOs keys2edges, LOs keys2midverts, Int prod_dim,
              LOs keys2prods, LOs prods2new_ents, LOs same_ents2old_ents,
              LOs same_ents2new_ents) override;

  /**
   * Handle class_parametric transfer during coarsening.
   * Copies parametric coordinates for surviving vertices.
   */
  void coarsen(Mesh& old_mesh, Mesh& new_mesh,
               LOs keys2verts, Adj keys2doms, Int prod_dim,
               LOs prods2new_ents, LOs same_ents2old_ents,
               LOs same_ents2new_ents) override;

  /**
   * Handle class_parametric transfer during edge/face swaps.
   * Vertices don't move, so just copy their parametric coordinates.
   */
  void swap(Mesh& old_mesh, Mesh& new_mesh, Int prod_dim,
            LOs keys2edges, LOs keys2prods, LOs prods2new_ents,
            LOs same_ents2old_ents, LOs same_ents2new_ents) override;

  /**
   * Copy vertex data during swap operations.
   */
  void swap_copy_verts(Mesh& old_mesh, Mesh& new_mesh) override;
};

}  // end namespace Omega_h

#endif

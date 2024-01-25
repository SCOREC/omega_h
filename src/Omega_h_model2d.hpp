#ifndef OMEGA_H_MODEL2D_HPP
#define OMEGA_H_MODEL2D_HPP

#include <Omega_h_graph.hpp>

/* A 2d non-manifold model that supports:
 * - multiple disconnected faces
 * - edges embedded within a face
 * - self-loops and multigraphs
 * - loops with at least one edge
 * It does not support:
 * - shells
 * - wire(frame) edges
 * - strut edges
 * - vertex uses
 * Terminology follows Kevin Weiler's 1986 RPI Ph.D. Thesis
 */

namespace Omega_h {

struct Model2d {
}

}
#endif


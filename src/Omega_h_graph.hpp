#ifndef OMEGA_H_GRAPH_HPP
#define OMEGA_H_GRAPH_HPP

#include <utility>
#include <vector>

#include "Omega_h_array.hpp"

namespace Omega_h {

/**
 * \brief directed graph (as defined by graph theory) in compressed row format
 *
 * \details the typical access pattern: using a serial CPU backend
 * for (LO a = 0; a < na; ++a) {
 *   for (auto ab = a2ab[a]; ab < a2ab[a + 1]; ++ab) {
 *     auto b = ab2b[ab];
 *     // do something with the (a,b) pair
 *   }
 * }
 */
struct Graph {
  OMEGA_H_INLINE Graph() {}
  explicit Graph(LOs ab2b_) : ab2b(ab2b_) {}
  Graph(LOs a2ab_, LOs ab2b_) : a2ab(a2ab_), ab2b(ab2b_) {}
  LOs a2ab; //offset array
  LOs ab2b; //values array
  LO nnodes() const;
  LO nedges() const;
};

/** \brief combine the edges of two graphs that have the same set of nodes */
Graph add_edges(Graph g1, Graph g2);
/** \brief traverse two graphs a2b and b2c to form and return a graph from a2c */
Graph unmap_graph(LOs a2b, Graph b2c);
/**
 * \brief apply reduction operation op to the edge data associated with each source node
 * \param a2b (in) graph from source nodes to edges
 * \param b_data (in) edge data, size = number of edges * width
 * \param width (in) number of data points per edge 
 * \param op (in) the reduction operation, i.e., min, max, sum
 * \return an array with width data points per source node
 */
template <typename T>
Read<T> graph_reduce(Graph a2b, Read<T> b_data, Int width, Omega_h_Op op);
Reals graph_weighted_average_arc_data(
    Graph a2b, Reals ab_weights, Reals ab_data, Int width);
Reals graph_weighted_average(
    Graph a2b, Reals ab_weights, Reals b_data, Int width);
Graph filter_graph_edges(Graph g, Read<I8> keep_edge);
Graph filter_graph_nodes(Graph g, Read<I8> keep_node);
bool operator==(Graph a, Graph b);
Graph identity_graph(LO nnodes);

Graph add_self_edges(Graph g);

template <typename T>
void map_into(Read<T> a_data, Graph a2b, Write<T> b_data, Int width);
template <typename T>
Read<T> map_onto(Read<T> a_data, Graph a2b, LO nb, T init_val, Int width);

#define INST_DECL(T)                                                           \
  extern template Read<T> graph_reduce(Graph, Read<T>, Int, Omega_h_Op);       \
  extern template void map_into(                                               \
      Read<T> a_data, Graph a2b, Write<T> b_data, Int width);                  \
  extern template Read<T> map_onto(                                            \
      Read<T> a_data, Graph a2b, LO nb, T, Int width);
INST_DECL(I8)
INST_DECL(I32)
INST_DECL(I64)
INST_DECL(Real)
#undef INST_DECL

}  // end namespace Omega_h

#endif

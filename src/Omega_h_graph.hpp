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
  LOs a2ab; //nodes-to-edges array
  LOs ab2b; //edges-to-nodes array
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

/**
 * \brief filter a graph by removing edges marked for deletion
 * \param g (in) the input graph
 * \param keep_edge (in) array of size g.nedges() (the number of entries in the
 *        ab2b edges-to-nodes array) where keep_edge[i] == 1 keeps edge i, == 0
 *        removes it
 * \return a new graph with the same nodes but only the edges where keep_edge[i] == 1
 *
 * \details Creates a new graph with all nodes from the input graph but only edges
 * that are marked to be kept. The node indices remain unchanged. Edge indices in the
 * returned graph are compacted (renumbered sequentially starting from 0).
 *
 * Using this API requires a valid graph structure.  Specifically, each entry
 * in the edges array (a2ab) must be a valid node index
 * (i.e., a2ab[j] >=0 && a2ab[j] < g.nnodes() , for j >= 0 && j < g.nedges() ).
 */
Graph filter_graph_edges(Graph g, Read<I8> keep_edge);

/**
 * \brief filter a graph by removing nodes marked for deletion
 * \param g (in) the input graph
 * \param keep_node (in) array of size g.nnodes() (the number of entries in the
 *        a2ab nodes-to-edges array, minus one) where keep_node[i] == 1 keeps
 *        node i, == 0 removes it
 * \return a new graph containing only the kept nodes with edges reindexed to
 *         the new node numbering
 *
 * \details Creates a new graph containing only nodes where keep_node[i] == 1.
 * All edges connecting to removed nodes are also removed.  Remaining nodes
 * are renumbered sequentially starting from 0, preserving their relative order.
 * Edge destinations are updated to reference the new node indices. For example,
 * if nodes [0,2,3] are kept from a four-node graph (node 1 removed), they become
 * new nodes [0,1,2], and all edge destinations are remapped accordingly.
 *
 * Using this API requires a valid graph structure.  Specifically, each entry
 * in the edges array (a2ab) must be a valid node index
 * (i.e., a2ab[j] >=0 && a2ab[j] < g.nnodes() , for j >= 0 && j < g.nedges() ).
 */
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

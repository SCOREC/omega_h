#ifndef OMEGA_H_MAP_HPP
#define OMEGA_H_MAP_HPP

#include <Omega_h_array.hpp>
#include <Omega_h_graph.hpp>

namespace Omega_h {

template <typename T>
void add_into(Read<T> a_data, LOs a2b, Write<T> b_data, Int width);

template <typename T>
void map_into(Read<T> a_data, LOs a2b, Write<T> b_data, Int width);

template <typename T>
void map_value_into(T a_value, LOs a2b, Write<T> b_data);

template <typename T>
void map_into_range(
    Read<T> a_data, LO begin, LO end, Write<T> b_data, Int width);

template <typename T>
Read<T> map_onto(Read<T> a_data, LOs a2b, LO nb, T init_val, Int width);

template <typename T>
Write<T> unmap(LOs a2b, Read<T> b_data, Int width);

template <typename T>
Read<T> unmap_range(LO begin, LO end, Read<T> b_data, Int width);

template <typename T>
Read<T> expand(Read<T> a_data, LOs a2b, Int width);

template <typename T>
void expand_into(Read<T> a_data, LOs a2b, Write<T> b_data, Int width);

template <typename T>
Read<T> permute(Read<T> a_data, LOs a2b, Int width);

LOs multiply_fans(LOs a2b, LOs a2c);

LOs compound_maps(LOs a2b, LOs b2c);

LOs invert_permutation(LOs a2b);

Read<I8> invert_marks(Read<I8> marks);

LOs collect_marked(Read<I8> marks);

Read<I8> mark_image(LOs a2b, LO nb);

void inject_map(LOs a2b, Write<LO> b2a);

LOs invert_injective_map(LOs a2b, LO nb);

/**
 * \brief given the source node index in A for a list of edges
 *        between nodes in set A and B, sorted by their source
 *        node in A, and the number of nodes in A, construct the
 *        map from source nodes to edges (the 'offset' array, 'a2ab').
 * \details the term 'funnel' is in the context of 'incoming' edges
 *          being collected/grouped into the source nodes.  This interface
 *          counts the degree/offset for the source nodes in the 'outgoing'
 *          graph.  The implementation does not use atomics.
 * \param ab2a (in) list of source node indices for edges between nodes in A and
 *                  B, sorted by source node index
 * \param na (in) number of nodes in set A
 * \return map from source nodes to edges (the 'offset' array, 'a2ab')
 *         with size = na + 1
 */
LOs invert_funnel(LOs ab2a, LO na);

/**
 * \brief see invert_map_by_atomics
 */
Graph invert_map_by_sorting(LOs a2b, LO nb);

/**
 * \brief given a bipartite graph from set A to B with nodes
 *        in A having degree 1 and nodes in B having degree > 1,
 *        and the array of indices mapping A to B (a2b), construct
 *        the graph from B to A.
 * \details see Appendix A of Dan Ibanez's 2016 Ph.D. Thesis,
 *  "CONFORMAL MESH ADAPTATION ON HETEROGENEOUS SUPERCOMPUTERS"
 * \param a2b (in) map of indices in A to B
 * \param nb (in) size of set B
 * \param b2ba_name (in) name of offset array in returned Graph
 * \param ba2a_name (in) name of values array in returned Graph
 * \return Graph of B to A
 */
Graph invert_map_by_atomics(LOs const a2b, LO const nb,
    std::string const& b2ba_name = "", std::string const& ba2a_name = "");

LOs get_degrees(LOs offsets, std::string const& name = "");

/**
 * \brief the opposite of invert_funnel
 * \param a2b (in) map from source nodes to edges (the 'offset' array, 'a2ab')
 *        with size = na + 1
 * \return list of source node indices for edges between nodes in A and
 *         B, sorted by source node index
 */
LOs invert_fan(LOs a2b);

Bytes mark_fan_preimage(LOs a2b);

template <typename T>
Read<T> fan_sum(LOs a2b, Read<T> b_data);
template <typename T>
Read<T> fan_max(LOs a2b, Read<T> b_data);
template <typename T>
Read<T> fan_min(LOs a2b, Read<T> b_data);
template <typename T>
Read<T> fan_reduce(LOs a2b, Read<T> b_data, Int width, Omega_h_Op op);

#define INST_T(T)                                                              \
  extern template Read<T> permute(Read<T> a_data, LOs a2b, Int width);         \
  extern template void add_into(                                               \
      Read<T> a_data, LOs a2b, Write<T> b_data, Int width);                    \
  extern template void map_value_into(T a_value, LOs a2b, Write<T> b_data);    \
  extern template void map_into(                                               \
      Read<T> a_data, LOs a2b, Write<T> b_data, Int width);                    \
  extern template void map_into_range(                                         \
      Read<T> a_data, LO begin, LO end, Write<T> b_data, Int width);           \
  extern template Read<T> map_onto(                                            \
      Read<T> a_data, LOs a2b, LO nb, T, Int width);                           \
  extern template Write<T> unmap(LOs a2b, Read<T> b_data, Int width);          \
  extern template Read<T> unmap_range(                                         \
      LO begin, LO end, Read<T> b_data, Int width);                            \
  extern template Read<T> expand(Read<T> a_data, LOs a2b, Int width);          \
  extern template void expand_into(                                            \
      Read<T> a_data, LOs a2b, Write<T> b_data, Int width);                    \
  extern template Read<T> fan_reduce(                                          \
      LOs a2b, Read<T> b_data, Int width, Omega_h_Op op);
INST_T(I8)
INST_T(I32)
INST_T(I64)
INST_T(Real)
#undef INST_T

}  // end namespace Omega_h

#endif

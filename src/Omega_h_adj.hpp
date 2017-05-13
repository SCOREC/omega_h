#ifndef OMEGA_H_ADJ_HPP
#define OMEGA_H_ADJ_HPP

#include <Omega_h_few.hpp>
#include <Omega_h_graph.hpp>
#include <Omega_h_matrix.hpp>
#include <Omega_h_vector.hpp>

namespace Omega_h {

struct Adj : public Graph {
  Adj() {}
  explicit Adj(LOs ab2b_) : Graph(ab2b_) {}
  Adj(LOs ab2b_, Read<I8> codes_) : Graph(ab2b_), codes(codes_) {}
  Adj(LOs a2ab_, LOs ab2b_, Read<I8> codes_)
      : Graph(a2ab_, ab2b_), codes(codes_) {}
  Adj(LOs a2ab_, LOs ab2b_) : Graph(a2ab_, ab2b_) {}
  Adj(Graph g) : Graph(g) {}
  Read<I8> codes;
};

void find_matches(
    Int dim, LOs av2v, LOs bv2v, Adj v2b, LOs* a2b_out, Read<I8>* codes_out);

Adj reflect_down(LOs hv2v, LOs lv2v, Adj v2l, Int high_dim, Int low_dim);

Adj unmap_adjacency(LOs a2b, Adj b2c);

Adj invert_adj(Adj down, Int nlows_per_high, LO nlows);

/* given the vertex lists for high entities,
   create vertex lists for all uses of low
   entities by high entities */
LOs form_uses(LOs hv2v, Int high_dim, Int low_dim);

LOs find_unique(LOs hv2v, Int high_dim, Int low_dim);

/* for each entity (or entity use), sort its vertex list
   and express the sorting transformation as an alignment code */
template <typename T>
Read<I8> get_codes_to_canonical(Int deg, Read<T> ev2v);

Read<I8> find_canonical_jumps(Int deg, LOs canon, LOs e_sorted2e);

/* given entity uses and unique entities,
   both defined by vertex lists, match
   uses to unique entities and derive their
   respective alignment codes.

   even though this is a downward adjacency, we'll
   define the code as describing how to transform
   the boundary entity into the entity use,
   since typically data is being pulled into an element
   from its boundary
*/
template <typename T>
void find_matches_ex(Int deg, LOs a2fv, Read<T> av2v, Read<T> bv2v, Adj v2b,
    LOs* a2b_out, Read<I8>* codes_out);

/* for testing only, internally computes upward
   adjacency */
Adj reflect_down(LOs hv2v, LOs lv2v, LO nv, Int high_dim, Int low_dim);

Adj transit(Adj h2m, Adj m2l, Int high_dim, Int low_dim);

Graph verts_across_edges(Adj e2v, Adj v2e);
Graph edges_across_tris(Adj f2e, Adj e2f);
Graph edges_across_tets(Adj r2e, Adj e2r);
Graph elements_across_sides(
    Int dim, Adj elems2sides, Adj sides2elems, Read<I8> side_is_exposed);

template <Int nhhl>
OMEGA_H_DEVICE Few<LO, nhhl> gather_down(LOs const& hl2l, Int h) {
  Few<LO, nhhl> hhl2l;
  for (Int i = 0; i < nhhl; ++i) {
    auto hl = h * nhhl + i;
    hhl2l[i] = hl2l[hl];
  }
  return hhl2l;
}

template <Int neev>
OMEGA_H_DEVICE Few<LO, neev> gather_verts(LOs const& ev2v, Int e) {
  return gather_down<neev>(ev2v, e);
}

template <Int neev, typename T>
OMEGA_H_DEVICE Few<T, neev> gather_scalars(Read<T> const& a, Few<LO, neev> v) {
  Few<T, neev> x;
  for (Int i = 0; i < neev; ++i) x[i] = a[v[i]];
  return x;
}

template <Int neev, Int dim>
OMEGA_H_DEVICE Few<Vector<dim>, neev> gather_vectors(
    Reals const& a, Few<LO, neev> v) {
  Few<Vector<dim>, neev> x;
  for (Int i = 0; i < neev; ++i) x[i] = get_vector<dim>(a, v[i]);
  return x;
}

template <Int neev, Int dim>
OMEGA_H_DEVICE Few<Matrix<dim, dim>, neev> gather_symms(
    Reals const& a, Few<LO, neev> v) {
  Few<Matrix<dim, dim>, neev> x;
  for (Int i = 0; i < neev; ++i) x[i] = get_symm<dim>(a, v[i]);
  return x;
}

#define INST_DECL(T)                                                           \
  extern template Read<I8> get_codes_to_canonical(Int deg, Read<T> ev2v);      \
  extern template void find_matches_ex(Int deg, LOs a2fv, Read<T> av2v,        \
      Read<T> bv2v, Adj v2b, LOs* a2b_out, Read<I8>* codes_out);
INST_DECL(LO)
INST_DECL(GO)
#undef INST_DECL

}  // end namespace Omega_h

#endif

#include "Omega_h_mixedMesh.hpp"

#include <algorithm>
#include <cctype>
#include <iostream>

#include "Omega_h_array_ops.hpp"
#include "Omega_h_bcast.hpp"
#include "Omega_h_compare.hpp"
#include "Omega_h_element.hpp"
#include "Omega_h_for.hpp"
#include "Omega_h_ghost.hpp"
#include "Omega_h_inertia.hpp"
#include "Omega_h_map.hpp"
#include "Omega_h_mark.hpp"
#include "Omega_h_migrate.hpp"
#include "Omega_h_quality.hpp"
#include "Omega_h_shape.hpp"
#include "Omega_h_timer.hpp"
#include "Omega_h_int_scan.hpp"
#include "Omega_h_atomics.hpp"
#include "Omega_h_reduce.hpp"
#include "Omega_h_print.hpp"
#include "Omega_h_dbg.hpp"

namespace Omega_h {

MixedMesh::MixedMesh() {
  family_ = OMEGA_H_MIXED;
  dim_ = -1;
  for (Int i = 0; i <= 7; ++i) nents_type_[i] = -1;
  library_ = nullptr;
}

MixedMesh::MixedMesh(Library* library_in) : MixedMesh() {
  set_library(library_in);
};

void MixedMesh::set_library(Library* library_in) {
  OMEGA_H_ALWAYS_CHECK(library_in != nullptr);
  library_ = library_in;
}

void MixedMesh::set_comm(CommPtr const& new_comm) {
  OMEGA_H_ALWAYS_CHECK(new_comm->size() == 1);
  comm_ = new_comm;
}

void MixedMesh::set_dim(Int dim_in) {
  OMEGA_H_ALWAYS_CHECK(dim_ == -1);
  OMEGA_H_ALWAYS_CHECK(dim_in >= 1);
  OMEGA_H_ALWAYS_CHECK(dim_in <= 3);
  dim_ = dim_in;
}

void MixedMesh::set_verts_type(LO nverts_in) { nents_type_[int(Topo_type::vertex)] = nverts_in; }

void MixedMesh::set_ents(Topo_type high_type, Topo_type low_type, Adj h2l) {
  OMEGA_H_TIME_FUNCTION;
  check_type(high_type);
  check_type(low_type);
  if (int(high_type) < 6) {
    OMEGA_H_CHECK(!has_ents(high_type));
  }
  auto deg = element_degree(high_type, low_type);
  nents_type_[int(high_type)] = divide_no_remainder(h2l.ab2b.size(), deg);
  add_adj(high_type, low_type, h2l);
}

LO MixedMesh::nents(Topo_type ent_type) const {
  check_type2(ent_type);
  return nents_type_[int(ent_type)];
}

Int MixedMesh::ent_dim(Topo_type ent_type) const {
  Int ent_dim;
  if (int(ent_type) == 0) {
    ent_dim = 0;
  }
  else if (int(ent_type) == 1) {
    ent_dim = 1;
  }
  else if ((int(ent_type) > 1) && (int(ent_type) < 4)) {
    ent_dim = 2;
  }
  else {
    ent_dim = 3;
  }
  return ent_dim;
}

LO MixedMesh::npyrams() const { return nents(Topo_type::pyramid); }

LO MixedMesh::nwedges() const { return nents(Topo_type::wedge); }

LO MixedMesh::nhexs() const { return nents(Topo_type::hexahedron); }

LO MixedMesh::ntets() const { return nents(Topo_type::tetrahedron); }

LO MixedMesh::nquads() const { return nents(Topo_type::quadrilateral); }

LO MixedMesh::ntris() const { return nents(Topo_type::triangle); }

LO MixedMesh::nedges_mix() const { return nents(Topo_type::edge); }

LO MixedMesh::nverts_mix() const { return nents(Topo_type::vertex); }

LO MixedMesh::nregions_mix() const {
  return (nents(Topo_type::tetrahedron) +
          nents(Topo_type::hexahedron) +
          nents(Topo_type::wedge) +
          nents(Topo_type::pyramid));
}

LO MixedMesh::nfaces_mix() const {
  return (nents(Topo_type::triangle) +
          nents(Topo_type::quadrilateral));
}

template <typename T>
void MixedMesh::add_tag(Topo_type ent_type, std::string const& name, Int ncomps) {
  if (has_tag(ent_type, name)) remove_tag(ent_type, name);
  check_type2(ent_type);
  check_tag_name(name);
  OMEGA_H_CHECK(ncomps >= 0);
  OMEGA_H_CHECK(ncomps <= Int(INT8_MAX));
  OMEGA_H_CHECK(tags_type_[int(ent_type)].size() < size_t(INT8_MAX));
  TagPtr ptr(new Tag<T>(name, ncomps));
  tags_type_[int(ent_type)].push_back(std::move(ptr));
}

template <typename T>
void MixedMesh::add_tag(Topo_type ent_type, std::string const& name, Int ncomps,
    Read<T> array, bool internal) {
  check_type2(ent_type);
  auto it = tag_iter(ent_type, name);
  auto had_tag = (it != tags_type_[int(ent_type)].end());
  auto ptr = std::make_shared<Tag<T>>(name, ncomps);
  ptr->set_array(array);
  if (had_tag) {
    OMEGA_H_CHECK(ncomps == ptr->ncomps());
    *it = std::move(ptr);
  } else {
    check_tag_name(name);
    OMEGA_H_CHECK(ncomps >= 0);
    OMEGA_H_CHECK(ncomps <= Int(INT8_MAX));
    OMEGA_H_CHECK(tags_type_[int(ent_type)].size() < size_t(INT8_MAX));
    tags_type_[int(ent_type)].push_back(std::move(ptr));
  }
  OMEGA_H_CHECK(array.size() == nents_type_[int(ent_type)] * ncomps);
  if (!internal) react_to_set_tag(ent_type, name);
}

template <typename T>
void MixedMesh::set_tag(
    Topo_type ent_type, std::string const& name, Read<T> array, bool internal) {
  if (!has_tag(ent_type, name)) {
    Omega_h_fail("set_tag(%s, %s): tag doesn't exist (use add_tag first)\n",
      dimensional_plural_name(ent_type), name.c_str());
  }
  auto it = tag_iter(ent_type,name);
  this->add_tag(ent_type, name, (*it)->ncomps(), array, internal);
}

void MixedMesh::react_to_set_tag(Topo_type ent_type, std::string const& name) {
  bool is_coordinates = (name == "coordinates");
  if ((int(ent_type) == 0) && (is_coordinates || (name == "metric"))) {
    remove_tag(Topo_type::edge, "length");

    remove_tag(Topo_type::pyramid, "quality");
    remove_tag(Topo_type::wedge, "quality");
    remove_tag(Topo_type::hexahedron, "quality");
    remove_tag(Topo_type::tetrahedron, "quality");
    remove_tag(Topo_type::quadrilateral, "quality");
    remove_tag(Topo_type::triangle, "quality");
  }
  if ((int(ent_type) == 0) && is_coordinates) {
    remove_tag(Topo_type::pyramid, "size");
    remove_tag(Topo_type::wedge, "size");
    remove_tag(Topo_type::hexahedron, "size");
    remove_tag(Topo_type::tetrahedron, "size");
    remove_tag(Topo_type::quadrilateral, "size");
    remove_tag(Topo_type::triangle, "size");
  }
}

TagBase const* MixedMesh::get_tagbase(Topo_type ent_type, std::string const& name) const {
  check_type2(ent_type);
  auto it = tag_iter(ent_type, name);
  if (it == tags_type_[int(ent_type)].end()) {
    Omega_h_fail("get_tagbase(%s, %s): doesn't exist\n",
        dimensional_plural_name(ent_type), name.c_str());
  }
  return it->get();
}

template <typename T>
Tag<T> const* MixedMesh::get_tag(Topo_type ent_type, std::string const& name) const {
  return as<T>(get_tagbase(ent_type, name));
}

template <typename T>
Read<T> MixedMesh::get_array(Topo_type ent_type, std::string const& name) const {
  return get_tag<T>(ent_type, name)->array();
}

void MixedMesh::remove_tag(Topo_type ent_type, std::string const& name) {
  if (!has_tag(ent_type, name)) return;
  check_type2(ent_type);
  OMEGA_H_CHECK(has_tag(ent_type, name));
  tags_type_[int(ent_type)].erase(tag_iter(ent_type, name));
}

bool MixedMesh::has_tag(Topo_type ent_type, std::string const& name) const {
  check_type(ent_type);
  if (!has_ents(ent_type)) return false;
  return tag_iter(ent_type, name) != tags_type_[int(ent_type)].end();
}

Int MixedMesh::ntags(Topo_type ent_type) const {
  check_type2(ent_type);
  return static_cast<Int>(tags_type_[int(ent_type)].size());
}

TagBase const* MixedMesh::get_tag(Topo_type ent_type, Int i) const {
  check_type2(ent_type);
  OMEGA_H_CHECK(0 <= i);
  OMEGA_H_CHECK(i <= ntags(ent_type));
  return tags_type_[int(ent_type)][static_cast<std::size_t>(i)].get();
}

bool MixedMesh::has_ents(Topo_type ent_type) const {
  check_type(ent_type);
  return nents_type_[int(ent_type)] >= 0;
}

bool MixedMesh::has_adj(Topo_type from_type, Topo_type to_type) const {
  check_type(from_type);
  check_type(to_type);
  return bool(adjs_type_[int(from_type)][int(to_type)]);
}

Adj MixedMesh::get_adj(Topo_type from_type, Topo_type to_type) const {
  check_type2(from_type);
  check_type2(to_type);
  OMEGA_H_CHECK(has_adj(from_type, to_type));
  return *(adjs_type_[int(from_type)][int(to_type)]);
}

Adj MixedMesh::ask_down(Topo_type from_type, Topo_type to_type) {
  OMEGA_H_CHECK(int(to_type) < int(from_type));
  return ask_adj(from_type, to_type);
}

LOs MixedMesh::ask_verts_of(Topo_type ent_type) { return ask_adj(ent_type, Topo_type::vertex).ab2b; }

Adj MixedMesh::ask_up(Topo_type from_type, Topo_type to_type) {
  OMEGA_H_CHECK(int(from_type) < int(to_type));
  return ask_adj(from_type, to_type);
}

MixedMesh::TagIter MixedMesh::tag_iter(Topo_type ent_type, std::string const& name) {
  return std::find_if(tags_type_[int(ent_type)].begin(), tags_type_[int(ent_type)].end(),
      [&](TagPtr const& a) { return a->name() == name; });
}

MixedMesh::TagCIter MixedMesh::tag_iter(Topo_type ent_type, std::string const& name) const {
  return std::find_if(tags_type_[int(ent_type)].begin(), tags_type_[int(ent_type)].end(),
      [&](TagPtr const& a) { return a->name() == name; });
}

void MixedMesh::check_type(Topo_type ent_type) const {
  OMEGA_H_CHECK(Topo_type::vertex <= ent_type);
  OMEGA_H_CHECK(ent_type <= Topo_type::pyramid);
}

void MixedMesh::check_type2(Topo_type ent_type) const {
  check_type(ent_type);
  OMEGA_H_CHECK(has_ents(ent_type));
}

void MixedMesh::add_adj(Topo_type from_type, Topo_type to_type, Adj adj) {
  check_type2(from_type);
  check_type(to_type);
  OMEGA_H_CHECK(adj.ab2b.exists());
  const int from = int(from_type);
  const int to = int(to_type);

  if (to < from) {
    OMEGA_H_CHECK(!adj.a2ab.exists());
    if (to_type == Topo_type::vertex) {
      OMEGA_H_CHECK(!adj.codes.exists());
    } else {
      OMEGA_H_CHECK(adj.codes.exists());
    }
    OMEGA_H_CHECK(
        adj.ab2b.size() == nents(from_type) * element_degree(from_type, to_type));
  } else {
    if (from < to) {
      OMEGA_H_CHECK(adj.a2ab.exists());
      OMEGA_H_CHECK(adj.codes.exists());
      OMEGA_H_CHECK(
          adj.ab2b.size() == nents(to_type) * element_degree(to_type, from_type));
    }
    OMEGA_H_CHECK(adj.a2ab.size() == nents(from_type) + 1);
  }
  adjs_type_[from][to] = std::make_shared<Adj>(adj);
}

Adj MixedMesh::derive_adj(Topo_type from_type, Topo_type to_type) {
  OMEGA_H_TIME_FUNCTION;
  check_type(from_type);
  check_type2(to_type);
  const int from = int(from_type);
  const int to = int(to_type);
  if (from < to) {
    Adj down = ask_adj(to_type, from_type);
    Int nlows_per_high = element_degree(to_type, from_type);
    LO nlows = nents(from_type);
    Adj up = invert_adj(down, nlows_per_high, nlows, to_type, from_type);
    return up;
  }
  else if (to < from) {
    OMEGA_H_CHECK(to + 1 < from);
    Topo_type mid_type;
    if (to_type == Topo_type::vertex) {
      mid_type = Topo_type::edge;
    }
    else if ((from_type == Topo_type::tetrahedron) ||
             (from_type == Topo_type::pyramid)) {
      mid_type = Topo_type::triangle;
    }
    else {
      mid_type = Topo_type::quadrilateral;
    }
    Adj h2m = ask_adj(from_type, mid_type);
    Adj m2l = ask_adj(mid_type, to_type);
    Adj h2l = transit(h2m, m2l, from_type, to_type, mid_type);
    return h2l;
  }
  /* todo: add second order adjacency derivation */
  Omega_h_fail("can't derive adjacency from %s to %s\n",
      dimensional_plural_name(from_type),
      dimensional_plural_name(to_type));
  OMEGA_H_NORETURN(Adj());
}

Adj MixedMesh::ask_adj(Topo_type from_type, Topo_type to_type) {
  OMEGA_H_TIME_FUNCTION;
  check_type2(from_type);
  check_type2(to_type);
  if (has_adj(from_type, to_type)) {
    return get_adj(from_type, to_type);
  }
  Adj derived = derive_adj(from_type, to_type);
  adjs_type_[int(from_type)][int(to_type)] = std::make_shared<Adj>(derived);
  return derived;
}

void MixedMesh::add_coords_mix(Reals array) {
  add_tag<Real>(Topo_type::vertex, "coordinates", dim(), array);
}

Reals MixedMesh::coords_mix() const { return get_array<Real>(Topo_type::vertex, "coordinates"); }

void get_all_type_tags(MixedMesh* mesh, Int dim, Topo_type ent_type, TagSet* tags) {
  for (Int j = 0; j < mesh->ntags(ent_type); ++j) {
    auto tagbase = mesh->get_tag(ent_type, j);
    (*tags)[size_t(dim)].insert(tagbase->name());
  }
}

#define OMEGA_H_INST(T)                                                        \
  template Tag<T> const* MixedMesh::get_tag<T>(Topo_type ent_type, std::string const& name)    \
      const;                                                                   \
  template Read<T> MixedMesh::get_array<T>(Topo_type ent_type, std::string const& name) const; \
  template void MixedMesh::add_tag<T>(                                              \
      Topo_type ent_type, std::string const& name, Int ncomps);                           \
  template void MixedMesh::add_tag<T>(Topo_type ent_type, std::string const& name, Int ncomps, \
      Read<T> array, bool internal);                                           \
  template void MixedMesh::set_tag(                                                 \
      Topo_type ent_type, std::string const& name, Read<T> array, bool internal);
OMEGA_H_INST(I8)
OMEGA_H_INST(I32)
OMEGA_H_INST(I64)
OMEGA_H_INST(Real)
#undef OMEGA_H_INST

}  // end namespace Omega_h

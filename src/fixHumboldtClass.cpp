/* Fix the classification of the 2d triangular Humboldt mesh.
 *
 * See fixHumboldtClass.md.  Vertices classified on (dim,id) pairs on the left
 * are reclassified onto the pair on the right:
 *
 *   (0,4) -> (2,1)
 *   (0,6) -> (1,3)
 *   (0,7) -> (1,2)
 *
 * and the class_sets are remapped as:
 *
 *   boundary_side_set   (1,1) -> (1,2),(1,3)
 *   boundary_node_set_1 (0,6) -> (1,3)
 *   boundary_node_set_3 (0,7) -> (1,2)
 *   node_set            (0,4) -> (1,2),(1,3),(2,1)
 */

#include <Omega_h_array_ops.hpp>
#include <Omega_h_cmdline.hpp>
#include <Omega_h_file.hpp>
#include <Omega_h_for.hpp>
#include <Omega_h_mesh.hpp>

#include <array>
#include <iostream>
#include <vector>

using namespace Omega_h;

namespace {

struct VertMap {
  Byte from_dim;
  ClassId from_id;
  Byte to_dim;
  ClassId to_id;
};

// (current_class_dim, current_class_id) : (target_class_dim, target_class_id)
std::array<VertMap, 3> const vert_maps = {{
    {0, 4, 2, 1},
    {0, 6, 1, 3},
    {0, 7, 1, 2},
}};

struct SetMap {
  std::string name;
  ClassPair from;
  std::vector<ClassPair> to;
};

std::vector<SetMap> const set_maps = {
    {"boundary_side_set", ClassPair(1, 1), {ClassPair(1, 2), ClassPair(1, 3)}},
    {"boundary_node_set_1", ClassPair(0, 6), {ClassPair(1, 3)}},
    {"boundary_node_set_3", ClassPair(0, 7), {ClassPair(1, 2)}},
    {"node_set", ClassPair(0, 4),
        {ClassPair(1, 2), ClassPair(1, 3), ClassPair(2, 1)}},
};

void fix_vert_classification(Mesh* mesh) {
  auto class_dim = deep_copy(mesh->get_array<Byte>(VERT, "class_dim"));
  auto class_id = deep_copy(mesh->get_array<ClassId>(VERT, "class_id"));
  for (auto const& m : vert_maps) {
    auto from_dim = m.from_dim;
    auto from_id = m.from_id;
    auto to_dim = m.to_dim;
    auto to_id = m.to_id;
    auto f = OMEGA_H_LAMBDA(LO v) {
      if (class_dim[v] == from_dim && class_id[v] == from_id) {
        class_dim[v] = to_dim;
        class_id[v] = to_id;
      }
    };
    parallel_for(mesh->nverts(), f, "fix_vert_classification");
  }
  mesh->set_tag(VERT, "class_dim", Read<Byte>(class_dim));
  mesh->set_tag(VERT, "class_id", Read<ClassId>(class_id));
}

void fix_class_sets(Mesh* mesh) {
  for (auto const& m : set_maps) {
    auto it = mesh->class_sets.find(m.name);
    if (it == mesh->class_sets.end()) {
      std::cerr << "ERROR: class set \"" << m.name << "\" not found\n";
      exit(EXIT_FAILURE);
    }
    it->second = m.to;
  }
}

}  // end anonymous namespace

int main(int argc, char** argv) {
  auto lib = Library(&argc, &argv);
  CmdLine cmdline;
  cmdline.add_arg<std::string>("input.osh");
  cmdline.add_arg<std::string>("output.osh");
  if (!cmdline.parse_final(lib.world(), &argc, argv)) return -1;
  auto path_in = cmdline.get<std::string>("input.osh");
  auto path_out = cmdline.get<std::string>("output.osh");
  auto mesh = binary::read(path_in, lib.world());
  fix_vert_classification(&mesh);
  fix_class_sets(&mesh);
  binary::write(path_out, &mesh);
  return 0;
}

#include <fstream>
#include <iostream>

#include "Omega_h_cmdline.hpp"
#include "Omega_h_file.hpp"
#include "Omega_h_mark.hpp"
#include "Omega_h_mesh.hpp"

Omega_h::Read<Omega_h::I8> mark_exposed_nodes(Omega_h::Mesh* mesh);

int main(int argc, char** argv) {
  auto lib = Omega_h::Library(&argc, &argv);
  auto comm = lib.world();

  Omega_h::CmdLine cmdline;
  cmdline.add_arg<std::string>("mesh-in (osh)");
  cmdline.add_arg<std::string>("mesh-out (xgc)");

  if (!cmdline.parse_final(comm, &argc, argv)) return -1;
  auto mesh_in = cmdline.get<std::string>("mesh-in (osh)");
  auto mesh_out = cmdline.get<std::string>("mesh-out (xgc)");
  auto ele_file = mesh_out + ".ele";
  auto node_file = mesh_out + ".node";

  Omega_h::Mesh mesh(&lib);
  Omega_h::binary::read(mesh_in, lib.world(), &mesh);

  const auto dim = mesh.dim();
  if (dim != 2) {
    // ref
    // https://xgc.pppl.gov/html/mesh_file_format.html#:~:text=XGC%20only%20supports,boundary%20markers%20%3D%201
    std::cerr << "Only 2D meshes are supported in XGC Mesh File Format."
              << std::endl;
    return -1;
  }

  const auto num_elements = mesh.nelems();
  const auto num_vertices = mesh.nverts();
  const auto coords = mesh.coords();
  const auto e2v = mesh.ask_elem_verts();
  const auto exposed_nodes = mark_exposed_nodes(&mesh);

  Omega_h::HostRead coords_host(coords);
  Omega_h::HostRead e2v_host(e2v);
  Omega_h::HostRead exposed_nodes_host(exposed_nodes);

  assert(coords_host.size() == dim * num_vertices);
  assert(e2v_host.size() == num_elements * (dim + 1));

  // write the node file
  std::ofstream node_out(node_file);
  node_out << num_vertices << " 2 0 1\n";
  for (Omega_h::LO i = 0; i < num_vertices; ++i) {
    node_out << i + 1 << " " << coords_host[i * dim] << " "
             << coords_host[i * dim + 1] << " "
             << static_cast<int>(exposed_nodes_host[i]) << "\n";
  }
  node_out << "\n";
  node_out.close();

  // write the ele file
  std::ofstream ele_out(ele_file);
  ele_out << num_elements << " 3 0\n";
  for (Omega_h::LO i = 0; i < num_elements; ++i) {
    ele_out << i + 1;
    for (Omega_h::LO j = 0; j < dim + 1; ++j) {
      ele_out << " "
              << e2v_host[i * (dim + 1) + j] +
                     1;  // convert to 1-based indexing
    }
    ele_out << "\n";
  }
  ele_out << "\n";
  ele_out.close();

  return 0;
}

Omega_h::Read<Omega_h::I8> mark_exposed_nodes(Omega_h::Mesh* mesh) {
  auto exposed_sides = mark_exposed_sides(mesh);
  auto exposed_nodes = mark_down(mesh, mesh->dim() - 1, 0, exposed_sides);

  return exposed_nodes;
}

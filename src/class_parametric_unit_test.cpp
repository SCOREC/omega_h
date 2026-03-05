#include <Omega_h_adapt.hpp>
#include <Omega_h_array_ops.hpp>
#include <Omega_h_bsplineModel2d.hpp>
#include <Omega_h_class_parametric_transfer.hpp>
#include <Omega_h_file.hpp>
#include <Omega_h_library.hpp>
#include <Omega_h_mesh.hpp>
#include <iostream>
#include <cmath>

int main(int argc, char** argv) {
  auto lib = Omega_h::Library(&argc, &argv);
  auto world = lib.world();

  // This is a serial-only unit test
  OMEGA_H_CHECK(world->size() == 1);

  std::cout << "=== ClassParametricTransfer Unit Test ===\n";
  std::cout << "Testing parametric coordinate interpolation during edge refinement\n\n";

  // 1. Load the mesh from file
  auto mesh_path = "meshes/test_single_edge.osh";
  auto model_path = "meshes/test_single_edge.smd";
  auto spline_path = "meshes/test_single_edge_spline.oshb";

  std::cout << "Loading mesh from: " << mesh_path << "\n";
  std::cout << "Loading model from: " << model_path << " and " << spline_path << "\n";

  Omega_h::Mesh mesh(&lib);
  Omega_h::binary::read(mesh_path, world, &mesh);

  // 2. Load the BsplineModel2D
  auto model = Omega_h::BsplineModel2D(model_path, spline_path);

  // 3. Verify initial mesh state
  std::cout << "\nInitial mesh state:\n";
  std::cout << "  Vertices: " << mesh.nverts() << "\n";
  std::cout << "  Edges: " << mesh.nedges() << "\n";

  OMEGA_H_CHECK(mesh.nverts() == 2);
  OMEGA_H_CHECK(mesh.nedges() == 1);

  // Verify classification tags exist
  OMEGA_H_CHECK(mesh.has_tag(Omega_h::VERT, "class_dim"));
  OMEGA_H_CHECK(mesh.has_tag(Omega_h::VERT, "class_id"));
  OMEGA_H_CHECK(mesh.has_tag(Omega_h::EDGE, "class_dim"));
  OMEGA_H_CHECK(mesh.has_tag(Omega_h::EDGE, "class_id"));
  OMEGA_H_CHECK(mesh.has_tag(Omega_h::VERT, "class_parametric"));

  // Verify initial parametric coordinates (copy to host for access)
  auto initial_params_host = Omega_h::HostRead<Omega_h::Real>(
      mesh.get_array<Omega_h::Real>(Omega_h::VERT, "class_parametric"));
  auto vert_class_dim_host = Omega_h::HostRead<Omega_h::I8>(
      mesh.get_array<Omega_h::I8>(Omega_h::VERT, "class_dim"));
  auto vert_class_id_host = Omega_h::HostRead<Omega_h::LO>(
      mesh.get_array<Omega_h::LO>(Omega_h::VERT, "class_id"));

  std::cout << "\nInitial vertex classification and parametric coords:\n";
  for (Omega_h::LO i = 0; i < mesh.nverts(); ++i) {
    std::cout << "  v" << i << ": class_dim=" << (int)vert_class_dim_host[i]
              << ", class_id=" << vert_class_id_host[i]
              << ", param=(" << initial_params_host[i * 2 + 0] << ", "
              << initial_params_host[i * 2 + 1] << ")\n";
  }

  // Expect vertices classified on model vertices (dim=0)
  OMEGA_H_CHECK(vert_class_dim_host[0] == 0);
  OMEGA_H_CHECK(vert_class_dim_host[1] == 0);

  // Expect one vertex at param 0.0, one at param 1.0
  Omega_h::Real param0 = initial_params_host[0 * 2 + 0];
  Omega_h::Real param1 = initial_params_host[1 * 2 + 0];

  bool has_zero = (std::abs(param0) < 1e-10) || (std::abs(param1) < 1e-10);
  bool has_one = (std::abs(param0 - 1.0) < 1e-10) || (std::abs(param1 - 1.0) < 1e-10);

  OMEGA_H_CHECK(has_zero);
  OMEGA_H_CHECK(has_one);

  std::cout << "\nInitial state verification: PASSED\n";

  // 4. Set up adaptation with ClassParametricTransfer
  auto xfer = std::make_shared<Omega_h::ClassParametricTransfer>(&model);

  Omega_h::AdaptOpts opts(&mesh);
  opts.xfer_opts.user_xfer = xfer;
  opts.verbosity = Omega_h::EXTRA_STATS;

  // Force edge refinement by setting a very small max_length
  auto current_max_length = mesh.max_length();
  opts.max_length_allowed = current_max_length * 0.5;

  std::cout << "\nRunning adaptation:\n";
  std::cout << "  Current max edge length: " << current_max_length << "\n";
  std::cout << "  Target max edge length: " << opts.max_length_allowed << "\n";

  // 5. Run adaptation
  Omega_h::adapt(&mesh, opts);

  // 6. Verify results
  std::cout << "\nFinal mesh state:\n";
  std::cout << "  Vertices: " << mesh.nverts() << "\n";
  std::cout << "  Edges: " << mesh.nedges() << "\n";

  OMEGA_H_CHECK(mesh.nverts() == 3);
  OMEGA_H_CHECK(mesh.nedges() == 2);

  // Verify parametric coordinates after refinement (copy to host)
  auto final_params_host = Omega_h::HostRead<Omega_h::Real>(
      mesh.get_array<Omega_h::Real>(Omega_h::VERT, "class_parametric"));
  auto final_class_dim_host = Omega_h::HostRead<Omega_h::I8>(
      mesh.get_array<Omega_h::I8>(Omega_h::VERT, "class_dim"));
  auto final_class_id_host = Omega_h::HostRead<Omega_h::LO>(
      mesh.get_array<Omega_h::LO>(Omega_h::VERT, "class_id"));

  std::cout << "\nFinal vertex classification and parametric coords:\n";
  for (Omega_h::LO i = 0; i < mesh.nverts(); ++i) {
    std::cout << "  v" << i << ": class_dim=" << (int)final_class_dim_host[i]
              << ", class_id=" << final_class_id_host[i]
              << ", param=(" << final_params_host[i * 2 + 0] << ", "
              << final_params_host[i * 2 + 1] << ")\n";
  }

  // Find the new midpoint vertex
  bool found_midpoint = false;
  Omega_h::LO midpoint_idx = -1;
  Omega_h::Real midpoint_param = -1.0;

  for (Omega_h::LO i = 0; i < mesh.nverts(); ++i) {
    Omega_h::Real p = final_params_host[i * 2 + 0];
    // Look for vertex with param close to 0.5
    if (std::abs(p - 0.5) < 1e-6) {
      found_midpoint = true;
      midpoint_idx = i;
      midpoint_param = p;

      // Verify it's classified on the model edge (dim=1)
      OMEGA_H_CHECK(final_class_dim_host[i] == 1);

      std::cout << "\nFound midpoint vertex: v" << i
                << " with param=" << p << "\n";
      std::cout << "  Classified on model edge " << final_class_id_host[i] << "\n";
      break;
    }
  }

  if (!found_midpoint) {
    std::cerr << "\nERROR: No vertex found with parametric coordinate ~0.5!\n";
    std::cerr << "Parametric coordinates found:\n";
    for (Omega_h::LO i = 0; i < mesh.nverts(); ++i) {
      std::cerr << "  v" << i << ": " << final_params_host[i * 2 + 0] << "\n";
    }
  }

  OMEGA_H_CHECK(found_midpoint);

  // Verify original vertices still have params 0.0 and 1.0
  int count_zero = 0;
  int count_one = 0;
  for (Omega_h::LO i = 0; i < mesh.nverts(); ++i) {
    Omega_h::Real p = final_params_host[i * 2 + 0];
    if (std::abs(p) < 1e-10) count_zero++;
    if (std::abs(p - 1.0) < 1e-10) count_one++;
  }

  OMEGA_H_CHECK(count_zero == 1);
  OMEGA_H_CHECK(count_one == 1);

  std::cout << "\n=== Test PASSED ===\n";
  std::cout << "Parametric coordinate transfer works correctly:\n";
  std::cout << "  - Original vertices retained params 0.0 and 1.0\n";
  std::cout << "  - New midpoint vertex has param 0.5\n";

  return 0;
}

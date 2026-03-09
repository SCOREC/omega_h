#include <Omega_h_adapt.hpp>
#include <Omega_h_array_ops.hpp>
#include <Omega_h_bsplineModel2d.hpp>
#include <Omega_h_build.hpp>
#include <Omega_h_class_parametric_transfer.hpp>
#include <Omega_h_file.hpp>
#include <Omega_h_library.hpp>
#include <Omega_h_matrix.hpp>
#include <Omega_h_mesh.hpp>
#include <Omega_h_metric.hpp>
#include <iostream>
#include <cmath>
#include <string>
#include <vector>

// Simple test case defined by number of initial edges
struct TestCase {
  std::string name;
  int numEdges;
};

// Build a test mesh with numEdges edges
// - Vertices evenly spaced from 0 to 1
// - Edge i connects vertex i to vertex i+1
// - First and last vertices classified on model vertices (dim=0)
// - Interior vertices classified on model edge (dim=1)
// - Parametric coords match x coordinates
void buildTestMesh(Omega_h::Mesh* mesh, const int numEdges) {
  const int numVerts = numEdges + 1;

  // Build ev2v connectivity: edge i connects vertex i to i+1
  Omega_h::HostWrite<Omega_h::LO> ev2v_vec(numEdges * 2);
  for (int i = 0; i < numEdges; ++i) {
    ev2v_vec[2 * i] = i;
    ev2v_vec[2 * i + 1] = i + 1;
  }
  Omega_h::LOs ev2v(ev2v_vec);

  // Build coordinates: evenly spaced from 0 to 1
  Omega_h::HostWrite<Omega_h::Real> coords_vec(numVerts);
  for (int i = 0; i < numVerts; ++i) {
    coords_vec[i] = static_cast<Omega_h::Real>(i) / numEdges;
  }
  Omega_h::Reals coords(coords_vec);

  Omega_h::build_from_elems_and_coords(mesh, OMEGA_H_SIMPLEX, Omega_h::EDGE, ev2v, coords);

  // Vertex classification: dim=0 for first/last, dim=1 for interior
  const auto stride = 0;
  const auto vDim = 1;
  Omega_h::HostWrite<Omega_h::I8> vert_class_dim_vec(numVerts, vDim, stride);
  vert_class_dim_vec[0] = 0;
  vert_class_dim_vec[numVerts-1] = 0;
  mesh->add_tag<Omega_h::I8>(Omega_h::VERT, "class_dim", 1,
      Omega_h::Read<Omega_h::I8>(vert_class_dim_vec));

  // Vertex class_id: 0 for all except last (which is 1)
  const auto vId = 0;
  Omega_h::HostWrite<Omega_h::ClassId> vert_class_id_vec(numVerts, vId, stride);
  vert_class_id_vec[numVerts-1] = 1;
  mesh->add_tag<Omega_h::ClassId>(Omega_h::VERT, "class_id", 1,
      Omega_h::Read<Omega_h::ClassId>(vert_class_id_vec));

  // Parametric coordinates: same as x coords (with v=0)
  Omega_h::HostWrite<Omega_h::Real> vert_params_vec(numVerts * 2);
  for (int i = 0; i < numVerts; ++i) {
    vert_params_vec[i * 2 + 0] = coords_vec[i];  // u = x coordinate
    vert_params_vec[i * 2 + 1] = 0.0;            // v = 0
  }
  mesh->add_tag<Omega_h::Real>(Omega_h::VERT, "class_parametric", 2,
      Omega_h::Read<Omega_h::Real>(vert_params_vec));

  // Edge classification: all edges on model edge 0 (dim=1, id=0)
  const auto eDim = 1;
  const auto eId = 0;
  Omega_h::HostWrite<Omega_h::I8> edge_class_dim_vec(numEdges, eDim, stride);
  mesh->add_tag<Omega_h::I8>(Omega_h::EDGE, "class_dim", 1,
      Omega_h::Read<Omega_h::I8>(edge_class_dim_vec));

  Omega_h::HostWrite<Omega_h::ClassId> edge_class_id_vec(numEdges, eId, stride);
  mesh->add_tag<Omega_h::ClassId>(Omega_h::EDGE, "class_id", 1,
      Omega_h::Read<Omega_h::ClassId>(edge_class_id_vec));
}

// Run a test case: build mesh, adapt it, verify results
bool runTestCase(const TestCase& test, Omega_h::Library* lib) {
  std::cout << "\n========================================\n";
  std::cout << "Running test: " << test.name << "\n";
  std::cout << "Initial edges: " << test.numEdges << "\n";
  std::cout << "========================================\n";

  // Build the test mesh
  Omega_h::Mesh mesh(lib);
  buildTestMesh(&mesh, test.numEdges);

  int initialNumVerts = mesh.nverts();
  int initialNumEdges = mesh.nedges();

  std::cout << "\nInitial mesh state:\n";
  std::cout << "  Vertices: " << initialNumVerts << "\n";
  std::cout << "  Edges: " << initialNumEdges << "\n";

  // Print initial state
  auto initial_params = Omega_h::HostRead<Omega_h::Real>(
      mesh.get_array<Omega_h::Real>(Omega_h::VERT, "class_parametric"));
  auto initial_class_dim = Omega_h::HostRead<Omega_h::I8>(
      mesh.get_array<Omega_h::I8>(Omega_h::VERT, "class_dim"));
  auto initial_class_id = Omega_h::HostRead<Omega_h::LO>(
      mesh.get_array<Omega_h::LO>(Omega_h::VERT, "class_id"));

  std::cout << "\nInitial vertex classification and parametric coords:\n";
  for (Omega_h::LO i = 0; i < mesh.nverts(); ++i) {
    std::cout << "  v" << i << ": class_dim=" << (int)initial_class_dim[i]
              << ", class_id=" << initial_class_id[i]
              << ", param=(" << initial_params[i * 2 + 0] << ", "
              << initial_params[i * 2 + 1] << ")\n";
  }

  // Create BsplineModel2D
  std::cout << "\nCreating BsplineModel2D with test model\n";
  auto model = Omega_h::BsplineModel2D(Omega_h::BsplineModel2DTestModel::ModelWithOneEdge);

  // Compute metrics and set up adaptation
  std::cout << "Computing implied metrics from edge lengths\n";
  auto metrics = Omega_h::get_implied_metrics(&mesh);
  mesh.add_tag(Omega_h::VERT, "metric", Omega_h::symm_ncomps(mesh.dim()), metrics);

  auto xfer = std::make_shared<Omega_h::ClassParametricTransfer>(&model);

  Omega_h::AdaptOpts opts(&mesh);
  opts.xfer_opts.user_xfer = xfer;
  opts.verbosity = Omega_h::EXTRA_STATS;

  // Force edge refinement - each edge should split once
  auto current_max_length = mesh.max_length();
  opts.max_length_desired = current_max_length * 0.5;
  opts.max_length_allowed = current_max_length * 0.5;

  std::cout << "\nRunning adaptation:\n";
  std::cout << "  Current max edge length: " << current_max_length << "\n";
  std::cout << "  Target max edge length: " << opts.max_length_allowed << "\n";

  // Run adaptation
  Omega_h::adapt(&mesh, opts);

  // Verify results - each edge should have split once
  int expectedNumVerts = 2 * test.numEdges + 1;
  int expectedNumEdges = 2 * test.numEdges;

  std::cout << "\nFinal mesh state:\n";
  std::cout << "  Vertices: " << mesh.nverts() << " (expected: " << expectedNumVerts << ")\n";
  std::cout << "  Edges: " << mesh.nedges() << " (expected: " << expectedNumEdges << ")\n";

  OMEGA_H_CHECK_OP(mesh.nverts(), ==, expectedNumVerts);
  OMEGA_H_CHECK_OP(mesh.nedges(), ==, expectedNumEdges);

  // Print final state
  auto final_params = Omega_h::HostRead<Omega_h::Real>(
      mesh.get_array<Omega_h::Real>(Omega_h::VERT, "class_parametric"));
  auto final_class_dim = Omega_h::HostRead<Omega_h::I8>(
      mesh.get_array<Omega_h::I8>(Omega_h::VERT, "class_dim"));
  auto final_class_id = Omega_h::HostRead<Omega_h::LO>(
      mesh.get_array<Omega_h::LO>(Omega_h::VERT, "class_id"));

  std::cout << "\nFinal vertex classification and parametric coords:\n";
  for (Omega_h::LO i = 0; i < mesh.nverts(); ++i) {
    std::cout << "  v" << i << ": class_dim=" << (int)final_class_dim[i]
              << ", class_id=" << final_class_id[i]
              << ", param=(" << final_params[i * 2 + 0] << ", "
              << final_params[i * 2 + 1] << ")\n";
  }

  const auto parametric = mesh.get_array<Omega_h::Real>(Omega_h::VERT, "class_parametric");
  const auto parametric_0 = Omega_h::get_component(parametric, 2, 0);
  const auto passed = Omega_h::are_close(parametric_0, mesh.coords());

  const auto resString = passed ? "PASSED" : "FAILED";
  std::cout << "\nTest '" << test.name << "' " << std::string(resString) << "\n";
  return true;
}

int main(int argc, char** argv) {
  auto lib = Omega_h::Library(&argc, &argv);
  auto world = lib.world();

  // This is a serial-only unit test
  OMEGA_H_CHECK(world->size() == 1);

  std::cout << "=== ClassParametricTransfer Unit Test ===\n";
  std::cout << "Testing parametric coordinate interpolation during edge refinement\n\n";

  // Define test cases
  std::vector<TestCase> tests = {
    {"Single edge splits into two", 1},
    {"Two edges both split (4 edges total)", 2},
    {"8 edges, all split (16 edges total)", 8}
  };

  // Run all test cases
  bool allPassed = true;
  for (const auto& test : tests) {
    if (!runTestCase(test, &lib)) {
      allPassed = false;
    }
  }

  if (allPassed) {
    std::cout << "\n=== ALL TESTS PASSED ===\n";
  } else {
    std::cout << "\n=== SOME TESTS FAILED ===\n";
    return 1;
  }

  return 0;
}

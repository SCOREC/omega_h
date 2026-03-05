#include <Omega_h_adapt.hpp>
#include <Omega_h_array_ops.hpp>
#include <Omega_h_class_parametric_transfer.hpp>
#include <Omega_h_cmdline.hpp>
#include <Omega_h_file.hpp>
#include <Omega_h_metric.hpp>
#include <iostream>

int main(int argc, char** argv) {
  auto lib = Omega_h::Library(&argc, &argv);
  Omega_h::CmdLine cmdline;
  cmdline.add_arg<std::string>("mesh.osh");
  cmdline.add_arg<std::string>("geomSimModel.smd");
  cmdline.add_arg<std::string>("splines.oshb");

  auto const world = lib.world();
  if (!cmdline.parse_final(world, &argc, argv)) {
    return -1;
  }

  auto const world_rank = world->rank();
  auto const mesh_path = cmdline.get<std::string>("mesh.osh");
  auto const smd_path = cmdline.get<std::string>("geomSimModel.smd");
  auto const splines_path = cmdline.get<std::string>("splines.oshb");

  if (world_rank == 0) {
    std::cout << "Loading BsplineModel2D from: " << smd_path
              << " and " << splines_path << "\n";
  }

  auto model = Omega_h::BsplineModel2D(smd_path, splines_path);

  Omega_h::Mesh mesh(&lib);
  mesh = Omega_h::binary::read(mesh_path, world, true);

  if (world_rank == 0) {
    std::cout << "Mesh loaded with " << mesh.nverts() << " vertices, "
              << mesh.nedges() << " edges, " << mesh.nfaces() << " faces\n";
  }

  // Check if class_parametric tag exists
  if (!mesh.has_tag(Omega_h::VERT, "class_parametric")) {
    if (world_rank == 0) {
      std::cerr << "ERROR: Mesh does not have class_parametric tag on vertices\n";
    }
    return -1;
  }

  // Save initial parametric coordinates for comparison
  auto initial_params = mesh.get_array<double>(Omega_h::VERT, "class_parametric");
  auto initial_nverts = mesh.nverts();

  if (world_rank == 0) {
    std::cout << "Initial mesh has " << initial_nverts << " vertices\n";
    std::cout << "class_parametric tag has " << initial_params.size()
              << " values (2 per vertex)\n";
  }

  // Set up adaptation with ClassParametricTransfer
  Omega_h::AdaptOpts opts(&mesh);
  opts.xfer_opts.user_xfer = std::make_shared<Omega_h::ClassParametricTransfer>(&model);

#ifdef OMEGA_H_USE_KOKKOS
  opts.bspline_model = &model;
  opts.should_smooth_snap = false;
#endif

  // Add a metric to force refinement
  if (!mesh.has_tag(Omega_h::VERT, "metric")) {
    if (world_rank == 0) {
      std::cout << "Adding implied metric for adaptation\n";
    }
    Omega_h::add_implied_metric_tag(&mesh);
  }

  // Scale metric to force refinement
  auto metrics = mesh.get_array<double>(Omega_h::VERT, "metric");
  metrics = Omega_h::multiply_each_by(metrics, 1.5);
  auto const metric_ncomps =
      Omega_h::divide_no_remainder(metrics.size(), mesh.nverts());
  mesh.add_tag(Omega_h::VERT, "metric", metric_ncomps, metrics);

  if (world_rank == 0) {
    std::cout << "Running adaptation with ClassParametricTransfer\n";
  }

  // Run adaptation
  bool adapted = Omega_h::adapt(&mesh, opts);

  if (adapted) {
    if (world_rank == 0) {
      std::cout << "Adaptation completed\n";
      std::cout << "New mesh has " << mesh.nverts() << " vertices\n";
    }

    // Check that class_parametric tag still exists
    if (!mesh.has_tag(Omega_h::VERT, "class_parametric")) {
      if (world_rank == 0) {
        std::cerr << "ERROR: class_parametric tag was lost during adaptation!\n";
      }
      return -1;
    }

    auto final_params = mesh.get_array<double>(Omega_h::VERT, "class_parametric");
    auto final_nverts = mesh.nverts();

    if (world_rank == 0) {
      std::cout << "Final mesh has " << final_nverts << " vertices\n";
      std::cout << "class_parametric tag has " << final_params.size()
                << " values (2 per vertex)\n";
    }

    // Verify that parametric coordinates are in valid range [0,1]
    auto num_invalid = 0;
    for (Omega_h::LO i = 0; i < final_nverts; ++i) {
      auto param = final_params[i * 2 + 0];
      // Only check vertices classified on model edges
      if (mesh.has_tag(Omega_h::VERT, "class_dim")) {
        auto class_dims = mesh.get_array<Omega_h::I8>(Omega_h::VERT, "class_dim");
        if (class_dims[i] == 1) {  // Classified on model edge
          if (param < 0.0 || param > 1.0) {
            num_invalid++;
            if (num_invalid <= 10 && world_rank == 0) {
              std::cerr << "WARNING: Vertex " << i << " has invalid parametric coord: "
                        << param << "\n";
            }
          }
        }
      }
    }

    if (num_invalid > 0 && world_rank == 0) {
      std::cerr << "ERROR: Found " << num_invalid
                << " vertices with invalid parametric coordinates!\n";
      return -1;
    }

    if (world_rank == 0) {
      std::cout << "SUCCESS: All parametric coordinates are valid\n";
    }

  } else {
    if (world_rank == 0) {
      std::cout << "No adaptation performed (mesh already at target)\n";
    }
  }

  if (world_rank == 0) {
    std::cout << "Test completed successfully!\n";
  }

  return 0;
}

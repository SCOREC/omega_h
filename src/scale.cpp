/*
 * Scale Omega_h mesh coordinates by a given factor.
 */

#include <Omega_h_array.hpp>
#include <Omega_h_cmdline.hpp>
#include <Omega_h_fence.hpp>
#include <Omega_h_file.hpp>
#include <Omega_h_for.hpp>
#include <Omega_h_mesh.hpp>

int main(int argc, char** argv) {
  auto lib = Omega_h::Library(&argc, &argv);
  auto world = lib.world();

  Omega_h::CmdLine cmdline;
  cmdline.add_arg<std::string>("mesh-in");
  cmdline.add_arg<std::string>("mesh-out");
  cmdline.add_arg<double>("scale");
  if (!cmdline.parse_final(world, &argc, argv)) return -1;

  auto mesh_in = cmdline.get<std::string>("mesh-in");
  auto mesh_out = cmdline.get<std::string>("mesh-out");
  // todo: make scale a vector for anisotropic scaling
  auto scale = cmdline.get<double>("scale");

  if (scale <= 0.0) {
    if (world->rank() == 0) {
      printf("ErrScale factor must be positive. Given %f.\n", scale);
    }
    return -1;
  }

  auto mesh = Omega_h::Mesh(&lib);
  Omega_h::binary::read(mesh_in, lib.world(), &mesh);

  auto dim = mesh.dim();
  auto nverts = mesh.nverts();
  auto coords = mesh.coords();
  Omega_h::Write<Omega_h::Real> scaled_coords(coords.size());
  OMEGA_H_ALWAYS_CHECK(coords.size() == dim * nverts);

  Omega_h::parallel_for(
      nverts, OMEGA_H_LAMBDA(const Omega_h::LO vert_id) {
        for (int i = 0; i < dim; ++i) {  // for future scale each coordinate
          scaled_coords[vert_id * dim + i] = coords[vert_id * dim + i] * scale;
        }
      });
  Omega_h::fence();

  mesh.set_coords(scaled_coords);
  Omega_h::fence();
  Omega_h::binary::write(mesh_out, &mesh);

  return 0;
}

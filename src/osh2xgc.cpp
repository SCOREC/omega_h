#include <Omega_h_cmdline.hpp>
#include <Omega_h_file.hpp>
#include <Omega_h_mesh.hpp>

int main(int argc, char** argv) {
  auto lib = Omega_h::Library(&argc, &argv);
  auto comm = lib.world();

  Omega_h::CmdLine cmdline;
  cmdline.add_arg<std::string>("mesh-in (osh)");
  cmdline.add_arg<std::string>("mesh-out (xgc)");

  if (!cmdline.parse_final(comm, &argc, argv)) return -1;
  auto mesh_in = cmdline.get<std::string>("mesh-in (osh)");
  auto mesh_out = cmdline.get<std::string>("mesh-out (xgc)");

  Omega_h::Mesh mesh(&lib);
  Omega_h::binary::read(mesh_in, lib.world(), &mesh);
  ::Omega_h::detail::require_same_tdim_sdim(mesh, __func__);
  Omega_h::xgc::write(mesh_out, &mesh);

  return 0;
}

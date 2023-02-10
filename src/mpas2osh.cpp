#include <Omega_h_file.hpp>
#include <Omega_h_cmdline.hpp>

int main(int argc, char** argv) {
  auto lib = Omega_h::Library(&argc, &argv);
  auto world = lib.world();
  Omega_h::CmdLine cmdline;
  cmdline.add_arg<std::string>("input.nc");
  cmdline.add_arg<std::string>("vtxFieldList.txt");
  cmdline.add_arg<std::string>("output.osh");
  if (!cmdline.parse_final(world, &argc, argv)) return -1;
  auto inpath = cmdline.get<std::string>("input.nc");
  auto inVtxFieldList = cmdline.get<std::string>("vtxFieldList.txt");
  auto outpath = cmdline.get<std::string>("output.osh");
  auto mesh = Omega_h::mpas::read(inpath, inVtxFieldList, world);
  Omega_h::binary::write(outpath, &mesh);
}

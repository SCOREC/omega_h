/*
 *
 * .cpp : adios2 low-level API example to write and read arrays and string
 *
 */
#include <Omega_h_timer.hpp>
#include <Omega_h_file.hpp>
#include <Omega_h_cmdline.hpp>
#include "Omega_h_build.hpp" // build_box
#include "Omega_h_library.hpp" // world
#include "Omega_h_mesh.hpp"
#include "Omega_h_inertia.hpp" // Rib
#include "Omega_h_tag.hpp" //class_ids
#include "Omega_h_defines.hpp"
#include "Omega_h_adios2.hpp"
#include <Omega_h_compare.hpp> // MeshCompareOpts
#include "Omega_h_array_ops.hpp" // each_eq_to
#include "Omega_h_element.hpp" // topological_singular_name
#include "Omega_h_mixedMesh.hpp" // family
#include <iostream>
#include <stdexcept>
#include <iomanip>

#include <adios2.h>
#include <mpi.h>

// to check the content of .bp, run bin/bpls
// ex. ./bpls mesh.bp
int main(int argc, char *argv[])
{
  auto lib = Omega_h::Library(&argc, &argv);
  auto world = lib.world();

  if (lib.world()->size()>1)
  {
    if (!lib.world()->rank())
      fprintf(stderr, "ADIOS2 file I/O with partitioned mesh is not supported yet\n");
    exit(EXIT_FAILURE);
  }

  Omega_h::CmdLine cmdline;

  cmdline.add_arg<std::string>("input1.osh");
  cmdline.add_arg<std::string>("input2.osh");
  cmdline.add_arg<std::string>("output.bp");
  if (!cmdline.parse_final(world, &argc, argv)) return -1;
  Omega_h::filesystem::path inpath1 = cmdline.get<std::string>("input1.osh");
  Omega_h::filesystem::path inpath2 = cmdline.get<std::string>("input2.osh");
  Omega_h::filesystem::path outpath=cmdline.get<std::string>("output.bp");

  Omega_h::Mesh mesh1(&lib);
  Omega_h::binary::read(inpath1, world, &mesh1);
  Omega_h::Mesh mesh2(&lib);
  Omega_h::binary::read(inpath2, world, &mesh2);

  Omega_h::binary::write("omegah1.osh", &mesh1);
  Omega_h::binary::write("omegah2.osh", &mesh2);

  try
  {
    std::map<Omega_h::Mesh*, std::string> mmap;
    mmap[&mesh1]="m1";
    mmap[&mesh2]="m2";
    Omega_h::adios::write(outpath, mmap);
    Omega_h::Mesh mesh3 = Omega_h::adios::read(outpath, &lib, std::string("m1"));
    Omega_h::Mesh mesh4 = Omega_h::adios::read(outpath, &lib, std::string("m2"));

    double tol = 1e-6, floor = 0.0;
    bool allow_superset = false;
    auto opts = Omega_h::MeshCompareOpts::init(
                &mesh1, Omega_h::VarCompareOpts{Omega_h::VarCompareOpts::RELATIVE, tol, floor});
    auto res = compare_meshes(&mesh1, &mesh3, opts, true);
    if (res == OMEGA_H_SAME || (allow_superset && res == OMEGA_H_MORE))
    {
      opts = Omega_h::MeshCompareOpts::init(
                &mesh2, Omega_h::VarCompareOpts{Omega_h::VarCompareOpts::RELATIVE, tol, floor});
      res = compare_meshes(&mesh2, &mesh4, opts, true);
      if (res == OMEGA_H_SAME || (allow_superset && res == OMEGA_H_MORE))
      {  
	std::cout << "\nSUCCESS! Meshes loaded from .osh and .bp are the same\n";
        return 0;
      }
    }
    std::cout << "\nFAIL! Two meshes (.osh and .bp) are NOT the same\n";
     return 2;
   }
   catch (std::exception &e)
   {
     std::cout << "\nERROR: ADIOS2 exception: " << e.what() << "\n";
     MPI_Abort(lib.world()->get_impl(), -1);
   }
    return 0;
}

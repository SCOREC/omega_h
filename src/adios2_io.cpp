/*
 *
 * .cpp : adios2 low-level API example to write and read arrays and string
 *
 *  Created on: Aug 25, 2024
 *  Author: Seegyoung Seol seols@rpi.edu
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
#if ADIOS2_USE_MPI
#include <mpi.h>
#endif

using namespace Omega_h;
using namespace std;

// printTagInfo and getNumEq copied from describe.cpp
template <typename T>
void printTagInfo(Omega_h::Mesh mesh, std::ostringstream& oss, int dim, int tag, std::string type) {
    auto tagbase = mesh.get_tag(dim, tag);
    auto array = Omega_h::as<T>(tagbase)->array();

    Omega_h::Real min = get_min(array);
    Omega_h::Real max = get_max(array);

    oss << std::setw(18) << std::left << tagbase->name().c_str()
        << std::setw(5) << std::left << dim
        << std::setw(7) << std::left << type
        << std::setw(5) << std::left << tagbase->ncomps()
        << std::setw(10) << std::left << min
        << std::setw(10) << std::left << max
        << "\n";
}

template <typename T>
int getNumEq(Omega_h::Mesh mesh, std::string tagname, int dim, int value) {
    auto array = mesh.get_array<T>(dim, tagname);
    auto each_eq_to = Omega_h::each_eq_to<T>(array, value);
    return Omega_h::get_sum(each_eq_to);
}

void print_info(Library* lib, Omega_h::Mesh mesh);

// to check the content of .bp, run /lore/seols/romulus-install/bin/bpls
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

  cout<<"\n--- Mesh loaded from \""<<inpath1<<"\" ---\n";
  print_info(&lib, mesh1);
  cout<<"\n--- Mesh loaded from \""<<inpath2<<"\" ---\n";
  print_info(&lib, mesh2);

//  Omega_h::Mesh mesh = build_box(world, OMEGA_H_SIMPLEX, 1., 1., 0., 2, 2, 0);
  Omega_h::binary::write("omegah1.osh", &mesh1);
  Omega_h::binary::write("omegah2.osh", &mesh2);
 // Omega_h::vtk::write_parallel("omegah.vtk", &mesh);

  try
  {
    map<Mesh*, std::string> mmap;
    mmap[&mesh1]="m1";
    mmap[&mesh2]="m2";
    write_adios2(outpath, mmap);
    Omega_h::Mesh mesh3 = read_adios2(outpath, &lib, std::string("m1"));
    Omega_h::Mesh mesh4 = read_adios2(outpath, &lib, std::string("m2"));
    //Omega_h::vtk::write_parallel("adios2.vtk", &mesh2);

    cout<<"\n\n--- Two meshes loaded back from \""<<outpath<<"\" ---\n";
    print_info(&lib, mesh3);
    print_info(&lib, mesh4);

    double tol = 1e-6, floor = 0.0;
    bool allow_superset = false;
    auto opts = MeshCompareOpts::init(
                &mesh1, VarCompareOpts{VarCompareOpts::RELATIVE, tol, floor});
    auto res = compare_meshes(&mesh1, &mesh3, opts, true);
    if (res == OMEGA_H_SAME || (allow_superset && res == OMEGA_H_MORE))
    {
      opts = MeshCompareOpts::init(
                &mesh2, VarCompareOpts{VarCompareOpts::RELATIVE, tol, floor});
      res = compare_meshes(&mesh2, &mesh4, opts, true);
      if (res == OMEGA_H_SAME || (allow_superset && res == OMEGA_H_MORE))
      {  
        cout << "\nSUCCESS! Meshes loaded from .osh and .bp are the same\n";
        return 0;
      }
    }
    cout << "\nFAIL! Two meshes (.osh and .bp) are NOT the same\n";
     return 2;
   }
   catch (std::exception &e)
   {
     std::cout << "\nERROR: ADIOS2 exception: " << e.what() << "\n";
#if ADIOS2_USE_MPI
     MPI_Abort(lib.world()->get_impl(), -1);
#endif
   }
    return 0;
}

// serial only at the moment
void print_info(Library* lib, Omega_h::Mesh mesh)
{
  auto rank = lib->world()->rank();
  
  ostringstream oss;
  // always print two places to the right of the decimal
  // for floating point types (i.e., imbalance)
  oss.precision(2);
  oss << std::fixed;

  if (!rank) 
  {
    oss << "\nEntity Type: " << Omega_h::topological_singular_name(mesh.family(), mesh.dim()) << "\n";

    oss << "\nEntity Count and Imbalance: (Dim, #Global, #Local, Max/Ave Imbalance)\n";
    for (int dim=0; dim <= mesh.dim(); dim++)
      oss << "(" << dim << ", " 
  	  << mesh.nglobal_ents(dim) << ", " 
    	  <<mesh.nents(dim)<< ", " 
	  << mesh.imbalance(dim) << ")\n";

    oss << "\nTag by Dimension: (Name, Dim, Type, #Components, Min/Max Value)\n";
    for (int dim=0; dim <= mesh.dim(); dim++)
      for (int tag=0; tag < mesh.ntags(dim); tag++) {
        auto tagbase = mesh.get_tag(dim, tag);
        if (tagbase->type() == OMEGA_H_I8)
          printTagInfo<Omega_h::I8>(mesh, oss, dim, tag, "I8");
        if (tagbase->type() == OMEGA_H_I32)
          printTagInfo<Omega_h::I32>(mesh, oss, dim, tag, "I32");
        if (tagbase->type() == OMEGA_H_I64)
     	  printTagInfo<Omega_h::I64>(mesh, oss, dim, tag, "I64");
        if (tagbase->type() == OMEGA_H_F64)
          printTagInfo<Omega_h::Real>(mesh, oss, dim, tag, "F64");
      }
    std::cout << oss.str();
  }
}

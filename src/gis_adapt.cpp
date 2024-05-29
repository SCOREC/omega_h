#include <iostream>

#include "Omega_h_adapt.hpp"
#include "Omega_h_array_ops.hpp"
#include "Omega_h_build.hpp"
#include "Omega_h_class.hpp"
#include "Omega_h_compare.hpp"
#include "Omega_h_for.hpp"
#include "Omega_h_shape.hpp"
#include "Omega_h_timer.hpp"
#include <Omega_h_file.hpp> //Omega_h::binary
#include <sstream> //ostringstream
#include <iomanip> //precision

using namespace Omega_h;

const Real ICE_DENSITY = 910; //kg/m^3

template <typename T>
void printTagInfo(Omega_h::Mesh& mesh, std::ostringstream& oss, int dim, int tag, std::string type) {
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

Reals getVolumeFromIceThickness(Omega_h::Mesh& mesh) {
  auto coords = mesh.coords();
  auto triArea = mesh.ask_sizes();
  auto iceThickness = mesh.get_array<Real>(OMEGA_H_VERT, "ice_thickness");
  auto faces2verts = mesh.ask_elem_verts();
  Write<Real> vol(mesh.nfaces()); 
  auto getVol = OMEGA_H_LAMBDA(LO faceIdx) {
    auto triVerts = Omega_h::gather_verts<3>(faces2verts, faceIdx);
    auto triThickness = Omega_h::gather_scalars<3>(iceThickness, triVerts);
    Real avgThickness = 0;
    for(LO i=0; i<3; i++)
       avgThickness += triThickness[i];
    avgThickness /= 3;
    vol[faceIdx] = triArea[faceIdx]*avgThickness;
  };
  parallel_for(mesh.nfaces(), getVol);
  return vol;
}

Reals getIceMass(Mesh& mesh, Reals vol) {
  return multiply_each_by(vol, ICE_DENSITY);
}

void printTags(Mesh& mesh) {
    std::ostringstream oss;
    // always print two places to the right of the decimal
    // for floating point types (i.e., imbalance)
    oss.precision(2);
    oss << std::fixed;

    if (!mesh.comm()->rank()) {
        oss << "\nTag Properties by Dimension: (Name, Dim, Type, Number of Components, Min. Value, Max. Value)\n";
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

static void check_total_mass(Mesh& mesh) {
  auto vol = getVolumeFromIceThickness(mesh);
  auto masses = getIceMass(mesh, vol);
  auto owned_masses = mesh.owned_array(mesh.dim(), masses, 1);
  auto mass = get_sum(mesh.comm(), owned_masses);
  if (!mesh.comm()->rank()) {
    std::cout << "mass " << mass << '\n';
  }
}

void printTriCount(Mesh* mesh) {
  const auto nTri = mesh->nglobal_ents(2);
  if (!mesh->comm()->rank())
    std::cout << "nTri: " << nTri << "\n";
}

AdaptOpts setupFieldTransfer(Mesh& mesh) {
  auto opts = AdaptOpts(&mesh);
  opts.xfer_opts.type_map["ice_thickness"] = OMEGA_H_LINEAR_INTERP;
  const int numLayers = 11;
  for(int i=1; i<=numLayers; i++) {
    std::stringstream ss;
    ss << "temperature_" << std::setfill('0') << std::setw(2) << i;
    std::cout << ss.str();
    opts.xfer_opts.type_map[ss.str()] = OMEGA_H_LINEAR_INTERP;
  }
  return opts;
}

int main(int argc, char** argv) {
  auto lib = Library(&argc, &argv);
  if( argc != 3 ) {
    fprintf(stderr, "Usage: %s inputMesh.osh outputMeshPrefix\n", argv[0]);
    exit(EXIT_FAILURE);
  }
  auto world = lib.world();
  Omega_h::Mesh mesh(&lib);
  Omega_h::binary::read(argv[1], world, &mesh);

  //set size field - TODO: use ice thickness
  mesh.set_parting(OMEGA_H_GHOSTED);
  {
    auto metrics = get_implied_isos(&mesh);
    auto scalar = metric_eigenvalue_from_length(0.5);
    metrics = multiply_each_by(metrics, scalar);
    mesh.add_tag(VERT, "metric", 1, metrics);
  }
  mesh.set_parting(OMEGA_H_ELEM_BASED);

  auto opts = setupFieldTransfer(mesh);
  printTriCount(&mesh);  
  printTags(mesh);
  Omega_h::vtk::write_parallel("beforeAdapt.vtk", &mesh, 2);
  check_total_mass(mesh);

  adapt(&mesh, opts);

  printTriCount(&mesh);  
  printTags(mesh);
  check_total_mass(mesh);
  const std::string vtkFileName = std::string(argv[2]) + ".vtk";
  Omega_h::vtk::write_parallel(vtkFileName, &mesh, 2);
  return 0;
}

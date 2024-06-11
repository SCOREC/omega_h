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


void printTriCount(Mesh* mesh) {
  const auto nTri = mesh->nglobal_ents(2);
  if (!mesh->comm()->rank())
    std::cout << "nTri: " << nTri << "\n";
}

int main(int argc, char** argv) {
  auto lib = Library(&argc, &argv);
  if( argc != 6 ) {
    fprintf(stderr, "Usage: %s inputMesh.osh outputMeshPrefix enforceMetricSize=[0:off|1:on] minLength maxLength\n", argv[0]);
    exit(EXIT_FAILURE);
  }
  auto world = lib.world();
  auto nx = 3;
  auto mesh = build_box(world, OMEGA_H_SIMPLEX, 1, 1, 0, nx, nx, 0);
  const auto enforceSize = std::atoi(argv[3]) > 0 ? true : false;
  const auto minLength = std::stof(argv[4]);
  const auto maxLength = std::stof(argv[5]);
  fprintf(stderr, "enforceMetricSize %d minLength %f maxLength %f\n", enforceSize, minLength, maxLength);

  //create analytic velocity field
  auto velocity = Write<Real>(mesh.nverts() * mesh.dim());
  auto coords = mesh.coords();
  auto f = OMEGA_H_LAMBDA(LO vert) {
    auto x = get_vector<2>(coords, vert);
    auto x2 = x[0] - 0.3;
    auto w = sign(x2) * std::sqrt(std::abs(x2));
    set_vector(velocity, vert, vector_2(1, 0) * w);
  };
  parallel_for(mesh.nverts(), f);
  mesh.add_tag(VERT, "velocity", mesh.dim(), Reals(velocity));

  //set size field - TODO: use ice thickness
  mesh.set_parting(OMEGA_H_GHOSTED);
//  auto target_error = Omega_h::Real(0.01); //this is a measure of 'error' when computing a metric from a hessian - see Omega_h_metric.cpp metric_from_messian(...)
  auto target_error = Omega_h::Real(1.0);
//  auto gradation_rate = Omega_h::Real(1.0);
//  auto max_metric_length = Omega_h::Real(2.8); // restrict edge length in metric space - how does this differ from MetricInput.max_length?

  auto genopts = Omega_h::MetricInput();
  genopts.sources.push_back(
      Omega_h::MetricSource{OMEGA_H_VARIATION, target_error, "velocity"});
  genopts.verbose = true;
  genopts.should_limit_lengths = enforceSize; //enforce [min|max]_length
  genopts.min_length = Omega_h::Real(minLength);
  genopts.max_length = Omega_h::Real(maxLength); //in metric space? restricts eigenvalues of metric, see Omega_h_metric.hpp
//  genopts.should_limit_gradation = true;
//  genopts.max_gradation_rate = gradation_rate; //how should this be set?
  Omega_h::generate_target_metric_tag(&mesh, genopts);
  Omega_h::add_implied_metric_tag(&mesh);
  Omega_h::AdaptOpts opts(&mesh);
//  opts.max_length_allowed = max_metric_length;
  opts.xfer_opts.type_map["velocity"] = OMEGA_H_LINEAR_INTERP;

  printTriCount(&mesh);  
  printTags(mesh);
  Omega_h::vtk::write_parallel("beforeAdapt.vtk", &mesh, 2);

  //Create a tag named 'metric' based on the current element size and the ideal
  //size.  As I understand, 'approach_metric(...)' uses the 'metric' tag as the
  //starting point and 'approaches' the 'target_metric' defined above
  //incrementally.
  while (Omega_h::approach_metric(&mesh, opts)) {
    adapt(&mesh, opts);
  }

  printTriCount(&mesh);  
  printTags(mesh);
  const std::string vtkFileName = std::string(argv[2]) + ".vtk";
  Omega_h::vtk::write_parallel(vtkFileName, &mesh, 2);
  return 0;
}

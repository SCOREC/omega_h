#include <Omega_h_library.hpp>
#include <Omega_h_defines.hpp> //OMEGA_H_EDGE
#include <Omega_h_for.hpp> //parallel_for
#include "Omega_h_int_scan.hpp" //offset_scan
#include <Omega_h_bsplineModel2d.hpp>
#include <fstream>

void writeSamplesToCsv(Omega_h::BsplineModel2D& model, std::string filename) {
  std::ofstream file(filename);
  assert(file.is_open());
  file << "splineId, x, y\n";

  const auto numEdges = model.getNumEnts(OMEGA_H_EDGE);
  const int factor = 4;
  const int numSamples = numEdges*factor;
  Omega_h::Write<Omega_h::Real> samplePts(numSamples, "splineSamplePoints");
  Omega_h::Write<Omega_h::LO> ids(numEdges, "splineIds");
  Omega_h::Write<Omega_h::LO> edgeSampleDegree(numEdges, "edgeSampleDegree");

  const double invFactor = 1.0/factor;
  Omega_h::parallel_for(numEdges, OMEGA_H_LAMBDA(Omega_h::LO& i) {
    for(int j = 0; j < factor; ++j) {
      auto t = invFactor * j;
      samplePts[i*factor+j] = t;
      ids[i] = i;
    }
    edgeSampleDegree[i] = factor;
  });

  auto edgeToSamples = offset_scan(Omega_h::read(edgeSampleDegree));

  const auto pts = model.eval(ids,edgeToSamples,samplePts);

  Omega_h::HostRead<Omega_h::Real> pts_h(pts);
  for(int i=0; i<numEdges; i++) {
    for(int j=0; j<factor; j++) {
      file << i << "," << pts_h[i*factor+(j*2)]
                << "," << pts_h[i*factor+(j*2)+1]
                << "\n";
    }
  }

}

int main(int argc, char** argv) {
  auto lib = Omega_h::Library(&argc, &argv);
  if( argc != 3 ) {
    fprintf(stderr, "Usage: %s inputSimModel.smd inputSplines.bin\n", argv[0]);
    exit(EXIT_FAILURE);
  }
  OMEGA_H_CHECK(argc == 3);
  auto model = Omega_h::BsplineModel2D(argv[1], argv[2]);
  writeSamplesToCsv(model, "samples.csv");

  model.printInfo();
  return 0;
}

#include <Omega_h_library.hpp>
#include <Omega_h_defines.hpp> //OMEGA_H_EDGE
#include <Omega_h_bsplineModel2d.hpp>
#include <fstream>

void writeSamplesToCsv(Omega_h::BsplineModel2D& model, std::string filename) {
  std::ofstream file(filename);
  assert(file.is_open());
  file << "splineId, x, y\n";

  const auto s2k = model.getSplineToKnots();
  const int numKnots = s2k.last();
  const int factor = 4;
  const int numSamples = numKnots*factor;
  Omega_h::Write<Omega_h::Real> samplePts(numSamples, "splineSamplePoints");
  Omega_h::Write<Omega_h::LO> ids(numSamples, "splineIds");

  const auto numEdges = model.getNumEnts(OMEGA_H_EDGE);
  Omega_h::parallel_for(numEdges, OMEGA_H_LAMBDA(Omega_h::LO& i) {
    const int numKnots = s2k[i+1]-s2k[i];
    const int numSamples = numKnots * factor;
    for(int j = 0; j < numSamples; ++j) {
      auto t = 1.0 * j / numSamples;
      samplePts[i] = t;
      ids[i] = i;
    }
  });

  const auto pts = model.eval(ids,samplePts); 

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

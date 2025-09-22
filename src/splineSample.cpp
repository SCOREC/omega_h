#include <Omega_h_library.hpp>
#include <Omega_h_defines.hpp> //OMEGA_H_EDGE
#include <Omega_h_for.hpp> //parallel_for
#include "Omega_h_int_scan.hpp" //offset_scan
#include <Omega_h_bsplineModel2d.hpp>
#include <fstream>

void writeSamplesToCsv(Omega_h::BsplineModel2D& model, std::string filename) {
  const auto numEdges = model.getNumEnts(OMEGA_H_EDGE);
  Omega_h::Write<Omega_h::LO> samplesPerEdge(numEdges, "samplesPerEdge");
  const auto s2k = model.getSplineToKnots();
  const int factor = 4;
  Omega_h::parallel_for(numEdges, OMEGA_H_LAMBDA(Omega_h::LO& i) {
      samplesPerEdge[i] = (s2k[i+1]-s2k[i]) * factor;
  });
  const auto edgeToSamples = Omega_h::offset_scan(Omega_h::read(samplesPerEdge));

  const int numSamples = edgeToSamples.last();
  Omega_h::Write<Omega_h::Real> samplePts(numSamples, "splineSamplePoints");

  Omega_h::parallel_for(numEdges, OMEGA_H_LAMBDA(Omega_h::LO& i) {
    const auto n = edgeToSamples[i+1]-edgeToSamples[i];
    //will result in overlapping points and the geometric model vertices
    const double step = 1.0/(n-1);
    for(int j = 0; j < n; ++j) {
      const auto t = step * j;
      const auto idx = edgeToSamples[i]+j;
      samplePts[idx] = t;
    }
  });

  Omega_h::Write<Omega_h::LO> ids(numEdges, 0, 1, "splineIds"); //array from 0..numEdges-1
  const auto pts = model.eval(ids,edgeToSamples,samplePts);

  std::ofstream file(filename);
  assert(file.is_open());
  file << "splineId, x, y\n";
  Omega_h::HostRead<Omega_h::Real> pts_h(pts);

  Omega_h::HostRead<Omega_h::LO> edgeToSamples_h(edgeToSamples);
  Omega_h::HostRead<Omega_h::LO> ids_h(ids);
  for(int i=0; i<ids_h.size(); i++) {
    const auto id = ids_h[i];
    for(int j = edgeToSamples_h[i]; j < edgeToSamples_h[i+1]; j++) {
      file << id
           << "," << pts_h[j*2]
           << "," << pts_h[j*2+1]
           << "\n";
    }
  };
  file.close();
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

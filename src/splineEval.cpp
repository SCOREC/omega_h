#include <Omega_h_library.hpp>
#include <Omega_h_defines.hpp> //OMEGA_H_EDGE
#include <Omega_h_for.hpp> //parallel_for
#include "Omega_h_int_scan.hpp" //offset_scan
#include <Omega_h_bsplineModel2d.hpp>
#include <Omega_h_file.hpp> //binary::<write|read>
#include <Omega_h_array_ops.hpp> //LOs::==, are_close
#include <fstream>
#include <filesystem> //std::filesystem


unsigned char const magic[2] = {0xa1, 0x1a};

struct Samples {
  Omega_h::LOs ids;
  Omega_h::LOs edgeToSamples;
  Omega_h::Reals pts; //the cartesian locations along the splines
};

bool areSamplesClose(const Samples& ref, const Samples& test) {
  auto isSameIds = (ref.ids == test.ids);
  auto isSameE2S = (ref.edgeToSamples == test.edgeToSamples);
  auto isClosePts = Omega_h::are_close(ref.pts, test.pts);
  return (isSameIds && isSameE2S && isClosePts);
}

Samples getSamples(Omega_h::BsplineModel2D& model) {
  const auto numEdges = model.getNumEnts(OMEGA_H_EDGE);
  Omega_h::Write<Omega_h::LO> samplesPerEdge(numEdges, "samplesPerEdge");
  const auto s2k = model.getSplineToKnots();
  const int factor = 4;
  Omega_h::parallel_for(numEdges, OMEGA_H_LAMBDA(Omega_h::LO i) {
      samplesPerEdge[i] = (s2k[i+1]-s2k[i]) * factor;
  });
  const auto edgeToSamples = Omega_h::offset_scan(Omega_h::read(samplesPerEdge));

  const int numSamples = edgeToSamples.last();
  Omega_h::Write<Omega_h::Real> samplePts(numSamples, "splineSamplePoints");

  Omega_h::parallel_for(numEdges, OMEGA_H_LAMBDA(Omega_h::LO i) {
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

  return {ids,edgeToSamples,pts};
}

void writeSamplesToBinary(const Samples& samples, std::string filename) {
  std::ofstream file(filename);
  assert(file.is_open());

  const int compressed = 0;
  //the following is from src/Omega_h_file.cpp write(...)
  file.write(reinterpret_cast<const char*>(magic), sizeof(magic));
  bool needs_swapping = !Omega_h::is_little_endian_cpu();
  Omega_h::binary::write_value(file, compressed, needs_swapping);

  Omega_h::binary::write_array(file, samples.ids, compressed, needs_swapping);
  Omega_h::binary::write_array(file, samples.edgeToSamples, compressed, needs_swapping);
  Omega_h::binary::write_array(file, samples.pts, compressed, needs_swapping);
  file.close();
}

Samples readSamplesFromBinary(std::string filename) {
  std::ifstream file(filename);
  assert(file.is_open());
  //the following is from src/Omega_h_file.cpp read(...)
  unsigned char magic_in[2];
  file.read(reinterpret_cast<char*>(magic_in), sizeof(magic));
  OMEGA_H_CHECK(magic_in[0] == magic[0]);
  OMEGA_H_CHECK(magic_in[1] == magic[1]);
  bool needs_swapping = !Omega_h::is_little_endian_cpu();
  int compressed;
  Omega_h::binary::read_value(file, compressed, needs_swapping);

  Samples in;
  Omega_h::binary::read_array(file, in.ids, compressed, needs_swapping);
  Omega_h::binary::read_array(file, in.edgeToSamples, compressed, needs_swapping);
  Omega_h::binary::read_array(file, in.pts, compressed, needs_swapping);
  file.close();
  return in;
}

void writeSamplesToCsv(const Samples& samples, std::string filename) {
  std::ofstream file(filename);
  assert(file.is_open());
  file << "splineId, x, y\n";
  Omega_h::HostRead<Omega_h::Real> pts_h(samples.pts);

  Omega_h::HostRead<Omega_h::LO> edgeToSamples_h(samples.edgeToSamples);
  Omega_h::HostRead<Omega_h::LO> ids_h(samples.ids);
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
  if( argc != 4 ) {
    fprintf(stderr, "Usage: %s inputSimModel.smd inputSplines.oshb referenceSamples.oshb\n", argv[0]);
    fprintf(stderr, "inputSimModel.smd: Simmetrix GeomSim geometric model\n"
                    "inputSplines.oshb: Omega_h binary file containing spline information\n"
                    "                   associated with the topology of inputSimModel.smd\n"
                    "referenceSamples.oshb: Omega_h binary file containing the expected\n"
                    "                       result of evaluation\n");
    exit(EXIT_FAILURE);
  }
  OMEGA_H_CHECK(argc == 4);
  auto model = Omega_h::BsplineModel2D(argv[1], argv[2]);
  const auto samples = getSamples(model);

  std::filesystem::path splineFile(argv[2]);
  const auto outputCsvFile = splineFile.stem().string()+"_eval.csv";
  writeSamplesToCsv(samples, outputCsvFile);

  const auto outputBinFile = splineFile.stem().string()+"_eval.oshb";
  writeSamplesToBinary(samples, outputBinFile);

  const auto refSamples = readSamplesFromBinary(argv[3]);
  assert( areSamplesClose(refSamples, samples) );

  model.printInfo();
  return 0;
}

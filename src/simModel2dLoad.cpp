#include <Omega_h_library.hpp>
#include <Omega_h_model2d.hpp>
#include <Omega_h_array_ops.hpp> // operator==(LOs,LOs)
#include <string_view> // operator==(LOs,LOs)

void printGraph(const Omega_h::Graph& g, std::string_view name) {
  std::cout << name << " {\n";
  Omega_h::HostRead<Omega_h::LO> offsets(g.a2ab);
  Omega_h::HostRead<Omega_h::LO> vals(g.ab2b);
  for(int i=0; i<offsets.size()-1; i++) {
    std::cout << i << ": ";
    for(int j=offsets[i]; j<offsets[i+1]; j++) 
      std::cout << vals[j] << " ";
    std::cout << "\n";
  }
  std::cout << "}\n";
}

void printLOs(const Omega_h::LOs& arr, std::string_view name) {
  std::cout << name << " { ";
  Omega_h::HostRead<Omega_h::LO> a(arr);
  for(int i=0; i<a.size(); i++)
    std::cout << a[i] << " ";
  std::cout << "}\n";
}

int main(int argc, char** argv) {
  auto lib = Omega_h::Library(&argc, &argv);
  if( argc != 2 ) {
    fprintf(stderr, "Usage: %s inputSimModel.smd\n", argv[0]);
    exit(EXIT_FAILURE);
  }
  OMEGA_H_CHECK(argc == 2);
  auto model = Omega_h::Model2D::SimModel2D_load(argv[1]);
  model.printInfo();
  //the following were checked in SimModeler
  Omega_h::LOs expectedVtxIds = {19,15,11,10};
  OMEGA_H_CHECK(model.vtxIds == expectedVtxIds);
  Omega_h::LOs expectedEdgeIds = {6,12,16,20};
  OMEGA_H_CHECK(model.edgeIds == expectedEdgeIds);
  Omega_h::LOs expectedFaceIds = {2};
  OMEGA_H_CHECK(model.faceIds == expectedFaceIds);
  //TODO check adjs

  printGraph(model.vtxToEdgeUse, "vtxToEdgeUse");
  //Omega_h::Graph expected_v2e = Omega_h::Graph({0,2,4,6,8},{3,4,2,5,1,6,0,7});
  //OMEGA_H_CHECK(model.edgeToEdgeUse == expected_e2eu);

  //use ids and adjacencies are **not** accessible in SimModeler
  Omega_h::LOs expectedEdgeUseIds = {8,6,4,2,1,3,5,7};
  OMEGA_H_CHECK(model.edgeUseIds == expectedEdgeUseIds);
  Omega_h::LOs expectedLoopUseIds = {2,1};
  OMEGA_H_CHECK(model.loopUseIds == expectedLoopUseIds);

  Omega_h::Graph expected_e2eu = Omega_h::Graph({0,2,4,6,8},{3,4,2,5,1,6,0,7});
  OMEGA_H_CHECK(model.edgeToEdgeUse == expected_e2eu);

  Omega_h::Graph expected_lu2eu = Omega_h::Graph({0,4,8},{0,1,2,3,4,5,6,7});
  OMEGA_H_CHECK(model.loopUseToEdgeUse == expected_lu2eu);

  return 0;
}

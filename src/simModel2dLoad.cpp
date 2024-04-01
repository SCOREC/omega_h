#include <Omega_h_library.hpp>
#include <Omega_h_model2d.hpp>

int main(int argc, char** argv) {
  auto lib = Omega_h::Library(&argc, &argv);
  if( argc != 2 ) {
    fprintf(stderr, "Usage: %s inputSimModel.smd\n", argv[0]);
    exit(EXIT_FAILURE);
  }
  OMEGA_H_CHECK(argc == 2);
  auto model = Omega_h::Model2D::SimModel2D_load(argv[1]);
  model.printInfo();
  return 0;
}

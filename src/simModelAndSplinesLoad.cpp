#include <Omega_h_library.hpp>
#include <Omega_h_bsplineModel2d.hpp>

int main(int argc, char** argv) {
  auto lib = Omega_h::Library(&argc, &argv);
  if( argc != 3 ) {
    fprintf(stderr, "Usage: %s inputSimModel.smd inputSplines.bin\n", argv[0]);
    exit(EXIT_FAILURE);
  }
  OMEGA_H_CHECK(argc == 3);
  auto model = Omega_h::BsplineModel2D(argv[1], argv[2]);

  model.printInfo();
  return 0;
}

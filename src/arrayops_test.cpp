#include "Omega_h_for.hpp"
#include "Omega_h_library.hpp"
#include "Omega_h_array_ops.hpp"

int main(int argc, char** argv) {
  using namespace Omega_h;
  auto lib = Library(&argc, &argv);
  auto world = lib.world();
  {
    Reals a = {1,0,2};
    bool res = is_sorted(a);
    printf("sorted: %s\n", res ? "yes" : "no");
    OMEGA_H_CHECK(res == false);
  }
  {
    Reals a = {1,0,1,2};
    LO res = find_last(a,1.0);
    printf("index: %d\n", res);
    OMEGA_H_CHECK(res == 2);
  }
  {
    Reals a = {1,0.0,1,2};
    Reals b = {1,0.1,1,2};
    bool res = are_close_abs(a,b,1e-3);
    printf("are_close: %s\n", res ? "yes" : "no");
    OMEGA_H_CHECK(res == false);
    Reals c = {1,0.000001,1,2};
    res = are_close_abs(a,c,1e-3);
    printf("are_close: %s\n", res ? "yes" : "no");
    OMEGA_H_CHECK(res == true);
  }
  return 0;
}

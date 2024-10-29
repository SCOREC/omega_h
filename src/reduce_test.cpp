#include "Omega_h_library.hpp"
#include "Omega_h_reduce.hpp"

int main(int argc, char** argv) {
  using namespace Omega_h;
  auto lib = Library(&argc, &argv);
  auto world = lib.world();
  {
    struct sum_reduce {
      using value_type=LO;
      void init(value_type& res) {
        res = 0;
      }
      OMEGA_H_INLINE
      void operator()(const LO& i, LO& lsum) const {
        lsum += i;
      }
    };
  {
    sum_reduce sr;
    sum_reduce::value_type result;
    sr.init(result);
    LO lsum=0;
    LO two=2;
    sr(two, lsum);
  }
    sum_reduce sr;
    LO n = 3;
    auto result = parallel_reduce<sum_reduce>(n, sr, "my_reduce");
    //auto kernel = OMEGA_H_LAMBDA(const LO& i, LO& lsum) { lsum += i; };
    //auto result = parallel_reduce(n, kernel, "reduce");
    fprintf(stderr, "result %d\n", result);
  }
  return 0;
}

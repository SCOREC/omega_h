#include "Omega_h_library.hpp"
#include "Omega_h_reduce.hpp"

int main(int argc, char** argv) {
  using namespace Omega_h;
  auto lib = Library(&argc, &argv);
  auto world = lib.world();
  {
    struct sum_reduce {
      using value_type=LO;
      OMEGA_H_INLINE void init(value_type& update) const { update = 0; }
      OMEGA_H_INLINE
      void operator()(const value_type& i, LO& lsum) const {
        lsum += i;
      }
      OMEGA_H_INLINE void join(value_type& update, const value_type& input) const {
        update = update + input;
      }
    };

    LO n = 5;
    LO expected = 10;
    {
    //this doesn't compile - cuda 'invalid template' compilation error, the
    //source of the error isn't obvious
    sum_reduce sr;
    auto result = Omega_h::parallel_reduce<sum_reduce>(n, sr, "my_reduce");
    fprintf(stderr, "result %d\n", result);
    assert(result == expected);
    }

    {
    //this doesn't compile - invalid args to Omega_h::parallel_reduce
    //I wouldn't expect it to compile as it doesn't define 'init()' or 'value_type'
    //neded by Omega_h::parallel_reduce.
    //auto kernel = OMEGA_H_LAMBDA(const LO& i, LO& lsum) { lsum += i; };
    //auto result = Omega_h::parallel_reduce(n, kernel, "my_reduce");
    //fprintf(stderr, "result %d\n", result);
    //assert(result == expected);
    }

    {
    //this works
    int result = 0;
    auto kernel = OMEGA_H_LAMBDA(const LO& i, LO& lsum) { lsum += i; };
    Kokkos::parallel_reduce("my_reduce", policy(n), kernel, result);
    fprintf(stderr, "result %d\n", result);
    assert(result == expected);
    }

  }
  return 0;
}

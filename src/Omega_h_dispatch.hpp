#ifndef OMEGA_H_DISPATCH_HPP
#define OMEGA_H_DISPATCH_HPP

#include <Omega_h_fail.hpp>
#include <Omega_h_defines.hpp>
#include <type_traits>

namespace Omega_h {
namespace detail {
template <class F>
decltype(auto) dispatch_sdim_edim(Int sdim, Int edim, F f) {
  if (sdim == 1 && edim == 1)
    return f(std::integral_constant<Int, 1>{}, std::integral_constant<Int, 1>{});
  if (sdim == 2 && edim == 1)
    return f(std::integral_constant<Int, 2>{}, std::integral_constant<Int, 1>{});
  if (sdim == 2 && edim == 2)
    return f(std::integral_constant<Int, 2>{}, std::integral_constant<Int, 2>{});
  if (sdim == 3 && edim == 1)
    return f(std::integral_constant<Int, 3>{}, std::integral_constant<Int, 1>{});
  if (sdim == 3 && edim == 2)
    return f(std::integral_constant<Int, 3>{}, std::integral_constant<Int, 2>{});
  if (sdim == 3 && edim == 3)
    return f(std::integral_constant<Int, 3>{}, std::integral_constant<Int, 3>{});
  Omega_h_fail("dispatch_sdim_edim: unsupported (space_dim, entity_dim) = (%d, %d)\n", sdim, edim);
}
}  // namespace detail
}  // namespace Omega_h
#endif

#ifndef OMEGA_H_MDSPAN_HPP
#define OMEGA_H_MDSPAN_HPP

#include <Omega_h_config.h>

#ifdef OMEGA_H_USE_KOKKOS
#include <Kokkos_Core.hpp>
#else
#define MDSPAN_IMPL_STANDARD_NAMESPACE Kokkos
#define MDSPAN_IMPL_PROPOSED_NAMESPACE Experimental
#endif
#include <mdspan/mdspan.hpp>

namespace Omega_h {
using Kokkos::mdspan;
using Kokkos::extents;
using Kokkos::dextents;
using Kokkos::layout_right;
using Kokkos::dynamic_extent;
}  // namespace Omega_h

#endif

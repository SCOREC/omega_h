# mdspan

Unmodified single-header distribution of https://github.com/kokkos/mdspan,
branch `single-header`, commit `86eefa3d6653f8ce09cd8bbff9a8ac0047a72d39`.
The upstream LICENSE is retained alongside the header; individual source
sections also retain their SPDX license identifiers.

Used and installed only for Omega_h builds without Kokkos. Kokkos builds
use their own bundled mdspan. Include Omega_h_mdspan.hpp from Omega_h code.
The standard/proposed namespaces are configured as Kokkos/Experimental.

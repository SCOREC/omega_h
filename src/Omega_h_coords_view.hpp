#ifndef OMEGA_H_COORDS_VIEW_HPP
#define OMEGA_H_COORDS_VIEW_HPP

#include <Omega_h_adj.hpp>
#include <Omega_h_mdspan.hpp>
#include <type_traits>
#include <vector>

namespace Omega_h {

#ifdef OMEGA_H_USE_KOKKOS
using HostCoordsMemorySpace = Kokkos::HostSpace;
using CoordsMemorySpace = Space;
#else
struct HostCoordsMemorySpace {};
using CoordsMemorySpace = HostCoordsMemorySpace;
#endif

// Like PCMS's accessor, retain memory-space information without owning storage.
template <class T, class MemorySpace>
struct memory_space_accessor : Kokkos::default_accessor<T> {
  using memory_space = MemorySpace;
  using offset_policy = memory_space_accessor;
  OMEGA_H_INLINE constexpr memory_space_accessor() noexcept = default;
  template <class U, std::enable_if_t<
      std::is_convertible<U (*)[], T (*)[]>::value, int> = 0>
  OMEGA_H_INLINE constexpr memory_space_accessor(
      memory_space_accessor<U, MemorySpace> const&) noexcept {}
};

template <std::size_t N = dynamic_extent, class MemorySpace = CoordsMemorySpace,
    class Layout = layout_right>
using CoordsView = mdspan<const Real, extents<LO, dynamic_extent, N>, Layout,
    memory_space_accessor<const Real, MemorySpace>>;
template <std::size_t N = dynamic_extent, class MemorySpace = CoordsMemorySpace,
    class Layout = layout_right>
using CoordsWriteView = mdspan<Real, extents<LO, dynamic_extent, N>, Layout,
    memory_space_accessor<Real, MemorySpace>>;
template <std::size_t N = dynamic_extent>
using HostCoordsView = CoordsView<N, HostCoordsMemorySpace>;

namespace detail {
template <class T, std::size_t N, class MemorySpace>
auto make_coords_span(T* data, LO size, Int sdim) {
  if (sdim < 1 || sdim > 3)
    Omega_h_fail("CoordsView: space_dim (%d) must be between 1 and 3\n", sdim);
  if (N != dynamic_extent && N != static_cast<std::size_t>(sdim))
    Omega_h_fail("CoordsView: static extent (%zu) differs from space_dim (%d)\n", N, sdim);
  if (size % sdim != 0)
    Omega_h_fail("CoordsView: coordinate count is not divisible by space_dim (%d)\n", sdim);
  using Span = mdspan<T, extents<LO, dynamic_extent, N>, layout_right,
      memory_space_accessor<T, MemorySpace>>;
  return Span(data, typename Span::extents_type(size / sdim, sdim));
}
}  // namespace detail

// These factories borrow storage. Keep the supplied owner alive while using it.
template <std::size_t N = dynamic_extent>
CoordsView<N> make_coords_view(Reals const& data, Int sdim = static_cast<Int>(N)) {
  return detail::make_coords_span<const Real, N, CoordsMemorySpace>(data.data(), data.size(), sdim);
}
template <std::size_t N = dynamic_extent>
CoordsWriteView<N> make_coords_view(Write<Real> const& data, Int sdim = static_cast<Int>(N)) {
  return detail::make_coords_span<Real, N, CoordsMemorySpace>(data.data(), data.size(), sdim);
}
template <std::size_t N = dynamic_extent>
HostCoordsView<N> make_coords_view(HostRead<Real> const& data, Int sdim = static_cast<Int>(N)) {
  return detail::make_coords_span<const Real, N, HostCoordsMemorySpace>(data.data(), data.size(), sdim);
}
template <std::size_t N = dynamic_extent>
HostCoordsView<N> make_coords_view(std::vector<Real> const& data, Int sdim = static_cast<Int>(N)) {
  return detail::make_coords_span<const Real, N, HostCoordsMemorySpace>(data.data(), LO(data.size()), sdim);
}
template <std::size_t N = dynamic_extent>
void make_coords_view(Reals&&, Int = static_cast<Int>(N)) = delete;
template <std::size_t N = dynamic_extent>
void make_coords_view(Write<Real>&&, Int = static_cast<Int>(N)) = delete;
template <std::size_t N = dynamic_extent>
void make_coords_view(HostRead<Real>&&, Int = static_cast<Int>(N)) = delete;
template <std::size_t N = dynamic_extent>
void make_coords_view(std::vector<Real>&&, Int = static_cast<Int>(N)) = delete;

template <std::size_t N = dynamic_extent>
void make_coords_view(Reals const&&, Int = static_cast<Int>(N)) = delete;
template <std::size_t N = dynamic_extent>
void make_coords_view(Write<Real> const&&, Int = static_cast<Int>(N)) = delete;
template <std::size_t N = dynamic_extent>
void make_coords_view(HostRead<Real> const&&, Int = static_cast<Int>(N)) = delete;
template <std::size_t N = dynamic_extent>
void make_coords_view(std::vector<Real> const&&, Int = static_cast<Int>(N)) = delete;

template <std::size_t N, class T, std::size_t M, class Layout, class MemorySpace>
auto as_coords(mdspan<T, extents<LO, dynamic_extent, M>, Layout,
    memory_space_accessor<T, MemorySpace>> coords) {
  static_assert(N == dynamic_extent || M == dynamic_extent || N == M,
      "incompatible static coordinate extents");
  if (N != dynamic_extent && coords.extent(1) != static_cast<LO>(N))
    Omega_h_fail("CoordsView: static extent (%zu) differs from space_dim (%d)\n", N, int(coords.extent(1)));
  using Span = mdspan<T, extents<LO, dynamic_extent, N>, Layout,
      memory_space_accessor<T, MemorySpace>>;
  return Span(coords);
}

// Return an explicitly owned host copy, including for device/strided spans.
template <std::size_t N, class MemorySpace, class Layout>
std::vector<Real> copy_coords_to_host(CoordsView<N, MemorySpace, Layout> coords) {
  std::vector<Real> result(coords.size());
#ifdef OMEGA_H_USE_KOKKOS
  using Borrowed = Kokkos::View<const Real**, Kokkos::LayoutStride, MemorySpace,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  Borrowed source(coords.data_handle(), Kokkos::LayoutStride(
      coords.extent(0), coords.stride(0), coords.extent(1), coords.stride(1)));
  auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), source);
#else
  auto host = coords;
#endif
  for (LO v = 0; v < coords.extent(0); ++v)
    for (Int d = 0; d < coords.extent(1); ++d)
      result[v * coords.extent(1) + d] = host(v, d);
  return result;
}

template <Int n, std::size_t N, class MemorySpace, class Layout>
OMEGA_H_INLINE Vector<n> get_vector(CoordsView<N, MemorySpace, Layout> const& coords, LO v) {
  static_assert(N != dynamic_extent && N == n,
      "get_vector requires a matching static coordinate extent");
  Vector<n> result;
  for (Int d = 0; d < n; ++d) result[d] = coords(v, d);
  return result;
}

template <Int n, std::size_t N, class MemorySpace, class Layout>
OMEGA_H_INLINE Matrix<static_cast<Int>(N), n> gather_vectors(
    CoordsView<N, MemorySpace, Layout> const& coords, Few<LO, n> vertices) {
  static_assert(N != dynamic_extent, "gather_vectors needs a static coordinate extent");
  Matrix<static_cast<Int>(N), n> result;
  for (Int i = 0; i < n; ++i)
    for (Int d = 0; d < static_cast<Int>(N); ++d)
      result[i][d] = coords(vertices[i], d);
  return result;
}

class Mesh;
void require_same_tdim_sdim(Mesh const& mesh, char const* caller = "require_same_tdim_sdim");

}  // namespace Omega_h
#endif

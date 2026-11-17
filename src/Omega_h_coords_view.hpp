#ifndef OMEGA_H_COORDS_VIEW_HPP
#define OMEGA_H_COORDS_VIEW_HPP

#include <Omega_h_adj.hpp>
#include <Omega_h_mdspan.hpp>
#include <type_traits>

namespace Omega_h {

template <class T, std::size_t N>
class BasicCoordsView {
  static_assert(std::is_same<T, Real>::value || std::is_same<T, const Real>::value,
      "coordinate views hold Real or const Real");
  template <class U>
  using ArrayOf = std::conditional_t<std::is_const<U>::value, Reals, Write<Real>>;

 public:
  using extents_type = extents<LO, dynamic_extent, N>;
  using mdspan_type = mdspan<T, extents_type, layout_right>;

  BasicCoordsView(ArrayOf<T> data, Int sdim) : data_(data) {
    if (sdim < 1 || sdim > 3)
      Omega_h_fail("CoordsView: space_dim (%d) must be between 1 and 3\n", sdim);
    if (N != dynamic_extent && N != static_cast<std::size_t>(sdim))
      Omega_h_fail("CoordsView: static extent (%zu) differs from space_dim (%d)\n", N, sdim);
    if (data_.size() % sdim != 0)
      Omega_h_fail("CoordsView: coordinate count is not divisible by space_dim (%d)\n", sdim);
    span_ = mdspan_type(data_.data(), extents_type(data_.size() / sdim, sdim));
  }

  template <class U, std::size_t M,
      std::enable_if_t<(N == dynamic_extent || N == M) &&
          std::is_convertible<ArrayOf<U>, ArrayOf<T>>::value, int> = 0>
  BasicCoordsView(BasicCoordsView<U, M> const& other)
      : BasicCoordsView(ArrayOf<T>(other.raw()), other.extent(1)) {}

  OMEGA_H_INLINE T& operator()(LO v, Int d) const {
#ifndef NDEBUG
    OMEGA_H_CHECK(v >= 0 && v < extent(0));
    OMEGA_H_CHECK(d >= 0 && d < extent(1));
#endif
    return span_(v, d);
  }
  OMEGA_H_INLINE LO extent(Int r) const {
#ifndef NDEBUG
    OMEGA_H_CHECK(r >= 0 && r < 2);
#endif
    return span_.extent(r);
  }
  // The returned span borrows storage; keep an owning view alive while using it.
  OMEGA_H_INLINE mdspan_type span() const { return span_; }
  ArrayOf<T> const& raw() const { return data_; }

  template <std::size_t M>
  BasicCoordsView<T, M> as() const { return BasicCoordsView<T, M>(data_, extent(1)); }

 private:
  ArrayOf<T> data_;
  mdspan_type span_;
};

template <std::size_t N = dynamic_extent>
using CoordsView = BasicCoordsView<const Real, N>;
template <std::size_t N = dynamic_extent>
using CoordsWriteView = BasicCoordsView<Real, N>;

template <Int n, std::size_t N>
OMEGA_H_INLINE Matrix<static_cast<Int>(N), n> gather_vectors(
    CoordsView<N> const& coords, Few<LO, n> vertices) {
  static_assert(N != dynamic_extent, "gather_vectors needs a static coordinate extent");
  Matrix<static_cast<Int>(N), n> result;
  for (Int i = 0; i < n; ++i)
    for (Int d = 0; d < static_cast<Int>(N); ++d)
      result[i][d] = coords(vertices[i], d);
  return result;
}

}  // namespace Omega_h
#endif

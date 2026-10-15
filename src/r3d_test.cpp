#include "Omega_h_r3d.hpp"

static void test_3d() {
  r3d::Few<r3d::Vector<3>, 4> verts = {
      {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  auto faces = r3d::faces_from_verts(verts);
  OMEGA_H_CHECK(Omega_h::are_close(Omega_h::from_r3d(faces[0].n),
      -Omega_h::normalize(Omega_h::vector_3(1, 1, 1))));
  OMEGA_H_CHECK(Omega_h::are_close(faces[0].d, -faces[0].n[0]));
  for (Omega_h::Int i = 0; i < 3; ++i) {
    auto v = Omega_h::vector_3(0, 0, 0);
    v[i] = 1;
    OMEGA_H_CHECK(Omega_h::are_close(Omega_h::from_r3d(faces[i + 1].n), v));
    OMEGA_H_CHECK(Omega_h::are_close(faces[i + 1].d, 0));
  }
  r3d::Polytope<3> a;
  r3d::init(a, verts);
  OMEGA_H_CHECK(a.nverts == 4);
  auto b = a;
  r3d::clip(b, r3d::Few<r3d::Plane<3>, 1>({{{1, 0, 0}, -0.5}}));
  OMEGA_H_CHECK(b.nverts == 4);
  auto volume = r3d::measure(b);
  OMEGA_H_CHECK(Omega_h::are_close(volume, Omega_h::cube(0.5) / 6.0));
  auto c = a;
  r3d::clip(c, r3d::Few<r3d::Plane<3>, 1>({{{-1, 0, 0}, 0.5}}));
  OMEGA_H_CHECK(c.nverts == 6);
  volume = r3d::measure(c);
  OMEGA_H_CHECK(
      Omega_h::are_close(volume, (1. / 6.) - (Omega_h::cube(0.5) / 6.)));
  r3d::Few<r3d::Vector<3>, 4> verts2 = {
      {1, 0, 0}, {1, 1, 0}, {0, 0, 0}, {1, 0, 1}};
  r3d::Polytope<3> intersection;
  r3d::intersect_simplices(intersection, verts, verts2);
  volume = r3d::measure(intersection);
  OMEGA_H_CHECK(Omega_h::are_close(volume, (1. / 3.) * (1. / 4.) * (1. / 2.)));
  r3d::Few<r3d::Vector<3>, 4> far_verts = {
      {0, 0, 4}, {1, 0, 4}, {0, 1, 4}, {0, 0, 5}};
  r3d::Polytope<3> null_intersection;
  r3d::intersect_simplices(null_intersection, verts, far_verts);
  OMEGA_H_CHECK(null_intersection.nverts == 0);
}

static void test_2d() {
  r3d::Few<r3d::Vector<2>, 3> verts = {{0, 0}, {1, 0}, {0, 1}};
  auto faces = r3d::faces_from_verts(verts);
  OMEGA_H_CHECK(Omega_h::are_close(
      Omega_h::from_r3d(faces[0].n), Omega_h::vector_2(0, 1)));
  OMEGA_H_CHECK(Omega_h::are_close(faces[0].d, 0));
  OMEGA_H_CHECK(Omega_h::are_close(Omega_h::from_r3d(faces[1].n),
      -Omega_h::normalize(Omega_h::vector_2(1, 1))));
  OMEGA_H_CHECK(Omega_h::are_close(faces[1].d, -faces[1].n[0]));
  OMEGA_H_CHECK(Omega_h::are_close(
      Omega_h::from_r3d(faces[2].n), Omega_h::vector_2(1, 0)));
  OMEGA_H_CHECK(Omega_h::are_close(faces[2].d, 0));
  r3d::Polytope<2> a;
  r3d::init(a, verts);
  OMEGA_H_CHECK(a.nverts == 3);
  auto b = a;
  r3d::clip(b, r3d::Few<r3d::Plane<2>, 1>({{{1, 0}, -0.5}}));
  OMEGA_H_CHECK(b.nverts == 3);
  auto area = r3d::measure(b);
  OMEGA_H_CHECK(Omega_h::are_close(area, Omega_h::square(0.5) / 2.0));
  auto c = a;
  r3d::clip(c, r3d::Few<r3d::Plane<2>, 1>({{{-1, 0}, 0.5}}));
  OMEGA_H_CHECK(c.nverts == 4);
  area = r3d::measure(c);
  OMEGA_H_CHECK(
      Omega_h::are_close(area, (1. / 2.) - (Omega_h::square(0.5) / 2.0)));
  r3d::Few<r3d::Vector<2>, 3> verts2 = {{0, 0}, {1, 0}, {1, 1}};
  r3d::Polytope<2> intersection;
  r3d::intersect_simplices(intersection, verts, verts2);
  area = r3d::measure(intersection);
  OMEGA_H_CHECK(Omega_h::are_close(area, 1. / 4.));
  r3d::Few<r3d::Vector<2>, 3> far_verts = {{0, 4}, {1, 4}, {0, 5}};
  r3d::Polytope<2> null_intersection;
  r3d::intersect_simplices(null_intersection, verts, far_verts);
  OMEGA_H_CHECK(null_intersection.nverts == 0);
}

static void test_2d_self_intersection() {
  r3d::Few<r3d::Vector<2>, 3> verts = {
      {-1660.4524999999999, 211.69999999999999},
      {-1582.3433333333332, 211.69999999999999},
      {-1582.3433333333332, 289.80916666666667}};
  r3d::Polytope<2> intersection;
  r3d::intersect_simplices(intersection, verts, verts);
  auto expected = 0.5 * (1660.4524999999999 - 1582.3433333333332) *
                  (289.80916666666667 - 211.69999999999999);
  OMEGA_H_CHECK(Omega_h::are_close(r3d::measure(intersection), expected));
}

static void test_3d_self_intersection() {
  const double x0 = -1660.4524999999999, x1 = -1582.3433333333332;
  const double y0 = 211.69999999999999, y1 = 289.80916666666667;
  const double z0 = 35.130000000000003, z1 = 113.23916666666668;
  r3d::Few<r3d::Vector<3>, 4> verts = {
      {x0, y0, z0}, {x1, y0, z0}, {x1, y1, z0}, {x1, y0, z1}};
  r3d::Polytope<3> intersection;
  r3d::intersect_simplices(intersection, verts, verts);
  auto expected = (x1 - x0) * (y1 - y0) * (z1 - z0) / 6.0;
  OMEGA_H_CHECK(
      Omega_h::are_close(std::abs(r3d::measure(intersection)), expected));
}

int main() {
  test_2d();
  test_3d();
  test_2d_self_intersection();
  test_3d_self_intersection();
}

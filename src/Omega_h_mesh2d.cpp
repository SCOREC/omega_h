//
// Created by Matthew McCall on 1/29/24.
//

#include "Omega_h_mesh2d.hpp"

namespace Omega_h {

void Mesh2D::set_dim(Int dim_in) {
  OMEGA_H_CHECK(dim_ == -1);
  OMEGA_H_CHECK(dim_in == 1 || dim_in == 2);
  dim_ = dim_in;
}

std::optional<Model2D> Mesh2D::getModel() const {
  return model;
}

Model2D Mesh2D::updateModel() {
  model = Model2D(*this);
  return *model;
}

} // Omega_h
#include <SimModel.h>
#include <SimUtil.h>
#include "Omega_h_model2d.hpp"

namespace Omega_h {

Model2D Model2D::SimModel2D_load(std::string const& filename) {
  pNativeModel nm = NULL;
  pProgress p = NULL;
  pGModel g = GM_load(filename.c_str(), nm, p);
  return Model2D();
}
  
}//end namespace Omega_h

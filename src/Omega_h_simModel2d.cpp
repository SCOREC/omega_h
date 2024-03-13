#include <SimModel.h>
#include <SimUtil.h>
#include "Omega_h_model2d.hpp"

namespace Omega_h {

bool isModel2D(pGModel mdl) {
  return (GM_numRegions(mdl) == 0);
}

Model2D Model2D::SimModel2D_load(std::string const& filename) {
  pNativeModel nm = NULL;
  pProgress p = NULL;
  pGModel g = GM_load(filename.c_str(), nm, p);
  const char* msg = "Simmetrix GeomSim model is not 2D... exiting\n";
  OMEGA_H_CHECK_MSG(isModel2D(g), msg);
  return Model2D();
}
  
}//end namespace Omega_h

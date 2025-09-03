#include "Omega_h_bsplineModel2d.hpp"

namespace Omega_h {
  BsplineModel2D::BsplineModel2D(filesystem::path const& omegahGeomFile)
  {
    read(omegahGeomFile);
  }

  BsplineModel2D::BsplineModel2D(filesystem::path const& geomSimModelFile, filesystem::path const& spline) :
     Model2D( Omega_h::Model2D::SimModel2D_load(geomSimModelFile.string()) )
  {
    //read from spline file and set private arrays
  }
  
  Reals BsplineModel2D::eval(LOs splineIds, Reals localCoords) {
    //port Bspline.cpp::eval here
    return Reals();
  }

  void BsplineModel2D::write(filesystem::path const& outOmegahGeomFile) {
    //write the private member arrays to outOmegahGeomFile via write_array(...)
    //calls from Omega_h_file.hpp
  }

  void BsplineModel2D::read(filesystem::path const& inOmegahGeomFile) {
    //read the private member arrays from inOmegahGeomFile via read_array(...)
    //calls from Omega_h_file.hpp
  }

} //end namespace Omega_h

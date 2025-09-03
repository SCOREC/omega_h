#ifndef OMEGA_H_BSPLINEMODEL2D_HPP
#define OMEGA_H_BSPLINEMODEL2D_HPP

#include "Omega_h_file.hpp"
#include "Omega_h_model2d.hpp"

namespace Omega_h {

class BsplineModel2D : public Model2D {
  private:
    LOs splineToCtrlPts; //offset array mapping splines to control points
    Reals ctrlPts; //each ctrl point is defined by two doubles

    LOs splineToKnots; //offset array mapping splines to knots
    Reals knots; //each knot is defined by two doubles

    LOs order; //polynomial order of each spline

    void read(filesystem::path const& omegahGeomFile);

  public:
    /**
     * fill the topo and splines from a file of omegah binary arrays
     */
    BsplineModel2D(filesystem::path const& omegahGeomFile);

    /**
     * create the omegah model from a Simmetrix GeomSim model file and a file of omegah binary arrays with spline info
     */
    BsplineModel2D(filesystem::path const& geomSimModelFile, filesystem::path const& spline);

    /**
     * given a spline and a local coordinate (parametric) 
     * along that spline, return the corresponding cartesian
     * coordinate 
     * @remark splineIds.size() == localCoords.size()
     * @param splineIds (in) array of spline Ids
     * @param localCoord (in) array of local coordinates
     * @return array of cartesian coordinates in the order of the provided
     *         inputs, the layout is (x0, y0, x1, y1, ..., xN-1, yN-1)
     */
    Reals eval(LOs splineIds, Reals localCoords);

    /**
     * create an omegah model file with topology and spline info
     */
    void write(filesystem::path const& outOmegahGeomFile);

};//end BsplineModel2D

}//end namespace Omega_h

#endif

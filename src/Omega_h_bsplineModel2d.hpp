#ifndef OMEGA_H_BSPLINEMODEL2D_HPP
#define OMEGA_H_BSPLINEMODEL2D_HPP

#include "Omega_h_file.hpp"

namespace Omega_h {

class BsplineModel2D : public Model2D {
  private:
    LOs splineToCtrlPts; //offset array mapping splines to control points
    Reals ctrlPts; //each ctrl point is defined by two doubles

    LOs splineToKnots; //offset array mapping splines to knots
    Reals knots; //each knot is defined by two doubles

    LOs order; //polynomial order of each spline

  public:
    /**
     * fill the splines from a file of omegah binary arrays
     * it is assumed the file contains the needed model topology and spline info
     */
    BsplineModel2D(filesystem::path const& path);
    /**
     * not sure if this makes sense...
     */
    BsplineModel2D(LOs splineToCtrlPtsIn, Reals ctrlPtsIn, LOs splineToKnotsIn, Reals knotsIn, LOs orderIn) :
      splineToCtrlPts(splineToCtrlPtsIn), ctrlPts(ctrlPtsIn), 
      splineToKnots(splineToKnotsIn), knots(knotsIn),
      order(orderIn) { assert(splineToCtrlPts.size() == splineToKnots() }
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

};

}//end namespace Omega_h

#endif

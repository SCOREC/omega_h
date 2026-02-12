#ifndef OMEGA_H_BSPLINEMODEL2D_HPP
#define OMEGA_H_BSPLINEMODEL2D_HPP

#include "Omega_h_file.hpp"
#include "Omega_h_model2d.hpp"

namespace Omega_h {

class BsplineModel2D : public Model2D {
  private:
    LOs splineToCtrlPts; //offset array mapping splines to control points
    Reals ctrlPtsX; //each ctrl point is defined by two doubles
    Reals ctrlPtsY; //each ctrl point is defined by two doubles

    LOs splineToKnots; //offset array mapping splines to knots
    Reals knotsX; //each knot is defined by one double
    Reals knotsY; //each knot is defined by one double

    LOs order; //polynomial order of each spline

    Write<LO> edgeIdToSplineIdx; //maps model edge ID -> dense spline array index

    void read(filesystem::path const& omegahGeomFile);
    Write<LO> buildEdgeIdMapping();

  public:
    /**
     * fill the topo and splines from a file of omegah binary arrays
     */
    BsplineModel2D(filesystem::path const& omegahGeomFile);

    /**
     * create the omegah model from a Simmetrix GeomSim model file and a file of omegah binary arrays with spline info
     */
    BsplineModel2D(filesystem::path const& geomSimModelFile, filesystem::path const& spline);

    const LOs& getSplineToKnots() const {
      return splineToKnots;
    }

    /** \brief given a parametric coordinate and the id of the corresponding
    *        spline, return the corresponding cartesian coordinates of that point
    *
    * Not clear what the best API for this is:
    *  a) CSR to group evaluation points by spline
    *  b) two arrays of equal length: model edge ids and evaluation points
    *  c) ...
    * For now, taking approach (b).
    *
    * This implementation was ported from BSpline.cc in
    * github.com/scorec/simLandIceMeshGen main @ 9f85d2e .
    *
    * \param edgeIds (In) array of model edge ids for each pair of parametric coordinates
    *                     specified in localCoords
    * \param edgeToLocalCoords (In) offset array from edgeIds to the set of
    *                               localCoordinates to evaluate for that edge
    *                               e.g., edgeToLocalCoords[i] and
    *                               edgeToLocalCoords[i+1] respectively define the
    *                               starting and last (exclusive) indices into
    *                               localCoords for edge i
    * \param localCoords (In) array of parametric coordinates (x0,x1,x2,...,xN-1)
    * \return the array of coordinates (x0,y0,x1,y1,...,xN-1,yN-1) in order
    *         they were specified in the input arrays
    */
    Reals eval(Omega_h::LOs edgeIds, Omega_h::LOs edgeToLocalCoords, Omega_h::Reals localCoords);

    /**
     * create an omegah model file with topology and spline info
     */
    void write(filesystem::path const& outOmegahGeomFile);

};//end BsplineModel2D

Reals bspline_get_snap_warp(Mesh* mesh, BsplineModel2D* mdl, bool verbose);

}//end namespace Omega_h

#endif

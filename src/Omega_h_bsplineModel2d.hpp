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

    /** \brief Evaluate B-splines at given parametric coordinates to compute cartesian positions
    *
    * Evaluates the B-spline representation of geometric model edges at specified parametric
    * coordinates and returns the corresponding cartesian (x,y) positions.
    *
    * This implementation was ported from BSpline.cc in
    * github.com/scorec/simLandIceMeshGen main @ 9f85d2e .
    *
    * \param edgeIds (In) Array of model edge IDs (NOT dense spline indices) for which to
    *                     evaluate B-splines. These IDs must match the edge IDs from the
    *                     geometric model topology (obtained via getEdgeIds()). The function
    *                     internally maps these model IDs to dense spline array indices.
    *                     Size: number of edges to evaluate.
    *
    * \param edgeToLocalCoords (In) CSR-style offset array mapping each edge in edgeIds to
    *                               its set of parametric coordinates in localCoords.
    *                               For edge i, the parametric coordinates are stored in
    *                               localCoords[edgeToLocalCoords[i]] through
    *                               localCoords[edgeToLocalCoords[i+1]-1] (exclusive end).
    *                               Size: edgeIds.size() + 1
    *
    * \param localCoords (In) Parametric coordinates in [0,1] at which to evaluate the splines.
    *                         Each value represents a position along its corresponding edge's
    *                         parametric domain. Grouped by edge according to edgeToLocalCoords.
    *                         Size: edgeToLocalCoords.last()
    *
    * \return Reals array containing interleaved (x,y) cartesian coordinates for all evaluation
    *         points, in the same order as specified in localCoords. Format:
    *         (x0,y0, x1,y1, ..., xN-1,yN-1) where N = localCoords.size().
    *         Size: localCoords.size() * 2
    *
    * \pre edgeIds.size() + 1 == edgeToLocalCoords.size()
    * \pre edgeToLocalCoords.last() == localCoords.size()
    * \pre All values in edgeIds must be valid model edge IDs present in the geometric model
    * \pre All values in localCoords should be in range [0,1]
    *
    * \note Model edge IDs are arbitrary integers assigned by the CAD system and may be
    *       non-continuous or start at values other than 0. This function handles the
    *       mapping from model edge IDs to internal dense spline array indices.
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

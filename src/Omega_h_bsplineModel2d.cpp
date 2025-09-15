#include "Omega_h_bsplineModel2d.hpp"
#include "Omega_h_for.hpp" //parallel_for
#include <Kokkos_Core.hpp>
#include <fstream>

namespace Omega_h {
  bool areKnotsIncreasing(LOs splineToKnots, Reals x, Reals y) {
    assert(x.size() == y.size());
    parallel_for(splineToKnots.size()-1, OMEGA_H_LAMBDA(LO i) {
        for (auto j = splineToKnots[i]; j < splineToKnots[i+1]-1; j++) {
          OMEGA_H_CHECK(x[j] <= x[j+1]);
          OMEGA_H_CHECK(y[j] <= y[j+1]);
        }
    });
    return true;
  }

  BsplineModel2D::BsplineModel2D(filesystem::path const& omegahGeomFile)
  {
    read(omegahGeomFile);
  }

  BsplineModel2D::BsplineModel2D(filesystem::path const& geomSimModelFile, filesystem::path const& splineFile) :
     Model2D( Omega_h::Model2D::SimModel2D_load(geomSimModelFile.string()) )
  {
    std::ifstream file(splineFile.string());
    OMEGA_H_CHECK(file.is_open());
    //FIXME copied swap and compression test from read(stream, mesh, ...) in Omega_h_file.cpp
    bool compressed = true;
    bool needs_swapping = false;

    binary::read_array(file, splineToCtrlPts, compressed, needs_swapping);
    binary::read_array(file, splineToKnots, compressed, needs_swapping);
    binary::read_array(file, ctrlPtsX, compressed, needs_swapping);
    binary::read_array(file, ctrlPtsY, compressed, needs_swapping);
    binary::read_array(file, knotsX, compressed, needs_swapping);
    binary::read_array(file, knotsY, compressed, needs_swapping);
    binary::read_array(file, order, compressed, needs_swapping);

    const auto numEdges = getNumEnts(OMEGA_H_EDGE);
    OMEGA_H_CHECK(order.size() == numEdges);
    OMEGA_H_CHECK(splineToCtrlPts.size() == numEdges+1); //offset array
    OMEGA_H_CHECK(splineToKnots.size() == numEdges+1); //offset array
    OMEGA_H_CHECK(splineToCtrlPts.get(numEdges) == ctrlPtsX.size()); //last entry should be numCtrlPts
    OMEGA_H_CHECK(ctrlPtsX.size() == ctrlPtsY.size());
    OMEGA_H_CHECK(splineToKnots.get(numEdges) == knotsX.size()); //last entry should be numKnots
    OMEGA_H_CHECK(knotsX.size() == knotsY.size());
    OMEGA_H_CHECK(areKnotsIncreasing(splineToKnots, knotsX, knotsY));
  }

  /** \brief given a pair of parametric coordinates along each spline
    *        return the corresponding cartesian coordinates of that point
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
    *                       specified in localCoords
    * \param localCoords (In) array of parametric coordinates (x0,x1,x2,...,xN-1)
    * \return the array of coordinates (x0,y0,x1,y1,...,xN-1,yN-1) in order
    *         they were specified in the input arrays
    */
  Reals BsplineModel2D::eval(LOs edgeIds, Reals localCoords) {
    assert(edgeIds.size()*2 == localCoords.size());
    Write<Real> coords(localCoords.size()); 

    //localize class members for use in lambda
    const auto s2k = splineToKnots;
    const auto kx = knotsX;
    const auto ky = knotsY;
    const auto s2cp = splineToCtrlPts;
    const auto cx = ctrlPtsX;
    const auto cy = ctrlPtsY;
    const auto ord = order;

    Write<Real> pts(ctrlPtsX.size());

    //TODO use team based loop or expand index range by 2x to run eval
    //on X and Y separately
    parallel_for(edgeIds.size(), OMEGA_H_LAMBDA(LO i) {
      const auto spline = edgeIds[i];
      const auto sOrder = ord[i];
      auto sKnots = Kokkos::subview(knotsX.view(), std::make_pair(s2k[spline], s2k[spline+1]));
      auto sCtrlPts = Kokkos::subview(cx.view(), std::make_pair(s2cp[spline], s2cp[spline+1]));
      // first find the interval of x in knots
      int leftKnot = sOrder - 1;
      int leftPt = 0;
      while (sKnots(leftKnot + 1) < localCoords[i]) {
        leftKnot++;
        leftPt++;
        if (leftKnot == sKnots.size() - 1)
          break;
      }

      auto ptsLocal = Kokkos::subview(pts.view(), std::make_pair(leftPt, leftPt + sOrder));
      //vector<double> localKnots(&(knots[leftKnot - sOrder + 2]), &(knots[leftKnot + sOrder]));
      const auto localKnots = Kokkos::subview(sKnots, std::make_pair(leftKnot - sOrder + 2, leftKnot + sOrder));

      for (int r = 1; r <= sOrder; r++) {
        // from bottom to top to save a buff
        for (int i = sOrder - 1; i >= r; i--) {
          double a_left = localKnots(i - 1);
          double a_right = localKnots(i + sOrder - r - 1);
          double alpha;
          if (a_right == a_left)
            alpha = 0.; // not sure??
          else
            alpha = (localCoords[i] - a_left) / (a_right - a_left);
          ptsLocal(i) = (1. - alpha) * ptsLocal(i - 1) + alpha * ptsLocal(i);
        }
      }
      coords[i*2] = ptsLocal(sOrder - 1);
    });
    return coords;
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

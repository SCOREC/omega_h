#include "Omega_h_bsplineModel2d.hpp"
#include "Omega_h_for.hpp" //parallel_for
#include "Omega_h_array_ops.hpp" //get_max
#include "Omega_h_int_scan.hpp" //offset_scan
#include <Kokkos_Core.hpp>
#include <fstream>

namespace {
  template<typename ViewConstT, typename ViewT>
  OMEGA_H_INLINE double splineEval(const int sOrder,
      ViewConstT sKnots, ViewConstT sCtrlPts, ViewT work,
      const double parametricCoord) {
    // first find the interval of parametricCoord in knots
    int leftKnot = sOrder - 1;
    int leftPt = 0;
    while (sKnots(leftKnot + 1) < parametricCoord) {
      leftKnot++;
      leftPt++;
      if (leftKnot == sKnots.size() - 1)
        break;
    }

    //initialize the work array
    int j = 0;
    for(int i=leftPt; i<leftPt+sOrder; i++) {
      work(j++) = sCtrlPts(i);
    }

    const auto localKnots = Kokkos::subview(sKnots, std::make_pair(leftKnot - sOrder + 2, leftKnot + sOrder));

    for (int r = 1; r <= sOrder; r++) {
      // from bottom to top to save a buff
      for (int i = sOrder - 1; i >= r; i--) {
        const double a_left = localKnots(i - 1);
        const double a_right = localKnots(i + sOrder - r - 1);
        double alpha = 0;
        if (a_right != a_left)
          alpha = (parametricCoord - a_left) / (a_right - a_left);
        work(i) = (1. - alpha) * work(i - 1) + alpha * work(i);
      }
    }
    return work(sOrder - 1);
  }
};

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
    OMEGA_H_CHECK(splineToCtrlPts.last() == ctrlPtsX.size()); //last entry should be numCtrlPts
    OMEGA_H_CHECK(ctrlPtsX.size() == ctrlPtsY.size());
    OMEGA_H_CHECK(splineToKnots.last() == knotsX.size()); //last entry should be numKnots
    OMEGA_H_CHECK(knotsX.size() == knotsY.size());
    OMEGA_H_CHECK(areKnotsIncreasing(splineToKnots, knotsX, knotsY));
  }


  Reals BsplineModel2D::eval(LOs edgeIds, LOs edgeToLocalCoords, Reals localCoords) {
    assert(edgeIds.size()+1 == edgeToLocalCoords.size());
    assert(edgeToLocalCoords.last() == localCoords.size());
    Write<Real> coords(localCoords.size()*2);  //x and y for each coordinate

    //localize class members for use in lambda
    const auto s2k = splineToKnots;
    const auto kx = knotsX;
    const auto ky = knotsY;
    const auto s2cp = splineToCtrlPts;
    const auto cx = ctrlPtsX;
    const auto cy = ctrlPtsY;
    const auto ord = order;

    std::cout << "max order " << get_max(order) << "\n";

    // Need a work array whose length is sum(spline[i].order) where i is the index
    // for an edge specified in edgeIds and an offset array constructed from an
    // array containing the order of each edge in edgeIds.
    // The offset array is used to create a kokkos::subview of the work array
    // for use in splineEval(...).
    // The work array needs to be initialized with the control points for the
    // given spline; this is done inside splineEval(...).
    Write<LO> edgeOrder(edgeIds.size());
    parallel_for(edgeIds.size(), OMEGA_H_LAMBDA(LO i) {
        const auto sIdx = edgeIds[i];
        edgeOrder[i] = order[sIdx];
    });
    auto edgeOrderOffset = offset_scan(Omega_h::read(edgeOrder));
    Write<Real> workArray(edgeOrderOffset.last()); //storage for intermediate values 

    //TODO use team based loop or expand index range by 2x to run eval
    //on X and Y separately
    parallel_for(edgeIds.size(), OMEGA_H_LAMBDA(LO i) {
        const auto spline = edgeIds[i];
        const auto sOrder = ord[i];
        const auto knotRange = std::make_pair(s2k[spline], s2k[spline+1]);
        const auto ctrlPtRange = std::make_pair(s2cp[spline], s2cp[spline+1]);

        auto xKnots = Kokkos::subview(kx.view(), knotRange);
        auto xCtrlPts = Kokkos::subview(cx.view(), ctrlPtRange);

        auto yKnots = Kokkos::subview(ky.view(), knotRange);
        auto yCtrlPts = Kokkos::subview(cy.view(), ctrlPtRange);

        const auto orderRange = std::make_pair(edgeOrderOffset[i], edgeOrderOffset[i+1]);
        auto work = Kokkos::subview(workArray.view(), orderRange);

        for(int j = edgeToLocalCoords[i]; j < edgeToLocalCoords[i+1]; j++) {
          coords[i*2] = splineEval(sOrder, xKnots, xCtrlPts, work, localCoords[j]);
          coords[i*2+1] = splineEval(sOrder, yKnots, yCtrlPts, work, localCoords[j]);
        }
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

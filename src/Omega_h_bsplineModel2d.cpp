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

  Reals bspline_get_snap_warp(Mesh* mesh, BsplineModel2D* mdl, bool verbose) {
    OMEGA_H_CHECK(mesh->dim() == 2);
    auto class_dims = mesh->get_array<I8>(VERT, "class_dim");
    auto class_ids = mesh->get_array<ClassId>(VERT, "class_id");
    auto class_parametric = mesh->get_array<Real>(VERT, "class_parametric");
    OMEGA_H_CHECK(class_parametric.size() == mesh->nverts()*2);

    const auto sideDim = mesh->dim()-1;
    auto sidesToMeshVerts = mesh->ask_revClass_downAdj(sideDim, VERT);

    //get the location of mesh vertices along the geometric model entity they
    //are classified on using their parametric coordinates along that model
    //entity
    const auto sidesToSamples = sidesToMeshVerts.a2ab;
    const auto meshEntIds = sidesToMeshVerts.ab2b;
    const int numSides = sidesToSamples.size()-1;
    const int numSamples = sidesToSamples.last();
    auto sideIds = Write<Omega_h::LO>(numSides, 0, 1, "splineIds");
    auto setSideIds = OMEGA_H_LAMBDA(Omega_h::LO side) {
      //all mesh entities associated with this side will have the same side id,
      //use the first one to store the id
      const auto firstMeshEnt = meshEntIds[sidesToSamples[side]];
      sideIds[side] = class_ids[firstMeshEnt];
    };
    Omega_h::parallel_for(sidesToSamples.size(), setSideIds, "setSideIds");

    auto samplePts = Omega_h::Write<Omega_h::Real>(numSamples, "splineSamplePoints");
    auto setSamplePts = OMEGA_H_LAMBDA(Omega_h::LO side) {
      for( auto ab = sidesToSamples[side]; ab < sidesToSamples[side+1]; ab++ ) {
         const auto meshEnt = meshEntIds[ab];
         //class_parametric is a tag with two components per ent
         //only need first component for parametric coord along a spline
         samplePts[ab] = class_parametric[meshEnt*2];
      }
    };
    Omega_h::parallel_for(sidesToSamples.size(), setSamplePts, "setSamplePts");

    const auto pts = mdl->eval(sideIds,sidesToSamples,samplePts);

    //compute warp vector for mesh vertices classified on the model boundary (sides)
    auto coords = mesh->coords();
    auto warp = Write<Real>(mesh->nverts() * 2, 0);
    auto setWarpVectors = OMEGA_H_LAMBDA(Omega_h::LO side) {
      for( auto ab = sidesToSamples[side]; ab < sidesToSamples[side+1]; ab++ ) {
         const auto meshEnt = meshEntIds[ab];
         const auto currentPt = get_vector<2>(coords, meshEnt);
         const auto targetPt = get_vector<2>(pts, meshEnt);
         auto d = vector_2(0, 0);
         d = targetPt - currentPt;
         set_vector(warp, meshEnt, d);
      }
    };
    Omega_h::parallel_for(sidesToSamples.size(), setWarpVectors, "setWarpVectors");
    return warp;
  }

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

    //the following is from src/Omega_h_file.cpp read(...)
    unsigned char const magic[2] = {0xa1, 0x1a};
    unsigned char magic_in[2];
    file.read(reinterpret_cast<char*>(magic_in), sizeof(magic));
    OMEGA_H_CHECK(magic_in[0] == magic[0]);
    OMEGA_H_CHECK(magic_in[1] == magic[1]);
    bool needs_swapping = !is_little_endian_cpu();
    int compressed;
    binary::read_value(file, compressed, needs_swapping);

    binary::read_array(file, splineToCtrlPts, compressed, needs_swapping);
    binary::read_array(file, splineToKnots, compressed, needs_swapping);
    binary::read_array(file, ctrlPtsX, compressed, needs_swapping);
    binary::read_array(file, ctrlPtsY, compressed, needs_swapping);
    binary::read_array(file, knotsX, compressed, needs_swapping);
    binary::read_array(file, knotsY, compressed, needs_swapping);
    binary::read_array(file, order, compressed, needs_swapping);

    file.close();

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
    const auto s2cp = splineToCtrlPts;
    const auto kx = knotsX.view();
    const auto ky = knotsY.view();
    const auto cx = ctrlPtsX.view();
    const auto cy = ctrlPtsY.view();
    const auto ord = order;

    std::cout << "max order " << get_max(ord) << "\n";

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
        edgeOrder[i] = ord[sIdx];
    });
    auto edgeOrderOffset = offset_scan(Omega_h::read(edgeOrder));
    Kokkos::View<Real*> workArray("workArray", edgeOrderOffset.last()); //storage for intermediate values

    //TODO use team based loop or expand index range by 2x to run eval
    //on X and Y separately
    parallel_for(edgeIds.size(), OMEGA_H_LAMBDA(LO i) {
        const auto spline = edgeIds[i];
        const auto sOrder = ord[i];
        const auto knotRange = std::make_pair(s2k[spline], s2k[spline+1]);
        const auto ctrlPtRange = std::make_pair(s2cp[spline], s2cp[spline+1]);

        auto xKnots = Kokkos::subview(kx, knotRange);
        auto xCtrlPts = Kokkos::subview(cx, ctrlPtRange);

        auto yKnots = Kokkos::subview(ky, knotRange);
        auto yCtrlPts = Kokkos::subview(cy, ctrlPtRange);

        const auto orderRange = std::make_pair(edgeOrderOffset[i], edgeOrderOffset[i+1]);
        auto work = Kokkos::subview(workArray, orderRange);

        for(int j = edgeToLocalCoords[i]; j < edgeToLocalCoords[i+1]; j++) {
          coords[j*2] = splineEval(sOrder, xKnots, xCtrlPts, work, localCoords[j]);
          coords[j*2+1] = splineEval(sOrder, yKnots, yCtrlPts, work, localCoords[j]);
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

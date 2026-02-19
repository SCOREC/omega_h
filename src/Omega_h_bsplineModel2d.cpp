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

    const auto localKnots = Kokkos::subview(sKnots, Kokkos::make_pair(leftKnot - sOrder + 2, leftKnot + sOrder));

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

  Omega_h::Write<Omega_h::LO> buildEdgeIdMapping(Omega_h::BsplineModel2D& mdl) {
    auto edgeIds = mdl.getEdgeIds();  // Model entity IDs from parent class
    if (edgeIds.size() == 0) return Omega_h::Write<Omega_h::LO>();

    auto maxId = get_max(edgeIds);
    Omega_h::Write<Omega_h::LO> mapping(maxId + 1, -1);  // Initialize with -1 (invalid)

    // Map each model edge ID to its dense spline array index
    Omega_h::parallel_for(edgeIds.size(), OMEGA_H_LAMBDA(Omega_h::LO i) {
      mapping[edgeIds[i]] = i;
    });

    return mapping;
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

    //////////////////////////////////////////////////////////////////////////
    // ask_revClass_downAdj returns a CSR where the offset array is indexed by
    // model entity ID (from derive_revClass: n_gents = get_max(class_ids)+1).
    // When model entity IDs are non-contiguous (e.g., start at 1 or have gaps),
    // this CSR has degree-zero entries for the missing IDs. These entries produce
    // empty inner loops and are harmless — no filtering is needed.
    //////////////////////////////////////////////////////////////////////////
    auto sidesToMeshVerts_revClass = mesh->ask_revClass_downAdj(sideDim, VERT);
    const auto sidesToSamples = sidesToMeshVerts_revClass.a2ab;
    // Sparse size: sidesToSamples.size()-1 = max_model_id + 1
    const int numSidesSparse = sidesToSamples.size()-1;

    //get the location of mesh vertices along the geometric model entity they
    //are classified on using their parametric coordinates along that model
    //entity
    const auto meshEntIds = sidesToMeshVerts_revClass.ab2b;
    const int numSamples = sidesToSamples.last();

    auto samplePts = Omega_h::Write<Omega_h::Real>(numSamples, "splineSamplePoints");
    auto setSamplePts = OMEGA_H_LAMBDA(Omega_h::LO side) {
      for( auto ab = sidesToSamples[side]; ab < sidesToSamples[side+1]; ab++ ) {
         const auto meshEnt = meshEntIds[ab];
         //class_parametric is a tag with two components per ent
         //only need first component for parametric coord along a spline
         const auto pos = class_parametric[meshEnt*2];
         samplePts[ab] = pos;
      }
    };
    Omega_h::parallel_for(numSidesSparse, setSamplePts, "setSamplePts");

    OMEGA_H_CHECK(sideDim==1); //no support for 1d or 3d models yet
    // Position i in the sparse CSR equals model entity ID i (derive_revClass indexes
    // by class_id directly).  Pass [0..numSidesSparse-1] as model IDs; eval() skips
    // zero-degree entries (gaps) so no spline lookup is attempted for them.
    LOs sideIds(numSidesSparse, 0, 1);  // [0, 1, 2, ..., numSidesSparse-1]
    const auto pts = mdl->eval(sideIds,sidesToSamples,samplePts);

    //compute warp vector for mesh vertices classified on the model boundary (sides)
    auto coords = mesh->coords();
    auto warp = Write<Real>(mesh->nverts() * 2, 0);
    auto setWarpVectors = OMEGA_H_LAMBDA(Omega_h::LO side) {
      //see note in setSamplePts lambda
      for( auto ab = sidesToSamples[side]; ab < sidesToSamples[side+1]; ab++ ) {
         const auto meshEnt = meshEntIds[ab];
         const auto currentPt = get_vector<2>(coords, meshEnt);
         const auto targetPt = get_vector<2>(pts, ab);
         auto d = vector_2(0, 0);
         d = targetPt - currentPt;
         set_vector(warp, meshEnt, d);
      }
    };
    Omega_h::parallel_for(numSidesSparse, setWarpVectors, "setWarpVectors");
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

#if defined(OMEGA_H_USE_SIMMODSUITE)
  BsplineModel2D::BsplineModel2D(filesystem::path const& geomSimModelFile, filesystem::path const& splineFile) :
     Model2D( Omega_h::Model2D::SimModel2D_load(geomSimModelFile.string()) ) //need model topology
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

    edgeIdToSplineIdx = buildEdgeIdMapping(*this);
  }
#else
  BsplineModel2D::BsplineModel2D(filesystem::path const&, filesystem::path const&)
  {
    std::cerr << "Error: BsplineModel2D requires building Omega_h with SimModSuite enabled... exiting\n";
    exit(EXIT_FAILURE);
  }
#endif


  Reals BsplineModel2D::eval(LOs edgeIds, LOs edgeToLocalCoords, Reals localCoords) {
    assert(edgeIds.size()+1 == edgeToLocalCoords.size());
    assert(edgeToLocalCoords.last() == localCoords.size());
    Write<Real> coords(localCoords.size()*2);  //x and y for each coordinate

    // Convert model entity IDs to spline indices
    Write<LO> splineIndices(edgeIds.size());
    auto idToIdx = edgeIdToSplineIdx;  // Capture for lambda
    parallel_for(edgeIds.size(), OMEGA_H_LAMBDA(LO i) {
      LO modelId = edgeIds[i];
      OMEGA_H_CHECK(modelId < idToIdx.size());
      splineIndices[i] = idToIdx[modelId];
    });
    auto sIdx = LOs(splineIndices);

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
    // Zero-degree entries (gaps in sparse model entity ID space) contribute order=0
    // so their work array sub-range is empty and the inner eval loop does not execute.
    // FIXME - replace the following with reduction over ord
    Write<LO> edgePolynomialOrder(edgeIds.size(), 0);
    parallel_for(edgeIds.size(), OMEGA_H_LAMBDA(LO i) {
        if (edgeToLocalCoords[i] < edgeToLocalCoords[i+1]) {  // non-zero degree
          const auto spline = sIdx[i];
          assert(spline < ord.size());
          edgePolynomialOrder[i] = ord[spline];
        }
    });
    auto edgePolyOrderOffset = offset_scan(Omega_h::read(edgePolynomialOrder));
    Kokkos::View<Real*> workArray("workArray", edgePolyOrderOffset.last()); //storage for intermediate values

    //TODO use team based loop or expand index range by 2x to run eval
    //on X and Y separately
    parallel_for(edgeIds.size(), OMEGA_H_LAMBDA(LO i) {
        if (edgeToLocalCoords[i] < edgeToLocalCoords[i+1]) {  // non-zero degree (skip gaps)
          const auto spline = sIdx[i];
          const auto sOrder = ord[spline];
          const auto knotRange = Kokkos::make_pair(s2k[spline], s2k[spline+1]);
          const auto ctrlPtRange = Kokkos::make_pair(s2cp[spline], s2cp[spline+1]);

          auto xKnots = Kokkos::subview(kx, knotRange);
          auto xCtrlPts = Kokkos::subview(cx, ctrlPtRange);

          auto yKnots = Kokkos::subview(ky, knotRange);
          auto yCtrlPts = Kokkos::subview(cy, ctrlPtRange);

          const auto orderRange = Kokkos::make_pair(edgePolyOrderOffset[i], edgePolyOrderOffset[i+1]);
          auto work = Kokkos::subview(workArray, orderRange);

          for(int j = edgeToLocalCoords[i]; j < edgeToLocalCoords[i+1]; j++) {
            coords[j*2] = splineEval(sOrder, xKnots, xCtrlPts, work, localCoords[j]);
            coords[j*2+1] = splineEval(sOrder, yKnots, yCtrlPts, work, localCoords[j]);
          }
        }  // end non-zero degree
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

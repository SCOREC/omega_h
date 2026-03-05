#include "Omega_h_bsplineModel2d.hpp"
#include "Omega_h_for.hpp" //parallel_for
#include "Omega_h_array_ops.hpp" //get_max
#include "Omega_h_int_scan.hpp" //offset_scan
#include "Omega_h_atomics.hpp" //atomic_increment
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
    auto class_parametric = mesh->get_array<Real>(VERT, "class_parametric");
    OMEGA_H_CHECK(class_parametric.size() == mesh->nverts()*2);

    const auto sideDim = mesh->dim()-1;

    //////////////////////////////////////////////////////////////////////////
    // ask_revClass_downAdj returns a CSR where the offset array is indexed by
    // model entity ID (from derive_revClass: n_gents = get_max(class_ids)+1).
    // When model entity IDs are non-contiguous (e.g., start at 1 or have gaps),
    // this CSR has degree-zero entries for the missing IDs. These entries produce
    // empty inner loops and are harmless — no filtering is needed.
    // This returns mesh entities on the closure of model entities... bug?
    // The returned array does not remove duplicate downward adjacent entities
    //////////////////////////////////////////////////////////////////////////
    auto sidesToMeshVerts_revClass = mesh->ask_revClass_downAdj(sideDim, VERT);
    const auto sidesToSamples = sidesToMeshVerts_revClass.a2ab;
    // Sparse size: sidesToSamples.size()-1 = max_model_id + 1
    const int numSidesSparse = sidesToSamples.size()-1;

    const auto meshEntIds = sidesToMeshVerts_revClass.ab2b;

    { //debug
      auto s2s_h = HostRead<LO>(sidesToSamples);
      auto meshEntIds_h = HostRead<LO>(meshEntIds);
      std::cerr << "side, ab, meshEntIdx\n";
      for(int side = 0; side < numSidesSparse; side++) {
        for(auto ab = s2s_h[side]; ab < s2s_h[side+1]; ab++) {
           const auto meshEntIdx = meshEntIds_h[ab];
           std::cerr << side << ", " 
                     << ab << ", " 
                     << meshEntIdx << "\n";
        }
      }
    } //end debug

    //filter out mesh entities that are classified on model vertices
    //by definition, these vertices should not move/snap
    //need to create new filtered CSR from the revClass CSR
    // - X count number of vertices to keep per model edge - classified on an model edge
    // - X offset_scan to build offset array from degree array
    // - X allocate value array with size = offset.last()
    // - X run the counting loop again to copy into new values array using 
    //     atomic_increment to keep track of position to write for each model edge
    auto numSamplesPerSide = Omega_h::Write<LO>(numSidesSparse, 0, "numSamplesPerSide");
    auto countSamplesPerSide = OMEGA_H_LAMBDA(Omega_h::LO side) {
      for( auto ab = sidesToSamples[side]; ab < sidesToSamples[side+1]; ab++ ) {
         const auto meshEnt = meshEntIds[ab];
         if( class_dims[meshEnt] == 1 ) {
           atomic_increment(&numSamplesPerSide[side]);
         }
      }
    };
    Omega_h::parallel_for(numSidesSparse, countSamplesPerSide, "countSamplesPerSide");
    const auto samplesPerMdlEdge = offset_scan(read(numSamplesPerSide));
    //clear the degree array to reuse for recording position
    fill(numSamplesPerSide,0);
    auto samplePts = Write<Real>(samplesPerMdlEdge.last());
    auto meshVerts = Write<LO>(samplesPerMdlEdge.last());
    auto setSamplePts = OMEGA_H_LAMBDA(Omega_h::LO side) {
      for( auto ab = sidesToSamples[side]; ab < sidesToSamples[side+1]; ab++ ) {
         const auto startIdx = samplesPerMdlEdge[side];
         const auto meshEnt = meshEntIds[ab];
         if( class_dims[meshEnt] == 1 ) {
           const auto inc = atomic_fetch_add(&numSamplesPerSide[side],1);
           const auto idx = startIdx + inc;
           //class_parametric is a tag with two components per ent
           //only need first component for parametric coord along a spline
           const auto pos = class_parametric[meshEnt*2];
           samplePts[idx] = pos;
           meshVerts[idx] = meshEnt;
         }
      }
    };
    Omega_h::parallel_for(numSidesSparse, setSamplePts, "setSamplePts");

    OMEGA_H_CHECK(sideDim==1); //no support for 1d or 3d models yet
    // Position i in the sparse CSR equals model entity ID i (derive_revClass indexes
    // by class_id directly).  Pass [0..numSidesSparse-1] as model IDs; eval() skips
    // zero-degree entries (gaps) so no spline lookup is attempted for them.
    LOs sideIds(numSidesSparse, 0, 1);  // [0, 1, 2, ..., numSidesSparse-1]
    const auto pts = mdl->eval(sideIds,samplesPerMdlEdge,samplePts);
    //compute warp vector for mesh vertices classified on the model boundary (sides)
    auto coords = mesh->coords();
    auto warp = Write<Real>(mesh->nverts() * 2, 0);
    auto setWarpVectors = OMEGA_H_LAMBDA(Omega_h::LO side) {
      for( auto ab = samplesPerMdlEdge[side]; ab < samplesPerMdlEdge[side+1]; ab++ ) {
         const auto meshEnt = meshVerts[ab];
         const auto currentPt = get_vector<2>(coords, meshEnt);
         const auto targetPt = get_vector<2>(pts, ab);
         auto d = vector_2(0, 0);
         d = targetPt - currentPt;
         set_vector(warp, meshEnt, d);
      }
    };
    Omega_h::parallel_for(numSidesSparse, setWarpVectors, "setWarpVectors");

    { //debug
      auto sideIds_h = HostRead<LO>(sideIds);
      auto s2s_h = HostRead<LO>(samplesPerMdlEdge);
      auto sPts_h = HostRead<Real>(samplePts);
      auto pts_h = HostRead<Real>(pts);
      auto warp_h = HostRead<Real>(warp);
      auto coords_h = HostRead<Real>(mesh->coords());
      auto meshEntIds_h = HostRead<LO>(meshVerts);
      std::cerr << "side, ab, meshEntIdx, x, y, sPts_h[ab], evalPt_x, evalPt_y, warpVec_x, warpVec_y\n";
      for(int side = 0; side < sideIds_h.size(); side++) {
        for(auto ab = s2s_h[side]; ab < s2s_h[side+1]; ab++) {
           const auto meshEntIdx = meshEntIds_h[ab];
           const auto warpVec = get_vector<2>(warp, meshEntIdx);
           std::cerr << side << ", "
                     << ab << ", "
                     << meshEntIdx << ", "
                     << coords_h[meshEntIdx*2] << ", "
                     << coords_h[meshEntIdx*2+1] << ", "
                     << sPts_h[ab] << ", "
                     << pts_h[ab*2] << ", "
                     << pts_h[ab*2+1] << ", "
                     << warpVec[0] << ", "
                     << warpVec[1] << "\n";
        }
      }
    } //end debug

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

  BsplineModel2D::BsplineModel2D(BsplineModel2DTestModel testModel)
  {
    switch(testModel) {
      case BsplineModel2DTestModel::ModelWithOneEdge:
      {
        // Create a single edge from (0,0,0) to (1,0,0)

        // 1. Set up Model2D topology
        // Vertices
        LOs vtxIds({0, 1});
        Reals vtxCoords({0.0, 0.0, 0.0,   // vertex 0: (0,0,0)
                         1.0, 0.0, 0.0}); // vertex 1: (1,0,0)
        setVertexInfo(vtxIds, vtxCoords);

        // Edges and edge uses
        LOs edgeIds({0});  // one edge with ID 0
        LOs edgeUseIds({0, 1});  // two edge uses (forward and backward)
        LOs edgeUseOrientation({1, 0});  // use 0: same direction, use 1: opposite
        setEdgeInfo(edgeIds, edgeUseIds, edgeUseOrientation);

        // Faces (empty - no faces for a single edge)
        LOs faceIds({});
        setFaceIds(faceIds);

        // Loop uses (empty - no loops without faces)
        LOs loopUseIds({});
        LOs loopUseOrientation({});
        setLoopUseIdsAndDir(loopUseIds, loopUseOrientation);

        // Adjacencies
        // Edge 0 has two uses: 0 and 1
        Graph edgeToEdgeUse({0, 2}, {0, 1});

        // Each edge use connects to its two vertices
        // Edge use 0 (forward): vertex 0 -> vertex 1
        // Edge use 1 (backward): vertex 1 -> vertex 0
        LOs edgeUseToVtx({0, 1,   // edge use 0: vertices 0->1
                          1, 0}); // edge use 1: vertices 1->0

        // Empty adjacencies (no loops/faces)
        LOs edgeUseToLoopUse({});
        LOs loopUseToFace({});

        setAdjInfo(edgeToEdgeUse, edgeUseToVtx, loopUseToFace, edgeUseToLoopUse);

        // 2. Set up B-spline geometry
        // Linear B-spline (order 2) from (0,0) to (1,0)

        // One spline corresponding to edge 0
        splineToCtrlPts = LOs({0, 2});  // Spline 0 has 2 control points (indices 0-1)
        ctrlPtsX = Reals({0.0, 1.0});   // X coordinates of control points
        ctrlPtsY = Reals({0.0, 0.0});   // Y coordinates of control points

        // Clamped knot vector for linear spline (order 2)
        splineToKnots = LOs({0, 4});    // Spline 0 has 4 knots (indices 0-3)
        knotsX = Reals({0.0, 0.0, 1.0, 1.0});  // Clamped knot vector [0,0,1,1]
        knotsY = Reals({0.0, 0.0, 1.0, 1.0});  // Same for Y

        order = LOs({2});  // Linear spline has order 2

        // Mapping from model edge ID to dense spline array index
        // Edge ID 0 -> spline index 0
        edgeIdToSplineIdx = buildEdgeIdMapping(*this);

        break;
      }
      default:
        std::cerr << "Error: Unknown test model type\n";
        exit(EXIT_FAILURE);
    }
  }


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

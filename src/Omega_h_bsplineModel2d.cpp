#include "Omega_h_bsplineModel2d.hpp"
#include "Omega_h_for.hpp" //parallel_for
#include "Omega_h_array_ops.hpp" //get_max
#include "Omega_h_int_scan.hpp" //offset_scan
#include "Omega_h_atomics.hpp" //atomic_[add|increment]
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
    //ask_revClass_downAdj(...) will return a CSR/graph (offset & value arrays)
    //that may have 'a' nodes that have no connection to 'b' nodes (i.e., their
    //degree is 0).  For the use here, these entries have to be ignored/skipped.
    //The following code creates a CSR/graph without those entries.
    //////////////////////////////////////////////////////////////////////////
    auto sidesToMeshVerts_revClass = mesh->ask_revClass_downAdj(sideDim, VERT);
    const auto offsets_revClass = sidesToMeshVerts_revClass.a2ab;


    auto keep_node = Write<I8>(offsets_revClass.size()-1);

    auto numDegreeZeroEntries = Write<LO>(1, 0, "lazyWayToCount");
    Omega_h::parallel_for(offsets_revClass.size()-1, OMEGA_H_LAMBDA(Omega_h::LO side) {
      //The following will count model entities with no associated mesh
      //entities. For the current implementation of ask_revClass_downAdj, this
      //is required to avoid 'gaps' in the model entity ids.
      keep_node[side] = 1;
      if(offsets_revClass[side] == offsets_revClass[side+1]) {
        atomic_increment( &(numDegreeZeroEntries[0]) );
        keep_node[side] = 0;
      } 
    });

    const auto numMdlEnts = (offsets_revClass.size()-1)-numDegreeZeroEntries.get(0);
    OMEGA_H_CHECK(mdl->getNumEnts(sideDim) == numMdlEnts);

    //auto degree = Write<LO>(numMdlEnts);
    //auto values = Write<LO>(offsets_revClass.last());
    //auto offsets = offset_scan(read(degree));
    //auto sidesToMeshVerts = Graph(offsets, values);
    auto sidesToMeshVerts = filter_graph_nodes(sidesToMeshVerts_revClass,keep_node);
    OMEGA_H_CHECK(mdl->getNumEnts(sideDim) == sidesToMeshVerts.a2ab.size()-1);
    //////////////////////////////////////////////////////////////////////////
    
    //get the location of mesh vertices along the geometric model entity they
    //are classified on using their parametric coordinates along that model
    //entity
    const auto sidesToSamples = sidesToMeshVerts.a2ab;
    const auto meshEntIds = sidesToMeshVerts.ab2b;
    const int numSamples = sidesToSamples.last();
    const int numSides = mdl->getNumEnts(sideDim);

    auto samplePts = Omega_h::Write<Omega_h::Real>(numSamples, "splineSamplePoints");
    auto setSamplePts = OMEGA_H_LAMBDA(Omega_h::LO side) {
      //The following loop will skip model entities with no associated mesh
      //entities. For the current implementation of ask_revClass_downAdj, this
      //is required to avoid 'gaps' in the model entity ids.
      for( auto ab = sidesToSamples[side]; ab < sidesToSamples[side+1]; ab++ ) {
         const auto meshEnt = meshEntIds[ab];
         //class_parametric is a tag with two components per ent
         //only need first component for parametric coord along a spline
         const auto pos = class_parametric[meshEnt*2];
         samplePts[ab] = pos;
      }
    };
    Omega_h::parallel_for(numSides, setSamplePts, "setSamplePts");

    OMEGA_H_CHECK(sideDim==1); //no support for 1d or 3d models yet
    auto sideIds = mdl->getEdgeIds();
    const auto pts = mdl->eval(sideIds,sidesToSamples,samplePts);

    //compute warp vector for mesh vertices classified on the model boundary (sides)
    auto coords = mesh->coords();
    auto warp = Write<Real>(mesh->nverts() * 2, 0);
    auto setWarpVectors = OMEGA_H_LAMBDA(Omega_h::LO side) {
      //see note in setSamplePts lambda
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
    Write<LO> edgePolynomialOrder(edgeIds.size());
    parallel_for(edgeIds.size(), OMEGA_H_LAMBDA(LO i) {
        const auto sIdx = edgeIds[i];
        assert(sIdx <= ord.size());
        edgePolynomialOrder[i] = ord[sIdx];
    });
    auto edgePolyOrderOffset = offset_scan(Omega_h::read(edgePolynomialOrder));
    Kokkos::View<Real*> workArray("workArray", edgePolyOrderOffset.last()); //storage for intermediate values

    //TODO use team based loop or expand index range by 2x to run eval
    //on X and Y separately
    parallel_for(edgeIds.size(), OMEGA_H_LAMBDA(LO i) {
        const auto spline = edgeIds[i];
        const auto sOrder = ord[i];
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

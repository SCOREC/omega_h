#include <Omega_h_defines.hpp>
#include <PyOmega_h.hpp>

namespace Omega_h {

void pybind11_defines(py::module& m) {
  py::enum_<Omega_h_Type>(m, "Type")
      .value("I8", OMEGA_H_I8)
      .value("I32", OMEGA_H_I32)
      .value("I64", OMEGA_H_I64)
      .value("F64", OMEGA_H_F64)
      .value("REAL", OMEGA_H_REAL)
      .export_values();
  py::enum_<Omega_h_EntDim>(m, "EntDim", py::arithmetic())
      .value("VERT", OMEGA_H_VERT)
      .value("EDGE", OMEGA_H_EDGE)
      .value("FACE", OMEGA_H_FACE)
      .value("REGION", OMEGA_H_REGION)
      .export_values();
  py::enum_<Omega_h_Op>(m, "Op")
      .value("MIN", OMEGA_H_MIN)
      .value("MAX", OMEGA_H_MAX)
      .value("SUM", OMEGA_H_SUM)
      .export_values();
  py::enum_<Omega_h_Comparison>(m, "Comparison")
      .value("SAME", OMEGA_H_SAME)
      .value("MORE", OMEGA_H_MORE)
      .value("DIFF", OMEGA_H_DIFF)
      .export_values();
  py::enum_<Omega_h_Transfer>(m, "Transfer")
      .value("INHERIT", OMEGA_H_INHERIT)
      .value("LINEAR_INTERP", OMEGA_H_LINEAR_INTERP)
      .value("METRIC", OMEGA_H_METRIC)
      .value("DENSITY", OMEGA_H_DENSITY)
      .value("CONSERVE", OMEGA_H_CONSERVE)
      .value("MOMENTUM_VELOCITY", OMEGA_H_MOMENTUM_VELOCITY)
      .value("POINTWISE", OMEGA_H_POINTWISE)
      .export_values();
  py::enum_<Omega_h_Parting>(m, "Parting")
      .value("ELEM_BASED", OMEGA_H_ELEM_BASED)
      .value("GHOSTED", OMEGA_H_GHOSTED)
      .value("VERT_BASED", OMEGA_H_VERT_BASED)
      .export_values();
  py::enum_<Omega_h_Source>(
      m, "Source", "The type of source of a metric field")
      .value("CONSTANT", OMEGA_H_CONSTANT)
      .value("VARIATION", OMEGA_H_VARIATION)
      .value("DERIVATIVE", OMEGA_H_DERIVATIVE)
      .value("GIVEN", OMEGA_H_GIVEN)
      .value("IMPLIED", OMEGA_H_IMPLIED)
      .value("CURVATURE", OMEGA_H_CURVATURE)
      .export_values();
  py::enum_<Omega_h_Isotropy>(m, "Isotropy",
      "Whether and how to convert an anisotropic metric into an isotropic one")
      .value("ANISOTROPIC", OMEGA_H_ANISOTROPIC)
      .value("ISO_LENGTH", OMEGA_H_ISO_LENGTH)
      .value("ISO_SIZE", OMEGA_H_ISO_SIZE)
      .export_values();
  py::enum_<Omega_h_Scales>(m, "Scales",
      "Whether a metric source scales to satisfy element counts")
      .value("ABSOLUTE", OMEGA_H_ABSOLUTE)
      .value("SCALES", OMEGA_H_SCALES)
      .export_values();
  py::enum_<Omega_h_Family>(
      m, "Family", "Whether elements are simplices, hypercubes, or mixed")
      .value("SIMPLEX", OMEGA_H_SIMPLEX)
      .value("HYPERCUBE", OMEGA_H_HYPERCUBE)
      .value("MIXED", OMEGA_H_MIXED)
      .export_values();
  py::enum_<Topo_type>(m, "TopoType")
      .value("vertex", Topo_type::vertex)
      .value("edge", Topo_type::edge)
      .value("triangle", Topo_type::triangle)
      .value("quadrilateral", Topo_type::quadrilateral)
      .value("tetrahedron", Topo_type::tetrahedron)
      .value("hexahedron", Topo_type::hexahedron)
      .value("wedge", Topo_type::wedge)
      .value("pyramid", Topo_type::pyramid)
      .export_values();
}

}  // namespace Omega_h

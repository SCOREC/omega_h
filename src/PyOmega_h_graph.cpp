#include <Omega_h_graph.hpp>
#include <PyOmega_h.hpp>
#include <PyOmega_h_numpy_transform.hpp>


namespace Omega_h {
void pybind11_graph(py::module& m) {
  py::class_<Omega_h::Graph>(m, "OmegaHGraph")
    .def_property(
      "a2ab",
      [](const Omega_h::Graph& graph) {
        return omega_h_read_to_numpy(graph.a2ab);
      },
      [](Omega_h::Graph& graph, py::array_t<Omega_h::LO> values) {
        graph.a2ab = numpy_to_omega_h_read<Omega_h::LO>(values);
      })
    .def_property(
      "ab2b",
      [](const Omega_h::Graph& graph) {
        return omega_h_read_to_numpy(graph.ab2b);
      },
      [](Omega_h::Graph& graph, py::array_t<Omega_h::LO> values) {
        graph.ab2b = numpy_to_omega_h_read<Omega_h::LO>(values);
      })
    .def("nnodes", &Omega_h::Graph::nnodes, "Get number of graph nodes");
    }
} // namespace Omega_h
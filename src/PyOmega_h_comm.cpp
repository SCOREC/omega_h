#include <Omega_h_comm.hpp>
#include <Omega_h_library.hpp>
#include <PyOmega_h.hpp>
#include <PyOmega_h_numpy_transform.hpp>

namespace Omega_h {

void pybind11_comm(py::module& m) {
  // Bind the Comm class
#ifdef OMEGA_H_USE_MPI
  using CommHandle = std::uintptr_t;
#endif

  py::class_<Omega_h::Comm, std::shared_ptr<Omega_h::Comm>>(m, "Comm")
  // Constructors
#ifdef OMEGA_H_USE_MPI
    .def(py::init([](Omega_h::Library* library, CommHandle impl_handle) {
           MPI_Comm impl;
           std::memcpy(&impl, &impl_handle, sizeof(MPI_Comm));
           return new Omega_h::Comm(library, impl);
         }),
         py::arg("library"), py::arg("impl_handle"))

    .def(py::init([](Omega_h::Library* library, CommHandle impl_handle,
                     py::array_t<const Omega_h::I32> srcs,
                     py::array_t<const Omega_h::I32> dsts) {
           MPI_Comm impl;
           std::memcpy(&impl, &impl_handle, sizeof(MPI_Comm));
           auto srcs_view = numpy_to_omega_h_read<Omega_h::I32>(srcs);
           auto dsts_view = numpy_to_omega_h_read<Omega_h::I32>(dsts);
           return new Omega_h::Comm(library, impl, srcs_view, dsts_view);
         }),
         py::arg("library"), py::arg("impl_handle"), py::arg("srcs"),
         py::arg("dsts"))

    .def(
      "get_impl_handle",
      [](Omega_h::Comm const& self) -> CommHandle {
        MPI_Comm impl = self.get_impl();
        CommHandle handle = 0;
        std::memcpy(&handle, &impl, sizeof(MPI_Comm));
        return handle;
      },
      "Get the underlying MPI communicator as an opaque integer handle")
#else
    .def(py::init<Omega_h::Library*, bool, bool>(), py::arg("library"),
         py::arg("is_graph"), py::arg("sends_to_self"))
#endif
    // Methods
    .def("library", &Omega_h::Comm::library, py::return_value_policy::reference,
         "Get the library pointer")
    .def("rank", &Omega_h::Comm::rank, "Get the rank of this process")
    .def("size", &Omega_h::Comm::size, "Get the total number of processes")
    .def("dup", &Omega_h::Comm::dup, "Duplicate the communicator")
    .def("split", &Omega_h::Comm::split, py::arg("color"), py::arg("key"),
         "Split the communicator")
    .def("graph", &Omega_h::Comm::graph, py::arg("dsts"),
         "Create a graph communicator")
    .def("graph_adjacent", &Omega_h::Comm::graph_adjacent, py::arg("srcs"),
         py::arg("dsts"), "Create an adjacent graph communicator")
    .def("graph_inverse", &Omega_h::Comm::graph_inverse,
         "Get the inverse graph communicator")
    .def("sources", &Omega_h::Comm::sources, "Get source ranks")
    .def("destinations", &Omega_h::Comm::destinations, "Get destination ranks")
    .def("reduce_or", &Omega_h::Comm::reduce_or, py::arg("x"),
         "Reduce using logical OR")
    .def("reduce_and", &Omega_h::Comm::reduce_and, py::arg("x"),
         "Reduce using logical AND")
    .def("add_int128", &Omega_h::Comm::add_int128, py::arg("x"),
         "Add Int128 values across processes")
    .def("bcast_string", &Omega_h::Comm::bcast_string, py::arg("s"),
         py::arg("root_rank") = 0, "Broadcast a string")
    .def("barrier", &Omega_h::Comm::barrier, "Synchronize all processes");
}

}  // namespace Omega_h

#include <Omega_h_library.hpp>
#include <PyOmega_h.hpp>
#include <iostream>

namespace Omega_h {

/* The lifetime of the Library object is quite important (it must contain the
   lifetime of pretty much all other Omega_h objects), and
   I'm unsure about the order in which destructors will be called.
   So, for now, I'll take the approach that the Python interface will have
   the Library as a hidden global variable.
   This is consistent with how mpi4py seems to work.
   I tried using the Python atexit mechanism, but that seems to execute prior
   to final garbage collection.
   
   Note: We use a raw pointer and intentionally leak it to avoid CUDA/Kokkos
   finalization issues during Python shutdown. The OS will clean up the memory.
 */

Library* pybind11_global_library = nullptr;

void pybind11_library(py::module& module) {
  // Bind Omega_h::Library
  py::class_<Omega_h::Library>(
    module, "OmegaHLibrary")
    .def(py::init<>(), "Default constructor")
    .def("world", &Omega_h::Library::world, "Get the world communicator");
}

}  // namespace Omega_h

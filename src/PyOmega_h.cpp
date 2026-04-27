#include <PyOmega_h.hpp>

PYBIND11_MODULE(PyOmega_h, m) {
  m.doc() = "Omega_h: simplex mesh adaptation";
  Omega_h::pybind11_defines(m);
  Omega_h::pybind11_array(m);
  Omega_h::pybind11_comm(m);
  Omega_h::pybind11_library(m);
  Omega_h::pybind11_tag(m);
  Omega_h::pybind11_graph(m);
  Omega_h::pybind11_mesh(m);
  Omega_h::pybind11_build(m);
  Omega_h::pybind11_adapt(m);
  Omega_h::pybind11_file(m);
  Omega_h::pybind11_class(m);
#ifdef OMEGA_H_USE_DOLFIN
  Omega_h::pybind11_dolfin(m);
#endif
}

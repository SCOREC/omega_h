#include <PyOmega_h_numpy_transform.hpp>
#include <Omega_h_tag.hpp>
#include <PyOmega_h.hpp>

namespace Omega_h {
void pybind11_tag(py::module& module) {
   // Bind ArrayType enum for tag array types
  py::enum_<Omega_h::ArrayType>(module, "ArrayType")
    .value("VectorND", Omega_h::ArrayType::VectorND)
    .value("SymmetricSquareMatrix", Omega_h::ArrayType::SymmetricSquareMatrix)
    .export_values();

  // Bind TagBase class for tag metadata access
  py::class_<Omega_h::TagBase>(module, "TagBase")
    .def("name", &Omega_h::TagBase::name, "Get tag name")
    .def("ncomps", &Omega_h::TagBase::ncomps, "Get number of components")
    .def("type", &Omega_h::TagBase::type, "Get tag data type (Omega_h_Type)")
    .def("array_type", &Omega_h::TagBase::array_type,
         "Get array type (VectorND or SymmetricSquareMatrix)")
    .def(
      "class_ids",
      [](const Omega_h::TagBase& tag) {
        auto class_ids = tag.class_ids();
        return omega_h_read_to_numpy(class_ids);
      },
      "Get class IDs for rcField tags");
}
} // namespace Omega_h
#ifndef OMEGA_H_PY_NUMPY_TRANSFORM_HPP
#define OMEGA_H_PY_NUMPY_TRANSFORM_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <Omega_h_array.hpp>

namespace py = pybind11;

namespace Omega_h {
// Helper to convert 1D numpy array to Omega_h::Read
template <typename T>
Omega_h::Read<T> numpy_to_omega_h_read(py::array_t<T> arr)
{
  py::buffer_info buf = arr.request();
  if (buf.ndim != 1) {
    throw std::runtime_error("Number of dimensions must be 1");
  }
  Kokkos::View<T*, Kokkos::DefaultExecutionSpace::memory_space,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
    view(reinterpret_cast<T*>(buf.ptr), buf.shape[0]);
  Omega_h::Write<T> write_view(view);
  Omega_h::Read<T> read_view(write_view);
  return read_view;
}

// Helper to convert Omega_h::Read to numpy array (creates a copy)
template <typename T>
py::array_t<T> omega_h_read_to_numpy(Omega_h::Read<T> read_view)
{
  auto read_view_host = Omega_h::HostRead<T>(read_view);
  py::array_t<T> result(read_view_host.size());
  py::buffer_info buf = result.request();
  T* ptr = static_cast<T*>(buf.ptr);
  for (Omega_h::LO i = 0; i < read_view_host.size(); ++i) {
    ptr[i] = read_view_host[i];
  }
  return result;
}

// Helper to convert 1D numpy array to Omega_h::Write
template <typename T>
Omega_h::Write<T> numpy_to_omega_h_write(py::array_t<T> arr)
{
  py::buffer_info buf = arr.request();
  if (buf.ndim != 1) {
    throw std::runtime_error("Number of dimensions must be 1");
  }
  // Get host mirror and copy data
  auto write_view_host = Omega_h::HostWrite<T>(buf.shape[0]);
  T* ptr = static_cast<T*>(buf.ptr);
  for (Omega_h::LO i = 0; i < buf.shape[0]; ++i) {
    write_view_host[i] = ptr[i];
  }
  auto write_view = Omega_h::Write<T>(write_view_host);
  return write_view;
}

// Helper to convert Omega_h::Write to numpy array (creates a reference)
template <typename T>
py::array_t<T> omega_h_write_to_numpy(Omega_h::Write<T> write_view)
{
  return py::array_t<T>({static_cast<py::ssize_t>(write_view.size())}, // shape
                        {sizeof(T)},         // strides
                        write_view.data(),   // data pointer
                        py::cast(write_view) // base object to manage lifetime
  );
}
} // namespace Omega_h

#endif
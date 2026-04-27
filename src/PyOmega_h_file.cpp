#include <Omega_h_file.hpp>
#include <Omega_h_filesystem.hpp>
#include <PyOmega_h.hpp>

#ifdef OMEGA_H_USE_ADIOS2
#include <Omega_h_adios2.hpp>
#endif

namespace Omega_h {

void pybind11_file(py::module& m) {
  py::class_<Omega_h::filesystem::path>(m, "path")
      .def(py::init<char const*>());
  m.def(
    "read_mesh_file",
    [](const std::string& filepath, std::shared_ptr<Omega_h::Comm> comm) {
      return Omega_h::read_mesh_file(filepath, comm);
    },
    py::arg("filepath"), py::arg("comm"),
    "Read mesh from file (auto-detects format)", py::return_value_policy::move);

  // Binary format I/O
  m.def(
    "read_mesh_binary",
    [](const std::string& filepath, Omega_h::Library* lib) {
      Omega_h::Mesh mesh(lib);
      Omega_h::binary::read(filepath, lib->world(), &mesh);
      return mesh;
    },
    py::arg("filepath"), py::arg("library"), "Read mesh from binary file",
    py::return_value_policy::move);

  m.def(
    "read_mesh_binary",
    [](const std::string& filepath, std::shared_ptr<Omega_h::Comm> comm) {
      return Omega_h::binary::read(filepath, comm);
    },
    py::arg("filepath"), py::arg("comm"), "Read mesh from binary file",
    py::return_value_policy::move);

  m.def(
    "write_mesh_binary",
    [](const std::string& filepath, Omega_h::Mesh& mesh) {
      Omega_h::binary::write(filepath, &mesh);
    },
    py::arg("filepath"), py::arg("mesh"), "Write mesh to binary file");

  // Gmsh format I/O
  m.def(
    "read_mesh_gmsh",
    [](const std::string& filepath, std::shared_ptr<Omega_h::Comm> comm) {
      return Omega_h::gmsh::read(filepath, comm);
    },
    py::arg("filepath"), py::arg("comm"), "Read mesh from Gmsh file",
    py::return_value_policy::move);

  m.def(
    "write_mesh_gmsh",
    [](const std::string& filepath, Omega_h::Mesh& mesh) {
      Omega_h::gmsh::write(filepath, &mesh);
    },
    py::arg("filepath"), py::arg("mesh"), "Write mesh to Gmsh file");

#ifdef OMEGA_H_USE_GMSH
  m.def(
    "read_mesh_gmsh_parallel",
    [](const std::string& filepath, std::shared_ptr<Omega_h::Comm> comm) {
      return Omega_h::gmsh::read_parallel(filepath, comm);
    },
    py::arg("filepath"), py::arg("comm"), "Read parallel Gmsh mesh (MSH 4.1+)",
    py::return_value_policy::move);

  m.def("write_mesh_gmsh_parallel", &Omega_h::gmsh::write_parallel,
    py::arg("filepath"), py::arg("mesh"), "Write parallel Gmsh mesh (MSH 4.1)");
#endif

  // VTK format I/O
  m.def(
    "write_mesh_vtu",
    [](const std::string& filepath, Omega_h::Mesh& mesh, bool compress) {
      Omega_h::vtk::write_vtu(filepath, &mesh, compress);
    },
    py::arg("filepath"), py::arg("mesh"),
    py::arg("compress") = OMEGA_H_DEFAULT_COMPRESS, "Write mesh to VTU file");

  m.def(
    "write_mesh_vtu",
    [](const std::string& filepath, Omega_h::Mesh& mesh, Omega_h::Int cell_dim,
       bool compress) {
      Omega_h::vtk::write_vtu(filepath, &mesh, cell_dim, compress);
    },
    py::arg("filepath"), py::arg("mesh"), py::arg("cell_dim"),
    py::arg("compress") = OMEGA_H_DEFAULT_COMPRESS,
    "Write mesh to VTU file with specified cell dimension");

  m.def(
    "write_mesh_parallel_vtk",
    [](const std::string& filepath, Omega_h::Mesh& mesh, bool compress) {
      Omega_h::vtk::write_parallel(filepath, &mesh, compress);
    },
    py::arg("filepath"), py::arg("mesh"),
    py::arg("compress") = OMEGA_H_DEFAULT_COMPRESS,
    "Write mesh to parallel VTK files");

  m.def(
    "read_mesh_parallel_vtk",
    [](const std::string& pvtupath, std::shared_ptr<Omega_h::Comm> comm) {
      Omega_h::Mesh mesh;
      Omega_h::vtk::read_parallel(pvtupath, comm, &mesh);
      return mesh;
    },
    py::arg("pvtupath"), py::arg("comm"), "Read parallel VTK mesh",
    py::return_value_policy::move);

#ifdef OMEGA_H_USE_SEACASEXODUS
  // Exodus ClassifyWith enum
  py::enum_<Omega_h::exodus::ClassifyWith>(m, "ExodusClassifyWith")
    .value("NODE_SETS", Omega_h::exodus::NODE_SETS)
    .value("SIDE_SETS", Omega_h::exodus::SIDE_SETS)
    .export_values();

  // Exodus file operations
  m.def(
    "exodus_open",
    [](const std::string& filepath, bool verbose) {
      return Omega_h::exodus::open(filepath, verbose);
    },
    py::arg("filepath"), py::arg("verbose") = false,
    "Open an Exodus file and return file handle");

  m.def(
    "exodus_close",
    [](int exodus_file) { Omega_h::exodus::close(exodus_file); },
    py::arg("exodus_file"), "Close an Exodus file");

  m.def(
    "exodus_get_num_time_steps",
    [](int exodus_file) {
      return Omega_h::exodus::get_num_time_steps(exodus_file);
    },
    py::arg("exodus_file"), "Get the number of time steps in an Exodus file");

  m.def(
    "read_mesh_exodus",
    [](int exodus_file, Omega_h::Mesh& mesh, bool verbose, int classify_with) {
      Omega_h::exodus::read_mesh(exodus_file, &mesh, verbose, classify_with);
    },
    py::arg("exodus_file"), py::arg("mesh"), py::arg("verbose") = false,
    py::arg("classify_with") =
      Omega_h::exodus::NODE_SETS | Omega_h::exodus::SIDE_SETS,
    "Read mesh from an open Exodus file");

  m.def(
    "read_exodus_nodal_fields",
    [](int exodus_file, Omega_h::Mesh& mesh, int time_step,
       const std::string& prefix, const std::string& postfix, bool verbose) {
      Omega_h::exodus::read_nodal_fields(exodus_file, &mesh, time_step, prefix,
                                         postfix, verbose);
    },
    py::arg("exodus_file"), py::arg("mesh"), py::arg("time_step"),
    py::arg("prefix") = "", py::arg("postfix") = "", py::arg("verbose") = false,
    "Read nodal fields from an open Exodus file at a specific time step");

  m.def(
    "read_exodus_element_fields",
    [](int exodus_file, Omega_h::Mesh& mesh, int time_step,
       const std::string& prefix, const std::string& postfix, bool verbose) {
      Omega_h::exodus::read_element_fields(exodus_file, &mesh, time_step,
                                           prefix, postfix, verbose);
    },
    py::arg("exodus_file"), py::arg("mesh"), py::arg("time_step"),
    py::arg("prefix") = "", py::arg("postfix") = "", py::arg("verbose") = false,
    "Read element fields from an open Exodus file at a specific time step");

  m.def(
    "read_mesh_exodus_sliced",
    [](const std::string& filepath, std::shared_ptr<Omega_h::Comm> comm,
       bool verbose, int classify_with, int time_step) {
      return Omega_h::exodus::read_sliced(filepath, comm, verbose,
                                          classify_with, time_step);
    },
    py::arg("filepath"), py::arg("comm"), py::arg("verbose") = false,
    py::arg("classify_with") =
      Omega_h::exodus::NODE_SETS | Omega_h::exodus::SIDE_SETS,
    py::arg("time_step") = -1, "Read sliced Exodus mesh in parallel",
    py::return_value_policy::move);

  m.def(
    "write_mesh_exodus",
    [](const std::string& filepath, Omega_h::Mesh& mesh, bool verbose,
       int classify_with) {
      Omega_h::exodus::write(filepath, &mesh, verbose, classify_with);
    },
    py::arg("filepath"), py::arg("mesh"), py::arg("verbose") = false,
    py::arg("classify_with") =
      Omega_h::exodus::NODE_SETS | Omega_h::exodus::SIDE_SETS,
    "Write mesh to Exodus file");
#endif

#ifdef OMEGA_H_USE_LIBMESHB
  // MESHB format I/O
  m.def(
    "read_mesh_meshb",
    [](Omega_h::Mesh& mesh, const std::string& filepath) {
      Omega_h::meshb::read(&mesh, filepath);
    },
    py::arg("mesh"), py::arg("filepath"), "Read mesh from MESHB file");

  m.def(
    "write_mesh_meshb",
    [](Omega_h::Mesh& mesh, const std::string& filepath, int version) {
      Omega_h::meshb::write(&mesh, filepath, version);
    },
    py::arg("mesh"), py::arg("filepath"), py::arg("version") = 2,
    "Write mesh to MESHB file");

  m.def(
    "read_meshb_sol",
    [](Omega_h::Mesh& mesh, const std::string& filepath,
       const std::string& sol_name) {
      Omega_h::meshb::read_sol(&mesh, filepath, sol_name);
    },
    py::arg("mesh"), py::arg("filepath"), py::arg("sol_name"),
    "Read solution/resolution data from MESHB .sol file");

  m.def(
    "write_meshb_sol",
    [](Omega_h::Mesh& mesh, const std::string& filepath,
       const std::string& sol_name, int version) {
      Omega_h::meshb::write_sol(&mesh, filepath, sol_name, version);
    },
    py::arg("mesh"), py::arg("filepath"), py::arg("sol_name"),
    py::arg("version") = 2,
    "Write solution/resolution data to MESHB .sol file");
#endif

#ifdef OMEGA_H_USE_ADIOS2
  // ADIOS2 format I/O
  m.def(
    "read_mesh_adios2",
    [](const std::string& filepath, Omega_h::Library* lib,
       const std::string& prefix) {
      return Omega_h::adios::read(filepath, lib, prefix);
    },
    py::arg("filepath"), py::arg("library"), py::arg("prefix") = "",
    "Read mesh from ADIOS2 file", py::return_value_policy::move);

  m.def(
    "write_mesh_adios2",
    [](const std::string& filepath, Omega_h::Mesh& mesh,
       const std::string& prefix) {
      Omega_h::adios::write(filepath, &mesh, prefix);
    },
    py::arg("filepath"), py::arg("mesh"), py::arg("prefix") = "",
    "Write mesh to ADIOS2 file");
#endif
}

}  // namespace Omega_h

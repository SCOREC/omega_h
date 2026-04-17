#include <Omega_h_mesh.hpp>
#include <PyOmega_h.hpp>
#include <pybind11/pybind11.h>
#include <PyOmega_h_numpy_transform.hpp>

namespace Omega_h {

#define OMEGA_H_DECL_TYPE(T, name)                                             \
  void (Mesh::*add_tag_##name)(Int, std::string const&, Int, Read<T>, bool) =  \
      &Mesh::add_tag<T>;
#define OMEGA_H_DEF_TYPE(T, name)                                              \
  .def("get_array_" #name, &Mesh::get_array<T>)                                \
      .def("add_tag_" #name, add_tag_##name,                                   \
          "Add " #name " tag array to the mesh",                               \
          py::arg("ent_dim") = OMEGA_H_VERT, py::arg("name"),                  \
          py::arg("ncomps") = 1, py::arg("array"),                             \
          py::arg("internal_do_not_use_ever") = false)


void pybind11_mesh(py::module& module) {
  void (Mesh::*set_parting)(Omega_h_Parting, Int, bool) = &Mesh::set_parting;
  void (Mesh::*balance)(bool) = &Mesh::balance;
  py::class_<Omega_h::Mesh, std::shared_ptr<Omega_h::Mesh>>(module, "OmegaHMesh")
    .def(py::init<>(), "Default constructor")
    .def(py::init<Omega_h::Library*>(), py::arg("library"),
         "Constructor with library")

    .def("set_library", &Omega_h::Mesh::set_library, py::arg("library"),
         "Set the library")

    .def("library", &Omega_h::Mesh::library, py::return_value_policy::reference,
         "Get the library")

    .def("set_comm", &Omega_h::Mesh::set_comm, py::arg("comm"),
         "Set the communicator")

    .def("comm", &Omega_h::Mesh::comm, "Get the communicator")

    .def("set_dim", &Omega_h::Mesh::set_dim, py::arg("dim"),
         "Set mesh dimension")

    .def("dim", &Omega_h::Mesh::dim, "Get mesh dimension")

    .def("set_family", &Omega_h::Mesh::set_family, py::arg("family"),
         "Set mesh family (simplex, hypercube, etc.)")

    .def("family", &Omega_h::Mesh::family, "Get mesh family")

    .def("nverts", &Omega_h::Mesh::nverts, "Get number of vertices")

    .def("nedges", &Omega_h::Mesh::nedges, "Get number of edges")

    .def("nfaces", &Omega_h::Mesh::nfaces, "Get number of faces")

    .def("nregions", &Omega_h::Mesh::nregions, "Get number of regions")

    .def("nelems", &Omega_h::Mesh::nelems, "Get number of elements")

    .def("nents", &Omega_h::Mesh::nents, py::arg("ent_dim"),
         "Get number of entities of given dimension")

    .def("nglobal_ents", &Omega_h::Mesh::nglobal_ents, py::arg("ent_dim"),
         "Get global number of entities")

    .def(
      "coords",
      [](const Omega_h::Mesh& mesh) {
        auto coords = mesh.coords();
        return omega_h_read_to_numpy(coords);
      },
      "Get mesh coordinates as numpy array")

    .def(
      "set_coords",
      [](Omega_h::Mesh& mesh, py::array_t<Omega_h::Real> coords) {
        auto coords_write = numpy_to_omega_h_write<Omega_h::Real>(coords);
        mesh.set_coords(Omega_h::Reals(coords_write));
      },
      py::arg("coords"), "Set mesh coordinates from numpy array")

    .def(
      "add_coords",
      [](Omega_h::Mesh& mesh, py::array_t<Omega_h::Real> coords) {
        auto coords_write = numpy_to_omega_h_write<Omega_h::Real>(coords);
        mesh.add_coords(Omega_h::Reals(coords_write));
      },
      py::arg("coords"), "Add mesh coordinates from numpy array")

    .def("has_tag", &Omega_h::Mesh::has_tag, py::arg("ent_dim"),
         py::arg("name"), "Check if mesh has a tag")

    .def("ntags", &Omega_h::Mesh::ntags, py::arg("ent_dim"),
         "Get number of tags for entity dimension")

    .def("nrctags", &Omega_h::Mesh::nrctags, py::arg("ent_dim"),
         "Get number of rcField tags for entity dimension")
    
    .def("get_tag",
        static_cast<const Omega_h::TagBase* (Omega_h::Mesh::*)(Omega_h::Int, Omega_h::Int) const>(
          &Omega_h::Mesh::get_tag),
        py::arg("ent_dim"), py::arg("index"),
        py::return_value_policy::reference)

    .def("get_tagbase",
        static_cast<const Omega_h::TagBase* (Omega_h::Mesh::*)(Omega_h::Int, const std::string&) const>(
          &Omega_h::Mesh::get_tagbase),
        py::arg("ent_dim"), py::arg("name"),
        py::return_value_policy::reference)

    .def("remove_tag", &Omega_h::Mesh::remove_tag, py::arg("ent_dim"),
         py::arg("name"), "Remove a tag")
    

     // Tag creation with dtype string
    .def(
      "add_tag",
      [](Omega_h::Mesh& mesh, Omega_h::Int dim, const std::string& name,
         Omega_h::Int ncomps, const std::string& dtype,
         Omega_h::ArrayType array_type) {
        if (dtype == "int8" || dtype == "i1") {
          mesh.add_tag<Omega_h::I8>(dim, name, ncomps, array_type);
        } else if (dtype == "int32" || dtype == "i4") {
          mesh.add_tag<Omega_h::I32>(dim, name, ncomps, array_type);
        } else if (dtype == "int64" || dtype == "i8") {
          mesh.add_tag<Omega_h::I64>(dim, name, ncomps, array_type);
        } else if (dtype == "float64" || dtype == "f8" || dtype == "double") {
          mesh.add_tag<Omega_h::Real>(dim, name, ncomps, array_type);
        } else {
          throw std::runtime_error(
            "Unsupported dtype: " + dtype +
            ". Use 'int8', 'int32', 'int64', or 'float64'");
        }
      },
      py::arg("ent_dim"), py::arg("name"), py::arg("ncomps"),
      py::arg("dtype") = "float64",
      py::arg("array_type") = Omega_h::ArrayType::VectorND,
      "Add a tag with specified dtype (int8, int32, int64, float64)")

    // Add tag with numpy array (auto-detect type)
    .def(
      "add_tag",
      [](Omega_h::Mesh& mesh, Omega_h::Int dim, const std::string& name,
         Omega_h::Int ncomps, py::array array, bool internal,
         Omega_h::ArrayType array_type) {
        auto dtype = array.dtype();
        if (dtype.is(py::dtype::of<std::int8_t>())) {
          auto array_view = numpy_to_omega_h_read<Omega_h::I8>(
            array.cast<py::array_t<std::int8_t>>());
          mesh.add_tag<Omega_h::I8>(dim, name, ncomps, array_view, internal,
                                    array_type);
        } else if (dtype.is(py::dtype::of<std::int32_t>())) {
          auto array_view = numpy_to_omega_h_read<Omega_h::I32>(
            array.cast<py::array_t<std::int32_t>>());
          mesh.add_tag<Omega_h::I32>(dim, name, ncomps, array_view, internal,
                                     array_type);
        } else if (dtype.is(py::dtype::of<std::int64_t>())) {
          auto array_view = numpy_to_omega_h_read<Omega_h::I64>(
            array.cast<py::array_t<std::int64_t>>());
          mesh.add_tag<Omega_h::I64>(dim, name, ncomps, array_view, internal,
                                     array_type);
        } else if (dtype.is(py::dtype::of<double>())) {
          auto array_view = numpy_to_omega_h_read<Omega_h::Real>(
            array.cast<py::array_t<double>>());
          mesh.add_tag<Omega_h::Real>(dim, name, ncomps, array_view, internal,
                                      array_type);
        } else {
          throw std::runtime_error(
            "Unsupported numpy dtype. Use int8, int32, int64, or float64");
        }
      },
      py::arg("ent_dim"), py::arg("name"), py::arg("ncomps"), py::arg("array"),
      py::arg("internal") = false,
      py::arg("array_type") = Omega_h::ArrayType::VectorND,
      "Add a tag with initial values from numpy array (auto-detects type)")

    // Get tag data (auto-detect type from stored tag)
    .def(
      "get_tag",
      [](Omega_h::Mesh& mesh, Omega_h::Int dim,
         const std::string& name) -> py::array {
        auto tagbase = mesh.get_tagbase(dim, name);
        auto type = tagbase->type();

        if (type == OMEGA_H_I8) {
          auto array = mesh.get_array<Omega_h::I8>(dim, name);
          return omega_h_read_to_numpy(array);
        } else if (type == OMEGA_H_I32) {
          auto array = mesh.get_array<Omega_h::I32>(dim, name);
          return omega_h_read_to_numpy(array);
        } else if (type == OMEGA_H_I64) {
          auto array = mesh.get_array<Omega_h::I64>(dim, name);
          return omega_h_read_to_numpy(array);
        } else if (type == OMEGA_H_F64) {
          auto array = mesh.get_array<Omega_h::Real>(dim, name);
          return omega_h_read_to_numpy(array);
        } else {
          throw std::runtime_error("Unsupported tag type");
        }
      },
      py::arg("ent_dim"), py::arg("name"),
      "Get tag data as numpy array by name (auto-detects type)")

    .def(
      "get_tag_array",
      [](Omega_h::Mesh& mesh, Omega_h::Int dim,
         const std::string& name) -> py::array {
        auto tagbase = mesh.get_tagbase(dim, name);
        auto type = tagbase->type();

        if (type == OMEGA_H_I8) {
          auto array = mesh.get_array<Omega_h::I8>(dim, name);
          return omega_h_read_to_numpy(array);
        } else if (type == OMEGA_H_I32) {
          auto array = mesh.get_array<Omega_h::I32>(dim, name);
          return omega_h_read_to_numpy(array);
        } else if (type == OMEGA_H_I64) {
          auto array = mesh.get_array<Omega_h::I64>(dim, name);
          return omega_h_read_to_numpy(array);
        } else if (type == OMEGA_H_F64) {
          auto array = mesh.get_array<Omega_h::Real>(dim, name);
          return omega_h_read_to_numpy(array);
        } else {
          throw std::runtime_error("Unsupported tag type");
        }
      },
      py::arg("ent_dim"), py::arg("name"),
      "Get tag data as numpy array by name (alias for clarity)")

    // Typed getters that return Read<T> arrays directly (for C++ API use)
    .def("get_array_int8",
      [](Omega_h::Mesh& mesh, Omega_h::Int dim, const std::string& name) {
        return mesh.get_array<Omega_h::I8>(dim, name);
      },
      py::arg("ent_dim"), py::arg("name"),
      "Get int8 tag data as Read_int8 array")
    .def("get_array_int32",
      [](Omega_h::Mesh& mesh, Omega_h::Int dim, const std::string& name) {
        return mesh.get_array<Omega_h::I32>(dim, name);
      },
      py::arg("ent_dim"), py::arg("name"),
      "Get int32 tag data as Read_int32 array")
    .def("get_array_int64",
      [](Omega_h::Mesh& mesh, Omega_h::Int dim, const std::string& name) {
        return mesh.get_array<Omega_h::I64>(dim, name);
      },
      py::arg("ent_dim"), py::arg("name"),
      "Get int64 tag data as Read_int64 array")
    .def("get_array_float64",
      [](Omega_h::Mesh& mesh, Omega_h::Int dim, const std::string& name) {
        return mesh.get_array<Omega_h::Real>(dim, name);
      },
      py::arg("ent_dim"), py::arg("name"),
      "Get float64 tag data as Read_float64 array")

    // Set tag data from numpy array (auto-detect type)
    .def(
      "set_tag",
      [](Omega_h::Mesh& mesh, Omega_h::Int dim, const std::string& name,
         py::array array, bool internal, Omega_h::ArrayType array_type) {
        auto dtype = array.dtype();
        if (dtype.is(py::dtype::of<std::int8_t>())) {
          auto array_view = numpy_to_omega_h_read<Omega_h::I8>(
            array.cast<py::array_t<std::int8_t>>());
          mesh.set_tag<Omega_h::I8>(dim, name, array_view, internal,
                                    array_type);
        } else if (dtype.is(py::dtype::of<std::int32_t>())) {
          auto array_view = numpy_to_omega_h_read<Omega_h::I32>(
            array.cast<py::array_t<std::int32_t>>());
          mesh.set_tag<Omega_h::I32>(dim, name, array_view, internal,
                                     array_type);
        } else if (dtype.is(py::dtype::of<std::int64_t>())) {
          auto array_view = numpy_to_omega_h_read<Omega_h::I64>(
            array.cast<py::array_t<std::int64_t>>());
          mesh.set_tag<Omega_h::I64>(dim, name, array_view, internal,
                                     array_type);
        } else if (dtype.is(py::dtype::of<double>())) {
          auto array_view = numpy_to_omega_h_read<Omega_h::Real>(
            array.cast<py::array_t<double>>());
          mesh.set_tag<Omega_h::Real>(dim, name, array_view, internal,
                                      array_type);
        } else {
          throw std::runtime_error(
            "Unsupported numpy dtype. Use int8, int32, int64, or float64");
        }
      },
      py::arg("ent_dim"), py::arg("name"), py::arg("array"),
      py::arg("internal") = false,
      py::arg("array_type") = Omega_h::ArrayType::VectorND,
      "Set tag data from numpy array (auto-detects type)")

    // Overloads for direct Read<T> inputs for set_tag
    .def("set_tag",
    [](Omega_h::Mesh& mesh, Omega_h::Int dim, const std::string& name,
        Omega_h::Read<Omega_h::I8> array, bool internal,
        Omega_h::ArrayType array_type) {
      mesh.set_tag<Omega_h::I8>(dim, name, array, internal, array_type);
    },
    py::arg("ent_dim"), py::arg("name"), py::arg("array"),
    py::arg("internal") = false,
    py::arg("array_type") = Omega_h::ArrayType::VectorND,
    "Set tag data with int8 Read array")
    .def("set_tag",
    [](Omega_h::Mesh& mesh, Omega_h::Int dim, const std::string& name,
        Omega_h::Read<Omega_h::I32> array, bool internal,
        Omega_h::ArrayType array_type) {
      mesh.set_tag<Omega_h::I32>(dim, name, array, internal, array_type);
    },
    py::arg("ent_dim"), py::arg("name"), py::arg("array"),
    py::arg("internal") = false,
    py::arg("array_type") = Omega_h::ArrayType::VectorND,
    "Set tag data with int32 Read array")
    .def("set_tag",
    [](Omega_h::Mesh& mesh, Omega_h::Int dim, const std::string& name,
        Omega_h::Read<Omega_h::I64> array, bool internal,
        Omega_h::ArrayType array_type) {
      mesh.set_tag<Omega_h::I64>(dim, name, array, internal, array_type);
    },
    py::arg("ent_dim"), py::arg("name"), py::arg("array"),
    py::arg("internal") = false,
    py::arg("array_type") = Omega_h::ArrayType::VectorND,
    "Set tag data with int64 Read array")
    .def("set_tag",
    [](Omega_h::Mesh& mesh, Omega_h::Int dim, const std::string& name,
        Omega_h::Read<Omega_h::Real> array, bool internal,
        Omega_h::ArrayType array_type) {
      mesh.set_tag<Omega_h::Real>(dim, name, array, internal, array_type);
    },
    py::arg("ent_dim"), py::arg("name"), py::arg("array"),
    py::arg("internal") = false,
    py::arg("array_type") = Omega_h::ArrayType::VectorND,
    "Set tag data with Real Read array")

    // Synchronize and reduce tags in parallel
    .def("sync_tag", &Omega_h::Mesh::sync_tag, py::arg("ent_dim"),
         py::arg("name"), "Synchronize tag values across processors")

    .def("reduce_tag", &Omega_h::Mesh::reduce_tag, py::arg("ent_dim"),
         py::arg("name"), py::arg("op"),
         "Reduce tag values across processors with specified operation")

    .def("has_ents", &Omega_h::Mesh::has_ents, py::arg("ent_dim"),
         "Check if mesh has entities of given dimension")

    .def(
      "ask_lengths",
      [](Omega_h::Mesh& mesh) {
        auto lengths = mesh.ask_lengths();
        return omega_h_read_to_numpy(lengths);
      },
      "Get edge lengths")

    .def(
      "ask_qualities",
      [](Omega_h::Mesh& mesh) {
        auto qualities = mesh.ask_qualities();
        return omega_h_read_to_numpy(qualities);
      },
      "Get element qualities")

    .def("set_parting", set_parting, py::arg("parting"), py::arg("nlayers"),
        py::arg("verbose") = false)
    .def("min_quality", &Omega_h::Mesh::min_quality)
    .def("max_length", &Omega_h::Mesh::max_length)
    .def("balance", balance, py::arg("predictive") = false)
    // Overloads for direct Read<T> inputs
    .def("add_tag",
    [](Omega_h::Mesh& mesh, Omega_h::Int dim, const std::string& name,
        Omega_h::Int ncomps, Omega_h::Read<Omega_h::I8> array, bool internal,
        Omega_h::ArrayType array_type) {
      mesh.add_tag<Omega_h::I8>(dim, name, ncomps, array, internal, array_type);
    },
    py::arg("ent_dim"), py::arg("name"), py::arg("ncomps"), py::arg("array"),
    py::arg("internal") = false,
    py::arg("array_type") = Omega_h::ArrayType::VectorND,
    "Add a tag with int8 Read array")
    .def("add_tag",
    [](Omega_h::Mesh& mesh, Omega_h::Int dim, const std::string& name,
        Omega_h::Int ncomps, Omega_h::Read<Omega_h::I32> array, bool internal,
        Omega_h::ArrayType array_type) {
      mesh.add_tag<Omega_h::I32>(dim, name, ncomps, array, internal, array_type);
    },
    py::arg("ent_dim"), py::arg("name"), py::arg("ncomps"), py::arg("array"),
    py::arg("internal") = false,
    py::arg("array_type") = Omega_h::ArrayType::VectorND,
    "Add a tag with int32 Read array")
    .def("add_tag",
    [](Omega_h::Mesh& mesh, Omega_h::Int dim, const std::string& name,
        Omega_h::Int ncomps, Omega_h::Read<Omega_h::I64> array, bool internal,
        Omega_h::ArrayType array_type) {
      mesh.add_tag<Omega_h::I64>(dim, name, ncomps, array, internal, array_type);
    },
    py::arg("ent_dim"), py::arg("name"), py::arg("ncomps"), py::arg("array"),
    py::arg("internal") = false,
    py::arg("array_type") = Omega_h::ArrayType::VectorND,
    "Add a tag with int64 Read array")
    .def("add_tag",
    [](Omega_h::Mesh& mesh, Omega_h::Int dim, const std::string& name,
        Omega_h::Int ncomps, Omega_h::Read<Omega_h::Real> array, bool internal,
        Omega_h::ArrayType array_type) {
      mesh.add_tag<Omega_h::Real>(dim, name, ncomps, array, internal, array_type);
    },
    py::arg("ent_dim"), py::arg("name"), py::arg("ncomps"), py::arg("array"),
    py::arg("internal") = false,
    py::arg("array_type") = Omega_h::ArrayType::VectorND,
    "Add a tag with Real Read array")
    .def("parting", &Omega_h::Mesh::parting, "Get mesh partitioning type")

    .def("nghost_layers", &Omega_h::Mesh::nghost_layers,
         "Get number of ghost layers")

    .def(
      "owned",
      [](Omega_h::Mesh& mesh, Omega_h::Int ent_dim) {
        auto owned = mesh.owned(ent_dim);
        return omega_h_read_to_numpy(owned);
      },
      py::arg("ent_dim"), "Get ownership flags for entities")

    // Adjacency queries
    .def(
      "ask_verts_of",
      [](Omega_h::Mesh& mesh, Omega_h::Int dim) {
        auto verts = mesh.ask_verts_of(dim);
        return omega_h_read_to_numpy(verts);
      },
      py::arg("ent_dim"), "Get vertex IDs for entities of given dimension")

    .def(
      "ask_elem_verts",
      [](Omega_h::Mesh& mesh) {
        auto elem_verts = mesh.ask_elem_verts();
        return omega_h_read_to_numpy(elem_verts);
      },
      "Get vertex IDs for elements")

    .def("get_vtx_patches", &Omega_h::Mesh::get_vtx_patches,
      py::arg("min_patch_size"), py::arg("tgt_dim"),
      py::return_value_policy::move, "Get vertex-centered patch graph")

    .def("has_adj", &Omega_h::Mesh::has_adj, py::arg("from_dim"),
         py::arg("to_dim"),
         "Check if adjacency information exists from one dimension to another")

    // Global IDs
    .def(
      "globals",
      [](Omega_h::Mesh& mesh, Omega_h::Int dim) {
        auto globals = mesh.globals(dim);
        return omega_h_read_to_numpy(globals);
      },
      py::arg("ent_dim"), "Get global IDs for entities")

    // Mesh sizing queries
    .def(
      "ask_sizes",
      [](Omega_h::Mesh& mesh) {
        auto sizes = mesh.ask_sizes();
        return omega_h_read_to_numpy(sizes);
      },
      "Get element sizes");
  module.def(
      "new_empty_mesh", []() { return Mesh(pybind11_global_library); });
  // Mesh utility functions
  module.def(
    "average_field",
    [](Omega_h::Mesh* mesh, Omega_h::Int dim, Omega_h::Int ncomps,
       py::array_t<Omega_h::Real> v2x) {
      auto v2x_view = numpy_to_omega_h_read<Omega_h::Real>(v2x);
      auto result = Omega_h::average_field(mesh, dim, ncomps, v2x_view);
      return omega_h_read_to_numpy(result);
    },
    py::arg("mesh"), py::arg("ent_dim"), py::arg("ncomps"),
    py::arg("vertex_data"),
    "Average vertex field data to entity centers (e.g., get face coordinates "
    "from vertex coordinates)");

  module.def(
    "average_field",
    [](Omega_h::Mesh* mesh, Omega_h::Int dim, py::array_t<Omega_h::LO> a2e,
       Omega_h::Int ncomps, py::array_t<Omega_h::Real> v2x) {
      auto a2e_view = numpy_to_omega_h_read<Omega_h::LO>(a2e);
      auto v2x_view = numpy_to_omega_h_read<Omega_h::Real>(v2x);
      auto result =
        Omega_h::average_field(mesh, dim, a2e_view, ncomps, v2x_view);
      return omega_h_read_to_numpy(result);
    },
    py::arg("mesh"), py::arg("ent_dim"), py::arg("a2e"), py::arg("ncomps"),
    py::arg("vertex_data"),
    "Average vertex field data to subset of entities specified by a2e");
}

#undef OMEGA_H_DECL_TYPE
#undef OMEGA_H_DEF_TYPE

}  // namespace Omega_h

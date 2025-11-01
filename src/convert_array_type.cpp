#include <Omega_h_array.hpp>
#include <Omega_h_comm.hpp>
#include <Omega_h_file.hpp>
#include <Omega_h_library.hpp>
#include <Omega_h_filesystem.hpp>
#include <Omega_h_mesh.hpp>
#include <Omega_h_tag.hpp>

// helper function to convert tag array types in files

namespace detail {
  template <typename T>
  void convert_mesh_array_type(
      Omega_h::Mesh* mesh, std::vector<std::string> const& tag_names, std::vector<Omega_h::Int> dims, Omega_h::ArrayType array_type) {
    for (size_t i = 0; i < tag_names.size(); ++i) {
      auto tag_name = tag_names[i];
      auto d = dims[i];
      if (mesh->has_tag(d, tag_name)) {
        auto tag = mesh->get_tag<T>(d, tag_name);
        mesh->set_tag(d, tag_name, tag->array(), true, array_type);
      }
    }
  }
}

template <typename T>
void convert_file_array_type(
    const std::string& input_path, const std::string& output_path, Omega_h::Library* lib, std::vector<std::string> const& tag_names, std::vector<Omega_h::Int> dims, Omega_h::ArrayType array_type = Omega_h::ArrayType::SymmetricSquareMatrix) {
  if (tag_names.size() != dims.size()) {
    Omega_h_fail("convert_file_array_type: tag_names and dims must have the same size\n");
  }
  auto const extension = Omega_h::filesystem::path(input_path).extension().string();
  if (extension == ".osh") {
    Omega_h::Mesh mesh = Omega_h::binary::read(input_path, lib->world());
    detail::convert_mesh_array_type<T>(&mesh, tag_names, dims, array_type);
    Omega_h::binary::write(output_path, &mesh);
  } else if (extension == ".vtu") {
    std::ifstream file(input_path);
    Omega_h::Mesh mesh = Omega_h::Mesh();
    if (file.is_open()) {
      Omega_h::vtk::read_vtu(file, lib->world(), &mesh);
      file.close();
    }
    detail::convert_mesh_array_type<T>(&mesh, tag_names, dims, array_type);
    Omega_h::vtk::write_vtu(output_path, &mesh, mesh.dim());
  } else if (extension == ".pvtu") {
    Omega_h::Mesh mesh = Omega_h::Mesh(lib);
    Omega_h::vtk::read_parallel(input_path, lib->world(), &mesh);
    detail::convert_mesh_array_type<T>(&mesh, tag_names, dims, array_type);
    Omega_h::vtk::write_parallel(output_path, &mesh, mesh.dim());
  } else if (extension == ".bp") {
    #ifdef OMEGA_H_USE_ADIOS
    Omega_h::Mesh mesh = Omega_h::adios::read(input_path, lib);
    detail::convert_mesh_array_type<T>(&mesh, tag_names, dims, array_type);
    Omega_h::adios::write(output_path, &mesh);
    #else
    Omega_h_fail(
        "Omega_h: Can't convert %s without reconfiguring with "
        "OMEGA_H_USE_ADIOS=ON\n",
        input_path.c_str());
    #endif
  } else if (extension == "meshb") {
  #ifdef OMEGA_H_USE_LIBMESHB
    Omega_h::Mesh mesh = Omega_h::Mesh(lib);
    for (size_t i = 0; i < tag_names.size(); ++i) {
      auto tag_name = tag_names[i];
      Omega_h::meshb::read_sol(&mesh, input_path.c_str(), tag_name);
      auto d = dims[i];
      auto tag = mesh.get_tag<T>(d, tag_name);
      mesh.set_tag(d, tag_name, tag->array(), true, array_type);
      Omega_h::meshb::write_sol(&mesh, output_path.c_str(), tag_name);
    }
    #else
    Omega_h_fail(
        "Omega_h: Can't convert %s without reconfiguring with "
        "OMEGA_H_USE_libMeshb=ON\n",
        input_path.c_str());
    #endif
  } else {
    Omega_h_fail("convert_file_array_type: unsupported file extension \"%s\" on \"%s\"\n",
        extension.c_str(), input_path.c_str());
  }
}

int main(int argc, char** argv) {
  auto lib = Omega_h::Library(&argc, &argv);
  if (argc < 5) {
    if (!lib.world()->rank())
      fprintf(stderr, "Usage: %s inputMesh outputMesh tagName1 dim1 [tagName2 dim2 ...]\n", argv[0]);
    exit(EXIT_FAILURE);
  }
  std::string input_path = argv[1];
  std::string output_path = argv[2];
  fprintf(stderr, "Converting file: %s to %s\n", input_path.c_str(), output_path.c_str());
  argc -= 3;
  argv += 3;
  std::vector<std::string> tag_names;
  std::vector<Omega_h::Int> dims;
  while (argc >= 2) {
    tag_names.push_back(argv[0]);
    dims.push_back(atoi(argv[1]));
    argc -= 2;
    argv += 2;
  }
  Omega_h::ArrayType array_type = Omega_h::ArrayType::SymmetricSquareMatrix;
  if (argc > 0) {
    std::string array_type_str = argv[0];
    array_type = Omega_h::NamesToArrayType.at(array_type_str);
  }
  fprintf(stderr, "Converting tags to array type: %s\n", Omega_h::ArrayTypeNames.at(array_type).c_str());
  convert_file_array_type<Omega_h::Real>(input_path, output_path, &lib, tag_names, dims, array_type);
  return 0;
}
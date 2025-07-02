#ifndef OMEGA_H_ADIOS2_HPP
#define OMEGA_H_ADIOS2_HPP
#include <adios2.h>
#include <Omega_h_mesh.hpp>
#include "Omega_h_filesystem.hpp" // filesystem
#include "Omega_h_library.hpp"
#include "Omega_h_simplex.hpp"

namespace Omega_h {

namespace adios {
void write(filesystem::path const& path,
                  std::map<Mesh*, std::string>& mesh_map);
void write(filesystem::path const& path, Mesh *mesh, std::string pref="");

void write_mesh(adios2::IO &io, adios2::Engine & writer,
                  Mesh* mesh, std::string pref);

Mesh read(filesystem::path const& path, Library* lib, std::string pref="");

} // namespace adios

} // namespace Omega_h
#endif

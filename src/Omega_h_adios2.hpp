#ifndef OMEGA_H_ADIOS2_HPP
#define OMEGA_H_ADIOS2_HPP
#include <adios2.h>
#include <Omega_h_mesh.hpp>
#if ADIOS2_USE_MPI
#include <mpi.h>
#endif
#include "Omega_h_filesystem.hpp" // filesystem
#include "Omega_h_library.hpp"

namespace Omega_h {

template <typename T>
void write_value(adios2::IO &io, adios2::Engine &writer,
        T val, std::string &name, bool global=false);

template <typename T>
void read_value(adios2::IO &io, adios2::Engine &reader,
        T *val, std::string &name, bool global=false);

void write_adios2(filesystem::path const& path, Mesh *mesh);
Mesh read_adios2(filesystem::path const& path, Library* lib);
}  // namespace Omega_h
#endif

![Omega\_h Logo][1]

# Omega\_h
> Reliable mesh adaptation

Omega\_h is a C++17 library that implements tetrahedron and triangle mesh adaptativity,
with a focus on scalable HPC performance using (optionally) MPI and OpenMP or CUDA.
It is intended to provided adaptive functionality to existing simulation codes.
Mesh adaptivity allows one to minimize both discretization error and number
of degrees of freedom live during the simulation, as well as enabling moving
object and evolving geometry simulations.
Omega\_h will do this for you in a way that is fast, memory-efficient, and
portable across many different architectures.

## Extensions

This fork of Omega\_h from https://github.com/sandialabs/omega_h v9.34.12 adds support for:

- loading serial Simmetrix meshes
- reverse classification, and defining tags on sets of mesh entities defined by a reverse classification relation
- representing single-process mixed meshes
- domains with matched/periodic entities

## Installing / Getting started

For a bare minimum setup with no parallelism, you just need [CMake][0],
a C++14 compiler, and preferably [ZLib][6] installed.

```shell
git clone git@github.com:SCOREC/omega_h.git
cd omega_h
cmake . -DCMAKE_INSTALL_PREFIX=/your/choice
make install
```

This should install Omega\_h under the given prefix in a way you can
access from your own CMake files using these CMake commands:

```cmake
find_package(Omega_h)
target_link_libraries(myprogram Omega_h::omega_h)
```

## Features

Omega\_h provides at least the following:
* Adaptation of tetrahedral and triangle meshes in parallel
* Anisotropic metric field support
* Given good input element quality, the output element
  quality is also guaranteed.
* Scalable MPI parallelism
* On-node OpenMP or CUDA parallelism
* Fully deterministic execution
* Given the same mesh, global numbering, and size field,
  results will be independent of parallel partitioning
  and ordering.

## Configuration

Kokkos builds require Kokkos 4.4 or newer with bundled mdspan enabled
(`Kokkos_ENABLE_IMPL_MDSPAN` in Kokkos 4.x). Builds without Kokkos use
an installed, vendored mdspan copy through `Omega_h_mdspan.hpp`; avoid
including another mdspan distribution in the same translation unit because
these distributions share internal include guards and namespaces.

New osh outputs use format version 12. Existing older osh files remain
readable; older Omega_h versions cannot read the new outputs.

Below we document some key CMake configuration options:

#### Omega\_h\_USE\_MPI
Default: `OFF`

Whether to enable MPI parallelism.
We recommend using [MPICH][3] or another MPI 3.0 implementation,
but we also support MPI version 2.1.
If this is `ON`, set `CMAKE_CXX_COMPILER` to your MPI compiler wrapper.

#### Omega\_h\_USE\_OpenMP
Default: `OFF`

Whether to enable OpenMP thread parallelism.
The `-fopenmp` flag will automatically be added.

#### Omega\_h\_USE\_CUDA
Default: `OFF`

Whether to enable CUDA GPU parallelism.

#### Omega\_h\_USE\_SEACASExodus
Default: `OFF`

Whether to use the Exodus subpackage of the SEACAS toolkit.
This allows reading and writing Exodus files from Omega\_h.
By default, it will look for this dependency in `SEACASExodus_PREFIX`.

## Embedded meshes

The existing `Mesh` supports edges in 2D/3D and triangles in 3D.
`dim()` (also `tdim()`) describes topology; `space_dim()` describes coordinates
and defaults to `dim()` for ordinary meshes.

```cpp
Omega_h::Mesh mesh(&library);
Omega_h::build_from_elems_and_coords(
    &mesh, OMEGA_H_SIMPLEX, 2, triangle_vertices, xyz_coordinates, 3);
auto xyz = mesh.coords_view<3>(); // owns storage; checks the dimension on the host
// In an Omega_h device kernel: xyz(vertex, component)
```

`coords_view()` provides a dynamic extent; `coords_view<N>()` checks a static
extent, including in Release builds. `CoordsWriteView<N>` wraps writable
coordinate arrays, and `set_coords(CoordsView<>)` validates mesh extents.
`HostCoordsView<N>` owns a host mirror for inspecting coordinates from device
storage with the same `(vertex, component)` indexing.
The deprecated raw `coords()` and `set_coords(Reals)` methods work only for
ordinary meshes. `OMEGA_H_SUPPRESS_DEPRECATION` silences their warnings while
retaining the embedded-mesh checks.

Embedded meshes support real element measures and edge lengths, nonpredictive
and weighted balancing, migration/ghosting, osh I/O, and VTK output. Surface
areas are unsigned. Adaptation, metrics, quality, recovery, conservation, and
AMR reject embedded meshes. Unported tools also reject them. Topological
classification is supported; distributed classification needs a ghosted partition.

## Contributing

Please open a Github issue to ask a question, report a bug,
request features, etc.
If you'd like to contribute, please fork the repository and use a feature
branch. Pull requests are welcome.

[0]: https://cmake.org
[1]: https://raw.githubusercontent.com/SNLComputation/omega_h/master/misc/omega_h.png
[3]: http://www.mpich.org
[6]: http://zlib.net

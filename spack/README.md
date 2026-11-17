# CUDA development environment

This follows `~/code/pcms/pcms-cuda-env`: GCC 12.3, external CUDA 12.1
and MPICH, and Kokkos 4.7.04 with CUDA architecture 75 (RTX 2060 SUPER).
Omega_h builds directly from this checkout in `RelWithDebInfo` mode, with
MPI, shared libraries, and the CTest suite enabled. GPU-aware MPI is disabled
because the site's external MPICH does not support it.

The environment uses the existing user Spack configuration for the SCOREC
and builtin repositories, compiler, CUDA, and MPICH externals. The local
`omega_dev` recipe extends SCOREC's recipe with a `develop` version, a
`tests` variant, and explicit CUDA architecture settings. Modern Omega_h
selects CUDA through Kokkos, rather than an Omega_h `+cuda` variant.

The checked-in `spack.lock` was concretized successfully with Spack 1.2.2.
All dependencies resolve to existing installations or site externals;
only the development Omega_h needs building. The development path in the
lockfile is specific to this checkout; reconcretize when moving it.

The CUDA build and all 51 registered tests passed on 2026-11-16, including
2-rank and 4-rank MPI cases. See [validation.txt](validation.txt) for the
source revision, configuration, commands, and result.

From the checkout root, review and concretize without building:

```sh
source ~/spack/spack/share/spack/setup-env.sh
spack -e ./spack concretize
spack -e ./spack find -dlv
```

Build and install:

```sh
spack -e ./spack install -j 4
```

Spack keeps this development build under its staging directory, with sources
pointing at this checkout. Query the build directory, then run the tests in Spack's build environment
so its compiler and runtime dependencies are available:

```sh
omega_build_dir=$(spack -e ./spack location -b omega-h)
spack -e ./spack build-env omega-h -- ctest --test-dir "$omega_build_dir" -N
spack -e ./spack build-env omega-h -- ctest --test-dir "$omega_build_dir" --output-on-failure -j 1
```

Run tests sequentially to avoid competing for GPU memory. CTest includes
multi-rank MPI cases. Tests needing the optional external `omega_h-data`
dataset are omitted unless `Omega_h_DATA` is set at CMake configuration time.
Python, Exodus, Gmsh, and proprietary geometry interfaces are not enabled.

After source edits, rerun `spack -e ./spack install -j 4` and the CTest command.
Spack detects changes to development sources and rebuilds incrementally.
For a direct build iteration, use:

```sh
spack -e ./spack build-env omega-h -- cmake --build "$omega_build_dir" -j 4
```

GPU execution requires a working NVIDIA driver and visible GPU; check
`nvidia-smi` before running tests. The sandbox cannot communicate with the
driver, but the host reports an RTX 2060 SUPER with driver 590.48.01.
Builds and GPU tests in Codex therefore run with sandbox escalation.

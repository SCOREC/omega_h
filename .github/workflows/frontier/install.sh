# #!/bin/bash

branch=$1

cd /lustre/orion/phy122/scratch/castia5/globus-compute/omega_h-test

module load rocm
module load craype-accel-amd-gfx90a
module load cray-mpich
export CRAYPE_LINK_TYPE=dynamic
export MPICH_GPU_SUPPORT_ENABLED=1

# # Kokkos
# rm kokkos -rf
# rm build-kokkos -rf
# git clone -b 4.1.00 git@github.com:Kokkos/kokkos.git
# cmake -S kokkos -B build-kokkos \
#  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
#  -DCMAKE_CXX_COMPILER=CC \
#  -DCMAKE_CXX_EXTENSIONS=OFF \
#  -DKokkos_ENABLE_TESTS=OFF \
#  -DKokkos_ENABLE_EXAMPLES=OFF \
#  -DKokkos_ENABLE_HIP=ON \
#  -DKokkos_ARCH_VEGA90A=ON \
#  -DKokkos_ENABLE_SERIAL=ON \
#  -DKokkos_ENABLE_OPENMP=OFF \
#  -DKokkos_ENABLE_DEBUG=OFF \
#  -DCMAKE_INSTALL_PREFIX=build-kokkos/install
# cmake --build build-kokkos -j8 --target install

# Omega_h
rm omega_h -rf
rm build-omega_h -rf
git clone git@github.com:SCOREC/omega_h.git
cd omega_h && git checkout $branch && cd -
cmake -S omega_h -B build-omega_h \
  -DCMAKE_INSTALL_PREFIX=build-omega_h/install \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DBUILD_SHARED_LIBS=OFF \
  -DOmega_h_USE_MPI=ON \
  -DOmega_h_USE_OpenMP=OFF \
  -DCMAKE_CXX_COMPILER=CC \
  -DOMEGA_H_USE_HIP=ON \
  -DOmega_h_USE_Kokkos=ON \
  -DKokkos_PREFIX=$PWD/build-kokkos/install \
  -DBUILD_TESTING=ON
cmake --build build-omega_h -j24 --target install

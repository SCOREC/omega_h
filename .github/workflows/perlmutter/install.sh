# !/bin/bash -e

branch=$1

cd $SCRATCH/globus-compute/omega_h-test

# # kokkos
# rm kokkos -rf
# rm build-kokkos -rf
# git clone -b 4.5.00 https://github.com/kokkos/kokkos.git
# cmake -S kokkos -B build-kokkos \
#   -DCMAKE_INSTALL_PREFIX=build-kokkos/install \
#   -DCMAKE_BUILD_TYPE="Release" \
#   -DCMAKE_CXX_COMPILER=$PWD/kokkos/bin/nvcc_wrapper \
#   -DKokkos_ARCH_AMPERE80=ON \
#   -DKokkos_ENABLE_SERIAL=ON \
#   -DKokkos_ENABLE_OPENMP=off \
#   -DKokkos_ENABLE_CUDA=on \
#   -DKokkos_ENABLE_CUDA_LAMBDA=on \
#   -DKokkos_ENABLE_DEBUG=off
# cmake --build build-kokkos -j 24 --target install

# omegah
rm omega_h -rf
rm build-omega_h -rf
git clone https://github.com/SCOREC/omega_h.git
cd omega_h && git checkout $branch && cd -
cmake -S omega_h -B build-omega_h \
  -DCMAKE_INSTALL_PREFIX=build-omega_h/install \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=off \
  -DOmega_h_USE_Kokkos=ON \
  -DOmega_h_USE_CUDA=on \
  -DOmega_h_CUDA_ARCH=80 \
  -DOmega_h_USE_MPI=on  \
  -DBUILD_TESTING=on  \
  -DCMAKE_C_COMPILER=cc \
  -DCMAKE_CXX_COMPILER=CC \
  -DKokkos_PREFIX=$PWD/build-kokkos/install/lib64/cmake
cmake --build build-omega_h -j 24 --target install
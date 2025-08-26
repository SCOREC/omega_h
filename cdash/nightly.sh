#!/bin/bash -x
source /etc/profile
source /users/yus9/.bash_profile

#setup lmod
export PATH=/usr/share/lmod/lmod/libexec:$PATH

#setup spack modules
unset MODULEPATH

module use /opt/scorec/spack/rhel9/v0201_4/lmod/linux-rhel9-x86_64/Core/
module load gcc/12.3.0-iil3lno
module load mpich/4.1.1-xpoyz4t
module load cmake/3.26.3-2duxfcd
module load cuda/12.1.1-zxa4msk

#cdash output root
d=/lore/yus9/nightlyBuilds/omega_h_build
cd $d
#remove compilation directories created by previous nightly.cmake runs
[ -d build ] && rm -rf build/

#install kokkos
[ ! -d kokkos ] && git clone https://github.com/kokkos/kokkos.git
cd kokkos && git pull && cd -
[ -d build-kokkos ] && rm -rf build-kokkos
cmake -S kokkos -B build-kokkos \
  -DCMAKE_INSTALL_PREFIX=build-kokkos/install \
  -DCMAKE_BUILD_TYPE="Release" \
  -DCMAKE_CXX_COMPILER=$d/kokkos/bin/nvcc_wrapper \
  -DKokkos_ARCH_AMPERE80=ON \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_OPENMP=OFF \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ENABLE_CUDA_LAMBDA=ON \
  -DKokkos_ENABLE_DEBUG=OFF
cmake --build build-kokkos -j 24 --target install

touch $d/startedCoreNightly
#run nightly.cmake script
ctest -V --script $d/nightly.cmake
touch $d/doneCoreNightly

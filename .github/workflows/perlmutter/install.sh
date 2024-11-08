# !/bin/bash -e

branch=$1

cd $SCRATCH/globus-compute/omega_h-test

export root=$PWD
module load PrgEnv-gnu
module load cudatoolkit
module load cmake

export kk=$root/build-kokkos/install   # This is where kokkos will be (or is) installed
export oh=$root/build-omega_h/install  # This is where omega_h will be (or is) installed
export CMAKE_PREFIX_PATH=$kk:$kk/lib64/cmake:$oh:$CMAKE_PREFIX_PATH
export MPICH_CXX=$root/kokkos/bin/nvcc_wrapper

export SLURM_CPU_BIND="cores"


# #kokkos
# rm ${kk%%install} -rf
# rm kokkos -rf
# git clone -b 4.2.00 https://github.com/kokkos/kokkos.git
# cmake -S kokkos -B ${kk%%install} \
#   -DCMAKE_INSTALL_PREFIX=$kk \
#   -DCMAKE_BUILD_TYPE="Release" \
#   -DCMAKE_CXX_COMPILER=$root/kokkos/bin/nvcc_wrapper \
#   -DKokkos_ARCH_AMPERE80=ON \
#   -DKokkos_ENABLE_SERIAL=ON \
#   -DKokkos_ENABLE_OPENMP=off \
#   -DKokkos_ENABLE_CUDA=on \
#   -DKokkos_ENABLE_CUDA_LAMBDA=on \
#   -DKokkos_ENABLE_DEBUG=off
# cmake --build ${kk%%install} -j 24 --target install

#omegah
rm ${oh%%install} -rf
rm omega_h -rf
git clone https://github.com/SCOREC/omega_h.git
cd omega_h && git checkout $branch && cd -
cmake -S omega_h -B ${oh%%install} \
  -DCMAKE_INSTALL_PREFIX=$oh \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=off \
  -DOmega_h_USE_Kokkos=ON \
  -DOmega_h_USE_CUDA=on \
  -DOmega_h_CUDA_ARCH=80 \
  -DOmega_h_USE_MPI=on  \
  -DMPIEXEC_EXECUTABLE=srun \
  -DBUILD_TESTING=on  \
  -DCMAKE_C_COMPILER=cc \
  -DCMAKE_CXX_COMPILER=CC \
  -DKokkos_PREFIX=$kk/lib64/cmake
cmake --build ${oh%%install} -j 24 --target install
from spack_repo.scorec.packages.omega_h.package import OmegaH as ScorecOmegaH

from spack.package import *


class OmegaH(ScorecOmegaH):
    """Build and test this checkout with the SCOREC dependency setup."""

    version("develop", branch="master", git="https://github.com/SCOREC/omega_h.git")
    variant("tests", default=True, description="Build the CTest suite")

    def cmake_args(self):
        args = super().cmake_args()
        args.extend([
            self.define_from_variant("BUILD_TESTING", "tests"),
            # The site's external MPICH does not support device pointers.
            self.define("OMEGA_H_USE_GPU_AWARE_MPI", False),
        ])
        if self.spec.satisfies("+kokkos ^kokkos+cuda"):
            arch = self.spec["kokkos"].variants["cuda_arch"].value
            args.extend([
                self.define("Omega_h_CUDA_ARCH", arch),
                self.define("CMAKE_CUDA_ARCHITECTURES", arch),
            ])
        return args

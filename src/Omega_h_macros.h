#ifndef OMEGA_H_MACROS_H
#define OMEGA_H_MACROS_H

#include <Omega_h_config.h>

#if defined(OMEGA_H_USE_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

#define OMEGA_H_STRINGIFY(s) #s
/* apparently you need two macros to make a string */
#define OMEGA_H_TOSTRING(s) OMEGA_H_STRINGIFY(s)

#define OMEGA_H_PRAGMA(x) _Pragma(#x)

#if defined(_MSC_VER)
#define OMEGA_H_SYSTEM_HEADER
#elif defined(__clang__)
#define OMEGA_H_SYSTEM_HEADER OMEGA_H_PRAGMA(clang system_header)
#elif defined(__GNUC__)
#define OMEGA_H_SYSTEM_HEADER OMEGA_H_PRAGMA(GCC system_header)
#endif

#define OMEGA_H_PI 3.14159265358979323
//                  ^    ^    ^    ^
//                  0    5   10   15
#define OMEGA_H_EPSILON 1e-10
#define OMEGA_H_MANTISSA_BITS 52

#if defined(__clang__)
#define OMEGA_H_FALLTHROUGH [[clang::fallthrough]]
/* Test for GCC >= 7.0.0 */
#elif defined(__GNUC__) && (__GNUC__ > 7 || __GNUC__ == 7)
#define OMEGA_H_FALLTHROUGH [[gnu::fallthrough]]
#else
#define OMEGA_H_FALLTHROUGH ((void)0)
#endif

#ifdef _MSC_VER
#define OMEGA_H_NODISCARD
#else
#define OMEGA_H_NODISCARD __attribute__((warn_unused_result))
#endif

#if defined(OMEGA_H_USE_KOKKOS) && defined(OMEGA_H_USE_CUDA)
#define OMEGA_H_INLINE __host__ __device__ inline
#define OMEGA_H_INLINE_BIG OMEGA_H_INLINE
#define OMEGA_H_DEVICE __host__ __device__ inline
#define OMEGA_H_LAMBDA [=] __host__ __device__
#if defined(__CUDA_ARCH__)
#define OMEGA_H_CONSTANT_DATA __constant__
#else
#define OMEGA_H_CONSTANT_DATA
#endif
#elif defined(OMEGA_H_USE_KOKKOS)
#define OMEGA_H_INLINE KOKKOS_INLINE_FUNCTION
#define OMEGA_H_INLINE_BIG OMEGA_H_INLINE
#define OMEGA_H_DEVICE KOKKOS_INLINE_FUNCTION
#define OMEGA_H_LAMBDA KOKKOS_LAMBDA
#define OMEGA_H_CONSTANT_DATA constexpr
#elif defined(OMEGA_H_USE_CUDA)
#define OMEGA_H_INLINE __host__ __device__ inline
#define OMEGA_H_INLINE_BIG OMEGA_H_INLINE
#define OMEGA_H_DEVICE __host__ __device__ inline
#define OMEGA_H_LAMBDA [=] __host__ __device__
#define OMEGA_H_CONSTANT_DATA constexpr
#elif defined(_MSC_VER)
#define OMEGA_H_INLINE __forceinline
#define OMEGA_H_INLINE_BIG inline
#define OMEGA_H_DEVICE __forceinline
#define OMEGA_H_LAMBDA [=]
#define OMEGA_H_CONSTANT_DATA
#else
#define OMEGA_H_INLINE __attribute__((always_inline)) inline
#define OMEGA_H_INLINE_BIG inline
#define OMEGA_H_DEVICE __attribute__((always_inline)) inline
#define OMEGA_H_LAMBDA [=]
#define OMEGA_H_CONSTANT_DATA
#endif

#if defined(OMEGA_H_CHECK_BOUNDS) && defined(OMEGA_H_THROW)
#define OMEGA_H_NOEXCEPT
#else
#define OMEGA_H_NOEXCEPT noexcept
#endif

#if defined(_MSC_VER) && defined(OMEGA_H_IS_SHARED)
#ifdef omega_h_EXPORTS
#define OMEGA_H_DLL __declspec(dllexport)
#else
#define OMEGA_H_DLL __declspec(dllimport)
#endif
#else
#define OMEGA_H_DLL
#endif

#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__CUDA_ARCH__) && !defined(__SYCL_DEVICE_ONLY__)
#define OMEGA_H_COMPILING_FOR_HOST
#endif

#endif

# - Try to find ADIOS2
# Once done this will define
#  ADIOS2_FOUND - System has ADIOS2
#  ADIOS2_INCLUDE_DIR - The ADIOS2 include directories
#  ADIOS2_LIBS - The libraries needed to use ADIOS2 
#  ADIOS2_<library>_FOUND - System has <library>
#  ADIOS2_MAJOR_VERSION - the leading integer of the version string
#  ADIOS2_MINOR_VERSION - the date code from the version string
#
# Based on input variables:
#  ADIOS2_LIB_DIR
#  ADIOS2_INCLUDE_DIR
# And environment variable:
#  CMAKE_PREFIX_PATH
#
# This implementation assumes an adios2 install has the following structure
# VERSION/
#         include/*.h
#         lib64/*.a

macro(adios2LibCheck libs isRequired)
  foreach(lib ${libs})
    unset(adios2lib CACHE)
    find_library(adios2lib "${lib}" PATHS ${ADIOS2_LIB_DIR})
    if(adios2lib MATCHES "^adios2lib-NOTFOUND$")
      if(${isRequired})
	      message(FATAL_ERROR "adios2 library ${lib} not found in ${ADIOS2_LIB_DIR}")
      else()
	      message("adios2 library ${lib} not found in ${ADIOS2_LIB_DIR}")
      endif()
    else()
	    set("ADIOS2_${lib}_FOUND" TRUE CACHE INTERNAL "ADIOS2 library present")
      set(ADIOS2_LIBS ${ADIOS2_LIBS} ${adios2lib})
    endif()
  endforeach()
endmacro(adios2LibCheck)

find_path(ADIOS2_INCLUDE_DIR
  NAMES adios2_c.h  adios2.h
  PATHS ${ADIOS2_INCLUDE_DIR})
if(NOT EXISTS "${ADIOS2_INCLUDE_DIR}")
  message(FATAL_ERROR "adios2 include dir not found")
endif()

string(REGEX REPLACE
  "/include$" ""
  ADIOS2_INSTALL_DIR
  "${ADIOS2_INCLUDE_DIR}")

string(REGEX MATCH
  "[0-9]+[.][0-9]+-[0-9]+"
  ADIOS2_VERSION
  "${ADIOS2_INCLUDE_DIR}")

#VERSION_LESS and VERSION_GREATER need '.' delimited version strings.
string(REGEX REPLACE
  "([0-9]+[.][0-9]+)-([0-9]+)"
  "\\1.\\2" ADIOS2_DOT_VERSION
  "${ADIOS2_VERSION}")
string(REGEX REPLACE
  "([0-9]+)[.]([0-9]+)-([0-9]+)"
  "\\1" ADIOS2_MAJOR_VERSION
  "${ADIOS2_VERSION}")
string(REGEX REPLACE
  "([0-9]+)[.]([0-9]+)-([0-9]+)"
  "\\3" ADIOS2_MINOR_VERSION
  "${ADIOS2_VERSION}")

set(MIN_VALID_ADIOS2_VERSION 2.10.0)
set(MAX_VALID_ADIOS2_VERSION 2.10.10)
#if( ${SKIP_ADIOS2_VERSION_CHECK} )
#	message(STATUS "Skipping ADIOS2 version check."
#    " This may result in undefined behavior")
#elseif( (ADIOS2_DOT_VERSION VERSION_LESS MIN_VALID_ADIOS2_VERSION) OR
#	(ADIOS2_DOT_VERSION VERSION_GREATER MAX_VALID_ADIOS2_VERSION) )
#  MESSAGE(FATAL_ERROR 
#	  "invalid ADIOS2 version: ${ADIOS2_DOT_VERSION}, \
#    valid versions are ${MIN_VALID_ADIOS2_VERSION} to ${MAX_VALID_ADIOS2_VERSION}")
#endif()
message(STATUS "Building with ADIOS2 ${ADIOS2_DOT_VERSION}")

set(ADIOS2_LIBS "")

if (Omega_h_USE_MPI)
set(ADIOS2_LIB_NAMES
	adios2_core_mpi
	adios2_cxx11_mpi
)
else()
set(ADIOS2_LIB_NAMES
   	adios2_core
    	adios2_cxx11
)
endif()

adios2LibCheck("${ADIOS2_LIB_NAMES}" TRUE)

# handle the QUIETLY and REQUIRED arguments and set ADIOS2_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(ADIOS2 DEFAULT_MSG
  ADIOS2_LIBS ADIOS2_INCLUDE_DIR
)

mark_as_advanced(ADIOS2_INCLUDE_DIR ADIOS2_LIBS)

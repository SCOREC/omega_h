cmake_minimum_required(VERSION 3.10)
project(TestOmegaHConfig CXX)

set(CMAKE_CXX_EXTENSIONS OFF)

# Enable testing
enable_testing()

# Find Omega_h package
find_package(Omega_h REQUIRED)

# Create test executable
add_executable(test_export test_export.cpp)

# Link with Omega_h
target_link_libraries(test_export Omega_h::omega_h)

# Set C++ standard to match Omega_h (C++17)
set_property(TARGET test_export PROPERTY CXX_STANDARD 17)
set_property(TARGET test_export PROPERTY CXX_STANDARD_REQUIRED ON)

# Add test
add_test(NAME test_export COMMAND test_export)
set_tests_properties(test_export PROPERTIES 
  PASS_REGULAR_EXPRESSION "Test completed successfully!")
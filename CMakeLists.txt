cmake_minimum_required(VERSION 3.16)
project(ParallelVision VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(PV_WITH_MPI  "Build MPI backend (row strips + halo exchange)" OFF)
option(PV_WITH_CUDA "Build CUDA backend (needs nvcc)"               OFF)
option(PV_BUILD_CLI "Build the pv command-line tool (needs OpenCV)" ON)

# Everything lands in build/bin so libpv.dll sits next to the executables.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

find_package(OpenMP REQUIRED COMPONENTS CXX)

# ---- core shared library (no OpenCV dependency) ----
add_library(pv SHARED
  src/dispatcher.cpp
  src/calibrate.cpp
  src/color.cpp
  src/filters.cpp
  src/edges.cpp
  src/threshold.cpp)
target_include_directories(pv PUBLIC include PRIVATE src)
target_compile_definitions(pv PRIVATE PV_BUILD_DLL)
target_compile_options(pv PRIVATE -Wall -Wextra -O3)
target_link_libraries(pv PUBLIC OpenMP::OpenMP_CXX)

if(PV_WITH_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
  target_sources(pv PRIVATE src/mpi_backend.cpp)
  target_compile_definitions(pv PUBLIC PV_HAVE_MPI)
  target_link_libraries(pv PUBLIC MPI::MPI_CXX)
endif()

# ---- tests ----
enable_testing()
add_executable(pv_tests tests/test_main.cpp)
target_link_libraries(pv_tests PRIVATE pv)
add_test(NAME pv_tests COMMAND pv_tests)

# ---- benchmarks (synthetic images, CSV output) ----
add_executable(pv_bench bench/bench.cpp)
target_link_libraries(pv_bench PRIVATE pv)

# heuristic vs calibrated dispatcher comparison (CSV + accuracy summary)
add_executable(pv_dispatch_report bench/dispatch_report.cpp)
target_link_libraries(pv_dispatch_report PRIVATE pv)

# ---- CLI ----
if(PV_BUILD_CLI)
  find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)
  add_executable(pv_cli cli/pv.cpp)
  set_target_properties(pv_cli PROPERTIES OUTPUT_NAME pv)
  target_link_libraries(pv_cli PRIVATE pv ${OpenCV_LIBS})
  target_include_directories(pv_cli PRIVATE ${OpenCV_INCLUDE_DIRS})
endif()

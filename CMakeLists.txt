cmake_minimum_required(VERSION 3.24)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86-real)
endif()
project(qwen35_engine LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# Zen 3 host code: AVX2 + FMA + F16C + BMI2. No -march=native so the Nix build is reproducible.
set(QE_CPU_FLAGS -mavx2 -mfma -mf16c -mbmi2 -mtune=znver3)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")

file(GLOB QE_CORE_CPP src/*.cpp)
list(FILTER QE_CORE_CPP EXCLUDE REGEX ".*/(main|server)\\.cpp$")
file(GLOB QE_CORE_CU src/*.cu)

add_library(qecore STATIC ${QE_CORE_CPP} ${QE_CORE_CU})
target_include_directories(qecore PUBLIC src)
target_compile_options(qecore PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:${QE_CPU_FLAGS} -Wall -Wno-unused-function>
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -lineinfo --expt-relaxed-constexpr -Xcompiler=-mavx2,-mfma,-mf16c>)
set_target_properties(qecore PROPERTIES CUDA_SEPARABLE_COMPILATION OFF POSITION_INDEPENDENT_CODE ON)
target_link_libraries(qecore PUBLIC CUDA::cudart CUDA::cublas CUDA::cuda_driver Threads::Threads)

add_executable(qwen35-engine src/main.cpp)
target_link_libraries(qwen35-engine PRIVATE qecore)
target_compile_options(qwen35-engine PRIVATE ${QE_CPU_FLAGS})

if(EXISTS ${CMAKE_SOURCE_DIR}/src/server.cpp)
  add_executable(qwen35-server src/server.cpp)
  target_link_libraries(qwen35-server PRIVATE qecore)
  target_compile_options(qwen35-server PRIVATE ${QE_CPU_FLAGS})
  install(TARGETS qwen35-server RUNTIME DESTINATION bin)
endif()

file(GLOB QE_TOOLS tools/qe_*.cpp tools/qe_*.cu)
foreach(t ${QE_TOOLS})
  get_filename_component(n ${t} NAME_WE)
  add_executable(${n} ${t})
  target_link_libraries(${n} PRIVATE qecore)
  target_compile_options(${n} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${QE_CPU_FLAGS}>)
endforeach()

install(TARGETS qwen35-engine RUNTIME DESTINATION bin)

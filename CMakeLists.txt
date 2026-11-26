cmake_minimum_required(VERSION 3.18)
project(freesched VERSION 0.1.0 LANGUAGES C CXX)

option(FREESCHED_BUILD_TESTS "Build freesched C11/C++17 tests and benchmark" ON)

add_library(freesched STATIC
  src/fs_topology.cpp
  src/fs_dmabuf.cpp
  src/fs_router.cpp
  src/fs_executor.cpp
  src/fs_backend_openbuenos.cpp
  src/fs_backend_vulkan_dmabuf.cpp
  src/ggml_freesilicon.cpp
  src/pc_gguf.cpp
  src/pc_kv_ring.cpp
  src/pc_soa_neon.cpp
  src/pc_dmabuf_graph.cpp
)

add_library(freesched::freesched ALIAS freesched)

target_compile_features(freesched PRIVATE cxx_std_17)
target_compile_options(freesched PRIVATE
  -Wall -Wextra -Werror -fno-exceptions -fno-rtti
)

target_include_directories(freesched
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if (UNIX AND NOT APPLE)
  if (ANDROID)
    target_link_libraries(freesched PUBLIC m ${CMAKE_DL_LIBS})
  else()
    target_link_libraries(freesched PUBLIC m pthread ${CMAKE_DL_LIBS})
  endif()
elseif (APPLE)
  target_link_libraries(freesched PUBLIC m)
endif()

add_executable(freesched_dmabuf_smoke tools/freesched_dmabuf_smoke.cpp)
target_link_libraries(freesched_dmabuf_smoke PRIVATE freesched)
target_compile_features(freesched_dmabuf_smoke PRIVATE cxx_std_17)
target_compile_options(freesched_dmabuf_smoke PRIVATE -Wall -Wextra -Werror -fno-exceptions -fno-rtti)

add_executable(pocketcoder-run tools/pocketcoder_run.cpp)
target_link_libraries(pocketcoder-run PRIVATE freesched)
target_compile_features(pocketcoder-run PRIVATE cxx_std_17)
target_compile_options(pocketcoder-run PRIVATE -Wall -Wextra -Werror -fno-exceptions -fno-rtti)

if (FREESCHED_BUILD_TESTS)
  enable_testing()
  add_test(NAME freesched_dmabuf_smoke COMMAND freesched_dmabuf_smoke --profile laguna --json)
  add_test(NAME pocketcoder_run_bench_synthetic COMMAND pocketcoder-run --bench-synthetic --iters 20)

  add_executable(test_c11_abi tests/test_c11_abi.c)
  target_link_libraries(test_c11_abi PRIVATE freesched)
  set_target_properties(test_c11_abi PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS OFF
  )
  target_compile_options(test_c11_abi PRIVATE -Wall -Wextra -Werror)
  add_test(NAME test_c11_abi COMMAND test_c11_abi)

  add_executable(test_freesched tests/test_freesched.cpp)
  target_link_libraries(test_freesched PRIVATE freesched)
  target_compile_features(test_freesched PRIVATE cxx_std_17)
  target_compile_options(test_freesched PRIVATE -Wall -Wextra -Werror -fno-exceptions -fno-rtti)
  add_test(NAME test_freesched COMMAND test_freesched)

  add_executable(test_pocketcoder tests/test_pocketcoder.cpp)
  target_link_libraries(test_pocketcoder PRIVATE freesched)
  target_compile_features(test_pocketcoder PRIVATE cxx_std_17)
  target_compile_options(test_pocketcoder PRIVATE -Wall -Wextra -Werror -fno-exceptions -fno-rtti)
  add_test(NAME test_pocketcoder COMMAND test_pocketcoder)

  add_executable(bench_heterogeneous tests/bench_heterogeneous.cpp)
  target_link_libraries(bench_heterogeneous PRIVATE freesched)
  target_compile_features(bench_heterogeneous PRIVATE cxx_std_17)
  target_compile_options(bench_heterogeneous PRIVATE -Wall -Wextra -Werror -fno-exceptions -fno-rtti)

  find_package(Python3 COMPONENTS Interpreter)
  if (Python3_FOUND)
    add_test(
      NAME test_twin
      COMMAND ${Python3_EXECUTABLE} -m unittest discover -s ${CMAKE_CURRENT_SOURCE_DIR}/tests -p "test_*.py" -v
    )
  endif()
endif()

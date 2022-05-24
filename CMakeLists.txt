cmake_minimum_required(VERSION 3.9)
project(main)


find_package(autodiff)
find_package(pybind11)
find_package(Eigen3)

if(SKBUILD)
  # Scikit-Build does not add your site-packages to the search path
  # automatically, so we need to add it _or_ the pybind11 specific directory
  # here.
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c
            "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE _tmp_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
  list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")
endif()

pybind11_add_module(_core MODULE src/main.cpp)

set_property(TARGET _core PROPERTY CXX_STANDARD 17)
target_link_libraries(_core PRIVATE Eigen3::Eigen autodiff::autodiff ${NLOPT_LIBRARIES})
install(TARGETS _core DESTINATION .)
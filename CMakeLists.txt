cmake_minimum_required(VERSION 3.18)
project(hpfold LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# Writes compile_commands.json, read by clangd (editor diagnostics).
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Simulation logic, independent of Python.
add_library(hp_core STATIC
    cpp/src/conformation.cpp
    cpp/src/pull_moves.cpp
    cpp/src/remc.cpp
    cpp/src/sample_writer.cpp
    cpp/src/vshd_moves.cpp
)
target_include_directories(hp_core PUBLIC cpp/include)
target_compile_options(hp_core PRIVATE -Wall -Wextra)
# Required because the static library is linked into a shared module.
set_target_properties(hp_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Python module `hpfold._core` (pybind11 wrapper).
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_core cpp/bindings/bindings.cpp)
target_link_libraries(_core PRIVATE hp_core)

install(TARGETS _core LIBRARY DESTINATION hpfold)

// pybind11 wrapper: exposes the C++ code as the Python module `hpfold._core`.

#include <pybind11/pybind11.h>

#include "hp/remc.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "C++ core of hpfold: REMC for HP proteins on a 2D lattice.";

    // TODO(phase 1): expose the parameter struct and the final signature.
    m.def("run_remc", &hp::run_remc, "Runs the REMC simulation (not implemented yet).");
}

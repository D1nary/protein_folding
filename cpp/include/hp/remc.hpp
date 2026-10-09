#pragma once

// Replica Exchange Monte Carlo (Thachuk et al., 2007).
// See notes/samplings.md.

namespace hp {

// TODO(phase 1): struct with the simulation parameters, filled from Python
// with the values of the configuration file (no default values here).

// Runs the REMC simulation: local sampling with the Metropolis criterion
// and swaps between replicas at adjacent temperatures (even-odd scheme).
//
// TODO(phase 1): define parameters and return value. For now it throws
// std::logic_error.
void run_remc();

}  // namespace hp

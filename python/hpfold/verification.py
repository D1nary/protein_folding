"""Check of phase 1 — is the conformation found by the REMC the ground state?

The true minimum energy of the sequence is computed by exhaustive enumeration
(C++ function ``exact_ground_state``) and compared with the energy of the
conformation saved by the simulation. The enumeration is expensive, so it is
not part of the simulation: it runs only when this phase is requested.
"""

from hpfold import _core
from hpfold.conformation import load_conformation


def exact_energy(config: dict, sequence: str) -> int | None:
    """Exact minimum energy of the sequence, or None if the chain is too long.

    The enumeration visits every conformation that fits in the lattice, so its
    cost grows exponentially with the length: chains longer than
    ``exact.max_length`` are not checked.
    """
    if len(sequence) > config["exact"]["max_length"]:
        return None
    lattice = config["lattice"]
    return _core.exact_ground_state(sequence, lattice["width"], lattice["height"]).energy


def run(config: dict) -> None:
    """Compares the conformation saved by phase 1 with the exact minimum."""
    lattice = config["lattice"]
    path = config["output"]["native_conformation"]

    sequence, coordinates = load_conformation(path)
    # The energy is computed again from the coordinates instead of being read
    # from the file.
    found = _core.Conformation(sequence, coordinates, lattice["width"], lattice["height"]).energy()

    print(f"Conformation:   {path}")
    print(f"Sequence:       {sequence}")
    print(f"REMC energy:    {found}")

    exact = exact_energy(config, sequence)
    if exact is None:
        print("Exact minimum:  not computed (chain longer than exact.max_length)")
    elif found == exact:
        print(f"Exact minimum:  {exact} -> the REMC found the ground state")
    else:
        print(f"Exact minimum:  {exact} -> the REMC did NOT reach the ground state")

#pragma once

// Canonical form of a conformation, used by the dataset (phase 2).
// See notes/dataset.md.
//
// A chain is described by its turns, which do not change when the chain is
// translated or rotated on the lattice. Of the two mirror images, the
// canonical form keeps the one whose first turn is LEFT: two conformations
// related by a translation, a rotation or a reflection have the same
// canonical form, and so the same key.

#include <cstdint>
#include <string>
#include <vector>

#include "hp/conformation.hpp"

namespace hp {

// Turn of the chain at an inner residue i (1 ... N-2): direction of the bond
// (i, i+1) compared with the direction of the bond (i-1, i). LEFT turns the
// direction (dx, dy) into (-dy, dx), RIGHT into (dy, -dx). A reflection of
// the lattice exchanges LEFT and RIGHT.
enum Turn : int { FORWARD = 0, LEFT = 1, RIGHT = 2 };

// Longest chain whose key fits in 64 bits: N-2 turns in base 3, 3^40 < 2^64.
constexpr int MAX_KEY_LENGTH = 42;

// Direction turned by 90 degrees, as LEFT and RIGHT above.
Coord turn_left(Coord d);
Coord turn_right(Coord d);

// The N-2 turns of the conformation, in chain order.
std::vector<int> turns(const Conformation& conformation);

// Turns of the mirror image: LEFT and RIGHT exchanged.
std::vector<int> mirror_turns(std::vector<int> t);

// The lexicographically smaller of the turns and of their mirror: the first
// turn that is not FORWARD is LEFT.
std::vector<int> canonical_turns(const Conformation& conformation);

// Key of a list of turns: the turns read as a number in base 3, the first one
// as the most significant digit (so the order of the keys is the
// lexicographic order of the turns). Throws std::invalid_argument for chains
// longer than MAX_KEY_LENGTH.
std::uint64_t turns_key(const std::vector<int>& t);

// Inverse of turns_key for a chain of n residues.
std::vector<int> key_turns(std::uint64_t key, int n);

// Key of the canonical form.
std::uint64_t canonical_key(const Conformation& conformation);

// Side of the square lattice on which the canonical states are placed. The
// chain is centred with at least two free nodes on every side, and no move
// takes a residue more than two nodes away from the box that contains the
// chain: the lattice behaves as if it were infinite.
int free_lattice_side(int n);

// Conformation with the given turns on the free lattice: the first bond goes
// to the right and the box that contains the chain is centred. Throws
// std::invalid_argument if the turns do not give a self-avoiding walk.
Conformation conformation_from_turns(const std::string& sequence, const std::vector<int>& t);

// Canonical conformation of `sequence` with the given key.
Conformation conformation_from_key(const std::string& sequence, std::uint64_t key);

}  // namespace hp

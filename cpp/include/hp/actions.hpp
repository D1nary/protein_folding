#pragma once

// Discrete action space of fixed size for the dataset (phase 2): every move
// returned by vshd_moves() or pull_moves() gets a slot, identified by a
// residue and a move type. See notes/dataset.md.
//
// The move types are defined in the frame of the chain, never with absolute
// directions: L and R are turns of 90 degrees with respect to a bond (as in
// canonical.hpp), so a rotation of the lattice does not change the action
// and a reflection only exchanges L and R.

#include <string>
#include <vector>

#include "hp/conformation.hpp"

namespace hp {

enum MoveType : int {
    // VSHD end move of the first or the last residue k, around its chain
    // neighbour a: k goes to a + d turned left, right or backwards, where d is
    // the direction from a to k (forwards is where k already is).
    END_L = 0,
    END_R,
    END_B,
    // VSHD corner move of an inner residue. The pull move of the same residue
    // that ends at once (its fourth corner holds the other chain neighbour)
    // is the same move and gets this slot too.
    CORNER,
    // VSHD crankshaft of the pair (k, k+1), k = 1 ... N-3.
    CRANKSHAFT,
    // Pull move of an inner residue k next to m = k+1 (NEXT) or m = k-1
    // (PREV): k goes to m + d turned left or right, where d is the direction
    // from k to m. The rest of the chain is pulled on the other side.
    PULL_NEXT_L,
    PULL_NEXT_R,
    PULL_PREV_L,
    PULL_PREV_R,
    // Pull move of the first or the last residue k, whose chain neighbour is
    // j: the neighbour goes to A, adjacent to k, and k goes to B, adjacent to
    // A. The first letter is A seen from k (with respect to the direction from
    // j to k), the second is B seen from A (with respect to the direction
    // from k to A); F is forwards. Type = PULL_END_FF + 3 * first + second,
    // with F = 0, L = 1, R = 2 as in hp::Turn.
    PULL_END_FF,
    PULL_END_FL,
    PULL_END_FR,
    PULL_END_LF,
    PULL_END_LL,
    PULL_END_LR,
    PULL_END_RF,
    PULL_END_RL,
    PULL_END_RR,
};

constexpr int N_MOVE_TYPES = 18;

// Name of a move type, as written above (for example "PULL_NEXT_L").
std::string move_type_name(int type);

// Move type of the mirror image: L and R exchanged.
int mirror_move_type(int type);

// A slot of the action space.
struct Action {
    int residue;
    int type;
};

// Number of slots for a chain of n >= 3 residues: 6n + 11.
//
// Each end residue has 12 slots (3 end moves, 9 end pulls), each inner
// residue has CORNER and 4 pull moves, the residues 1 ... N-3 also have
// CRANKSHAFT. The slots are numbered residue by residue, and for every
// residue in the order of MoveType.
int action_count(int n);

// The slots in order: action_table(n)[a] is the slot a.
std::vector<Action> action_table(int n);

// Index of the slot (residue, type), or -1 if that residue has no such type.
int action_index(int n, int residue, int type);

// Permutation of the slots under a reflection of the lattice:
// mirror_permutation(n)[a] is the slot of the mirror image of the action a.
// It is an involution.
std::vector<int> mirror_permutation(int n);

// Slot of a move returned for residue k by vshd_moves() (pull = false) or by
// pull_moves() (pull = true). Throws std::logic_error if the move does not
// match its type.
int classify_move(const Conformation& conformation, int k, const Move& move, bool pull);

// All the feasible moves of the conformation, one per slot: the result has
// action_count(n) elements, and an empty move means that the slot is not
// feasible. Every move returned by vshd_moves() and pull_moves() is in its
// slot; the moves found twice by those functions (a crankshaft from either
// of its residues, a corner move as VSHD and as pull move) are the same
// move. Throws std::logic_error if two different moves fall in one slot.
std::vector<Move> action_moves(const Conformation& conformation);

}  // namespace hp

#pragma once

// HP model on a 2D lattice: sequence, residue positions and energy.

#include <string>
#include <utility>
#include <vector>

namespace hp {

// Lattice node (x, y): x is the column, y the row.
using Coord = std::pair<int, int>;

// A move is the list of the residues that change position, each with its new
// node. Residues are numbered from 0, in chain order.
using Move = std::vector<std::pair<int, Coord>>;

// True if the two nodes are joined by an edge of the lattice.
bool are_adjacent(Coord a, Coord b);

// True if the two nodes are the opposite corners of a unit square.
bool are_diagonal(Coord a, Coord b);

// The four nodes adjacent to `c` (some may be outside the lattice).
std::vector<Coord> neighbours(Coord c);

// A chain of H/P residues placed on a lattice of width x height nodes.
class Conformation {
public:
    // Throws std::invalid_argument if the conformation is not valid: sequence
    // not made only of H and P, one coordinate per residue missing, residues
    // outside the lattice, overlapping or not connected.
    Conformation(std::string sequence, std::vector<Coord> coordinates, int width, int height);

    int size() const { return static_cast<int>(coordinates_.size()); }
    int width() const { return width_; }
    int height() const { return height_; }
    const std::string& sequence() const { return sequence_; }
    const std::vector<Coord>& coordinates() const { return coordinates_; }
    Coord position(int i) const { return coordinates_[i]; }

    bool in_bounds(Coord c) const;
    // Index of the residue on node `c`, or -1 if the node is empty or outside
    // the lattice.
    int residue_at(Coord c) const;
    // True if `c` is inside the lattice and not occupied.
    bool is_free(Coord c) const;

    // Self-avoiding walk inside the lattice: every residue on its own node,
    // consecutive residues on adjacent nodes.
    bool is_valid() const;

    // Energy of the HP model: -1 for every pair of H residues that are
    // adjacent on the lattice but not consecutive in the chain.
    int energy() const;

    // Applies the move and returns the move that undoes it. The move is not
    // checked: it must come from vshd_moves() or pull_moves().
    Move apply(const Move& move);

private:
    int cell(Coord c) const { return c.second * width_ + c.first; }

    std::string sequence_;
    std::vector<Coord> coordinates_;
    int width_;
    int height_;
    // For every node, the index of the residue on it or -1: makes the
    // "is this node free?" question O(1).
    std::vector<int> occupancy_;
};

}  // namespace hp

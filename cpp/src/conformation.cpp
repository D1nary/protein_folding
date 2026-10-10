#include "hp/conformation.hpp"

#include <cstdlib>
#include <stdexcept>

namespace hp {

bool are_adjacent(Coord a, Coord b) {
    return std::abs(a.first - b.first) + std::abs(a.second - b.second) == 1;
}

bool are_diagonal(Coord a, Coord b) {
    return std::abs(a.first - b.first) == 1 && std::abs(a.second - b.second) == 1;
}

std::vector<Coord> neighbours(Coord c) {
    return {
        {c.first + 1, c.second},
        {c.first - 1, c.second},
        {c.first, c.second + 1},
        {c.first, c.second - 1},
    };
}

Conformation::Conformation(std::string sequence, std::vector<Coord> coordinates, int width,
                           int height)
    : sequence_(std::move(sequence)),
      coordinates_(std::move(coordinates)),
      width_(width),
      height_(height) {
    if (width_ <= 0 || height_ <= 0) {
        throw std::invalid_argument("the lattice must have positive width and height");
    }
    if (sequence_.empty()) {
        throw std::invalid_argument("the conformation is empty");
    }
    if (sequence_.size() != coordinates_.size()) {
        throw std::invalid_argument("the sequence and the coordinates have different lengths");
    }
    if (sequence_.find_first_not_of("HP") != std::string::npos) {
        throw std::invalid_argument("the sequence must contain only H and P");
    }

    occupancy_.assign(static_cast<std::size_t>(width_) * height_, -1);
    for (int i = 0; i < size(); ++i) {
        const Coord c = coordinates_[i];
        if (!in_bounds(c)) {
            throw std::invalid_argument("residue " + std::to_string(i) + " is outside the lattice");
        }
        if (occupancy_[cell(c)] != -1) {
            throw std::invalid_argument("residue " + std::to_string(i) +
                                        " is on an occupied node");
        }
        if (i > 0 && !are_adjacent(coordinates_[i - 1], c)) {
            throw std::invalid_argument("residue " + std::to_string(i) +
                                        " is not adjacent to the previous one");
        }
        occupancy_[cell(c)] = i;
    }
}

bool Conformation::in_bounds(Coord c) const {
    return c.first >= 0 && c.first < width_ && c.second >= 0 && c.second < height_;
}

int Conformation::residue_at(Coord c) const {
    return in_bounds(c) ? occupancy_[cell(c)] : -1;
}

bool Conformation::is_free(Coord c) const {
    return in_bounds(c) && occupancy_[cell(c)] == -1;
}

bool Conformation::is_valid() const {
    for (int i = 0; i < size(); ++i) {
        const Coord c = coordinates_[i];
        // Two residues on the same node cannot both be recorded in
        // occupancy_, so this also detects overlaps.
        if (!in_bounds(c) || occupancy_[cell(c)] != i) {
            return false;
        }
        if (i > 0 && !are_adjacent(coordinates_[i - 1], c)) {
            return false;
        }
    }
    return true;
}

int Conformation::energy() const {
    int contacts = 0;
    for (int i = 0; i < size(); ++i) {
        if (sequence_[i] != 'H') {
            continue;
        }
        const Coord c = coordinates_[i];
        // Looking only right and down counts every pair of nodes once.
        for (const Coord n : {Coord{c.first + 1, c.second}, Coord{c.first, c.second + 1}}) {
            const int j = residue_at(n);
            if (j != -1 && sequence_[j] == 'H' && std::abs(i - j) > 1) {
                ++contacts;
            }
        }
    }
    return -contacts;
}

Move Conformation::apply(const Move& move) {
    Move undo;
    undo.reserve(move.size());
    // Two passes: in a pull move a residue lands on the node that another
    // moved residue is leaving, so all the old nodes are freed first.
    for (const auto& [i, target] : move) {
        undo.emplace_back(i, coordinates_[i]);
        occupancy_[cell(coordinates_[i])] = -1;
    }
    for (const auto& [i, target] : move) {
        coordinates_[i] = target;
        occupancy_[cell(target)] = i;
    }
    return undo;
}

}  // namespace hp

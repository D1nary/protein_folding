#include "hp/exact.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

namespace hp {

namespace {

// Smallest rectangle that contains the residues placed so far.
struct Box {
    int min_x, max_x, min_y, max_y;

    int columns() const { return max_x - min_x + 1; }
    int rows() const { return max_y - min_y + 1; }
};

class Search {
public:
    Search(const std::string& sequence, int width, int height)
        : sequence_(sequence),
          n_(static_cast<int>(sequence.size())),
          width_(width),
          height_(height),
          side_(2 * n_ + 1),
          occupancy_(static_cast<std::size_t>(side_) * side_, -1),
          path_(n_),
          max_future_contacts_(n_ + 1, 0) {
        // max_future_contacts_[i]: upper bound on the contacts formed by
        // placing the residues i ... n-1. A residue has 4 adjacent nodes: one
        // is taken by the previous residue and one by the next, so 2 are left
        // for the contacts (3 for the last residue, which has no next).
        for (int i = n_ - 1; i >= 0; --i) {
            const int capacity = sequence_[i] != 'H' ? 0 : (i == n_ - 1 ? 3 : 2);
            max_future_contacts_[i] = max_future_contacts_[i + 1] + capacity;
        }
    }

    void run() {
        // The walk is built on a private grid large enough for any shape,
        // starting from its centre; it is moved into the lattice at the end.
        const Coord start{n_, n_};
        place(0, start);
        if (n_ == 1) {
            visit_leaf(0, Box{n_, n_, n_, n_});
        } else {
            // Rotations: the first bond always goes to the right.
            extend(1, {n_ + 1, n_}, 0, Box{n_, n_, n_, n_}, true);
        }
    }

    bool found() const { return best_contacts_ >= 0; }
    int best_contacts() const { return best_contacts_; }
    long long explored_nodes() const { return explored_nodes_; }

    // Best walk, translated (and transposed if needed) into the lattice.
    std::vector<Coord> best_coordinates() const {
        const bool transpose = !(best_box_.columns() <= width_ && best_box_.rows() <= height_);
        std::vector<Coord> coordinates;
        coordinates.reserve(n_);
        for (const Coord c : best_path_) {
            const int x = c.first - best_box_.min_x;
            const int y = c.second - best_box_.min_y;
            coordinates.push_back(transpose ? Coord{y, x} : Coord{x, y});
        }
        return coordinates;
    }

private:
    int cell(Coord c) const { return c.second * side_ + c.first; }

    void place(int i, Coord c) {
        occupancy_[cell(c)] = i;
        path_[i] = c;
    }

    // The walk can be rotated by 90 degrees, so its box may fit either way.
    bool fits(const Box& box) const {
        return (box.columns() <= width_ && box.rows() <= height_) ||
               (box.columns() <= height_ && box.rows() <= width_);
    }

    // H-H contacts that residue `i` forms on node `c` with the residues
    // already placed (the previous one in the chain does not count).
    int new_contacts(int i, Coord c) const {
        if (sequence_[i] != 'H') {
            return 0;
        }
        int contacts = 0;
        for (const Coord n : neighbours(c)) {
            const int j = occupancy_[cell(n)];
            if (j != -1 && j != i - 1 && sequence_[j] == 'H') {
                ++contacts;
            }
        }
        return contacts;
    }

    void visit_leaf(int contacts, const Box& box) {
        if (contacts > best_contacts_) {
            best_contacts_ = contacts;
            best_path_ = path_;
            best_box_ = box;
        }
    }

    // Puts residue `i` on the free node `c` and tries every continuation.
    // `contacts` counts the contacts among the residues 0 ... i-1, `box`
    // contains them, `straight` is true while they all lie on one line.
    void extend(int i, Coord c, int contacts, Box box, bool straight) {
        box.min_x = std::min(box.min_x, c.first);
        box.max_x = std::max(box.max_x, c.first);
        box.min_y = std::min(box.min_y, c.second);
        box.max_y = std::max(box.max_y, c.second);
        if (!fits(box)) {
            return;
        }

        ++explored_nodes_;
        contacts += new_contacts(i, c);
        if (i == n_ - 1) {
            place(i, c);
            visit_leaf(contacts, box);
            occupancy_[cell(c)] = -1;
            return;
        }
        // Bound: not even the best completion improves on the current best.
        if (contacts + max_future_contacts_[i + 1] <= best_contacts_) {
            return;
        }

        place(i, c);
        for (const Coord next : neighbours(c)) {
            if (occupancy_[cell(next)] != -1) {
                continue;
            }
            // Mirror images: the first turn of the walk always goes up.
            const bool turns = next.second != c.second;
            if (straight && next.second < c.second) {
                continue;
            }
            extend(i + 1, next, contacts, box, straight && !turns);
        }
        occupancy_[cell(c)] = -1;
    }

    const std::string& sequence_;
    int n_;
    int width_;
    int height_;
    int side_;  // side of the private grid
    // For every node of the private grid, the residue on it or -1.
    std::vector<int> occupancy_;
    std::vector<Coord> path_;
    std::vector<int> max_future_contacts_;

    int best_contacts_ = -1;  // -1: no complete walk found yet
    std::vector<Coord> best_path_;
    Box best_box_{};
    long long explored_nodes_ = 0;
};

}  // namespace

ExactResult exact_ground_state(const std::string& sequence, int width, int height) {
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument("the lattice must have positive width and height");
    }
    if (sequence.empty()) {
        throw std::invalid_argument("the sequence is empty");
    }
    if (sequence.find_first_not_of("HP") != std::string::npos) {
        throw std::invalid_argument("the sequence must contain only H and P");
    }

    Search search(sequence, width, height);
    search.run();
    if (!search.found()) {
        throw std::invalid_argument("no conformation of this length fits in the lattice");
    }
    return ExactResult{Conformation(sequence, search.best_coordinates(), width, height),
                       -search.best_contacts(), search.explored_nodes()};
}

}  // namespace hp

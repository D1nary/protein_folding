"""Tests of the parts of the GUI module that do not need a window."""

import pytest

from hpfold.conformation import load_conformation, validate_conformation

# The module imports Tkinter: skip these tests where it is not installed.
pytest.importorskip("tkinter")

from hpfold import gui  # noqa: E402


def test_pixel_to_cell_snaps_to_the_closest_node():
    origin = (60, 60)
    assert gui.pixel_to_cell(60, 60, 60, origin) == (0, 0)
    assert gui.pixel_to_cell(85, 40, 60, origin) == (0, 0)
    assert gui.pixel_to_cell(95, 60, 60, origin) == (1, 0)
    assert gui.pixel_to_cell(5, 5, 60, origin) == (-1, -1)  # outside the grid


def test_pixel_and_cell_conversions_are_inverse():
    for cell in [(0, 0), (3, 5), (7, 7)]:
        pixel = gui.cell_to_pixel(cell, 37.5, (12.5, 80))
        assert gui.pixel_to_cell(*pixel, 37.5, (12.5, 80)) == cell


def test_fit_grid_fills_a_canvas_with_the_same_proportions():
    # 8x8 nodes plus one cell of margin per side: 9 cells in each direction.
    assert gui.fit_grid(540, 540, 8, 8) == (60, (60, 60))
    assert gui.fit_grid(270, 270, 8, 8) == (30, (30, 30))


def test_fit_grid_is_limited_by_the_shorter_side_and_centred():
    cell_size, origin = gui.fit_grid(900, 540, 8, 8)
    assert cell_size == 60
    # The grid spans 7 cells (420 px): the rest is split between the two sides.
    assert origin == (240, 60)

    cell_size, origin = gui.fit_grid(540, 900, 8, 8)
    assert cell_size == 60
    assert origin == (60, 240)


def test_fit_grid_keeps_the_whole_grid_inside_the_canvas():
    cell_size, origin = gui.fit_grid(333, 211, 5, 3)
    right, bottom = gui.cell_to_pixel((4, 2), cell_size, origin)
    assert origin[0] >= cell_size / 2 and origin[1] >= cell_size / 2
    assert right <= 333 - cell_size / 2 and bottom <= 211 - cell_size / 2


def make_config(tmp_path, sequence="HPHPPH", length=6):
    return {
        "lattice": {"width": 4, "height": 4},
        "protein": {
            "length": length,
            "sequence": sequence,
            "initial_conformation": str(tmp_path / "initial.json"),
        },
    }


def test_write_default_conformation(tmp_path):
    path = gui.write_default_conformation(make_config(tmp_path))

    sequence, coordinates = load_conformation(path)
    assert sequence == "HPHPPH"
    validate_conformation(sequence, coordinates, 4, 4)


def test_write_default_conformation_checks_the_length(tmp_path):
    config = make_config(tmp_path, length=7)
    with pytest.raises(ValueError):
        gui.write_default_conformation(config)
    assert not (tmp_path / "initial.json").exists()

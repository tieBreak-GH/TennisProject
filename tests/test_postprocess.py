import pytest

from postprocess import line_intersection


def test_line_intersection_diagonals_of_unit_square():
    px, py = line_intersection((0, 0, 2, 2), (0, 2, 2, 0))
    assert px == pytest.approx(1.0)
    assert py == pytest.approx(1.0)


def test_line_intersection_perpendicular_lines():
    px, py = line_intersection((0, 0, 4, 0), (2, -2, 2, 2))
    assert px == pytest.approx(2.0)
    assert py == pytest.approx(0.0)


def test_line_intersection_parallel_lines_return_none():
    assert line_intersection((0, 0, 2, 0), (0, 1, 2, 1)) is None


def test_line_intersection_coincident_lines_return_none():
    # denom is 0 for coincident lines too (they're "parallel" to themselves)
    assert line_intersection((0, 0, 2, 0), (1, 0, 3, 0)) is None

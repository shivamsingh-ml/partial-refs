"""V2V2R mask conversion (synthetic arrays only)."""

import numpy as np
import pytest

from partialref.baseline.v2v2r_masks import convert_labels, dice, to_tf4_grid


def test_labels_map_to_v2v2r_scheme_with_and_without_mirror():
    L = np.array([[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 26, 36, 46, 116, 146, 103, 104, 105]]], np.int16)
    plain = convert_labels(L, mirror=False).ravel().tolist()
    assert plain == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 26, 36, 46, 50, 50, 51, 52, 53]
    mir = convert_labels(L, mirror=True).ravel().tolist()
    assert mir == [0, 1, 2, 4, 3, 6, 5, 7, 8, 9, 10, 26, 16, 46, 36, 50, 50, 52, 51, 53]
    with pytest.raises(ValueError):
        convert_labels(np.array([[[0, 77]]], np.int16), mirror=False)


def test_grid_flip_and_dice():
    a = np.arange(24).reshape(2, 3, 4)
    f = to_tf4_grid(a)
    assert f[0, 0, 0] == a[0, 2, 3] and np.array_equal(to_tf4_grid(f), a)
    x = np.zeros((4, 4, 4), np.uint8)
    y = x.copy()
    x[:2] = 36
    y[:1] = 36
    y[3] = 46
    d = dice(x, y)
    assert abs(d[36] - 2 * 16 / (32 + 16)) < 1e-12 and d[46] == 0

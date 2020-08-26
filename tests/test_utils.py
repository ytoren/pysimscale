import pytest
from importlib.util import find_spec
from numpy import array, nan, array_equal

HAS_PANDAS = False

if find_spec('pandas') is not None:
    HAS_PANDAS = True


def test_series2array2D():
    l = [[1, 2.2, 3], [1, 2.2, 3], None, [1, 2.2, 3]]
    a = array([[1, 2.2, 3], [1, 2.2, 3], [nan, nan, nan], [1, 2.2, 3]])

    if HAS_PANDAS:
        from pysimscale import series2array2D
        from pandas import Series

        s2a = series2array2D(Series(l))

        assert array_equal(s2a, a, equal_nan=True)

    else:
        print('Could not find a Pandas instalation, skipping test')

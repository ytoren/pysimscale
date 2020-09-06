import pytest
from importlib.util import find_spec
from numpy import array, nan, array_equal
from pysimscale import is_permutation, is_partition, sort_partition, series2array2D, id_block_matrix

def test_is_permutation():
    p = [9,4,5,7,2,6,3,8,0,1]
    assert is_permutation(p)

def test_not_permutation1():
    p = [9,4,7,2,6,3,8,0,1]
    assert not is_permutation(p)

def test_not_permutation2():
    p = [9,10,5,7,2,6,3,8,0,1]
    assert not is_permutation(p)

def test_not_permutation3():
    p = [9,10,5,7,2,6,5,8,0,1]
    assert not is_permutation(p)


def test_is_partition():
    p = [[0], [1,2,3], [4,5]]
    assert is_partition(p)

def test_is_partition():
    p = [[10], [11,12,13], [14,15]]
    assert is_partition(p, start=10)

def test_not_partition():
    p = [[10], [11,12,13], [14,15]]
    assert not is_partition(p, start=10, end=16)

def test_not_partition1():
    p = [[0], [1,2], [4,5]]
    assert not is_partition(p)

def test_not_partition2():
    p = [[0], [1,2,3], [4,6]]
    assert not is_partition(p)


def test_sorted_partition():
    p = [[0], [1,2,3], [4,5]]
    assert p == sort_partition(p)

def test_sorted_partition_reverse():
    p = [[4,5], [1,2,3], [0]]
    assert p == sort_partition(p, by=min, reverse=True)

def test_unsorted_partition1():
    p = [[1,2,3], [0], [4,5]]
    assert not p == sort_partition(p)

def test_unsorted_partition1():
    p = [[1,2,3], [0], [4,5]]
    assert not p == sort_partition(p, by=min)


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


def test_id_block_wrong_blocks():
    with pytest.raises(ValueError):
         id_block_matrix([2,1,2])

def test_id_block_matrix_ones():
    assert array_equal(
        id_block_matrix([1,2,2,3]).todense(),
        array([[1, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 1]])
    )

def test_block_matrix_2():
    assert array_equal(
        id_block_matrix([1,1,2], value=2).todense(),
        array([[2,2,0], [2,2,0], [0,0,2]])
    )

def test_block_matrix_2_diag0():
    assert array_equal(
        id_block_matrix([1,1,2], value=2, diag_value=0).todense(),
        array([[0,2,0], [2,0,0], [0,0,0]])
    )

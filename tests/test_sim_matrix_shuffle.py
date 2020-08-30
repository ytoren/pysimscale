from numpy import allclose, array
from pysimscale import row_shuffle_matrix

permutation = [0, 5, 2, 3, 4, 1]

def test_row_shuffle_matrix():
    expected_shuffle_matrix = array([
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0]
    ])

    assert allclose(row_shuffle_matrix(permutation).toarray(), expected_shuffle_matrix)

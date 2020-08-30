from numpy import allclose, array
from scipy.sparse import csr_matrix, issparse
from pysimscale import row_shuffle_matrix, sim_matrix_shuffle

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

def test_sparse_output():
    m = csr_matrix(array([
        [1.0, 0.9, 0.0],
        [0.9, 1.0, 0.6],
        [0.0, 0.6, 1.0],
    ]))

    assert issparse(sim_matrix_shuffle(m, row_order=[1,0,2]))


def test_same_order():
    m = array([
        [1.0, 0.9, 0.0],
        [0.9, 1.0, 0.6],
        [0.0, 0.6, 1.0],
    ])

    assert allclose(sim_matrix_shuffle(m, row_order=[0,1,2]), m)


def test_sim_matrix_shuffle():
    m = csr_matrix(array([
        [1.0, 0.9, 0.0, 0.2, 0.0, 0.1],
        [0.9, 1.0, 0.6, 0.0, 0.0, 0.0],
        [0.0, 0.6, 1.0, 0.0, 0.5, 0.0],
        [0.2, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.0, 1.0, 0.8],
        [0.1, 0.0, 0.0, 0.0, 0.8, 1.0]
    ]))

    expected = array([
        [1.0, 0.1, 0.0, 0.2, 0.0, 0.9],
        [0.1, 1.0, 0.0, 0.0, 0.8, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.5, 0.6],
        [0.2, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.8, 0.5, 0.0, 1.0, 0.0],
        [0.9, 0.0, 0.6, 0.0, 0.0, 1.0]
    ])

    m_shuffled = sim_matrix_shuffle(m, row_order=permutation)

    assert allclose(m_shuffled.todense(), expected)

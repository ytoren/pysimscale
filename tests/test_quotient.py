from numpy import allclose, array
from scipy.sparse import csr_matrix, issparse
from pysimscale import merge_row_partition, quotient_similarity, sort_partition

partition = [[1, 2, 3], [4, 5], [0]]

m = csr_matrix(array([
    [1.0, 0.9, 0.0, 0.2, 0.0, 0.1],
    [0.9, 1.0, 0.6, 0.0, 0.0, 0.0],
    [0.0, 0.6, 1.0, 0.0, 0.5, 0.0],
    [0.2, 0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.5, 0.0, 1.0, 0.8],
    [0.1, 0.0, 0.0, 0.0, 0.8, 1.0]
]))

m_merge_rows = array([
    [1.0, 0.9, 0.0, 0.2, 0.0, 0.1],
    [1.1, 1.6, 1.6, 1.0, 0.5, 0.0],
    [0.1, 0.0, 0.5, 0.0, 1.8, 1.8]
])

m_merged = array([
    [1.0, 1.1, 0.1],
    [1.1, 4.2, 0.5],
    [0.1, 0.5, 3.6]
])

def test_row_merge():
    assert allclose(merge_row_partition(m, sort_partition(partition), agg='sum').todense(), m_merge_rows)

def test_full_merge():
    assert allclose(quotient_similarity(m, partition, agg='sum').todense(), m_merged)

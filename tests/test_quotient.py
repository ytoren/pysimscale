import pytest
from importlib.util import find_spec
from numpy import allclose, array
from scipy.sparse import csr_matrix, issparse
from pysimscale import merge_row_partition, quotient_similarity

if find_spec('networkx') is not None:
    HAS_NX = True

if HAS_NX:
    from networkx import from_scipy_sparse_matrix, quotient_graph, to_numpy_array, Graph

m = csr_matrix(array([
    [1.0, 0.9, 0.0, 0.2, 0.0, 0.1],
    [0.9, 1.0, 0.6, 0.0, 0.0, 0.0],
    [0.0, 0.6, 1.0, 0.0, 0.5, 0.0],
    [0.2, 0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.5, 0.0, 1.0, 0.8],
    [0.1, 0.0, 0.0, 0.0, 0.8, 1.0]
]))

partition = [[2], [1, 0, 3], [4, 5]]

m_merge_rows = array([
    [0.0, 0.6, 1.0, 0.0, 0.5, 0.0],
    [2.1, 1.9, 0.6, 1.2, 0.0, 0.1],
    [0.1, 0.0, 0.5, 0.0, 1.8, 1.8]
])

m_merged = array([
    [1.0, 0.6, 0.5],
    [0.6, 5.2, 0.1],
    [0.5, 0.1, 3.6]
])

custom_agg = lambda m,p: csr_matrix(getattr(m[p, :], 'sum')(axis=0))


def test_row_merge():
    assert allclose(merge_row_partition(m, partition, custom_agg , 1).todense(), m_merge_rows)

def test_row_merge_parallel():
    assert allclose(
        merge_row_partition(m, partition, lambda m,p: csr_matrix(getattr(m[p, :], 'sum')(axis=0)), -1).todense(),
        m_merge_rows
    )


def test_quotient_wrong_agg_str():
    with pytest.raises(ValueError):
        quotient_similarity(m, partition, agg='ABCD', n_cpu=1)

def test_quotient_wrong_agg_else():
    with pytest.raises(ValueError):
        quotient_similarity(m, partition, agg=array([]), n_cpu=1)


def test_quotient_loop():
    assert allclose(quotient_similarity(m, partition, agg='sum', n_cpu=1).todense(), m_merged)

def test_quotient_parallel():
    assert allclose(quotient_similarity(m, partition, agg='sum', n_cpu=-1).todense(), m_merged)

def test_quotient_custom():
    assert allclose(quotient_similarity(m, partition, agg=custom_agg, n_cpu=-1).todense(), m_merged)

def test_quotient_bad_partition():
    with pytest.raises(ValueError):
        quotient_similarity(m, partition=[[0,2], [3,7]], check=True)


if HAS_NX:
    G = from_scipy_sparse_matrix(m)

    def edge_sum(u, v):
        Guv = Graph(G.subgraph(list(u) + list(v)))
        Guv.remove_edges_from(G.subgraph(list(u)).edges)
        Guv.remove_edges_from(G.subgraph(list(v)).edges)
        return {'weight': Guv.size(weight='weight')}

    def node_sum(u):
        return {'weight': G.subgraph(list(u)).size(weight='weight')}

    def test_nx():
        mq1 = quotient_similarity(m, partition, agg='sum', diag_value=0).todense()
        Gq = quotient_graph(G=G, partition=partition, edge_data=edge_sum)
        mq2 = to_numpy_array(Gq)
        assert allclose(mq1, mq2)

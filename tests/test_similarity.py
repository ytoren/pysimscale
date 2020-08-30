import pytest
from importlib.util import find_spec

from numpy import array, allclose
from scipy.sparse import coo_matrix, issparse
from pysimscale import truncated_sparse_similarity

HAS_JOBLIB = False
if find_spec('joblib') is not None:
    HAS_JOBLIB = True

a1 = array([
    [1, 1, 1, 1],
    [0, 1, 0, 2],
    [2.2, 2, 2.2, 0.5]
])
expected_cosine_sim = array([
    [1.0, 0.6708204, 0.9243651],
    [0.6708204, 1.0, 0.3594684],
    [0.9243651, 0.3594684, 1]
])
# coo_matrix(expected_cosine_sim)

def test_sim_wrong_metric():
    with pytest.raises(ValueError):
         truncated_sparse_similarity(a1, metric=None, n_jobs=1)

def test_cosine_no_thresh():
    assert allclose(
        truncated_sparse_similarity(a1, metric='cosine', thresh=0, diag_value=None, n_jobs=1).todense(),
        expected_cosine_sim
    )

def test_cosine_block_no_thresh():
    assert allclose(
        truncated_sparse_similarity(a1, metric='cosine', block_size = 2, thresh=0, diag_value=None, n_jobs=1).todense(),
        expected_cosine_sim
    )

def test_cosine_parallel():
    if HAS_JOBLIB:
        assert allclose(
            truncated_sparse_similarity(a1, metric='cosine', thresh=0, diag_value=None, n_jobs=-1).todense(),
            expected_cosine_sim
        )
    else:
        print('Could not find a Joblib instalation, skipping test')

def test_sim_sparsity():
    assert issparse(truncated_sparse_similarity(a1, metric='cosine', thresh=0.9, diag_value=0, n_jobs=1))

def test_cosine_thresh():
    sim_thresh = 0.9
    expected = expected_cosine_sim.copy()
    expected[expected < sim_thresh] = 0

    assert allclose(
        truncated_sparse_similarity(a1, metric='cosine', thresh=sim_thresh, diag_value=None, n_jobs=1).todense(),
        expected
    )


a2 = array([[1, 1, 1, 1], [0, 1, 0, 1], [0, 0, 0, 1]])
expected_hamming_sim = array([[0.0, 0.5, 0.25], [0.5, 0.0, 0.75], [0.25, 0.75, 0.0]])

def test_cosine_no_thresh_dense():
    assert allclose(
        truncated_sparse_similarity(a2, metric='hamming', thresh=0, diag_value=0, n_jobs=1).todense(),
        expected_hamming_sim
    )

def test_cosine_thresh_dense():
    sim_thresh = 0.75
    expected = expected_hamming_sim.copy()
    expected[expected < sim_thresh] = 0

    if HAS_JOBLIB:
        n_cpu = -1
    else:
        n_cpu = 1

    assert allclose(
        truncated_sparse_similarity(a2, metric='hamming', thresh=sim_thresh, diag_value=0, n_jobs=n_cpu).todense(),
        expected
    )

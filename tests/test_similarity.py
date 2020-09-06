import pytest
from importlib.util import find_spec

from numpy import array, allclose, matmul
from numpy.linalg import norm
from scipy.sparse import coo_matrix, issparse
from pysimscale import truncated_sparse_similarity, similarity_sparse_block

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

a2 = array([[1, 1, 1, 1], [0, 1, 0, 1], [0, 0, 0, 1]])
expected_hamming_sim = array([[0.0, 0.5, 0.25], [0.5, 0.0, 0.75], [0.25, 0.75, 0.0]])

def custom_cosine_matrix_similarity(X, Y):
    X = X / norm(X, ord=2, axis=1).reshape(X.shape[0], 1)
    Y = Y / norm(Y, ord=2, axis=1).reshape(Y.shape[0], 1)
    return matmul(X, Y.T)


def test_local_block_cosine():
    assert allclose(
        similarity_sparse_block(a1, [0,1], thresh=0, metric='cosine').todense(),
        expected_cosine_sim[[0,1], :]
    )

def test_local_block_hamming():
    assert allclose(
        similarity_sparse_block(a2, [0,1], thresh=0, metric='hamming').todense(),
        array([[1.0, 0.5, 0.25], [0.5, 1.0, 0.75]])
    )

def test_local_block_custom():
    assert allclose(
        similarity_sparse_block(a1, [0,1], thresh=0, metric=custom_cosine_matrix_similarity).todense(),
        expected_cosine_sim[[0,1], :]
    )

def test_local_block_cosine_thresh():
    e = expected_cosine_sim[[0,1], :]
    e[e < 0.9] = 0
    assert allclose(similarity_sparse_block(a1, [0,1], thresh=0.9, metric='cosine').todense(), e)

def test_local_block_cosine_thresh_binary():
    e = expected_cosine_sim[[0,1], :]
    e = e >= 0.9
    assert allclose(similarity_sparse_block(a1, [0,1], thresh=0.9, binary=True, metric='cosine').todense(), e)


def test_sim_wrong_metric():
    with pytest.raises(ValueError):
         truncated_sparse_similarity(a1, metric=None, n_jobs=1)

def test_cosine_no_thresh():
    assert allclose(
        truncated_sparse_similarity(a1, metric='cosine', thresh=0, diag_value=None, n_jobs=1).todense(),
        expected_cosine_sim
    )

def test_cosine_convertable_dtype():
    assert allclose(
        truncated_sparse_similarity(a1.astype('object'), metric='cosine', thresh=0, diag_value=None, n_jobs=1).todense(),
        expected_cosine_sim
    )

def test_cosine_bad_dtype():
    with pytest.raises(TypeError):
        truncated_sparse_similarity(array([lambda x: x, lambda y: y]))

def test_cosine_block_no_thresh():
    assert allclose(
        truncated_sparse_similarity(a1, metric='cosine', block_size=2, thresh=0, diag_value=None, n_jobs=1).todense(),
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

def test_hamming_no_thresh_dense():
    assert allclose(
        truncated_sparse_similarity(a2, metric='hamming', thresh=0, diag_value=0, n_jobs=1).todense(),
        expected_hamming_sim
    )

def test_hamming_thresh_dense():
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

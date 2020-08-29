__all__ = ['truncated_sparse_similarity']

from scipy.sparse import coo_matrix, vstack
from numpy import matmul, isnan, clip
from numpy.linalg import norm
from math import ceil

from importlib.util import find_spec

if find_spec('joblib') is not None:
    from joblib import Parallel, delayed, cpu_count
    DEFAULT_CPUS = -1
else:
    print('Could not find `joblib` library. Parallelisation is disabled by default')
    DEFAULT_CPUS = 1

def similarity_sparse_block_(a, ind_range, thresh, metric='hamming', lower=None, binary=False, sparse=True, normalized=True):
    '''Calculate a Hamming similarity matrix (1 - distance) for a subset of indices (against the entire dataset).

    params:
    - a: A 2D Numpy array, each row representing an embedding vector
    - ind_range: A list of integers representing the subset of chosen indices
    - metric: A string with the name of a built in metric (currently `cosine` and `hamming` are supported) or a function that takes two matrices and returns row-wise distnaces
    - thresh: a lower threshold for similarity. Values below wll be set to 0. Default is None (no filtering)
    - binary: Should the result be a 1/0 matrix (is the similarity above or below threshold) or return the actual similarities. Default is `False`
    - sparse: Should the returned matrix be a Scipy COO matrix. Default is True
    - normalized: Are the rows of the array normalized? (used only to decide if we normalize when calculating cosine similarity)
    '''
    if metric == 'hamming':
        m = (1.0 * matmul(a[ind_range], a.T) + matmul((1 - a[ind_range]), (1 - a).T)) / a.shape[1]
    elif metric == 'cosine':
        a = a / norm(a, ord=2, axis=1).reshape(a.shape[0], 1)
        m = matmul(a[ind_range], a.T)
    elif callable(metric):
        m = metric(a[ind_range])
    else:
        raise ValueError('Invalid value of `metric` parameter. Please use one of the built-in options of specify a function (see documentation)')


    if thresh is not None:
        if binary:
            m = (m >= thresh).astype(int)
        else:
            m[m < thresh] = 0

    if sparse:
        m = coo_matrix(m)

    return m


def truncated_sparse_similarity(a, metric='hamming', block_size=1, thresh=0.9, diag_value=0, binary=False, n_jobs=DEFAULT_CPUS, dtype_fallback='int64'):
    '''Calculate similarity measures between rows of a 2D Numpy array or a Pandas series of lists

    Params:
    - a: A `numpy` matrix / array with one of the following types: `boolean`, `int32/64`, `float32/64`. All rows must have the same number of elements (you can use `simscale.util.allign2Darray` to ensure that)
    - metric: A string with the name of a built in metric (currently `cosine` and `hamming` are supported) or a function that takes two matrices and returns row-wise distnaces
    - block_size: An integer. Maximal number of rows per block when breaking down the similarity calculation int o components. Default is 1, which means we calculate similarity one row at a time (against all other rows).
    - thresh: a lower threshold for similarity. Values under threshold are set to 0. Default is 0.9
    - diag_value: What value should be assigned to the diagonal (`None` means no assignment). Default is 0
    - filter_nan: Should `nan` values be converted to 0. Default is True
    - binary: Should the result be a 1/0 matrix (is the similarity above or belowe threshold) or a matrix with the actual similarities. Default is False
    - n_jobs: Number of jobs to be passed to `joblib`.
              Default depends on whether `joblib` is installed:
              * Installed: Default is -1,  which means `cpu_count() - 1`
              * Not installed: Default is 1, which means simple python loops.
              * You can force a number at your own risk
    - dtype_fallcack: if the array's `dtype` is not `boolean`, `int32/64`, `float32/64` then the function will try and convert the array to this type. Defaults to `float64` which should cover most cases (but is not very memory efficient)
    '''
    if a.dtype not in ('bool', 'int32', 'int64', 'float32', 'float64'):
        try:
            a = a.astype(dtype_fallback)
        except ValueError:
            raise TypeError('Supported data types are `boolean`, `int32`, `int64`, `float32`, `float64`. Do all of your lines have the same number of items? Maybe there is `None` hiding somewhere? Try using `simscale.util.allign2Darray`')

    l = list(range(a.shape[0]))
    blocks = [l[i:(i + block_size)] for i in range(0, a.shape[0], block_size)]

    if n_jobs == 1:
        sim = [similarity_sparse_block_(a=a, ind_range=b, metric=metric, thresh=thresh, binary=binary) for b in blocks]
    else:
        with Parallel(n_jobs=n_jobs) as p:
            f = delayed(similarity_sparse_block_)
            sim = p(f(a=a, ind_range=b, metric=metric, thresh=thresh, binary=binary) for b in blocks)

    sim = vstack(sim)

    if diag_value is not None:
        sim.setdiag(diag_value)

    sim.eliminate_zeros()

    return sim

from pysimscale import is_partition
from scipy.sparse import vstack, csr_matrix
from importlib.util import find_spec

if find_spec('joblib') is not None:
    from joblib import Parallel, delayed
    DEFAULT_CPUS = -1
else:
    print('Could not find `joblib` library. Parallelisation is disabled by default')
    DEFAULT_CPUS = 1

MATRIX_METHOD_STR = ('sum', 'min', 'max', 'mean', 'getnnz')
MATRIX_METHOD_STR_ERR = 'Unknown string for aggregation method. Please pick one of (' + ','.join(MATRIX_METHOD_STR) + ') or specify a function'


def merge_row_partition(m, partition, agg, n_cpu):
    '''Merge rows in a symmetric similarity matrix according to a partition

    Params:
    - m: A symmetric 2D array (can be sparse) containing the similarity matrix
    - partition: A list-of-lists partitionning the rows of `m`
    - agg: a function that takes as parameters the matrix and a range of indices from the partition and aggregates the values across the rows

    Returns a similarity matrix reduced to the dimension induced by the partition
    '''
    if n_cpu == 1:
        m_merged = [agg(m, p) for p in partition]
    else:
        with Parallel(n_jobs=n_cpu) as p:
            m_merged = vstack(p(delayed(agg)(m, p) for p in partition))

    return(vstack(m_merged))


def quotient_similarity(m, partition, agg='sum', diag_value=None, check=False, n_cpu=DEFAULT_CPUS):
    '''Generate quotient similarity matrix based on the given partition of matrix rows

    Params:
    - m: A symmetric 2D array (can be sparse) containing the similarity matrix
    - partition: A list-of-lists partitionning the rows/columns of `m`
    - agg: One of ('sum', 'min'. 'max', 'mean', 'getnnz') or a function that takes as parameters the matrix and a range of indices from the partition and aggregates the values across the rows
    - check: Logical. Should the dunction check that `partition` is a valid partition of the rows of m? Default is 'True'

    Returns a similarity matrix reduced to the dimension induced by the partition
    '''

    if callable(agg):
        f_agg = agg
    elif isinstance(agg, str):
        if agg in MATRIX_METHOD_STR:
            f_agg = lambda m,p: csr_matrix(getattr(m[p, :], agg)(axis=0))
        else:
            raise ValueError(MATRIX_METHOD_STR_ERR)
    else:
        raise ValueError(MATRIX_METHOD_STR_ERR)

    if check:
        if not is_partition(partition, start=0, end=m.shape[0]-1):
            raise ValueError('Please provide a proper partition')

    result = merge_row_partition(merge_row_partition(m, partition, f_agg, n_cpu).T, partition, f_agg, n_cpu)

    if diag_value is not None:
        result.setdiag(diag_value)

    result.eliminate_zeros()

    return result

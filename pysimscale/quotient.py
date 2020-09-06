from pysimscale import is_partition
from scipy.sparse import vstack, csr_matrix
from importlib.util import find_spec

if find_spec('joblib') is not None:
    from joblib import Parallel, delayed
    DEFAULT_CPUS = -1
else:
    print('Could not find `joblib` library. Parallelisation is disabled by default')
    DEFAULT_CPUS = 1

CLASS_STR = ('sum', 'min', 'max', 'mean', 'getnnz')


def merge_row_partition(m, partition, agg, n_cpu):
    '''Merge rows in a symmetric similarity matrix according to a partition

    Params:
    - m: A symmetric 2D array (can be sparse) containing the similarity matrix
    - partition: A list-of-lists partitionning the rows of `m`
    - agg: One of ('sum', 'min'. 'max', 'mean', 'getnnz') or a function that takes as parameters the matrix and a range of indices from the partition and aggregates the values across the rows
    '''
    if isinstance(agg, str):
        if agg in CLASS_STR:
            def csr_agg(m, p):
                return csr_matrix(getattr(m[p, :], agg)(axis=0))
            f_agg = csr_agg
        elif callble(agg):
            f_agg = agg
        else:
            raise ValueError('Unknown string for aggregation method. Please pick one of (' + ','.join(CLASS_STR) + ') or specify a function')

    if n_cpu == 1:
        m_merged = [f_agg(m, p) for p in partition]
    else:
        with Parallel(n_jobs=n_cpu) as p:
            m_merged = vstack(p(delayed(f_agg)(m, p) for p in partition))

    return(vstack(m_merged))


def quotient_similarity(m, partition, agg='sum', diag_value=None, check=False, n_cpu=DEFAULT_CPUS):
    '''Generate quotient similarity matrix based on the given partition of matrix rows

    Params:
    - m: A symmetric 2D array (can be sparse) containing the similarity matrix
    - partition: A list-of-lists partitionning the rows/columns of `m`
    - agg: One of ('sum', 'min'. 'max', 'mean', 'getnnz') or a function that takes as parameters the matrix and a range of indices from the partition and aggregates the values across the rows
    - check: Logical. Should the dunction check that `partition` is a valid partition of the rows of m? Default is 'True'

    '''
    if not callable(agg) and (isinstance(agg, str) and not agg in CLASS_STR):
        raise ValueError('''Agg has to be one of ('sum', 'min'. 'max', 'mean', 'getnnz') or a function''')

    if check:
        if not is_partition(partition, start=0, end=m.shape[0]-1):
            raise ValueError('Please provide a proper partition')

    result = merge_row_partition(merge_row_partition(m, partition, agg, n_cpu).T, partition, agg, n_cpu)

    if diag_value is not None:
        result.setdiag(diag_value)

    result.eliminate_zeros()

    return result

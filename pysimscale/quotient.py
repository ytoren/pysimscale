from pysimscale import sort_partition, is_partition
from scipy.sparse import vstack, csr_matrix

def merge_row_partition(m, partition, agg='sum', diag_value=None, sort=True, check=True):
    '''Merge rows in a symmetric similarity matrix according to a partition

    Params:
    - m: A symmetric 2D array (can be sparse) containing the similarity matrix
    - partition: A list-of-lists partitionning the rows of `m`
    - agg: One of ('sum', 'min'. 'max', 'mean', 'getnnz') or a function that takes as parameters the matrix and a range of indices from the partition and aggregates the values across the rows
    - sort: Logical. Should the partition be sorted? Default is `True`
    - check: Logical. Should the dunction check that `partition` is a valid partition of the rows of m? Default is 'True'

    '''
    CLASS_STR = ('sum', 'min', 'max', 'mean', 'getnnz')

    if not callable(agg) and (isinstance(agg, str) and not agg in CLASS_STR):
        raise ValueError('''Agg has to be one of ('sum', 'min'. 'max', 'mean', 'getnnz') or a function''')

    if sort:
        partition = sort_partition(partition, min)

    if check:
        if not is_partition(partition, start=0, end=m.shape[0]-1):
            raise ValueError('Please provide a proper partition')

    if isinstance(agg, str) and agg in CLASS_STR:
        result = vstack([csr_matrix(getattr(m[p, :], agg)(axis=0)) for p in partition])
    elif callable(agg):
        result = vstack([agg(m, p) for p in partition])

    if diag_value is not None:
        result.setdiag(diag_value)

    result.eliminate_zeros()

    return result

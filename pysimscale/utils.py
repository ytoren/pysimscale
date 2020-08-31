from importlib.util import find_spec
from numpy import stack, nan, unique, full
from scipy.sparse import block_diag

def is_permutation(p):
    '''Check that the integer vector `p` is a permutation of {min(p) ... max(p)}'''
    if len(p) == max(p) - min(p) + 1:
        if len(set(range(min(p), max(p) + 1, 1)).difference(set(p))) == 0:
            return True
        else:
            return False
    else:
        return False


def is_partition(p, start=None, end=None):
    '''Check that a "list of lists" `p` is a partition of the index range it applies to'''
    p_flat = sum(p, [])
    if start is None:
        start = min(p_flat)
    if end is None:
        end = max(p_flat)

    if len(p_flat) == end - start + 1:
        p_range = set(range(start, end + 1, 1))
        if len(p_range.difference(set(p_flat))) == 0:
            if all([is_permutation(pi) for pi in p]):
                return True
            else:
                return False
        else:
            return False
    else:
        return False


def sort_partition(p, by=min, reverse=False):
    '''Sort a partition `p`

    Params:
    - p: list of lists of integers, representing a partition of the matching range
    - by: Function. Maps a list to a single value (real/integer) which is used to sort the lists in the partition
    - reverse: Direction of the sort (passed to `sorted`). Default is `False`
    '''
    return [y for x,y in sorted(zip(map(by, p), p), reverse=reverse)]


def series2array2D(s, none_treament='row', width=None, replicate=False):
    '''Convert a Pandas series containing arrays or list (and possibly rows with a single `None` value) into a "flat" 2D Numpy array

    params:
    - s: A Pandas series
    - none_treament: String. Possible values are `row` (convert a single `None` into a row of Numpy `nan`), `remove` (remove row completely)
    - width: Int. Width of the 2D array. Defaults to `None` with will take the first non-`None` value with length > 1.
    - replicate: Should rows with a single value be replicated to match `width`? Default to `False`
    '''
    a = s.copy().values

    # Find array width
    if width is None:
        width = 0
        i = 1

        while width == 0 and i <= a.shape[0]:
            if a[i] is not None and len(a[i]) > 1:
                width = len(a[i])
            i += 1

        if width == 0:
            raise ValueError('no vectors found')

    for i in range(a.shape[0]):
        if a[i] is None or len(a[i]) == 0:
            a[i] = [nan] * width
        elif len(a[i]) == 1 and replicate:
            a[i] = a[i] * width

    return stack(a)


def id_block_matrix(ids, value=1, diag_value=None):
    '''Generate a block matrix of 1's based on a sorted list of IDs

    Params:
    - ids: A sorted list of IDs. Repeated ID's will generate a block of 1's in the matrix
    - diag_value: What value should be assigned to the diagonal (None means not assignment). Default is 0

    Usage: Add information from a higher hierarcy to the similarity matrix.
    '''
    if ids != sorted(ids):
        raise ValueError('List must ne sorted to ensure additivity works')

    l_unique, l_counts = unique(ids, return_counts=True)
    block_list = []
    n_range = range(len(l_unique))

    for i in n_range:
        n_i = l_counts[i]
        block_list.append(full((n_i, n_i), value))

    if l_counts.tolist() != list(map(lambda x: x.shape[0], block_list)):
        raise ValueError('Counts do not match block sizes')

    result = block_diag(block_list)

    if diag_value is not None:
        result.setdiag(diag_value)
        result.eliminate_zeros()

    return block_diag(block_list)

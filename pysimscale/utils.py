from importlib.util import find_spec
from numpy import stack, nan, unique, ones
from scipy.sparse import block_diag

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


def block_matrix(l, diag_value=0):
    '''Generate a block matrix of 1's based on a sorted list of IDs

    Params:
    - l: A sorted list of IDs. Repeated ID's will generate a block of 1's in the matrix
    - diag_value: What value should be assigned to the diagonal (None means not assignment). Default is 0

    Usage: Add information from a higher hierarcy to the similarity matrix.
    '''
    if l != sorted(l):
        raise ValueError('List must ne sorted to ensure additivity works')

    l_unique, l_counts = unique(l, return_counts=True)
    block_list = []
    n_range = range(len(l_unique))

    for i in n_range:
        n_i = l_counts[i]
        block_list.append(ones((n_i, n_i)))

    if l_counts.tolist() != list(map(lambda x: x.shape[0], block_list)):
        raise ValueError('Counts do not match block sizes')

    result = block_diag(block_list)
    if diag_value is not None:
        result.setdiag(diag_value)
        result.eliminate_zeros()

    return block_diag(block_list)

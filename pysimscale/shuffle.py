from scipy.sparse import coo_matrix
from pysimscale import is_permutation


def row_shuffle_matrix(permutation):
    '''Permutation matrix:

    Params:
    - permutation: A list of integers representing a permutation of permutation. Must contain all values in `range(max(permutation))`

    Returns a sparse CSR row/column permutation matrix based on a permutation of row indices
    '''
    return coo_matrix(([1] * len(permutation), (sorted(permutation), permutation)))


def sim_matrix_shuffle(m, row_order, check=True):
    '''Re-arrange a similarity matrix based on a new row order

    Params:
    - m: Similarity matrix. Can be dense (Numpy array / matrix) or sparse (as long as it supports arithmetic operations)
    - row_order: A list of integers, which makes a permutation of the matrix rows
    - check: Should the permutation be checked before creating the matrix (can be turned off in case you are sure and want to save some time)

    Returns: A re-ordered similarity matrix (type depends on the input type. Sparse should return sparse)
    '''
    if check:
        if not m.shape[0] == len(row_order):
            raise ValueError('''Length of `row_order` must match the number of rows in the matrix''')
        if not is_permutation(row_order):
            raise ValueError('''`row_order` is not a permutation''')

    sh = row_shuffle_matrix(row_order)

    return sh * m * sh

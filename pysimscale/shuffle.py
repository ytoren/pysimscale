from scipy.sparse import coo_matrix


def row_shuffle_matrix(permutation):
    '''Permutation matrix:

    Params:
    - permutation: A list of integers representing a permutation of permutation. Must contain all values in `range(max(permutation))`

    Returns a sparse CSR row/column permutation matrix based on a permutation of row indices
    '''

    diff_set = set(range(min(permutation), max(permutation) + 1, 1)).difference(set(permutation))
    if len(permutation) != max(permutation) + 1 or len(diff_set) > 0:
        raise ValueError('Input is not a permutation')

    return coo_matrix(([1] * len(permutation), (sorted(permutation), permutation)))


def sim_matrix_shuffle(m, row_order):
    '''Re-arrange a similarity matrix based on a new row order

    Params:
    - m: Similarity matrix. Can be dense (Numpy array / matrix) or sparse (as long as it supports arithmetic operations)
    - row_order: A list of integers, which makes a permutation of the matrix rows

    Returns: A re-ordered similarity matrix (type depends on the input type. Sparse should return sparse)
    '''

    sh = row_shuffle_matrix(row_order)

    return sh * m * sh

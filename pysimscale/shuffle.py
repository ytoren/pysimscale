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

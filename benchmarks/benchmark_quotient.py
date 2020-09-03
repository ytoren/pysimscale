import timeit
from math import sqrt
from numpy import array, where, maximum, fill_diagonal, allclose
from numpy.random import uniform
from random import choices
from scipy.sparse import csr_matrix
import networkx as nx
from pysimscale import sim_matrix_shuffle, quotient_similarity
from matplotlib import pyplot as plt

n_features = 10
n_rows = [100, 1000, 2000, 3000, 4000, 5000]
sim_thresh = 0.8

nx_time = []
pss_time = []

for r in n_rows:
    # Generate a similarity matrix (symmetrix, diag = 1)
    m = uniform(0, 1, (r, r))
    m = abs(maximum(m, m.T))
    fill_diagonal(m, 1)
    m[m < sim_thresh] = 0
    m = csr_matrix(m)
    # Generate IDs for partition
    ids = list(range(int(sqrt(r))))
    id_map = array(choices(ids, k=r))
    partition = [where(id_map == i)[0].tolist() for i in ids]
    # Networkx implementation
    start_time = timeit.default_timer()
    nx_G = nx.from_scipy_sparse_matrix(m)
    nx_Gq = nx.quotient_graph(G=nx_G, partition=partition)
    nx_q = nx.to_scipy_sparse_matrix(nx_Gq)
    nx_q.eliminate_zeros()
    nx_time.append(timeit.default_timer() - start_time)
    # PSS implementation
    start_time = timeit.default_timer()
    pss_q = quotient_similarity(m, partition, diag_value=0)
    pss_time.append(timeit.default_timer() - start_time)
    # Test and print
    if not allclose(nx_q.todense(), pss_q.todense()):
        raise ValueError('Quotient similarity matrix values diverged')
    print('END {} rows'.format(r))

plt.plot(n_rows, nx_time, color='black', linestyle='dashed', linewidth=2, label='Networkx')
plt.plot(n_rows, pss_time, color='green', linewidth=2, label='pysimscale')
plt.title('Quotient Similarity Graph')
plt.legend()
plt.xlabel('Number of rows processed (square root of similarity matrix size)')
plt.ylabel('Run Time (sec.)')
plt.savefig('benchmarks/benchmark_quotient.png')
plt.close('all')

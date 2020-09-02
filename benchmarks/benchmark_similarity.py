import timeit
from numpy import array
from numpy.random import normal
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from pysimscale import truncated_sparse_similarity
from matplotlib import pyplot as plt

n_features = 10
n_rows = [1000, 5000, 10000, 20000, 30000, 40000, 50000]
sim_thresh = 0.5

sk_time = []
pss_row_time = []
pss_block_time = []

for r in n_rows:
    m = normal(0, 1, (r, n_features))
    start_time = timeit.default_timer()
    sk = cosine_similarity(m, dense_output=False)
    sk[sk < sim_thresh] = 0
    sk = csr_matrix(sk)
    sk.setdiag(0)
    sk.eliminate_zeros()
    sk_time.append(timeit.default_timer() - start_time)
    start_time = timeit.default_timer()
    pss_row = truncated_sparse_similarity(m, metric='cosine', thresh=sim_thresh, block_size=1)
    pss_row_time.append(timeit.default_timer() - start_time)
    start_time = timeit.default_timer()
    pss_block = truncated_sparse_similarity(m, metric='cosine', thresh=sim_thresh, block_size=1000)
    pss_block_time.append(timeit.default_timer() - start_time)
    print('END {} rows'.format(r))

plt.plot(n_rows, sk_time, color='black', linestyle='dashed', linewidth=2, label='SKlearn: single thread')
plt.plot(n_rows, pss_row_time, color='blue', linewidth=2, label='pysimscale: parallel rows')
plt.plot(n_rows, pss_block_time, color='green', linewidth=2, label='pysimscale: parallel blocks')
plt.legend()
plt.xlabel('Number of rows processed (square root of similarity matrix size)')
plt.ylabel('Run Time (sec.)')
plt.savefig('benchmarks/benchmark_similarity.png')
plt.close('all')

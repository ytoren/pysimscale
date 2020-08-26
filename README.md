# pysimscale


Large scale similarity calculations with support for:

* Thresholds & sparse representations of the similarity matrix (using [sparse API from Scipy](https://docs.scipy.org/doc/scipy/reference/sparse.html) package)

* Parallel of calculations, using the [Joblib](https://github.com/joblib/joblib) package.

## Background

A smart dev-ops engineer once told me:
> Before I give you a cluster, show me you can fully utilise a single machine

With that in mind I created this package to share my experiences working on large scale similarity projects. One of the main problems I've encountered was scaling up similarity calculations and representation. Specifically how to better distribute calculations (focus on utilising a single, multi-core machine as efficiently as possible) and efficiently store the result, especially when low values are not very interesting (making the similarity matrix very sparse)

This package contains tools for handling this kind of the above problems.

### Assumptions

* Data is numeric (binary, integers or real numbers). For categorical data please convert first (embedding, 1-hot encoding or other methods)

* The $N\times M$ matrix of features ($N$ rows, $M$ features) can be contained in memory and can expose a Numpy array API.


## Installation
`git clone git clone git://github.com/ytoren/pysimscale.git`

## Usage

### Similarity calculations

* To enable parallel calculations please install the `joblib` package (it is not a dependency)

* Built-in, fully parallel support is available for [Cosine](https://en.wikipedia.org/wiki/Cosine_similarity) and [Hamming](https://en.wikipedia.org/wiki/Hamming_distance) similarities. You can specify your own similarity function as long as it supports a "single row against the entire matrix" kind of output.

#### Example: cosine similarity

`a = array([[1, 1, 1, 1], [0, 1, 0, 2],[2.2, 2, 2.2, 0.5]])`

To return the full similarity matrix (no thresholding):

```
sim = truncated_sparse_similarity(a1, metric='cosine', thresh=0, diag_value=None, n_jobs=1)
print(sim.todense())
```

More practically, we don't care about low similarity values (let's say <0.9) so we can use a sparse representation. We also don't need a diagonal of `1`s (save some memory), and we want to run this in parallel:

```
sim = truncated_sparse_similarity(a1, metric='cosine', thresh=0.0, diag_value=0, n_jobs=-1)
print(sim.todense())
```

### Pandas tools

[Pandas](https://pandas.pydata.org/) is not a dependency for this package, but I did include some tools to handle series data. Especially cases where each row contains a vector but some contain `None/NAN/nan` values. See `print(series2array2D.__doc__)` for details.

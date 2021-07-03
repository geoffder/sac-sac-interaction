import numpy as np


def bin_reduce(x, sz, reducer):
    n_bins = int(np.ceil(len(x) / sz))
    return np.array([reducer(a) for a in np.array_split(x, n_bins)])


def bin_sum(x, sz):
    return bin_reduce(x, sz, np.sum)


def bin_mean(x, sz):
    return bin_reduce(x, sz, np.mean)


def raster(bins, thresh=1., max_q=3):
    def quantize(a):
        return np.clip(np.floor(a / thresh), a_min=0, a_max=max_q)

    return np.array([quantize(b) for b in bins])


def sum_quanta(bins, edges, q, dt):
    edges -= edges[0]  # offset back to begin at zero
    n_pts = int(edges[-1] / dt) + len(q)
    all_pad = n_pts - len(q)
    sm = np.zeros(n_pts)
    for b, e in zip(bins, edges):
        lead = int(e / dt)
        trail = all_pad - lead
        sm += np.concatenate([np.zeros(lead), q * b, np.zeros(trail)])
    return sm

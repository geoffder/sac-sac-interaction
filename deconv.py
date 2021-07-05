import numpy as np


def bin_reduce(x, sz, reducer):
    n_bins = int(np.ceil(x.shape[-1] / sz))
    return np.concatenate(
        [reducer(a, axis=-1, keepdims=True) for a in np.array_split(x, n_bins, axis=-1)],
        axis=-1
    )


def bin_sum(x, sz):
    return bin_reduce(x, sz, np.sum)


def bin_mean(x, sz):
    return bin_reduce(x, sz, np.mean)


def raster(bins, thresh=1., max_q=3):
    return np.clip(np.floor(bins / thresh), a_min=0, a_max=max_q)


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


def quantal_size_estimate(arr):
    return 2.0 * np.var(arr) / np.mean(arr)


def get_quanta(recs, quantum, dt, bin_t=0.05, ceiling=0.9, max_q=5, scale_mode=False):
    """Expects recs to be either a 1-d ndarray, or an ndarray of shape (n_rois,
    trials, time)."""
    orig_shape = recs.shape
    if recs.ndim == 1:
        recs = recs.reshape(1, 1, -1)
    n_rois, trials, n_pts = recs.shape

    quantum_fft = np.fft.rfft(quantum, n=recs.shape[-1]).reshape(1, 1, -1)
    rec_fft = np.fft.rfft(recs, axis=-1)
    inv_trans = np.fft.irfft(rec_fft / quantum_fft, axis=-1)

    sz = int(bin_t / dt)
    quanta_xaxis = np.arange(sz, inv_trans.shape[-1] + sz, sz) * dt
    binned = bin_mean(inv_trans, sz)

    scaled_quantum = quantum / max_q if scale_mode else quantum

    quanta = raster(
        binned,
        thresh=np.max(binned, axis=-1, keepdims=True) * ceiling / max_q,
        max_q=max_q
    )
    quantal_sum = np.stack(
        [
            sum_quanta(qs, quanta_xaxis, scaled_quantum, dt)
            for qs in quanta.reshape(-1, quanta.shape[-1])
        ],
        axis=0
    ).reshape(n_rois, trials, -1)
    quantal_sum_xaxis = np.arange(quantal_sum.shape[-1]) * dt

    if orig_shape != recs.shape:
        quanta, quantal_sum = np.squeeze(quanta), np.squeeze(quantal_sum)

    return quanta, quanta_xaxis, quantal_sum, quantal_sum_xaxis

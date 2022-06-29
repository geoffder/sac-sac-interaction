import numpy as np
from utils import find_rise_bsln, find_bsln_return


def bin_reduce(x, sz, reducer):
    n_bins = int(np.ceil(x.shape[-1] / sz))
    return np.concatenate(
        [
            reducer(a, axis=-1, keepdims=True)
            for a in np.array_split(x, n_bins, axis=-1)
        ],
        axis=-1,
    )


def bin_sum(x, sz):
    return bin_reduce(x, sz, np.sum)


def bin_mean(x, sz):
    return bin_reduce(x, sz, np.mean)


def raster(bins, thresh=1.0, max_q=3):
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


def static_to_motion(
    rec, dt, rf=60, spot=400, vel=500, rise_start=None, **find_rise_kwargs
):
    """Zero out recording (intended to be response to static stimuli) starting
    from a duration after the rise of the response determined by the given `rf`
    and `spot` diameters and the `vel`ocity of the spot. If the `rise_start` index
    is not provided, it will be calculated with `utils.find_rise_bsln`
    (using `**find_rise_kwargs`)."""
    dur = int((rf + spot) / vel / dt)
    start = (
        find_rise_bsln(rec, **find_rise_kwargs) if rise_start is None else rise_start
    )
    end = min((dur + start), len(rec))
    mask = np.append(np.ones(end), np.zeros(len(rec) - end))
    return rec * mask


def clip_response(r, lead=0, tail=0, **find_kwargs):
    start = find_rise_bsln(r, **find_kwargs) - lead
    end = find_bsln_return(r, **find_kwargs) + tail
    return r[start:end]


def release_rate(event, quantum):
    """Inverse fourier transform of event recording by representative quantum."""
    event_fft = np.fft.rfft(event)
    quantum_fft = np.fft.rfft(quantum, n=len(event))
    return np.fft.irfft(event_fft / quantum_fft)


def get_quanta(recs, quantum, dt, bin_t=0.05, ceiling=0.9, max_q=5, scale_mode=False):
    """Expects recs to be either a 1-d ndarray, or an ndarray of shape (n_rois,
    trials, time)."""
    orig_shape = recs.shape
    if recs.ndim == 1:
        recs = recs.reshape(1, 1, -1)
    n_rois, trials, _ = recs.shape

    quantum_fft = np.fft.rfft(quantum, n=recs.shape[-1]).reshape(1, 1, -1)
    rec_fft = np.fft.rfft(recs, axis=-1)
    inv_trans = np.fft.irfft(rec_fft / quantum_fft, axis=-1)

    sz = int(bin_t / dt)
    quanta_xaxis = np.arange(sz, inv_trans.shape[-1] + sz, sz) * dt
    binned = bin_mean(inv_trans, sz)

    scaled_quantum = quantum / np.max(quantum) / max_q if scale_mode else quantum

    quanta = raster(
        binned,
        thresh=np.max(binned, axis=-1, keepdims=True) * ceiling / max_q,
        max_q=max_q,
    )
    quantal_sum = np.stack(
        [
            sum_quanta(qs, quanta_xaxis, scaled_quantum, dt)
            for qs in quanta.reshape(-1, quanta.shape[-1])
        ],
        axis=0,
    ).reshape(n_rois, trials, -1)
    quantal_sum_xaxis = np.arange(quantal_sum.shape[-1]) * dt

    if orig_shape != recs.shape:
        quanta, quantal_sum = np.squeeze(quanta), np.squeeze(quantal_sum)

    return quanta, quanta_xaxis, quantal_sum, quantal_sum_xaxis


# TODO: should use the new generator class way that numpy encourages. Create a py_rng
# object with a seed (as done for hoc random) that is used for number pulling on the
# python side.
def poisson_quanta(rate, dt, duration):
    n = int(duration / dt)
    return np.random.poisson(lam=dt * rate, size=n)


def quanta_to_times(qs, dt):
    ts = []
    t = 0.0
    for n in qs:
        for _ in range(n):
            ts.append(t)
        t += dt
    return np.array(ts)


def times_to_quanta(ts, dt, duration):
    counts, _ = np.histogram(ts, bins=int(duration / dt))
    return counts


def poisson_bipolar(trans_rate, trans_dur, sust_rate, sust_dur, dt):
    """Simplistic biphasic train, starting with a transient rate and duration,
    followed by the sustained rate and duration. Output is in intervals of
    dt."""
    trans = poisson_quanta(trans_rate, dt, trans_dur)
    sust = poisson_quanta(sust_rate, dt, sust_dur)
    return np.concatenate([trans, sust])


def poisson_of_release(rng, rate):
    """Takes a 1d ndarray representing a variable release rate and returns an
    array of the same size with poisson generated count of events per interval.
    """
    return np.concatenate([rng.poisson(lam=max(r, 0.0), size=1) for r in rate])


def train_maker(rate, dt):
    """Takes a 1d ndarray representing a variable release rate with an interval
    specified by dt (in seconds). Returns a function which takes an onset time
    (ms), and returns a train of event times in milliseconds to be fed to a
    NetQuanta object."""
    return lambda rng, t: quanta_to_times(poisson_of_release(rng, rate), dt) * 1000 + t

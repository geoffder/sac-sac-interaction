from neuron import h
import h5py as h5
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy import interpolate
import symfit as sf


def nrn_section(name):
    """Create NEURON hoc section, and return a corresponding python object."""
    h("create " + name)
    return h.__getattribute__(name)


def nrn_objref(name):
    """Create NEURON hoc objref, and return a corresponding python object."""
    h("objref " + name)
    return h.__getattribute__(name)


def pack_hdf(pth, data_dict):
    """Takes data organized in a python dict, and creates an hdf5 with the
    same structure. Keys are converted to strings to comply to hdf5 group naming
    convention. In `unpack_hdf`, if the key is all digits, it will be converted
    back from string."""

    def rec(data, grp):
        for k, v in data.items():
            k = str(k) if type(k) != str else k
            if type(v) is dict:
                rec(v, grp.create_group(k))
            else:
                grp.create_dataset(k, data=v)

    with h5.File(pth + ".h5", "w") as pckg:
        rec(data_dict, pckg)


def unpack_hdf(group):
    """Recursively unpack an hdf5 of nested Groups (and Datasets) to dict."""
    return {
        int(k)
        if k.isdigit()
        else k: v[()]
        if type(v) is h5._hl.dataset.Dataset
        else unpack_hdf(v)
        for k, v in group.items()
    }


def rotate(origin, X, Y, angle):
    """
    Rotate a point (X[i],Y[i]) counterclockwise an angle around an origin.
    The angle should be given in radians.
    """
    ox, oy = origin
    X, Y = np.array(X), np.array(Y)
    rotX = ox + np.cos(angle) * (X - ox) - np.sin(angle) * (Y - oy)
    rotY = oy + np.sin(angle) * (X - ox) + np.cos(angle) * (Y - oy)
    return rotX, rotY


def measure_response(vm_rec, threshold=20):
    vm = np.array(vm_rec)
    psp = vm + 70
    area = sum(psp[70:]) / len(psp[70:])
    thresh_count, _ = find_spikes(vm, thresh=threshold)
    return vm, area, thresh_count


def calc_DS(dirs, response):
    xpts = np.multiply(response, np.cos(dirs))
    ypts = np.multiply(response, np.sin(dirs))
    xsum = np.sum(xpts)
    ysum = np.sum(ypts)
    DSi = np.sqrt(xsum ** 2 + ysum ** 2) / np.sum(response)
    theta = np.arctan2(ysum, xsum) * 180 / np.pi

    return DSi, theta


def polar_plot(dirs, metrics, show_plot=True):
    # resort directions and make circular for polar axes
    circ_vals = metrics["spikes"].T[np.array(dirs).argsort()]
    circ_vals = np.concatenate([circ_vals, circ_vals[0, :].reshape(1, -1)], axis=0)
    circle = np.radians([0, 45, 90, 135, 180, 225, 270, 315, 0])

    peak = np.max(circ_vals)  # to set axis max
    avg_theta = np.radians(metrics["avg_theta"])
    avg_DSi = metrics["avg_DSi"]
    thetas = np.radians(metrics["thetas"])
    DSis = np.array(metrics["DSis"])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="polar")

    # plot trials lighter
    ax.plot(circle, circ_vals, color=".75")
    ax.plot([thetas, thetas], [np.zeros_like(DSis), DSis * peak], color=".75")

    # plot avg darker
    ax.plot(circle, np.mean(circ_vals, axis=1), color=".0", linewidth=2)
    ax.plot([avg_theta, avg_theta], [0.0, avg_DSi * peak], color=".0", linewidth=2)

    # misc settings
    ax.set_rlabel_position(-22.5)  # labels away from line
    ax.set_rmax(peak)
    ax.set_rticks([peak])
    ax.set_thetagrids([0, 90, 180, 270])

    if show_plot:
        plt.show()

    return fig


def stack_trials(n_trials, n_dirs, data_list):
    """Stack a list of run recordings [ndarrays of shape (recs, samples)]
    into a single ndarray of shape (trials, directions, recs, samples).
    """
    stack = np.stack(data_list, axis=0)
    return stack.reshape(n_trials, n_dirs, *stack.shape[1:])


def find_spikes(Vm, thresh=20):
    """use scipy.signal.find_peaks to get spike count and times"""
    spikes, _ = find_peaks(Vm, height=thresh)  # returns indices
    count = spikes.size
    times = spikes * h.dt

    return count, times


def thresholded_area(x, thresh=None):
    if thresh is not None:
        x = np.clip(x, thresh, None) - thresh
    else:
        x = x - np.min(x, axis=-1)

    return np.sum(x, axis=-1)


def peak_vm_deflection(x):
    return np.max(x - np.min(x, axis=-1, keepdims=True), axis=-1)


def pn_dsi(a, b, eps=0.0000001):
    return (a - b) / (a + b + eps)


def clean_axes(axes, remove_spines=["right", "top"], ticksize=11):
    """A couple basic changes I often make to pyplot axes. If input is an
    iterable of axes (e.g. from plt.subplots()), apply recursively."""
    if hasattr(axes, "__iter__"):
        for a in axes:
            clean_axes(a, remove_spines=remove_spines)
    else:
        for r in remove_spines:
            axes.spines[r].set_visible(False)
        axes.tick_params(
            axis="both",
            which="both",  # applies to major and minor
            **{r: False for r in remove_spines},  # remove ticks
            **{"label%s" % r: False for r in remove_spines}  # remove labels
        )
        for ticks in axes.get_yticklabels():
            ticks.set_fontsize(ticksize)
        for ticks in axes.get_xticklabels():
            ticks.set_fontsize(ticksize)


def nearest_index(arr, v):
    """Index of value closest to v in ndarray `arr`"""
    return np.abs(arr - v).argmin()


def apply_to_data(f, data):
    """Recursively apply the same operation to all ndarrays stored in the given
    dictionary. It may have arbitary levels of nesting, as long as the leaves are
    arrays and they are a shape that the given function can operate on."""

    def applyer(val):
        if type(val) == dict:
            return {k: applyer(v) for k, v in val.items()}
        else:
            return f(val)

    return {k: applyer(v) for k, v in data.items()}


def apply_to_data2(f, d1, d2):
    def applyer(v1, v2):
        if type(v1) == dict:
            return {k: applyer(v1, v2) for (k, v1), v2 in zip(v1.items(), v2.values())}
        else:
            return f(v1, v2)

    return {k: applyer(v1, v2) for (k, v1), v2 in zip(d1.items(), d2.values())}


def stack_pair_data(exp):
    """Expects data in the form exported from SAC-SAC experiments.
    conditions -> section/synapses -> sacs -> metrics"""
    return {
        cond: {
            k: apply_to_data2(
                lambda a, b: np.stack([a, b], axis=0), ex[k]["a"], ex[k]["b"]
            )
            for k in ex.keys()
        }
        for cond, ex in exp.items()
    }


def linear_bp_sum(exp):
    """Linearly (simple sum) combine all bipolar inputs in SAC-SAC experiments.
    Output dict is in same structure as the input experiment, but the sole contents
    are the summed conductances and currents for each SAC."""
    return {
        cond: {
            n: {
                m: np.sum(
                    [bp[m] for bps in sac.values() for bp in bps.values()], axis=0
                )
                for m in ["g", "i"]
            }
            for n, sac in e["bps"].items()
        }
        for cond, e in exp.items()
    }


def inverse_transform(x, y):
    """Generate a sampling function from a distribution that recreates the relationship
    between the given x and y vectors. Since this method assumes the underlying data is
    actually a distribution, with x representing bin edges, an additional 0 edge will be
    added before the first position. Additionally, the y vector is normalized such that it
    sums to 1, so that probability can be distributed properly across the range.
    """
    x = np.concatenate([[0], x])
    cum = np.zeros(len(x))
    cum[1:] = np.cumsum(y / np.sum(y))
    inv_cdf = interpolate.interp1d(cum, x)
    return inv_cdf


def find_rise_start(arr, step=10):
    peak_idx = np.argmax(arr)

    def rec(last_min, idx):
        min_idx = np.argmin(arr[idx - step : idx]) + idx - step
        return rec(arr[min_idx], min_idx) if arr[min_idx] < last_min else idx

    return rec(arr[peak_idx], peak_idx)


def find_bsln_return(
    arr, bsln_start=0, bsln_end=None, offset=0.0, step=10, pre_step=False
):
    idx = np.argmax(arr[bsln_end:]) + (0 if bsln_end is None else bsln_end)
    last_min = idx
    bsln = np.mean(arr[bsln_start:bsln_end]) + offset
    if step > 0:
        last = len(arr) - 1
        stop = lambda next: next <= last
        get_min = lambda i: np.argmin(arr[i : i + step]) + i
    else:
        stop = lambda next: next >= 0
        get_min = lambda i: np.argmin(arr[i + step : i]) + i + step

    while stop(idx + step):
        min_idx = get_min(idx)
        if arr[min_idx] < bsln:
            return last_min if pre_step else min_idx
        else:
            last_min = min_idx
            idx += step

    return min_idx


def find_rise_bsln(
    arr, bsln_start=0, bsln_end=None, offset=0.0, step=10, pre_step=False
):
    return find_bsln_return(arr, bsln_start, bsln_end, offset, -step, pre_step)


class BiexpFitter:
    def __init__(self, est_tau1, est_tau2, amp=1.0, norm_amp=False):
        self.a0 = amp
        self.b0 = amp
        self.norm_amp = norm_amp
        a, b, g, t = sf.variables("a, b, g, t")
        tau1 = sf.Parameter("tau1", est_tau1)
        tau2 = sf.Parameter("tau2", est_tau2)
        y0 = sf.Parameter("y0", est_tau2)
        y0.value = 1.0
        y0.min = 0.5
        y0.max = 2.0
        self.ode_model = sf.ODEModel(
            {
                # sf.D(a, t): -a / tau1,
                # sf.D(b, t): -b / tau2,
                # HACK: trick model into fitting an initial value (always 1)
                # https://stackoverflow.com/questions/49149241/ode-fitting-with-symfit-for-python-how-to-get-estimations-for-intial-values
                sf.D(a, t): -a / tau1 * (sf.cos(y0) ** 2 + sf.sin(y0) ** 2),
                sf.D(b, t): -b / tau2 * (sf.cos(y0) ** 2 + sf.sin(y0) ** 2),
            },
            initial={
                t: 0,
                a: y0.value,
                b: y0.value,
                # a: amp,
                # b: amp
            },
        )
        self.model = sf.CallableNumericalModel(
            {g: self.g_func}, connectivity_mapping={g: {t, tau1, tau2, y0}}
        )
        self.constraints = [
            sf.GreaterThan(tau1, 0.0001),
            sf.GreaterThan(tau2, tau1),
        ]

    def g_func(self, t, tau1, tau2, y0):
        res = self.ode_model(t=t, tau1=tau1, tau2=tau2, y0=y0)
        g = res.b - res.a
        gmax = np.max(g)
        if self.norm_amp and not np.isclose(gmax, 0.0):
            return g * 1 / gmax
        else:
            # tp = (tau1 * tau2) / (tau2 - tau1) * np.log(tau2 / tau1)
            # factor = 1 / (-np.exp(-tp / tau1) + np.exp(-tp / tau2))
            # return g + factor
            return g

    def fit(self, x, y):
        self.results = sf.Fit(
            self.model, t=x, g=y, constraints=self.constraints
        ).execute()
        return self.results

    def calc_g(self, x):
        return self.model(t=x, **self.results.params)[0]


def biexp(x, m, t1, t2, b):
    return m * (np.exp(-t2 * x) - np.exp(-t1 * x)) + b


def aligned_avg(recs, bsln_start=50, bsln_end=150, step=1):
    """Aligns provided (positive going) recordings (2d ndarray, shape: (N, T))
    by their rises."""
    rise_idxs = np.array(
        [
            find_rise_bsln(r, bsln_start=bsln_start, bsln_end=bsln_end, step=step)
            for r in recs
        ]
    )
    shifts = rise_idxs - np.min(rise_idxs)
    trim = np.max(shifts)
    aligned = np.mean(
        [r[s : (-trim + s) if s != trim else None] for r, s in zip(recs, shifts)],
        axis=0,
    )
    return aligned

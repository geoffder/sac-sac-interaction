from neuron import h
import h5py as h5

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def nrn_section(name):
    """Create NEURON hoc section, and return a corresponding python object."""
    h("create " + name)
    return h.__getattribute__(name)


def pack_hdf(pth, data_dict):
    """
    Takes data organized in a python dict, and creates an hdf5 with the same
    structure.
    """
    pckg = h5.File(pth + '.h5', 'w')

    for key, dataset in data_dict.items():
        set_grp = pckg.create_group(key)
        for cond, data in dataset.items():
            set_grp.create_dataset(cond, data=data)

    pckg.close()


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
    psp = vm + 61.3
    area = sum(psp[70:]) / len(psp[70:])
    thresh_count, _ = find_spikes(vm, thresh=threshold)
    return vm, area, thresh_count


def calc_DS(dirs, response):
    xpts = np.multiply(response, np.cos(dirs))
    ypts = np.multiply(response, np.sin(dirs))
    xsum = np.sum(xpts)
    ysum = np.sum(ypts)
    DSi = np.sqrt(xsum**2 + ysum**2) / np.sum(response)
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

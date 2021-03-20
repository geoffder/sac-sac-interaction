from neuron import h, gui

# general libraries
import os
import h5py as h5
import json

import numpy as np  # arrays
import matplotlib.pyplot as plt

from utils import *


class Sac:
    def __init__(self, name, forward=True, seed=0, params=None):
        self.set_default_params()
        if params is not None:
            self.update_params(params)
        self.name = name
        self.forward = forward  # soma (left) to dendrite (right)
        self.seed = seed  # NOTE: if separate rands like this, must use far apart seeds
        self.rand = h.Random(seed)
        self.nz_seed = 1  # noise seed for HHst

        self.calc_xy_locs()
        self.create_neuron()  # builds and connects soma and dendrite

    def set_default_params(self):
        self.origin = (0, 0)

        # soma physical properties
        self.soma_l = 10
        self.soma_diam = 10
        self.soma_nseg = 1
        self.soma_ra = 100

        # dendrite physical properties
        self.dend_nseg = 25
        self.seg_step = 1 / self.dend_nseg
        self.dend_diam = .5
        self.dend_l = 150
        self.term_l = 10
        self.dend_ra = 100

        # soma active properties
        self.soma_na = .0  # [S/cm2]
        self.soma_k = .035  # [S/cm2]
        self.soma_km = .003  # [S/cm2]

        self.dend_cat = .0003
        self.dend_cal = .0003
        self.soma_gleak_hh = .0001667  # [S/cm2]
        self.soma_eleak_hh = -60.0  # [mV]
        self.soma_gleak_pas = .0001667  # [S/cm2]
        self.soma_eleak_pas = -60  # [mV]

        # dend compartment active properties
        self.dend_na = .03  # [S/cm2] .03
        self.dend_k = .025  # [S/cm2]
        self.dend_km = .003  # [S/cm2]
        self.dend_gleak_hh = 0.0001667  # [S/cm2]
        self.dend_eleak_hh = -60.0  # [mV]
        self.dend_gleak_pas = .0001667  # [S/cm2]
        self.dend_eleak_pas = -60  # [mV]

        # membrane noise
        self.dend_nz_factor = .1  # default NF_HHst = 1
        self.soma_nz_factor = .1

        self.bp_jitter = 0
        self.bp_locs = {"prox": [5], "dist": [25, 45, 65]}
        self.bp_props = {
            "prox":
                {
                    "tau1": 10,  # excitatory conductance rise tau [ms]
                    "tau2": 60,  # excitatory conductance decay tau [ms]
                    "rev": 0,  # excitatory reversal potential [mV]
                    "weight": .000275,  # weight of excitatory NetCons [uS] .00023
                },
            "dist":
                {
                    "tau1": .1,  # inhibitory conductance rise tau [ms]
                    "tau2": 12,  # inhibitory conductance decay tau [ms]
                    "rev": -65,  # inhibitory reversal potential [mV]
                    "weight": .000495,  # weight of inhibitory NetCons [uS]
                }
        }

        self.gaba_props = {
            "loc": 15,  # distance from soma [um]
            "thresh": -50,  # pre-synaptic release threshold
            "tau1": .5,  # inhibitory conductance rise tau [ms]
            "tau2": 60,  # inhibitory conductance decay tau [ms]
            "rev": -70,  # inhibitory reversal potential [mV]
            "weight": .001,  # weight of inhibitory NetCons [uS]
            "delay": .5,
        }

    def update_params(self, params):
        """Update self members with key-value pairs from supplied dict."""
        for k, v in params.items():
            self.__dict__[k] = v

    def get_params_dict(self):
        params = self.__dict__.copy()
        # remove the non-param entries (model objects)
        for key in ["soma", "dend", "term", "bps", "rand"]:
            params.pop(key)
        return params

    def create_soma(self):
        """Build and set membrane properties of soma compartment"""
        soma = nrn_section("soma_%s" % self.name)
        soma.L = self.soma_l
        soma.diam = self.soma_diam
        soma.nseg = self.soma_nseg
        soma.Ra = self.soma_ra

        soma.insert('HHst')
        soma.gnabar_HHst = self.soma_na
        soma.gkbar_HHst = self.soma_k
        soma.gkmbar_HHst = self.soma_km
        soma.gleak_HHst = self.soma_gleak_hh
        soma.eleak_HHst = self.soma_eleak_hh
        soma.NF_HHst = self.soma_nz_factor

        return soma

    def create_dend(self):
        dend = nrn_section("dend_%s" % (self.name))
        term = nrn_section("term_%s" % (self.name))
        for s in [dend, term]:
            s.nseg = self.dend_nseg
            s.Ra = self.dend_ra
            s.diam = self.dend_diam
            s.insert('HHst')
            s.gnabar_HHst = self.dend_na
            s.gkbar_HHst = self.dend_k
            s.gkmbar_HHst = self.dend_km
            s.gtbar_HHst = self.dend_cat
            s.glbar_HHst = self.dend_cal
            s.gleak_HHst = self.dend_gleak_hh  # (S/cm2)
            s.eleak_HHst = self.dend_eleak_hh
            s.NF_HHst = self.dend_nz_factor
            s.seed_HHst = self.nz_seed

        dend.L = self.dend_l
        dend.gtbar_HHst = 0
        dend.glbar_HHst = 0
        term.gtbar_HHst = self.dend_cat
        term.glbar_HHst = self.dend_cal
        term.L = self.term_l

        term.connect(dend)
        return dend, term

    def create_synapses(self):
        # access hoc compartment
        self.dend.push()
        # create *named* hoc objects for each synapse (for gui compatibility)
        h(
            "objref prox_bps[%i], dist_bps[%i]" %
            (len(self.bp_locs["prox"]), len(self.bp_locs["dist"]))
        )

        # complete synapses are made up of a NetStim, Syn, and NetCon
        self.bps = {
            "prox": {
                "stim": [],
                "syn": h.prox_bps,
                "con": []
            },
            "dist": {
                "stim": [],
                "syn": h.dist_bps,
                "con": []
            },
        }

        for (k, syns), locs, props in zip(
            self.bps.items(), self.bp_locs.values(), self.bp_props.values()
        ):
            for i in range(len(locs)):
                # 0 -> 1 position dendrite section
                # obtain fractional position from distance to soma
                pos = np.round(locs[i] / self.dend_l, decimals=5)

                # Synapse object (source of conductance)
                syns["syn"][i] = h.Exp2Syn(pos)
                syns["syn"][i].tau1 = props["tau1"]
                syns["syn"][i].tau2 = props["tau2"]
                syns["syn"][i].e = props["rev"]

                # Network Stimulus object (activates synaptic event)
                syns["stim"].append(h.NetStim(pos))
                syns["stim"][i].interval = 0
                syns["stim"][i].number = 0
                syns["stim"][i].noise = 0

                # Network Connection object (connects stimulus to synapse)
                syns["con"].append(
                    h.NetCon(
                        syns["stim"][i],
                        syns["syn"][i],
                        0,  # threshold
                        0,  # delay [ms]
                        props["weight"],  # conductance strength
                    )
                )

        # remove section from access stack
        h.pop_section()

    def calc_xy_locs(self):
        """Origin of the arena is (0, 0), so the dendrite is positioned with
        that as the centre. X locations are calculated for based on the
        distances of each bipolar cell from the soma."""
        total_l = self.dend_l + self.term_l
        dir_sign = 1 if self.forward else -1
        o_x, o_y = self.origin
        dend_x_origin = o_x + (total_l / 2 * -1 * dir_sign)
        self.bp_xy_locs = {
            k: {
                "x": [dir_sign * l + dend_x_origin for l in locs],
                "y": [o_y for _ in locs]
            }
            for k, locs in self.bp_locs.items()
        }
        return self.bp_xy_locs

    def create_neuron(self):
        # create compartments (using parameters in self.__dict__)
        self.soma = self.create_soma()
        self.dend, self.term = self.create_dend()
        self.dend.connect(self.soma)
        self.create_synapses()  # generate synapses on dendrite

    def rotate_sacs(self, rotation):
        rotated = {}
        for s, locs in self.bp_xy_locs.items():
            x, y = rotate(self.origin, locs["x"], locs["y"], rotation)
            rotated[s] = {"x": x, "y": y}
        return rotated

    def bar_sweep(self, bar, rad_angle):
        """Return activation time for the single synapse based on the light bar
        config and the bipolar locations on the presynaptic dendrites.
        """
        ax = "x" if bar["x_motion"] else "y"
        rot_locs = self.rotate_sacs(-rad_angle)
        on_times = {
            s: [
                bar["start_time"] + (l - bar[ax + "_start"]) / bar["speed"]
                for l in locs[ax]
            ]
            for s, locs in rot_locs.items()
        }

        return on_times

    def bar_onsets(self, bar, rad_direction):
        # bare base onset with added jitter
        for k, ts in self.bar_sweep(bar, rad_direction).items():
            for t, stim in zip(ts, self.bps[k]["stim"]):
                jit = self.rand.normal(0, 1)
                stim.start = t + self.bp_jitter * jit

    def update_noise(self):
        for s in [self.soma, self.dend, self.term]:
            s.seed_HHst = self.nz_seed
            self.nz_seed += 1


class SacPair:
    def __init__(self, sac_params=None):
        self.sacs = {
            "a": Sac("a", params=sac_params),
            "b": Sac("b", forward=False, params=sac_params)
        }
        self.wire_gaba()

    def wire_gaba(self):
        self.gaba_syns = {}
        for n, sac in self.sacs.items():
            pos = np.round(sac.gaba_props["loc"] / sac.dend_l, decimals=5)
            sac.dend.push()
            self.gaba_syns[n] = {"syn": h.Exp2Syn(pos)}
            h.pop_section()
        for pre, post in [("a", "b"), ("b", "a")]:
            sac = self.sacs[pre]
            sac.term.push()
            self.gaba_syns[pre]["conn"] = h.NetCon(
                sac.term(1)._ref_v,
                self.gaba_syns[post]["syn"],
                sac.gaba_props["thresh"],
                sac.gaba_props["delay"],
                sac.gaba_props["weight"],
            )
            h.pop_section()

    def get_params_dict(self):
        return {n: sac.get_params_dict() for n, sac in self.sacs.items()}

    def bar_onsets(self, stim, dir_idx):
        for sac in self.sacs.values():
            sac.bar_onsets(stim, dir_idx)

    def update_noise(self):
        for sac in self.sacs.values():
            sac.update_noise()


class Runner:
    def __init__(self, model, data_path=""):
        self.data_path = data_path
        if not os.path.isdir(data_path):
            os.makedirs(data_path)
        self.model = model

        # hoc environment parameters
        self.tstop = 2000  # [ms]
        self.steps_per_ms = 1  # [10 = 10kHz]
        self.dt = .1  # [ms, .1 = 10kHz]
        self.v_init = -70
        self.celsius = 36.9
        self.set_hoc_params()

        self.config_stimulus()

    def set_hoc_params(self):
        """Set hoc NEURON environment model run parameters."""
        h.tstop = self.tstop
        h.steps_per_ms = self.steps_per_ms
        h.dt = self.dt
        h.v_init = self.v_init
        h.celsius = self.celsius

    def config_stimulus(self):
        # light stimulus
        self.light_bar = {
            "start_time": 0.,  # vel -> start: .25 -> -900; .5 -> -400
            "speed": 1.0,  # speed of the stimulus bar (um/ms)
            "width": 500,  # width of the stimulus bar(um)
            "x_motion": True,  # move bar in x, if not, move bar in y
            "x_start": -175,  # start location (X axis) of the stim bar (um)
            "x_end": 175,  # end location (X axis)of the stimulus bar (um)
            "y_start": 25,  # start location (Y axis) of the stimulus bar (um)
            "y_end": 225,  # end location (Y axis) of the stimulus bar (um)
        }

        self.dir_labels = [225, 270, 315, 0, 45, 90, 135, 180]
        self.dir_rads = np.radians(self.dir_labels)
        self.dirs = [135, 90, 45, 0, 45, 90, 135, 180]
        self.dir_inds = np.array(self.dir_labels).argsort()
        self.circle = np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315, 0])

    def run(self, stim, dir_idx):
        """Initialize model, set synapse onset and release numbers, update
        membrane noise seeds and run the model. Calculate somatic response and
        return to calling function."""
        h.init()
        self.model.bar_onsets(stim, self.dir_rads[dir_idx])
        self.model.update_noise()

        self.clear_recordings()
        h.run()
        self.dump_recordings()

    def velocity_run(
        self,
        velocities=np.linspace(.1, 2, 10),
        n_trials=1,
        prefix="",
    ):
        """"""
        self.place_electrodes()

        n_vels = len(velocities)
        stim = {"type": "bar", "dir": 0}
        params = self.model.get_params_dict()  # for logging

        for j in range(n_trials):
            print("trial %d..." % j, end=" ", flush=True)

            for i in range(n_vels):
                print("%.2f" % velocities[i], end=" ", flush=True)
                self.light_bar["speed"] = velocities[i]
                self.run(self.light_bar, 3)  # index of 0 degrees

            print("")  # next line

        data = {
            "params": json.dumps(params),
            "data":
                {
                    sec: {
                        n: {
                            m: stack_trials(n_trials, n_vels, metric)
                            for m, metric in d.items()
                        }
                        for n, d in ds.items()
                    }
                    for sec, ds in self.data.items()
                },
        }

        return data

    def place_electrodes(self):
        self.recs = {"soma": {}, "term": {}}
        self.data = {"soma": {}, "term": {}}
        for n, sac in self.model.sacs.items():
            for p in ["soma", "term"]:
                self.recs[p][n] = h.Vector()
                self.data[p][n] = {"Vm": [], "area": [], "peak": []}
            self.recs["soma"][n].record(sac.soma(0.5)._ref_v)
            self.recs["term"][n].record(sac.term(1)._ref_v)

    def dump_recordings(self):
        for rs, ds in zip(self.recs.values(), self.data.values()):
            for n in self.model.sacs.keys():
                vm, area, _ = measure_response(rs[n])
                ds[n]["Vm"].append(np.round(vm, decimals=3))
                ds[n]["area"].append(np.round(area, decimals=3))
                ds[n]["peak"].append(np.round(np.max(vm + 61.3), decimals=3))

    def clear_recordings(self):
        for sec in self.recs.values():
            for rec in sec.values():
                rec.resize(0)


if __name__ == "__main__":
    base_path = "/mnt/Data/NEURONoutput/sac_sac/"
    os.makedirs(base_path, exist_ok=True)

    model = SacPair()
    runner = Runner(model, data_path=base_path)
    runner.velocity_run()

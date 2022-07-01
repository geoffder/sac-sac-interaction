from neuron import h, gui

import os
import json
from copy import deepcopy
import numpy as np
from utils import *


# TODO: a simple class that can handle the generation / acces of BP releasers
# that handles multiple velocities. Currently, bp_releasers is part of model config
# but velocities, which I want to control input, comes later. So instead of having
# it be set in stone, I need to revamp to allow generation of the velocity
# approriate rates and quanta generators from them in a way that maintains
# compatibility with simply providing a [time] (as is default right now, since a toy
# non-train input may be desired)
class Releaser:
    def __init__(
        self,
        input_rec,
        input_dt,
        quantum,
        lead_t=0.0,
        tail_t=0.1,
        rf=0.06,
        spot=0.4,
        model_dt=0.001,
    ):
        self.input_rec = input_rec
        self.input_dt = input_dt
        self.quantum = quantum
        self.lead_t = lead_t
        self.tail_t = tail_t
        self.rf = rf
        self.spot = spot
        self.model_dt = model_dt


class NetQuanta:
    def __init__(self, syn, weight, delay=0.0):
        self.con = h.NetCon(None, syn, 0.0, 0.0, weight)
        self.delay = delay  # NetCon does not impose delay on *scheduled* events
        self._events = []

    @property
    def weight(self):
        return self.con.weight[0]

    @property
    def events(self):
        return self._events

    @weight.setter
    def weight(self, w):
        self.con.weight[0] = w

    @events.setter
    def events(self, ts):
        self._events = [t + self.delay for t in ts]

    def clear_events(self):
        self._events = []

    def add_event(self, t):
        self._events.append(t + self.delay)

    def initialize(self):
        """Schedule events in the NetCon object. This must be called within the
        function given to h.FInitializeHandler in the model running class/functions.
        """
        for ev in self._events:
            self.con.event(ev)


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
        self.np_rng = np.random.default_rng(seed)

        self.calc_xy_locs()
        self.create_neuron()  # builds and connects soma and dendrite

    def set_default_params(self):
        self.origin = (0, 0)

        # soma physical properties
        self.soma_l = 7
        self.soma_diam = 7
        self.soma_nseg = 1
        self.soma_ra = 100

        # dendrite physical properties
        self.dend_nseg = 25
        self.seg_step = 1 / self.dend_nseg
        self.dend_diam = 0.2
        self.initial_dend_diam = 0.4
        self.initial_dend_l = 10
        self.dend_l = 120
        self.term_l = 20
        self.term_diam = 0.2
        self.dend_ra = 100

        # soma active properties
        self.soma_na = 0.0  # [S/cm2]
        self.soma_k = 0.005  # [S/cm2]
        self.soma_km = 0.0  # [S/cm2]
        self.soma_kv3 = 0.0
        self.soma_kv3_1 = 0.0
        self.soma_kv3_3 = 0.0

        self.soma_gleak_hh = 0.0001667  # [S/cm2]
        # self.soma_eleak_hh = -70.0  # [mV]
        self.soma_eleak_hh = -54.4  # [mV]

        # dend compartment active properties
        self.initial_k = 0.001  # [S/cm2]
        self.initial_km = 0.0  # [S/cm2]
        self.initial_kv3 = 0.0
        self.initial_kv3_1 = 0.0
        self.initial_kv3_3 = 0.0

        self.dend_na = 0.00  # [S/cm2] .03
        self.dend_k = 0.0
        self.dend_km = 0.0
        self.dend_kv3 = 0.0
        self.dend_kv3_1 = 0.0
        self.dend_kv3_3 = 0.0
        self.dend_can = 0.0003
        self.dend_cap = 0.0002
        self.dend_gleak_hh = 0.0001667  # [S/cm2]
        # self.dend_eleak_hh = -70.0  # [mV]
        self.dend_eleak_hh = -54.4  # [mV]

        self.term_na = 0.0
        self.term_k = 0.0  # [S/cm2]
        self.term_km = 0.0  # [S/cm2]
        self.term_kv3 = 0.0
        self.term_kv3_1 = 0.0
        self.term_kv3_3 = 0.0
        self.term_cat = 0.0003
        self.term_cal = 0.0
        self.term_can = 0.0003
        self.term_cap = 0.0003
        self.term_nav1_8 = 0.0
        self.term_nav1_9 = 0.0

        # sink dendrite parameters
        self.sink_dend_locs = [35, 65, 95, 125]
        self.sink_orders = [4, 3, 2, 1]
        self.sink_branch_len = 20
        self.sink_nseg = 1

        # membrane noise
        self.dend_nz_factor = 0.0  # .1  # default NF_HHst = 1
        self.soma_nz_factor = 0.0  # .1

        self.bp_jitter = 0
        self.bp_locs = {"sust": [5], "trans": [25, 45, 65]}
        self.bp_props = {
            "sust": {
                "tau1": 10,  # excitatory conductance rise tau [ms]
                "tau2": 60,  # excitatory conductance decay tau [ms]
                "rev": 0,  # excitatory reversal potential [mV]
                "weight": 0.000275,  # weight of excitatory NetCons [uS] .00023
                "delay": 0,
            },
            "trans": {
                "tau1": 0.1,  # inhibitory conductance rise tau [ms]
                "tau2": 12,  # inhibitory conductance decay tau [ms]
                "rev": 0,  # inhibitory reversal potential [mV]
                "weight": 0.000495,  # weight of inhibitory NetCons [uS]
                "delay": 0,
            },
        }
        # first argument is for providing an rng instance
        self.bp_releasers = {"sust": lambda _rng, t: [t], "trans": lambda _rng, t: [t]}
        self.loc_scaling = False
        self.bp_loc_scaling = {
            "sust": {
                "weight": (lambda loc, w: w),
                "tau1": (lambda loc, t1: t1),
                "tau2": (lambda loc, t2: t2),
            },
            "trans": {
                "weight": (lambda loc, w: w),
                "tau1": (lambda loc, t1: t1),
                "tau2": (lambda loc, t2: t2),
            },
        }

        self.vel_scaling = False
        self.bp_vel_scaling = {
            "sust": {
                "weight": (lambda v, w: w - (np.clip(v / 4, 0, 1) * w * 0.5)),
                "tau1": (lambda v, t1: t1),
                "tau2": (lambda v, t2: t2),
            },
            "trans": {
                "weight": (lambda v, w: w - (np.clip(v / 4, 0, 1) * w * 0.5)),
                "tau1": (lambda v, t1: t1),
                "tau2": (lambda v, t2: t2),
            },
        }

        self.gaba_props = {
            "locs": [15],  # distance from soma [um]
            "calcium": False,  # threshold is cai, rather than vm if True
            "thresh": -50,  # pre-synaptic release threshold
            "tau1": 0.5,  # inhibitory conductance rise tau [ms]
            "tau2": 60,  # inhibitory conductance decay tau [ms]
            "rev": -60,  # inhibitory reversal potential [mV]
            "weight": 0.001,  # weight of inhibitory NetCons [uS]
            "delay": 0.5,
        }
        self.gaba_vel_scaling = lambda _, w: w

    def update_params(self, params):
        """Update self members with key-value pairs from supplied dict."""
        for k, v in params.items():
            v = deepcopy(v) if type(v) == dict else v
            self.__dict__[k] = v

    def get_params_dict(self):
        params = self.__dict__.copy()
        # remove the non-param entries (model objects)
        for key in [
            "soma",
            "initial",
            "dend",
            "term",
            "sinks",
            "all_dends",
            "bps",
            "rand",
            "np_rng",
            "bp_loc_scaling",
            "bp_vel_scaling",
            "gaba_vel_scaling",
            "bp_releasers",
        ]:
            params.pop(key)
        return params

    def create_soma(self):
        """Build and set membrane properties of soma compartment"""
        soma = nrn_section("soma_%s" % self.name)
        soma.L = self.soma_l
        soma.diam = self.soma_diam
        soma.nseg = self.soma_nseg
        soma.Ra = self.soma_ra

        soma.insert("HHst")
        soma.insert("cad")
        soma.insert("Kv3")
        soma.insert("HT")
        soma.insert("Kv3_3")
        soma.gnabar_HHst = self.soma_na
        soma.gkbar_HHst = self.soma_k
        soma.gkmbar_HHst = self.soma_km
        soma.gbar_Kv3 = self.soma_kv3
        soma.gbar_HT = self.soma_kv3_1
        soma.gbar_Kv3_3 = self.soma_kv3_3
        soma.gleak_HHst = self.soma_gleak_hh
        soma.eleak_HHst = self.soma_eleak_hh
        soma.NF_HHst = self.soma_nz_factor

        return soma

    def sink_branch(self, name, n_orders):

        head = nrn_section("branch_%s_head_%s" % (name, self.name))
        last = [[(head, "")]]
        all_secs = [head]
        n = 1
        while n < n_orders:
            next = []
            for branch in last:
                for (sec, lbl) in branch:
                    left = nrn_section("branch_%s%s_l_%s" % (name, lbl, self.name))
                    right = nrn_section("branch%s%s_r_%s" % (name, lbl, self.name))
                    left.connect(sec)
                    right.connect(sec)
                    next.append([(left, lbl + "_l"), (right, lbl + "_r")])
                    all_secs.append(left)
                    all_secs.append(right)
            last = next
            n += 1

        for sec in all_secs:
            sec.nseg = self.sink_nseg
            sec.Ra = self.dend_ra
            sec.insert("HHst")
            sec.insert("cad")
            sec.insert("nav1p8")
            sec.insert("Kv3_3")
            sec.gleak_HHst = self.dend_gleak_hh  # (S/cm2)
            sec.eleak_HHst = self.dend_eleak_hh
            sec.NF_HHst = self.dend_nz_factor
            sec.seed_HHst = self.nz_seed
            sec.diam = self.dend_diam
            sec.L = self.sink_branch_len
            sec.gtbar_HHst = 0
            sec.glbar_HHst = 0
            sec.gnabar_HHst = 0

        return head, all_secs

    def create_dend(self):
        initial = nrn_section("initial_%s" % (self.name))
        dend = nrn_section("dend_%s" % (self.name))
        term = nrn_section("term_%s" % (self.name))

        # shared properties
        for s in [initial, dend, term]:
            s.nseg = self.dend_nseg
            s.Ra = self.dend_ra
            s.insert("HHst")
            s.insert("cad")
            s.insert("Kv3")
            s.insert("HT")
            s.insert("Kv3_3")
            s.gleak_HHst = self.dend_gleak_hh  # (S/cm2)
            s.eleak_HHst = self.dend_eleak_hh
            s.NF_HHst = self.dend_nz_factor
            s.seed_HHst = self.nz_seed

        initial.diam = self.initial_dend_diam
        initial.L = self.initial_dend_l
        dend.diam = self.dend_diam
        dend.L = self.dend_l
        term.diam = self.term_diam
        term.L = self.term_l

        # turn off calcium and sodium everywhere but the terminal
        for s in [initial, dend]:
            s.gtbar_HHst = 0
            s.glbar_HHst = 0
            s.gnabar_HHst = 0

        # non terminal potassium densities
        initial.gkbar_HHst = self.initial_k
        initial.gkmbar_HHst = self.initial_km
        initial.gbar_Kv3 = self.initial_kv3
        initial.gbar_HT = self.initial_kv3_1
        initial.gbar_Kv3_3 = self.initial_kv3_3
        dend.gkbar_HHst = self.dend_k
        dend.gkmbar_HHst = self.dend_km
        dend.gbar_Kv3 = self.dend_kv3
        dend.gbar_HT = self.dend_kv3_1
        dend.gbar_Kv3_3 = self.dend_kv3_3
        term.gkbar_HHst = self.term_k
        term.gkmbar_HHst = self.term_km
        term.gbar_Kv3 = self.dend_kv3
        term.gbar_HT = self.term_kv3_1
        term.gbar_Kv3_3 = self.term_kv3_3

        dend.insert("can")
        dend.insert("newCaP1")
        dend.gcanbar_can = self.dend_can
        dend.pcabar_newCaP1 = self.dend_cap

        # terminal active properties
        term.insert("can")
        term.insert("newCaP1")
        term.gnabar_HHst = self.term_na
        term.gtbar_HHst = self.term_cat
        term.glbar_HHst = self.term_cal
        term.gtbar_HHst = self.term_cat
        term.glbar_HHst = self.term_cal
        term.gcanbar_can = self.term_can
        term.pcabar_newCaP1 = self.term_cap
        term.insert("nav1p8")
        term.insert("nav1p9")
        term.gbar_nav1p8 = self.term_nav1_8
        term.gbar_nav1p9 = self.term_nav1_9

        dend.connect(initial)
        term.connect(dend)

        all_sinks = []
        for (loc, n_orders) in zip(self.sink_dend_locs, self.sink_orders):
            pos = np.round(max(0, loc - self.initial_dend_l) / self.dend_l, decimals=5)
            sink, _ = self.sink_branch(str(loc), n_orders)
            sink.connect(dend(pos))
            all_sinks.append(sink)

        return initial, dend, term, all_sinks

    def create_synapses(self):
        # create *named* hoc objects for each synapse (for gui compatibility)
        h(
            "objref sust_bps_%s[%i], trans_bps_%s[%i]"
            % (
                self.name,
                len(self.bp_locs["sust"]),
                self.name,
                len(self.bp_locs["trans"]),
            )
        )

        # complete synapses are made up of a NetStim, Syn, and NetCon
        self.bps = {
            "sust": {
                # "stim": [],
                "syn": getattr(h, "sust_bps_%s" % self.name),
                "con": [],
            },
            "trans": {
                # "stim": [],
                "syn": getattr(h, "trans_bps_%s" % self.name),
                "con": [],
            },
        }

        for (k, syns), locs, props in zip(
            self.bps.items(), self.bp_locs.values(), self.bp_props.values()
        ):
            for i in range(len(locs)):
                # access hoc compartment and calculate fractional position
                # along dendritic section (0 -> 1)
                if locs[i] <= self.initial_dend_l:
                    self.initial.push()
                    pos = np.round(locs[i] / self.initial_dend_l, decimals=5)
                elif locs[i] <= (self.initial_dend_l + self.dend_l):
                    self.dend.push()
                    pos = np.round(
                        max(0, locs[i] - self.initial_dend_l) / self.dend_l, decimals=5
                    )
                else:
                    self.term.push()
                    pos = np.round(
                        max(0, locs[i] - self.initial_dend_l - self.dend_l)
                        / self.term_l,
                        decimals=5,
                    )

                # Synapse object (source of conductance)
                syns["syn"][i] = h.Exp2Syn(pos)
                syns["syn"][i].tau1 = props["tau1"]
                syns["syn"][i].tau2 = props["tau2"]
                syns["syn"][i].e = props["rev"]

                # NetQuanta (wraps NetCon) for scheduling and applying conductance events
                syns["con"].append(
                    NetQuanta(syns["syn"][i], props["weight"], delay=props["delay"])
                )

                h.pop_section()  # remove section from access stack

    def init_synapses(self):
        """Initialize the events in each NetQuanta (NetCon wrapper)."""
        for syns in self.bps.values():
            for nq in syns["con"]:
                nq.initialize()

    def clear_synapses(self):
        for syns in self.bps.values():
            for nq in syns["con"]:
                nq.clear_events()

    def calc_xy_locs(self):
        """Origin of the arena is (0, 0), so the dendrite is positioned with
        that as the centre. X locations are calculated based on the distances of
        each bipolar cell from the soma. Dendrite origins are refer to the 0
        position of the section, meaning it extends from that in a different
        direction depending on the orientation of the SAC."""
        pre_term_l = self.initial_dend_l + self.dend_l
        total_l = pre_term_l + self.term_l
        dir_sign = 1 if self.forward else -1
        o_x, o_y = self.origin
        self.initial_dend_x_origin = o_x + (
            (total_l + self.gaba_props["locs"][0]) / -2 * dir_sign
        )
        self.dend_x_origin = self.initial_dend_x_origin + (
            self.initial_dend_l * dir_sign
        )
        self.term_x_origin = self.initial_dend_x_origin + (pre_term_l * dir_sign)
        self.soma_x_origin = self.initial_dend_x_origin + (
            self.soma_l / 2 * dir_sign * -1
        )
        self.gaba_x_loc = self.initial_dend_x_origin + (total_l * dir_sign)
        self.bp_xy_locs = {
            k: {
                "x": [dir_sign * l + self.initial_dend_x_origin for l in locs],
                "y": [o_y for _ in locs],
            }
            for k, locs in self.bp_locs.items()
        }
        return self.bp_xy_locs

    def create_neuron(self):
        # create compartments (using parameters in self.__dict__)
        self.soma = self.create_soma()
        self.initial, self.dend, self.term, self.sinks = self.create_dend()
        self.all_dends = [self.initial, self.dend, self.term] + self.sinks
        self.initial.connect(self.soma)
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

    def bar_onsets(self, bar, rad_direction, uniform=False):
        # bare base onset with added jitter
        for k, ts in self.bar_sweep(bar, rad_direction).items():
            rel_k = "trans" if uniform else k
            for t, nq in zip(ts, self.bps[k]["con"]):
                jit = self.rand.normal(0, 1)
                nq.events = self.bp_releasers[rel_k](
                    self.np_rng, t + self.bp_jitter * jit
                )

    def flash(self, onset, uniform=False):
        for k in self.bps.keys():
            rel_k = "trans" if uniform else k
            for nq in self.bps[k]["con"]:
                jit = self.rand.normal(0, 1)
                nq.events = self.bp_releasers[rel_k](
                    self.np_rng, onset + self.bp_jitter * jit
                )

    def update_noise(self):
        for s in self.all_dends:
            s.seed_HHst = self.nz_seed
            self.nz_seed += 1


class SacPair:
    def __init__(self, sac_params=None):
        self.sacs = {
            "a": Sac("a", params=sac_params),
            "b": Sac("b", forward=False, params=sac_params),
        }
        self.wire_gaba()

    def wire_gaba(self):
        self.gaba_syns = {}
        for n, sac in self.sacs.items():
            self.gaba_syns[n] = {"syn": [], "con": []}
            for loc in sac.gaba_props["locs"]:
                # TODO: compare this to updated BP positioning code
                if loc < sac.initial_dend_l:
                    sac.initial.push()
                    pos = np.round(loc / sac.initial_dend_l, decimals=5)
                else:
                    sac.dend.push()
                    pos = np.round((loc - sac.initial_dend_l) / sac.dend_l, decimals=5,)
                syn = h.Exp2Syn(pos)
                syn.tau1 = sac.gaba_props["tau1"]
                syn.tau2 = sac.gaba_props["tau2"]
                syn.e = sac.gaba_props["rev"]
                self.gaba_syns[n]["syn"].append(syn)
                h.pop_section()
        for pre, post in [("a", "b"), ("b", "a")]:
            pre_sac = self.sacs[pre]
            pre_sac.term.push()

            for post_loc, post_syn in zip(
                self.sacs[post].gaba_props["locs"], self.gaba_syns[post]["syn"]
            ):
                tip_dist = post_loc - self.sacs[post].gaba_props["locs"][0]
                if tip_dist < sac.term_l:
                    pre_sec = pre_sac.term
                    pos = np.round(1 - tip_dist / sac.term_l, decimals=5)
                else:
                    pre_sec = pre_sac.dend
                    pos = np.round(
                        (sac.dend_l - (tip_dist - sac.term_l)) / sac.dend_l, decimals=5,
                    )

                pre_sec.push()
                ref = (
                    pre_sec(pos)._ref_v
                    if not pre_sac.gaba_props["calcium"]
                    else pre_sec(pos)._ref_cai
                )
                self.gaba_syns[pre]["con"].append(
                    h.NetCon(
                        ref,
                        post_syn,
                        float(pre_sac.gaba_props["thresh"]),
                        float(pre_sac.gaba_props["delay"]),
                        float(pre_sac.gaba_props["weight"]),
                    )
                )
            h.pop_section()

    def get_params_dict(self):
        return {n: sac.get_params_dict() for n, sac in self.sacs.items()}

    def flash(self, onset, uniform=False):
        for sac in self.sacs.values():
            sac.flash(onset, uniform=uniform)

    def bar_onsets(self, stim, dir_idx, uniform=False):
        for sac in self.sacs.values():
            sac.bar_onsets(stim, dir_idx, uniform=uniform)

    def update_noise(self):
        for sac in self.sacs.values():
            sac.update_noise()

    def init_bipolars(self):
        for sac in self.sacs.values():
            sac.init_synapses()

    def clear_bipolar_events(self):
        for sac in self.sacs.values():
            sac.clear_synapses()


class Runner:
    def __init__(self, model, data_path=""):
        self.data_path = data_path
        if not os.path.isdir(data_path):
            os.makedirs(data_path)
        self.model = model

        # hoc environment parameters
        # self.tstop = 4000  # [ms]
        self.tstop = 6000  # [ms]
        self.steps_per_ms = 1  # [10 = 10kHz]
        self.dt = 1  # [ms, .1 = 10kHz]
        # self.v_init = -70
        self.v_init = -60
        self.celsius = 36.9
        self.set_hoc_params()

        self.config_stimulus()
        self.place_electrodes()
        self.orig_gaba_weights = None
        self.orig_bp_props = None
        self.orig_bp_vel_scaling = None

        self.vc = None
        self.vc_rec = None
        self.vc_data = None
        self.empty_vc_data = None

        # schedules the events for the NetCons during model initialization
        self.initialize_handler = h.FInitializeHandler(self.model.init_bipolars)

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
            "start_time": 0.0,  # vel -> start: .25 -> -900; .5 -> -400
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

    def get_params_dict(self):
        params = self.__dict__.copy()
        # remove the non-param entries (model objects)
        for key in [
            "model",
            "recs",
            "data",
            "dir_labels",
            "dir_rads",
            "dirs",
            "dir_inds",
            "circle",
            "empty_data",
            "empty_vc_data",
            "orig_gaba_weights",
            "orig_bp_props",
            "orig_bp_vel_scaling",
            "initialize_handler",
            "vc",
            "vc_rec",
            "vc_data",
        ]:
            params.pop(key)
        return params

    def run(self, stim, dir_idx):
        """Initialize model, set synapse onset and release numbers, update
        membrane noise seeds and run the model. Calculate somatic response and
        return to calling function."""
        h.init()
        self.model.bar_onsets(
            stim, self.dir_rads[dir_idx], uniform=self.orig_bp_props is not None
        )
        self.model.update_noise()

        self.clear_recordings()
        h.run()
        self.dump_recordings()
        self.model.clear_bipolar_events()

    def flash_run(self, onset, n_trials=10):

        model_params = self.model.get_params_dict()  # for logging
        exp_params = self.get_params_dict()

        for j in range(n_trials):
            print("trial %d..." % j, end=" ", flush=True)
            h.init()
            self.model.flash(onset, uniform=self.orig_bp_props is not None)
            self.model.update_noise()

            self.clear_recordings()
            h.run()
            self.dump_recordings()
            self.model.clear_bipolar_events()

        if self.vc_data is None:
            all_recs = self.data
        else:
            all_recs = {**self.data, "vc": self.vc_data}
            self.vc_data = deepcopy(self.empty_vc_data)

        data = {
            "model_params": json.dumps(model_params),
            "exp_params": json.dumps(exp_params),
            "data": self.stack_data(all_recs, n_trials, 1),
        }
        self.data = deepcopy(self.empty_data)  # clear out stored data

        return data

    def velocity_run(
        self,
        velocities=[0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2],
        n_trials=1,
        prefix="",
    ):
        """"""
        n_vels = len(velocities)
        stim = {"type": "bar", "dir": 0}
        model_params = self.model.get_params_dict()  # for logging
        exp_params = self.get_params_dict()
        exp_params["velocities"] = velocities

        for j in range(n_trials):
            print("trial %d..." % j, end=" ", flush=True)

            for i in range(n_vels):
                print("%.2f" % velocities[i], end=" ", flush=True)
                self.light_bar["speed"] = velocities[i]
                self.scale_bps(velocities[i])
                self.scale_gaba(velocities[i])
                self.run(self.light_bar, 3)  # index of 0 degrees

            print("")  # next line

        if self.vc_data is None:
            all_recs = self.data
        else:
            all_recs = {**self.data, "vc": self.vc_data}
            self.vc_data = deepcopy(self.empty_vc_data)

        self.unscale_bps()
        data = {
            "model_params": json.dumps(model_params),
            "exp_params": json.dumps(exp_params),
            "data": self.stack_data(all_recs, n_trials, n_vels),
        }
        self.data = deepcopy(self.empty_data)  # clear out stored data

        return data

    def scale_bps(self, vel):
        for n, sac in self.model.sacs.items():
            for props, locs, loc_scale, vel_scale, bps in zip(
                sac.bp_props.values(),
                sac.bp_locs.values(),
                sac.bp_loc_scaling.values(),
                sac.bp_vel_scaling.values(),
                sac.bps.values(),
            ):
                for i, loc in enumerate(locs):
                    w = props["weight"]
                    t1 = props["tau1"]
                    t2 = props["tau2"]
                    if sac.loc_scaling:
                        w = loc_scale["weight"](loc, w)
                        t1 = loc_scale["tau1"](loc, t1)
                        t2 = loc_scale["tau2"](loc, t2)
                    if sac.vel_scaling:
                        w = vel_scale["weight"](vel, w)
                        t1 = vel_scale["tau1"](vel, t1)
                        t2 = vel_scale["tau2"](vel, t2)
                    # bps["con"][i].weight[0] = w
                    bps["con"][i].weight = w
                    bps["syn"][i].tau1 = t1
                    bps["syn"][i].tau2 = t2

    def unscale_bps(self):
        for n, sac in self.model.sacs.items():
            for props, bps in zip(sac.bp_props.values(), sac.bps.values()):
                for i in range(len(bps["syn"])):
                    # bps["con"][i].weight[0] = props["weight"]
                    bps["con"][i].weight = props["weight"]
                    bps["syn"][i].tau1 = props["tau1"]
                    bps["syn"][i].tau2 = props["tau2"]

    def scale_gaba(self, vel):
        for (n, sac), syn in zip(
            self.model.sacs.items(), self.model.gaba_syns.values()
        ):
            for con in syn["con"]:
                con.weight[0] = sac.gaba_vel_scaling(vel, sac.gaba_props["weight"])

    def unscale_bps(self):
        for n, sac in self.model.sacs.items():
            for props, bps in zip(sac.bp_props.values(), sac.bps.values()):
                for i in range(len(bps["syn"])):
                    # bps["con"][i].weight[0] = props["weight"]
                    bps["con"][i].weight = props["weight"]
                    bps["syn"][i].tau1 = props["tau1"]
                    bps["syn"][i].tau2 = props["tau2"]

    def remove_gaba(self):
        if self.orig_gaba_weights is None:
            self.orig_gaba_weights = {}
            for (n, sac), syn in zip(
                self.model.sacs.items(), self.model.gaba_syns.values()
            ):
                self.orig_gaba_weights[n] = float(sac.gaba_props["weight"])
                sac.gaba_props["weight"] = 0.0
                for con in syn["con"]:
                    con.weight[0] = 0.0

    def restore_gaba(self):
        if self.orig_gaba_weights is not None:
            for (n, sac), syn in zip(
                self.model.sacs.items(), self.model.gaba_syns.values()
            ):
                self.model.sacs[n].gaba_props["weight"] = float(
                    self.orig_gaba_weights[n]
                )
                for con in syn["con"]:
                    con.weight[0] = float(self.orig_gaba_weights[n])
            self.orig_gaba_weights = None

    def unify_bps(self, taus):
        if self.orig_bp_props is None:
            self.orig_bp_props = {}
            self.orig_bp_vel_scaling = {}
            for n, sac in self.model.sacs.items():
                self.orig_bp_props[n] = deepcopy(sac.bp_props)
                self.orig_bp_vel_scaling[n] = deepcopy(sac.bp_vel_scaling)
                sac.bp_props = {k: {**v, **taus} for k, v in sac.bp_props.items()}
                sac.bp_vel_scaling["sust"] = sac.bp_vel_scaling["trans"]
                for bps in sac.bps.values():
                    for syn in bps["syn"]:
                        syn.tau1 = taus["tau1"]
                        syn.tau2 = taus["tau2"]

    def restore_bps(self):
        if self.orig_bp_props is not None:
            for n, sac in self.model.sacs.items():
                sac.bp_props = deepcopy(self.orig_bp_props[n])
                sac.bp_vel_scaling = deepcopy(self.orig_bp_vel_scaling[n])
                for bps, props in zip(sac.bps.values(), sac.bp_props.values()):
                    for syn in bps["syn"]:
                        syn.tau1 = props["tau1"]
                        syn.tau2 = props["tau2"]
            self.orig_bp_props, self.orig_bp_vel_scaling = None, None

    def velocity_mechanism_run(
        self,
        velocities=[0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2],
        mech_trials=1,
        uniform_props={"tau1": 1, "tau2": 12,},
        conds={"control", "no_gaba", "uniform", "no_mech"},
    ):
        data = {}
        if "control" in conds:
            print("Control run:")
            data["control"] = self.velocity_run(
                velocities=velocities, n_trials=mech_trials
            )

        if "no_gaba" in conds:
            print("No GABA run:")
            self.remove_gaba()
            data["no_gaba"] = self.velocity_run(
                velocities=velocities, n_trials=mech_trials
            )
            self.restore_gaba()

        if "uniform" in conds:
            print("Uniform Bipolar run:")
            self.unify_bps(uniform_props)
            data["uniform"] = self.velocity_run(
                velocities=velocities, n_trials=mech_trials
            )
            self.restore_bps()

        if "no_mech" in conds:
            print("No Mechanism run:")
            self.remove_gaba()
            self.unify_bps(uniform_props)
            data["no_mechs"] = self.velocity_run(
                velocities=velocities, n_trials=mech_trials
            )
            self.restore_gaba()
            self.restore_bps()

        return data

    def bp_distribution_run(
        self, distributions, dist_trials=1, mirror=False, **velocity_kwargs
    ):
        n_bps = {k: len(locs) for k, locs in self.model.sacs["a"].bp_locs.items()}
        data = {}
        for i in range(dist_trials):
            for j, sac in enumerate(self.model.sacs.values()):
                if not mirror or not j:
                    locs = {
                        k: dist(np.random.uniform(size=n_bps[k])).tolist()
                        for k, dist in distributions.items()
                    }
                sac.bp_locs = locs
                sac.calc_xy_locs()
            data[i] = self.velocity_mechanism_run(**velocity_kwargs)

        return data

    def place_electrodes(self):
        self.recs = {"soma": {}, "term": {}, "gaba": {}, "bps": {}}
        self.data = {"soma": {}, "term": {}, "gaba": {}, "bps": {}}
        for n, sac in self.model.sacs.items():
            for p in ["soma", "term"]:
                self.recs[p][n] = {
                    "v": h.Vector(),
                    "ica": h.Vector(),
                    "cai": h.Vector(),
                }
                self.data[p][n] = {k: [] for k in self.recs[p][n].keys()}
                for k, vec in self.recs[p][n].items():
                    vec.record(
                        getattr(
                            getattr(sac, p)(0.5 if p == "soma" else 1), "_ref_%s" % k
                        )
                    )

            self.recs["bps"][n], self.data["bps"][n] = {}, {}
            for (k, bps) in sac.bps.items():
                self.recs["bps"][n][k], self.data["bps"][n][k] = {}, {}
                for i, syn in enumerate(bps["syn"]):
                    self.recs["bps"][n][k][i], self.data["bps"][n][k][i] = {}, {}
                    for s in ["i", "g"]:
                        self.recs["bps"][n][k][i][s] = h.Vector()
                        self.data["bps"][n][k][i][s] = []
                        self.recs["bps"][n][k][i][s].record(getattr(syn, "_ref_%s" % s))

            self.recs["gaba"][n], self.data["gaba"][n] = {}, {}
            for i, syn in enumerate(self.model.gaba_syns[n]["syn"]):
                self.recs["gaba"][n][i] = {"i": h.Vector(), "g": h.Vector()}
                self.data["gaba"][n][i] = {"i": [], "g": []}
                self.recs["gaba"][n][i]["i"].record(syn._ref_i)
                self.recs["gaba"][n][i]["g"].record(syn._ref_g)
        self.empty_data = deepcopy(self.data)  # for resetting

    def place_vc(self, targets=["a", "b"], hold=-60, block_vgcs=True):
        self.vc, self.vc_rec, self.vc_data = {}, {}, {}

        for n in targets:
            sac = self.model.sacs[n]
            sac.soma.push()
            vc = nrn_objref("vc")
            vc = h.SEClamp(0.5)
            h.pop_section()

            # hold target voltage for entire duration
            vc.dur1 = h.tstop
            vc.dur2 = 0.0
            vc.dur3 = 0.0
            vc.amp1 = hold

            if block_vgcs:
                for sec in [sac.soma, sac.initial, sac.dend, sac.term]:
                    sec.gnabar_HHst = 0.0
                    sec.gkbar_HHst = 0.0
                    sec.gkmbar_HHst = 0.0

            vc_rec = h.Vector()
            vc_rec.record(vc._ref_i)

            self.vc_rec[n] = vc_rec
            self.vc_data[n] = []
            self.vc[n] = vc

        self.empty_vc_data = deepcopy(self.vc_data)

    def dump_recordings(self):
        for (p, rs), ds in zip(self.recs.items(), self.data.values()):
            for n in self.model.sacs.keys():
                if p in ["soma", "term"]:
                    for k, vec in rs[n].items():
                        ds[n][k].append(np.round(vec, decimals=3))
                elif p == "bps":
                    for typ, bp_rs in rs[n].items():
                        for i, r in bp_rs.items():
                            for k in ["i", "g"]:
                                ds[n][typ][i][k].append(np.array(r[k]))
                elif p == "gaba":
                    for i in rs[n].keys():
                        for k in ["i", "g"]:
                            ds[n][i][k].append(np.array(rs[n][i][k]))
                else:
                    for k in ["i", "g"]:
                        ds[n][k].append(np.array(rs[n][k]))

        if self.vc_rec is not None:
            for n in self.vc_rec.keys():
                self.vc_data[n].append(np.array(self.vc_rec[n]))

    def clear_recordings(self):
        """Clear out all of the recording vectors in the recs dict, accounting for
        arbitrary levels of nesting, as long as all of the leaves are hoc vectors."""

        def loop(rs):
            for r in rs:
                if type(r) == dict:
                    loop(r.values())
                else:
                    r.resize(0)

        if self.vc_rec is not None:
            loop(self.vc_rec.values())

        loop(self.recs.values())

    @staticmethod
    def stack_data(data, n_trials, n_vels):
        def stacker(val):
            if type(val) == dict:
                return {k: stacker(v) for k, v in val.items()}
            else:
                return stack_trials(n_trials, n_vels, val)

        return {k: stacker(v) for k, v in data.items()}

    def isolated_input_battery(self, times, n_trials=5):
        """March through each of the transient bipolar inputs (ideally regularly
        spaced along the dendrite) and play the given train of event timings for
        `n_trials` repetitions."""
        inputs = self.model.sacs["a"].bps["trans"]["con"]
        for i, nq in enumerate(inputs):
            print("synapse %i..." % i, end=" ", flush=True)
            h.init()
            nq.events = times
            for j in range(n_trials):
                print("%i" % j, end=" ", flush=True)
                self.model.update_noise()
                self.clear_recordings()
                h.run()
                self.dump_recordings()

            self.model.clear_bipolar_events()
            print("")

        if self.vc_data is None:
            all_recs = self.data
        else:
            all_recs = {**self.data, "vc": self.vc_data}

        data = {
            "model_params": json.dumps(self.model.get_params_dict()),
            "exp_params": json.dumps(self.get_params_dict()),
            "data": self.stack_data(all_recs, len(inputs), n_trials),
        }
        return data


if __name__ == "__main__":
    # h.xopen("sac_pair.ses")  # open neuron gui session
    h.xopen("pair.ses")  # open neuron gui session
    base_path = "/mnt/Data/NEURONoutput/sac_sac/"
    data_path = base_path + "test_run/"
    os.makedirs(data_path, exist_ok=True)

    model = SacPair()
    runner = Runner(model, data_path=data_path)
    # data = runner.velocity_run()
    # pack_hdf(data_path + "test", data)

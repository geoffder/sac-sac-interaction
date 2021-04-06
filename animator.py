import os
import h5py as h5
import json

import numpy as np  # arrays
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import TextBox, Slider
from matplotlib import cm

# local imports (found in this repo)
from utils import *


class SacSacAnimator:
    def __init__(
        self,
        exps,
        exp_params,
        model_params,
        y_off={
            "a": 2,
            "b": -2
        },
        bp_offset=2,
        bp_width=3,
        bp_height=5
    ):
        self.exp_params = exp_params
        self.model_params = model_params  # TODO: improve to a dict of conditions
        self.exps = exps
        self.y_off = y_off
        self.bp_offset = bp_offset
        self.bp_width = bp_width
        self.bp_height = bp_height
        self.schemes = self.build_schemes()
        self.cond = "control"
        self.vel_idx = 0
        self.t_idx = 0
        self.rec_xaxis = np.arange(
            0, exp_params["tstop"] + exp_params["dt"], exp_params["dt"]
        )
        self.avg_exps = apply_to_data(lambda a: np.mean(a, axis=0), exps)
        self.min_exps = apply_to_data(np.min, stack_pair_data(self.avg_exps))
        self.max_exps = apply_to_data(np.max, stack_pair_data(self.avg_exps))
        self.velocities = exp_params["velocities"]
        self.conds = [c for c in self.exps.keys()]
        self.cmap = cm.get_cmap("jet", 12)

    def build_schemes(self):
        return {
            n: {
                "soma":
                    patches.Circle(
                        (ps["soma_x_origin"], ps["origin"][1] + self.y_off[n]),
                        ps["soma_diam"] / 2,
                    ),
                "dend":
                    patches.Rectangle(
                        (
                            ps["dend_x_origin"] -
                            (ps["dend_l"] if not ps["forward"] else 0), ps["origin"][1] -
                            (ps["dend_diam"] / 2) + self.y_off[n]
                        ),
                        ps["dend_l"],
                        1,  # ps["dend_diam"],
                        fill=True,
                    ),
                "term":
                    patches.Rectangle(
                        (
                            ps["term_x_origin"] -
                            (ps["term_l"] if not ps["forward"] else 0), ps["origin"][1] -
                            (ps["dend_diam"] / 2) + self.y_off[n]
                        ),
                        ps["term_l"],
                        1,  # ps["dend_diam"],
                        fill=True,
                    ),
                "gaba":
                    patches.Arrow(
                        ps["gaba_x_loc"],
                        ps["origin"][1] + self.y_off[n],
                        0,
                        self.y_off[n] * -1.5,
                        width=10,
                    ),
                "bps":
                    {
                        k: [
                            patches.Rectangle(
                                (
                                    x - self.bp_width / 2, (
                                        y + (
                                            self.bp_offset
                                            if ps["forward"] else self.bp_offset * -1
                                        ) + self.y_off[n] -
                                        (self.bp_height if not ps["forward"] else 0)
                                    )
                                ),
                                self.bp_width,
                                self.bp_height,
                            ) for x, y in zip(ls["x"], ls["y"])
                        ]
                        for k, ls in ps["bp_xy_locs"].items()
                    },
            }
            for n, ps in self.model_params.items()
        }

    def apply_patches(self, ax):
        def loop(ps):
            if type(ps) == dict:
                for p in ps.values():
                    loop(p)
            elif type(ps) == list:
                for p in ps:
                    loop(p)
            elif isinstance(ps, patches.Patch):
                ax.add_patch(ps)
            else:
                raise TypeError("All leaves must be patches.")

        loop(self.schemes)

    def label_bps(self, ax, x_off=-3, y_off=1):
        for n, s in self.schemes.items():
            for k, bps in s["bps"].items():
                for bp in bps:
                    ax.text(
                        bp.get_x() + x_off,
                        bp.get_y() + (
                            (bp.get_height() +
                             y_off) if self.model_params[n]["forward"] else (-2 - y_off)
                        ),
                        k,
                    )

    def build_animation_fig(self, **plot_kwargs):
        if hasattr(self, "fig"):
            del (self.fig, self.ax, self.cond_slider, self.vel_slider, self.time_slider)
        if "gridspec_kw" not in plot_kwargs:
            plot_kwargs["gridspec_kw"] = {"height_ratios": [.45, .05, .05, .05, .4]}
        self.fig, self.ax = plt.subplots(5, **plot_kwargs)
        self.scheme_ax, self.cond_slide_ax, self.vel_slide_ax, self.time_slide_ax, self.rec_ax = self.ax
        self.build_cond_slide_ax()
        self.build_vel_slide_ax()
        self.build_time_slide_ax()
        self.build_rec_ax(-70, -40)
        self.build_scheme_ax()
        self.update_rec()
        self.update_scheme()
        self.connect_events()
        self.fig.tight_layout()
        return self.fig, self.ax

    def build_cond_slide_ax(self):
        self.cond_slider = Slider(
            self.cond_slide_ax,
            "",
            valmin=0,
            valmax=(len(self.conds) - 1),
            valinit=0,
            valstep=1,
            valfmt="%.0f"
        )
        self.cond_slide_ax.set_title("Condition = %s" % self.cond)

    def build_vel_slide_ax(self):
        self.vel_slider = Slider(
            self.vel_slide_ax,
            "",
            valmin=0,
            valmax=(len(self.velocities) - 1),
            valinit=0,
            valstep=1,
            valfmt="%.0f"
        )
        self.vel_slide_ax.set_title(
            "Velocity = %.2f mm/s" % self.velocities[self.vel_idx]
        )

    def build_time_slide_ax(self):
        self.time_slider = Slider(
            self.time_slide_ax,
            "Time (ms)",
            valmin=0,
            valmax=self.exp_params["tstop"],
            valinit=0,
            valstep=self.exp_params["dt"],
            valfmt="%.3f"
        )

    def bar_loc(self, t):
        bar = self.exp_params["light_bar"]
        travel_time = max(0, t - bar["start_time"])
        return travel_time * self.velocities[self.vel_idx] + bar["x_start"]

    def build_scheme_ax(self):
        self.scheme_ax.set_xlim(-180, 180)
        self.scheme_ax.set_ylim(-15, 15)
        self.scheme_ax.set_xlabel("μm")
        self.scheme_ax.set_ylabel("μm")
        self.apply_patches(self.scheme_ax)
        self.label_bps(self.scheme_ax)
        bar_x = self.bar_loc(self.rec_xaxis[self.t_idx])
        self.bar_rect = patches.Rectangle((bar_x, -15), 1, 30, color="black")
        self.scheme_ax.add_patch(self.bar_rect)

    def build_rec_ax(self, ymin, ymax):
        self.rec_ax.set_ylabel("Terminal Voltage (mV)")
        self.rec_ax.set_xlabel("Time (ms)")
        self.term_lines = {
            n: self.rec_ax.plot(rs["v"][self.vel_idx])[0]
            for n, rs in self.avg_exps[self.cond]["term"].items()
        }
        self.t_marker = self.rec_ax.plot(
            [self.rec_xaxis[self.t_idx] for _ in range(2)],
            [ymin, ymax],
            linestyle="--",
            c="black",
        )[0]

    def on_cond_slide(self, v):
        self.cond = self.conds[int(v)]
        self.cond_slide_ax.set_title("Condition = %s" % self.cond)
        self.update_scheme()
        self.update_rec()

    def on_vel_slide(self, v):
        self.vel_idx = int(v)
        self.vel_slide_ax.set_title(
            "Velocity = %.2f mm/s" % self.velocities[self.vel_idx]
        )
        self.update_scheme()
        self.update_rec()

    def on_time_slide(self, v):
        self.t_idx = nearest_index(self.rec_xaxis, v)
        self.update_scheme()
        self.update_rec()

    def connect_events(self):
        self.cond_slider.on_changed(self.on_cond_slide)
        self.vel_slider.on_changed(self.on_vel_slide)
        self.time_slider.on_changed(self.on_time_slide)

    def update_rec(self):
        for i, (n, line) in enumerate(self.term_lines.items()):
            line.set_ydata(self.avg_exps[self.cond]["term"][n]["v"][self.vel_idx])

        self.t_marker.set_xdata(self.rec_xaxis[self.t_idx])

    def update_scheme(self):
        ex = self.avg_exps[self.cond]
        mins = self.min_exps[self.cond]
        maxs = self.max_exps[self.cond]
        self.bar_rect.set_x(self.bar_loc(self.rec_xaxis[self.t_idx]))
        for n, s in self.schemes.items():
            s["soma"].set_color(
                self.cmap(
                    (ex["soma"][n]["v"][self.vel_idx, self.t_idx] - mins["soma"]["v"]) /
                    (maxs["soma"]["v"] - mins["soma"]["v"])
                )
            )
            s["term"].set_color(
                self.cmap(
                    (ex["term"][n]["v"][self.vel_idx, self.t_idx] - mins["term"]["v"]) /
                    (maxs["term"]["v"] - mins["term"]["v"])
                )
            )
            # GABA arrow coming from pre-synaptic side, so flip n
            s["gaba"].set_color(
                self.cmap(
                    ex["gaba"]["b" if n == "a" else "a"]["g"][self.vel_idx, self.t_idx] /
                    maxs["gaba"]["g"]
                )
            )
            for k, bps in s["bps"].items():
                for i, b in enumerate(bps):
                    b.set_color(
                        self.cmap(
                            ex["bps"][n][k][i]["g"][self.vel_idx, self.t_idx] /
                            (maxs["bps"][k][i]["g"] + .00001)
                        )
                    )

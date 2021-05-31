import os
import h5py as h5
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import TextBox, Slider
from matplotlib import cm
from matplotlib.animation import FuncAnimation

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
        bp_height=5,
        reveal_time=False,
    ):
        self.exp_params = exp_params
        self.model_params = model_params  # TODO: improve to a dict of conditions
        self.exps = exps
        # HACK: create basic summed bipolar measures for easier plotting for now
        for cond in self.exps.keys():
            self.exps[cond]["combined_bps"] = {
                n: {
                    m: np.sum(
                        [bp[m] for bps in sac.values() for bp in bps.values()], axis=0
                    )
                    for m in ["g", "i"]
                }
                for n, sac in exps[cond]["bps"].items()
            }
        self.y_off = y_off
        self.bp_offset, self.bp_width, self.bp_height = bp_offset, bp_width, bp_height
        self.reveal_time = reveal_time
        self.schemes = self.build_schemes()
        self.cond = "control"
        self.vel_idx = 0
        self.t_idx = 0
        self.rec_xaxis = np.arange(
            0, exp_params["tstop"] + exp_params["dt"], exp_params["dt"]
        )
        self.n_pts = self.rec_xaxis.size
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
                        "S" if k == "sust" else "T",
                        fontsize=12,
                    )

    def build_interactive_fig(self, **plot_kwargs):
        if hasattr(self, "fig"):
            del (self.fig, self.ax, self.cond_slider, self.vel_slider, self.time_slider)
        if "gridspec_kw" not in plot_kwargs:
            plot_kwargs["gridspec_kw"] = {
                "height_ratios": [.225, .033, .033, .033, .225, .225, .225],
                "hspace": .65
            }
        self.fig, self.ax = plt.subplots(7, **plot_kwargs)
        (
            self.scheme_ax, self.cond_slide_ax, self.vel_slide_ax, self.time_slide_ax,
            self.bp_g_ax, self.gaba_g_ax, self.term_vm_ax
        ) = self.ax
        self.build_cond_slide_ax()
        self.build_vel_slide_ax()
        self.build_time_slide_ax()
        self.build_term_vm_ax()
        self.build_gaba_g_ax()
        self.build_bp_g_ax()
        self.build_scheme_ax()
        self.update_rec_axes()
        self.update_scheme()
        self.connect_events()
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
        self.scheme_ax.set_ylabel("μm")
        self.apply_patches(self.scheme_ax)
        self.label_bps(self.scheme_ax)
        bar_x = self.bar_loc(self.rec_xaxis[self.t_idx])
        self.bar_rect = patches.Rectangle((bar_x, -15), 1, 30, color="black")
        self.scheme_ax.add_patch(self.bar_rect)

    def build_rec_ax(self, ax, loc, rec_key, ymin, ymax, xlbl=None, ylbl=None):
        ax.set_ylabel(ylbl)
        ax.set_xlabel(xlbl)
        ax.set_xlim(self.rec_xaxis.min(), self.rec_xaxis.max())
        end = (self.t_idx + 1) if self.reveal_time else self.n_pts
        lines = {
            n: ax.plot(
                self.rec_xaxis[:end], rs[rec_key][self.vel_idx][:end], label="sac %s" % n
            )[0]
            for n, rs in self.avg_exps[self.cond][loc].items()
        }
        t_marker = ax.plot(
            [self.rec_xaxis[self.t_idx] for _ in range(2)],
            [ymin, ymax],
            linestyle="--",
            c="black",
        )[0]
        ax.legend()
        return lines, t_marker

    def build_term_vm_ax(self):
        self.term_vm_lines, self.term_vm_t_marker = self.build_rec_ax(
            self.term_vm_ax, "term", "v", -70, -35, "Time (ms)", "Terminal Voltage (mV)"
        )

    def build_gaba_g_ax(self):
        self.gaba_g_lines, self.gaba_g_t_marker = self.build_rec_ax(
            self.gaba_g_ax,
            "gaba",
            "g",
            0,
            self.max_exps[self.cond]["gaba"]["g"],
            ylbl="GABA Conductance (μS)"
        )
        self.gaba_g_ax.set_xticklabels([])
        self.gaba_g_ax._shared_x_axes.join(self.gaba_g_ax, self.term_vm_ax)

    def build_bp_g_ax(self):
        self.bp_g_lines, self.bp_g_t_marker = self.build_rec_ax(
            self.bp_g_ax,
            "combined_bps",
            "g",
            0,
            self.max_exps[self.cond]["combined_bps"]["g"],
            ylbl="Total BPC Conductance (μS)"
        )
        self.bp_g_ax.set_xticklabels([])
        self.bp_g_ax._shared_x_axes.join(self.bp_g_ax, self.term_vm_ax)

    def on_cond_slide(self, v):
        self.cond = self.conds[int(v)]
        self.cond_slide_ax.set_title("Condition = %s" % self.cond)
        self.update_scheme()
        self.update_rec_axes()

    def on_vel_slide(self, v):
        self.vel_idx = int(v)
        self.vel_slide_ax.set_title(
            "Velocity = %.2f mm/s" % self.velocities[self.vel_idx]
        )
        self.update_scheme()
        self.update_rec_axes()

    def on_time_slide(self, v):
        self.t_idx = nearest_index(self.rec_xaxis, v)
        self.update_scheme()
        self.update_rec_axes()

    def connect_events(self):
        self.cond_slider.on_changed(self.on_cond_slide)
        self.vel_slider.on_changed(self.on_vel_slide)
        self.time_slider.on_changed(self.on_time_slide)

    def update_rec(self, lines, t_marker, loc, rec_key):
        end = (self.t_idx + 1) if self.reveal_time else self.n_pts
        t_marker.set_xdata(self.rec_xaxis[self.t_idx])
        for i, (n, line) in enumerate(lines.items()):
            line.set_data(
                self.rec_xaxis[:end],
                self.avg_exps[self.cond][loc][n][rec_key][self.vel_idx][:end]
            )

    def update_rec_axes(self):
        self.update_rec(self.term_vm_lines, self.term_vm_t_marker, "term", "v")
        self.update_rec(self.gaba_g_lines, self.gaba_g_t_marker, "gaba", "g")
        self.update_rec(self.bp_g_lines, self.bp_g_t_marker, "combined_bps", "g")

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

    def build_animation_fig(self, **plot_kwargs):
        if hasattr(self, "fig"):
            del (self.fig, self.ax, self.cond_slider, self.vel_slider, self.time_slider)
        if "gridspec_kw" not in plot_kwargs:
            plot_kwargs["gridspec_kw"] = {
                "height_ratios": [.25, .25, .25, .25],
                "hspace": .2
            }
        self.fig, self.ax = plt.subplots(4, **plot_kwargs)
        (self.scheme_ax, self.bp_g_ax, self.gaba_g_ax, self.term_vm_ax) = self.ax
        self.build_term_vm_ax()
        self.build_gaba_g_ax()
        self.build_bp_g_ax()
        self.build_scheme_ax()
        self.update_rec_axes()
        self.update_scheme()
        return self.fig, self.ax

    def play_velocity_exp(self, cond):
        pass

    def create_vel_gifs(self, out_path, n_frames, vel_idx=0, dt=10, dpi=100, gif_step=30):
        os.makedirs(out_path, exist_ok=True)
        self.term_vm_ax.set_xlim(0, n_frames * dt)
        self.vel_idx = vel_idx

        def update(t):
            self.t_idx = nearest_index(self.rec_xaxis, t)
            self.update_rec_axes()
            self.update_scheme()

        for cond in self.avg_exps.keys():
            self.cond = cond
            anim = FuncAnimation(
                self.fig, update, frames=np.arange(n_frames) * dt, interval=gif_step
            )
            name = os.path.join(
                out_path, "%s_vel%.2f.gif" % (cond, self.velocities[vel_idx])
            )
            anim.save(name, dpi=dpi, writer="imagemagick")

        self.term_vm_ax.set_xlim(0, self.rec_xaxis.max())

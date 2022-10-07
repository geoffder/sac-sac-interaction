import os
import h5py as h5
from utils import pack_dataset

from copy import deepcopy
import multiprocessing

import numpy as np
import sac_pair


def distribution_run(
    save_path,
    sac_params,
    sust_f,
    trans_f,
    n_sust=6,
    n_trans=12,
    dist_trials=16,
    batch=8,
    mech_trials=1,
    conds={"no_gaba"},
    velocities=[0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1, 1.25, 1.5],
):
    global distrib_repeat  # required to allow pickling for Pool

    def distrib_repeat(i):
        params = deepcopy(sac_params)
        rng = np.random.default_rng(i)
        params["bp_locs"] = {
            "sust": sust_f(rng.uniform(size=n_sust)).tolist(),
            "trans": trans_f(rng.uniform(size=n_trans)).tolist(),
        }
        model = sac_pair.SacPair(sac_params=params, seed=i)
        runner = sac_pair.Runner(model)
        data = runner.velocity_mechanism_run(
            velocities=velocities, conds=conds, mech_trials=mech_trials, quiet=True,
        )
        return data

    with multiprocessing.Pool(batch) as pool, h5.File(save_path, "w") as pckg:
        idx = 0
        while idx < dist_trials:
            n = min(batch, dist_trials - idx)
            print(
                "distribution trials %i to %i (of %i)..."
                % (idx + 1, idx + n, dist_trials),
                flush=True,
            )
            res = pool.map(distrib_repeat, [idx + i for i in range(n)])
            for i in range(n):
                pack_dataset(pckg, {str(idx + i): res[0]})
                del res[0]  # delete head
            idx = idx + n
    print("Done!")

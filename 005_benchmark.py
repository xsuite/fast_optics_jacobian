import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd

from utils import ir8_optics, switch_to_ad, switch_to_fd, load_hllhc_b1

# Import custom timing module
from pyprof import timing

COUNT = 2
BROYDEN_MAX = 3

plt.style.use("../../latex.mplstyle")

plt.rcParams.update(
    {
        "font.size": 18,  # default text size
        "axes.titlesize": 20,  # title
        "axes.labelsize": 18,  # x and y labels
        "xtick.labelsize": 16,  # x tick labels
        "ytick.labelsize": 16,  # y tick labels
        "legend.fontsize": 14,  # legend
    }
)

fonts_presentation = {
    "font.size": 16,  # default text size
    "axes.titlesize": 18,  # title
    "axes.labelsize": 16,  # x and y labels
    "xtick.labelsize": 14,  # x tick labels
    "ytick.labelsize": 14,  # y tick labels
    "legend.fontsize": 16,  # legend
}

fonts_paper = {
    "font.size": 12,  # default text size
    "axes.titlesize": 14,  # title
    "axes.labelsize": 12,  # x and y labels
    "xtick.labelsize": 10,  # x tick labels
    "ytick.labelsize": 10,  # y tick labels
    "legend.fontsize": 12,  # legend
}

# Load LHC model (collider + optics knobs + per-circuit match limits/steps)
collider, line = load_hllhc_b1()

# Disable edge effects
# for elem in line.elements:
#     if isinstance(elem, xt.Bend) or isinstance(elem, xt.RBend):
#         elem.edge_entry_active=0
#         elem.edge_exit_active=0

# Initial twiss
tw0 = line.twiss()

# Inspect IPS
tw0.rows["ip.*"].cols["betx bety mux muy x y"]


# Inspect for one circuit
collider.vars.vary_default["kq4.l2b2"]

# s.ds.l8.b1 -> ip1  (the IR8 optics match; see utils.ir8_optics).
# Finite-difference object (switch_to_fd toggles use_ad, ignored on this branch).
opt = ir8_optics(line, tw0)

# MAD-NG TPSA Jacobian backend (same IR8 match, MAD-NG twiss columns;
# see utils.ir8_optics).
opt_ng = ir8_optics(line, use_tpsa=True)

# Exact-physics JAX Jacobian backend (same problem as `opt`, use_jax instead
# of use_ad).  Targets are all TargetSet tuples, so use_jax stays active.
opt_jax = ir8_optics(line, tw0, use_jax=True, use_jax_residual=True)

opt_tpsa_direct = ir8_optics(line, tw0, use_tpsa_direct=True)

opt.check_limits = False
opt_ng.check_limits = False
opt_jax.check_limits = False
opt_tpsa_direct.check_limits = False

timing.start_timing("jitcompile")
opt.step(1)  # JIT-compile
timing.stop_timing()

timing.start_timing("madng_setup")
opt_ng.step(1)
timing.stop_timing()

timing.start_timing("jax_jitcompile")
opt_jax.step(1)  # trace + compile + build JAX jacobian
timing.stop_timing()

timing.start_timing("tpsa-sa_setup")
opt_tpsa_direct.step(1)
timing.stop_timing()


def reset_benchmark(opt):
    opt.reload(0)
    opt.clear_log()
    opt._err.call_counter = 0
    opt.solver._last_jac = None


def populate_dictionary(dict, opt, key, ind):
    dict[key + f"_{ind}"] = {
        "n_calls": opt._err.call_counter,
        "n_steps": len(opt.log()) - 1,  # Exclude first entry
        "runtime": timing.times[key][-1]
        - np.sum(
            timing.times["lib_time"][-2:]
        ),  # Exclude library time from last runtime sample
    }

    return dict


def benchmark(opt, fname=None, save=True):
    reset_benchmark(opt)
    reset_benchmark(opt_ng)
    reset_benchmark(opt_jax)
    reset_benchmark(opt_tpsa_direct)
    # Do AD benchmark

    data_dict = {}
    # --- AD backend not available on this branch (feature/ad_jac_exact); the
    # --- AD loops are kept for the combined branch but commented out here.
    # for i in range(COUNT):
    #     timing.start_timing('ad_solve')
    #     opt.step(100)
    #     timing.stop_timing()
    #     data_dict = populate_dictionary(data_dict, opt, f'ad_solve', i)
    #     reset_benchmark(opt)

    # for i in range(COUNT):
    #     for j in range(1, BROYDEN_MAX + 1):
    #         timing.start_timing(f'ad_solve_broyden_{j}')
    #         opt.step(100, broyden=j)
    #         timing.stop_timing()
    #         data_dict = populate_dictionary(data_dict, opt, f'ad_solve_broyden_{j}', i)
    #         reset_benchmark(opt)

    #     timing.start_timing('ad_solve_broyden_full')
    #     opt.step(100, broyden=True)
    #     timing.stop_timing()
    #     data_dict = populate_dictionary(data_dict, opt, f'ad_solve_broyden_full', i)
    #     reset_benchmark(opt)

    switch_to_fd(opt)

    for i in range(COUNT):
        timing.start_timing("fd_solve")
        opt.step(100)
        timing.stop_timing()
        data_dict = populate_dictionary(data_dict, opt, f"fd_solve", i)
        reset_benchmark(opt)

    for i in range(COUNT):
        for j in range(1, BROYDEN_MAX + 1):
            timing.start_timing(f"fd_solve_broyden_{j}")
            opt.step(100, broyden=j)
            timing.stop_timing()
            data_dict = populate_dictionary(data_dict, opt, f"fd_solve_broyden_{j}", i)
            reset_benchmark(opt)

        timing.start_timing(f"fd_solve_broyden_full")
        opt.step(100, broyden=True)
        timing.stop_timing()
        data_dict = populate_dictionary(data_dict, opt, f"fd_solve_broyden_full", i)
        reset_benchmark(opt)

    # Exact-physics JAX Jacobian benchmark (jacobian compiled once at warmup)
    for i in range(COUNT):
        timing.start_timing("jax_solve")
        opt_jax.step(100)
        timing.stop_timing()
        data_dict = populate_dictionary(data_dict, opt_jax, f"jax_solve", i)
        reset_benchmark(opt_jax)

    for i in range(COUNT):
        for j in range(1, BROYDEN_MAX + 1):
            timing.start_timing(f"jax_solve_broyden_{j}")
            opt_jax.step(100, broyden=j)
            timing.stop_timing()
            data_dict = populate_dictionary(
                data_dict, opt_jax, f"jax_solve_broyden_{j}", i
            )
            reset_benchmark(opt_jax)

        timing.start_timing(f"jax_solve_broyden_full")
        opt_jax.step(100, broyden=True)
        timing.stop_timing()
        data_dict = populate_dictionary(
            data_dict, opt_jax, f"jax_solve_broyden_full", i
        )
        reset_benchmark(opt_jax)

    # for i in range(COUNT):
    #     timing.start_timing('madngfd_solve')
    #     opt_ng.step(100)
    #     timing.stop_timing()
    #     data_dict = populate_dictionary(data_dict, opt_ng, f'madngfd_solve', i)
    #     reset_benchmark(opt_ng)

    # for i in range(COUNT):
    #     for j in range(1, BROYDEN_MAX + 1):
    #         timing.start_timing(f'madngfd_solve_broyden_{j}')
    #         opt_ng.step(100, broyden=j)
    #         timing.stop_timing()
    #         data_dict = populate_dictionary(data_dict, opt_ng, f'madngfd_solve_broyden_{j}', i)
    #         reset_benchmark(opt_ng)

    #     timing.start_timing(f'madngfd_solve_broyden_full')
    #     opt_ng.step(100, broyden=True)
    #     timing.stop_timing()
    #     data_dict = populate_dictionary(data_dict, opt_ng, f'madngfd_solve_broyden_full', i)
    #     reset_benchmark(opt_ng)

    for i in range(COUNT):
        timing.start_timing("tpsa-ng_solve")
        opt_ng.step(100)
        timing.stop_timing()
        data_dict = populate_dictionary(data_dict, opt_ng, f"tpsa-ng_solve", i)
        reset_benchmark(opt_ng)

    for i in range(COUNT):
        for j in range(1, BROYDEN_MAX + 1):
            timing.start_timing(f"tpsa-ng_solve_broyden_{j}")
            opt_ng.step(100, broyden=j)
            timing.stop_timing()
            data_dict = populate_dictionary(
                data_dict, opt_ng, f"tpsa-ng_solve_broyden_{j}", i
            )
            reset_benchmark(opt_ng)

        timing.start_timing(f"tpsa-ng_solve_broyden_full")
        opt_ng.step(100, broyden=True)
        timing.stop_timing()
        data_dict = populate_dictionary(
            data_dict, opt_ng, f"tpsa-ng_solve_broyden_full", i
        )
        reset_benchmark(opt_ng)

    for i in range(COUNT):
        timing.start_timing("tpsa-sa_solve")
        opt_tpsa_direct.step(100)
        timing.stop_timing()
        data_dict = populate_dictionary(data_dict, opt_tpsa_direct, f"tpsa-sa_solve", i)
        reset_benchmark(opt_tpsa_direct)

    for i in range(COUNT):
        for j in range(1, BROYDEN_MAX + 1):
            timing.start_timing(f"tpsa-sa_solve_broyden_{j}")
            opt_tpsa_direct.step(100, broyden=j)
            timing.stop_timing()
            data_dict = populate_dictionary(
                data_dict, opt_tpsa_direct, f"tpsa-sa_solve_broyden_{j}", i
            )
            reset_benchmark(opt_tpsa_direct)

        timing.start_timing(f"tpsa-sa_solve_broyden_full")
        opt_tpsa_direct.step(100, broyden=True)
        timing.stop_timing()
        data_dict = populate_dictionary(
            data_dict, opt_tpsa_direct, f"tpsa-sa_solve_broyden_full", i
        )
        reset_benchmark(opt_tpsa_direct)

    if save:
        if fname is None:
            fname = "generic_benchmark.json"

        if not fname.endswith(".json"):
            fname += ".json"

        with open(fname, "w") as f:
            json.dump(data_dict, f, indent=4)

    return data_dict


def benchmark_other_algorithms(opt, fname=None):
    opt._err.use_ad = False
    reset_benchmark(opt)
    # Benchmark ls trf and ls dogbox

    data_dict = {}
    for i in range(COUNT):
        timing.start_timing("ls_trf_solve")
        opt.run_ls_trf(verbose=2)
        timing.stop_timing()
        data_dict = populate_dictionary(data_dict, opt, f"ls_trf_solve", i)
        opt.target_status()
        reset_benchmark(opt)

    for i in range(COUNT):
        timing.start_timing("ls_dogbox_solve")
        opt.run_ls_dogbox(verbose=2)
        timing.stop_timing()
        data_dict = populate_dictionary(data_dict, opt, f"ls_dogbox_solve", i)
        opt.target_status()
        reset_benchmark(opt)

    if fname is None:
        fname = "generic_benchmark.json"

    if not fname.endswith(".json"):
        fname += ".json"

    with open(fname, "w") as f:
        json.dump(data_dict, f, indent=4)
    return data_dict


def get_dict_from_file(fname):
    with open(fname, "r") as f:
        data_dict = json.load(f)
    return data_dict


def convert_dict_to_df(data_dict):
    import pandas as pd

    df = pd.DataFrame.from_dict(data_dict, orient="index")
    df = df.reset_index().rename(columns={"index": "label"})
    return df


def set_fd_step(step):
    for vary in opt.vary:
        vary.step = step


def measure_time():
    opt.step(1)
    opt.reload(0)
    opt.solver._last_jac = None
    opt._err.call_counter = 0

    times = np.zeros((2, 15), dtype=float)
    call_counts = np.zeros((2, 15), dtype=int)

    for use_ad in [False, True]:
        ad_ind = [False, True].index(use_ad)
        opt._err.use_ad = ad_ind

        for i in range(15):
            i_ind = i

            if i == 0:
                i = False

            import time

            t0 = time.perf_counter()
            opt.step(60, broyden=i)
            t1 = time.perf_counter()
            times[ad_ind][i_ind] = t1 - t0
            call_counts[ad_ind][i_ind] = opt._err.call_counter
            opt.reload(0)
            opt.solver._last_jac = None
            opt._err.call_counter = 0
    return times, call_counts


def parse_key(key):
    parts = key.split("_")

    method = parts[0]  # "ad" or "fd"

    if "broyden" in parts:
        broyden = True
        sample = int(parts[4])
        interval = parts[3]  # int()
    else:
        broyden = False
        sample = int(parts[2])
        interval = "no_broyden"

    return method, broyden, sample, interval


def dict_to_dataframe(data_dict):
    rows = []
    for key, metrics in data_dict.items():
        method, broyden, sample, interval = parse_key(key)
        row = {
            "method": method,
            "broyden": broyden,
            "sample": sample,
            "interval": interval,
            **metrics,  # expand runtime info (calls, steps, runtime_ms, ...)
        }
        rows.append(row)
    return pd.DataFrame(rows)


def get_stats(df):
    stats = (
        df.groupby(["method", "broyden", "interval"])
        .agg(
            mean_runtime=("runtime", "mean"),
            std_runtime=("runtime", "std"),
            calls=("n_calls", "mean"),
            steps=("n_steps", "mean"),
        )
        .reset_index()
    )
    stats["mean_runtime"] /= 1000
    stats["std_runtime"] /= 1000

    # throw error message "hello" if assert fails
    assert np.all(stats["calls"] == np.floor(stats["calls"])), (
        "Non-integer call counts found!"
    )
    assert np.all(stats["steps"] == np.floor(stats["steps"])), (
        "Non-integer step counts found!"
    )

    stats["calls"] = stats["calls"].astype(int)
    stats["steps"] = stats["steps"].astype(int)

    return stats


def _normalize_intervals(stats):
    stats = stats.copy()
    # force everything to str
    stats["interval"] = stats["interval"].apply(lambda x: str(x))
    interval_len = len(stats["interval"].unique())
    # custom order
    interval_order = (
        ["no_broyden"] + [str(i) for i in range(1, interval_len - 1)] + ["full"]
    )
    stats["interval"] = pd.Categorical(
        stats["interval"], categories=interval_order, ordered=True
    )
    return stats, interval_order


def _fix_ticks_and_legend(ax, interval_order):
    from matplotlib.ticker import FixedLocator

    # ticks
    ticks = range(len(interval_order))
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    labels = [
        lbl.replace("no_broyden", "None").replace("full", "Full")
        for lbl in interval_order
    ]
    ax.set_xticklabels(labels)

    # legend
    handles, labels = ax.get_legend_handles_labels()
    labels = [lbl.upper() for lbl in labels]
    labels[-1] = labels[-1].replace("MADNGFD", "MAD-NG Twiss FD")
    ax.legend(handles, labels)


def plot_runtime(stats, savefig=False, fname=None, figsize=(6.4, 4.8), fonts=None):
    stats, interval_order = _normalize_intervals(stats)
    plt.figure(figsize=figsize)
    ax = plt.gca()

    for method in stats["method"].unique():
        method_data = stats[stats["method"] == method]
        ax.errorbar(
            method_data["interval"],
            method_data["mean_runtime"],
            yerr=method_data["std_runtime"],
            label=method,
            capsize=5,
            marker="o",
            linestyle="-",
        )

    ax.set_xlabel(
        "Consecutive Broyden Usage", fontsize=fonts["axes.labelsize"] if fonts else 16
    )
    ax.set_ylabel("Mean Runtime (s)", fontsize=fonts["axes.labelsize"] if fonts else 16)
    ax.set_title(
        "Runtime Optics Matching IP1", fontsize=fonts["axes.titlesize"] if fonts else 18
    )
    ax.tick_params(
        axis="both", which="major", labelsize=fonts["xtick.labelsize"] if fonts else 14
    )
    ax.grid(True, which="both", ls="--", linewidth=0.5)
    ax.legend(fontsize=fonts["legend.fontsize"] if fonts else 14)
    _fix_ticks_and_legend(ax, interval_order)

    plt.tight_layout()
    if savefig:
        if fname is None:
            fname = "benchmark_ad_vs_fd_runtime.pdf"

        if not fname.endswith(".pdf") or not fname.endswith(".json"):
            fname += "_runtime.pdf"
        else:
            fname = fname.replace(".pdf", "_runtime.pdf")
            fname = fname.replace(".json", "_runtime.pdf")
        if not fname.endswith(".pdf"):
            fname += ".pdf"
        plt.savefig(fname)
    plt.show()


def plot_call_counts(stats, savefig=False, fname=None, figsize=(6.4, 4.8), fonts=None):
    stats, interval_order = _normalize_intervals(stats)
    plt.figure(figsize=figsize)
    ax = plt.gca()

    linewidth_arr = np.linspace(
        1.5 + len(stats["method"].unique()), 1.5, len(stats["method"].unique())
    )
    markersize_arr = np.linspace(
        6 + len(stats["method"].unique()), 6, len(stats["method"].unique())
    )
    linewidth_arr = [2.5, 1.5, 1.5]

    for method, linewidth, markersize in zip(
        stats["method"].unique(), linewidth_arr, markersize_arr
    ):
        method_data = stats[stats["method"] == method]
        ax.plot(
            method_data["interval"],
            method_data["calls"],
            label=method,
            marker="o",
            linestyle="-",
            linewidth=linewidth,
            markersize=markersize,
        )

    ax.set_xlabel(
        "Consecutive Broyden Usage", fontsize=fonts["axes.labelsize"] if fonts else 16
    )
    ax.set_ylabel("Function Calls", fontsize=fonts["axes.labelsize"] if fonts else 16)
    ax.set_title(
        "Twiss Calls Matching IP1", fontsize=fonts["axes.titlesize"] if fonts else 18
    )
    ax.tick_params(
        axis="both", which="major", labelsize=fonts["xtick.labelsize"] if fonts else 14
    )
    ax.grid(True, which="both", ls="--", linewidth=0.5)
    ax.legend(fontsize=fonts["legend.fontsize"] if fonts else 14)

    _fix_ticks_and_legend(ax, interval_order)

    plt.tight_layout()
    if savefig:
        if fname is None:
            fname = "benchmark_ad_vs_fd_calls.pdf"
        if not fname.endswith(".pdf") or not fname.endswith(".json"):
            fname += "_calls.pdf"
        else:
            fname = fname.replace(".pdf", "_calls.pdf")
            fname = fname.replace(".json", "_calls.pdf")
        plt.savefig(fname)
    plt.show()


def plot_step_counts(stats, savefig=False, fname=None, figsize=(6.4, 4.8), fonts=None):
    stats, interval_order = _normalize_intervals(stats)
    plt.figure(figsize=figsize)
    ax = plt.gca()

    linewidth_arr = np.linspace(
        1.5 + len(stats["method"].unique()), 1.5, len(stats["method"].unique())
    )
    markersize_arr = np.linspace(
        6 + len(stats["method"].unique()), 6, len(stats["method"].unique())
    )

    for method, linewidth, markersize in zip(
        stats["method"].unique(), linewidth_arr, markersize_arr
    ):
        method_data = stats[stats["method"] == method]
        ax.plot(
            method_data["interval"],
            method_data["steps"],
            label=method,
            marker="o",
            linestyle="-",
            linewidth=linewidth,
            markersize=markersize,
        )

    ax.set_xlabel(
        "Consecutive Broyden Usage", fontsize=fonts["axes.labelsize"] if fonts else 16
    )
    ax.set_ylabel("Steps", fontsize=fonts["axes.labelsize"] if fonts else 16)
    ax.set_title(
        "Optimization Steps Optics Matching",
        fontsize=fonts["axes.titlesize"] if fonts else 18,
    )
    ax.tick_params(
        axis="both", which="major", labelsize=fonts["xtick.labelsize"] if fonts else 14
    )
    ax.grid(True, which="both", ls="--", linewidth=0.5)
    ax.legend(fontsize=fonts["legend.fontsize"] if fonts else 14)

    _fix_ticks_and_legend(ax, interval_order)

    plt.tight_layout()
    if savefig:
        if fname is None:
            fname = "benchmark_ad_vs_fd_steps.pdf"
        if not fname.endswith(".pdf") or not fname.endswith(".json"):
            fname += "_steps.pdf"
        else:
            fname = fname.replace(".pdf", "_steps.pdf")
            fname = fname.replace(".json", "_steps.pdf")
        plt.savefig(fname)
    plt.show()


def pipeline_visualization(
    data=None,
    do_benchmark=False,
    savefig=False,
    figsize=(6.4, 4.8),
    methods=["ad", "fd", "jax", "tpsa-ng", "tpsa-sa"],
    benchmark_dir="benchmarks",
    fonts=None,
):
    if do_benchmark:
        assert isinstance(data, str), (
            "If benchmark is True, data must be a filename string."
        )
        if benchmark_dir not in data:
            data = f"{benchmark_dir}/{data}"
        fname = data
        data = benchmark(opt, fname=data)
    else:
        if isinstance(data, str):
            if benchmark_dir not in data:
                data = f"{benchmark_dir}/{data}"
            if not data.endswith(".json"):
                data += ".json"
            fname = data
            data = get_dict_from_file(data)
        assert isinstance(data, dict), (
            "Input data must be a dictionary or a filename string."
        )
    df = dict_to_dataframe(data)
    stats = get_stats(df)
    if fname.endswith(".json"):
        fname = fname[:-5]
    stats = stats[stats["method"].isin(methods)]
    fonts_dict = fonts_paper if fonts is None else fonts

    plot_runtime(stats, savefig=savefig, fname=fname, figsize=figsize, fonts=fonts_dict)
    plot_call_counts(
        stats, savefig=savefig, fname=fname, figsize=figsize, fonts=fonts_dict
    )
    plot_step_counts(
        stats, savefig=savefig, fname=fname, figsize=figsize, fonts=fonts_dict
    )

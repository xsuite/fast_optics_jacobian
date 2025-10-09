import xtrack as xt
from xtrack._temp import lhc_match as lm
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd

# Import custom timing module
from pyprof import timing

COUNT = 10
BROYDEN_MAX = 7

plt.style.use('../../latex_presentation.mplstyle')

plt.rcParams.update({
    "font.size": 18,        # default text size
    "axes.titlesize": 20,   # title
    "axes.labelsize": 18,   # x and y labels
    "xtick.labelsize": 16,  # x tick labels
    "ytick.labelsize": 16,  # y tick labels
    "legend.fontsize": 16,  # legend
})

# Load LHC model
collider = xt.Environment.from_json(
    '../xtrack/test_data/hllhc15_thick/hllhc15_collider_thick.json')
collider.vars.load_madx(
    '../xtrack/test_data/hllhc15_thick/opt_round_150_1500.madx')

collider.build_trackers()

line = collider.lhcb1

# Disable edge effects
for elem in line.elements:
    if isinstance(elem, xt.Bend) or isinstance(elem, xt.RBend):
        elem.edge_entry_active=0
        elem.edge_exit_active=0

# Initial twiss
tw0 = line.twiss()

# Inspect IPS
tw0.rows['ip.*'].cols['betx bety mux muy x y']


# Prepare for optics matching: set limits and steps for all circuits
lm.set_var_limits_and_steps(collider)

# Inspect for one circuit
collider.vars.vary_default['kq4.l2b2']

# s.ds.l8.b1 -> ip1
opt = line.match(
    solve=False,
    default_tol={None: 1e-8, 'betx': 1e-6, 'bety': 1e-6, 'alfx': 1e-6, 'alfy': 1e-6},
    start='s.ds.l8.b1', end='ip1',
    init=tw0, init_at=xt.START,
    vary=[
        # Only IR8 quadrupoles including DS
        #xt.VaryList(['kq6.l8b1', 'kq8.l8b1'], step=fd_step)],
        xt.VaryList(['kq6.l8b1', 'kq7.l8b1', 'kq8.l8b1', 'kq9.l8b1', 'kq10.l8b1',
            'kqtl11.l8b1', 'kqt12.l8b1', 'kqt13.l8b1',
            'kq4.l8b1', 'kq5.l8b1', 'kq4.r8b1', 'kq5.r8b1',
            'kq6.r8b1', 'kq7.r8b1', 'kq8.r8b1', 'kq9.r8b1',
            'kq10.r8b1', 'kqtl11.r8b1', 'kqt12.r8b1', 'kqt13.r8b1'])],
    targets=[
        xt.TargetSet(at='ip8', tars=('betx', 'bety', 'alfx', 'alfy', 'dx', 'dpx'), value=tw0, weight=1),
        xt.TargetSet(at='ip1', betx=0.15, bety=0.1, alfx=0, alfy=0, dx=0, dpx=0, weight=1),
        xt.TargetRelPhaseAdvance('mux', value = tw0['mux', 'ip1.l1'] - tw0['mux', 's.ds.l8.b1'], weight=1),
        xt.TargetRelPhaseAdvance('muy', value = tw0['muy', 'ip1.l1'] - tw0['muy', 's.ds.l8.b1'], weight=1),
    ],
    use_ad=True)

tw_ng = line.madng_twiss()

opt_ng = line.match(
    solve=False,
    default_tol={None: 1e-8, 'beta11_ng': 1e-6, 'beta22_ng': 1e-6, 'alfa11_ng': 1e-6, 'alfa22_ng': 1e-6},
    start='s.ds.l8.b1', end='ip1',
    init=tw_ng, init_at=xt.START,
    vary=[
        xt.VaryList(['kq6.l8b1', 'kq7.l8b1', 'kq8.l8b1', 'kq9.l8b1', 'kq10.l8b1',
            'kqtl11.l8b1', 'kqt12.l8b1', 'kqt13.l8b1',
            'kq4.l8b1', 'kq5.l8b1', 'kq4.r8b1', 'kq5.r8b1',
            'kq6.r8b1', 'kq7.r8b1', 'kq8.r8b1', 'kq9.r8b1',
            'kq10.r8b1', 'kqtl11.r8b1', 'kqt12.r8b1', 'kqt13.r8b1'])],
    targets=[
        xt.TargetSet(at='ip8', tars=('beta11_ng', 'beta22_ng', 'alfa11_ng', 'alfa22_ng', 'dx_ng', 'dpx_ng'), value=tw_ng, weight=1),
        xt.TargetSet(at='ip1', beta11_ng=0.15, beta22_ng=0.1, alfa11_ng=0, alfa22_ng=0, dx_ng=0, dpx_ng=0, weight=1),
        xt.TargetRelPhaseAdvance('mu1_ng', start='s.ds.l8.b1', end='ip1.l1',
                                  value = tw_ng['mu1_ng', 'ip1.l1'] - tw_ng['mu1_ng', 's.ds.l8.b1'], weight=1),
        xt.TargetRelPhaseAdvance('mu2_ng', start='s.ds.l8.b1', end='ip1.l1',
                                  value = tw_ng['mu2_ng', 'ip1.l1'] - tw_ng['mu2_ng', 's.ds.l8.b1'], weight=1),
    ])

opt.check_limits = False
opt_ng.check_limits = False

timing.start_timing('jitcompile')
opt.step(1) # JIT-compile
timing.stop_timing()

opt_ng.step(1)

def reset_benchmark(opt):
    opt.reload(0)
    opt.clear_log()
    opt._err.call_counter = 0
    opt.solver._last_jac = None

def switch_to_ad(opt):
    opt._err.use_ad = True

def switch_to_fd(opt):
    opt._err.use_ad = False

def populate_dictionary(dict, opt, key, ind):
    dict[key + f'_{ind}'] = {
        'n_calls': opt._err.call_counter,
        'n_steps': len(opt.log()) - 1,  # Exclude first entry
        'runtime': timing.times[key][-1] - np.sum(timing.times['lib_time'][-2:]) # Exclude library time from last runtime sample
    }

    return dict

def benchmark(opt, fname=None):
    reset_benchmark(opt)
    reset_benchmark(opt_ng)
    # Do AD benchmark

    data_dict = {}
    for i in range(COUNT):
        timing.start_timing('ad_solve')
        opt.step(100)
        timing.stop_timing()
        data_dict = populate_dictionary(data_dict, opt, f'ad_solve', i)
        reset_benchmark(opt)

    for i in range(COUNT):
        for j in range(1, BROYDEN_MAX + 1):
            timing.start_timing(f'ad_solve_broyden_{j}')
            opt.step(100, broyden=j)
            timing.stop_timing()
            data_dict = populate_dictionary(data_dict, opt, f'ad_solve_broyden_{j}', i)
            reset_benchmark(opt)

        timing.start_timing('ad_solve_broyden_full')
        opt.step(100, broyden=True)
        timing.stop_timing()
        data_dict = populate_dictionary(data_dict, opt, f'ad_solve_broyden_full', i)
        reset_benchmark(opt)

    switch_to_fd(opt)

    for i in range(COUNT):
        timing.start_timing('fd_solve')
        opt.step(100)
        timing.stop_timing()
        data_dict = populate_dictionary(data_dict, opt, f'fd_solve', i)
        reset_benchmark(opt)

    for i in range(COUNT):
        for j in range(1, BROYDEN_MAX + 1):
            timing.start_timing(f'fd_solve_broyden_{j}')
            opt.step(100, broyden=j)
            timing.stop_timing()
            data_dict = populate_dictionary(data_dict, opt, f'fd_solve_broyden_{j}', i)
            reset_benchmark(opt)

        timing.start_timing(f'fd_solve_broyden_full')
        opt.step(100, broyden=True)
        timing.stop_timing()
        data_dict = populate_dictionary(data_dict, opt, f'fd_solve_broyden_full', i)
        reset_benchmark(opt)

    for i in range(COUNT):
        timing.start_timing('madngfd_solve')
        opt_ng.step(100)
        timing.stop_timing()
        data_dict = populate_dictionary(data_dict, opt_ng, f'madngfd_solve', i)
        reset_benchmark(opt_ng)

    for i in range(COUNT):
        for j in range(1, BROYDEN_MAX + 1):
            timing.start_timing(f'madngfd_solve_broyden_{j}')
            opt_ng.step(100, broyden=j)
            timing.stop_timing()
            data_dict = populate_dictionary(data_dict, opt_ng, f'madngfd_solve_broyden_{j}', i)
            reset_benchmark(opt_ng)

        timing.start_timing(f'madngfd_solve_broyden_full')
        opt_ng.step(100, broyden=True)
        timing.stop_timing()
        data_dict = populate_dictionary(data_dict, opt_ng, f'madngfd_solve_broyden_full', i)
        reset_benchmark(opt_ng)

    if fname is None:
        fname = "generic_benchmark.json"

    if not fname.endswith('.json'):
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
        timing.start_timing('ls_trf_solve')
        opt.run_ls_trf(verbose=2)
        timing.stop_timing()
        data_dict = populate_dictionary(data_dict, opt, f'ls_trf_solve', i)
        opt.target_status()
        reset_benchmark(opt)

    for i in range(COUNT):
        timing.start_timing('ls_dogbox_solve')
        opt.run_ls_dogbox(verbose=2)
        timing.stop_timing()
        data_dict = populate_dictionary(data_dict, opt, f'ls_dogbox_solve', i)
        opt.target_status()
        reset_benchmark(opt)

    if fname is None:
        fname = "generic_benchmark.json"

    if not fname.endswith('.json'):
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
    df = pd.DataFrame.from_dict(data_dict, orient='index')
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
        interval = parts[3] # int()
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
    stats = df.groupby(['method', 'broyden', 'interval']).agg(
        mean_runtime=('runtime', 'mean'),
        std_runtime=('runtime', 'std'),
        calls=('n_calls', 'mean'),
        steps=('n_steps', 'mean'),
    ).reset_index()
    stats["mean_runtime"] /= 1000
    stats["std_runtime"] /= 1000

    # throw error message "hello" if assert fails
    assert np.all(stats["calls"] == np.floor(stats["calls"])), "Non-integer call counts found!"
    assert np.all(stats["steps"] == np.floor(stats["steps"])), "Non-integer step counts found!"

    stats["calls"] = stats["calls"].astype(int)
    stats["steps"] = stats["steps"].astype(int)

    return stats

def _normalize_intervals(stats):
    stats = stats.copy()
    # force everything to str
    stats["interval"] = stats["interval"].apply(lambda x: str(x))
    interval_len = len(stats["interval"].unique())
    # custom order
    interval_order = ["no_broyden"] + [str(i) for i in range(1, interval_len - 1)] + ["full"]
    stats["interval"] = pd.Categorical(stats["interval"], categories=interval_order, ordered=True)
    return stats, interval_order

def _fix_ticks_and_legend(ax, interval_order):
    from matplotlib.ticker import FixedLocator
    # ticks
    ticks = range(len(interval_order))
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    labels = [lbl.replace("no_broyden", "None").replace("full", "Full") for lbl in interval_order]
    ax.set_xticklabels(labels)

    # legend
    handles, labels = ax.get_legend_handles_labels()
    labels = [lbl.upper() for lbl in labels]
    labels[-1] = labels[-1].replace("MADNGFD", "MAD-NG Twiss FD")
    ax.legend(handles, labels)

def plot_runtime(stats, savefig=False, fname=None, figsize=(6.4, 4.8)):
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
            linestyle="-"
        )

    ax.set_xlabel("Consecutive Broyden Usage", fontsize=16)
    ax.set_ylabel("Mean Runtime (s)", fontsize=16)
    ax.set_title("Runtime Optics Matching IP1", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True, which="both", ls="--", linewidth=0.5)

    _fix_ticks_and_legend(ax, interval_order)

    plt.tight_layout()
    if savefig:
        if fname is None:
            fname = "benchmark_ad_vs_fd_runtime.pdf"

        if not fname.endswith('.pdf') or not fname.endswith('.json'):
            fname += "_runtime.pdf"
        else:
            fname = fname.replace('.pdf', '_runtime.pdf')
            fname = fname.replace('.json', '_runtime.pdf')
        if not fname.endswith('.pdf'):
            fname += ".pdf"
        plt.savefig(fname)
    plt.show()


def plot_call_counts(stats, savefig=False, fname=None, figsize=(6.4, 4.8)):
    stats, interval_order = _normalize_intervals(stats)
    plt.figure(figsize=figsize)
    ax = plt.gca()

    for method in stats["method"].unique():
        method_data = stats[stats["method"] == method]
        ax.plot(
            method_data["interval"],
            method_data["calls"],
            label=method,
            marker="o",
            linestyle="-"
        )

    ax.set_xlabel("Consecutive Broyden Usage", fontsize=16)
    ax.set_ylabel("Function Calls", fontsize=16)
    ax.set_title("Twiss Calls Matching IP1", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True, which="both", ls="--", linewidth=0.5)

    _fix_ticks_and_legend(ax, interval_order)

    plt.tight_layout()
    if savefig:
        if fname is None:
            fname = "benchmark_ad_vs_fd_calls.pdf"
        if not fname.endswith('.pdf') or not fname.endswith('.json'):
            fname += "_calls.pdf"
        else:
            fname = fname.replace('.pdf', '_calls.pdf')
            fname = fname.replace('.json', '_calls.pdf')
        plt.savefig(fname)
    plt.show()


def plot_step_counts(stats, savefig=False, fname=None, figsize=(6.4, 4.8)):
    stats, interval_order = _normalize_intervals(stats)
    plt.figure(figsize=figsize)
    ax = plt.gca()

    for method in stats["method"].unique():
        method_data = stats[stats["method"] == method]
        ax.plot(
            method_data["interval"],
            method_data["steps"],
            label=method,
            marker="o",
            linestyle="-"
        )

    ax.set_xlabel("Consecutive Broyden Usage", fontsize=16)
    ax.set_ylabel("Steps", fontsize=16)
    ax.set_title("Optimization Steps Optics Matching", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True, which="both", ls="--", linewidth=0.5)

    _fix_ticks_and_legend(ax, interval_order)

    plt.tight_layout()
    if savefig:
        if fname is None:
            fname = "benchmark_ad_vs_fd_steps.pdf"
        if not fname.endswith('.pdf') or not fname.endswith('.json'):
            fname += "_steps.pdf"
        else:
            fname = fname.replace('.pdf', '_steps.pdf')
            fname = fname.replace('.json', '_steps.pdf')
        plt.savefig(fname)
    plt.show()

def pipeline_visualization(data=None, do_benchmark=False, savefig=False, figsize=(6.4, 4.8), methods=['ad', 'fd']):
    if do_benchmark:
        assert isinstance(data, str), "If benchmark is True, data must be a filename string."
        fname = data
        data = benchmark(opt, fname=data)
    else:
        if isinstance(data, str):
            fname = data
            if not data.endswith('.json'):
                data += ".json"
            data = get_dict_from_file(data)
        assert isinstance(data, dict), "Input data must be a dictionary or a filename string."
    df = dict_to_dataframe(data)
    stats = get_stats(df)
    if fname.endswith('.json'):
        fname = fname[:-5]
    stats = stats[stats["method"].isin(methods)]
    plot_runtime(stats, savefig=savefig, fname=fname, figsize=figsize)
    plot_call_counts(stats, savefig=savefig, fname=fname, figsize=figsize)
    plot_step_counts(stats, savefig=savefig, fname=fname, figsize=figsize)
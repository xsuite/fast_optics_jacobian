import xtrack as xt
import numpy as np
import matplotlib.pyplot as plt

from utils import load_hllhc_b1, ir8_optics, switch_to_ad, switch_to_fd

# Load LHC model (collider + optics knobs + per-circuit match limits/steps)
collider, line = load_hllhc_b1()

for elem in line.elements:
    if isinstance(elem, xt.Bend) or isinstance(elem, xt.RBend):
        elem.edge_entry_active = 0
        elem.edge_exit_active = 0

# Initial twiss
tw0 = line.twiss()

# Inspect IPS
tw0.rows["ip.*"].cols["betx bety mux muy x y"]

fd_step = 1e-6

# Inspect for one circuit
collider.vars.vary_default['kq4.l2b2']

# Twiss on a part of the machine (bidirectional)
tw_81_12 = line.twiss(start='ip8', end='ip2', init_at='ip1',
                                betx=0.15, bety=0.15)


# s.ds.l8.b1 -> ip1  (the IR8 optics match; see utils.ir8_optics)
opt = ir8_optics(line, tw0, use_ad=True)

opt.check_limits = False

opt.step(1)

def set_fd_step(step):
    for vary in opt.vary:
        vary.step = step

def measure_time():
    opt.step(1)
    opt.reload(0)
    opt.solver._last_jac = None
    opt._err.call_counter = 0

    times = np.zeros((2, 14), dtype=float)
    call_counts = np.zeros((2, 14), dtype=int)

    for use_ad in [False, True]:
        ad_ind = [False, True].index(use_ad)
        opt._err.use_ad = ad_ind

        for i in range(1, 15):
            i_ind = i - 1

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

def plot_times(times, call_counts):
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.plot(np.arange(1, len(times[0]) + 1), times[0], label='FD')
    plt.plot(np.arange(1, len(times[1]) + 1), times[1], label='AD')
    plt.xlabel('Broyden interval')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.title('Optimization Time for different broyden intervals')

    plt.subplot(122)
    plt.plot(np.arange(1, len(call_counts[0]) + 1), call_counts[0], label='FD')
    plt.plot(np.arange(1, len(call_counts[1]) + 1), call_counts[1], label='AD')
    plt.xlabel('Broyden interval')
    plt.ylabel('Function calls')
    plt.legend()
    plt.title('Function calls for different broyden intervals')
    plt.tight_layout()
    #plt.savefig("rt_adfd_broyden_new.pdf")


# plt.show()

# switch_to_ad / switch_to_fd now come from utils.

# Save plot from s.ds.l8.b1 to ip1

plt.style.use('../../latex_presentation.mplstyle')
tw_s_ip1 = line.twiss(start='s.ds.l8.b1', end='ip1', init=tw0, init_at=xt.START)
quads = opt._err.quad_sources_ord
quad_pos = tw_s_ip1['s', quads]

tw_plot = tw_s_ip1.plot('betx bety')

for iq, q in enumerate(quads):
    plt.axvline(quad_pos[iq], color='k', linestyle='--', alpha=0.7)

tw_plot.ax.set_xticks([23000, 24000, 25000, 26000])
plt.tight_layout()
plt.show()

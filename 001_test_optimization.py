import xtrack as xt
from xtrack._temp import lhc_match as lm
import numpy as np
import matplotlib.pyplot as plt

# Load LHC model
collider = xt.Environment.from_json(
    '../xtrack/test_data/hllhc15_thick/hllhc15_collider_thick.json')
collider.vars.load_madx(
    '../xtrack/test_data/hllhc15_thick/opt_round_150_1500.madx')

collider.build_trackers()

line = collider.lhcb1

for elem in line.elements:
    if isinstance(elem, xt.Bend) or isinstance(elem, xt.RBend):
        elem.edge_entry_active=0
        elem.edge_exit_active=0

# Initial twiss
tw0 = line.twiss()

# Inspect IPS
tw0.rows['ip.*'].cols['betx bety mux muy x y']

fd_step = 1e-6


# Prepare for optics matching: set limits and steps for all circuits
lm.set_var_limits_and_steps(collider)

# Inspect for one circuit
collider.vars.vary_default['kq4.l2b2']

# Twiss on a part of the machine (bidirectional)
tw_81_12 = line.twiss(start='ip8', end='ip2', init_at='ip1',
                                betx=0.15, bety=0.15)

line['myvar'] = 0.5 * line['kq7.l8b1']
line['kq7.l8b1'] = '2 * myvar'

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
        xt.TargetSet(at='ip8', tars=('betx', 'bety', 'alfx', 'alfy', 'dx', 'dpx'), value=tw0),
        xt.TargetSet(at='ip1', betx=0.15, bety=0.1, alfx=0, alfy=0, dx=0, dpx=0),
        # xt.TargetRelPhaseAdvance('mux', value = tw0['mux', 'ip1.l1'] - tw0['mux', 'ip8'], start='ip8', end='ip1.l1'),
        # xt.TargetRelPhaseAdvance('muy', value = tw0['muy', 'ip1.l1'] - tw0['muy', 'ip8'], start='ip8', end='ip1.l1'),
        xt.TargetRelPhaseAdvance('mux', value = tw0['mux', 'ip1.l1'] - tw0['mux', 's.ds.l8.b1']),
        xt.TargetRelPhaseAdvance('muy', value = tw0['muy', 'ip1.l1'] - tw0['muy', 's.ds.l8.b1']),
    ],
    use_ad=True)

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

def switch_to_ad(opt):
    # import xdeps as xd
    # import jacobian_mod as jmod
    # xd.optimize.optimize.MeritFunctionForMatch.get_jacobian = jmod.get_jacobian
    opt._err.use_ad = True

def switch_to_fd(opt):
    # import xdeps as xd
    # xd.optimize.optimize.MeritFunctionForMatch.get_jacobian = xd.optimize.optimize.MeritFunctionForMatch.get_jacobian
    opt._err.use_ad = False

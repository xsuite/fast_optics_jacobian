import xtrack as xt
from xtrack._temp import lhc_match as lm
import numpy as np
import sympy
import time

# Add missing method to twiss table
import twiss_deriv

#import jacobian_mod as jacmod

# Load LHC model
collider = xt.Environment.from_json(
    '../xtrack/test_data/hllhc15_thick/hllhc15_collider_thick.json')
collider.vars.load_madx(
    '../xtrack/test_data/hllhc15_thick/opt_round_150_1500.madx')

collider.build_trackers()

line = collider.lhcb1

line.cycle('ip7', inplace=True)

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
            'kq10.r8b1', 'kqtl11.r8b1', 'kqt12.r8b1', 'kqt13.r8b1'], step=fd_step)],
    targets=[
        # xt.TargetSet(at='ip8', tars=('bety'), value=tw0),
        # xt.TargetSet(at='ip1', bety=0.1),
        xt.TargetSet(at='ip8', tars=('betx', 'bety', 'alfx', 'alfy', 'dx', 'dpx'), value=tw0),
        xt.TargetSet(at='ip1', betx=0.15, bety=0.1, alfx=0, alfy=0, dx=0, dpx=0),
        xt.TargetRelPhaseAdvance('mux', value = tw0['mux', 'ip1.l1'] - tw0['mux', 'ip8'], start='ip8', end='ip1.l1'),
        xt.TargetRelPhaseAdvance('muy', value = tw0['muy', 'ip1.l1'] - tw0['muy', 'ip8'], start='ip8', end='ip1.l1'),
        # xt.TargetRelPhaseAdvance('mux', value = tw0['mux', 'ip1.l1'] - tw0['mux', 's.ds.l8.b1']),
        # xt.TargetRelPhaseAdvance('muy', value = tw0['muy', 'ip1.l1'] - tw0['muy', 's.ds.l8.b1']),
    ])

opt.check_limits = False

def set_fd_step(step):
    for vary in opt.vary:
        vary.step = step
# Match for target bety: 0.15 --> [0.1, 0.14, 0.149, 0.1499, 0.15]

opt.target_status()

#import jacobian_mod as jacmod

#opt.step(40)

# import matplotlib.pyplot as plt
# penalty = opt.log()['penalty']

# plt.plot(penalty)
# plt.xlabel('Iteration')
# plt.ylabel('Penalty')
# plt.title(f'Optimization Penalty over Iterations (normal case) with fd step {fd_step:.0e}')
# # use log scale
# plt.savefig(f'optimization_penalty_{fd_step:.0e}.png')
# plt.yscale('log')
# plt.savefig(f'optimization_penalty_{fd_step:.0e}_log.png')


# plt.show()

def switch_to_ad():
    import xdeps as xd
    import jacobian_mod as jmod
    xd.optimize.optimize.MeritFunctionForMatch.get_jacobian = jmod.get_jacobian

def switch_to_fd():
    import xdeps as xd
    xd.optimize.optimize.MeritFunctionForMatch.get_jacobian = xd.optimize.optimize.MeritFunctionForMatch.get_jacobian
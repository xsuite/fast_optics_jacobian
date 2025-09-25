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

# Initial twiss
tw0 = line.madng_twiss()

# Prepare for optics matching: set limits and steps for all circuits
lm.set_var_limits_and_steps(collider)

# Inspect for one circuit
collider.vars.vary_default['kq4.l2b2']

# s.ds.l8.b1 -> ip1
opt = line.match(
    solve=False,
    default_tol={None: 1e-8, 'beta11_ng': 1e-6, 'beta22_ng': 1e-6, 'alfa11_ng': 1e-6, 'alfa22_ng': 1e-6},
    start='s.ds.l8.b1', end='ip1',
    init=tw0, init_at=xt.START,
    vary=[
        xt.VaryList(['kq6.l8b1', 'kq7.l8b1', 'kq8.l8b1', 'kq9.l8b1', 'kq10.l8b1',
            'kqtl11.l8b1', 'kqt12.l8b1', 'kqt13.l8b1',
            'kq4.l8b1', 'kq5.l8b1', 'kq4.r8b1', 'kq5.r8b1',
            'kq6.r8b1', 'kq7.r8b1', 'kq8.r8b1', 'kq9.r8b1',
            'kq10.r8b1', 'kqtl11.r8b1', 'kqt12.r8b1', 'kqt13.r8b1'])],
    targets=[
        xt.TargetSet(at='ip8', tars=('beta11_ng', 'beta22_ng', 'alfa11_ng', 'alfa22_ng', 'dx_ng', 'dpx_ng'), value=tw0),
        xt.TargetSet(at='ip1', beta11_ng=0.15, beta22_ng=0.1, alfa11_ng=0, alfa22_ng=0, dx_ng=0, dpx_ng=0),
        xt.TargetRelPhaseAdvance('mu1_ng', start='s.ds.l8.b1', end='ip1.l1',\
                                  value = tw0['mu1_ng', 'ip1.l1'] - tw0['mu1_ng', 's.ds.l8.b1']),
        xt.TargetRelPhaseAdvance('mu2_ng', start='s.ds.l8.b1', end='ip1.l1',\
                                  value = tw0['mu2_ng', 'ip1.l1'] - tw0['mu2_ng', 's.ds.l8.b1']),
    ])

from pyprof import timing
timing.start_timing("Xsuite_Opt_MADNG")
opt.step(60, broyden=True)
timing.stop_timing()
timing.report()
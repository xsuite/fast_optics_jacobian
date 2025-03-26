import xtrack as xt
from xtrack._temp import lhc_match as lm
import numpy as np
import sympy
import time

# Add missing method to twiss table
import twiss_deriv

# Load LHC model
collider = xt.Environment.from_json(
    '../../xtrack/test_data/hllhc15_thick/hllhc15_collider_thick.json')
collider.vars.load_madx(
    '../../xtrack/test_data/hllhc15_thick/opt_round_150_1500.madx')

collider.build_trackers()

line = collider.lhcb1

line.cycle('ip7', inplace=True)

# Initial twiss
tw0 = line.twiss()

# Inspect IPS
tw0.rows['ip.*'].cols['betx bety mux muy x y']


# Prepare for optics matching: set limits and steps for all circuits
lm.set_var_limits_and_steps(collider)

# Inspect for one circuit
collider.vars.vary_default['kq4.l2b2']

# Twiss on a part of the machine (bidirectional)
tw_81_12 = line.twiss(start='ip8', end='ip2', init_at='ip1',
                                betx=0.15, bety=0.15)

line['myvar'] = 0.5 * line['kq7.l8b1']
line['kq7.l8b1'] = '2 * myvar'


opt = line.match(
    solve=False,
    default_tol={None: 1e-8, 'betx': 1e-6, 'bety': 1e-6, 'alfx': 1e-6, 'alfy': 1e-6},
    start='s.ds.l8.b1', end='ip1',
    init=tw0, init_at=xt.START,
    vary=[
        # Only IR8 quadrupoles including DS
        xt.VaryList(['kq6.l8b1', 'myvar', 'kq8.l8b1', 'kq9.l8b1', 'kq10.l8b1',
            'kqtl11.l8b1', 'kqt12.l8b1', 'kqt13.l8b1',
            'kq4.l8b1', 'kq5.l8b1', 'kq4.r8b1', 'kq5.r8b1',
            'kq6.r8b1', 'kq7.r8b1', 'kq8.r8b1', 'kq9.r8b1',
            'kq10.r8b1', 'kqtl11.r8b1', 'kqt12.r8b1', 'kqt13.r8b1'])],
    targets=[
        xt.TargetSet(at='ip8', tars=('betx', 'bety', 'alfx', 'alfy', 'dx', 'dpx'), value=tw0),
        xt.TargetSet(at='ip1', betx=0.15, bety=0.10, alfx=0, alfy=0, dx=0, dpx=0),
        # xt.TargetRelPhaseAdvance('mux', value = tw0['mux', 'ip1.l1'] - tw0['mux', 's.ds.l8.b1']),
        # xt.TargetRelPhaseAdvance('muy', value = tw0['muy', 'ip1.l1'] - tw0['muy', 's.ds.l8.b1']),
    ])

opt.check_limits = False

# Match for target bety: 0.15 --> [0.1, 0.14, 0.149, 0.1499, 0.15]

opt.target_status()

class FakeQuad:
    pass

env = collider

t0 = time.perf_counter()
dkq_dvv = {} # Derivatives of quadrupole strengths with respect to the knobs
a = sympy.var("a")
for ivv in range(len(opt.vary)):
    vv = opt.vary[ivv].name

    k1 = []
    myelems = {}
    for dd in env.ref_manager.find_deps([env.vars[vv]]):
        if dd.__class__.__name__ == "AttrRef" and dd._key == "k1":
            k1.append((dd._owner._key, dd._expr))
            myelems[dd._owner._key] = FakeQuad()

    fdef = env.ref_manager.mk_fun("myfun", a=env.vars[vv])
    gbl = {
        "vars": env.ref_manager.containers["vars"]._owner.copy(),
        "element_refs": myelems,
    }
    lcl = {}
    exec(fdef, gbl, lcl)
    fff = lcl["myfun"]

    fff(a)
    dk1_dvv = {}
    for kk, expr in k1:
        dd = gbl["element_refs"][kk].k1.diff(a)
        dk1_dvv[kk] = dd

    dkq_dvv[vv] = dk1_dvv

t1 = time.perf_counter()
print('Sympy derivatives in', t1-t0, 's')

# TODO make more efficient
jac_estim = np.zeros((len(opt.targets), len(opt.vary)))
for itt, tt in enumerate(opt.targets):

    assert isinstance(tt.tar, tuple)

    tar_quantitiy = tt.tar[0]
    tar_place = tt.tar[1]
    tar_weight = tt.weight

    for ivv in range(len(opt.vary)):
        vv = opt.vary[ivv].name
        quad_names = dkq_dvv[vv].keys()

        # Extract relevant twiss derivatives
        twiss_derivs = {}
        for qqnn in quad_names:
            twiss_derivs[qqnn] = tw0.get_twiss_param_derivative(src=qqnn, observation=tar_place)

            # Refer to k1 instead of k1l
            for nn in twiss_derivs[qqnn].keys():
                twiss_derivs[qqnn][nn] *= env[qqnn].length

        dtar_dvv = 0
        for qqnn in quad_names:
            dtar_dvv += twiss_derivs[qqnn]['d'+tar_quantitiy] * dkq_dvv[vv][qqnn]

        dtar_dvv *= tar_weight

    jac_estim[itt, ivv] = dtar_dvv
t1 = time.perf_counter()
print('Estimated in', t1-t0, 's')


err = opt.get_merit_function()

t1_fd = time.perf_counter()
jac = err.get_jacobian(err.get_x())
print('                                         ')
t2_fd = time.perf_counter()
print('Finite difference in', t2_fd-t1_fd, 's')

print('\n\nEstimated Jacobian vs real Jacobian')
i_col = 10
for jj, tt in enumerate(opt.targets):
    print(jj, jac_estim[jj, i_col], jac[jj, i_col])
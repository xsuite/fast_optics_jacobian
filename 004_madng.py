import xtrack as xt
from xtrack._temp import lhc_match as lm
import numpy as np
import time
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
        elem.edge_entry_active=1
        elem.edge_exit_active=1

mng = line.to_madng(sequence_name='lhcb1')

tw0 = mng.twiss(sequence="lhcb1")[0]

varylist = ['kq6.l8b1', 'kq7.l8b1', 'kq8.l8b1', 'kq9.l8b1', 'kq10.l8b1',
            'kqtl11.l8b1', 'kqt12.l8b1', 'kqt13.l8b1',
            'kq4.l8b1', 'kq5.l8b1', 'kq4.r8b1', 'kq5.r8b1',
            'kq6.r8b1', 'kq7.r8b1', 'kq8.r8b1', 'kq9.r8b1',
            'kq10.r8b1', 'kqtl11.r8b1', 'kqt12.r8b1', 'kqt13.r8b1']

def set_var_string(varylist):
    varstr = ''
    for var in varylist:
        var_split = var.split('.')
        var_name = f"{var_split[0]}_{var_split[1]}"
        varstr += f"{{ var='MADX.{var_name}', name='{var_name}',  rtol=1e-6}}"
        if var != varylist[-1]:
            varstr += ',\n'
    return varstr


df = tw0.to_df()


mng.send("twpart = twiss {sequence = lhcb1, range = 's.ds.l8.b1/ip1'}")
mng.send("mu1diff = twpart['ip1.l1'].mu1 - twpart['s.ds.l8.b1'].mu1")
mng.send("mu2diff = twpart['ip1.l1'].mu2 - twpart['s.ds.l8.b1'].mu2")


mng.send("""
match {
    command := twiss {sequence = lhcb1, range = 's.ds.l8.b1/ip1'},
    variables = {
         """ + set_var_string(varylist) + """
    },
    equalities = {
        { expr=\\t -> t.ip8.beta11-0.15, kind='beta', name='betx_ip8', tol=1e-4 },
        { expr=\\t -> t.ip8.beta22-0.15, kind='beta', name='bety_ip8', tol=1e-4 },
        { expr=\\t -> t.ip8.alfa11, kind='alfa', name='alfx_ip8', tol=1e-4 },
        { expr=\\t -> t.ip8.alfa22, kind='alfa', name='alfy_ip8', tol=1e-4 },
        { expr=\\t -> t.ip8.dx, kind='dx', name='dx_ip8', tol=1e-4 },
        { expr=\\t -> t.ip8.dpx, kind='dpx', name='dy_ip8', tol=1e-4 },
        { expr=\\t -> t.ip1.beta11-0.15, kind='beta', name='betx_ip1', tol=1e-4 },
        { expr=\\t -> t.ip1.beta22-0.15, kind='beta', name='bety_ip1', tol=1e-4 },
        { expr=\\t -> t.ip1.alfa11, kind='alfa', name='alfx_ip1', tol=1e-4 },
        { expr=\\t -> t.ip1.alfa22, kind='alfa', name='alfy_ip1', tol=1e-4 },
        { expr=\\t -> t.ip1.dx, kind='dx', name='dx_ip1', tol=1e-4 },
        { expr=\\t -> t.ip1.dpx, kind='dpx', name='dy_ip1', tol=1e-4 },

    },
    objective = { fmin=1e-10, broyden=true },
    info = 3,
    debug = 1,
    maxcall = 1000,
}
""")

# { expr=\\t -> t["ip1.l1"].mu1 - t["s.ds.l8.b1"].mu1, kind='mux', name='mux_ip1', tol=1e-4 },
# { expr=\\t -> t["ip1.l1"].mu2 - t["s.ds.l8.b1"].mu2, kind='muy', name='muy_ip1', tol=1e-4 },

# mng.match(
#     command="twiss {sequence=lhcb1}",
#     variables=[
#         {"var": "'MADX.kq6_l8b1'", "name": "'kq6_l8b1'", "rtol": 1e-6}
#     ],
#     equalities=[
#         {"expr": "\\t -> t.IP1.beta11-0.15", "kind":"'beta'", "name": "'betx_ip8'", "tol": 1e-4},
#     ],
#     objective={"fmin": 1e-3},
#     info=2
# )


# # Initial twiss
# tw0 = line.twiss()

# # Inspect IPS
# tw0.rows['ip.*'].cols['betx bety mux muy x y']

# fd_step = 1e-6


# # Prepare for optics matching: set limits and steps for all circuits
# lm.set_var_limits_and_steps(collider)

# # Inspect for one circuit
# collider.vars.vary_default['kq4.l2b2']

# # Twiss on a part of the machine (bidirectional)
# tw_81_12 = line.twiss(start='ip8', end='ip2', init_at='ip1',
#                                 betx=0.15, bety=0.15)

# # s.ds.l8.b1 -> ip1
# opt = line.match(
#     solve=False,
#     default_tol={None: 1e-8, 'betx': 1e-6, 'bety': 1e-6, 'alfx': 1e-6, 'alfy': 1e-6},
#     start='s.ds.l8.b1', end='ip1',
#     init=tw0, init_at=xt.START,
#     vary=[
#         # Only IR8 quadrupoles including DS
#         #xt.VaryList(['kq6.l8b1', 'kq8.l8b1'], step=fd_step)],
#         xt.VaryList(['kq6.l8b1', 'kq7.l8b1', 'kq8.l8b1', 'kq9.l8b1', 'kq10.l8b1',
#             'kqtl11.l8b1', 'kqt12.l8b1', 'kqt13.l8b1',
#             'kq4.l8b1', 'kq5.l8b1', 'kq4.r8b1', 'kq5.r8b1',
#             'kq6.r8b1', 'kq7.r8b1', 'kq8.r8b1', 'kq9.r8b1',
#             'kq10.r8b1', 'kqtl11.r8b1', 'kqt12.r8b1', 'kqt13.r8b1'], step=fd_step)],
#     targets=[
#         xt.TargetSet(at='ip8', tars=('betx', 'bety', 'alfx', 'alfy', 'dx', 'dpx'), value=tw0),
#         xt.TargetSet(at='ip1', betx=0.15, bety=0.1, alfx=0, alfy=0, dx=0, dpx=0),
#         xt.TargetRelPhaseAdvance('mux', value = tw0['mux', 'ip1.l1'] - tw0['mux', 's.ds.l8.b1']),
#         xt.TargetRelPhaseAdvance('muy', value = tw0['muy', 'ip1.l1'] - tw0['muy', 's.ds.l8.b1']),
#     ])

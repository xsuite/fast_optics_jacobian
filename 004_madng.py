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
        varstr += f"{{ var='MADX.{var_name}', name='{var_name}'}}"
        if var != varylist[-1]:
            varstr += ',\n'
    return varstr


df = tw0.to_df()


mng.send("""
    local observed in MAD.element.flags
    lhcb1:select(observed, {list = {'s.ds.l8.b1', 'ip1.l1', 'ip1', 'ip8'}})
    twpart, mf = twiss {sequence = lhcb1, observe = 1, savemap = true, info = 2}
    X0 = twpart['s.ds.l8.b1'].__map
    X0.status = "Aset" ! Bug corrected in next version
    print("PRE SECOND TWISS")
    local tws = twiss {sequence = lhcb1, X0 = X0, range = 's.ds.l8.b1/ip1', info = 2}
    mu1_to_ip1 = tws['ip1.l1'].mu1
    mu2_to_ip1 = tws['ip1.l1'].mu2
    -- tws2 = twiss {sequence = lhcb1, X0 = X0, range = 's.ds.l8.b1/ip1', info = 2}

    local printf in MAD.utility
    time_1 = os.clock()
    n = 0
    while n < 10 do
        twiss {sequence = lhcb1, X0=X0, range = 's.ds.l8.b1/ip1'}
        n = n + 1
    end
    time_2 = os.clock()-time_1
    print("TIME: ",time_2 / 10, " s per twiss")
""")

mng.send("""
match {
    command := twiss {sequence = lhcb1, X0=X0, range = 's.ds.l8.b1/ip1'},
    variables = { rtol=1e-6,
         """ + set_var_string(varylist) + """
    },
    equalities = { tol=1e-4,
        { expr=\\t -> t.ip8.beta11-0.15, kind='beta', name='betx_ip8'},
        { expr=\\t -> t.ip8.beta22-0.15, kind='beta', name='bety_ip8'},
        { expr=\\t -> t.ip8.alfa11, kind='alfa', name='alfx_ip8'},
        { expr=\\t -> t.ip8.alfa22, kind='alfa', name='alfy_ip8'},
        { expr=\\t -> t.ip8.dx, kind='dx', name='dx_ip8'},
        { expr=\\t -> t.ip8.dpx, kind='dpx', name='dy_ip8'},
        { expr=\\t -> t.ip1.beta11-0.15, kind='beta', name='betx_ip1'},
        { expr=\\t -> t.ip1.beta22-0.1, kind='beta', name='bety_ip1'},
        { expr=\\t -> t.ip1.alfa11, kind='alfa', name='alfx_ip1'},
        { expr=\\t -> t.ip1.alfa22, kind='alfa', name='alfy_ip1'},
        { expr=\\t -> t.ip1.dx, kind='dx', name='dx_ip1'},
        { expr=\\t -> t.ip1.dpx, kind='dpx', name='dy_ip1'},
        { expr=\\t -> t["ip1.l1"].mu1 - mu1_to_ip1, kind='mu', name='mux_ip1'},
        { expr=\\t -> t["ip1.l1"].mu2 - mu2_to_ip1, kind='mu', name='muy_ip1'},
        ! { expr=\\t -> t.q1 - mu1_to_ip1, kind='mu', name='mux_ip1'},
        ! { expr=\\t -> t.q2 - mu2_to_ip1, kind='mu', name='muy_ip1'},

    },
    -- weights = {
    --    beta = 1.,
    --    alfa = 1.,
    --    dx = 1.,
    --    dpx = 1.,
    --    mu = 1.,
    --    qx = 1.,
    --    qy = 1.,
    --},
    objective = { broyden=true },
    info = 2,
    maxcall = 1000,
}
""")
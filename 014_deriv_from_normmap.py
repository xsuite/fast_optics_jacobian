import xtrack as xt
from xtrack._temp import lhc_match as lm
import numpy as np
import matplotlib.pyplot as plt
from xtrack.madng_interface import madng_get_init
from tpsa_util import TPSA
from tabulate import tabulate

# Load LHC model
collider = xt.Environment.from_json(
    '../xtrack/test_data/hllhc15_thick/hllhc15_collider_thick.json')
collider.vars.load_madx(
    '../xtrack/test_data/hllhc15_thick/opt_round_150_1500.madx')

collider.build_trackers()

line = collider.lhcb1

# Initial twiss
tw0 = line.twiss()

# Prepare for optics matching: set limits and steps for all circuits
lm.set_var_limits_and_steps(collider)

dk1 = 1e-5

start = 's.ds.l8.b1'
end = 'ip1'

opt = line.match(
    solve=False,
    default_tol={None: 1e-8, 'betx': 1e-6, 'bety': 1e-6, 'alfx': 1e-6, 'alfy': 1e-6},
    start=start, end=end,
    init=tw0, init_at=xt.START,
    vary=[
        xt.VaryList(['kq6.l8b1', 'kq7.l8b1', 'kq8.l8b1', 'kq9.l8b1', 'kq10.l8b1',
            'kqtl11.l8b1', 'kqt12.l8b1', 'kqt13.l8b1',
            'kq4.l8b1', 'kq5.l8b1', 'kq4.r8b1', 'kq5.r8b1',
            'kq6.r8b1', 'kq7.r8b1', 'kq8.r8b1', 'kq9.r8b1',
            'kq10.r8b1', 'kqtl11.r8b1', 'kqt12.r8b1', 'kqt13.r8b1'])],
    targets=[
        xt.TargetSet(at='ip8', tars=('betx', 'bety', 'alfx', 'alfy', 'dx', 'dpx'), value=tw0, weight=1),
        xt.TargetSet(at='ip1', betx=0.15, bety=0.1, alfx=0, alfy=0, dx=0, dpx=0, weight=1),
        # xt.TargetRelPhaseAdvance('mux', value = tw0['mux', 'ip1.l1'] - tw0['mux', 's.ds.l8.b1'], weight=1),
        # xt.TargetRelPhaseAdvance('muy', value = tw0['muy', 'ip1.l1'] - tw0['muy', 's.ds.l8.b1'], weight=1),
    ])

mng = line.to_madng(sequence_name='seq')

# get initial conditions x, px, y, py, t, pt
beta0 = line.particle_ref.beta0[0]
start = end
init_coord = np.array([tw0['x', start], tw0['px', start],
                       tw0['y', start], tw0['py', start],
                       tw0['zeta', start] * beta0, 0])

coord_str = ''
part_order = ['x', 'px', 'y', 'py', 't', 'pt']
for part, val in zip(part_order, init_coord):
    if np.abs(val) > 1e-12:
        coord_str += f'X0.{part} = {val} '

mng_str = r'''
        local obs_flag = MAD.element.flags.observed

        local pts={'ip8', 'ip1', 'ip2', 'ip3', 'ip4', 'ip5', 'ip6', 'ip7'}

        MADX.seq:select(obs_flag, {list=pts})

        local X0 = MAD.damap {
            nv=6, -- number of variables
            mo=2, -- max order of variables
            np=2, -- number of parameters
            po=1, -- max order of parameters
            pn={'kq6.l8b1', 'kq7.l8b1'}, -- parameter names
        }
        ''' + coord_str + r'''
        MADX['kq6.l8b1'] = MADX['kq6.l8b1'] + X0['kq6.l8b1'] -- Converting to TPSA (mutating type)
        MADX['kq7.l8b1'] = MADX['kq7.l8b1'] + X0['kq7.l8b1']

        -- X0:print()

        trk, mflw = MAD.track{sequence=MADX.seq, X0=X0, savemap=true} -- mflw is the final one!

        local ip8_map = trk['ip8'].__map -- coordinates at ip8

        local nf = MAD.gphys.normal(mflw[1]) -- Compute normal form for ip8-map (instead mflw[1])

        --[[ for i, v in ipairs(pts) do
            local map_i = trk[v].__map
            print('--- Map at '..v..' ---')
            local nfi = MAD.gphys.normal(map_i)
            local B0 = MAD.gphys.map2bet(nfi.a:real())
            print(B0.beta11)
        end
        ]]--

        local clearkeys in MAD.utility
        py:send(clearkeys(nf.a.__vn), true) -- Send keys as a list (ordered)

        for i, v in ipairs(nf.a.__vn) do
            py:send(nf.a[v]) -- Send TPSAs (Normal Forms) over in order
        end

        local B0 = MAD.gphys.map2bet(nf.a:real())
        -- print(B0.beta11)

        local a_re = nf.a:real()
        -- print(a_re.x:get("100000")^2 + a_re.x:get("010000")^2) -- deriv instead of get
        -- print(a_re.x:get("1000001"))

    '''

mng.send(mng_str)

tpsas = {k: mng.recv() for k in mng.recv()} # Create dict out of TPSAs
tpsa = TPSA(tpsas, num_variables=6) # Create TPSA object out of madng-dict

line['kq6.l8b1'] += dk1

tw_pert = line.twiss()

mng2 = line.to_madng(sequence_name='seq')
mng2.send(mng_str)

tpsas_pert = {k: mng2.recv() for k in mng2.recv()} # Create dict out of TPSAs
tpsa_pert = TPSA(tpsas_pert, num_variables=6)

line['kq6.l8b1'] -= dk1

x_ = np.array([1,0,0,0,0,0,0,0])
x_kq6 = np.array([1,0,0,0,0,0,1,0])
px_ = np.array([0,1,0,0,0,0,0,0])
px_kq6 = np.array([0,1,0,0,0,0,1,0])
y_ = np.array([0,0,1,0,0,0,0,0])
y_kq6 = np.array([0,0,1,0,0,0,1,0])
py_ = np.array([0,0,0,1,0,0,0,0])
py_kq6 = np.array([0,0,0,1,0,0,1,0])

x_x_coeff = tpsa.get_coeff('x', x_)
x_x_kq6_coeff = tpsa.get_coeff('x', x_kq6)
x_pert_x_coeff = tpsa_pert.get_coeff('x', x_)
x_pert_x_kq6_coeff = tpsa_pert.get_coeff('x', x_kq6)
x_px_coeff = tpsa.get_coeff('x', px_)
x_px_kq6_coeff = tpsa.get_coeff('x', px_kq6)
x_pert_px_coeff = tpsa_pert.get_coeff('x', px_)
x_pert_px_kq6_coeff = tpsa_pert.get_coeff('x', px_kq6)
y_y_coeff = tpsa.get_coeff('y', y_)
y_py_coeff = tpsa.get_coeff('y', py_)
y_y_kq6_coeff = tpsa.get_coeff('y', y_kq6)
y_py_kq6_coeff = tpsa.get_coeff('y', py_kq6)
y_pert_y_coeff = tpsa_pert.get_coeff('y', y_)
y_pert_py_coeff = tpsa_pert.get_coeff('y', py_)
y_pert_y_kq6_coeff = tpsa_pert.get_coeff('y', y_kq6)
y_pert_py_kq6_coeff = tpsa_pert.get_coeff('y', py_kq6)
px_x_coeff = tpsa.get_coeff('px', x_)
px_px_coeff = tpsa.get_coeff('px', px_)
px_x_kq6_coeff = tpsa.get_coeff('px', x_kq6)
px_px_kq6_coeff = tpsa.get_coeff('px', px_kq6)
px_pert_x_coeff = tpsa_pert.get_coeff('px', x_)
px_pert_px_coeff = tpsa_pert.get_coeff('px', px_)
px_pert_x_kq6_coeff = tpsa_pert.get_coeff('px', x_kq6)
px_pert_px_kq6_coeff = tpsa_pert.get_coeff('px', px_kq6)
py_y_coeff = tpsa.get_coeff('py', y_)
py_py_coeff = tpsa.get_coeff('py', py_)
py_y_kq6_coeff = tpsa.get_coeff('py', y_kq6)
py_py_kq6_coeff = tpsa.get_coeff('py', py_kq6)
py_pert_y_coeff = tpsa_pert.get_coeff('py', y_)
py_pert_py_coeff = tpsa_pert.get_coeff('py', py_)
py_pert_y_kq6_coeff = tpsa_pert.get_coeff('py', y_kq6)
py_pert_py_kq6_coeff = tpsa_pert.get_coeff('py', py_kq6)
x_zeta_coeff = tpsa.get_coeff('x', np.array([0,0,0,0,1,0,0,0]))
x_delta_coeff = tpsa.get_coeff('x', np.array([0,0,0,0,0,1,0,0]))
zeta_zeta_coeff = tpsa.get_coeff('t', np.array([0,0,0,0,1,0,0,0]))
delta_delta_coeff = tpsa.get_coeff('pt', np.array([0,0,0,0,0,1,0,0]))
zeta_delta_coeff = tpsa.get_coeff('t', np.array([0,0,0,0,0,1,0,0]))
delta_zeta_coeff = tpsa.get_coeff('pt', np.array([0,0,0,0,1,0,0,0]))

x_kq6_delta_coeff = tpsa_pert.get_coeff('x', np.array([0,0,0,0,0,1,1,0]))
x_kq6_zeta_coeff = tpsa_pert.get_coeff('x', np.array([0,0,0,0,1,0,1,0]))
zeta_kq6_delta_coeff = tpsa_pert.get_coeff('t', np.array([0,0,0,0,0,1,1,0]))
delta_kq6_delta_coeff = tpsa_pert.get_coeff('pt', np.array([0,0,0,0,0,1,1,0]))
delta_kq6_zeta_coeff = tpsa_pert.get_coeff('pt', np.array([0,0,0,0,1,0,1,0]))


x_pert_delta_coeff = tpsa_pert.get_coeff('x', np.array([0,0,0,0,0,1,0,0]))
x_pert_zeta_coeff = tpsa_pert.get_coeff('x', np.array([0,0,0,0,1,0,0,0]))

m = x_delta_coeff - (x_zeta_coeff * zeta_delta_coeff)/zeta_zeta_coeff
n = delta_delta_coeff - (delta_zeta_coeff * zeta_delta_coeff)/zeta_zeta_coeff
dm_dkq6 = x_kq6_delta_coeff - (x_kq6_zeta_coeff * zeta_delta_coeff - x_zeta_coeff * zeta_kq6_delta_coeff)/zeta_zeta_coeff**2
dn_dkq6 = delta_kq6_delta_coeff - (delta_kq6_zeta_coeff * zeta_delta_coeff - delta_zeta_coeff * zeta_kq6_delta_coeff)/zeta_zeta_coeff**2

m_pert = x_pert_delta_coeff - (x_pert_zeta_coeff * zeta_delta_coeff)/zeta_zeta_coeff
n_pert = delta_delta_coeff - (delta_zeta_coeff * zeta_delta_coeff)/zeta_zeta_coeff

print(tabulate([
    ['betx', tw0.betx[-1], x_x_coeff**2 + x_px_coeff**2, "---"],
    ['dbetx/dkq6', (tw_pert.betx[-1] - tw0.betx[-1]) / dk1,
        ((x_pert_x_coeff**2 + x_pert_px_coeff**2) -
         (x_x_coeff**2 + x_px_coeff**2)) / dk1,
        2 * (x_x_coeff * x_x_kq6_coeff + x_px_coeff * x_px_kq6_coeff)],
    ['bety', tw0.bety[-1], y_y_coeff**2 + y_py_coeff**2, "---"],
    ['dbety/dkq6', (tw_pert.bety[-1] - tw0.bety[-1]) / dk1,
        ((y_pert_y_coeff**2 + y_pert_py_coeff**2) -
         (y_y_coeff**2 + y_py_coeff**2)) / dk1,
        2 * (y_y_coeff * y_y_kq6_coeff + y_py_coeff * y_py_kq6_coeff)],
    ['alfx', tw0.alfx[-1], - x_x_coeff *
        px_x_coeff - x_px_coeff * px_px_coeff, "---"],
    ['dalfx/dkq6', (tw_pert.alfx[-1] - tw0.alfx[-1]) / dk1,
        - ((x_pert_x_coeff * px_pert_x_coeff + x_pert_px_coeff * px_pert_px_coeff)
            - (x_x_coeff * px_x_coeff + x_px_coeff * px_px_coeff)) / dk1,
        - (x_x_kq6_coeff * px_x_coeff + x_x_coeff * px_x_kq6_coeff
            + x_px_kq6_coeff * px_px_coeff + x_px_coeff * px_px_kq6_coeff)],
    ['alfy', tw0.alfy[-1], - y_y_coeff *
        py_y_coeff - y_py_coeff * py_py_coeff, "---"],
    ['dalfy/dkq6',
        (tw_pert.alfy[-1] - tw0.alfy[-1]) / dk1,
        - ((y_pert_y_coeff * py_pert_y_coeff + y_pert_py_coeff * py_pert_py_coeff)
            - (y_y_coeff * py_y_coeff + y_py_coeff * py_py_coeff)) / dk1,
        - (y_y_kq6_coeff * py_y_coeff + y_y_coeff * py_y_kq6_coeff
            + y_py_kq6_coeff * py_py_coeff + y_py_coeff * py_py_kq6_coeff)],
    ['dx', tw0.dx[-1], m * n**(-1), "---"],
    ['ddx/dkq6',
        (tw_pert.dx[-1] - tw0.dx[-1]) / dk1,
        ((m_pert * n_pert ** (-1)) - (m * n**(-1))) / dk1,
        (dm_dkq6 * n - m * dn_dkq6) / n**2],
], tablefmt="fancy_grid", headers=["Parameter", "Xsuite (Twiss/FD)", "TPSA (FD)", "TPSA Deriv"]))

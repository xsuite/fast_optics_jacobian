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

tw0_range = line.twiss(start=start, end=end, init=tw0, init_at=xt.START)

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
        MADX['kq6.l8b1'] = MADX['kq6.l8b1']:get0()
        MADX['kq7.l8b1'] = MADX['kq7.l8b1']:get0()

    '''

mng.send(mng_str)

tpsas = {k: mng.recv() for k in mng.recv()} # Create dict out of TPSAs
tpsa = TPSA(tpsas, num_variables=6) # Create TPSA object out of madng-dict

line['kq6.l8b1'] += dk1

tw_pert = line.twiss()
tw_range_pert = line.twiss(start=start, end=end, init=tw_pert, init_at=xt.START)
tw_range_pert_base = line.twiss(start=start, end=end, init=tw0, init_at=xt.START) # This is the one that yields the proper jacobians

mng2 = line.to_madng(sequence_name='seq')
mng2.send(mng_str)

tpsas_pert = {k: mng2.recv() for k in mng2.recv()} # Create dict out of TPSAs
tpsa_pert = TPSA(tpsas_pert, num_variables=6)

line['kq6.l8b1'] -= dk1

obsv = 'ip1'

jac = opt._err.get_jacobian(opt._err._get_x())

mng_str_update = r'''
    local obs_flag = MAD.element.flags.observed

    local pts={'ip1', 'ip2', 'ip3', 'ip4', 'ip5', 'ip6', 'ip7', 'ip8'}

    MADX.seq:select(obs_flag, {list=pts})

    local tw = MAD.twiss{sequence=MADX.seq}

    local X0 = MAD.damap {
        nv=6, -- number of variables
        mo=3, -- max order of variables
        np=2, -- number of parameters
        po=1, -- max order of parameters
        pn={'kq6.l8b1', 'kq7.l8b1'}, -- parameter names
    }

    MADX['kq6.l8b1'] = MADX['kq6.l8b1'] + X0['kq6.l8b1'] -- Converting to TPSA (mutating type)
    MADX['kq7.l8b1'] = MADX['kq7.l8b1'] + X0['kq7.l8b1']

    -- for i, v in ipairs({'x', 'px', 'y', 'py', 't', 'pt'}) do
    --     local derivs = MADX.seq.X0[v]:getvec(1,7)
    --     X0[v]:setvec(1, derivs)
    -- end

    local trk, mflw = MAD.track{sequence=MADX.seq, X0=X0}

    local nf = MAD.gphys.normal(mflw[1])
    local B0 = MAD.gphys.map2bet(nf.a:real())
    print(B0.beta11) -- 0.15

    local pos = 's.ds.l8.b1'

    local beta11 = tw[pos].beta11
    local beta22 = tw[pos].beta22
    local alfa11 = tw[pos].alfa11
    local alfa22 = tw[pos].alfa22
    local dx = tw[pos].dx
    local dpx = tw[pos].dpx
    local dy = tw[pos].dy
    local dpy = tw[pos].dpy
    local betas = 1

    local mat = {
        {math.sqrt(beta11), 0, 0, 0, 0, dx},
        {-alfa11/math.sqrt(beta11), 1/math.sqrt(beta11), 0, 0, 0, dpx},
        {0, 0, math.sqrt(beta22), 0, 0, dy},
        {0, 0, -alfa22/math.sqrt(beta22), 1/math.sqrt(beta22), 0, dpy},
        {0, 0, 0, 0, betas, 0},
        {0, 0, 0, 0, 0, 1/betas},
    }

    local cmat = MAD.cmatrix(mat)

    local a_clear = nf.a:clear()
    local a_deriv = nf.a:set1(cmat):real()

    local trk2, mflw2 = MAD.track{sequence=MADX.seq, X0=a_deriv, range='s.ds.l8.b1/ip1'}
    local a_re_exit = mflw2[1]

    local B0_exit = MAD.gphys.map2bet(a_re_exit:real())


    print("Beta11 at ip2:")
    print(tw['ip2'].beta11) -- from twiss: 9.999999
    print(B0_exit.beta11) -- 9.999993!

    local clearkeys in MAD.utility
    py:send(clearkeys(a_re_exit.__vn), true) -- Send keys as a list (ordered)

    for i, v in ipairs(a_re_exit.__vn) do
        py:send(a_re_exit[v]) -- Send TPSAs (Normal Forms) over in order
    end

    '''

mng.send(mng_str_update)

tpsas_range = {k: mng.recv() for k in mng.recv()} # Create dict out of TPSAs
tpsa_range = TPSA(tpsas_range, num_variables=6) # Create TPSA object out of madng-dict

# Base vectors
vars_base = {
    "x": np.array([1,0,0,0,0,0,0,0]),
    "px": np.array([0,1,0,0,0,0,0,0]),
    "y": np.array([0,0,1,0,0,0,0,0]),
    "py": np.array([0,0,0,1,0,0,0,0]),
    "zeta": np.array([0,0,0,0,1,0,0,0]),
    "delta": np.array([0,0,0,0,0,1,0,0]),
}

# kq6 perturbations
vars_kq6 = {
    "x": np.array([1,0,0,0,0,0,1,0]),
    "px": np.array([0,1,0,0,0,0,1,0]),
    "y": np.array([0,0,1,0,0,0,1,0]),
    "py": np.array([0,0,0,1,0,0,1,0]),
    "zeta": np.array([0,0,0,0,1,0,1,0]),
    "delta": np.array([0,0,0,0,0,1,1,0]),
}

tpsa_objs = {
    "ref": tpsa,
    "pert": tpsa_pert,
    "range": tpsa_range
}

x_x_coeff = tpsa.get_coeff('x', vars_base['x'])
x_x_kq6_coeff = tpsa.get_coeff('x', vars_kq6['x'])
x_pert_x_coeff = tpsa_pert.get_coeff('x', vars_base['x'])
x_px_coeff = tpsa.get_coeff('x', vars_base['px'])
x_px_kq6_coeff = tpsa.get_coeff('x', vars_kq6['px'])
x_pert_px_coeff = tpsa_pert.get_coeff('x', vars_base['px'])
y_y_coeff = tpsa.get_coeff('y', vars_base['y'])
y_py_coeff = tpsa.get_coeff('y', vars_base['py'])
y_y_kq6_coeff = tpsa.get_coeff('y', vars_kq6['y'])
y_py_kq6_coeff = tpsa.get_coeff('y', vars_kq6['py'])
y_pert_y_coeff = tpsa_pert.get_coeff('y', vars_base['y'])
y_pert_py_coeff = tpsa_pert.get_coeff('y', vars_base['py'])
px_x_coeff = tpsa.get_coeff('px', vars_base['x'])
px_px_coeff = tpsa.get_coeff('px', vars_base['px'])
px_x_kq6_coeff = tpsa.get_coeff('px', vars_kq6['x'])
px_px_kq6_coeff = tpsa.get_coeff('px', vars_kq6['px'])
px_pert_x_coeff = tpsa_pert.get_coeff('px', vars_base['x'])
px_pert_px_coeff = tpsa_pert.get_coeff('px', vars_base['px'])
py_y_coeff = tpsa.get_coeff('py', vars_base['y'])
py_py_coeff = tpsa.get_coeff('py', vars_base['py'])
py_y_kq6_coeff = tpsa.get_coeff('py', vars_kq6['y'])
py_py_kq6_coeff = tpsa.get_coeff('py', vars_kq6['py'])
py_pert_y_coeff = tpsa_pert.get_coeff('py', vars_base['y'])
py_pert_py_coeff = tpsa_pert.get_coeff('py', vars_base['py'])
x_zeta_coeff = tpsa.get_coeff('x', vars_base['zeta'])
x_delta_coeff = tpsa.get_coeff('x', vars_base['delta'])
zeta_zeta_coeff = tpsa.get_coeff('t', vars_base['zeta'])
delta_delta_coeff = tpsa.get_coeff('pt', vars_base['delta'])
zeta_delta_coeff = tpsa.get_coeff('t', vars_base['delta'])
delta_zeta_coeff = tpsa.get_coeff('pt', vars_base['zeta'])

x_kq6_delta_coeff = tpsa_pert.get_coeff('x', vars_kq6['delta'])
x_kq6_zeta_coeff = tpsa_pert.get_coeff('x', vars_kq6['zeta'])
zeta_kq6_delta_coeff = tpsa_pert.get_coeff('t', vars_kq6['delta'])
delta_kq6_delta_coeff = tpsa_pert.get_coeff('pt', vars_kq6['delta'])
delta_kq6_zeta_coeff = tpsa_pert.get_coeff('pt', vars_kq6['zeta'])


x_pert_delta_coeff = tpsa_pert.get_coeff('x', vars_base['delta'])
x_pert_zeta_coeff = tpsa_pert.get_coeff('x', vars_base['zeta'])

m = x_delta_coeff - (x_zeta_coeff * zeta_delta_coeff)/zeta_zeta_coeff
n = delta_delta_coeff - (delta_zeta_coeff * zeta_delta_coeff)/zeta_zeta_coeff
dm_dkq6 = x_kq6_delta_coeff - (x_kq6_zeta_coeff * zeta_delta_coeff - x_zeta_coeff * zeta_kq6_delta_coeff)/zeta_zeta_coeff**2
dn_dkq6 = delta_kq6_delta_coeff - (delta_kq6_zeta_coeff * zeta_delta_coeff - delta_zeta_coeff * zeta_kq6_delta_coeff)/zeta_zeta_coeff**2

m_pert = x_pert_delta_coeff - (x_pert_zeta_coeff * zeta_delta_coeff)/zeta_zeta_coeff
n_pert = delta_delta_coeff - (delta_zeta_coeff * zeta_delta_coeff)/zeta_zeta_coeff



tpsa_objs['range'] = tpsa_range

x_x_coeff_3 = tpsa_range.get_coeff('x', vars_base['x'])
x_x_kq6_coeff_3 = tpsa_range.get_coeff('x', vars_kq6['x'])
x_px_coeff_3 = tpsa_range.get_coeff('x', vars_base['px'])
x_px_kq6_coeff_3 = tpsa_range.get_coeff('x', vars_kq6['px'])
px_x_coeff_3 = tpsa_range.get_coeff('px', vars_base['x'])
px_px_coeff_3 = tpsa_range.get_coeff('px', vars_base['px'])
px_x_kq6_coeff_3 = tpsa_range.get_coeff('px', vars_kq6['x'])
px_px_kq6_coeff_3 = tpsa_range.get_coeff('px', vars_kq6['px'])
y_y_coeff_3 = tpsa_range.get_coeff('y', vars_base['y'])
y_y_kq6_coeff_3 = tpsa_range.get_coeff('y', vars_kq6['y'])
y_py_coeff_3 = tpsa_range.get_coeff('y', vars_base['py'])
y_py_kq6_coeff_3 = tpsa_range.get_coeff('y', vars_kq6['py'])
py_y_coeff_3 = tpsa_range.get_coeff('py', vars_base['y'])
py_py_coeff_3 = tpsa_range.get_coeff('py', vars_base['py'])
py_y_kq6_coeff_3 = tpsa_range.get_coeff('py', vars_kq6['y'])
py_py_kq6_coeff_3 = tpsa_range.get_coeff('py', vars_kq6['py'])
x_zeta_coeff_3 = tpsa_range.get_coeff('x', vars_base['zeta'])
x_delta_coeff_3 = tpsa_range.get_coeff('x', vars_base['delta'])
zeta_zeta_coeff_3 = tpsa_range.get_coeff('t', vars_base['zeta'])
delta_delta_coeff_3 = tpsa_range.get_coeff('pt', vars_base['delta'])
zeta_delta_coeff_3 = tpsa_range.get_coeff('t', vars_base['delta'])
delta_zeta_coeff_3 = tpsa_range.get_coeff('pt', vars_base['zeta'])
x_kq6_delta_coeff_3 = tpsa_range.get_coeff('x', vars_kq6['delta'])
x_kq6_zeta_coeff_3 = tpsa_range.get_coeff('x', vars_kq6['zeta'])
zeta_kq6_delta_coeff_3 = tpsa_range.get_coeff('t', vars_kq6['delta'])
delta_kq6_delta_coeff_3 = tpsa_range.get_coeff('pt', vars_kq6['delta'])
delta_kq6_zeta_coeff_3 = tpsa_range.get_coeff('pt', vars_kq6['zeta'])

m_3 = x_delta_coeff_3 - (x_zeta_coeff_3 * zeta_delta_coeff_3)/zeta_zeta_coeff_3
n_3 = delta_delta_coeff_3 - (delta_zeta_coeff_3 * zeta_delta_coeff_3)/zeta_zeta_coeff_3
dm_dkq6_3 = x_kq6_delta_coeff_3 - (x_kq6_zeta_coeff_3 * zeta_delta_coeff_3 - x_zeta_coeff_3 * zeta_kq6_delta_coeff_3)/zeta_zeta_coeff_3**2
dn_dkq6_3 = delta_kq6_delta_coeff_3 - (delta_kq6_zeta_coeff_3 * zeta_delta_coeff_3 - delta_zeta_coeff_3 * zeta_kq6_delta_coeff_3)/zeta_zeta_coeff_3**2

print(tabulate([
    ['betx', tw0['betx', obsv], x_x_coeff**2 + x_px_coeff**2, "---", "---", "---", "---"],
    ['dbetx/dkq6', (tw_pert['betx', obsv] - tw0['betx', obsv]) / dk1,
        ((x_pert_x_coeff**2 + x_pert_px_coeff**2) -
         (x_x_coeff**2 + x_px_coeff**2)) / dk1,
        2 * (x_x_coeff * x_x_kq6_coeff + x_px_coeff * x_px_kq6_coeff),
        (tw_range_pert_base['betx', obsv] - tw0_range['betx', obsv]) / dk1,
        jac[6][0],
        2 * (x_x_coeff_3 * x_x_kq6_coeff_3 + x_px_coeff_3 * x_px_kq6_coeff_3)
    ],
    ['bety', tw0['bety', obsv], y_y_coeff**2 + y_py_coeff**2, "---", "---", "---", "---"],
    ['dbety/dkq6', (tw_pert['bety', obsv] - tw0['bety', obsv]) / dk1,
        ((y_pert_y_coeff**2 + y_pert_py_coeff**2) -
         (y_y_coeff**2 + y_py_coeff**2)) / dk1,
        2 * (y_y_coeff * y_y_kq6_coeff + y_py_coeff * y_py_kq6_coeff),
        (tw_range_pert_base['bety', obsv] - tw0_range['bety', obsv]) / dk1,
        jac[7][0],
        2 * (y_y_coeff_3 * y_y_kq6_coeff_3 + y_py_coeff_3 * y_py_kq6_coeff_3)
    ],
    ['alfx', tw0['alfx', obsv], - x_x_coeff *
        px_x_coeff - x_px_coeff * px_px_coeff, "---", "---", "---", "---"],
    ['dalfx/dkq6', (tw_pert['alfx', obsv] - tw0['alfx', obsv]) / dk1,
        - ((x_pert_x_coeff * px_pert_x_coeff + x_pert_px_coeff * px_pert_px_coeff)
            - (x_x_coeff * px_x_coeff + x_px_coeff * px_px_coeff)) / dk1,
        - (x_x_kq6_coeff * px_x_coeff + x_x_coeff * px_x_kq6_coeff
            + x_px_kq6_coeff * px_px_coeff + x_px_coeff * px_px_kq6_coeff),
        (tw_range_pert_base['alfx', obsv] - tw0_range['alfx', obsv]) / dk1,
        jac[8][0],
        - (x_x_kq6_coeff_3 * px_x_coeff_3 + x_x_coeff_3 * px_x_kq6_coeff_3
            + x_px_kq6_coeff_3 * px_px_coeff_3 + x_px_coeff_3 * px_px_kq6_coeff_3)
    ],
    ['alfy', tw0['alfy', obsv], - y_y_coeff *
        py_y_coeff - y_py_coeff * py_py_coeff, "---", "---", "---", "---"],
    ['dalfy/dkq6',
        (tw_pert['alfy', obsv] - tw0['alfy', obsv]) / dk1,
        - ((y_pert_y_coeff * py_pert_y_coeff + y_pert_py_coeff * py_pert_py_coeff)
            - (y_y_coeff * py_y_coeff + y_py_coeff * py_py_coeff)) / dk1,
        - (y_y_kq6_coeff * py_y_coeff + y_y_coeff * py_y_kq6_coeff
            + y_py_kq6_coeff * py_py_coeff + y_py_coeff * py_py_kq6_coeff),
        (tw_range_pert_base['alfy', obsv] - tw0_range['alfy', obsv]) / dk1,
        jac[9][0],
        - (y_y_kq6_coeff_3 * py_y_coeff_3 + y_y_coeff_3 * py_y_kq6_coeff_3
            + y_py_kq6_coeff_3 * py_py_coeff_3 + y_py_coeff_3 * py_py_kq6_coeff_3)
    ],
    ['dx', tw0['dx', obsv], m * n**(-1), "---", "---", "---", "---"],
    ['ddx/dkq6',
        (tw_pert['dx', obsv] - tw0['dx', obsv]) / dk1,
        ((m_pert * n_pert ** (-1)) - (m * n**(-1))) / dk1,
        (dm_dkq6 * n - m * dn_dkq6) / n**2,
        (tw_range_pert_base['dx', obsv] - tw0_range['dx', obsv]) / dk1,
        jac[10][0],
        (dm_dkq6_3 * n_3 - m_3 * dn_dkq6_3) / n_3**2
    ],
], tablefmt="fancy_grid", headers=["Parameter", "Xsuite (Twiss/FD)", "TPSA (FD)", "TPSA Deriv", "Xsuite Deriv (Twiss Range FD)", "Xsuite Jac", "MNG Range TPSA"]))

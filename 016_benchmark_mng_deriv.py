import xtrack as xt
from xtrack._temp import lhc_match as lm
import numpy as np
import matplotlib.pyplot as plt
from xtrack.madng_interface import madng_get_init
from tpsa_util import TPSA
from tabulate import tabulate
from pyprof import timing

timing.reset()

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

start = 's.ds.l8.b1'
end = 'ip1'

vary_names = ['kq6.l8b1', 'kq7.l8b1', 'kq8.l8b1', 'kq9.l8b1', 'kq10.l8b1',
            'kqtl11.l8b1', 'kqt12.l8b1', 'kqt13.l8b1',
            'kq4.l8b1', 'kq5.l8b1', 'kq4.r8b1', 'kq5.r8b1',
            'kq6.r8b1', 'kq7.r8b1', 'kq8.r8b1', 'kq9.r8b1',
            'kq10.r8b1', 'kqtl11.r8b1', 'kqt12.r8b1', 'kqt13.r8b1']

tw0_range = line.twiss(start=start, end=end, init=tw0, init_at=xt.START)

opt = line.match(
    solve=False,
    default_tol={None: 1e-8, 'betx': 1e-6, 'bety': 1e-6, 'alfx': 1e-6, 'alfy': 1e-6},
    start=start, end=end,
    init=tw0, init_at=xt.START,
    vary=[
        xt.VaryList(vary_names)
    ],
    targets=[
        xt.TargetSet(at='ip8', tars=('betx', 'bety', 'alfx', 'alfy', 'dx', 'dpx'), value=tw0, weight=1),
        xt.TargetSet(at='ip1', betx=0.15, bety=0.1, alfx=0, alfy=0, dx=0, dpx=0, weight=1),
        # xt.TargetRelPhaseAdvance('mux', value = tw0['mux', 'ip1.l1'] - tw0['mux', 's.ds.l8.b1'], weight=1),
        # xt.TargetRelPhaseAdvance('muy', value = tw0['muy', 'ip1.l1'] - tw0['muy', 's.ds.l8.b1'], weight=1),
    ])

seq_name = 'seq'
mng = line.to_madng(sequence_name=seq_name)

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

def _ng_build_param_assignment(vary_names):
    param_str = ''
    for name in vary_names:
        param_str += f"MADX['{name}'] = MADX['{name}'] + X0['{name}'] \n"

    return param_str

def _ng_build_param_list(vary_names):
    param_str = '{'
    for name in vary_names:
        param_str += f"'{name}', "
    param_str = param_str[:-2] + '}'
    return param_str

def _ng_build_observables_list(start, end, target_locations=None):
    loc_str = '{' + f"'{start}', '{end}', "
    if target_locations is not None:
        for loc in target_locations:
            if loc != start and loc != end:
                loc_str += f"'{loc}', "
    loc_str = loc_str[:-2] + '}'
    return loc_str

tar_locations = {t.tar[1] for t in opt.targets if isinstance(t.tar, tuple)}

observables_str = _ng_build_observables_list(start, end, tar_locations)
param_list_str = _ng_build_param_list(vary_names)
param_assignment_str = _ng_build_param_assignment(vary_names)
range_str = f"range='{start}/{end}'"

init_cond_str = f"local beta11 = {tw0['betx', start]}\n" + f"local beta22 = {tw0['bety', start]}\n"\
                + f"local alfa11 = {tw0['alfx', start]}\n" + f"local alfa22 = {tw0['alfy', start]}\n"\
                + f"local dx = {tw0['dx', start]}\n" + f"local dpx = {tw0['dpx', start]}\n"\
                + f"local dy = {tw0['dy', start]}\n" + f"local dpy = {tw0['dpy', start]}\n"\
                + f"local betas = 1\n"

mng_init_str = r'''
    local obs_flag = MAD.element.flags.observed

    local pts=''' + observables_str + r'''

    MADX.seq:select(obs_flag, {list=pts})

    local X0 = MAD.damap {
        nv=6, -- number of variables
        mo=2, -- max order of variables
        np=''' + str(len(vary_names)) + r''', -- number of parameters
        po=1, -- max order of parameters
        pn=''' + param_list_str + r''', -- parameter names
    }

    ''' + coord_str + r'''

    -- Converting to TPSA (mutating type)
    ''' + param_assignment_str + r'''

    local pos = ''' + f"'{start}'" + r'''

    ''' + init_cond_str + r'''

    local mat = {
        {math.sqrt(beta11), 0, 0, 0, 0, dx},
        {-alfa11/math.sqrt(beta11), 1/math.sqrt(beta11), 0, 0, 0, dpx},
        {0, 0, math.sqrt(beta22), 0, 0, dy},
        {0, 0, -alfa22/math.sqrt(beta22), 1/math.sqrt(beta22), 0, dpy},
        {0, 0, 0, 0, betas, 0},
        {0, 0, 0, 0, 0, 1/betas},
    }

    local mat = MAD.matrix(mat)


    MADX.X0 = X0:set1(mat)


    py:send(nil) -- Send a signal that initialization is done
    '''

timing.start_timing("MAD-NG Twiss/init damap")
mng.send(mng_init_str)
mng.recv()
timing.stop_timing()

tpsa_dict = {}

mng_track_str = r'''
    local trk2, mflw2 = MAD.track{sequence=MADX.seq, X0=MADX.X0, savemap=true,''' + range_str + r'''}
    MADX.trk2 = trk2
    py:send(nil)
    '''

timing.start_timing("MAD-NG Track and TPSA")
mng.send(mng_track_str)
mng.recv()


for loc in tar_locations:
    loc_map_str = f"local a_re_exit = MADX.trk2['{loc}'].__map\n"

    mng_map_str = loc_map_str + r'''
    local clearkeys in MAD.utility
    py:send(clearkeys(a_re_exit.__vn), true) -- Send keys as a list (ordered)

    for i, v in ipairs(a_re_exit.__vn) do
        py:send(a_re_exit[v]) -- Send TPSAs (Normal Forms) over in order
    end

    '''

    mng.send(mng_map_str)

    tpsas = {k: mng.recv() for k in mng.recv()} # Create dict out of TPSAs
    tpsa = TPSA(tpsas, num_variables=6) # Create TPSA object out of madng-dict
    tpsa_dict[loc] = tpsa
timing.stop_timing()


jac = opt._err.get_jacobian(opt._err._get_x())

def _arr_from_pos(pos, length):
    return np.isin(np.arange(length), pos).astype(int)


def build_jacobian(opt, tpsa_dict):
    jac_tpsa = np.zeros((len(opt._err.targets), len(opt._err.vary)))

    for i, tar in enumerate(opt._err.targets):
        loc = ''
        if isinstance(tar.tar, tuple):
            param = tar.tar[0]
            loc = tar.tar[1]
        else:
            param = tar.tar
            loc = '--'
        tpsa = tpsa_dict[loc]

        for j, var in enumerate(opt._err.vary):
            if param[0] == 'd':
                #deriv = dispersion_deriv(tpsa, j + tpsa.num_variables, param[1:])
                deriv = tpsa.calc_dispersion_deriv(param[1:], j + tpsa.num_variables)
            elif param[:3] == 'bet':
                #deriv = beta_deriv(tpsa, j + tpsa.num_variables, param[-1])
                deriv = tpsa.calc_beta_deriv(param[-1], j + tpsa.num_variables)
            elif param[:3] == 'alf':
                #deriv = alfa_deriv(tpsa, j + tpsa.num_variables, param[-1])
                deriv = tpsa.calc_alpha_deriv(param[-1], j + tpsa.num_variables)
            else:
                raise ValueError(f"Unknown target location {param}")

            jac_tpsa[i, j] = deriv

    return jac_tpsa

def alfa_deriv(tpsa, var_index, dim):
    assert dim in ['x', 'y'], "Dimension should be 'x' yor 'y'"
    p_dim = 'p' + dim
    dim_ind = 2 if dim == 'y' else 0
    monom_length = tpsa.monom_length

    x_x_coeff = tpsa.get_coeff(dim, _arr_from_pos([dim_ind], monom_length))
    px_x_coeff = tpsa.get_coeff(p_dim, _arr_from_pos([dim_ind], monom_length))
    x_x_deriv_coeff = tpsa.get_coeff(dim, _arr_from_pos([dim_ind, var_index], monom_length))
    px_x_deriv_coeff = tpsa.get_coeff(p_dim, _arr_from_pos([dim_ind, var_index], monom_length))

    x_px_coeff = tpsa.get_coeff(dim, _arr_from_pos([dim_ind + 1], monom_length))
    px_px_coeff = tpsa.get_coeff(p_dim, _arr_from_pos([dim_ind + 1], monom_length))
    x_px_deriv_coeff = tpsa.get_coeff(dim, _arr_from_pos([dim_ind + 1, var_index], monom_length))
    px_px_deriv_coeff = tpsa.get_coeff(p_dim, _arr_from_pos([dim_ind + 1, var_index], monom_length))

    dalfa = - (x_x_deriv_coeff * px_x_coeff + x_x_coeff * px_x_deriv_coeff
               + x_px_deriv_coeff * px_px_coeff + x_px_coeff * px_px_deriv_coeff)

    return dalfa

def beta_deriv(tpsa, var_index, dim):
    assert dim in ['x', 'y'], "Dimension should be 'x' or 'y'"
    dim_ind = 2 if dim == 'y' else 0
    monom_length = tpsa.monom_length
    x_x_coeff = tpsa.get_coeff(dim, _arr_from_pos([dim_ind], monom_length))
    x_px_coeff = tpsa.get_coeff(dim, _arr_from_pos([dim_ind + 1], monom_length))
    x_x_deriv_coeff = tpsa.get_coeff(dim, _arr_from_pos([dim_ind, var_index], monom_length))
    x_px_deriv_coeff = tpsa.get_coeff(dim, _arr_from_pos([dim_ind + 1, var_index], monom_length))

    dbeta = 2 * (x_x_coeff * x_x_deriv_coeff + x_px_coeff * x_px_deriv_coeff)

    return dbeta

def dispersion_deriv(tpsa, var_index, dim):
    assert dim in ['x', 'y', 'px', 'py'], "Dimension should be 'x', 'y', 'px' or 'py'"
    monom_length = tpsa.monom_length

    x_delta_coeff = tpsa.get_coeff(dim, _arr_from_pos([5], monom_length))
    x_delta_deriv_coeff = tpsa.get_coeff(dim, _arr_from_pos([5, var_index], monom_length))
    x_zeta_deriv_coeff = tpsa.get_coeff(dim, _arr_from_pos([4, var_index], monom_length))
    zeta_delta_coeff = tpsa.get_coeff('t', _arr_from_pos([5], monom_length))
    x_zeta_coeff = tpsa.get_coeff(dim, _arr_from_pos([4], monom_length))
    zeta_delta_deriv_coeff = tpsa.get_coeff('t', _arr_from_pos([5, var_index], monom_length))
    zeta_zeta_coeff = tpsa.get_coeff('t', _arr_from_pos([4], monom_length))
    delta_zeta_coeff = tpsa.get_coeff('pt', _arr_from_pos([4], monom_length))
    delta_delta_coeff = tpsa.get_coeff('pt', _arr_from_pos([5], monom_length))

    delta_delta_deriv_coeff = tpsa.get_coeff('pt', _arr_from_pos([5, var_index], monom_length))
    delta_zeta_deriv_coeff = tpsa.get_coeff('pt', _arr_from_pos([4, var_index], monom_length))

    m = x_delta_coeff - (x_zeta_coeff * zeta_delta_coeff)/zeta_zeta_coeff
    n = delta_delta_coeff - (delta_zeta_coeff * zeta_delta_coeff)/zeta_zeta_coeff

    dm = x_delta_deriv_coeff - (x_zeta_deriv_coeff * zeta_delta_coeff - x_zeta_coeff * zeta_delta_deriv_coeff)/zeta_zeta_coeff**2
    dn = delta_delta_deriv_coeff - (delta_zeta_deriv_coeff * zeta_delta_coeff - delta_zeta_coeff * zeta_delta_deriv_coeff)/zeta_zeta_coeff**2

    dd = (dm * n - m * dn) / n**2

    return dd

timing.start_timing("Build Jacobian TPSA (only 6)")
jac_ng = build_jacobian(opt, tpsa_dict)
timing.stop_timing()

for i in range(20):
    timing.start_timing("Range Twiss")
    tw_range = line.twiss(start=start, end=end, init=tw0, init_at=xt.START)
    timing.stop_timing()

for i in range(20):
    timing.start_timing("Range Twiss Chromatic")
    tw_range = line.twiss(start=start, end=end, init=tw0, init_at=xt.START, compute_chromatic_properties=True)
    timing.stop_timing()

timing.report()
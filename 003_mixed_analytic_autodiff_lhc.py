import xtrack as xt
import numpy as np
import jax.numpy as jnp
import jax
from tabulate import tabulate
from xtrack._temp import lhc_match as lm

env = xt.Environment()
env.particle_ref = xt.Particles(p0c=7e12)

collider = xt.Environment.from_json(
    '../xtrack/test_data/hllhc15_thick/hllhc15_collider_thick.json')
collider.vars.load_madx(
    '../xtrack/test_data/hllhc15_thick/opt_round_150_1500.madx')

collider.build_trackers()

line = collider.lhcb1

#line.cycle('ip7', inplace=True)

# Initial twiss
tw0 = line.twiss()

tw_copy = tw0

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

# Bends -> bend
# Quadrupole -> Quad
# Drift, Sextupole, Octupole -> Drift

#opt.solve()

start_point = 'ip1'
limit = 4000
end_point = tw_copy.rows[limit].name[0]

for i in range(limit):
    if isinstance(line.elements[i], xt.Bend) or isinstance(line.elements[i], xt.RBend):
        line.elements[i].edge_entry_active=0
        line.elements[i].edge_exit_active=0

#jax.config.update("jax_enable_x64", True)

tw0 = line.twiss4d(start=start_point, end=end_point, betx=0.15, bety=0.15)
trunc_elements = np.array(line.elements)[np.logical_and(line.get_s_position() >= np.float64(line.get_s_position(start_point)), line.get_s_position() <= np.float64(line.get_s_position(end_point)))]

quadrupoles = [elem for elem in trunc_elements if isinstance(elem, xt.Quadrupole) and elem.k1 != 0]

eps = 1e-6
grads_fd = []
for quad in quadrupoles:
    quad.k1 += eps
    tw_plus = line.twiss4d(init=tw0, start=start_point, end=end_point)
    quad.k1 -= 2 * eps
    tw_minus = line.twiss4d(init=tw0, start=start_point, end=end_point)
    quad.k1 += eps

    fd_dict = {
    'betx': (tw_plus.betx[-1] - tw_minus.betx[-1]) / (2 * eps),
    'bety': (tw_plus.bety[-1] - tw_minus.bety[-1]) / (2 * eps),
    'alfx': (tw_plus.alfx[-1] - tw_minus.alfx[-1]) / (2 * eps),
    'alfy': (tw_plus.alfy[-1] - tw_minus.alfy[-1]) / (2 * eps),
    'mux': (tw_plus.mux[-1] - tw_minus.mux[-1]) / (2 * eps),
    'muy': (tw_plus.muy[-1] - tw_minus.muy[-1]) / (2 * eps),
    'dx': (tw_plus.dx[-1] - tw_minus.dx[-1]) / (2 * eps),
    'dpx': (tw_plus.dpx[-1] - tw_minus.dpx[-1]) / (2 * eps),
    'dy': (tw_plus.dy[-1] - tw_minus.dy[-1]) / (2 * eps),
    'dpy': (tw_plus.dpy[-1] - tw_minus.dpy[-1]) / (2 * eps),
    }
    grads_fd.append(fd_dict)

tw0 = line.twiss4d(start=start_point, end=end_point, betx=0.15, bety=0.15)

# Reorder list of dictionaries to have Dictionary of lists
grads_fd = {key: [d[key] for d in grads_fd] for key in grads_fd[0].keys()}

# This can be differentiated when assuming that no quadrupole, that should be differentiated, has k1 = 0.
@jax.jit
def get_transfer_matrix_quad(k1, l, beta0, gamma0):
    kx = jnp.sqrt(k1.astype(complex))
    ky = jnp.sqrt(-k1.astype(complex))
    sx = l * jnp.sinc(kx * l / jnp.pi)
    cx = jnp.cos(kx * l)
    sy = l * jnp.sinc(ky * l / jnp.pi) # limit of sin(ky * l) / ky when ky -> 0
    cy = jnp.cos(ky * l)

    f_matrix = jnp.array([
        [cx, sx, 0, 0, 0, 0],
        [-kx**2 * sx, cx, 0, 0, 0, 0],
        [0, 0, cy, sy, 0, 0],
        [0, 0, -ky**2 * sy, cy, 0, 0],
        [0, 0, 0, 0, 1, l/(beta0**2 * gamma0**2)],
        [0, 0, 0, 0, 0, 1]
    ])

    return f_matrix.real

@jax.jit
def get_transfer_matrix_drift(l, beta0, gamma0):
    f_matrix = jnp.array([
        [1, l, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, l, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, l/(beta0**2 * gamma0**2)],
        [0, 0, 0, 0, 0, 1]
    ])
    return f_matrix

@jax.jit
def get_transfer_matrix_bend(k0, k1, l, h, beta0, gamma0):
    kx = jnp.sqrt((h * k0 + k1).astype(complex))
    ky = jnp.sqrt(-k1.astype(complex)) # for dipoles usually 0
    sx = l * jnp.sinc(kx * l / jnp.pi)
    cx = jnp.cos(kx * l)
    sy = l * jnp.sinc(ky * l / jnp.pi)
    cy = jnp.cos(ky * l)
    dx = (1 - cx) / kx**2
    j1 = (l - sx) / kx**2

    f_matrix = jnp.array([
        [cx, sx, 0, 0, 0, h/beta0 * dx],
        [-kx**2 * sx, cx, 0, 0, 0, h/beta0 * sx],
        [0, 0, cy, sy, 0, 0],
        [0, 0, -ky**2 * sy, cy, 0, 0],
        [-h/beta0 * sx, -h/beta0 * dx, 0, 0, 1, l/(beta0**2 * gamma0**2) - h**2/beta0**2 * j1],
        [0, 0, 0, 0, 0, 1]
    ])

    return f_matrix.real

def get_values_from_transfer_matrix(transfer_matrix, tw0):
    betx = 1/tw0.betx[0] * ((transfer_matrix[0, 0] * tw0.betx[0] - transfer_matrix[0, 1] * tw0.alfx[0])**2 + transfer_matrix[0, 1]**2)
    bety = 1/tw0.bety[0] * ((transfer_matrix[2, 2] * tw0.bety[0] - transfer_matrix[2, 3] * tw0.alfy[0])**2 + transfer_matrix[2, 3]**2)
    alfx = -1/tw0.betx[0] * ((transfer_matrix[0, 0] * tw0.betx[0] - transfer_matrix[0, 1] * tw0.alfx[0]) *
                             (transfer_matrix[1, 0] * tw0.betx[0] - transfer_matrix[1, 1] * tw0.alfx[0]) + transfer_matrix[0, 1] * transfer_matrix[1, 1])
    alfy = -1/tw0.bety[0] * ((transfer_matrix[2, 2] * tw0.bety[0] - transfer_matrix[2, 3] * tw0.alfy[0]) *
                             (transfer_matrix[3, 2] * tw0.bety[0] - transfer_matrix[3, 3] * tw0.alfy[0]) + transfer_matrix[2, 3] * transfer_matrix[3, 3])
    mux = tw0.mux[0] + jnp.arctan2(transfer_matrix[0, 1], transfer_matrix[0, 0] * tw0.betx[0] - transfer_matrix[0, 1] * tw0.alfx[0]) / (2 * jnp.pi)
    muy = tw0.muy[0] + jnp.arctan2(transfer_matrix[2, 3], transfer_matrix[2, 2] * tw0.bety[0] - transfer_matrix[2, 3] * tw0.alfy[0]) / (2 * jnp.pi)
    dx = transfer_matrix[0,0] * tw0.dx[0] + transfer_matrix[0,1] * tw0.dpx[0] + transfer_matrix[0, 5]
    dy = transfer_matrix[2,2] * tw0.dy[0] + transfer_matrix[2,3] * tw0.dpy[0] + transfer_matrix[2, 5]
    dpx = transfer_matrix[1,0] * tw0.dx[0] + transfer_matrix[1,1] * tw0.dpx[0] + transfer_matrix[1, 5]
    dpy = transfer_matrix[3,2] * tw0.dy[0] + transfer_matrix[3,3] * tw0.dpy[0] + transfer_matrix[3, 5]

    param_dict = {
        'betx': betx,
        'bety': bety,
        'alfx': alfx,
        'alfy': alfy,
        'mux': mux,
        'muy': muy,
        'dx': dx,
        'dy': dy,
        'dpx': dpx,
        'dpy': dpy,
    }
    return param_dict

@jax.jit
def get_values_new_from_transfer_matrix(r_mat, param_values):
    # Order: betx, bety, alfx, alfy, mux, muy, dx, dy, dpx, dpy
    betx = 1/param_values[0] * ((r_mat[0, 0] * param_values[0] - r_mat[0, 1] * param_values[2])**2 + r_mat[0, 1]**2)
    bety = 1/param_values[1] * ((r_mat[2, 2] * param_values[1] - r_mat[2, 3] * param_values[3])**2 + r_mat[2, 3]**2)
    alfx = -1/param_values[0] * ((r_mat[0, 0] * param_values[0] - r_mat[0, 1] * param_values[2]) *
                             (r_mat[1, 0] * param_values[0] - r_mat[1, 1] * param_values[2]) + r_mat[0, 1] * r_mat[1, 1])
    alfy = -1/param_values[1] * ((r_mat[2, 2] * param_values[1] - r_mat[2, 3] * param_values[3]) *
                             (r_mat[3, 2] * param_values[1] - r_mat[3, 3] * param_values[3]) + r_mat[2, 3] * r_mat[3, 3])
    mux = param_values[4] + jnp.arctan2(r_mat[0, 1], r_mat[0, 0] * param_values[0] - r_mat[0, 1] * param_values[2]) / (2 * jnp.pi)
    muy = param_values[5] + jnp.arctan2(r_mat[2, 3], r_mat[2, 2] * param_values[1] - r_mat[2, 3] * param_values[3]) / (2 * jnp.pi)
    dx = r_mat[0,0] * param_values[6] + r_mat[0,1] * param_values[8] + r_mat[0, 5]
    dy = r_mat[2,2] * param_values[7] + r_mat[2,3] * param_values[9] + r_mat[2, 5]
    dpx = r_mat[1,0] * param_values[6] + r_mat[1,1] * param_values[8] + r_mat[1, 5]
    dpy = r_mat[3,2] * param_values[7] + r_mat[3,3] * param_values[9] + r_mat[3, 5]

    return jnp.array([betx, bety, alfx, alfy, mux, muy, dx, dy, dpx, dpy])

def derive_values_by_backtrack(elements, tw0):
    # Get the transfer matrix for each element
    transfer_matrices = []
    for elem in elements:
        if isinstance(elem, xt.Quadrupole):
            transfer_matrix = get_transfer_matrix_quad(elem.k1, elem.length, tw0.particle_on_co.beta0[0], tw0.particle_on_co.gamma0[0])
            transfer_matrices.append(transfer_matrix)
        elif isinstance(elem, xt.Bend) or isinstance(elem, xt.RBend):
            transfer_matrix = get_transfer_matrix_bend(elem.k0, elem.k1, elem.length, elem.h, tw0.particle_on_co.beta0[0], tw0.particle_on_co.gamma0[0])
            transfer_matrices.append(transfer_matrix)
        elif isinstance(elem, xt.Multipole):
            # ignore
            transfer_matrix = np.eye(6)
            transfer_matrices.append(transfer_matrix)
        elif isinstance(elem, xt.Drift) or hasattr(elem, 'length'):
            transfer_matrix = get_transfer_matrix_drift(elem.length, tw0.particle_on_co.beta0[0], tw0.particle_on_co.gamma0[0])
            transfer_matrices.append(transfer_matrix)
        else:
            transfer_matrix = np.eye(6)
            transfer_matrices.append(transfer_matrix)

    # Calculate the total transfer matrix
    total_transfer_matrix = jnp.eye(6)
    parameter_values = jnp.array([tw0.betx[0], tw0.bety[0], tw0.alfx[0], tw0.alfy[0], tw0.mux[0], tw0.muy[0], tw0.dx[0], tw0.dy[0], tw0.dpx[0], tw0.dpy[0]])

    for i, tm in enumerate(transfer_matrices):
        total_transfer_matrix = tm @ total_transfer_matrix
        parameter_values = get_values_new_from_transfer_matrix(tm, parameter_values)
    values = get_values_from_transfer_matrix(total_transfer_matrix, tw0)

    return values, total_transfer_matrix, transfer_matrices, parameter_values

def compute_param_derivatives(elements, elem_to_deriv, tw0):
    def get_values(k1_arr):
        transfer_matrices = []

        i = 0
        for elem in elements:
            if elem in elem_to_deriv:
                assert isinstance(elem, xt.Quadrupole)
                transfer_matrices.append(get_transfer_matrix_quad(k1_arr[i], elem.length, tw0.particle_on_co.beta0[0], tw0.particle_on_co.gamma0[0]))
                i += 1
            elif isinstance(elem, xt.Quadrupole) and elem.k1 != 0:
                transfer_matrices.append(get_transfer_matrix_quad(elem.k1, elem.length, tw0.particle_on_co.beta0[0], tw0.particle_on_co.gamma0[0]))
            elif isinstance(elem, xt.Bend) or isinstance(elem, xt.RBend):
                transfer_matrices.append(get_transfer_matrix_bend(elem.k0, elem.k1, elem.length, elem.h, tw0.particle_on_co.beta0[0], tw0.particle_on_co.gamma0[0]))
            elif isinstance(elem, xt.Multipole):
                transfer_matrices.append(jnp.eye(6))
            elif isinstance(elem, xt.Drift) or hasattr(elem, 'length'):
                transfer_matrices.append(get_transfer_matrix_drift(elem.length, tw0.particle_on_co.beta0[0], tw0.particle_on_co.gamma0[0]))
            else:
                transfer_matrices.append(jnp.eye(6))

        parameter_values = jnp.array([tw0.betx[0], tw0.bety[0], tw0.alfx[0], tw0.alfy[0], tw0.mux[0], tw0.muy[0], tw0.dx[0], tw0.dy[0], tw0.dpx[0], tw0.dpy[0]])

        for i, tm in enumerate(transfer_matrices):
            parameter_values = get_values_new_from_transfer_matrix(tm, parameter_values)
        return parameter_values

    k1_arr = jnp.array([elem.k1 for elem in elem_to_deriv])
    return jax.jacfwd(get_values)(k1_arr)


print("-----------------------------------------------------------")
print(f"Compare Twiss parameters and Backtracked parameters")

backtracked_values, transfer_matrix, transfer_matrices, parameter_values = derive_values_by_backtrack(trunc_elements, tw0)

print(tabulate([
    ['bety', tw0.bety[-1], backtracked_values['bety'], parameter_values[1]],
    ['betx', tw0.betx[-1], backtracked_values['betx'], parameter_values[0]],
    ['alfx', tw0.alfx[-1], backtracked_values['alfx'], parameter_values[2]],
    ['alfy', tw0.alfy[-1], backtracked_values['alfy'], parameter_values[3]],
    ['mux', tw0.mux[-1], backtracked_values['mux'], parameter_values[4]],
    ['muy', tw0.muy[-1], backtracked_values['muy'], parameter_values[5]],
    ['dx', tw0.dx[-1], backtracked_values['dx'], parameter_values[6]],
    ['dy', tw0.dy[-1], backtracked_values['dy'], parameter_values[7]],
    ['dpx', tw0.dpx[-1], backtracked_values['dpx'], parameter_values[8]],
    ['dpy', tw0.dpy[-1], backtracked_values['dpy'], parameter_values[9]],
], tablefmt="fancy_grid", headers=["Parameter", "Twiss", "Backtracked", "Backtracked Continuous"]))

print("-----------------------------------------------------------")

#print("Finite difference gradient betx: ", grads_fd['alfy'])

#print(f"Automatic gradient: {compute_param_derivatives(trunc_elements, tw0)['betx']}")

# deriv_sympy, symbols = compute_beta_derivative_sym(line, tw0)
# sympy_grad = []
# # Evaluate sympy expressions and print
# for i, elem, symbol in zip(range(len(quadrupoles)), quadrupoles, symbols):
#     sympy_grad.append(deriv_sympy[i].evalf(subs={symbol: quadrupole.k1 for symbol, quadrupole in zip(symbols, quadrupoles)}))
# print(f"Sympy gradient: {sympy_grad}")for elem, tab in zip(trunc_elements, tw0.rows):

def print_elements_diff(trunc_elements, tw0):
    tmp_sum = 0
    for i, elem, tab in zip(range(len(trunc_elements)), trunc_elements, tw0.rows):
        flag = hasattr(elem, "length")
        msg = f"Element of type {type(elem).__name__} at {round(tab.s, 4)}"
        if flag:
            tmp_sum += elem.length
            msg += f" with length {elem.length}, reaching {round(tmp_sum, 4)}."
            if elem != trunc_elements[-1]:
                msg += f" Difference: {(tmp_sum - tw0.rows[i+1].s).round(4)[0]}"
        print(msg)

def plot_betx_twiss_and_bt(transfer_matrices, tw0):
    import matplotlib.pyplot as plt

    bt_bety = []
    walking_mat = np.eye(6)
    param_values = jnp.array([tw0.betx[0], tw0.bety[0], tw0.alfx[0], tw0.alfy[0], tw0.mux[0], tw0.muy[0], tw0.dx[0], tw0.dy[0], tw0.dpx[0], tw0.dpy[0]])
    for i in transfer_matrices:
        walking_mat = i @ walking_mat
        param_values = get_values_new_from_transfer_matrix(i, param_values)
        bt_bety.append(param_values[0])
    bt_bety = np.array(bt_bety)

    plt.plot(tw0.s, tw0.betx, label='Twiss')
    plt.plot(tw0.s, bt_bety, label='Backtracked', linestyle='--')

    plt.xlabel('s [m]')
    plt.ylabel('Beta function [m]')
    plt.title('Beta function along the lattice')
    plt.legend()
    plt.grid()
    plt.show()

def get_closest_id_for_s(target):
    return np.abs(np.array(line.get_s_position()) - target).argmin()

def get_norm_diff_mat_for_elements(transfer_matrices, tw0):
    transfer_matrices = np.array(transfer_matrices)
    matrix_norm_diffs = np.zeros(len(transfer_matrices - 1))
    for i, tm in enumerate(transfer_matrices[1:]):
        matrix_twiss = tw0.get_R_matrix(tw0.name[i], tw0.name[i+1])
        matrix_norm_diffs[i] = np.linalg.norm(tm - matrix_twiss)
    return matrix_norm_diffs

mat_diffs = get_norm_diff_mat_for_elements(transfer_matrices, tw0)

import jacobian_mod as jmod

all_quad_sources, target_places, dkq_dvv = jmod.get_dependency_derivatives(opt)

start = 's.ds.l8.b1'
end = 'ip1'
opt_tw = line.twiss(start=start, end=end, init=tw_copy, init_at=xt.START)
tw_names = opt_tw.name[:-1]

# sort all_quad_sources based on occurence in opt_tw.name, which is a np.ndarray
all_quad_sources = sorted(all_quad_sources, key=lambda x: np.where(tw_names == x)[0][0])

# truncate line to analyze for start - end
twiss_derivs = {}
for place in target_places: # ip1, ip8
    # Calc derivative for all quadrupoles for target place
    # Source point = qqnn, Observation point = target
    twiss_derivs[place] = {}
    trunc_elements = np.array([line.element_dict[name] for name in opt_tw.rows[:place].name])
    nonzero_qq = []
    nonzero_qqn = []
    for qqnn in all_quad_sources:
        if opt_tw['s', place] < opt_tw['s', qqnn]:
            twiss_derivs[place][qqnn] = np.zeros(10)
        else:
            nonzero_qqn.append(qqnn)
            nonzero_qq.append(line.element_dict[qqnn])
            # add to list to be calculated
    nonzero_deriv = compute_param_derivatives(trunc_elements, nonzero_qq, opt_tw)
    for i, qqn in enumerate(nonzero_qqn):
        twiss_derivs[place][qqn] = nonzero_deriv[i]
    for qqn, deriv in zip(nonzero_qqn, nonzero_deriv.T):
        twiss_derivs[place][qqn] = deriv

# Convert list to dict
def convert_list_to_dict(lst):
    return {'betx': lst[0], 'bety': lst[1], 'alfx': lst[2], 'alfy': lst[3],
            'mux': lst[4], 'muy': lst[5], 'dx': lst[6], 'dy': lst[7], 'dpx': lst[8], 'dpy': lst[9]}

for i in twiss_derivs.keys():
    for j in twiss_derivs[i].keys():
        twiss_derivs[i][j] = convert_list_to_dict(twiss_derivs[i][j])

jac_estim = np.zeros((len(opt.targets), len(opt.vary)))
for itt, tt in enumerate(opt.targets):

    assert isinstance(tt.tar, tuple)

    tar_quantity = tt.tar[0]
    tar_place = tt.tar[1]
    tar_weight = tt.weight

    for ivv in range(len(opt.vary)):

        vv = opt.vary[ivv].name

        quad_names = dkq_dvv[vv].keys()

        dtar_dvv = 0
        for qqnn in quad_names:
            dtar_dvv += twiss_derivs[tar_place][qqnn][tar_quantity] * float(dkq_dvv[vv][qqnn])

        dtar_dvv *= tar_weight

        jac_estim[itt, ivv] = dtar_dvv


err = opt.get_merit_function()
jac = err.get_jacobian(err.get_x(), opt)


# analyze ip1_bety: 0,15 -> 0,1 impact from first variable: kq6.l8b1 (0) to jacobian at place (0,7)
# two quadrupoles: mqm.6l8.b1 and mqml.6l8.b1
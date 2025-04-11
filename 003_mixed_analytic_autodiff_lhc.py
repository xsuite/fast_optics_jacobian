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
limit = 14520
end_point = tw_copy.rows[limit].name[0]

for i in range(limit):
    if isinstance(line.elements[i], xt.Bend):
        #line.elements[i].k0 = 0
        #line.elements[i].h = 0
        line.elements[i].edge_entry_active=0
        line.elements[i].edge_exit_active=0

#jax.config.update("jax_enable_x64", True)

tw0 = line.twiss4d(start=start_point, end=end_point, betx=0.15, bety=0.15)
trunc_elements = np.array(line.elements)[np.logical_and(line.get_s_position() >= np.float64(line.get_s_position(start_point)), line.get_s_position() <= np.float64(line.get_s_position(end_point)))]

quadrupoles = [elem for elem in trunc_elements if isinstance(elem, xt.Quadrupole)]
# eps = 1e-6
# grads_fd = []
# for quad in quadrupoles:
#     quad.k1 += eps
#     tw_plus = line.twiss4d(init=tw0, start=xt.START, end=xt.END)
#     quad.k1 -= 2 * eps
#     tw_minus = line.twiss4d(init=tw0, start=xt.START, end=xt.END)
#     quad.k1 += eps

#     fd_dict = {
#     'betx': (tw_plus.betx[-1] - tw_minus.betx[-1]) / (2 * eps),
#     'bety': (tw_plus.bety[-1] - tw_minus.bety[-1]) / (2 * eps),
#     'alfx': (tw_plus.alfx[-1] - tw_minus.alfx[-1]) / (2 * eps),
#     'alfy': (tw_plus.alfy[-1] - tw_minus.alfy[-1]) / (2 * eps),
#     'mux': (tw_plus.mux[-1] - tw_minus.mux[-1]) / (2 * eps),
#     'muy': (tw_plus.muy[-1] - tw_minus.muy[-1]) / (2 * eps),
#     'dx': (tw_plus.dx[-1] - tw_minus.dx[-1]) / (2 * eps),
#     'dpx': (tw_plus.dpx[-1] - tw_minus.dpx[-1]) / (2 * eps),
#     'dy': (tw_plus.dy[-1] - tw_minus.dy[-1]) / (2 * eps),
#     'dpy': (tw_plus.dpy[-1] - tw_minus.dpy[-1]) / (2 * eps),
#     }
#     grads_fd.append(fd_dict)

# tw0 = line.twiss4d(start=xt.START, end=xt.END, betx=0.15, bety=0.15)

# # Reorder list of dictionaries to have Dictionary of lists
# grads_fd = {key: [d[key] for d in grads_fd] for key in grads_fd[0].keys()}

def get_transfer_matrix_quad(k1, l, beta0, gamma0):
    kx = jnp.sqrt(k1.astype(complex))
    ky = jnp.sqrt(-k1.astype(complex))
    sx = jnp.sin(kx * l) / kx
    cx = jnp.cos(kx * l)
    sy = jnp.sin(ky * l) / ky # limit of sin(ky * l) / ky when ky -> 0
    cy = jnp.cos(ky * l)
    if k1 == 0:
        return get_transfer_matrix_drift(l, beta0, gamma0)

    f_matrix = jnp.array([
        [cx, sx, 0, 0, 0, 0],
        [-kx**2 * sx, cx, 0, 0, 0, 0],
        [0, 0, cy, sy, 0, 0],
        [0, 0, -ky**2 * sy, cy, 0, 0],
        [0, 0, 0, 0, 1, l/(beta0**2 * gamma0**2)],
        [0, 0, 0, 0, 0, 1]
    ])

    return f_matrix.real

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

def get_transfer_matrix_bend(k0, k1, l, h, beta0, gamma0):
    kx = jnp.sqrt((h * k0 + k1).astype(complex))
    ky = jnp.sqrt(-k1.astype(complex)) # for dipoles usually 0
    sx = jnp.sin(kx * l) / kx
    cx = jnp.cos(kx * l)
    sy = jnp.sin(ky * l) / ky
    cy = jnp.cos(ky * l)
    dx = h * ((1 - cx) / kx**2)
    j1 = (l - sx) / kx**2

    if k1 == 0:
        sy = l # sin(ky * l) / ky converges against l for ky -> 0
        cy = 1.0
    if k0 == 0 or h == 0:
        return get_transfer_matrix_drift(l, beta0, gamma0)

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
    dx = transfer_matrix[0,0] * tw0.dx[0] + transfer_matrix[0,1] * tw0.dpx[0]
    dy = transfer_matrix[2,2] * tw0.dy[0] + transfer_matrix[2,3] * tw0.dpy[0]
    dpx = transfer_matrix[1,0] * tw0.dx[0] + transfer_matrix[1,1] * tw0.dpx[0]
    dpy = transfer_matrix[3,2] * tw0.dy[0] + transfer_matrix[3,3] * tw0.dpy[0]

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

def derive_values_by_backtrack(elements, tw0):
    # Get the transfer matrix for each element
    transfer_matrices = []
    for elem in elements:
        if isinstance(elem, xt.Quadrupole):
            transfer_matrix = get_transfer_matrix_quad(elem.k1, elem.length, tw0.particle_on_co.beta0[0], tw0.particle_on_co.gamma0[0])
            if np.any(np.isnan(transfer_matrix)):
                print("Turned to NaN in transfer matrix!!")
                print(f"Element: {elem}")
                quit()
            transfer_matrices.append(transfer_matrix)
        elif isinstance(elem, xt.Bend):
            transfer_matrix = get_transfer_matrix_bend(elem.k0, elem.k1, elem.length, elem.h, tw0.particle_on_co.beta0[0], tw0.particle_on_co.gamma0[0])
            if np.any(np.isnan(transfer_matrix)):
                print("Turned to NaN in transfer matrix!!")
                print(f"Element: {elem}")
                quit()
            transfer_matrices.append(transfer_matrix)
        elif isinstance(elem, xt.Multipole):
            # ignore
            transfer_matrix = np.eye(6)
            transfer_matrices.append(transfer_matrix)
        elif isinstance(elem, xt.Drift) or hasattr(elem, 'length'):
            transfer_matrix = get_transfer_matrix_drift(elem.length, tw0.particle_on_co.beta0[0], tw0.particle_on_co.gamma0[0])
            if np.any(np.isnan(transfer_matrix)):
                print("Turned to NaN in transfer matrix!!")
                print(f"Element: {elem}")
                quit()
            transfer_matrices.append(transfer_matrix)
        else:
            transfer_matrix = np.eye(6)
            transfer_matrices.append(transfer_matrix)

    # Calculate the total transfer matrix
    total_transfer_matrix = np.eye(6)

    for i, tm in enumerate(reversed(transfer_matrices)):
        total_transfer_matrix = total_transfer_matrix @ tm
    values = get_values_from_transfer_matrix(total_transfer_matrix, tw0)

    # matrix_copy = transfer_matrices.copy()
    # def balanced_matrix_prod(mats):
    #     n = len(mats)
    #     if n == 1:
    #         return mats[0]
    #     mid = n // 2
    #     return balanced_matrix_prod(mats[mid:]) @ balanced_matrix_prod(mats[:mid])

    # second_matrix = balanced_matrix_prod(matrix_copy)

    #values = get_values_from_transfer_matrix(second_matrix, tw0)

    return values, total_transfer_matrix, transfer_matrices

def compute_param_derivatives(elements, tw0):
    def get_values(k1_arr):
        transfer_matrices = []
        # assert that you have the same number of quadrupoles as k1_arr
        assert len(k1_arr) == len([elem for elem in elements if isinstance(elem, xt.Quadrupole)])

        i = 0
        for elem in line.elements:
            if isinstance(elem, xt.Quadrupole):
                transfer_matrices.append(get_transfer_matrix_quad(k1_arr[i], elem.length, tw0.particle_on_co.beta0[0], tw0.particle_on_co.gamma0[0]))
                i += 1
            elif isinstance(elem, xt.Bend):
                transfer_matrices.append(get_transfer_matrix_bend(elem.k0, elem.k1, elem.length, elem.h, tw0.particle_on_co.beta0[0], tw0.particle_on_co.gamma0[0]))
            elif isinstance(elem, xt.Drift) or hasattr(elem, 'length'):
                transfer_matrices.append(get_transfer_matrix_drift(elem.length, tw0.particle_on_co.beta0[0], tw0.particle_on_co.gamma0[0]))

        total_transfer_matrix = jnp.eye(6)
        for tm in reversed(transfer_matrices):
            total_transfer_matrix = total_transfer_matrix @ tm
        values = get_values_from_transfer_matrix(total_transfer_matrix, tw0)
        return values

    return jax.jacfwd(get_values)(jnp.array([elem.k1 for elem in elements if isinstance(elem, xt.Quadrupole)]))


print("-----------------------------------------------------------")
print(f"Compare Twiss parameters and Backtracked parameters")

backtracked_values, transfer_matrix, transfer_matrices = derive_values_by_backtrack(trunc_elements, tw0)

print(tabulate([
    ['betx', tw0.betx[-1], backtracked_values['betx']],
    ['bety', tw0.bety[-1], backtracked_values['bety']],
    ['alfx', tw0.alfx[-1], backtracked_values['alfx']],
    ['alfy', tw0.alfy[-1], backtracked_values['alfy']],
    ['mux', tw0.mux[-1], backtracked_values['mux']],
    ['muy', tw0.muy[-1], backtracked_values['muy']],
    ['dx', tw0.dx[-1], backtracked_values['dx']],
    ['dy', tw0.dy[-1], backtracked_values['dy']],
    ['dpx', tw0.dpx[-1], backtracked_values['dpx']],
    ['dpy', tw0.dpy[-1], backtracked_values['dpy']],
], tablefmt="fancy_grid", headers=["Parameter", "Twiss", "Backtracked"]))

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
    for i in transfer_matrices:
        walking_mat = walking_mat @ i
        bt_bety.append(get_values_from_transfer_matrix(walking_mat, tw0)['bety'])
    bt_bety = np.array(bt_bety)

    plt.plot(tw0.s, tw0.bety, label='Twiss')
    plt.plot(tw0.s, bt_bety, label='Backtracked', linestyle='--')

    plt.xlabel('s [m]')
    plt.ylabel('Beta function [m]')
    plt.title('Beta function along the lattice')
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure()

    plt.plot(tw0.s, tw0.bety / bt_bety - 1)

    plt.grid()
    plt.show()

plot_betx_twiss_and_bt(transfer_matrices, tw0)

#index_pos_elem_tuple = [(elem, s) for elem, s in zip(line.elements, line.get_s_position())]

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
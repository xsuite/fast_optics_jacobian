import xtrack as xt
import numpy as np
import jax.numpy as jnp
import jax
from tabulate import tabulate

env = xt.Environment()
env.particle_ref = xt.Particles(p0c=7e12)

env['kq'] = 0.1
env['bphi'] = 0.01

env.new('qf', 'Quadrupole', k1='kq', length=1.0, anchor='start')
env.new('qd', 'Quadrupole', k1='-kq', length=1.0, anchor='start')
env.new('drift', 'Drift', length=2.0)
env.new('drift2', 'Drift', length=1.5)
env.new('end', 'Marker', at=10., from_='qd@end')
env.new('bendh', 'Bend', angle='bphi', k0_from_h=True, length=1.0),
env.new('bendv', 'Bend', angle='bphi', rot_s_rad=np.pi/2, k0_from_h=True, length=1.0),
env.new('qq', 'Quadrupole', k1=0., length=1.0, anchor='start')

line = env.new_line(components=[
    env.place('qf', anchor='start', at=0.),
    env.place('drift', at=1., from_='qf@end'),
    env.place('bendh', anchor='start', at=5., from_='drift@end'),
    #env.place('bendv', anchor='start', at=18., from_='bendh@end'),
    env.place('qd', anchor='start', at=10., from_='qf@end'),
    env.place('drift2', anchor='start', at=11., from_='qd@end'),
    env.place('qq', anchor='start', at=12., from_='drift2@end'),
    env.place('end', at=10., from_='drift@end'),
])

# Bends -> bend
# Quadrupole -> Quad
# Drift, Sextupole, Octupole -> Drift

opt = line.match(
    method='4d',
    solve=False,
    vary=xt.Vary('kq', step=1e-4),
    targets=xt.Target('qx', 0.166666, tol=1e-6),
)
opt.solve()

tw0 = line.twiss4d(start=xt.START, end=xt.END, betx=0.15, bety=0.15)

quadrupoles = [elem for elem in line.elements if isinstance(elem, xt.Quadrupole)]
eps = 1e-6
grads_fd = []
for quad in quadrupoles:
    quad.k1 += eps
    tw_plus = line.twiss4d(init=tw0, start=xt.START, end=xt.END)
    quad.k1 -= 2 * eps
    tw_minus = line.twiss4d(init=tw0, start=xt.START, end=xt.END)
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

tw0 = line.twiss4d(start=xt.START, end=xt.END, betx=0.15, bety=0.15)

# Reorder list of dictionaries to have Dictionary of lists
grads_fd = {key: [d[key] for d in grads_fd] for key in grads_fd[0].keys()}

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
    dx = (1 - cx) / kx**2
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
            if np.any(np.isnan(transfer_matrix)):
                print("Turned to NaN in transfer matrix!!")
                print(f"Element: {elem}")
                quit()
            transfer_matrices.append(transfer_matrix)
        elif isinstance(elem, xt.Bend) or isinstance(elem, xt.RBend):
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
    total_transfer_matrix = jnp.eye(6)
    parameter_matrix = []
    parameter_values = jnp.array([tw0.betx[0], tw0.bety[0], tw0.alfx[0], tw0.alfy[0], tw0.mux[0], tw0.muy[0], tw0.dx[0], tw0.dy[0], tw0.dpx[0], tw0.dpy[0]])

    for i, tm in enumerate(transfer_matrices):
        total_transfer_matrix = tm @ total_transfer_matrix
        parameter_values = get_values_new_from_transfer_matrix(tm, parameter_values)
        parameter_matrix.append(parameter_values)
    values = get_values_from_transfer_matrix(total_transfer_matrix, tw0)

    return values, total_transfer_matrix, transfer_matrices, parameter_values, jnp.array(parameter_matrix)

def compute_param_derivatives(line, tw0):
    def get_values(k1_arr):
        transfer_matrices = []
        # assert that you have the same number of quadrupoles as k1_arr
        assert len(k1_arr) == len([elem for elem in line.elements if isinstance(elem, xt.Quadrupole)])

        i = 0
        for elem in line.elements:
            if isinstance(elem, xt.Quadrupole):
                transfer_matrices.append(get_transfer_matrix_quad(k1_arr[i], elem.length, tw0.particle_on_co.beta0[0], tw0.particle_on_co.gamma0[0]))
                i += 1
            elif isinstance(elem, xt.Bend):
                transfer_matrices.append(get_transfer_matrix_bend(elem.k0, elem.k1, elem.length, elem.h, tw0.particle_on_co.beta0[0], tw0.particle_on_co.gamma0[0]))
            elif isinstance(elem, xt.Drift):
                transfer_matrices.append(get_transfer_matrix_drift(elem.length, tw0.particle_on_co.beta0[0], tw0.particle_on_co.gamma0[0]))

        parameter_values = jnp.array([tw0.betx[0], tw0.bety[0], tw0.alfx[0], tw0.alfy[0], tw0.mux[0], tw0.muy[0], tw0.dx[0], tw0.dy[0], tw0.dpx[0], tw0.dpy[0]])

        total_transfer_matrix = jnp.eye(6)
        for i, tm in enumerate(transfer_matrices):
            total_transfer_matrix = tm @ total_transfer_matrix
            parameter_values = get_values_new_from_transfer_matrix(tm, parameter_values)
        return parameter_values

    return jax.jacrev(get_values)(jnp.array([elem.k1 for elem in line.elements if isinstance(elem, xt.Quadrupole)]))


import sympy as sp

def get_values_from_transfer_matrix_sp(transfer_matrix, tw0):
    betx = 1/tw0.betx[0] * ((transfer_matrix[0, 0] * tw0.betx[0] - transfer_matrix[0, 1] * tw0.alfx[0])**2 + transfer_matrix[0, 1]**2)
    bety = 1/tw0.bety[0] * ((transfer_matrix[2, 2] * tw0.bety[0] - transfer_matrix[2, 3] * tw0.alfy[0])**2 + transfer_matrix[2, 3]**2)
    alfx = -1/tw0.betx[0] * ((transfer_matrix[0, 0] * tw0.betx[0] - transfer_matrix[0, 1] * tw0.alfx[0]) *
                             (transfer_matrix[1, 0] * tw0.betx[0] - transfer_matrix[1, 1] * tw0.alfx[0]) + transfer_matrix[0, 1] * transfer_matrix[1, 1])
    alfy = -1/tw0.bety[0] * ((transfer_matrix[2, 2] * tw0.bety[0] - transfer_matrix[2, 3] * tw0.alfy[0]) *
                             (transfer_matrix[3, 2] * tw0.bety[0] - transfer_matrix[3, 3] * tw0.alfy[0]) + transfer_matrix[2, 3] * transfer_matrix[3, 3])
    mux = tw0.mux[0] + sp.atan(transfer_matrix[0, 1] / (transfer_matrix[0, 0] * tw0.betx[0] - transfer_matrix[0, 1] * tw0.alfx[0])) / (2 * sp.pi)
    muy = tw0.muy[0] + sp.atan(transfer_matrix[2, 3] / (transfer_matrix[2, 2] * tw0.bety[0] - transfer_matrix[2, 3] * tw0.alfy[0])) / (2 * sp.pi)
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

def get_transfer_matrix_quad_sym(k1, l, beta0, gamma0):
    kx = sp.sqrt(k1)
    ky = sp.sqrt(-k1)
    sx = sp.sin(kx * l) / kx
    cx = sp.cos(kx * l)
    sy = sp.sin(ky * l) / ky
    cy = sp.cos(ky * l)

    f_matrix = sp.Matrix([
        [cx, sx, 0, 0, 0, 0],
        [-k1 * sx, cx, 0, 0, 0, 0],
        [0, 0, cy, sy, 0, 0],
        [0, 0, k1 * sy, cy, 0, 0],
        [0, 0, 0, 0, 1, l/(beta0**2 * gamma0**2)],
        [0, 0, 0, 0, 0, 1]
    ])

    return f_matrix

def get_transfer_matrix_drift_sym(l, beta0, gamma0):
    return sp.Matrix([
        [1, l, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, l, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, l/(beta0**2 * gamma0**2)],
        [0, 0, 0, 0, 0, 1]
    ])

def get_transfer_matrix_bend_sym(k0, k1, l, h, beta0, gamma0):
    kx = sp.sqrt(h * k0 + k1)
    ky = sp.sqrt(-k1)  # for dipoles usually 0
    sx = sp.sin(kx * l) / kx
    cx = sp.cos(kx * l)
    sy = 1.0  # limit of sin(ky * l) / ky when ky -> 0
    cy = 1.0  # limit of cos(ky * l) when ky -> 0
    dx = h * ((1 - cx) / kx**2)
    j1 = (l - sx) / kx**2

    f_matrix = sp.Matrix([
        [cx, sx, 0, 0, 0, h/beta0 * dx],
        [-kx**2 * sx, cx, 0, 0, 0, h/beta0 * sx],
        [0, 0, cy, sy, 0, 0],
        [0, 0, -ky**2 * sy, cy, 0, 0],
        [-h/beta0 * sx, -h/beta0 * dx, 0, 0, 1, l/(beta0**2 * gamma0**2) - h**2/beta0**2 * j1],
        [0, 0, 0, 0, 0, 1]
    ])

    return f_matrix

def compute_beta_derivative_sym(line, tw0):
    quadrupoles = [line.elements[i] for i in range(len(line.elements)) if isinstance(line.elements[i], xt.Quadrupole)]
    k1_vars = [sp.Symbol(f'k1_{i}') for i, _ in enumerate(quadrupoles)]

    transfer_matrices = []
    k1_index = 0
    for elem in line.elements:
        if isinstance(elem, xt.Quadrupole):
            transfer_matrices.append(get_transfer_matrix_quad_sym(k1_vars[k1_index], elem.length, tw0.particle_on_co.beta0[0], tw0.particle_on_co.gamma0[0]))
            k1_index += 1
        elif isinstance(elem, xt.Bend):
            transfer_matrices.append(get_transfer_matrix_bend_sym(elem.k0, elem.k1, elem.length, elem.h, tw0.particle_on_co.beta0[0], tw0.particle_on_co.gamma0[0]))
        elif isinstance(elem, xt.Drift):
            transfer_matrices.append(get_transfer_matrix_drift_sym(elem.length, tw0.particle_on_co.beta0[0], tw0.particle_on_co.gamma0[0]))

    total_transfer_matrix = sp.eye(6)
    for tm in reversed(transfer_matrices):
        total_transfer_matrix = total_transfer_matrix @ tm

    betx_sym = get_values_from_transfer_matrix_sp(total_transfer_matrix, tw0)['alfy']

    beta_derivatives = [sp.diff(betx_sym, k1) for k1 in k1_vars]

    return beta_derivatives, k1_vars

print("-----------------------------------------------------------")
print(f"Compare Twiss parameters and Backtracked parameters")

backtracked_values, transfer_matrix, transfer_matrices, parameter_values, parameter_matrix = derive_values_by_backtrack(line, tw0)

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
print("Finite difference gradient betx: ", grads_fd['alfy'])

print(f"Automatic betx gradient: {compute_param_derivatives(line, tw0)}")

# deriv_sympy, symbols = compute_beta_derivative_sym(line, tw0)
# sympy_grad = []
# # Evaluate sympy expressions and print
# for i, elem, symbol in zip(range(len(quadrupoles)), quadrupoles, symbols):
#     sympy_grad.append(deriv_sympy[i].evalf(subs={symbol: quadrupole.k1 for symbol, quadrupole in zip(symbols, quadrupoles)}))
# print(f"Sympy gradient: {sympy_grad}")

def plot_betx_twiss_and_bt(transfer_matrices, tw0):
    import matplotlib.pyplot as plt

    bt_bety = []
    walking_mat = np.eye(6)
    param_values = jnp.array([tw0.betx[0], tw0.bety[0], tw0.alfx[0], tw0.alfy[0], tw0.mux[0], tw0.muy[0], tw0.dx[0], tw0.dy[0], tw0.dpx[0], tw0.dpy[0]])
    idx = 6
    bt_bety.append(param_values[idx])
    for i in transfer_matrices:
        walking_mat = i @ walking_mat
        param_values = get_values_new_from_transfer_matrix(i, param_values)
        bt_bety.append(param_values[idx])
    bt_bety = np.array(bt_bety)

    print(bt_bety)

    plt.plot(tw0.s, tw0.dx, label='Twiss')
    plt.plot(tw0.s, bt_bety, label='Backtracked', linestyle='--')

    plt.xlabel('s [m]')
    plt.ylabel('Beta function [m]')
    plt.title('Beta function along the lattice')
    plt.legend()
    plt.grid()
    plt.show()
    return bt_bety


def get_norm_diff_mat_for_elements(transfer_matrices, tw0):
    transfer_matrices = np.array(transfer_matrices)
    matrix_norm_diffs = np.zeros(len(transfer_matrices - 1))
    index_05_diff = np.zeros(len(transfer_matrices - 1))
    for i, tm in enumerate(transfer_matrices):
        matrix_twiss = tw0.get_R_matrix(tw0.name[i], tw0.name[i+1])
        matrix_norm_diffs[i] = np.linalg.norm(tm - matrix_twiss)
        index_05_diff[i] = tm[0, 5] - matrix_twiss[0, 5]
    return matrix_norm_diffs, index_05_diff
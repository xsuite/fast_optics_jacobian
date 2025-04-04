import xtrack as xt
import numpy as np
import jax.numpy as jnp
import jax
from tabulate import tabulate

env = xt.Environment()
env.particle_ref = xt.Particles(p0c=7e12)

env['kq'] = 0.1

env.new('qf', 'Quadrupole', k1='kq', length=1.0, anchor='start')
env.new('qd', 'Quadrupole', k1='-kq', length=1.0, anchor='start')
env.new('drift', 'Drift', length=2.0)
env.new('drift2', 'Drift', length=1.5)
env.new('end', 'Marker', at=10., from_='qd@end')

line = env.new_line(components=[
    env.place('qf', anchor='start', at=0.),
    env.place('drift', at=1., from_='qf@end'),
    env.place('qd', anchor='start', at=10., from_='qf@end'),
    env.place('drift2', anchor='start', at=11., from_='qd@end'),
    env.place('end', at=10., from_='drift@end'),
])

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

def get_transfer_matrix_quad(k1, l):
    kx = jnp.sqrt(k1.astype(complex))
    ky = jnp.sqrt(-k1.astype(complex))
    sx = jnp.sin(kx * l) / kx
    cx = jnp.cos(kx * l)
    sy = jnp.sin(ky * l) / ky
    cy = jnp.cos(ky * l)

    f_matrix = jnp.array([
        [cx, sx, 0, 0],
        [-k1 * sx, cx, 0, 0],
        [0, 0, cy, sy],
        [0, 0, k1 * sy, cy]
    ])

    return f_matrix.real

def get_transfer_matrix_drift(l):
    f_matrix = jnp.array([
        [1, l, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, l],
        [0, 0, 0, 1]
    ])
    return f_matrix

def get_values_from_transfer_matrix(transfer_matrix, tw0):
    betx = 1/tw0.betx[0] * ((transfer_matrix[0, 0] * tw0.betx[0] - transfer_matrix[0, 1] * tw0.alfx[0])**2 + transfer_matrix[0, 1]**2)
    bety = 1/tw0.bety[0] * ((transfer_matrix[2, 2] * tw0.bety[0] - transfer_matrix[2, 3] * tw0.alfy[0])**2 + transfer_matrix[2, 3]**2)
    alfx = -1/tw0.betx[0] * ((transfer_matrix[0, 0] * tw0.betx[0] - transfer_matrix[0, 1] * tw0.alfx[0]) *
                             (transfer_matrix[1, 0] * tw0.betx[0] - transfer_matrix[1, 1] * tw0.alfx[0]) + transfer_matrix[0, 1] * transfer_matrix[1, 1])
    alfy = -1/tw0.bety[0] * ((transfer_matrix[2, 2] * tw0.bety[0] - transfer_matrix[2, 3] * tw0.alfy[0]) *
                             (transfer_matrix[3, 2] * tw0.bety[0] - transfer_matrix[3, 3] * tw0.alfy[0]) + transfer_matrix[2, 3] * transfer_matrix[3, 3])
    mux = tw0.mux[0] + jnp.arctan2(transfer_matrix[0, 1], transfer_matrix[0, 0] * tw0.betx[0] - transfer_matrix[0, 1] * tw0.alfx[0]) / (2 * jnp.pi)
    muy = tw0.muy[0] + jnp.arctan2(transfer_matrix[2, 3], transfer_matrix[2, 2] * tw0.bety[0] - transfer_matrix[2, 3] * tw0.alfy[0]) / (2 * jnp.pi)

    param_dict = {
        'betx': betx,
        'bety': bety,
        'alfx': alfx,
        'alfy': alfy,
        'mux': mux,
        'muy': muy,
    }
    return param_dict

def derive_values_by_backtrack(line, tw0):
    # Get the transfer matrix for each element
    transfer_matrices = []
    for elem in line.elements:
        if isinstance(elem, xt.Quadrupole):
            transfer_matrices.append(get_transfer_matrix_quad(elem.k1, elem.length))
        elif isinstance(elem, xt.Drift):
            transfer_matrices.append(get_transfer_matrix_drift(elem.length))

    # Calculate the total transfer matrix
    total_transfer_matrix = np.eye(4)
    for tm in reversed(transfer_matrices):
        total_transfer_matrix = total_transfer_matrix @ tm
    print(total_transfer_matrix)
    values = get_values_from_transfer_matrix(total_transfer_matrix, tw0)

    return values

def compute_param_derivatives(line, tw0):
    def get_values(k1_arr):
        transfer_matrices = []
        # assert that you have the same number of quadrupoles as k1_arr
        assert len(k1_arr) == len([elem for elem in line.elements if isinstance(elem, xt.Quadrupole)])

        i = 0

        for elem in line.elements:
            if isinstance(elem, xt.Quadrupole):
                transfer_matrices.append(get_transfer_matrix_quad(k1_arr[i], elem.length))
                i += 1
            elif isinstance(elem, xt.Drift):
                transfer_matrices.append(get_transfer_matrix_drift(elem.length))

        total_transfer_matrix = jnp.eye(4)
        for tm in reversed(transfer_matrices):
            total_transfer_matrix = total_transfer_matrix @ tm
        values = get_values_from_transfer_matrix(total_transfer_matrix, tw0)
        return values

    return jax.jacfwd(get_values)(jnp.array([elem.k1 for elem in line.elements if isinstance(elem, xt.Quadrupole)]))


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

    param_dict = {
        'betx': betx,
        'bety': bety,
        'alfx': alfx,
        'alfy': alfy,
        'mux': mux,
        'muy': muy,
    }
    return param_dict

def get_transfer_matrix_quad_sym(k1, l):
    kx = sp.sqrt(k1)
    ky = sp.sqrt(-k1)
    sx = sp.sin(kx * l) / kx
    cx = sp.cos(kx * l)
    sy = sp.sin(ky * l) / ky
    cy = sp.cos(ky * l)

    f_matrix = sp.Matrix([
        [cx, sx, 0, 0],
        [-k1 * sx, cx, 0, 0],
        [0, 0, cy, sy],
        [0, 0, k1 * sy, cy]
    ])

    return f_matrix

def get_transfer_matrix_drift_sym(l):
    return sp.Matrix([
        [1, l, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, l],
        [0, 0, 0, 1]
    ])

def compute_beta_derivative_sym(line, tw0):
    quadrupoles = [line.elements[i] for i in range(len(line.elements)) if isinstance(line.elements[i], xt.Quadrupole)]
    k1_vars = [sp.Symbol(f'k1_{i}') for i, _ in enumerate(quadrupoles)]

    transfer_matrices = []
    k1_index = 0
    for elem in line.elements:
        if isinstance(elem, xt.Quadrupole):
            transfer_matrices.append(get_transfer_matrix_quad_sym(k1_vars[k1_index], elem.length))
            k1_index += 1
        elif isinstance(elem, xt.Drift):
            transfer_matrices.append(get_transfer_matrix_drift_sym(elem.length))

    total_transfer_matrix = sp.eye(4)
    for tm in reversed(transfer_matrices):
        total_transfer_matrix = total_transfer_matrix @ tm

    betx_sym = get_values_from_transfer_matrix_sp(total_transfer_matrix, tw0)['alfy']

    beta_derivatives = [sp.diff(betx_sym, k1) for k1 in k1_vars]

    return beta_derivatives, k1_vars

print("-----------------------------------------------------------")
print(f"Compare Twiss parameters and Backtracked parameters")

backtracked_values = derive_values_by_backtrack(line, tw0)

print(tabulate([
    ['betx', tw0.betx[-1], backtracked_values['betx']],
    ['bety', tw0.bety[-1], backtracked_values['bety']],
    ['alfx', tw0.alfx[-1], backtracked_values['alfx']],
    ['alfy', tw0.alfy[-1], backtracked_values['alfy']],
    ['mux', tw0.mux[-1], backtracked_values['mux']],
    ['muy', tw0.muy[-1], backtracked_values['muy']],
], tablefmt="fancy_grid", headers=["Parameter", "Twiss", "Backtracked"]))

print("-----------------------------------------------------------")
print("Finite difference gradient betx: ", grads_fd['alfy'])

print(f"Automatic betx gradient: {compute_param_derivatives(line, tw0)['alfy']}")

deriv_sympy, symbols = compute_beta_derivative_sym(line, tw0)
sympy_grad = []
# Evaluate sympy expressions and print
for i, elem, symbol in zip(range(len(quadrupoles)), quadrupoles, symbols):
    sympy_grad.append(deriv_sympy[i].evalf(subs={symbol: quadrupole.k1 for symbol, quadrupole in zip(symbols, quadrupoles)}))
print(f"Sympy gradient: {sympy_grad}")
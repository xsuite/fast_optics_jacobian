import xtrack as xt
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
import twiss_deriv

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

eps = 1e-6

grad = []

quadrupoles = [elem for elem in line.elements if isinstance(elem, xt.Quadrupole)]
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
    grad.append(fd_dict)

# Reorder list of dictionaries to have Dictionary of lists
fd_dict = {key: [d[key] for d in grad] for key in grad[0].keys()}

elements = line.elements

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
        [0, 0, cy, sy / ky],
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

def get_betx_from_transfer_matrix(transfer_matrix, betx0, alfx0):
    betx = 1/betx0 * ((transfer_matrix[0, 0] * betx0 - transfer_matrix[0, 1] * alfx0)**2 + transfer_matrix[0, 1]**2)
    return betx

def get_bety_from_transfer_matrix(transfer_matrix, bety0, alfy0):
    bety = 1/bety0 * ((transfer_matrix[2, 2] * bety0 - transfer_matrix[2, 3] * alfy0)**2 + transfer_matrix[2, 3]**2)
    return bety

def derive_beta_by_backtrack(line, tw0):
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
        total_transfer_matrix = tm @ total_transfer_matrix

    # Calculate the beta function at the end of the line

    betx_end = get_betx_from_transfer_matrix(total_transfer_matrix, tw0.betx[0], tw0.alfx[0])

    return betx_end

def compute_betx_derivative(line, tw0):
    def get_betx(k1_arr):
        transfer_matrices = []
        # assert that you have the same number of quadrupoles as k1_arr
        assert len(k1_arr) == len([elem for elem in line.elements if isinstance(elem, xt.Quadrupole)])

        i = 0

        for elem in line.elements:
            if isinstance(elem, xt.Quadrupole):
                transfer_matrices.append(get_transfer_matrix_quad(k1_arr[i], elem.length))
                i += 1
            elif isinstance(elem, xt.Multipole):
                transfer_matrices.append(get_transfer_matrix_quad(k1_arr[i], 1.0))
                i += 1
            elif isinstance(elem, xt.Drift):
                transfer_matrices.append(get_transfer_matrix_drift(elem.length))

        total_transfer_matrix = jnp.eye(4)
        for tm in reversed(transfer_matrices):
            total_transfer_matrix = tm @ total_transfer_matrix
        betx_end = get_betx_from_transfer_matrix(total_transfer_matrix, tw0.betx[0], tw0.alfx[0])

        return betx_end

    return jax.jacfwd(get_betx)(jnp.array([elem.k1 for elem in line.elements if isinstance(elem, xt.Quadrupole)]))


import sympy as sp

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
        [0, 0, cy, sy / ky],
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

def get_betx_from_transfer_matrix_sym(transfer_matrix, betx0, alfx0):
    return 1 / betx0 * ((transfer_matrix[0, 0] * betx0 - transfer_matrix[0, 1] * alfx0)**2 + transfer_matrix[0, 1]**2)

def compute_beta_derivative_sym(line, tw0):
    k1_vars = [sp.Symbol(f'k1_{i}') for i, elem in enumerate(line.elements) if isinstance(elem, xt.Quadrupole)]
    print(f"Symbols for k1: {k1_vars}")

    transfer_matrices = []
    k1_index = 0
    for elem in line.elements:
        if isinstance(elem, xt.Quadrupole) or isinstance(elem, xt.Multipole):
            transfer_matrices.append(get_transfer_matrix_quad_sym(k1_vars[k1_index], elem.length))
            k1_index += 1
        elif isinstance(elem, xt.Drift):
            transfer_matrices.append(get_transfer_matrix_drift_sym(elem.length))

    total_transfer_matrix = sp.eye(4)
    for tm in reversed(transfer_matrices):
        total_transfer_matrix = tm @ total_transfer_matrix

    betx_sym = get_betx_from_transfer_matrix_sym(total_transfer_matrix, tw0.betx[0], tw0.alfx[0])

    beta_derivatives = [sp.diff(betx_sym, k1) for k1 in k1_vars]

    return beta_derivatives



print(f'Betx by twiss: {tw0.betx[-1]}\nBetx by backtrack: {derive_beta_by_backtrack(line, tw0)}')

print("--------------------------------------")
print(f"Computed betx Derivative: {compute_betx_derivative(line, tw0)}")

print("Finite difference gradient betx: ", fd_dict['betx'])

deriv_sympy = compute_beta_derivative_sym(line, tw0)

# Evaluate sympy expressions and print
for i, elem in enumerate(quadrupoles):
    if isinstance(elem, xt.Quadrupole):
        print(f"Symbolic Derivative of betx w.r.t k1 for element: {deriv_sympy[i].evalf(subs={f'k1_0': quadrupoles[0].k1, f'k1_3': quadrupoles[1].k1})}")
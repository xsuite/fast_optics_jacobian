import xdeps as xd
import numpy as np
import sympy
import jax
from jax import numpy as jnp
import xtrack as xt

all_quad_sources = None
target_places = None
dkq_dvv = None

# How to get line? opt.line
# How to get twiss? opt.action_twiss.run()
# How to get old twiss? opt.action_twiss._tw0

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

@jax.jit
def get_values_from_transfer_matrix(r_mat, param_values):
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

QTY_IDX = {
    "betx": 0,
    "bety": 1,
    "alfx": 2,
    "alfy": 3,
    "mux": 4,
    "muy": 5,
    "dx": 6,
    "dy": 7,
    "dpx": 8,
    "dpy": 9,
}

def encode_elements(elements, elem_to_deriv):
    etype_list = []
    data0_list = []
    data1_list = []
    data2_list = []
    data3_list = []
    k1_idx_list = []

    deriv_lookup = {id(elem): i for i, elem in enumerate(elem_to_deriv)}

    for elem in elements:
        if elem in elem_to_deriv:
            etype_list.append(0)
            data0_list.append(elem.length)
            data1_list.append(0.0)  # filler
            data2_list.append(0.0)
            data3_list.append(0.0)
            k1_idx_list.append(deriv_lookup[id(elem)])
        elif isinstance(elem, xt.Quadrupole):
            etype_list.append(1)
            data0_list.append(elem.k1)
            data1_list.append(elem.length)
            data2_list.append(0.0)
            data3_list.append(0.0)
            k1_idx_list.append(-1)
        elif isinstance(elem, xt.Bend):
            etype_list.append(2)
            data0_list.append(elem.k0)
            data1_list.append(elem.k1)
            data2_list.append(elem.length)
            data3_list.append(elem.h)
            k1_idx_list.append(-1)
        elif isinstance(elem, xt.Multipole):
            etype_list.append(4)
            data0_list.append(0.0)
            data1_list.append(0.0)
            data2_list.append(0.0)
            data3_list.append(0.0)
            k1_idx_list.append(-1)
        elif isinstance(elem, xt.Drift) or hasattr(elem, 'length'):
            etype_list.append(3)
            data0_list.append(elem.length)
            data1_list.append(0.0)
            data2_list.append(0.0)
            data3_list.append(0.0)
            k1_idx_list.append(-1)
        else:
            etype_list.append(4)
            data0_list.append(0.0)
            data1_list.append(0.0)
            data2_list.append(0.0)
            data3_list.append(0.0)
            k1_idx_list.append(-1)

    return {
        'etype': jnp.array(etype_list),
        'data0': jnp.array(data0_list),
        'data1': jnp.array(data1_list),
        'data2': jnp.array(data2_list),
        'data3': jnp.array(data3_list),
        'k1_idx': jnp.array(k1_idx_list),
    }


@jax.jit
def get_values(k1_arr, encoded_elements, beta0, gamma0, initial_params):
    def scan_step(params, elem):
        etype = elem['etype']
        d0, d1, d2, d3 = elem['data0'], elem['data1'], elem['data2'], elem['data3']
        k1_idx = elem['k1_idx']

        def quad_param():
            k1 = k1_arr[k1_idx]
            return get_transfer_matrix_quad(k1, d0, beta0, gamma0)

        def quad_const():
            return get_transfer_matrix_quad(d0, d1, beta0, gamma0)

        def bend():
            return get_transfer_matrix_bend(d0, d1, d2, d3, beta0, gamma0)

        def drift():
            return get_transfer_matrix_drift(d0, beta0, gamma0)

        def identity():
            return jnp.eye(6)
        tm = jax.lax.switch(etype, [quad_param, quad_const, bend, drift, identity])
        new_params = get_values_from_transfer_matrix(tm, params)
        return new_params, None


    final_params, _ = jax.lax.scan(scan_step, initial_params, encoded_elements)
    return final_params

def compute_param_derivatives(elements, elem_to_deriv, tw0):
    beta0 = tw0.particle_on_co.beta0[0]
    gamma0 = tw0.particle_on_co.gamma0[0]

    encoded_elements = encode_elements(elements, elem_to_deriv)
    k1_arr = jnp.array([elem.k1 for elem in elem_to_deriv])

    initial_params = jnp.array([
        tw0.betx[0], tw0.bety[0], tw0.alfx[0], tw0.alfy[0],
        tw0.mux[0], tw0.muy[0], tw0.dx[0], tw0.dy[0],
        tw0.dpx[0], tw0.dpy[0]
    ])

    def wrapped_get_values(k1_arr):
        return get_values(k1_arr, encoded_elements, beta0, gamma0, initial_params)

    return jax.jacfwd(wrapped_get_values)(k1_arr)


# def compute_param_derivatives(elements, elem_to_deriv, tw0):
#     def get_values(k1_arr):
#         transfer_matrices = []

#         i = 0
#         for elem in elements:
#             if elem in elem_to_deriv:
#                 assert isinstance(elem, xt.Quadrupole)
#                 transfer_matrices.append(get_transfer_matrix_quad(k1_arr[i], elem.length, tw0.particle_on_co.beta0[0], tw0.particle_on_co.gamma0[0]))
#                 i += 1
#             elif isinstance(elem, xt.Quadrupole) and elem.k1 != 0:
#                 transfer_matrices.append(get_transfer_matrix_quad(elem.k1, elem.length, tw0.particle_on_co.beta0[0], tw0.particle_on_co.gamma0[0]))
#             elif isinstance(elem, xt.Bend):
#                 transfer_matrices.append(get_transfer_matrix_bend(elem.k0, elem.k1, elem.length, elem.h, tw0.particle_on_co.beta0[0], tw0.particle_on_co.gamma0[0]))
#             elif isinstance(elem, xt.Multipole):
#                 transfer_matrices.append(jnp.eye(6))
#             elif isinstance(elem, xt.Drift) or hasattr(elem, 'length'):
#                 transfer_matrices.append(get_transfer_matrix_drift(elem.length, tw0.particle_on_co.beta0[0], tw0.particle_on_co.gamma0[0]))
#             else:
#                 transfer_matrices.append(jnp.eye(6))

#         parameter_values = jnp.array([tw0.betx[0], tw0.bety[0], tw0.alfx[0], tw0.alfy[0], tw0.mux[0], tw0.muy[0], tw0.dx[0], tw0.dy[0], tw0.dpx[0], tw0.dpy[0]])

#         total_transfer_matrix = jnp.eye(6)
#         for i, tm in enumerate(transfer_matrices):
#             total_transfer_matrix = tm @ total_transfer_matrix
#             parameter_values = get_values_from_transfer_matrix(tm, parameter_values)
#         return parameter_values

#     k1_arr = jnp.array([elem.k1 for elem in elem_to_deriv])
#     return jax.jacfwd(get_values)(k1_arr)


def get_dependency_derivatives():
    return all_quad_sources, target_places, dkq_dvv

# collider, opt
def get_dependency_derivatives(opt):
    class FakeQuad:
        pass

    dkq_dvv = {} # Derivatives of quadrupole strengths with respect to the knobs
    a = sympy.var("a")
    for ivv in range(len(opt.vary)):
        vv = opt.vary[ivv].name

        k1 = []
        myelems = {}
        for dd in opt.line.ref_manager.find_deps([opt.line.vars[vv]]):
            if dd.__class__.__name__ == "AttrRef" and dd._key == "k1":
                k1.append((dd._owner._key, dd._expr))
                myelems[dd._owner._key] = FakeQuad()

        fdef = opt.line.ref_manager.mk_fun("myfun", a=opt.line.vars[vv])
        gbl = {
            "vars": opt.line.ref_manager.containers["vars"]._owner.copy(),
            "element_refs": myelems,
        }
        lcl = {}
        exec(fdef, gbl, lcl)
        fff = lcl["myfun"]

        fff(a)
        dk1_dvv = {}
        for kk, expr in k1:
            dd = gbl["element_refs"][kk].k1.diff(a)
            dk1_dvv[kk] = dd

        dkq_dvv[vv] = dk1_dvv

    all_quad_sources = set()
    for vv in dkq_dvv.keys():
        all_quad_sources.update(dkq_dvv[vv].keys())

    target_places = set()
    for tt in opt.targets:
        assert isinstance(tt.tar, tuple)
        target_places.add(tt.tar[1])

    # Sort all quad sources by appearance in the line
    quad_sources_list = []
    for i in opt.action_twiss._tw0.name:
        if i in all_quad_sources:
            quad_sources_list.append(i)
    return quad_sources_list, target_places, dkq_dvv

def get_jac(opt, all_quad_sources, target_places, dkq_dvv):
    # Get twiss for current knobs
    opt_tw = opt.action_twiss.run()
    twiss_derivs = {}
    for place in target_places: # ip1, ip8
        # Calc derivative for all quadrupoles for target place
        # Source point = qqnn, Observation point = target
        twiss_derivs[place] = {}
        trunc_elements = np.array([opt.line.element_dict[name] for name in opt_tw.rows[:place].name])
        nonzero_qq = []
        nonzero_qqn = []
        for qqnn in all_quad_sources:
            if opt_tw['s', place] < opt_tw['s', qqnn]:
                twiss_derivs[place][qqnn] = np.zeros(10)
            else:
                nonzero_qqn.append(qqnn)
                nonzero_qq.append(opt.line.element_dict[qqnn])
                # add to list to be calculated
        nonzero_deriv = compute_param_derivatives(trunc_elements, nonzero_qq, opt_tw)

        for i, qqn in enumerate(nonzero_qqn):
            twiss_derivs[place][qqn] = nonzero_deriv[i]
        for qqn, deriv in zip(nonzero_qqn, nonzero_deriv.T):
            twiss_derivs[place][qqn] = deriv

    jac_estim = np.zeros((len(opt.targets), len(opt.vary)))
    for itt, tt in enumerate(opt.targets):

        assert isinstance(tt.tar, tuple)

        tar_quantity = tt.tar[0]
        quantity_idx = QTY_IDX[tar_quantity]
        tar_place = tt.tar[1]
        tar_weight = tt.weight

        for ivv in range(len(opt.vary)):

            vv = opt.vary[ivv].name

            quad_names = dkq_dvv[vv].keys()

            dtar_dvv = 0
            for qqnn in quad_names:
                dtar_dvv += twiss_derivs[tar_place][qqnn][quantity_idx] * float(dkq_dvv[vv][qqnn])

            dtar_dvv *= tar_weight

            jac_estim[itt, ivv] = dtar_dvv

    return jac_estim

def get_jacobian(self, x, opt, f0=None):
    if len(self.targets) == 12 and len(self.vary) == 20:
        x = np.array(x).copy()
        steps = self._knobs_to_x(self.steps_for_jacobian)
        # get twiss
        global all_quad_sources, target_places, dkq_dvv
        if all_quad_sources is None or target_places is None or dkq_dvv is None:
            all_quad_sources, target_places, dkq_dvv = get_dependency_derivatives(opt)
        jacobian = get_jac(opt, all_quad_sources, target_places, dkq_dvv)
        return jacobian
    else:
        if hasattr(self, "_force_jacobian"):
            return self._force_jacobian
        x = np.array(x).copy()
        steps = self._knobs_to_x(self.steps_for_jacobian)
        assert len(x) == len(steps)
        if f0 is None:
            f0 = self(x)
        if np.isscalar(f0):
            jac = np.zeros((1, len(x)))
        else:
            jac = np.zeros((len(f0), len(x)))
        mask_input = self.mask_input
        for ii in range(len(x)):
            if not mask_input[ii]:
                continue
            x[ii] += steps[ii]
            jac[:, ii] = (self(x, check_limits=False) - f0) / steps[ii]
            x[ii] -= steps[ii]

        self._last_jac = jac
        return jac


#xd.optimize.optimize.MeritFunctionForMatch.get_jacobian = get_jacobian
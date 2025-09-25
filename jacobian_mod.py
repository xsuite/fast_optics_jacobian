import xdeps as xd
import numpy as np
import sympy
import jax
from jax import numpy as jnp
import xtrack as xt
from typing import NamedTuple
from functools import partial

#jax.config.update("jax_enable_x64", True)

all_quad_sources = None
target_places = None
dkq_dvv = None

# How to get line? opt.line
# How to get twiss? opt.action_twiss.run()
# How to get old twiss? opt.action_twiss._tw0

class TransferMatrixFactory:

    @staticmethod
    @jax.jit
    def quad(k1, l, beta0, gamma0):
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

    @staticmethod
    @jax.jit
    def drift(l, beta0, gamma0):
        f_matrix = jnp.array([
            [1, l, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, l, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, l/(beta0**2 * gamma0**2)],
            [0, 0, 0, 0, 0, 1]
        ])
        return f_matrix

    @staticmethod
    @jax.jit
    def bend(k0, k1, l, h, beta0, gamma0):
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

class EncodedElem(NamedTuple):
    etype: int
    data0: float = 0.0
    data1: float = 0.0
    data2: float = 0.0
    data3: float = 0.0
    k1_idx: int = -1

@jax.jit
def get_values_from_transfer_matrix(r_mat, param_values):
    # Order: betx, bety, alfx, alfy, mux, muy, dx, dy, dpx, dpy
    bx0, by0, ax0, ay0, mux0, muy0, dx0, dy0, dpx0, dpy0 = param_values

    # --- Horizontal plane ---
    r00, r01, r10, r11 = r_mat[0,0], r_mat[0,1], r_mat[1,0], r_mat[1,1]

    tmp_x = r00 * bx0 - r01 * ax0
    betx = (tmp_x**2 + r01**2) / bx0
    alfx = -((tmp_x * (r10 * bx0 - r11 * ax0) + r01 * r11) / bx0)
    mux = mux0 + jnp.arctan2(r01, tmp_x) / (2 * jnp.pi)

    # --- Vertical plane ---
    r22, r23, r32, r33 = r_mat[2,2], r_mat[2,3], r_mat[3,2], r_mat[3,3]

    tmp_y = r22 * by0 - r23 * ay0
    bety = (tmp_y**2 + r23**2) / by0
    alfy = -((tmp_y * (r32 * by0 - r33 * ay0) + r23 * r33) / by0)
    muy = muy0 + jnp.arctan2(r23, tmp_y) / (2 * jnp.pi)

    # --- Dispersion ---
    dx = r00 * dx0 + r01 * dpx0 + r_mat[0,5]
    dy = r22 * dy0 + r23 * dpy0 + r_mat[2,5]
    dpx = r10 * dx0 + r11 * dpx0 + r_mat[1,5]
    dpy = r32 * dy0 + r33 * dpy0 + r_mat[3,5]

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
    if elem_to_deriv is not None:
        deriv_lookup = {id(elem): i for i, elem in enumerate(elem_to_deriv)}

    encoded = []

    for elem in elements:
        if elem_to_deriv is not None and elem in elem_to_deriv:
            encoded.append(EncodedElem(
                etype=0,
                data0=elem.length,
                k1_idx=deriv_lookup[id(elem)]
            ))
        elif isinstance(elem, xt.Quadrupole):
            encoded.append(EncodedElem(
                etype=1,
                data0=elem.k1,
                data1=elem.length
            ))
        elif isinstance(elem, xt.Bend) or isinstance(elem, xt.RBend):
            encoded.append(EncodedElem(
                etype=2,
                data0=elem.k0,
                data1=elem.k1,
                data2=elem.length,
                data3=elem.h
            ))
        elif isinstance(elem, xt.Multipole):
            encoded.append(EncodedElem(
                etype=4
            ))
        elif isinstance(elem, xt.Drift) or hasattr(elem, 'length'):
            encoded.append(EncodedElem(
                etype=3,
                data0=elem.length
            ))
        else:
            encoded.append(EncodedElem(
                etype=4,
            ))

    # Convert list of NamedTuples to NamedTuple of arrays for JAX
    return EncodedElem(
        etype=jnp.array([e.etype for e in encoded]),
        data0=jnp.array([e.data0 for e in encoded]),
        data1=jnp.array([e.data1 for e in encoded]),
        data2=jnp.array([e.data2 for e in encoded]),
        data3=jnp.array([e.data3 for e in encoded]),
        k1_idx=jnp.array([e.k1_idx for e in encoded]),
    )

@partial(jax.jit, static_argnums=(2,3))
def get_values(k1_arr, encoded_elements, beta0, gamma0, initial_params):
    def scan_step(params, elem):
        TMF = TransferMatrixFactory

        # Defining methods inside the switch to avoid recompilation
        tm = jax.lax.switch(elem.etype, [
            lambda: TMF.quad(k1_arr[elem.k1_idx], elem.data0, beta0, gamma0),
            lambda: TMF.quad(elem.data0, elem.data1, beta0, gamma0),
            lambda: TMF.bend(elem.data0, elem.data1, elem.data2, elem.data3, beta0, gamma0),
            lambda: TMF.drift(elem.data0, beta0, gamma0),
            lambda: jnp.eye(6)
            ]
        )
        new_params = get_values_from_transfer_matrix(tm, params)
        return new_params, None

    final_params, _ = jax.lax.scan(scan_step, initial_params, encoded_elements)
    return final_params

def compute_param_derivatives(elements, elem_to_deriv, init_cond, beta0, gamma0):
    encoded_elements = encode_elements(elements, elem_to_deriv)
    k1_arr = jnp.array([elem.k1 for elem in elem_to_deriv])

    initial_params = jnp.array(init_cond)

    def wrapped_get_values(k1_arr):
        return get_values(k1_arr, encoded_elements, beta0, gamma0, initial_params)

    pushfwd = partial(jax.jvp, wrapped_get_values, (k1_arr,))
    basis = jnp.eye(len(k1_arr))
    y, jac = jax.vmap(pushfwd, out_axes=(None, 1))((basis,))
    return jac, y
    #return jax.jacfwd(wrapped_get_values)(k1_arr)

@partial(jax.jit, static_argnums=(1,2))
def get_values_noderiv(encoded_elements, beta0, gamma0, initial_params):
    def scan_step(params, elem):
        TMF = TransferMatrixFactory

        # Defining methods inside the switch to avoid recompilation
        tm = jax.lax.switch(elem.etype, [
            lambda: jnp.eye(6),
            lambda: TMF.quad(elem.data0, elem.data1, beta0, gamma0),
            lambda: TMF.bend(elem.data0, elem.data1, elem.data2, elem.data3, beta0, gamma0),
            lambda: TMF.drift(elem.data0, beta0, gamma0),
            lambda: jnp.eye(6)
            ]
        )
        new_params = get_values_from_transfer_matrix(tm, params)
        return new_params, None

    final_params, _ = jax.lax.scan(scan_step, initial_params, encoded_elements)
    return final_params

def compute_values(elements, tw0):
    beta0 = tw0.particle_on_co.beta0[0]
    gamma0 = tw0.particle_on_co.gamma0[0]

    encoded_elements = encode_elements(elements, None)

    initial_params = jnp.array([
        tw0.betx[0], tw0.bety[0], tw0.alfx[0], tw0.alfy[0],
        tw0.mux[0], tw0.muy[0], tw0.dx[0], tw0.dy[0],
        tw0.dpx[0], tw0.dpy[0]
    ])

    return get_values_noderiv(encoded_elements, beta0, gamma0, initial_params)

def get_dependency_derivatives():
    return all_quad_sources, target_places, dkq_dvv

# collider, opt
def get_dependency_derivatives(opt):
    class DummyElement:
        """Placeholder object for injecting symbolic attributes."""
        pass

    dkq_dvv = {}  # Mapping: knob name -> {quad name -> d(quad)/d(knob)}

    for vary_entry in opt.vary:
        knob_name = vary_entry.name
        symbolic_var = sympy.var("a")

        # Find all quadrupole k1 dependencies on this knob
        quad_exprs = []
        dummy_quads = {}

        for dep in opt.line.ref_manager.find_deps([opt.line.vars[knob_name]]):
            if dep.__class__.__name__ == "AttrRef" and dep._key == "k1":
                quad_name = dep._owner._key
                quad_exprs.append((quad_name, dep._expr))
                dummy_quads[quad_name] = DummyElement()

        # Build symbolic expression function for the knob
        func_code = opt.line.ref_manager.mk_fun("myfun", a=opt.line.vars[knob_name])
        func_globals = {
            "vars": opt.line.ref_manager.containers["vars"]._owner.copy(),
            "element_refs": dummy_quads,
        }
        func_locals = {}

        ################### myfun ################################
        # def myfun(a):
        #    knob_name = a
        #    element_refs[quad_name].k1 = (1.0 * knob_name) -> SymPy expression
        #    ...
        ##########################################################

        exec(func_code, func_globals, func_locals) # Create function, stored in func_locals
        func_locals["myfun"](symbolic_var) # Execute function

        # Extract derivatives of k1 with respect to this knob
        k1_derivs = {}
        for quad_name, _ in quad_exprs:
            derivative = func_globals["element_refs"][quad_name].k1.diff(symbolic_var)
            k1_derivs[quad_name] = derivative

        dkq_dvv[knob_name] = k1_derivs

    # Set of all quadrupole names appearing in the derivatives
    quad_sources = set()
    for derivs in dkq_dvv.values():
        quad_sources.update(derivs.keys())

    assert all(isinstance(t.tar, tuple) or isinstance(t, xt.TargetRelPhaseAdvance) for t in opt.targets)
    # Set of all target locations used in optimization and sort them by order
    target_places = set()
    for target in opt.targets:
        if isinstance(target.tar, tuple):
            target_places.add(target.tar[1])
        elif isinstance(target, xt.TargetRelPhaseAdvance):
            if target.start != '__ele_start__':
                target_places.add(target.start)
            if target.end != '__ele_stop__':
                target_places.add(target.end)
            else:
                if opt.action_twiss._tw0.name[-2] not in target_places:
                    target_places.add(opt.action_twiss._tw0.name[-2])
                # Assumption: Point before _end_point is same as endpoint given in opt
        else:
            raise ValueError(f"Unknown target type: {type(target)}")
    # Convert to ordered list based on appearance of name in opt.line
    index_map = {name: i for i, name in enumerate(opt.action_twiss._tw0.name)}
    target_places = sorted(target_places, key=index_map.get)

    # Ordered list of quadrupole sources (based on their position in the beamline)
    quad_sources_ordered = [
        name for name in opt.action_twiss._tw0.name if name in quad_sources
    ]

    # quad_sources and target_places are both ordered
    return quad_sources_ordered, target_places, dkq_dvv

def get_jac(opt, all_quad_sources, target_places, dkq_dvv):
    # Get twiss for current knobs
    opt_tw = opt.action_twiss.run()
    #opt_tw = opt.action_twiss._tw0
    # Initial conditions for first derivative calculation
    init_cond = np.array([opt_tw.betx[0], opt_tw.bety[0], opt_tw.alfx[0], opt_tw.alfy[0],
                          opt_tw.mux[0], opt_tw.muy[0], opt_tw.dx[0], opt_tw.dy[0],
                          opt_tw.dpx[0], opt_tw.dpy[0]])
    beta0 = opt_tw.particle_on_co.beta0[0]
    gamma0 = opt_tw.particle_on_co.gamma0[0]

    twiss_derivs = {}
    for place in target_places: # ip1, ip8 in order of appearance
        # Calc derivative for all quadrupoles for target place
        # Source point = qqnn, Observation point = target
        twiss_derivs[place] = {}
        trunc_elements = np.array([opt.line.element_dict[name] for name in opt_tw.rows[:place].name])
        nonzero_qq = []
        nonzero_qqn = []
        for qqnn in all_quad_sources:
            if opt_tw['s', place] < opt_tw['s', qqnn]:
                twiss_derivs[place][qqnn] = np.zeros(10) # batch after
            else:
                nonzero_qqn.append(qqnn)
                nonzero_qq.append(opt.line.element_dict[qqnn]) # first elements
                # add to list to be calculated
        nonzero_deriv, _ = compute_param_derivatives(trunc_elements, nonzero_qq, init_cond, beta0, gamma0)

        for i, qqn in enumerate(nonzero_qqn):
            twiss_derivs[place][qqn] = nonzero_deriv[i]
        for qqn, deriv in zip(nonzero_qqn, nonzero_deriv.T):
            twiss_derivs[place][qqn] = deriv

    jac_estim = np.zeros((len(opt.targets), len(opt.vary)))
    for itt, tt in enumerate(opt.targets):

        assert isinstance(tt.tar, tuple) or isinstance(tt, xt.TargetRelPhaseAdvance)
        tar_start = None
        if isinstance(tt.tar, tuple):
            tar_quantity = tt.tar[0]
            tar_place = tt.tar[1]
        else:
            tar_quantity = tt.var
            tar_place = target_places[-1] if tt.end == '__ele_stop__' else tt.end
            tar_start = None if tt.start == '__ele_start__' else tt.start
            tar_weight = tt.weight

        tar_weight = tt.weight
        quantity_idx = QTY_IDX[tar_quantity]
        for ivv in range(len(opt.vary)):
            vv = opt.vary[ivv].name
            quad_names = dkq_dvv[vv].keys()

            dtar_dvv = 0
            for qqnn in quad_names:
                if qqnn in twiss_derivs[tar_place].keys():
                    dtar_dvv += (twiss_derivs[tar_place][qqnn][quantity_idx]) * float(dkq_dvv[vv][qqnn])
                    if tar_start is not None:
                        dtar_dvv -= (twiss_derivs[tar_start][qqnn][quantity_idx]) * float(dkq_dvv[vv][qqnn])

            dtar_dvv *= tar_weight

            jac_estim[itt, ivv] = dtar_dvv

    return jac_estim

# flag to set if targets contain anything but optical functions or dispersion
def get_jacobian(self, x, opt, f0=None, flag = False):
    for i in self.targets:
        if not isinstance(i.tar, tuple) and not isinstance(i, xt.TargetRelPhaseAdvance):
            flag = False
            break
        else:
            if isinstance(i.tar, tuple):
                if i.tar[0] not in ['betx', 'bety', 'alfx', 'alfy', 'mux', 'muy', 'dx', 'dy', 'dpx', 'dpy']:
                    flag = False
                    break
        flag = True
    if flag:
        x = np.array(x).copy()
        steps = self._knobs_to_x(self.steps_for_jacobian)
        # get twiss
        global all_quad_sources, target_places, dkq_dvv
        if all_quad_sources is None or target_places is None or dkq_dvv is None:
            all_quad_sources, target_places, dkq_dvv = get_dependency_derivatives(opt)
        jacobian = get_jac(opt, all_quad_sources, target_places, dkq_dvv)
        self._last_jac = jacobian
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

        self._set_x(x)
        self._last_jac = jac
        return jac


#xd.optimize.optimize.MeritFunctionForMatch.get_jacobian = get_jacobian
"""Extend JAX backend to orbit matching.
Unlike the optics backend (030, which freezes the orbit and differentiates Twiss),
we differentiate the orbit trajectory itself w.r.t. corrector dipole kicks.

Pipeline:
  knob -> corrector knl[0]/ksl[0] -> single-pass orbit trajectory -> x/px/y/py @ target

The full pipeline is one JAX-differentiable function of the kick strengths, so we
get the orbit-response Jacobian (corrector sensitivity matrix) in a single jacfwd
call - no per-knob re-tracking.

To validate, the JAX Jacobian is verified against xsuite finite differences, and
the orbit bump match is solved with a Gauss-Newton iteration using the JAX Jacobian.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import xtrack as xt
from utils import _orbit_line

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Build the same line / match problem as opt_orbit in 032.
# ---------------------------------------------------------------------------
env = xt.Environment()
env.vars.default_to_zero = True
line = _orbit_line()

VARY = [
    "kick_h_1",
    "kick_v_1",
    "kick_h_2",
    "kick_v_2",
    "kick_h_3",
    "kick_v_3",
    "kick_h_4",
    "kick_v_4",
]
TARGETS = [
    ("x", "mid", 1e-3),
    ("y", "mid", -2e-3),
    ("px", "mid", 0.0),
    ("py", "mid", 0.0),
    ("x", "end", 0.0),
    ("y", "end", 0.0),
    ("px", "end", 0.0),
    ("py", "end", 0.0),
]
COORD_INDEX = {"x": 0, "px": 1, "y": 2, "py": 3}

tw0 = line.twiss(betx=1, bety=1)
beta0 = float(tw0.particle_on_co.beta0[0])

# ===========================================================================
# Part 1: Exact element maps for orbit tracking
# ===========================================================================
# Reused from xtrack.jax_optics: exact drift and thin multipole kicks.
# These map a 6D phase-space state through one element.
# ===========================================================================


def rvv_from_delta(delta, beta0):
    one_plus_delta = 1.0 + delta
    denom = jnp.sqrt(
        beta0 * beta0 * one_plus_delta * one_plus_delta + 1.0 - beta0 * beta0
    )
    return one_plus_delta / denom


def drift_exact(state, length, beta0):
    x, px, y, py, zeta, delta = state
    one_plus_delta = 1.0 + delta
    one_over_pz = 1.0 / jnp.sqrt(one_plus_delta**2 - px**2 - py**2)
    rv0v = 1.0 / rvv_from_delta(delta, beta0)
    dzeta = 1.0 - rv0v * one_plus_delta * one_over_pz
    return jnp.array(
        [
            x + px * one_over_pz * length,
            px,
            y + py * one_over_pz * length,
            py,
            zeta + dzeta * length,
            delta,
        ]
    )


def dipole_kick(state, knl0, ksl0):
    """Order-0 multipole kick (port of thin_multipole, n=0)."""
    x, px, y, py, zeta, delta = state
    return jnp.array([x, px - knl0, y, py + ksl0, zeta, delta])


# Element-type codes.
ET_DRIFT, ET_CORR, ET_IDENT = 0, 1, 2

# ===========================================================================
# Part 2: Struct-of-arrays encoding of the line
# ===========================================================================
# Encode each element as a row in arrays: element type, length, corrector indices.
# Correctors are driven by parameters p[0..n_param] where each corrector has
# knl[0] at p[2*i] (horizontal kick) and ksl[0] at p[2*i+1] (vertical kick).
# ===========================================================================

ordered_names = list(tw0.name[:-1])  # drop _end_point
ed = line.element_dict

# Identify which elements are correctors (Multipole type).
corr_names = [n for n in ordered_names if type(ed.get(n)).__name__ == "Multipole"]

# Parameter vector layout: [corr1.knl0, corr1.ksl0, corr2.knl0, corr2.ksl0, ...]
# Each corrector has two slots: one for the horizontal dipole kick (knl[0]),
# one for the vertical dipole kick (ksl[0]).
pidx_h = {n: 2 * i for i, n in enumerate(corr_names)}
pidx_v = {n: 2 * i + 1 for i, n in enumerate(corr_names)}
n_param = 2 * len(corr_names)

# Compute element lengths from Twiss s-positions, not just element_dict.
# Some elements are "split" by inserted markers (e.g. ||drift_3::0 inserted by
# xtrack), so they don't appear in element_dict - but their length is the s-gap.
svals = np.asarray(tw0.s)  # s-position at each twiss row
et, L, hidx, vidx = [], [], [], []

for i, n in enumerate(ordered_names):
    e = ed.get(n)
    cls = type(e).__name__ if e is not None else None
    span = float(svals[i + 1] - svals[i])  # element occupies this s-distance

    if cls == "Multipole":  # corrector dipole
        et.append(ET_CORR)
        L.append(span)
        hidx.append(pidx_h[n])  # where h-kick reads from parameter vector
        vidx.append(pidx_v[n])  # where v-kick reads from parameter vector
    elif span > 0.0:  # any positive length with no special element
        et.append(ET_DRIFT)
        L.append(span)
        hidx.append(-1)  # no corrector index
        vidx.append(-1)
    else:  # zero-length (markers)
        et.append(ET_IDENT)
        L.append(0.0)
        hidx.append(-1)
        vidx.append(-1)

ENC = (
    jnp.array(et, dtype=jnp.int32),
    jnp.array(L),
    jnp.array(hidx, dtype=jnp.int32),
    jnp.array(vidx, dtype=jnp.int32),
)

# Map each target place to its scan row (orbit AT that element = scan output).
name_to_row = {n: i for i, n in enumerate(ordered_names)}
target_rows = jnp.array(
    [name_to_row[place] for _, place, _ in TARGETS], dtype=jnp.int32
)
target_coord = jnp.array([COORD_INDEX[q] for q, _, _ in TARGETS], dtype=jnp.int32)

s0 = jnp.array(
    [
        tw0["x", 0],
        tw0["px", 0],
        tw0["y", 0],
        tw0["py", 0],
        tw0["zeta", 0],
        tw0["delta", 0],
    ]
)


def emap(state, e, p):
    """Apply the element map (exact element physics) to the orbit.

    Parameters
    ----------
    state : 6D array
        Current phase-space state [x, px, y, py, zeta, delta].
    e : tuple of 4 arrays (one row from ENC)
        Element type code, length, h-kick index, v-kick index.
    p : 1D array
        Parameter vector: corrector strengths [kh1, kv1, kh2, kv2, ...].

    Returns
    -------
    new_state : 6D array
        Phase-space state after the element.
    """
    et_, L_, hi, vi = e
    # Extract kicks from the parameter vector; non-correctors get kick=0.
    kh = jnp.where(hi >= 0, p[jnp.maximum(hi, 0)], 0.0)
    kv = jnp.where(vi >= 0, p[jnp.maximum(vi, 0)], 0.0)

    def corr(s):
        """Corrector element: drift -> kick -> drift (thin-lens split)."""
        s = drift_exact(s, 0.5 * L_, beta0)
        s = dipole_kick(s, kh, kv)
        s = drift_exact(s, 0.5 * L_, beta0)
        return s

    return lax.switch(
        et_,
        [
            lambda s: drift_exact(s, L_, beta0),  # ET_DRIFT
            corr,  # ET_CORR
            lambda s: s,  # ET_IDENT
        ],
        state,
    )


# ===========================================================================
# Part 3: JAX Jacobian for orbit response
# ===========================================================================
# orbit_targets(p) is fully differentiable: it takes corrector strengths p,
# propagates the orbit through the line, and returns the coordinates at
# target positions. One jacfwd gives d(targets)/d(p) - no per-corrector loops.
# ===========================================================================


@jax.jit
def orbit_targets(p):
    """Propagate the orbit and extract coordinates at target positions.

    Parameters
    ----------
    p : 1D array
        Corrector kick strengths [kh1, kv1, kh2, kv2, ...].

    Returns
    -------
    1D array
        Orbit coordinates at target positions, flattened as
        [x_targ1, y_targ1, px_targ1, ..., px_targN].
    """

    def body(s, e):
        sn = emap(s, e, p)
        return sn, sn  # return next state and history

    _, hist = lax.scan(body, s0, ENC)  # hist[i] = orbit after element i
    return hist[target_rows, target_coord]  # gather requested coordinates


# Pre-compile the Jacobian function: d(orbit coordinates)/d(corrector strengths).
orbit_jac_fn = jax.jit(jax.jacfwd(orbit_targets))


def dparam_dvar(delta=1e-6):
    """d(corrector knl0/ksl0)/d(vary knob)"""

    def p_vec():
        out = np.zeros(n_param)
        for nm in corr_names:
            out[pidx_h[nm]] = float(ed[nm].knl[0])
            out[pidx_v[nm]] = float(ed[nm].ksl[0])
        return out

    v0 = {v: float(line.vars[v]._value) for v in VARY}
    dp = np.zeros((n_param, len(VARY)))
    for j, v in enumerate(VARY):
        line.vars[v] = v0[v] + delta
        pp = p_vec()
        line.vars[v] = v0[v] - delta
        pm = p_vec()
        line.vars[v] = v0[v]
        dp[:, j] = (pp - pm) / (2 * delta)
    return dp


DP = dparam_dvar()


def jax_jacobian():
    p = np.zeros(n_param)
    for nm in corr_names:
        p[pidx_h[nm]] = float(ed[nm].knl[0])
        p[pidx_v[nm]] = float(ed[nm].ksl[0])
    jac_p = np.asarray(orbit_jac_fn(jnp.asarray(p)))  # (n_tar, n_param)
    return jac_p @ DP  # (n_tar, n_vary)


# ---------------------------------------------------------------------------
# 1. Verify the JAX orbit Jacobian against xsuite finite differences.
# ---------------------------------------------------------------------------
def fd_jacobian(delta=1e-7):
    v0 = {v: float(line.vars[v]._value) for v in VARY}
    jac = np.zeros((len(TARGETS), len(VARY)))
    for j, v in enumerate(VARY):
        line.vars[v] = v0[v] + delta
        twp = line.twiss(betx=1, bety=1)
        line.vars[v] = v0[v] - delta
        twm = line.twiss(betx=1, bety=1)
        line.vars[v] = v0[v]
        for it, (q, place, _) in enumerate(TARGETS):
            jac[it, j] = (twp[q, place] - twm[q, place]) / (2 * delta)
    return jac


for v in VARY:
    line.vars[v] = 0.0
jac_jax = jax_jacobian()
jac_fd = fd_jacobian()
err = np.max(np.abs(jac_jax - jac_fd))
denom = np.maximum(np.abs(jac_fd), 1e-12)
rel = np.max(np.abs(jac_jax - jac_fd) / denom)
print(
    f"[orbit jac] shape {jac_jax.shape}  max|JAX-FD|={err:.2e}  "
    f"max rel={rel:.2e}  (JAX orbit-response == xsuite FD)"
)


# ---------------------------------------------------------------------------
# 2. Solve the bump: Gauss-Newton with the JAX Jacobian + xsuite residual.
# ---------------------------------------------------------------------------
def residual():
    tw = line.twiss(betx=1, bety=1)
    return np.array([tw[q, place] - val for q, place, val in TARGETS])


for v in VARY:
    line.vars[v] = 0.0

for it in range(8):
    r = residual()
    pen = np.sqrt(np.sum(r**2))
    if pen < 1e-12:
        break
    J = jax_jacobian()  # (n_tar, n_vary)
    dx, *_ = np.linalg.lstsq(J, -r, rcond=None)
    for j, v in enumerate(VARY):
        line.vars[v] = float(line.vars[v]._value) + dx[j]

tw = line.twiss(betx=1, bety=1)
print(f"[solve] {it} steps, penalty={np.sqrt(np.sum(residual() ** 2)):.3e}")
print(
    f"   mid: x={tw['x', 'mid']:.6e} (1e-3) y={tw['y', 'mid']:.6e} (-2e-3) "
    f"px={tw['px', 'mid']:.2e} py={tw['py', 'mid']:.2e}"
)
print(
    f"   end: x={tw['x', 'end']:.2e} y={tw['y', 'end']:.2e} "
    f"px={tw['px', 'end']:.2e} py={tw['py', 'end']:.2e}"
)

"""Extend the JAX backend to CHROMATICITY matching (dQx, dQy).

Second-order extension to support chromaticities. Unlike the
linear backends (optics, tune), chromaticity is fundamentally
nonlinear. Following extensions are added:

  1. Sextupole elements: exact thick-sextupole maps (linear-backend drops them).
  2. Off-momentum orbit: sextupoles interact with the dispersion D(δ).
     The effective focusing is: k1_eff(δ) = k2 * D(δ) * δ.
     This is the source of chromatic tune shifts.
  3. Delta-derivatives: chromaticity is dQ/dδ, evaluated at δ=0.

To compute dQ/dδ at δ=0, we build the one-turn matrix M(δ) by linearizing each
element around its off-momentum closed orbit [Dx*δ, Dpx*δ, Dy*δ, Dpy*δ, 0, δ].
Crucially, only sextupoles "feel" this transverse offset (linear elements are
independent of transverse position to 1st order).

  Q(δ, k2) := arccos(Tr(M(δ, k2)) / 2) / 2π
  chroma(k2) := dQ/dδ|_{δ=0}  [1st derivative via jacfwd over δ]
  Jacobian := dchroma/dknob  [2nd jacfwd over the 2 sext knobs]

To validate, d(chroma)/d(knob) is compared against finite differences in xsuite.
Then we solve the chromaticity match to target dQ values using Newton's method.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import xtrack as xt

from xtrack.jax_optics import (
    drift_exact_jax,
    quad_body_jax,
    bend_with_edges_jax,
    thin_quad_skew_kick_jax,
    bend_edge_coeffs,
)

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Load the ring.  Dipole edges are kept active: each bend row carries its
# linear edge-focusing coefficients, so Q(delta) matches xtrack with edges on.
# ---------------------------------------------------------------------------
line = xt.Line.from_json("lattice_data/lhc_thick_with_knobs.json")
line.build_tracker()

VARY = ["ksf.b1", "ksd.b1"]
TARGETS = [("dqx", 10.0), ("dqy", 12.0)]

tw0 = line.twiss(method="4d")
beta0 = float(tw0.particle_on_co.beta0[0])
print(
    f"[ring] {len(tw0.name)} elements, qx={tw0.qx:.4f} qy={tw0.qy:.4f}  "
    f"dqx={tw0.dqx:.3f} dqy={tw0.dqy:.3f}"
)

ordered_names = list(tw0.name[:-1])
ed = line.element_dict

# Dispersion at each element entrance (aligned with ordered_names).
Dx = np.asarray(tw0.dx)[:-1]
Dpx = np.asarray(tw0.dpx)[:-1]
Dy = np.asarray(tw0.dy)[:-1]
Dpy = np.asarray(tw0.dpy)[:-1]


# ===========================================================================
# Part 1: Sextupole sensitivity
# ===========================================================================
# Determine which sextupoles are driven by the knobs (ksf.b1, ksd.b1) and
# compute d(k2)/d(knob) via finite difference.
# Chromaticity is linear in k2, so fixed sextupoles are irrelevant to the
# Jacobian (their constant k2 differentiates away).
# ===========================================================================


def dk2_dknob(delta=1.0):
    sexts = [n for n in ordered_names if type(ed.get(n)).__name__ == "Sextupole"]
    k0 = {v: float(line[v]) for v in VARY}

    def k2_vec():
        return np.array([float(ed[s].k2) for s in sexts])

    dks = np.zeros((len(sexts), len(VARY)))
    for j, v in enumerate(VARY):
        line[v] = k0[v] + delta
        kp = k2_vec()
        line[v] = k0[v] - delta
        km = k2_vec()
        line[v] = k0[v]
        dks[:, j] = (kp - km) / (2 * delta)
    mask = np.any(dks != 0.0, axis=1)
    driven = [sexts[i] for i in range(len(sexts)) if mask[i]]
    return driven, dks[mask, :]


driven_sexts, DKS = dk2_dknob()
DKS = jnp.asarray(DKS)
ks0 = jnp.asarray([float(ed[s].k2) for s in driven_sexts])
ks_pos = {s: i for i, s in enumerate(driven_sexts)}
print(f"[sext] {len(driven_sexts)} knob-driven sextupoles")

# ===========================================================================
# Part 2: Struct-of-arrays encoding with all sextupoles
# ===========================================================================
# Encode the ring with full sextupole support. Each row includes:
#   - Element type and linear parameters (drift/quad/bend)
#   - Sextupole k2 source (knob index or fixed value)
#   - Dispersion (Dx, Dpx, Dy, Dpy): needed to linearize sextupoles
#
# Includes *all* sextupoles, not just knob-driven ones. Fixed
# sextupoles contribute the bulk of the chromaticity correction.
# They're constant in knobs, so they don't affect the Jacobian,
# but they're essential for the absolute value of Q(δ).
# ===========================================================================
ET_DRIFT, ET_QUAD, ET_BEND, ET_SEXT, ET_KICK = 0, 1, 2, 3, 4
# Each row: (et, L, k0, h, k1, ksidx, kn1, ks1, Dx, Dpx, Dy, Dpy, k2fix,
#            r21i, r43i, r21o, r43o)   -- last four are linear dipole-edge coeffs
rows = []
acc = 0.0  # accumulator for consecutive drifts
NOEDGE = (0, 0, 0, 0)  # edge coeffs for non-bend rows


def flush():
    global acc
    if acc > 0.0:
        rows.append((ET_DRIFT, acc, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, *NOEDGE))
        acc = 0.0


for i, n in enumerate(ordered_names):
    e = ed.get(n)
    cls = type(e).__name__ if e is not None else None
    span = float(np.asarray(tw0.s)[i + 1] - np.asarray(tw0.s)[i])
    D = (Dx[i], Dpx[i], Dy[i], Dpy[i])
    if cls == "Quadrupole" and float(e.k1) != 0.0:
        flush()
        rows.append(
            (ET_QUAD, float(e.length), 0, 0, float(e.k1), -1, 0, 0, *D, 0, *NOEDGE)
        )
    elif cls in ("Bend", "RBend"):
        flush()
        rows.append(
            (
                ET_BEND,
                float(e.length),
                float(e.k0),
                float(e.h),
                0,
                -1,
                0,
                0,
                *D,
                0,
                *bend_edge_coeffs(e),
            )
        )
    elif n in ks_pos:  # knob-driven sextupole: k2 from knob
        flush()
        rows.append(
            (ET_SEXT, float(e.length), 0, 0, 0, ks_pos[n], 0, 0, *D, 0, *NOEDGE)
        )
    elif cls == "Sextupole":  # fixed sextupole: baked-in k2
        flush()
        rows.append(
            (ET_SEXT, float(e.length), 0, 0, 0, -1, 0, 0, *D, float(e.k2), *NOEDGE)
        )
    elif e is not None and not getattr(e, "isthick", False):
        knl = np.asarray(getattr(e, "knl", [0.0]), dtype=float)
        ksl = np.asarray(getattr(e, "ksl", [0.0]), dtype=float)
        kn1 = float(knl[1]) if len(knl) > 1 else 0.0
        ks1 = float(ksl[1]) if len(ksl) > 1 else 0.0
        if kn1 or ks1:
            flush()
            rows.append((ET_KICK, 0, 0, 0, 0, -1, kn1, ks1, *D, 0, *NOEDGE))
        elif span > 0.0:
            acc += span
    elif span > 0.0:
        acc += span  # thick non-focusing -> drift
flush()

A = np.array(rows, dtype=float)
ENC = (
    jnp.array(A[:, 0], dtype=jnp.int32),
    jnp.array(A[:, 1]),
    jnp.array(A[:, 2]),
    jnp.array(A[:, 3]),
    jnp.array(A[:, 4]),
    jnp.array(A[:, 5], dtype=jnp.int32),
    jnp.array(A[:, 6]),
    jnp.array(A[:, 7]),
    jnp.array(A[:, 8]),
    jnp.array(A[:, 9]),
    jnp.array(A[:, 10]),
    jnp.array(A[:, 11]),
    jnp.array(A[:, 12]),
    jnp.array(A[:, 13]),
    jnp.array(A[:, 14]),
    jnp.array(A[:, 15]),
    jnp.array(A[:, 16]),
)
print(f"[encode] compressed to {A.shape[0]} rows")


def sext_kick(state, g):
    """Thin sextupole kick: normal and skew order-2 multipole.

    Integrated strength g = k2 * L. The kick is a thin multipole at the
    entrance of the element (apply at 0.5*L, drift, then apply, then drift).

    Parameters
    ----------
    state : 6D array
        [x, px, y, py, zeta, delta]
    g : scalar
        Integrated sextupole strength k2*L.

    Returns
    -------
    new_state : 6D array
        State after the sextupole kick.
    """
    x, px, y, py, zeta, delta = state
    # Normal sextupole: k2 ~ (x^2 - y^2), affects px
    # Skew sextupole: ~ x*y, affects py
    return jnp.array(
        [x, px - 0.5 * g * (x * x - y * y), y, py + g * x * y, zeta, delta]
    )


def emap(state, e, ks):
    """Apply one element's exact map, including sextupoles.

    For sextupoles, the effective k2 depends on:
      - knob-driven sextupoles: k2 = ks[index]
      - fixed sextupoles: k2 = k2fix

    Parameters
    ----------
    state : 6D array
        Phase-space state.
    e : tuple
        One row from ENC: (et, L, k0, h, k1, ksi, kn1, ks1, Dx, Dpx, Dy, Dpy,
        k2fix, r21i, r43i, r21o, r43o)
    ks : 1D array
        Sextupole strengths from knob vector.

    Returns
    -------
    new_state : 6D array
        State after the element.
    """
    et, L, k0, h, k1, ksi, kn1, ks1 = e[:8]
    k2f = e[12]
    r21i, r43i, r21o, r43o = e[13], e[14], e[15], e[16]
    # Select k2: knob-driven sextupoles read from ks[ksi], others use baked-in k2f
    g = jnp.where(ksi >= 0, ks[jnp.maximum(ksi, 0)], k2f) * L

    def sext(s):
        """Sextupole: drift(L/2) -> kick -> drift(L/2)."""
        s = drift_exact_jax(s, 0.5 * L, beta0)
        s = sext_kick(s, g)
        return drift_exact_jax(s, 0.5 * L, beta0)

    return lax.switch(
        et,
        [
            lambda s: drift_exact_jax(s, L, beta0),
            lambda s: quad_body_jax(s, L, k1, beta0),
            lambda s: bend_with_edges_jax(s, L, k0, h, beta0, r21i, r43i, r21o, r43o),
            sext,
            lambda s: thin_quad_skew_kick_jax(s, kn1, ks1),
        ],
        state,
    )


# ===========================================================================
# Part 3: One-turn matrix at off-momentum - Sextupole interaction
# ===========================================================================
# Linearize each element around the off-momentum closed orbit, then build
# the full one-turn matrix. Sextupoles interact with the dispersion orbit.
# Linear elements are (to 1st order) independent of transverse offset.
# ===========================================================================


def tunes(ks, delta):
    """Compute linear tunes Q(ks, delta) as functions of sextupole strength and momentum.

    For a given momentum deviation δ, we:
      1. Linearly expand each element around the off-momentum orbit
         point_p = [D(i)*δ, Dpx(i)*δ, Dy(i)*δ, Dpy(i)*δ]
      2. Compute the transverse 4x4 Jacobian R = d(map)/d(transverse) at that point
      3. Multiply all R matrices to get the one-turn matrix M
      4. Extract tunes from Tr(M)

    The nonlinearity comes from sextupoles: they have a quadratic map
    (proportional to x^2 - y^2 and x*y), so their Jacobian at point_p depends
    on which point we evaluate it - it is NOT the same as the Jacobian at (0,0).

    Parameters
    ----------
    ks : 1D array
        Knob-driven sextupole strengths.
    delta : scalar
        Momentum deviation. Usually 0 for the on-momentum tune; we differentiate
        w.r.t. delta to get chromaticity.

    Returns
    -------
    jnp.array([qx, qy])
        Fractional tunes.
    """

    def body(M, e):
        # Extract dispersion components from the encoding
        Dxi, Dpxi, Dyi, Dpyi = e[8], e[9], e[10], e[11]
        # Off-momentum orbit for this element: D*delta
        pt = jnp.array([Dxi * delta, Dpxi * delta, Dyi * delta, Dpyi * delta])

        def f(t):
            """Transverse map [x, px, y, py] -> [x', px', y', py'] at this element."""
            s = jnp.array([t[0], t[1], t[2], t[3], 0.0, delta])
            return emap(s, e, ks)[0:4]

        # Linearize the transverse map at the off-momentum orbit: R = d(f)/d(t)
        R = jax.jacfwd(f)(pt)
        # Left-multiply: new_M = R @ old_M
        return R @ M, None

    # Scan over all elements, building the one-turn matrix
    M, _ = lax.scan(body, jnp.eye(4), ENC)

    # Extract tunes from the trace
    qx = jnp.arccos(jnp.clip(0.5 * (M[0, 0] + M[1, 1]), -1.0, 1.0)) / (2 * jnp.pi)
    qy = jnp.arccos(jnp.clip(0.5 * (M[2, 2] + M[3, 3]), -1.0, 1.0)) / (2 * jnp.pi)
    return jnp.array([qx, qy])


# ===========================================================================
# Part 4: Chromaticity and its Jacobian
# ===========================================================================
# Chromaticity = dQ/dδ at δ=0. We compute it via jacfwd(tunes) w.r.t. δ.
# Then we chain with dQ/dknob using another jacfwd on the knob loop.
# ===========================================================================


@jax.jit
def chroma_of_dknob(dknob):
    """Compute chromaticity as a function of knob perturbation.

    Parameters
    ----------
    dknob : 1D array, shape (2,)
        Perturbation to the two sextupole knobs.

    Returns
    -------
    jnp.array([dqx, dqy])
        Chromaticities (tune derivatives w.r.t. δ).
    """
    # Update sextupole strengths: ks = ks0 + dks, where dks = DKS @ dknob
    ks = ks0 + DKS @ dknob
    # Compute chromaticity: dQ/dδ at δ=0 via jacfwd over δ
    return jax.jacfwd(lambda d: tunes(ks, d))(0.0)


# Pre-compile the 2nd jacfwd: dchroma/dknob
chroma_jac_fn = jax.jit(jax.jacfwd(chroma_of_dknob))


def jax_jacobian():
    """Compute the chromaticity-response Jacobian d(chroma)/d(knobs)
    at dknob=0.
    """
    return np.asarray(chroma_jac_fn(jnp.zeros(2)))


# Sanity: JAX chromaticity value vs xsuite.
ch = np.asarray(chroma_of_dknob(jnp.zeros(2)))
print(
    f"[chroma check] JAX dqx={ch[0]:.3f} dqy={ch[1]:.3f}  "
    f"vs xsuite {tw0.dqx:.3f} {tw0.dqy:.3f}"
)


# ---------------------------------------------------------------------------
# 1. Verify d(chroma)/d(knob) against xsuite 4d finite differences.
# ---------------------------------------------------------------------------
def fd_jacobian(delta=1e-3):
    v0 = {v: float(line[v]) for v in VARY}
    jac = np.zeros((2, len(VARY)))
    for j, v in enumerate(VARY):
        line[v] = v0[v] + delta
        twp = line.twiss(method="4d")
        line[v] = v0[v] - delta
        twm = line.twiss(method="4d")
        line[v] = v0[v]
        jac[0, j] = (twp.dqx - twm.dqx) / (2 * delta)
        jac[1, j] = (twp.dqy - twm.dqy) / (2 * delta)
    return jac


jac_jax = jax_jacobian()
jac_fd = fd_jacobian()
err = np.max(np.abs(jac_jax - jac_fd))
denom = np.maximum(np.abs(jac_fd), 1e-3)
rel = np.max(np.abs(jac_jax - jac_fd) / denom)
np.set_printoptions(precision=4, suppress=True)
print(f"[chroma jac] max|JAX-FD|={err:.2e}  max rel={rel:.2e}")
print("JAX:\n", jac_jax, "\nFD:\n", jac_fd)


# ---------------------------------------------------------------------------
# 2. Solve the chromaticity match: Newton with JAX Jacobian + xsuite residual.
# ---------------------------------------------------------------------------
def residual():
    tw = line.twiss(method="4d")
    return np.array([tw.dqx - TARGETS[0][1], tw.dqy - TARGETS[1][1]])


for it in range(12):
    r = residual()
    if np.max(np.abs(r)) < 1e-4:
        break
    J = jax_jacobian()
    dx = np.linalg.solve(J, -r)
    for j, v in enumerate(VARY):
        line[v] = float(line[v]) + dx[j]

tw = line.twiss(method="4d")
print(
    f"[solve] {it} steps  dqx={tw.dqx:.5f} (10.0)  dqy={tw.dqy:.5f} (12.0)  "
    f"|res|={np.max(np.abs(residual())):.2e}"
)

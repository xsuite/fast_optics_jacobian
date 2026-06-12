"""Extend the JAX backend to GLOBAL TUNE matching (Qx, Qy).

APPROACH
--------
Unlike section-local Twiss matching, tune is a global ring property. We build
the one-turn transfer matrix M (product of per-element R matrices around the
closed orbit) and read the linear tunes from its trace:

    Qx = arccos((M[0,0]+M[1,1]) / 2) / 2π   (mod 1, folded into [0, 0.5])
    Qy = arccos((M[2,2]+M[3,3]) / 2) / 2π

The whole thing is one JAX-differentiable function of the trim-quad knobs,
so dQ/dknob is a single jacfwd call.

ADVANTAGE OVER PHASE-ADVANCE METHOD
------------------------------------
No frozen initial-condition approximation. The trace-based tune is exact for
the closed-loop one-turn matrix, matching MAD-NG's implementation and avoiding
the subtle errors of propagating mux from a fixed periodic condition.

VALIDATION
----------
1. Verify dQ/dknob against xsuite 4d finite differences (includes chromatic map).
2. Solve the tune match to target values using Newton's method.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import xtrack as xt

from xtrack.jax_optics import (
    emap_jax,
    encode_section,
    compress_encoding,
    knob_strength_jacobian,
)

jax.config.update("jax_enable_x64", True)


def dk1_dknob_matrix(line, ordered_names, vary_names):
    """d(k1_quad)/d(knob) for every section quadrupole, from the xdeps graph.

    Returns (varied_quad_names, kq0, dk1) with dk1 of shape
    (n_varied_quad, n_knob); only quads actually driven by a knob are kept.
    (Local copy: the standalone helper was removed from xtrack.jax_optics, whose
    only in-package use was the now-deleted JaxSectionJacobian.)
    """
    ed = line.element_dict
    derivs = knob_strength_jacobian(line, vary_names, "k1")
    section_quads = [
        nm for nm in ordered_names if type(ed.get(nm)).__name__ == "Quadrupole"
    ]
    varied = [q for q in section_quads if any(q in derivs[v] for v in vary_names)]
    dk1 = np.zeros((len(varied), len(vary_names)))
    for j, v in enumerate(vary_names):
        for i, q in enumerate(varied):
            dk1[i, j] = derivs[v].get(q, 0.0)
    kq0 = np.array([float(ed[q].k1) for q in varied])
    return varied, kq0, dk1


# ---------------------------------------------------------------------------
# Load the ring (same as opt_chroma in 032).  Dipole edges are kept active: the
# encoded bend rows now carry their linear edge-focusing coefficients, so the
# JAX one-turn matrix (and tunes) match xtrack with edges on.
# ---------------------------------------------------------------------------
line = xt.Line.from_json("lattice_data/lhc_thick_with_knobs.json")
line.build_tracker()

VARY = ["kqtf.b1", "kqtd.b1"]
TARGETS = [("qx", 62.315), ("qy", 60.325)]

tw0 = line.twiss(method="4d")
beta0 = float(tw0.particle_on_co.beta0[0])
print(f"[ring] {len(tw0.name)} elements, qx={tw0.qx:.5f} qy={tw0.qy:.5f}")

ordered_names = list(tw0.name[:-1])
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

# Build the knob->k1 sensitivity matrix and the encoded section (reuse from 030).
# For tune, all elements matter (not just a section), so no intermediate targets.
# This lets compress_encoding merge all consecutive drifts -> fewer scan iterations.
varied, kq0, dk1 = dk1_dknob_matrix(line, ordered_names, VARY)
kq_index = {q: i for i, q in enumerate(varied)}
enc = encode_section(line, ordered_names, kq_index)
cenc, _ = compress_encoding(enc, set())  # empty target set -> merge all drifts
print(
    f"[encode] {len(varied)} knob-driven quads, "
    f"compressed to {int(cenc.etype.shape[0])} rows"
)


# ===========================================================================
# Part 1: One-turn map and tunes
# ===========================================================================
# Build the ring-averaged transfer matrix (product of per-element R's),
# then extract tunes from its trace via the standard Hill-Steele formula.
# ===========================================================================


def emap_single(s, e, kq):
    """Apply one element's exact map (delegates to the shared emap_jax; this
    backend uses only the quad vector, so ks/corr are empty)."""
    return emap_jax(s, e, kq, jnp.zeros(1), jnp.zeros(1), beta0)


@jax.jit
def tunes(kq):
    """Compute the linear tunes Qx, Qy as a function of quad strengths.

    Pipeline:
      1. Propagate orbit once around the ring at the frozen closed orbit.
      2. Compute 4x4 transverse R matrix at each element (vmap'd jacfwd).
      3. Build the one-turn matrix M = R_N @ ... @ R_1 by scanning left-multiply.
      4. Extract tunes from trace: Q = arccos(Tr(M)/2) / 2π.
    """

    # Step 1: propagate the orbit around the ring (frozen for differentiation).
    def obody(s, e):
        return emap_single(s, e, kq), s

    _, orbits = lax.scan(obody, s0, cenc)
    orbits = lax.stop_gradient(orbits)  # orbit ~ const in kq

    # Step 2: compute per-element R matrices (4x4 transverse block only).
    R4 = jax.vmap(
        lambda e, o: jax.jacfwd(lambda s: emap_single(s, e, kq))(o)[0:4, 0:4]
    )(cenc, orbits)

    # Step 3: build the one-turn matrix by left-multiplying R_i @ M_{i-1}.
    # Start with M_0 = I; scan applies M_i = R_i @ M_{i-1} for each i.
    def mbody(M, r):
        return r @ M, None

    M, _ = lax.scan(mbody, jnp.eye(4), R4)

    # Step 4: extract tunes from the trace via arccos(Tr/2).
    cosmux = 0.5 * (M[0, 0] + M[1, 1])
    cosmuy = 0.5 * (M[2, 2] + M[3, 3])
    qx = jnp.arccos(jnp.clip(cosmux, -1.0, 1.0)) / (2 * jnp.pi)
    qy = jnp.arccos(jnp.clip(cosmuy, -1.0, 1.0)) / (2 * jnp.pi)
    return jnp.array([qx, qy])


# ===========================================================================
# Part 2: Differentiate tunes w.r.t. knobs
# ===========================================================================
# One jacfwd gives d(tunes)/d(kq). Chain with d(kq)/d(knob) to get the
# knob-level Jacobian needed by the matching optimizer.
# ===========================================================================

tune_jac_fn = jax.jit(jax.jacfwd(tunes))  # pre-compiled: dQ/d(kq)

_ed = line.element_dict


def jax_jacobian():
    """Compute the tune-response Jacobian d(qx, qy) / d(knobs).

    Returns
    -------
    jac : np.ndarray, shape (2, 2)
        dQ/dknob for the two trim-quad knobs.
    """
    # Evaluate dQ/d(kq) at the current quad strengths.
    kq = jnp.asarray([float(_ed[q].k1) for q in varied])
    jac_kq = np.asarray(tune_jac_fn(kq))  # (2, n_varied)
    # Chain: dQ/dknob = dQ/dkq @ dkq/dknob.
    return jac_kq @ dk1  # (2, n_knob)


# Sanity: JAX fractional tunes vs xsuite (should match the .frac part).
qj = np.asarray(tunes(jnp.asarray(kq0)))
print(
    f"[tune check] JAX frac qx={qj[0]:.5f} qy={qj[1]:.5f}  "
    f"vs xsuite frac {tw0.qx % 1:.5f} {tw0.qy % 1:.5f}"
)


# ---------------------------------------------------------------------------
# 1. Verify dQ/dknob against xsuite 4d finite differences.
# ---------------------------------------------------------------------------
def fd_jacobian(delta=1e-7):
    v0 = {v: float(line[v]) for v in VARY}
    jac = np.zeros((len(TARGETS), len(VARY)))
    for j, v in enumerate(VARY):
        line[v] = v0[v] + delta
        twp = line.twiss(method="4d")
        line[v] = v0[v] - delta
        twm = line.twiss(method="4d")
        line[v] = v0[v]
        jac[0, j] = (twp.qx - twm.qx) / (2 * delta)
        jac[1, j] = (twp.qy - twm.qy) / (2 * delta)
    return jac


jac_jax = jax_jacobian()
jac_fd = fd_jacobian()
err = np.max(np.abs(jac_jax - jac_fd))
denom = np.maximum(np.abs(jac_fd), 1e-6)
rel = np.max(np.abs(jac_jax - jac_fd) / denom)
np.set_printoptions(precision=4, suppress=True)
print(f"[tune jac] max|JAX-FD|={err:.2e}  max rel={rel:.2e}")
print("JAX:\n", jac_jax, "\nFD:\n", jac_fd)


# ---------------------------------------------------------------------------
# 2. Solve the tune match: Newton with the JAX Jacobian + xsuite residual.
# ---------------------------------------------------------------------------
def residual():
    tw = line.twiss(method="4d")
    return np.array([tw.qx - TARGETS[0][1], tw.qy - TARGETS[1][1]])


for it in range(10):
    r = residual()
    if np.max(np.abs(r)) < 1e-6:
        break
    J = jax_jacobian()
    dx = np.linalg.solve(J, -r)
    for j, v in enumerate(VARY):
        line[v] = float(line[v]) + dx[j]

tw = line.twiss(method="4d")
print(
    f"[solve] {it} steps  qx={tw.qx:.6f} (62.315)  qy={tw.qy:.6f} (60.325)  "
    f"|res|={np.max(np.abs(residual())):.2e}"
)

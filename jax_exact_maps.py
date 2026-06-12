"""Exact-physics JAX element maps + linear-optics (Twiss) propagation.

Building blocks shared used in the LHC optics work (021, 030) prototypes.
The single-element maps are ports of the tracking code in
``xtrack/xtrack/beam_elements/elements_src`` (NOT the linear
transfer-matrix approximation from the first examples 002/003); the transfer matrix R
used for the optics is obtained by differentiating these exact maps around the
actual orbit (``jax.jacfwd`` over the state), so the optics are differentiated
through the same physics xtrack tracks.

State layout everywhere is the xsuite 6-vector ``[x, px, y, py, zeta, delta]``.
"""

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Single-element exact maps (ports of xtrack/beam_elements)
# ---------------------------------------------------------------------------


def rvv_from_delta(delta, beta0):
    """rvv = beta / beta0 as a function of delta (see particles.py)."""
    one_plus_delta = 1.0 + delta
    denom = jnp.sqrt(
        beta0 * beta0 * one_plus_delta * one_plus_delta + 1.0 - beta0 * beta0
    )
    return one_plus_delta / denom


def _cs(K, length):
    """Closed form of the focusing block, valid for any sign of K.

        K > 0 : C = cos(sqrt(K) L),    S = sin(sqrt(K) L) / sqrt(K)
        K < 0 : C = cosh(sqrt(-K) L),  S = sinh(sqrt(-K) L) / sqrt(-K)
        K = 0 : C = 1,                 S = L

    Safe denominators keep it autodiff-stable through K = 0.
    """
    absK = jnp.abs(K)
    is_zero = absK <= 0.0
    # Double-where so neither the value nor the (forward/reverse) gradient sees
    # sqrt(0): at K = 0 the sqrt is fed a safe 1.0 and r is overridden to 0.
    safe_absK = jnp.where(is_zero, 1.0, absK)
    r = jnp.where(is_zero, 0.0, jnp.sqrt(safe_absK))
    rl = r * length
    safe_r = jnp.where(is_zero, 1.0, r)
    C = jnp.where(K > 0, jnp.cos(rl), jnp.cosh(rl))
    S = jnp.where(
        is_zero,
        length,
        jnp.where(K > 0, jnp.sin(rl) / safe_r, jnp.sinh(rl) / safe_r),
    )
    return C, S


def drift_exact_jax(state, length, beta0):
    """Exact drift - port of Drift_single_particle_exact (track_drift.h)."""
    x, px, y, py, zeta, delta = (
        state[0],
        state[1],
        state[2],
        state[3],
        state[4],
        state[5],
    )
    one_plus_delta = 1.0 + delta
    one_over_pz = 1.0 / jnp.sqrt(one_plus_delta * one_plus_delta - px * px - py * py)
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


def quad_combined_map_jax(state, length, k0_, k1_, h, beta0):
    """Expanded combined dipole-quad map.

    Port of track_expanded_combined_dipole_quad_single_particle
    (track_magnet_drift.h, chi = 1). For a pure quad pass k0_ = h = 0.
    """
    x, px, y, py, zeta, delta = (
        state[0],
        state[1],
        state[2],
        state[3],
        state[4],
        state[5],
    )
    rvv = rvv_from_delta(delta, beta0)
    dp1 = 1.0 + delta

    k0 = k0_ / dp1
    k1 = k1_ / dp1
    Kx = k0 * h + k1
    Ky = -k1
    Cx, Sx = _cs(Kx, length)
    Cy, Sy = _cs(Ky, length)

    xp = px / dp1
    yp = py / dp1
    A = -Kx * x - k0 + h
    B = xp
    Cq = -Ky * y
    Dq = yp

    x_ = x * Cx + xp * Sx
    y_ = y * Cy + yp * Sy
    px_ = (A * Sx + B * Cx) * dp1
    py_ = (Cq * Sy + Dq * Cy) * dp1

    tol = 1e-9
    Kx_nz = jnp.abs(Kx) > tol
    Ky_nz = jnp.abs(Ky) > tol
    Kx_safe = jnp.where(Kx_nz, Kx, 1.0)
    Ky_safe = jnp.where(Ky_nz, Ky, 1.0)

    x_ = jnp.where(
        Kx_nz,
        x_ + (k0 - h) * (Cx - 1.0) / Kx_safe,
        x_ - (k0 - h) * 0.5 * length**2,
    )

    length_ = length
    term_x_nz = -(
        h * ((Cx - 1.0) * xp + Sx * A + length * (k0 - h))
    ) / Kx_safe + 0.5 * (
        -(A**2 * Cx * Sx) / (2.0 * Kx_safe)
        + (B**2 * Cx * Sx) / 2.0
        + (A**2 * length) / (2.0 * Kx_safe)
        + (B**2 * length) / 2.0
        - (A * B * Cx**2) / Kx_safe
        + (A * B) / Kx_safe
    )
    term_x_z = (
        h * length * (3.0 * length * xp + 6.0 * x - (k0 - h) * length**2) / 6.0
        + 0.5 * B**2 * length
    )
    length_ = length_ + jnp.where(Kx_nz, term_x_nz, term_x_z)

    term_y_nz = 0.5 * (
        -(Cq**2 * Cy * Sy) / (2.0 * Ky_safe)
        + (Dq**2 * Cy * Sy) / 2.0
        + (Cq**2 * length) / (2.0 * Ky_safe)
        + (Dq**2 * length) / 2.0
        - (Cq * Dq * Cy**2) / Ky_safe
        + (Cq * Dq) / Ky_safe
    )
    term_y_z = 0.5 * Dq**2 * length
    length_ = length_ + jnp.where(Ky_nz, term_y_nz, term_y_z)

    dzeta = length - length_ / rvv
    return jnp.array([x_, px_, y_, py_, zeta + dzeta, delta])


def curved_bend_map_jax(state, length, k0, h, beta0):
    """Exact curved sector-bend map (constant k0, h; assumes k0, h != 0).

    Port of track_curved_exact_bend_single_particle (track_magnet_drift.h).
    """
    x, px, y, py, zeta, delta = (
        state[0],
        state[1],
        state[2],
        state[3],
        state[4],
        state[5],
    )
    rvv = rvv_from_delta(delta, beta0)
    dp1 = 1.0 + delta
    s = length
    k0_chi = k0

    A = 1.0 / jnp.sqrt(dp1**2 - py**2)
    pz = jnp.sqrt(dp1**2 - px**2 - py**2)

    Cc = pz - k0_chi * (1.0 / h + x)
    new_px = px * jnp.cos(s * h) + Cc * jnp.sin(s * h)
    new_pz = jnp.sqrt(dp1**2 - new_px**2 - py**2)
    d_new_px_ds = Cc * h * jnp.cos(h * s) - h * px * jnp.sin(h * s)

    new_x = (new_pz * h - d_new_px_ds - k0_chi) / (h * k0_chi)
    D = jnp.arcsin(A * px) - jnp.arcsin(A * new_px)
    new_y = y + (py * s) / (k0_chi / h) + (py / k0_chi) * D
    delta_ell = (dp1 * s * h) / k0_chi + (dp1 / k0_chi) * D

    return jnp.array(
        [new_x, new_px, new_y, py, zeta + (length - delta_ell / rvv), delta]
    )


def thin_multipole_kick_jax(state, knl, ksl):
    """Thin multipolar kick from integrated normal/skew strengths knl/ksl.

    Port of kick_simple_single_coordinates (track_magnet_kick.h, chi = 1):
    builds dpx + i dpy = -sum_n (knl_n + i ksl_n)/n! * (x + i y)^n by Horner.
    knl/ksl are 1-D arrays of the same length (order+1).
    """
    x, px, y, py, zeta, delta = (
        state[0],
        state[1],
        state[2],
        state[3],
        state[4],
        state[5],
    )
    order = knl.shape[0] - 1
    # Horner from the highest order down (complex (x+iy) powers).
    inv_fact = 1.0
    for n in range(2, order + 1):
        inv_fact = inv_fact / n  # 1/order! at the top
    dpx = knl[order] * inv_fact
    dpy = ksl[order] * inv_fact
    idx = order
    while idx > 0:
        zre = dpx * x - dpy * y
        zim = dpx * y + dpy * x
        inv_fact = inv_fact * idx
        idx -= 1
        dpx = knl[idx] * inv_fact + zre
        dpy = ksl[idx] * inv_fact + zim
    return jnp.array([x, px - dpx, y, py + dpy, zeta, delta])


def thin_quad_kick_jax(state, k1l):
    """Thin quadrupole kick of integrated strength k1l (chi = 1, h = 0).

    Taken from track_magnet_kick.h with dpx = -k1l x, dpy = +k1l y.
    """
    x, px, y, py, zeta, delta = (
        state[0],
        state[1],
        state[2],
        state[3],
        state[4],
        state[5],
    )
    return jnp.array([x, px - k1l * x, y, py + k1l * y, zeta, delta])


def thin_quad_skew_kick_jax(state, knl1, ksl1):
    """Linear (n=1) part of a thin multipole kick: integrated normal knl1 and
    skew ksl1 quadrupole components.  This is the only part of a thin multipole
    that affects the linear optics at zero orbit (dipole/sextupole/... no
    first-order focusing there).  Derived from thin_multipole_kick_jax."""
    x, px, y, py, zeta, delta = (
        state[0],
        state[1],
        state[2],
        state[3],
        state[4],
        state[5],
    )
    dpx = knl1 * x - ksl1 * y
    dpy = knl1 * y + ksl1 * x
    return jnp.array([x, px - dpx, y, py + dpy, zeta, delta])


def quad_body_jax(state, length, k1, beta0, model="mat-kick-mat", nslice=1):
    """Quadrupole body for the xtrack model (uniform integrator).

    'mat-kick-mat'          : two half combined-maps (kick does nothing for
                              a pure quad).  This is the xsuite default.
    'drift-kick-drift-exact': nslice * [exact drift, thin kick, exact drift];
                              converges to MAD-NG as nslice grows.
    """
    if model == "mat-kick-mat":
        half = 0.5 * length
        state = quad_combined_map_jax(state, half, 0.0, k1, 0.0, beta0)
        state = quad_combined_map_jax(state, half, 0.0, k1, 0.0, beta0)
        return state
    if model == "drift-kick-drift-exact":
        ds = length / nslice
        for _ in range(nslice):
            state = drift_exact_jax(state, 0.5 * ds, beta0)
            state = thin_quad_kick_jax(state, k1 * ds)
            state = drift_exact_jax(state, 0.5 * ds, beta0)
        return state
    raise ValueError(f"Unsupported quad model: {model}")


def bend_body_jax(state, length, k0, h, beta0):
    """Pure-dipole body for model 'bend-kick-bend' (two half curved maps)."""
    half = 0.5 * length
    state = curved_bend_map_jax(state, half, k0, h, beta0)
    state = curved_bend_map_jax(state, half, k0, h, beta0)
    return state


def dipole_edge_linear_jax(state, r21, r43):
    """Linear dipole-edge kick: px += r21 x, py += r43 y.

    Port of DipoleEdgeLinear_single_particle (track_dipole_edge_linear.h).
    ``r21, r43`` are precomputed by ``bend_edge_coeffs`` and are constant during
    a quad match (they depend only on fixed bend strength/geometry), so jacfwd
    folds this linear kick straight into the transfer matrix R.
    """
    x, px, y, py, zeta, delta = (
        state[0],
        state[1],
        state[2],
        state[3],
        state[4],
        state[5],
    )
    return jnp.array([x, px + r21 * x, y, py + r43 * y, zeta, delta])


def bend_with_edges_jax(state, length, k0, h, beta0, r21_in, r43_in, r21_out, r43_out):
    """Curved-dipole body with entry/exit edge effects.

    Inactive/zero faces pass r21 = r43 = 0, so this is the code for
    both main ``Bend`` faces (e1 = e2 = 0) and ``RBend`` faces (effective face
    angle = h*L/2).
    """
    state = dipole_edge_linear_jax(state, r21_in, r43_in)
    state = bend_body_jax(state, length, k0, h, beta0)
    state = dipole_edge_linear_jax(state, r21_out, r43_out)
    return state


def bend_edge_coeffs(e):
    """Linear dipole-edge coefficients for a Bend/RBend's two faces.

    Python-side port of ``compute_dipole_edge_linear_coefficients`` plus the
    rbend curved-body face-angle augmentation (effective face angle = attribute
    + ``(h*L -/+ rbend_angle_diff)/2``). Returns
    ``(r21_in, r43_in, r21_out, r43_out)``; an inactive face contributes
    ``(0, 0)``.  Only the linear edge model is supported.
    """
    cls = type(e).__name__
    if cls not in ("Bend", "RBend"):
        return 0.0, 0.0, 0.0, 0.0
    k0 = float(e.k0)
    angle = float(e.h) * float(e.length)
    is_rbend = cls == "RBend"
    if is_rbend and str(getattr(e, "rbend_model", "adaptive")) == "straight":
        raise NotImplementedError(
            "rbend straight-body edges are not supported "
            f"(element {getattr(e, 'name', '?')!r})"
        )
    angle_diff = float(getattr(e, "rbend_angle_diff", 0.0)) if is_rbend else 0.0
    aug_in = (angle - angle_diff) / 2.0 if is_rbend else 0.0
    aug_out = (angle + angle_diff) / 2.0 if is_rbend else 0.0

    def _face(active, model, e_ang, e_fd, fint, hgap, aug):
        if not int(active):
            return 0.0, 0.0
        if str(model) != "linear":
            raise NotImplementedError(
                f"only the linear dipole-edge model is supported, got "
                f"{model!r} on element {getattr(e, 'name', '?')!r}"
            )
        e1 = float(e_ang) + aug
        r21 = k0 * np.tan(e1)
        e1v = e1 + float(e_fd)
        corr = 2.0 * k0 * float(hgap) * float(fint)
        temp = corr / np.cos(e1v) * (1.0 + np.sin(e1v) ** 2)
        r43 = -k0 * np.tan(e1v - temp)
        return r21, r43

    r21i, r43i = _face(
        e.edge_entry_active,
        e.edge_entry_model,
        e.edge_entry_angle,
        e.edge_entry_angle_fdown,
        e.edge_entry_fint,
        e.edge_entry_hgap,
        aug_in,
    )
    r21o, r43o = _face(
        e.edge_exit_active,
        e.edge_exit_model,
        e.edge_exit_angle,
        e.edge_exit_angle_fdown,
        e.edge_exit_fint,
        e.edge_exit_hgap,
        aug_out,
    )
    return r21i, r43i, r21o, r43o


# ---------------------------------------------------------------------------
# Linear optics (Twiss) propagation
# ---------------------------------------------------------------------------

TW_LABELS = ["betx", "alfx", "mux", "bety", "alfy", "muy", "dx", "dpx", "dy", "dpy"]
TW_INDEX = {name: i for i, name in enumerate(TW_LABELS)}


def propagate_twiss(R, params):
    """Propagate Twiss params through one transfer matrix R.

    Port from previous implementation. ``params`` follows TW_LABELS; phase
    advances accumulate (arctan2 per segment), so applying this element by
    element stays correct over many betatron oscillations.
    """
    betx0, alfx0, mux0, bety0, alfy0, muy0, dx0, dpx0, dy0, dpy0 = [
        params[i] for i in range(10)
    ]

    r00, r01, r10, r11 = R[0, 0], R[0, 1], R[1, 0], R[1, 1]
    tmp_x = r00 * betx0 - r01 * alfx0
    betx = (tmp_x**2 + r01**2) / betx0
    alfx = -((tmp_x * (r10 * betx0 - r11 * alfx0) + r01 * r11) / betx0)
    mux = mux0 + jnp.arctan2(r01, tmp_x) / (2 * jnp.pi)

    r22, r23, r32, r33 = R[2, 2], R[2, 3], R[3, 2], R[3, 3]
    tmp_y = r22 * bety0 - r23 * alfy0
    bety = (tmp_y**2 + r23**2) / bety0
    alfy = -((tmp_y * (r32 * bety0 - r33 * alfy0) + r23 * r33) / bety0)
    muy = muy0 + jnp.arctan2(r23, tmp_y) / (2 * jnp.pi)

    dx = r00 * dx0 + r01 * dpx0 + R[0, 5]
    dpx = r10 * dx0 + r11 * dpx0 + R[1, 5]
    dy = r22 * dy0 + r23 * dpy0 + R[2, 5]
    dpy = r32 * dy0 + r33 * dpy0 + R[3, 5]

    return jnp.array([betx, alfx, mux, bety, alfy, muy, dx, dpx, dy, dpy])


def jax_twiss(element_maps, knobs, params0, init_state, return_boundaries=False):
    """Propagate Twiss params through a list of (name, map_fn) elements.

    ``map_fn(state, knobs) -> state`` is one of the exact maps above.  At each
    element the local transfer matrix is R = d(map)/d(state) at the incoming
    orbit (jax.jacfwd), i.e. the exact linearization around the real
    trajectory.  Returns the final Twiss params (and optionally a dict of
    params at every element boundary).
    """
    state = init_state
    params = params0
    boundaries = {}
    for name, emap in element_maps:
        R = jax.jacfwd(lambda s, emap=emap: emap(s, knobs))(state)
        params = propagate_twiss(R, params)
        state = emap(state, knobs)
        boundaries[name] = params
    if return_boundaries:
        return params, boundaries
    return params

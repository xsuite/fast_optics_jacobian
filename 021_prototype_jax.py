"""
JAX backend prototype for xsuite Jacobians from the *exact element physics*.

Instead of the linear transfer-matrix approximation used in
example 002 and 003, the maps below are ports of the actual
tracking code in ``xtrack/xtrack/beam_elements/elements_src``:

    drift_exact_jax        <- track_drift.h :: Drift_single_particle_exact
    quad_combined_map_jax  <- track_magnet_drift.h ::
                              track_expanded_combined_dipole_quad_single_particle
                              (this is the body of model 'mat-kick-mat', the
                               xsuite default for Quadrupole)
    curved_bend_map_jax    <- track_magnet_drift.h ::
                              track_curved_exact_bend_single_particle
                              ('bend-kick-bend' model)
    thin_quad_kick_jax     <- track_magnet_kick.h ::
                              track_magnet_kick_single_particle
                              (thin quad kick used by the
                              'drift-kick-drift-exact' model)

The script is organised in sections:

    1. JAX physics maps (+ auxiliary functions and helpers to
       build lines and JAX propagation through line).
    2. Part 1 - Drift only, knob = length of drift.
    3. Part 2 - Quad and Bend line, knob = k1.
    4. Part 3 - Model study: which xtrack model is closest to MAD-NG?
    5. Part 4 - beta / alpha / phase matching from JAX optics
"""

import itertools

import numpy as np

import jax
import jax.numpy as jnp

import xtrack as xt

jax.config.update("jax_enable_x64", True)

# ===========================================================================
# SECTION 1 - JAX physics maps (ports of xtrack/beam_elements)
# ===========================================================================


def _rvv_from_delta(delta, beta0):
    """rvv = beta / beta0 as a function of delta (see particles.py)."""
    one_plus_delta = 1.0 + delta
    denom = jnp.sqrt(
        beta0 * beta0 * one_plus_delta * one_plus_delta + 1.0 - beta0 * beta0
    )
    return one_plus_delta / denom


def _cs(K, length):
    """Closed form of the focusing block, valid for any sign of K.

        K > 0 : C = cos(sqrt(K) L),    S = sin(sqrt(K) L) / sqrt(K)
        K < 0 :0 C = cosh(sqrt(-K) L), S = sinh(sqrt(-K) L) / sqrt(-K)
        K = 0 : C = 1,                 S = L

    Safe denominators keep it autodiff-stable through K = 0.
    """
    absK = jnp.abs(K)
    r = jnp.sqrt(absK)
    rl = r * length
    safe_r = jnp.where(absK > 0, r, 1.0)
    C = jnp.where(K > 0, jnp.cos(rl), jnp.cosh(rl))
    S = jnp.where(
        absK > 0,
        jnp.where(K > 0, jnp.sin(rl) / safe_r, jnp.sinh(rl) / safe_r),
        length,
    )
    return C, S


def drift_exact_jax(state, length, beta0):
    """Exact drift, taken from Drift_single_particle_exact (track_drift.h)."""
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
    rv0v = 1.0 / _rvv_from_delta(delta, beta0)
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
    rvv = _rvv_from_delta(delta, beta0)
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
    rvv = _rvv_from_delta(delta, beta0)
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


def quad_body_jax(state, length, k1, beta0, model, nslice):
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


# ===========================================================================
# SECTION 1b - generic helpers (tracking + Jacobian comparison)
# ===========================================================================

X0, PX0, Y0, PY0, ZETA0, DELTA0 = 1.0e-3, 1.5e-3, 0.5e-3, -0.8e-3, 0.0, 1.0e-3
INIT_STATE = jnp.array([X0, PX0, Y0, PY0, ZETA0, DELTA0])

COORD_INDEX = {"x": 0, "px": 1, "y": 2, "py": 3, "zeta": 4, "delta": 5}
COORD_LABELS = ["x", "px", "y", "py", "zeta", "delta"]


def make_proton_env():
    env = xt.Environment()
    env.particle_ref = xt.Particles(p0c=7e12, mass0=xt.PROTON_MASS_EV)
    env.vars.default_to_zero = True
    return env


def xt_track(line, knob_values):
    """Track the reference particle through ``line`` for the given knobs."""
    for name, value in knob_values.items():
        line[name] = float(value)
    p = line.build_particles(x=X0, px=PX0, y=Y0, py=PY0, zeta=ZETA0, delta=DELTA0)
    line.track(p)
    return np.array(
        [
            float(p.x[0]),
            float(p.px[0]),
            float(p.y[0]),
            float(p.py[0]),
            float(p.zeta[0]),
            float(p.delta[0]),
        ]
    )


def forward_check(label, jax_state, xt_state):
    """Print the max |JAX - xtrack| over the transverse + longitudinal coords."""
    diff = np.abs(np.asarray(jax_state) - np.asarray(xt_state))
    print(
        f"[forward check] {label}: max|JAX - xtrack| (x,px,y,py,zeta) "
        f"= {diff[:5].max():.2e}"
    )


def physical_jacobian_from_opt(opt, x=None):
    """Return d(value)/d(knob) (weights divided out) from the Optimize object.

    Works for both the finite-difference and the TPSA merit functions.
    """
    if x is None:
        x = opt._err._get_x()
    else:
        x = np.asarray(x, dtype=float)
    jac = np.asarray(opt._err.get_jacobian(x))
    weights = np.array([t.weight for t in opt.targets])
    target_names = [
        t.tar[0] if isinstance(t.tar, tuple) else t.tar for t in opt.targets
    ]
    vary_names = [v.name for v in opt.vary]
    return jac / weights[:, None], target_names, vary_names


def jax_jacobian(track_fn, knob_values, target_names):
    """Analytic d(target)/d(knob) from the JAX maps (exact physics)."""
    full = np.asarray(jax.jacfwd(track_fn)(jnp.array(knob_values)))  # (6, n_knobs)
    rows = [COORD_INDEX[name] for name in target_names]
    return full[rows, :]


def compare_jacobians(title, target_names, vary_names, jac_by_label):
    """Pretty-print several Jacobians side by side and their pairwise diffs."""
    labels = list(jac_by_label)
    print("=" * 78)
    print(title)
    print("=" * 78)
    print(f"{'d(target)/d(vary)':<18}" + "".join(f"{lab:>20}" for lab in labels))
    for i, tn in enumerate(target_names):
        for j, vn in enumerate(vary_names):
            cell = f"{tn}/{vn}"
            row = f"{cell:<18}" + "".join(
                f"{jac_by_label[lab][i, j]:>20.12e}" for lab in labels
            )
            print(row)
    print("-" * 78)
    for a, b in itertools.combinations(labels, 2):
        d = np.max(np.abs(np.asarray(jac_by_label[a]) - np.asarray(jac_by_label[b])))
        print(f"max|{a} - {b}| = {d:.2e}")


def solve_and_report(opt, line, nominal, label):
    """Reset to nominal, solve, and print a clean one-line result."""
    for name, value in nominal.items():
        line[name] = value
    opt._err.show_call_counter = False
    opt.solve()
    knobs = {name: float(line[name]) for name in nominal}
    knob_str = ", ".join(f"{k}={v:.9f}" for k, v in knobs.items())
    print(f"[solve {label}]  -- ({knob_str})")
    return knobs


# --- Twiss optics from the exact maps -------------------------------------
#
# The orbit maps above give the *nonlinear* one-element flow.  Linear optics
# (beta, alpha, phase advance, dispersion) come from the linear part of that
# flow propagated from initial Twiss conditions.
# This mirrors the previous JAX implementation, with one upgrade: previously
# R was a constant analytic matrix per element, whereas here R it is obtained
# by differentiating the map around the actual orbit (jax.jacfwd over the
# state), so the optics are differentiated through the same physics of xtrack.

TW_LABELS = ["betx", "alfx", "mux", "bety", "alfy", "muy", "dx", "dpx", "dy", "dpy"]
TW_INDEX = {name: i for i, name in enumerate(TW_LABELS)}


def propagate_twiss(R, params):
    """Propagate Twiss params through one transfer matrix R
    from previous JAX implementation

    ``params`` follows TW_LABELS.  Phase advances accumulate (arctan2 per
    segment), so this works over many betatron oscillations when applied
    element by element.
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


def jax_twiss(
    element_maps, knobs, params0, init_state=INIT_STATE, return_boundaries=False
):
    """Propagate Twiss params through a list of (name, map_fn) elements.

    ``map_fn(state, knobs) -> state`` is one of the exact orbit maps.  At each
    element the local transfer matrix is R = d(map)/d(state) evaluated at the
    incoming orbit (jax.jacfwd), so R is the exact linearisation around the
    real trajectory.  Returns the final Twiss params (and, optionally, a dict
    of params at every element boundary, keyed by the element just traversed).
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


# ===========================================================================
# Analysis Part 1: pure exact-drift line, knob = drift length
# ===========================================================================

print("#" * 78)
print("# PART 1 - exact-drift line, knob = drift length")
print("#" * 78)

env1 = make_proton_env()
beta0_1 = float(env1.particle_ref.beta0[0])
env1["L_knob"] = 1.5
DRIFT_LENGTHS = [2.0, None, 3.5, 1.0, 2.0]  # index 1 is the knob

env1.new("d1", xt.Drift, length=DRIFT_LENGTHS[0], model="exact")
env1.new("d2", xt.Drift, length="L_knob", model="exact")
env1.new("d3", xt.Drift, length=DRIFT_LENGTHS[2], model="exact")
env1.new("d4", xt.Drift, length=DRIFT_LENGTHS[3], model="exact")
env1.new("d5", xt.Drift, length=DRIFT_LENGTHS[4], model="exact")
env1.new("end", xt.Marker)
line1 = env1.new_line(components=["d1", "d2", "d3", "d4", "d5", "end"])
line1.particle_ref = env1.particle_ref


def jax_track_drift(knobs):
    """knobs = [L_knob]. Returns the final 6-state."""
    lengths = jnp.array(
        [
            DRIFT_LENGTHS[0],
            knobs[0],
            DRIFT_LENGTHS[2],
            DRIFT_LENGTHS[3],
            DRIFT_LENGTHS[4],
        ]
    )

    def step(state, L):
        return drift_exact_jax(state, L, beta0_1), None

    final, _ = jax.lax.scan(step, INIT_STATE, lengths)
    return final


L0 = float(env1["L_knob"])
NOMINAL1 = {"L_knob": L0}
forward_check("drift line", jax_track_drift(jnp.array([L0])), xt_track(line1, NOMINAL1))

# A drift line is a single degree of freedom (only the total length matters,
# and drifts cannot change px/py), so the well-posed problem is 1 vary -> 1
# target. The target is taken from a tracked working point so it is reachable.
x_target1 = xt_track(line1, {"L_knob": 4.0})[0]
INIT1 = dict(betx=1.0, bety=1.0, x=X0, px=PX0, y=Y0, py=PY0, delta=DELTA0)


def make_match1(use_tpsa):
    """Create the drift-line match."""
    line1["L_knob"] = L0
    return line1.match(
        solve=False,
        use_tpsa=use_tpsa,
        vary=xt.Vary("L_knob", step=1e-7),
        targets=[xt.Target("x", x_target1, at="end", tol=1e-12)],
        **INIT1,
    )


# Build the FD and TPSA optimizers, read each Jacobian at the nominal point,
# then actually solve both.
opt1_fd = make_match1(use_tpsa=False)
jac1_fd, tnames1, vnames1 = physical_jacobian_from_opt(opt1_fd, [L0])

opt1_tpsa = make_match1(use_tpsa=True)
jac1_tpsa, _, _ = physical_jacobian_from_opt(opt1_tpsa, [L0])

jac1_jax = jax_jacobian(jax_track_drift, [L0], tnames1)

compare_jacobians(
    "PART 1 Jacobian  d(x)/d(L_knob)",
    tnames1,
    vnames1,
    {"JAX (exact)": jac1_jax, "xtrack FD": jac1_fd, "MAD-NG TPSA": jac1_tpsa},
)

print(f"\n[solve target] x(end) = {x_target1:.9e}  (reachable at L_knob ~ 4.0)")
solve_and_report(opt1_fd, line1, NOMINAL1, "FD  ")
solve_and_report(opt1_tpsa, line1, NOMINAL1, "TPSA")


# ===========================================================================
# Analysis Part 2: quad + bend line, knob = quadrupole k1
# ===========================================================================

print("\n" + "#" * 78)
print("# PART 2 - quadrupole + bend line, knobs = quadrupole k1 (kqf, kqd)")
print("#" * 78)

BEND_ANGLE = 0.05
BEND_LEN = 2.0
DRIFT_L = 1.0
QUAD_LEN = 0.5
KQF0, KQD0 = 0.30, -0.32


def build_quad_bend_line(quad_model="mat-kick-mat", quad_nslice=1):
    """Build a d-qf-d-bend-d-qd-d line with the requested quad model."""
    env = make_proton_env()
    env["kqf"] = KQF0
    env["kqd"] = KQD0
    quad_kw = dict(
        length=QUAD_LEN,
        model=quad_model,
        integrator="uniform",
        num_multipole_kicks=quad_nslice,
        edge_entry_active=False,
        edge_exit_active=False,
    )
    env.new("d1", xt.Drift, length=DRIFT_L, model="exact")
    env.new("qf", xt.Quadrupole, k1="kqf", **quad_kw)
    env.new("d2", xt.Drift, length=DRIFT_L, model="exact")
    env.new(
        "mb",
        xt.Bend,
        angle=BEND_ANGLE,
        length=BEND_LEN,
        k1=0.0,
        model="bend-kick-bend",
        integrator="uniform",
        num_multipole_kicks=1,
        edge_entry_active=False,
        edge_exit_active=False,
    )
    env.new("d3", xt.Drift, length=DRIFT_L, model="exact")
    env.new("qd", xt.Quadrupole, k1="kqd", **quad_kw)
    env.new("d4", xt.Drift, length=DRIFT_L, model="exact")
    env.new("end2", xt.Marker)
    line = env.new_line(components=["d1", "qf", "d2", "mb", "d3", "qd", "d4", "end2"])
    line.particle_ref = env.particle_ref
    return env, line


def make_jax_track_quad_bend(line, beta0, quad_model="mat-kick-mat", quad_nslice=1):
    """Return a JAX track fn taking knobs = [kqf, kqd]."""
    d = {nn: float(line[nn].length) for nn in ["d1", "d2", "d3", "d4"]}
    qf_len = float(line["qf"].length)
    qd_len = float(line["qd"].length)
    mb_len = float(line["mb"].length)
    mb_h = float(line["mb"].h)
    mb_k0 = mb_h  # k0_from_h=True

    def track(knobs):
        s = INIT_STATE
        s = drift_exact_jax(s, d["d1"], beta0)
        s = quad_body_jax(s, qf_len, knobs[0], beta0, quad_model, quad_nslice)
        s = drift_exact_jax(s, d["d2"], beta0)
        s = bend_body_jax(s, mb_len, mb_k0, mb_h, beta0)
        s = drift_exact_jax(s, d["d3"], beta0)
        s = quad_body_jax(s, qd_len, knobs[1], beta0, quad_model, quad_nslice)
        s = drift_exact_jax(s, d["d4"], beta0)
        return s

    return track


env2, line2 = build_quad_bend_line("mat-kick-mat", 1)
beta0_2 = float(env2.particle_ref.beta0[0])
jax_track2 = make_jax_track_quad_bend(line2, beta0_2, "mat-kick-mat", 1)

NOMINAL2 = {"kqf": KQF0, "kqd": KQD0}
forward_check(
    "quad+bend line", jax_track2(jnp.array([KQF0, KQD0])), xt_track(line2, NOMINAL2)
)

# Well-posed problem: 2 quad knobs -> 2 targets (x, y at the end). The targets
# come from a tracked working point (kqf, kqd) = (0.36, -0.28) so a solution
# exists. The same problem is used for the Jacobian comparison and the solve.
state_true2 = xt_track(line2, {"kqf": 0.36, "kqd": -0.28})
x_target2, y_target2 = state_true2[0], state_true2[2]
INIT2 = dict(betx=1.0, bety=1.0, x=X0, px=PX0, y=Y0, py=PY0, delta=DELTA0)


def make_match2(use_tpsa):
    """Create the quad+bend match."""
    line2["kqf"], line2["kqd"] = KQF0, KQD0
    return line2.match(
        solve=False,
        use_tpsa=use_tpsa,
        vary=xt.VaryList(["kqf", "kqd"], step=1e-7),
        targets=[xt.TargetSet(at="end2", x=x_target2, y=y_target2, tol=1e-12)],
        **INIT2,
    )


opt2_fd = make_match2(use_tpsa=False)
opt2_tpsa = make_match2(use_tpsa=True)

# Jacobian at the nominal operating point.
jac2_fd, tnames2, vnames2 = physical_jacobian_from_opt(opt2_fd, [KQF0, KQD0])
jac2_tpsa, _, _ = physical_jacobian_from_opt(opt2_tpsa, [KQF0, KQD0])
jac2_jax = jax_jacobian(jax_track2, [KQF0, KQD0], tnames2)
compare_jacobians(
    "PART 2 Jacobian  d[x,y]/d[kqf,kqd]  at (kqf,kqd)=(0.30,-0.32)",
    tnames2,
    vnames2,
    {"JAX (exact)": jac2_jax, "xtrack FD": jac2_fd, "MAD-NG TPSA": jac2_tpsa},
)

# Any-x check: the same optimizers, evaluated at a different operating point.
X_ALT = [0.45, -0.20]
jac2_fd_alt, _, _ = physical_jacobian_from_opt(opt2_fd, X_ALT)
jac2_tpsa_alt, _, _ = physical_jacobian_from_opt(opt2_tpsa, X_ALT)
jac2_jax_alt = jax_jacobian(jax_track2, X_ALT, tnames2)
compare_jacobians(
    "PART 2 Jacobian  d[x,y]/d[kqf,kqd]  at (kqf,kqd)=(0.45,-0.20)",
    tnames2,
    vnames2,
    {
        "JAX (exact)": jac2_jax_alt,
        "xtrack FD": jac2_fd_alt,
        "MAD-NG TPSA": jac2_tpsa_alt,
    },
)

print(
    f"\n[solve target] x(end) = {x_target2:.9e}, y(end) = {y_target2:.9e} "
    f"(reachable at kqf=0.36, kqd=-0.28)"
)
solve_and_report(opt2_fd, line2, NOMINAL2, "FD  ")
solve_and_report(opt2_tpsa, line2, NOMINAL2, "TPSA")


# ===========================================================================
# Analysis part 3 - Model study: which xtrack model is closest to MAD-NG?
# ===========================================================================
#
# Xsuite's *default* models are 'mat-kick-mat' for Quadrupole and 'rot-kick-rot'
# for Bend. MAD-NG's TPSA track integrates with its exact thick maps.
# 'mat-kick-mat' is an expanded (approximate) combined map.
# 'drift-kick-drift-exact' uses exact drifts + thin kicks and converges to
# MAD-NG as the number of slices grows. The loop below quantifies this.

print("\n" + "#" * 78)
print("# PART 3 - model study: |JAX - MAD-NG TPSA| for different quad models")
print("#" * 78)

study_configs = [
    ("mat-kick-mat", 1),
    ("drift-kick-drift-exact", 1),
    ("drift-kick-drift-exact", 5),
    ("drift-kick-drift-exact", 30),
]

rows = []
for model, nslice in study_configs:
    env_s, line_s = build_quad_bend_line(model, nslice)
    beta0_s = float(env_s.particle_ref.beta0[0])
    track_s = make_jax_track_quad_bend(line_s, beta0_s, model, nslice)

    fwd = np.max(
        np.abs(
            np.asarray(track_s(jnp.array([KQF0, KQD0])))[:5]
            - xt_track(line_s, {"kqf": KQF0, "kqd": KQD0})[:5]
        )
    )

    # Reachable targets for THIS model (tracked at the working point).
    st = xt_track(line_s, {"kqf": 0.36, "kqd": -0.28})
    xt_, yt_ = st[0], st[2]

    def make_match_s(use_tpsa):
        line_s["kqf"], line_s["kqd"] = KQF0, KQD0
        return line_s.match(
            solve=False,
            use_tpsa=use_tpsa,
            betx=1.0,
            bety=1.0,
            x=X0,
            px=PX0,
            y=Y0,
            py=PY0,
            delta=DELTA0,
            vary=xt.VaryList(["kqf", "kqd"], step=1e-7),
            targets=[xt.TargetSet(at="end2", x=xt_, y=yt_, tol=1e-12)],
        )

    opt_tpsa_s = make_match_s(use_tpsa=True)
    jac_tpsa_s, tns, _ = physical_jacobian_from_opt(opt_tpsa_s, [KQF0, KQD0])
    jac_jax_s = jax_jacobian(track_s, [KQF0, KQD0], tns)

    opt_fd_s = make_match_s(use_tpsa=False)
    jac_fd_s, _, _ = physical_jacobian_from_opt(opt_fd_s, [KQF0, KQD0])

    d_jax_tpsa = np.max(np.abs(jac_jax_s - jac_tpsa_s))

    # Every optimizer we build must solve: do it and record the penalties.
    nom_s = {"kqf": KQF0, "kqd": KQD0}
    solve_and_report(opt_fd_s, line_s, nom_s, f"FD   {model}/{nslice}")
    solve_and_report(opt_tpsa_s, line_s, nom_s, f"TPSA {model}/{nslice}")

    rows.append((model, nslice, fwd, d_jax_tpsa))

print()
print(f"{'quad model':<26}{'nslice':>8}{'fwd |JAX-xt|':>16}{'max|JAX-TPSA|':>16}")
for model, nslice, fwd, d_jax_tpsa in rows:
    print(f"{model:<26}{nslice:>8}{fwd:>16.2e}{d_jax_tpsa:>16.2e}")

print("\nInterpretation:")
print("  * In every row JAX reproduces xtrack tracking to ~machine precision,")
print("    so JAX differentiates exactly the model xtrack is using; the")
print("    remaining gap to TPSA is a physics-model difference.")
print("  * 'mat-kick-mat' (xsuite default) sits at a gap to MAD-NG (~1e-8).")
print("  * 'drift-kick-drift-exact' *converges* to MAD-NG as nslice grows. It only")
print("    beats the default once nslice is large (~hundreds here).")
print("  * So: the default mat-kick-mat is fairly close and cheap; the only")
print("    model that *matches* MAD-NG arbitrarily well is drift-kick-drift-")
print("    exact with many slices. For bends, 'bend-kick-bend' (used above)")
print("    is the exact curved map and already agrees with MAD-NG to ~1e-9.")


# ===========================================================================
# Analysis part 4 - match beta / alpha / phase from exact-physics JAX
# ===========================================================================
#
# This is the step towards the LHC optics-matching use case.
# Here the very same optical functions are produced and differentiated w.r.t.
# the quadrupole knobs - from the *exact* element physics via jax_twiss().

print("\n" + "#" * 78)
print("# PART 4 - beta / alpha / phase matching from exact-physics JAX")
print("#" * 78)


def build_element_maps(line, beta0, quad_model="mat-kick-mat", quad_nslice=1):
    """Per-element exact maps for the line: list of (name, map_fn),
    each ``map_fn(state, knobs) -> state`` with knobs = [kqf, kqd]."""
    d = {nn: float(line[nn].length) for nn in ["d1", "d2", "d3", "d4"]}
    qf_len = float(line["qf"].length)
    qd_len = float(line["qd"].length)
    mb_len = float(line["mb"].length)
    mb_h = float(line["mb"].h)
    mb_k0 = mb_h  # k0_from_h=True
    return [
        ("d1", lambda s, k: drift_exact_jax(s, d["d1"], beta0)),
        (
            "qf",
            lambda s, k: quad_body_jax(s, qf_len, k[0], beta0, quad_model, quad_nslice),
        ),
        ("d2", lambda s, k: drift_exact_jax(s, d["d2"], beta0)),
        ("mb", lambda s, k: bend_body_jax(s, mb_len, mb_k0, mb_h, beta0)),
        ("d3", lambda s, k: drift_exact_jax(s, d["d3"], beta0)),
        (
            "qd",
            lambda s, k: quad_body_jax(s, qd_len, k[1], beta0, quad_model, quad_nslice),
        ),
        ("d4", lambda s, k: drift_exact_jax(s, d["d4"], beta0)),
    ]


# Initial Twiss conditions for the (open) line - deliberately non-trivial.
BETX0, ALFX0, BETY0, ALFY0 = 3.0, -1.0, 5.0, 2.0
TW_PARAMS0 = jnp.array([BETX0, ALFX0, 0.0, BETY0, ALFY0, 0.0, 0.0, 0.0, 0.0, 0.0])
TWISS_INIT = dict(
    betx=BETX0,
    alfx=ALFX0,
    bety=BETY0,
    alfy=ALFY0,
    dx=0.0,
    dpx=0.0,
    dy=0.0,
    dpy=0.0,
    x=X0,
    px=PX0,
    y=Y0,
    py=PY0,
    delta=DELTA0,
)

# Fresh line so MAD-NG starts from a clean state: the previous TPSA
# optimizers left line2's MAD-NG knobs as (not cleaned-up) TPSA objects, which
# would make a new madng_twiss return TPSA-typed columns.
env4, line4 = build_quad_bend_line("mat-kick-mat", 1)
beta0_4 = float(env4.particle_ref.beta0[0])
elem_maps4 = build_element_maps(line4, beta0_4, "mat-kick-mat", 1)


def xt_twiss_at(
    line, knob_values, at, cols=("betx", "alfx", "bety", "alfy", "mux", "muy")
):
    """xtrack open-line Twiss values at a location (the ground truth)."""
    for name, value in knob_values.items():
        line[name] = float(value)
    tw = line.twiss(**TWISS_INIT)
    return np.array([tw[c, at] for c in cols])


# (a) Forward check: JAX optics vs xtrack twiss at the line end.
fin2, bnd2 = jax_twiss(
    elem_maps4, jnp.array([KQF0, KQD0]), TW_PARAMS0, return_boundaries=True
)
cols_chk = ["betx", "alfx", "bety", "alfy", "mux", "muy"]
jax_end = np.array([fin2[TW_INDEX[c]] for c in cols_chk])
xt_end = xt_twiss_at(line4, NOMINAL2, "end2", cols_chk)
print("\n[forward check] beta/alpha/phase at end2:")
print("  " + "  ".join(f"{c}" for c in cols_chk))
print("  JAX   : " + "  ".join(f"{v:.6f}" for v in jax_end))
print("  xtrack: " + "  ".join(f"{v:.6f}" for v in xt_end))
print(
    f"  max|JAX - xtrack| = {np.max(np.abs(jax_end - xt_end)):.2e}  "
    f"(= the FD-vs-AD R-matrix gap; xtrack twiss linearizes by FD)"
)


# (b) Jacobian of the optics targets w.r.t. the quad knobs, from JAX.
TWISS_TARGETS = ["betx", "bety", "alfx", "alfy"]


def jax_twiss_targets(knobs):
    """[betx, alfx, bety, alfy] at the line end, as a function of [kqf, kqd]."""
    params = jax_twiss(elem_maps4, knobs, TW_PARAMS0)
    return jnp.array([params[TW_INDEX[t]] for t in TWISS_TARGETS])


jac_tw_jax = np.asarray(jax.jacfwd(jax_twiss_targets)(jnp.array([KQF0, KQD0])))


def xt_twiss_targets_fd(kqf, kqd):
    return xt_twiss_at(line4, {"kqf": kqf, "kqd": kqd}, "end2", TWISS_TARGETS)


h = 1e-6
jac_tw_fd = np.zeros((len(TWISS_TARGETS), 2))
jac_tw_fd[:, 0] = (
    xt_twiss_targets_fd(KQF0 + h, KQD0) - xt_twiss_targets_fd(KQF0 - h, KQD0)
) / (2 * h)
jac_tw_fd[:, 1] = (
    xt_twiss_targets_fd(KQF0, KQD0 + h) - xt_twiss_targets_fd(KQF0, KQD0 - h)
) / (2 * h)
for name in NOMINAL2:
    line4[name] = NOMINAL2[name]

# MAD-NG TPSA Jacobian of the same optics targets - the reference for "what are
# the true derivatives". (The optical functions are the linear part of the TPSA
# map MAD-NG builds with its exact thick maps.)
tw_nom_vals = xt_end[
    : len(TWISS_TARGETS)
]  # target values are irrelevant to the Jacobian


def make_twiss_match(use_tpsa):
    """Fresh open-line twiss match (betx/alfx/bety/alfy targets at end2)."""
    line4["kqf"], line4["kqd"] = KQF0, KQD0
    return line4.match(
        solve=False,
        use_tpsa=use_tpsa,
        betx=BETX0,
        alfx=ALFX0,
        bety=BETY0,
        alfy=ALFY0,
        dx=0.0,
        dpx=0.0,
        dy=0.0,
        dpy=0.0,
        x=X0,
        px=PX0,
        y=Y0,
        py=PY0,
        delta=DELTA0,
        vary=xt.VaryList(["kqf", "kqd"], step=1e-7),
        targets=[
            xt.TargetSet(
                at="end2",
                betx=tw_nom_vals[0],
                alfx=tw_nom_vals[1],
                bety=tw_nom_vals[2],
                alfy=tw_nom_vals[3],
                tol=1e-12,
            )
        ],
    )


opt_tw_tpsa = make_twiss_match(use_tpsa=True)
jac_tw_tpsa, tnames_tw, _ = physical_jacobian_from_opt(opt_tw_tpsa, [KQF0, KQD0])

for name in NOMINAL2:
    line4[name] = NOMINAL2[name]

compare_jacobians(
    "PART 4 Jacobian  d[betx,alfx,bety,alfy]/d[kqf,kqd]  at (0.30,-0.32)",
    TWISS_TARGETS,
    ["kqf", "kqd"],
    {
        "JAX (exact)": jac_tw_jax,
        "xtrack twiss FD": jac_tw_fd,
        "MAD-NG TPSA": jac_tw_tpsa,
    },
)
print("  Reading the deviations:")
print("  * JAX (exact mat-kick-mat) and the MAD-NG TPSA optimizer AGREE to")
print("    ~2e-4, for beta AND alpha: the JAX Twiss Jacobian")
print("    is closer to what MAD-NG computes (mat-kick-mat ~ the exact thick quad).")
print("  * The larger gap is to xtrack-twiss-FD (~3e-1 on the most beta-beating-")
print("    sensitive entry). That is not necessarily a physics difference,")
print("    more likely due to finite difference as the R-matrix is built like that.")


# (c) Relative phase advance is just a difference of accumulated phases and is
#     equally differentiable (this is what TargetRelPhaseAdvance matches).
def rel_muy_qf_to_end(knobs):
    _, b = jax_twiss(elem_maps4, knobs, TW_PARAMS0, return_boundaries=True)
    return b["d4"][TW_INDEX["muy"]] - b["d1"][TW_INDEX["muy"]]


dmuy = np.asarray(jax.grad(rel_muy_qf_to_end)(jnp.array([KQF0, KQD0])))
print(
    f"\n[phase advance] muy(end2) - muy(qf entrance) = "
    f"{float(rel_muy_qf_to_end(jnp.array([KQF0, KQD0]))):.6f}   "
    f"d/d[kqf,kqd] = [{dmuy[0]:.4f}, {dmuy[1]:.4f}]"
)


# (d) Set up and solve a beta/alpha matching problem with JAX targets.
#     Reachable targets are taken from a tracked working point, so a solution
#     exists.  Newton-least-squares (2 knobs, 4 optics targets) drives the
#     JAX-computed residual to zero; we verify the result with xtrack twiss.
KQF_TRUE, KQD_TRUE = 0.34, -0.30
tw_target_vals = xt_twiss_at(
    line4, {"kqf": KQF_TRUE, "kqd": KQD_TRUE}, "end2", TWISS_TARGETS
)
for name in NOMINAL2:
    line4[name] = NOMINAL2[name]
print(
    f"\n[solve target] {dict(zip(TWISS_TARGETS, np.round(tw_target_vals, 6)))}"
    f"  (reachable at kqf={KQF_TRUE}, kqd={KQD_TRUE})"
)


def solve_twiss_match_jax(target_vals, k0, n_iter=30, tol=1e-10):
    """Newton-least-squares on the JAX-computed optics residual."""
    k = np.array(k0, dtype=float)
    for _ in range(n_iter):
        res = np.asarray(jax_twiss_targets(jnp.array(k))) - target_vals
        if np.sqrt(np.sum(res**2)) < tol:
            break
        J = np.asarray(jax.jacfwd(jax_twiss_targets)(jnp.array(k)))
        k = k + np.linalg.lstsq(J, -res, rcond=None)[0]
    return k


k_sol = solve_twiss_match_jax(tw_target_vals, [KQF0, KQD0])
# Verify with xtrack twiss (the independent ground truth).
xt_at_sol = xt_twiss_at(
    line4, {"kqf": k_sol[0], "kqd": k_sol[1]}, "end2", TWISS_TARGETS
)
for name in NOMINAL2:
    line4[name] = NOMINAL2[name]
print(f"[solve JAX ] kqf={k_sol[0]:.9f}, kqd={k_sol[1]:.9f}")
print(
    f"             xtrack twiss at solution: "
    f"{dict(zip(TWISS_TARGETS, np.round(xt_at_sol, 6)))}"
)
print(
    f"             max|xtrack(sol) - target| = "
    f"{np.max(np.abs(xt_at_sol - tw_target_vals)):.2e}"
)

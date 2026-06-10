"""JAX optics on the real LHC lattice (HL-LHC b1).

Reproduces the IR8 match (betx/bety/alfx/alfy + relative phase advances
over s.ds.l8.b1 -> ip1) using JAX-differentiated exact element maps
instead of linear-transfer-matrix AD.

The factorized pipeline computes R = d(map)/d(state) once at the frozen
reference orbit, then propagates Twiss through a scan, avoiding nested
jacfwd and roughly halving trace+compile time.  The closed orbit through
IR8 is ~0 and nearly independent of the matching quads, so the freeze
is valid.
"""

import os
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import xtrack as xt

import jax_exact_maps as jm
from utils import load_hllhc_b1

jax.config.update("jax_enable_x64", True)

# Persistent XLA compilation cache: Python trace rebuilds each process (~2s)
# but XLA compile (~1.7s) is cached; subsequent runs pay ~0.1s.
_CACHE_DIR = os.environ.get(
    "JAX_LHC_CACHE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".jax_cache"),
)
jax.config.update("jax_compilation_cache_dir", _CACHE_DIR)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)

# Element-type codes for the jitted scan dispatch.
ET_DRIFT, ET_QUAD, ET_BEND, ET_KICK, ET_IDENT = 0, 1, 2, 3, 4

START, END = "s.ds.l8.b1", "ip1"

# ===========================================================================
# SECTION 1 - per-element exact maps
# ===========================================================================


def build_element_maps(line, ordered_names, beta0, kq_index=None):
    """Build per-element exact-physics maps for a lattice section.

    Returns a list of (name, map_fn) where ``map_fn(state, knobs) -> state``.
    When ``kq_index`` is given, varied quads read k1 from the knob vector
    at index ``kq_index[name]``; all other quads use their frozen k1.
    """
    ed = line.element_dict
    maps = []
    for nm in ordered_names:
        e = ed.get(nm)
        if e is None:
            continue
        cls = type(e).__name__
        length = float(getattr(e, "length", 0.0) or 0.0)
        isthick = bool(getattr(e, "isthick", False))

        if kq_index is not None and nm in kq_index:
            # varied quad: k1 from the knob vector
            j = kq_index[nm]
            maps.append(
                (
                    nm,
                    (
                        lambda s, k, L=length, j=j: jm.quad_body_jax(
                            s, L, k[j], beta0, "mat-kick-mat", 1
                        )
                    ),
                )
            )
        elif cls == "Quadrupole" and float(e.k1) != 0.0:
            # Fixed quad
            k1 = float(e.k1)
            maps.append(
                (
                    nm,
                    (
                        lambda s, k, L=length, k1=k1: jm.quad_body_jax(
                            s, L, k1, beta0, "mat-kick-mat", 1
                        )
                    ),
                )
            )
        elif cls in ("Bend", "RBend"):
            # Fixed bend
            k0, h, k1 = float(e.k0), float(e.h), float(e.k1)
            if k1 == 0.0 and h != 0.0:
                ec = jm.bend_edge_coeffs(e)
                maps.append(
                    (
                        nm,
                        (
                            lambda s, k, L=length, k0=k0, h=h, ec=ec: (
                                jm.bend_with_edges_jax(s, L, k0, h, beta0, *ec)
                            )
                        ),
                    )
                )
            else:  # combined-function (or h==0): expanded combined map.
                maps.append(
                    (
                        nm,
                        (
                            lambda s, k, L=length, k0=k0, k1=k1, h=h: (
                                jm.quad_combined_map_jax(s, L, k0, k1, h, beta0)
                            )
                        ),
                    )
                )
        elif isthick and length > 0.0:
            # thick with no linear focusing at zero orbit (sextupole, solenoid, …) -> drift
            maps.append((nm, (lambda s, k, L=length: jm.drift_exact_jax(s, L, beta0))))
        else:
            # thin: only knl[1]/ksl[1] contribute to linear optics at zero orbit
            knl = np.asarray(getattr(e, "knl", [0.0]), dtype=float)
            ksl = np.asarray(getattr(e, "ksl", [0.0]), dtype=float)
            n = max(len(knl), len(ksl), 1)
            knl = np.pad(knl, (0, n - len(knl)))
            ksl = np.pad(ksl, (0, n - len(ksl)))
            if not (np.any(knl) or np.any(ksl)):
                continue  # pure identity -> skip
            knl_j, ksl_j = jnp.array(knl), jnp.array(ksl)
            maps.append(
                (
                    nm,
                    (
                        lambda s, k, knl=knl_j, ksl=ksl_j: jm.thin_multipole_kick_jax(
                            s, knl, ksl
                        )
                    ),
                )
            )
    return maps


# ===========================================================================
# SECTION 2 - struct-of-arrays encoding + factorized pipeline
# ===========================================================================


class Enc(NamedTuple):
    """Struct-of-arrays encoding of a lattice section for ``lax.scan``.

    Each field is a 1-D JAX array of length N (one entry per element).
    Row i holds the parameters for element i; ``lax.scan`` iterates over rows
    and dispatches to the correct map via ``etype``.

    etype  : element type code (ET_DRIFT=0, ET_QUAD=1, ET_BEND=2, ET_KICK=3, ET_IDENT=4)
    L      : element length [m]
    k0, h  : bend dipole strength and reference curvature
    k1fix  : frozen quad strength (used when kqidx == -1)
    kqidx  : index into the differentiable kq vector, or -1 if fixed
    knl1, ksl1 : thin-multipole normal/skew quad strength (knl[1], ksl[1])
    r21i/r43i/r21o/r43o : linear dipole-edge coeffs (entry/exit)
    """

    etype: jnp.ndarray
    L: jnp.ndarray
    k0: jnp.ndarray
    h: jnp.ndarray
    k1fix: jnp.ndarray
    kqidx: jnp.ndarray
    knl1: jnp.ndarray
    ksl1: jnp.ndarray
    r21i: jnp.ndarray
    r43i: jnp.ndarray
    r21o: jnp.ndarray
    r43o: jnp.ndarray


def encode_section(line, ordered_names, kq_index):
    """Convert a lattice section into a struct-of-arrays (``Enc``) for ``lax.scan``.

    Each element maps to one row; identity elements (markers, zero-strength
    thin correctors) are assigned ET_IDENT and become no-ops in the scan.
    """
    ed = line.element_dict
    et, L, k0, h, k1fix, kqidx, knl1, ksl1 = ([] for _ in range(8))
    r21i, r43i, r21o, r43o = ([] for _ in range(4))
    for nm in ordered_names:
        e = ed.get(nm)
        cls = type(e).__name__ if e is not None else None
        length = float(getattr(e, "length", 0.0) or 0.0)
        isthick = bool(getattr(e, "isthick", False))
        code, a_L, a_k0, a_h, a_k1, a_idx, a_kn1, a_ks1 = (
            ET_IDENT,
            0.0,
            0.0,
            0.0,
            0.0,
            -1,
            0.0,
            0.0,
        )
        a_e = (0.0, 0.0, 0.0, 0.0)
        if e is None:
            pass
        elif nm in kq_index:
            code, a_L, a_idx = ET_QUAD, length, kq_index[nm]
        elif cls == "Quadrupole" and float(e.k1) != 0.0:
            code, a_L, a_k1 = ET_QUAD, length, float(e.k1)
        elif cls in ("Bend", "RBend"):
            k0v, hv, k1v = float(e.k0), float(e.h), float(e.k1)
            assert k1v == 0.0 and hv != 0.0, f"combined-function bend {nm}"
            code, a_L, a_k0, a_h = ET_BEND, length, k0v, hv
            a_e = jm.bend_edge_coeffs(e)
        elif isthick and length > 0.0:
            code, a_L = ET_DRIFT, length
        else:
            knl = np.asarray(getattr(e, "knl", [0.0]), dtype=float)
            ksl = np.asarray(getattr(e, "ksl", [0.0]), dtype=float)
            a_kn1 = float(knl[1]) if len(knl) > 1 else 0.0
            a_ks1 = float(ksl[1]) if len(ksl) > 1 else 0.0
            code = ET_KICK if (a_kn1 or a_ks1) else ET_IDENT
        for lst, v in zip(
            (et, L, k0, h, k1fix, kqidx, knl1, ksl1, r21i, r43i, r21o, r43o),
            (
                code,
                a_L,
                a_k0,
                a_h,
                a_k1,
                a_idx,
                a_kn1,
                a_ks1,
                a_e[0],
                a_e[1],
                a_e[2],
                a_e[3],
            ),
        ):
            lst.append(v)
    return Enc(
        jnp.array(et, dtype=jnp.int32),
        jnp.array(L),
        jnp.array(k0),
        jnp.array(h),
        jnp.array(k1fix),
        jnp.array(kqidx, dtype=jnp.int32),
        jnp.array(knl1),
        jnp.array(ksl1),
        jnp.array(r21i),
        jnp.array(r43i),
        jnp.array(r21o),
        jnp.array(r43o),
    )


def compress_encoding(enc, boundary_rows):
    """Drop ET_IDENT rows and merge consecutive drifts, preserving boundary rows.

    Boundary rows are target positions that must not be merged away, so that
    scan output index i still corresponds to the optics after a specific element.
    Shrinks the LHC IR8 section from ~1960 rows to ~560.
    """
    et = np.asarray(enc.etype)
    L = np.asarray(enc.L)
    k0 = np.asarray(enc.k0)
    h = np.asarray(enc.h)
    k1f = np.asarray(enc.k1fix)
    kqi = np.asarray(enc.kqidx)
    kn1 = np.asarray(enc.knl1)
    ks1 = np.asarray(enc.ksl1)
    e21i = np.asarray(enc.r21i)
    e43i = np.asarray(enc.r43i)
    e21o = np.asarray(enc.r21o)
    e43o = np.asarray(enc.r43o)
    boundary = set(int(b) for b in boundary_rows)

    rows = []
    orig_to_comp = np.empty(len(et), dtype=np.int64)
    acc = 0.0  # accumulated drift length

    def flush():
        nonlocal acc
        if acc > 0.0:
            rows.append(
                (ET_DRIFT, acc, 0.0, 0.0, 0.0, -1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            )
            acc = 0.0

    for i in range(len(et)):
        if et[i] == ET_IDENT:
            pass
        elif et[i] == ET_DRIFT:
            acc += float(L[i])
        else:
            flush()
            rows.append(
                (
                    int(et[i]),
                    float(L[i]),
                    float(k0[i]),
                    float(h[i]),
                    float(k1f[i]),
                    int(kqi[i]),
                    float(kn1[i]),
                    float(ks1[i]),
                    float(e21i[i]),
                    float(e43i[i]),
                    float(e21o[i]),
                    float(e43o[i]),
                )
            )
        if i in boundary:
            flush()
        orig_to_comp[i] = len(rows) - 1
    flush()

    a = np.array(rows, dtype=float)
    return Enc(
        jnp.array(a[:, 0], dtype=jnp.int32),
        jnp.array(a[:, 1]),
        jnp.array(a[:, 2]),
        jnp.array(a[:, 3]),
        jnp.array(a[:, 4]),
        jnp.array(a[:, 5], dtype=jnp.int32),
        jnp.array(a[:, 6]),
        jnp.array(a[:, 7]),
        jnp.array(a[:, 8]),
        jnp.array(a[:, 9]),
        jnp.array(a[:, 10]),
        jnp.array(a[:, 11]),
    ), orig_to_comp


def build_section_twiss(enc, beta0, p0, s0):
    """Build the jitted Twiss-target function for a compressed section encoding.

    Returns ``targets(kq, rows, qidx)`` which can be differentiated with
    ``jax.jacfwd`` to get d(Twiss)/d(kq).

    The three-step pipeline inside: (1) orbit scan with ``stop_gradient``
    to freeze the orbit; (2) vmap'd ``jacfwd`` over all elements to get R
    matrices; (3) Twiss propagation scan (pure linear algebra, no nested AD).
    """

    def emap_single(s, e, kq):
        """Apply the exact element map for one row of the Enc struct."""
        et, L, k0, h, k1f, kqi, kn1, ks1, r21i, r43i, r21o, r43o = e
        # varied quads read k1 from kq[kqi]; frozen quads use k1f
        k1 = jnp.where(kqi >= 0, kq[jnp.maximum(kqi, 0)], k1f)
        return lax.switch(
            et,
            [
                lambda s: jm.drift_exact_jax(s, L, beta0),
                lambda s: jm.quad_body_jax(s, L, k1, beta0, "mat-kick-mat", 1),
                lambda s: jm.bend_with_edges_jax(
                    s, L, k0, h, beta0, r21i, r43i, r21o, r43o
                ),
                lambda s: jm.thin_quad_skew_kick_jax(s, kn1, ks1),
                lambda s: s,
            ],
            s,
        )

    def all_params(kq):
        # Step 1: propagate the orbit (frozen - no gradient tracking)
        def obody(s, e):
            return emap_single(s, e, kq), s

        _, orbits_in = lax.scan(obody, s0, enc)
        orbits_in = lax.stop_gradient(orbits_in)  # freeze orbit for Jacobian

        # Step 2: R matrices at frozen orbit via vmap'd jacfwd
        Rs = jax.vmap(lambda e, o: jax.jacfwd(lambda s: emap_single(s, e, kq))(o))(
            enc, orbits_in
        )

        # Step 3: Twiss scan
        def tbody(params, R):
            pn = jm.propagate_twiss(R, params)
            return pn, pn

        _, hist = lax.scan(tbody, p0, Rs)
        return hist  # shape (N_elements, 10) basically Twiss table

    @jax.jit
    def targets(kq, rows, qidx):
        return all_params(kq)[rows, qidx]

    return targets


# ===========================================================================
# SECTION 3 - knob Jacobian: chain d(target)/d(kq) @ d(kq)/d(knob)
# ===========================================================================


def fast_knob_jacobian(line, beta0, vary_names, target_specs, p0, s0, ordered_names):
    """Compute d(Twiss targets)/d(knobs) via the fast factorized pipeline.

    Main entry point called by the match hook (``use_jax=True``).  Builds and
    compiles the targets function once; the returned ``jac_fn`` can be reused
    across Newton steps.
    """
    varied, kq0, dk1 = dk1_dknob_matrix(line, ordered_names, vary_names)
    kq_index = {q: i for i, q in enumerate(varied)}
    enc = encode_section(line, ordered_names, kq_index)

    name_to_row = {nm: i - 1 for i, nm in enumerate(ordered_names)}
    orig_rows = [name_to_row[pos] for _, pos in target_specs]
    # compress and get mapping from original to compressed row indices
    cenc, orig_to_comp = compress_encoding(enc, set(orig_rows))
    rows = jnp.array([int(orig_to_comp[r]) for r in orig_rows], dtype=jnp.int32)
    qidx = jnp.array([jm.TW_INDEX[q] for q, _ in target_specs], dtype=jnp.int32)

    targets_fn = build_section_twiss(cenc, beta0, p0, s0)
    jac_fn = jax.jit(jax.jacfwd(lambda kq: targets_fn(kq, rows, qidx)))
    jac_kq = np.asarray(jac_fn(jnp.array(kq0)))  # (ntar, nvar)
    jac_knob = jac_kq @ dk1  # (ntar, nknob)
    return jac_knob, jac_fn, kq0, dk1


def dk1_dknob_matrix(line, ordered_names, vary_names, delta=1e-4):
    """Build the d(k1_quad)/d(knob) sensitivity matrix via central differences.

    Returns ``(varied, kq0, dk1)`` where ``varied`` are the quads actually
    driven by at least one knob.
    """
    ed = line.element_dict
    section_quads = [
        nm for nm in ordered_names if type(ed.get(nm)).__name__ == "Quadrupole"
    ]
    k0 = {v: float(line[v]) for v in vary_names}

    def k1_vec():
        return np.array([float(ed[q].k1) for q in section_quads])

    dk1 = np.zeros((len(section_quads), len(vary_names)))
    for j, v in enumerate(vary_names):
        line[v] = k0[v] + delta
        kp = k1_vec()
        line[v] = k0[v] - delta
        km = k1_vec()
        line[v] = k0[v]
        dk1[:, j] = (kp - km) / (2 * delta)

    mask = np.any(dk1 != 0.0, axis=1)
    varied = [section_quads[i] for i in range(len(section_quads)) if mask[i]]
    kq0 = np.array([float(ed[q].k1) for q in varied])
    return varied, kq0, dk1[mask, :]


def knob_jacobian_jax(line, beta0, vary_names, targets, p0, s0, ordered_names):
    """d(target)/d(knob) Jacobian, slower version without factorized pipeline.

    targets : list of (quantity, position_name) along the section.
    Returns jac with shape (n_target, n_knob).
    """
    varied, kq0, dk1 = dk1_dknob_matrix(line, ordered_names, vary_names)
    kq_index = {q: i for i, q in enumerate(varied)}
    emaps = build_element_maps(line, ordered_names, beta0, kq_index=kq_index)
    mapdict = dict(emaps)

    want = {pos for _, pos in targets}
    pos_after = {}
    for i, nm in enumerate(ordered_names):
        nxt = ordered_names[i + 1] if i + 1 < len(ordered_names) else nm
        if nxt in want:
            pos_after[nm] = nxt

    def optics_at_positions(kq):
        state = s0
        params = p0
        out = {}
        for nm in ordered_names:
            f = mapdict.get(nm)
            if f is not None:
                R = jax.jacfwd(lambda s, f=f: f(s, kq))(state)
                params = jm.propagate_twiss(R, params)
                state = f(state, kq)
            if nm in pos_after:
                out[pos_after[nm]] = params
        return out

    def target_vec(kq):
        out = optics_at_positions(kq)
        return jnp.array([out[pos][jm.TW_INDEX[q]] for q, pos in targets])

    jac_kq = np.asarray(jax.jacfwd(target_vec)(jnp.array(kq0)))  # (ntar, nvar)
    jac_knob = jac_kq @ dk1  # (ntar, nknob)
    return jac_knob


# ===========================================================================
# SECTION 4 - validation
# ===========================================================================


def main():
    """Load LHC IR8 section, encode, and validate Twiss + Jacobian."""
    # Load the lattice
    collider, line = load_hllhc_b1(set_var_limits=False)
    beta0 = float(line.particle_ref.beta0[0])

    # Reference Twiss
    tw0 = line.twiss()
    tw = line.twiss(start=START, end=END, init=tw0, init_at=xt.START)

    ordered_names = list(tw.name[:-1])  # drop trailing _end_point marker
    end_name = (
        ordered_names[-1] if ordered_names[-1] in line.element_dict else tw.name[-2]
    )

    # Initial Twiss params [betx, alfx, mux, bety, alfy, muy, dx, dpx, dy, dpy]
    p0 = jnp.array(
        [
            tw["betx", 0],
            tw["alfx", 0],
            tw["mux", 0],
            tw["bety", 0],
            tw["alfy", 0],
            tw["muy", 0],
            tw["dx", 0],
            tw["dpx", 0],
            tw["dy", 0],
            tw["dpy", 0],
        ]
    )
    # Initial 6D orbit [x, px, y, py, zeta, delta]
    s0 = jnp.array(
        [
            tw["x", 0],
            tw["px", 0],
            tw["y", 0],
            tw["py", 0],
            tw["zeta", 0],
            tw["delta", 0],
        ]
    )

    # M1: forward Twiss via per-element exact maps
    elem_maps = build_element_maps(line, ordered_names, beta0)
    print(
        f"section {START} -> {END}: {len(ordered_names)} rows, "
        f"{len(elem_maps)} non-trivial element maps"
    )

    final, bnd = jm.jax_twiss(
        elem_maps, jnp.array([0.0]), p0, s0, return_boundaries=True
    )

    cols = ["betx", "alfx", "bety", "alfy", "mux", "muy", "dx", "dpx"]
    print("\n[forward check] optics at end (%s):" % END)
    print(f"  {'quantity':8s} {'JAX':>16s} {'xtrack':>16s} {'|diff|':>10s}")
    last = tw.name[-2]
    for c in cols:
        jv = float(final[jm.TW_INDEX[c]])
        xv = float(tw[c, last])
        print(f"  {c:8s} {jv:16.8f} {xv:16.8f} {abs(jv - xv):10.2e}")

    # Max deviation of betx/bety along the section
    err_betx = err_bety = 0.0
    for i, nm in enumerate(ordered_names):
        if nm not in bnd:
            continue
        nxt = ordered_names[i + 1] if i + 1 < len(ordered_names) else last
        err_betx = max(
            err_betx, abs(float(bnd[nm][jm.TW_INDEX["betx"]]) - float(tw["betx", nxt]))
        )
        err_bety = max(
            err_bety, abs(float(bnd[nm][jm.TW_INDEX["bety"]]) - float(tw["bety", nxt]))
        )
    print(f"\n  max|betx_JAX - betx_xtrack| along section = {err_betx:.2e}")
    print(f"  max|bety_JAX - bety_xtrack| along section = {err_bety:.2e}")

    # M2: knob Jacobian vs finite differences
    validate_knob_jacobian(line, beta0, tw0, tw, ordered_names, p0, s0)


# IR8 quadrupole knobs varied in 001_test_optimization.
VARY_001 = [
    "kq6.l8b1",
    "kq7.l8b1",
    "kq8.l8b1",
    "kq9.l8b1",
    "kq10.l8b1",
    "kqtl11.l8b1",
    "kqt12.l8b1",
    "kqt13.l8b1",
    "kq4.l8b1",
    "kq5.l8b1",
    "kq4.r8b1",
    "kq5.r8b1",
    "kq6.r8b1",
    "kq7.r8b1",
    "kq8.r8b1",
    "kq9.r8b1",
    "kq10.r8b1",
    "kqtl11.r8b1",
    "kqt12.r8b1",
    "kqt13.r8b1",
]


def validate_knob_jacobian(line, beta0, tw0, tw, ordered_names, p0, s0):
    """Validate d(targets)/d(knobs) against xtrack finite differences.

    Also measures compile time and warm steady-state call time to characterise
    what a matching loop pays per Newton step.
    """
    vary = VARY_001
    end_pos = tw.name[-2]
    targets = [
        ("betx", "ip8"),
        ("bety", "ip8"),
        ("alfx", "ip8"),
        ("alfy", "ip8"),
        ("betx", end_pos),
        ("bety", end_pos),
        ("alfx", end_pos),
        ("alfy", end_pos),
    ]

    import time

    # Build and compile (first call triggers XLA compilation + disk cache write).
    t0 = time.perf_counter()
    jac_jax, jac_fn, kq0, _dk1 = fast_knob_jacobian(
        line, beta0, vary, targets, p0, s0, ordered_names
    )
    kqa = jnp.array(kq0)
    jax.block_until_ready(jac_fn(kqa))
    t1 = time.perf_counter()

    # Warm steady-state: what a matching loop pays per Newton step.
    nrep = 20
    for _ in range(nrep):
        jax.block_until_ready(jac_fn(kqa))
    t2 = time.perf_counter()
    print(
        "\n"
        f"[perf] fast jacobian: build(trace+compile+1st)={t1 - t0:.1f}s, "
        f"warm call={(t2 - t1) / nrep * 1e3:.1f}ms (mean of {nrep})"
    )

    # FD reference via xtrack twiss
    k0 = {v: float(line[v]) for v in vary}
    d = 1e-6

    def tw_targets():
        t = line.twiss(start=START, end=END, init=tw0, init_at=xt.START)
        return np.array([t[q, pos] for q, pos in targets])

    jac_fd = np.zeros((len(targets), len(vary)))
    for j, v in enumerate(vary):
        line[v] = k0[v] + d
        tp = tw_targets()
        line[v] = k0[v] - d
        tm = tw_targets()
        line[v] = k0[v]
        jac_fd[:, j] = (tp - tm) / (2 * d)

    print("\n[M2] exact-physics knob Jacobian d[betx,bety,alfx,alfy @ip8,end]/d[knob]")
    print(
        f"     JAX build time {t1 - t0:.1f}s, {len(vary)} knobs, "
        f"{jac_jax.shape[0]} targets"
    )
    denom = np.maximum(np.abs(jac_fd), 1e-6)
    relerr = np.abs(jac_jax - jac_fd) / denom
    print(f"     max|JAX - FD|        = {np.max(np.abs(jac_jax - jac_fd)):.2e}")
    print(f"     max rel err (|J|>1e-6)= {np.max(relerr):.2e}")
    # Show the largest-magnitude column for a concrete sense of agreement.
    jcol = int(np.argmax(np.max(np.abs(jac_fd), axis=0)))
    print(f"     sample column knob={vary[jcol]}:")
    for i, (q, pos) in enumerate(targets):
        print(
            f"       d{q}@{pos:8s} JAX={jac_jax[i, jcol]:+.4e} "
            f"FD={jac_fd[i, jcol]:+.4e}"
        )


if __name__ == "__main__":
    main()

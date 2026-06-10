"""Shared helpers for the matching examples.

The three ``*`` builders below encapsulate the boilerplate of the matching
problems used throughout the studies, so the examples of this repository
simply create the match problem by calling the corresponding function.

    from utils import tune_chroma
    opt = tune_chroma(line, use_jax=True)
    opt.solve()

The backend selection is keyword-driven: every builder forwards its keywords
straight to ``line.match`` - ``use_jax`` (with or without)
``use_jax_residual`` / ``use_tpsa`` / ``use_tpsa_direct`` / ``use_ad`` and any
other ``match`` keyword.  (Unknown keywords are ignored by ``OptimizeLine``
where a given backend is not built, for example in xtrack branches
which only implemented one backend.

To switch the backend of an *already-built* opt in place (no rebuild), use
``use_backend(opt, 'jax')`` or the ``switch_to_*`` shortcuts.

All builders default to ``solve=False`` (so the caller can ``step`` / time /
``reset_opt`` / ``solve`` as it likes) and return the ``OptimizeLine``.
"""

import xtrack as xt

# HL-LHC collider data (same files 005/030/031/054 use), relative to the cwd the
# examples run from (the fast_optics_jacobian dir).
LHC_JSON = "../xtrack/test_data/hllhc15_thick/hllhc15_collider_thick.json"
LHC_MADX = "../xtrack/test_data/hllhc15_thick/opt_round_150_1500.madx"

# The 20 IR8 quadrupole circuits varied by the 005 optics match.
IR8_VARY = [
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


def reset_opt(opt):
    opt.reload(0)
    opt.clear_log()
    opt._err.call_counter = 0
    opt.solver._last_jac = None


# --------------------------------------------------------------------------- #
# Seamless backend switching on an already-built opt (no rebuild).
# Generalises 005's switch_to_ad / switch_to_fd to every backend.
# --------------------------------------------------------------------------- #
def use_backend(opt, backend="fd"):
    """Switch ``opt``'s Jacobian backend in place to one of:

        'fd' | 'ad' | 'jax' | 'tpsa_direct' | 'tpsa'

    Only the boolean flags are toggled, so the switch is instantaneous (the JAX
    / TPSA-direct objects are lazily (re)built on the next ``get_jacobian``).

    Note: 'tpsa' (MAD-NG) requires the opt to have been *built* with
    ``use_tpsa=True`` (it relies on a MAD-NG twiss action), so it cannot be
    switched onto a plain opt; the others can.
    """
    err = opt._err
    err.use_ad = backend == "ad"
    err.use_jax = backend == "jax"
    err.use_tpsa = backend == "tpsa"
    err.use_tpsa_direct = backend == "tpsa_direct"
    return opt


def switch_to_fd(opt):
    return use_backend(opt, "fd")


def switch_to_ad(opt):
    return use_backend(opt, "ad")


def switch_to_jax(opt):
    return use_backend(opt, "jax")


def switch_to_tpsa_direct(opt):
    return use_backend(opt, "tpsa_direct")


def switch_to_tpsa(opt):
    return use_backend(opt, "tpsa")


def load_hllhc_b1(set_var_limits=True):
    """Load the HL-LHC thick collider, beam 1.  Returns ``(collider, line)``."""
    collider = xt.Environment.from_json(LHC_JSON)
    collider.vars.load(
        LHC_MADX, format="madx"
    )  # xt.load (not the deprecated load_madx)
    collider.build_trackers()
    if set_var_limits:
        from xtrack._temp import lhc_match as lm

        lm.set_var_limits_and_steps(collider)
    return collider, collider.lhcb1


# --------------------------------------------------------------------------- #
# 1. IR8 optics match: s.ds.l8.b1 -> ip1, 20 knobs, 14 targets.
# --------------------------------------------------------------------------- #
def _ir8_targets(tw0, madng_names=False):
    """Targets + default_tol for the IR8 optics match (Xsuite or MAD-NG names)."""
    b1, b2 = ("beta11_ng", "beta22_ng") if madng_names else ("betx", "bety")
    a1, a2 = ("alfa11_ng", "alfa22_ng") if madng_names else ("alfx", "alfy")
    dx, dpx = ("dx_ng", "dpx_ng") if madng_names else ("dx", "dpx")
    mu1, mu2 = ("mu1_ng", "mu2_ng") if madng_names else ("mux", "muy")

    targets = [
        xt.TargetSet(at="ip8", tars=(b1, b2, a1, a2, dx, dpx), value=tw0, weight=1),
        xt.TargetSet(
            at="ip1", weight=1, **{b1: 0.15, b2: 0.1, a1: 0, a2: 0, dx: 0, dpx: 0}
        ),
        xt.TargetRelPhaseAdvance(
            mu1,
            start="s.ds.l8.b1",
            end="ip1.l1",
            value=tw0[mu1, "ip1.l1"] - tw0[mu1, "s.ds.l8.b1"],
            weight=1,
        ),
        xt.TargetRelPhaseAdvance(
            mu2,
            start="s.ds.l8.b1",
            end="ip1.l1",
            value=tw0[mu2, "ip1.l1"] - tw0[mu2, "s.ds.l8.b1"],
            weight=1,
        ),
    ]
    tol = {None: 1e-8, b1: 1e-6, b2: 1e-6, a1: 1e-6, a2: 1e-6}
    return targets, tol


def ir8_optics(line=None, tw0=None, *, solve=False, **match_kwargs):
    """Build the HL-LHC IR8 optics match example.

    Match betx/bety/alfx/alfy/dx/dpx at ip8 & ip1
    and keep phase advance the same. 20 quadrupole
    knobs are varied.

    Parameters
    ----------
    line, tw0 : optional
        The loaded ``lhcb1`` line and its closed twiss.  If ``line``
        is None the collider is loaded here. If ``tw0`` is None it is
        twissed here according to the madng flag.
    solve, **match_kwargs
        Forwarded to ``line.match`` (backend flags, ``default_tol``, ...).
    """
    madng = True if "use_tpsa" in match_kwargs and match_kwargs["use_tpsa"] else False
    # find madng_names from match_kwargs
    madng_names = match_kwargs.get("madng_names", madng)
    if line is None:
        _, line = load_hllhc_b1()
    if tw0 is None:
        tw0 = line.madng_twiss() if madng else line.twiss()
    targets, tol = _ir8_targets(tw0, madng_names=madng_names)
    match_kwargs.setdefault("default_tol", tol)
    return line.match(
        solve=solve,
        start="s.ds.l8.b1",
        end="ip1",
        init=tw0,
        init_at=xt.START,
        vary=[xt.VaryList(IR8_VARY)],
        targets=targets,
        **match_kwargs,
    )


# --------------------------------------------------------------------------- #
# 2. Tune + chromaticity match (used in 032): qx/qy via trim quads,
#    dqx/dqy via sextupoles, on the thick LHC b1 line with knobs.
# --------------------------------------------------------------------------- #
def tune_chroma(line=None, *, solve=False, **match_kwargs):
    """Build the tune (qx,qy) + chromaticity (dqx,dqy) match (example 032)."""
    if line is None:
        line = xt.Line.from_json("lattice_data/lhc_thick_with_knobs.json")
        line.build_tracker()
    return line.match(
        solve=solve,
        method="4d",
        vary=[
            xt.VaryList(["kqtf.b1", "kqtd.b1"], step=1e-8, tag="quad"),
            xt.VaryList(
                ["ksf.b1", "ksd.b1"], step=1e-4, limits=[-0.1, 0.1], tag="sext"
            ),
        ],
        targets=[
            xt.TargetSet(qx=62.315, qy=60.325, tol=1e-6, tag="tune", weight=1),
            xt.TargetSet(dqx=10.0, dqy=12.0, tol=0.01, tag="chrom", weight=1),
        ],
        **match_kwargs,
    )


# --------------------------------------------------------------------------- #
# 3. Orbit-bump correction (used in 018): four thin-corrector pairs steer
#    an orbit bump (x, px, y, py at mid & end) on a short synthetic line.
# --------------------------------------------------------------------------- #
def _orbit_line():
    env = xt.Environment()
    env.vars.default_to_zero = True
    line = env.new_line(
        length=10,
        components=[
            env.new(
                "corr1",
                xt.Multipole,
                isthick=True,
                knl=["kick_h_1"],
                ksl=["kick_v_1"],
                length=0.1,
                at=1,
            ),
            env.new(
                "corr2",
                xt.Multipole,
                isthick=True,
                knl=["kick_h_2"],
                ksl=["kick_v_2"],
                length=0.1,
                at=2,
            ),
            env.new(
                "corr3",
                xt.Multipole,
                isthick=True,
                knl=["kick_h_3"],
                ksl=["kick_v_3"],
                length=0.1,
                at=8,
            ),
            env.new(
                "corr4",
                xt.Multipole,
                isthick=True,
                knl=["kick_h_4"],
                ksl=["kick_v_4"],
                length=0.1,
                at=9,
            ),
            env.new("mid", xt.Marker, at=5),
            env.new("end", xt.Marker, at=10),
        ],
    )
    line.set_particle_ref("proton", p0c=26e9)
    return line


def orbit_correction(line=None, *, solve=False, **match_kwargs):
    """Build the closed-orbit bump match (used in example 018).

    Four horizontal/vertical corrector pairs are varied to reach a chosen orbit
    at the midpoint and close it at the end.  If ``line`` is None the
    short synthetic 10 m line is built here.
    """
    madng = True if "use_tpsa" in match_kwargs else False
    if line is None:
        line = _orbit_line()
    if madng:
        line.to_madng()
    return line.match(
        solve=solve,
        betx=1,
        bety=1,
        vary=xt.VaryList(
            [
                "kick_h_1",
                "kick_v_1",
                "kick_h_2",
                "kick_v_2",
                "kick_h_3",
                "kick_v_3",
                "kick_h_4",
                "kick_v_4",
            ]
        ),
        targets=[
            xt.TargetSet(x=1e-3, y=-2e-3, px=0, py=0, at="mid"),
            xt.TargetSet(x=0, y=0, px=0, py=0, at="end"),
        ],
        **match_kwargs,
    )

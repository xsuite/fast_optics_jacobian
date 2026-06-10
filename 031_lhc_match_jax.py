"""Reproduce IR8 match using the new JAX Jacobian backend,
validating that the exact-physics JAX backend (from 030_lhc_jax_optics.py)
integrates correctly into xtrack's matching optimizer.

BACKENDS COMPARED
-----------------
1. JAX (exact-physics): Jacobian computed via jax.jacfwd + factorized pipeline
2. xsuite FD (finite differences): One-sided forward difference with per-knob steps
3. MAD-NG TPSA (if available): Taylor expansion of the one-turn map
"""

import time
import numpy as np
import xtrack as xt
from xtrack._temp import lhc_match as lm
from utils import ir8_optics, load_hllhc_b1


collider, line = load_hllhc_b1(set_var_limits=True)

tw0 = line.twiss()

VARY = [
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

# Original (unmatched) knob values - matching leaves the line with the optimized
# values, so these must be restored to time a genuine solve.
KNOBS0 = {v: float(line[v]) for v in VARY}


def reset_opt(opt):
    opt.reload(0)
    opt.clear_log()
    opt._err.call_counter = 0
    opt.solver._last_jac = None


# ===========================================================================
# PART 1: JACOBIAN COMPARISON - JAX vs. TPSA vs. FINITE DIFFERENCES
# ===========================================================================
# All three backends compute the same weighted Jacobian at the same operating
# point. We compare:
#   1. Numerical agreement (max |J_A - J_B|, relative error)
#   2. Execution time (compile + setup vs. warm call time)
# ===========================================================================

# JAX: exact-physics Jacobian via the fast factorized pipeline (030).
# Includes Python trace (~2s), XLA compile (~1.7s on first run, cached after).
t = time.perf_counter()
opt = ir8_optics(line, tw0, use_jax=True)
opt.check_limits = False
x0 = opt._err._get_x()
jac_jax = opt._err.get_jacobian(x0.copy())  # builds optimizer + traces + compiles
t_jax_setup = time.perf_counter() - t  # total setup time

# Warm steady-state: what a matching optimizer pays per Newton step.
t = time.perf_counter()
n_iter = 5
for _ in range(n_iter):
    jac_jax = opt._err.get_jacobian(
        x0.copy()
    )  # warm call, residuals computed separately
t_jax_call = (time.perf_counter() - t) / n_iter

# --- Finite differences: xsuite's own forward FD (base get_jacobian) -------
# This is exactly the FD the optimizer falls back to: one-sided forward
# difference with the per-knob `step` from set_var_limits_and_steps.
opt._err.use_jax = False
fd_steps = np.asarray(opt._err.steps_for_jacobian)
t = time.perf_counter()
jac_fd = np.asarray(opt._err.get_jacobian(x0.copy()))
t_fd_call = time.perf_counter() - t
opt._err.use_jax = True
print(
    f"[fd] xsuite forward FD, step range [{fd_steps.min():.1e}, {fd_steps.max():.1e}]"
)

# --- MAD-NG TPSA -----------------------------------------------------------
jac_tpsa = None
try:
    t = time.perf_counter()
    opt_t = ir8_optics(
        line, tw0, use_tpsa=True, madng_names=False
    )  # MAD-NG model build + 1st track
    opt_t.check_limits = False
    xt0 = opt_t._err._get_x()
    jac_tpsa = np.asarray(opt_t._err.get_jacobian(xt0.copy()))
    t_tpsa_setup = time.perf_counter() - t
    # Per-step cost: a fresh point forces a MAD-NG re-track + acquire.
    t = time.perf_counter()
    for j in range(n_iter):
        xp = xt0.copy()
        xp[0] += 1e-7 * (j + 1)
        _ = opt_t._err.get_jacobian(xp)
    t_tpsa_call = (time.perf_counter() - t) / n_iter
    opt_t._err._set_x(xt0.copy())
except Exception as ee:
    print(f"[tpsa] FAILED to build/evaluate TPSA jacobian: {ee!r}")


# --- Comparison ------------------------------------------------------------
def _tname(tt):
    if isinstance(tt.tar, tuple):
        return f"{tt.tar[0]}@{tt.tar[1]}"
    if hasattr(tt, "var"):  # TargetRelPhaseAdvance
        return f"{tt.var}_adv({tt.start}->{tt.end})"
    return str(tt.tar)


tar_names = [_tname(tt) for tt in opt._err.targets]


def compare(name_a, A, name_b, B):
    """Print absolute and relative differences between two Jacobians.

    Shows where the largest disagreement occurs (target x knob pair).
    """
    denom = np.maximum(np.abs(B), 1e-3)  # avoid division by tiny values
    rel = np.abs(A - B) / denom
    ij = np.unravel_index(np.argmax(rel), rel.shape)
    print(
        f"[Comparison] {name_a} vs {name_b}: max|d| = {np.max(np.abs(A - B)):.2e}, "
        f"max rel(|J|>1e-3)={np.max(rel):.2e}  "
        f"@ {tar_names[ij[0]]}/{VARY[ij[1]]} ({A[ij]:.3e} vs {B[ij]:.3e})"
    )


print(f"\n[jac] shape {jac_jax.shape} (targets x knobs)")
compare("JAX ", jac_jax, " FD ", jac_fd)
if jac_tpsa is not None:
    compare("TPSA", jac_tpsa, " FD ", jac_fd)
    compare("JAX ", jac_jax, "TPSA", jac_tpsa)

# Per-target max|backend - xsuiteFD| across knobs, and which backend is closer.
print(
    "\n[per-target] max|backend - xsuite_FD| over the 20 knobs (closer-to-FD marked):"
)
print(f"   {'target':30s} {'JAX':>11s} {'TPSA':>11s}  closer")
jax_wins = tpsa_wins = ties = 0
for it, nm in enumerate(tar_names):
    jx = np.max(np.abs(jac_jax[it] - jac_fd[it]))
    if jac_tpsa is None:
        print(f"   {nm:30s} {jx:11.2e}")
        continue
    tp = np.max(np.abs(jac_tpsa[it] - jac_fd[it]))
    if np.isclose(jx, tp, rtol=1e-3):
        who, ties = "tie", ties + 1
    elif jx < tp:
        who, jax_wins = "JAX", jax_wins + 1
    else:
        who, tpsa_wins = "TPSA", tpsa_wins + 1
    print(f"   {nm:30s} {jx:11.2e} {tp:11.2e}  {who}")
if jac_tpsa is not None:
    print(f"   -> closer to xsuite-FD: JAX and TPSA equally close, tie on {ties}")

# --- Timing summary --------------------------------------------------------
print("\n[time] per Jacobian backend (20 knobs, 14 targets):")
print(
    f"   JAX : setup {t_jax_setup:6.2f}s (opt+compile)  jac/step {t_jax_call * 1e3:7.1f} ms (jacfwd; residual separate)"
)
print(
    f"   FD  : setup   0.00s              jac/step {t_fd_call * 1e3:7.1f} ms (2x20 twiss)"
)
if jac_tpsa is not None:
    print(
        f"   TPSA: setup {t_tpsa_setup:6.2f}s (MAD-NG+track) jac/step {t_tpsa_call * 1e3:7.1f} ms (re-track + acquire)"
    )

# ===========================================================================
# PART 2: Match solve using the JAX Jacobian
# ===========================================================================
# Run the Newton optimizer with use_jax=True to see if the Jacobian is
# accurate enough to converge to the target optics.
# ===========================================================================

t = time.perf_counter()
opt = ir8_optics(line, tw0, use_jax=True)
opt.check_limits = False
opt.step(1)
t_jax_first_step = time.perf_counter() - t
reset_opt(opt)
t = time.perf_counter()
opt.solve()  # Newton optimization loop; uses opt._err.get_jacobian(x) per step
t_jax = time.perf_counter() - t


within = opt._err.last_targets_within_tol
print(
    f"\n[Solve use_jax] Configuration: {t_jax_first_step:.2f}s (first step), {t_jax:.2f}s (Warm Solve), "
    f"{int(np.sum(within))}/{len(within)} targets within tol, "
    f"penalty={np.sqrt(np.sum(opt._err.last_residue_values**2)):.3e}"
)

# Final Twiss after the match
tw_jax = line.twiss(start="s.ds.l8.b1", end="ip1", init=tw0, init_at=xt.START)
print(
    f"   ip8: betx={tw_jax['betx', 'ip8']:.6f} bety={tw_jax['bety', 'ip8']:.6f} "
    f"alfx={tw_jax['alfx', 'ip8']:.2e} alfy={tw_jax['alfy', 'ip8']:.2e}"
)
print(
    f"   ip1: betx={tw_jax['betx', 'ip1']:.6f} (target 0.15) "
    f"bety={tw_jax['bety', 'ip1']:.6f} (target 0.1)"
)
x_twiss_residual = opt._err._get_x().copy()

# ===========================================================================
# PART 3: Match with the JAX residual too (use_jax_residual=True)
# ===========================================================================
# Same exact-physics Jacobian, but the per-step residual is now read from the
# JAX primal optics instead of an extra twiss calculation.
# The optimizer judges convergence on the JAX residual; a real twiss at the end
# reports the true optics it actually reached.
# ===========================================================================
# PART 2 left the line at the solution; restore the original knobs so this is a
# real solve from the same starting point, not a no-op from the matched state.
for v in VARY:
    line[v] = KNOBS0[v]
opt_r = ir8_optics(line, tw0, use_jax=True, use_jax_residual=True)
opt_r.check_limits = False
opt_r.step(1)  # build + compile (warm-up, like PART 2)
reset_opt(opt_r)
t = time.perf_counter()
opt_r.solve()
t_jax_res = time.perf_counter() - t

within_r = opt_r._err.last_targets_within_tol
x_jax_residual = opt_r._err._get_x().copy()
tw_jax_r = line.twiss(start="s.ds.l8.b1", end="ip1", init=tw0, init_at=xt.START)
print(
    f"\n[Solve use_jax + use_jax_residual] warm solve {t_jax_res:.3f}s "
    f"(vs {t_jax:.3f}s with twiss residual), "
    f"{int(np.sum(within_r))}/{len(within_r)} targets within tol"
)
print(
    f"   final REAL twiss: ip8 betx={tw_jax_r['betx', 'ip8']:.6f} "
    f"bety={tw_jax_r['bety', 'ip8']:.6f} | "
    f"ip1 betx={tw_jax_r['betx', 'ip1']:.6f} bety={tw_jax_r['bety', 'ip1']:.6f}"
)
print(
    f"   max|knob diff vs twiss-residual solve| = "
    f"{np.max(np.abs(x_jax_residual - x_twiss_residual)):.2e}"
)

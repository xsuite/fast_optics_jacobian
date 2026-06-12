"""Can jax.experimental.jet (Taylor-mode AD) give us TPSA-like Taylor maps and
normal-form quantities, instead of our jacfwd-based derivatives?

`jet` implements univariate Taylor-mode AD, and can give thus one expand
in one direction. One call gives a truncated Taylor series for one direction.

Short answer
------------
* YES for the *map itself* and for *single-direction* high-order expansions
  (momentum -> chromaticity to any order; amplitude -> detuning).  `jet` is
  Taylor-mode AD: one call returns the whole truncated Taylor series along a
  chosen direction, far cheaper and higher-order than nesting jacfwd.
* It is a BUILDING BLOCK, not a drop-in TPSA: `jet` is *univariate* (expansion
  along one direction in one call).  MAD-NG normal forms need the full
  *multivariate* one-turn DA map (all mixed monomials in x,px,y,py,...); with
  JAX you assemble that from directional jets / nested jacfwd, then run the
  normal-form (Lie) algebra on the extracted polynomial.  JAX has no native
  multivariate TPSA and no jet rule for arccos, so tunes are read off the
  map/trace as a post-step (which is what normal-form codes do anyway).

This file demonstrates the two pieces and validates them against our current
jacfwd derivatives.

Convention note (verified below): jet's output terms are the derivatives
f', f'', f''' ... (NOT the factorial-scaled Taylor coefficients).
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental.jet import jet
import time

jax.config.update("jax_enable_x64", True)

from xtrack.jax_optics import drift_exact_jax  # exact drift, reused

print("=" * 70)
print("0. jet convention check (output terms = derivatives f', f'', ...)")
print("=" * 70)
y0, (d1, d2, d3) = jet(lambda x: x**3, (2.0,), ([1.0, 0.0, 0.0],))
print(f"   x^3 at 2:  f={y0}  f'={d1} (=12)  f''={d2} (=12)  f'''={d3} (=6)")


# ===========================================================================
# 1. TAYLOR MAP of a nonlinear element via jet (the TPSA building block).
#    A thin sextupole followed by an exact drift; expand the one-element map
#    x_out(x_in) to 3rd order along a phase-space direction.
# ===========================================================================
print("\n" + "=" * 70)
print("1. Taylor map of (sextupole o drift) via jet  [state = x,px,y,py,z,d]")
print("=" * 70)
BETA0 = 0.999


def sext_drift(s, g=120.0, L=0.5):
    x, px, y, py, z, d = s
    px = px - 0.5 * g * (x * x - y * y)  # normal sextupole kick
    py = py + g * x * y
    return drift_exact_jax(jnp.array([x, px, y, py, z, d]), L, BETA0)


s0 = jnp.array([1e-3, 0.0, 0.5e-3, 0.0, 0.0, 0.0])  # expansion point
v = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # direction: +x
zero = jnp.zeros(6)
y0, (o1, o2, o3) = jet(sext_drift, (s0,), ([v, zero, zero],))

R_dir = jax.jacfwd(sext_drift)(s0) @ v  # our current way (linear)
print(f"   order-1 term o1 (= d map/d x):     {np.asarray(o1)}")
print(f"   jacfwd directional derivative J.v: {np.asarray(R_dir)}")
print(f"   -> match: {np.allclose(o1, R_dir)}  (jet's 1st order == jacfwd)")
print(f"   order-2 term o2 (d^2 map/dx^2):    {np.asarray(o2)}")
print(f"      = the sextupole's quadratic map term (-g x on px): -g = {-120.0}")
print(f"   order-3 term o3:                   {np.asarray(o3)}")
print("   -> one jet call yields the full Taylor MAP along x; these monomial")
print("      coefficients are exactly the DA/TPSA data normal forms consume.")


# ===========================================================================
# 2. HIGH-ORDER CHROMATICITY via jet on the one-turn-matrix TRACE.
#    A thin-lens FODO with the exact (1+delta) momentum scaling.  arccos has no
#    jet rule, so we jet the trace T(delta) (cos/sin/sqrt/div are supported) and
#    read the tune + chromaticities off it - the normal-form way.
# ===========================================================================
print("\n" + "=" * 70)
print("2. Chromaticity to high order: jet the one-turn TRACE T(delta)")
print("=" * 70)
K0, LD = 0.6, 1.0  # integrated half-quad strength, drift length


def one_turn_trace(delta):
    sc = 1.0 / (1.0 + delta)  # exact (1+delta) momentum scaling
    k = K0 * sc
    Le = LD * sc
    QF = jnp.array([[1.0, 0.0], [-k, 1.0]])
    QD = jnp.array([[1.0, 0.0], [k, 1.0]])
    D = jnp.array([[1.0, Le], [0.0, 1.0]])
    M = D @ QD @ D @ QF
    return M[0, 0] + M[1, 1]


def tune(delta):
    return jnp.arccos(jnp.clip(0.5 * one_turn_trace(delta), -1.0, 1.0)) / (2 * jnp.pi)


# --- jet: full delta-series of the trace in ONE call -----------------------
t_0 = time.perf_counter()
T0, (T1, T2, T3) = jet(one_turn_trace, (0.0,), ([1.0, 0.0, 0.0],))
c = 0.5 * T0
s = np.sqrt(1.0 - c * c)
Q0 = np.arccos(c) / (2 * np.pi)
# theta = 2 pi Q = arccos(T/2);  chain rule from the trace derivatives:
th1 = -0.5 * T1 / s
th2 = -(0.5 * T2 + c * th1**2) / s
Qp_jet, Qpp_jet = th1 / (2 * np.pi), th2 / (2 * np.pi)
t_1 = time.perf_counter()

jet_time = t_1 - t_0

# --- our current way: nested jacfwd through the tune (arccos OK under jacfwd)
t_0 = time.perf_counter()
Qp_ref = float(jax.jacfwd(tune)(0.0))
t_1 = time.perf_counter()
Qpp_ref = float(jax.jacfwd(jax.jacfwd(tune))(0.0))
t_2 = time.perf_counter()

qp_jax_time = t_1 - t_0
qpp_jax_time = t_2 - t_1

print(f"   tune Q0 = {Q0:.6f}")
print(
    f"   Q'  (linear chromaticity):  jet={Qp_jet:+.6f} ({jet_time * 1e3:.1f} ms)   jacfwd={Qp_ref:+.6f} ({qp_jax_time * 1e3:.1f} ms)"
)
print(
    f"   Q'' (2nd-order chroma):      jet={Qpp_jet:+.6f} ({jet_time * 1e3:.1f} ms)   "
    f"jacfwd(jacfwd)={Qpp_ref:+.6f} ({qpp_jax_time * 1e3:.1f} ms)"
)
print(
    f"   -> match Q': {np.isclose(Qp_jet, Qp_ref)}   "
    f"match Q'': {np.isclose(Qpp_jet, Qpp_ref)}"
)
print("   The single jet call already carries T''' (3rd order) too -> Q''' etc.")
print("   come for free, whereas matching them with jacfwd needs another nest.")


# ===========================================================================
# 3. Verdict.
# ===========================================================================
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)
print("""\
 * jet = Taylor-mode AD: ideal for the derivatives we actually compute that are
   expansions in ONE parameter - chromaticity dQ/ddelta (and dd, ddd...).
   One call, all orders, cheaper than nesting jacfwd.
 * Practical caveats: no jet rule for arccos (read tunes off the trace/map, as
   here) and a few other ops.  But every element map we
   use (drift/quad/bend/sextupole) is jet-compatible.""")

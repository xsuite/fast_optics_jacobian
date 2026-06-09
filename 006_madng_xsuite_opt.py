from pyprof import timing

from utils import ir8_optics, load_hllhc_b1

# Load LHC model (also sets the per-circuit match limits/steps)
collider, line = load_hllhc_b1()

# The IR8 optics match driven by the MAD-NG TPSA Jacobian (see utils.ir8_optics).
opt = ir8_optics(line, use_tpsa=True)

timing.reset()
timing.start_timing("Xsuite_Opt_MADNG")
opt.step(60)
timing.stop_timing()
timing.report()

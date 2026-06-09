from utils import orbit_correction

# Orbit-bump match driven by the MAD-NG TPSA Jacobian
opt = orbit_correction(use_tpsa=True)

jac_ng = opt._err.get_jacobian(opt._err._get_x())
opt._err.call_counter = 0
opt.clear_log()
opt.solve(cleanup_madng_tpsa=False)

# tw = line.madng_twiss(betx=1, bety=1)
# tw.plot('x y')

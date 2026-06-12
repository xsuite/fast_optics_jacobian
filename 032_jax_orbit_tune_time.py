import xtrack as xt
from pyprof import timing
from utils import reset_opt, tune_chroma, orbit_correction


# Load a line and build a tracker
line = xt.Line.from_json("lattice_data/lhc_thick_with_knobs.json")
line.build_tracker()

# Match tunes and chromaticities to assigned values
timing.start_timing("Chroma Setup Matching + 1st Step")
opt_chroma = tune_chroma(line, use_jax=True)
opt_chroma.step(1)
timing.stop_timing()
reset_opt(opt_chroma)
timing.start_timing("Chroma Warm solve")
opt_chroma.solve()
timing.stop_timing()

# Steer a closed-orbit bump with four corrector pairs (synthetic line built
# inside orbit_correction)
timing.start_timing("Orbit Setup + 1st Step")
opt_orbit = orbit_correction(use_jax=True)
opt_orbit.step(1)
timing.stop_timing()
reset_opt(opt_orbit)
timing.start_timing("Orbit Warm solve")
opt_orbit.solve()
timing.stop_timing()
timing.report()

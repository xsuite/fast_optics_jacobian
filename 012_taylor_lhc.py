import xtrack as xt
from xtrack._temp import lhc_match as lm
import numpy as np
import matplotlib.pyplot as plt
from xtrack.madng_interface import madng_get_init
from tpsa_util import TPSA

# Load LHC model
collider = xt.Environment.from_json(
    '../xtrack/test_data/hllhc15_thick/hllhc15_collider_thick.json')
collider.vars.load_madx(
    '../xtrack/test_data/hllhc15_thick/opt_round_150_1500.madx')

collider.build_trackers()

line = collider.lhcb1

# Initial twiss
tw0 = line.twiss()

# Prepare for optics matching: set limits and steps for all circuits
lm.set_var_limits_and_steps(collider)

mng = line.to_madng(sequence_name='lhcb1')

# Twiss on a part of the machine (bidirectional)
start = 's.ds.l8.b1'
end = 'ip1'
tw = line.twiss(start=start, end=end, init=tw0)

X0 = madng_get_init(line, at=start)
tw_ng = line.madng_twiss(start=start, end=end, X0=X0, init=tw0)

beta0 = line.particle_ref.beta0[0]

init_coord = np.array([1e-3, 0, 0, 0, 0, 0])
particles = line.build_particles(x=init_coord[0], px=init_coord[1], y=init_coord[2], py=init_coord[3], zeta=init_coord[4] * beta0, delta=init_coord[5])

mng.send(r'''
    local obs_flag = MAD.element.flags.observed

    local X0 = MAD.damap {
        nv=6, -- number of variables
        mo=4, -- max order of variables
        np=0, -- number of parameters
    }

    X0.x = 1e-3

    trk, mflw = MAD.track{sequence=MADX.lhcb1, X0=X0, range='s.ds.l8.b1/ip1', savemap=true}
    trk:print({'name', 's', 'x', 'px', 'y', 'py', 't', 'pt'})

    local clearkeys in MAD.utility
    py:send(clearkeys(mflw[1].__vn), true) -- Send keys as a list (ordered)

    for i, v in ipairs(mflw[1].__vn) do
        py:send(mflw[1][v]) -- Send TPSAs over in order
    end

''')

tpsas = {k: mng.recv() for k in mng.recv()} # Create dict out of TPSAs

tpsa = TPSA(tpsas, num_variables=6) # Create TPSA object out of madng-dict

line.track(particles, ele_start=start, ele_stop=end)

perturbed_coord = np.array([init_coord[0] * 0.9, init_coord[1] * 0.9, init_coord[2] * 1.1, init_coord[3] * 1.1, init_coord[4] * 1, init_coord[5] * 1])

delta = perturbed_coord - init_coord
x_pert = tpsa.get_taylor_expansion_all(delta)

particles = line.build_particles(x=perturbed_coord[0], px=perturbed_coord[1], y=perturbed_coord[2],
                                 py=perturbed_coord[3], t=perturbed_coord[4], pt=perturbed_coord[5])
line.track(particles, ele_start=start, ele_stop=end)
perturbed_particles_arr = np.array([particles.x[0], particles.px[0], particles.y[0],
                                   particles.py[0], particles.zeta[0], particles.delta[0]])

print(x_pert - perturbed_particles_arr)

from xtrack.beam_elements import TPSAMap

tpsa_map_element = TPSAMap(
    length=1.0,
    x_monomials=tpsas['x'][0],
    y_monomials=tpsas['y'][0],
    px_monomials=tpsas['px'][0],
    py_monomials=tpsas['py'][0],
    zeta_monomials=tpsas['t'][0],
    delta_monomials=tpsas['pt'][0],
    x_coefficients=tpsas['x'][1],
    y_coefficients=tpsas['y'][1],
    px_coefficients=tpsas['px'][1],
    py_coefficients=tpsas['py'][1],
    zeta_coefficients=tpsas['t'][1],
    delta_coefficients=tpsas['pt'][1],
    base_coordinates=init_coord,
    map_machine_vals=np.array([]),
    machine_vals=np.array([]),
)

from pyprof import timing

n_times = 10

for i in range(n_times):
    particles = line.build_particles(x=perturbed_coord[0], px=perturbed_coord[1], y=perturbed_coord[2],
                                        py=perturbed_coord[3], zeta=perturbed_coord[4] * beta0, delta=perturbed_coord[5])
    timing.start_timing("xsuite_track")
    line.track(particles)
    timing.stop_timing("xsuite_track")

for i in range(n_times):
    delta = perturbed_coord - init_coord
    timing.start_timing("tpsa_expansion")
    particles_pert_taylor = tpsa.get_taylor_expansion_all(delta)
    timing.stop_timing("tpsa_expansion")

for i in range(n_times):
    particles = line.build_particles(x=perturbed_coord[0], px=perturbed_coord[1], y=perturbed_coord[2],
                                        py=perturbed_coord[3], zeta=perturbed_coord[4] * beta0, delta=perturbed_coord[5])
    timing.start_timing("xsuite_tpsa_map_track")
    tpsa_map_element.track(particles)
    timing.stop_timing("xsuite_tpsa_map_track")

timing.report()
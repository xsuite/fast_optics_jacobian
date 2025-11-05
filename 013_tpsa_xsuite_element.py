from tkinter.font import names
import xtrack as xt
import numpy as np
from tpsa_util import TPSA

env = xt.Environment()
env['kk'] = 0.1

elems = [env.new('quad', xt.Quadrupole, length=1, k1='kk')]
start_elem = env.new('start', xt.Marker)
end_elem = env.new('end', xt.Marker)

line = env.new_line(components=[
    start_elem,
    elems[0],
    end_elem,
])

line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, q0=1)
beta0 = line.particle_ref.beta0[0]
n_part = 1
delta = 0
init_coord = np.array([-1e-3, 0, 0, 0, 0, 0]) # MAD-NG coordinates
part_order = ['x', 'px', 'y', 'py', 't', 'pt']

param_vec = np.concatenate((init_coord, [env['kk']]))

particles = line.build_particles(x=init_coord[0], px=init_coord[1], y=init_coord[2], py=init_coord[3], zeta=init_coord[4] * beta0, delta=delta)

if line.tracker is None:
    line.build_tracker()

coord_str = ''
for part, val in zip(part_order, init_coord):
    if np.abs(val) > 1e-12:
        coord_str += f'X0.{part} = {val} '

mng = line.to_madng()
mng.send(r'''
    local obs_flag = MAD.element.flags.observed

    MADX.seq:select(obs_flag, {list={'start', 'end'}})

    local X0 = MAD.damap {
        nv=6, -- number of variables
        mo=3, -- max order of variables
        np=1, -- number of parameters
        pn={'kk'}, -- parameter names
    }
    '''
    +
    coord_str
    +
    r'''
    MADX.kk = MADX.kk + X0.kk

    trk, mflw = MAD.track{sequence=MADX.seq, X0=X0, savemap=true}

    -- trk:print({'name', 's', 'x', 'px', 'y', 'py', 't', 'pt'})

    local clearkeys in MAD.utility
    py:send(clearkeys(mflw[1].__vn), true) -- Send keys as a list (ordered)

    for i, v in ipairs(mflw[1].__vn) do
        py:send(mflw[1][v]) -- Send TPSAs over in order
    end

''')

tpsas = {k: mng.recv() for k in mng.recv()} # Create dict out of TPSAs

tpsa = TPSA(tpsas, num_variables=6) # Create TPSA object out of madng-dict

#####

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
    map_machine_vals=np.array([env['kk']]),
    machine_vals=np.array([env['kk']]),
)

#####

perturbed_coord = init_coord.copy()
perturbed_coord = np.array([init_coord[0] * 0.9, init_coord[1] * 0.9, init_coord[2] * 1.1, init_coord[3] * 1.1, init_coord[4] * 1, init_coord[5] * 1])
perturbed_param_vec = np.concatenate((perturbed_coord, [env['kk']]))

delta = perturbed_param_vec - param_vec
particles_pert_taylor = tpsa.get_taylor_expansion_all(delta)

particles = line.build_particles(x=perturbed_coord[0], px=perturbed_coord[1], y=perturbed_coord[2],
                                    py=perturbed_coord[3], zeta=perturbed_coord[4] * beta0, delta=perturbed_coord[5])

line.track(particles)

perturbed_particles_arr = np.array([particles.x[0], particles.px[0], particles.y[0],
                                    particles.py[0], particles.zeta[0], particles.delta[0]])

particles = line.build_particles(x=perturbed_coord[0], px=perturbed_coord[1], y=perturbed_coord[2],
                                    py=perturbed_coord[3], zeta=perturbed_coord[4] * beta0, delta=perturbed_coord[5])
tpsa_map_element.track(particles)

perturbed_particles_arr_map = np.array([particles.x[0], particles.px[0], particles.y[0],
                                        particles.py[0], particles.zeta[0], particles.delta[0]])

print("Initial Particle: ", init_coord)
#print("Perturbed Particle: ", perturbed_coord)

print(f"After Track Xsuite Track: {perturbed_particles_arr}\n")
print(f"After Taylor Expansion TPSA: {particles_pert_taylor}\n")
print(f"Xsuite TPSAMap Track: {perturbed_particles_arr_map}\n")

print(f"Difference between TPSA and Xsuite Track: {particles_pert_taylor - perturbed_particles_arr}\n")
print(f"Difference between TPSA and Xsuite TPSAMap Track: {particles_pert_taylor - perturbed_particles_arr_map}\n")

print("Now let's do some timings")

from pyprof import timing

n_times = 10

for i in range(n_times):
    particles = line.build_particles(x=perturbed_coord[0], px=perturbed_coord[1], y=perturbed_coord[2],
                                        py=perturbed_coord[3], zeta=perturbed_coord[4] * beta0, delta=perturbed_coord[5])
    timing.start_timing("xsuite_track")
    line.track(particles)
    timing.stop_timing("xsuite_track")

for i in range(n_times):
    delta = perturbed_param_vec - param_vec
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
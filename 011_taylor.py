import xtrack as xt
import numpy as np
from tpsa_util import TPSA

# Create line with only one quadrupole of length 1

env = xt.Environment()
env['k1'] = 0.01

line = env.new_line(components=[
    env.new('start', xt.Marker),
    env.new('s1', xt.Quadrupole, length=1, k1='k1'),
    env.new('end', xt.Marker),
])

line.particle_ref = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=-1)
n_part = 1
init_coord = np.array([-1e-3, 0, 0, 0, 0, 0])
param_vec = np.concatenate((init_coord, [env['k1']]))

particles = line.build_particles(x=-1e-3, px=0, y=0, py=0, t=0, pt=0)
line.build_tracker()

mng = line.to_madng()
mng.send(r'''
    local obs_flag = MAD.element.flags.observed

    MADX.seq:select(obs_flag, {list={'start', 'end'}})

    local X0 = MAD.damap {
        nv=6, -- number of variables
        mo=4, -- max order of variables
        np=1, -- number of parameters
        pn={'k1'}, -- parameter names
    }

    X0.x = -1e-3

    MADX.k1 = MADX.k1 + X0.k1
    trk, mflw = MAD.track{sequence=MADX.seq, X0=X0, savemap=true}
    trk:print({'name', 's', 'x', 'px', 'y', 'py', 't', 'pt'})

    local clearkeys in MAD.utility
    py:send(clearkeys(mflw[1].__vn), true) -- Send keys as a list (ordered)

    for i, v in ipairs(mflw[1].__vn) do
        py:send(mflw[1][v]) -- Send TPSAs over in order
    end

''')

tpsas = {k: mng.recv() for k in mng.recv()} # Create dict out of TPSAs

tpsa = TPSA(tpsas, num_variables=6) # Create TPSA object out of madng-dict

line.track(particles)

perturbed_coord = np.array([-1e-3 * 0.9, 0, 0, 0, 0, 0])
perturbed_param_vec = np.concatenate((perturbed_coord, [env['k1']]))

delta = perturbed_param_vec - param_vec
x_pert = tpsa.get_taylor_expansion_all(delta)

particles = line.build_particles(x=perturbed_coord[0], px=perturbed_coord[1], y=perturbed_coord[2],
                                 py=perturbed_coord[3], t=perturbed_coord[4], pt=perturbed_coord[5])
line.track(particles)
perturbed_particles_arr = np.array([particles.x[0], particles.px[0], particles.y[0],
                                   particles.py[0], particles.zeta[0], particles.delta[0]])

print(x_pert - perturbed_particles_arr)
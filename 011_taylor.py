import xtrack as xt
import numpy as np
from tpsa_util import TPSA

# Create line with only one quadrupole of length 1

def dp2pt(dp, beta0):
    """Convert relative momentum deviation dp/p to transverse momentum pt/p.

    Parameters
    ----------
    dp : float
        Relative momentum deviation (dp/p, dimensionless).
    beta0 : float
        Particle relativistic beta (v/c).

    Returns
    -------
    float
        Transverse momentum relative to total momentum (pt/p, dimensionless).
    """
    _beta0 = 1 / beta0
    return np.sqrt((1 + dp) ** 2 + (_beta0**2 - 1)) - _beta0

env = xt.Environment()
env['kk'] = 0.1
env['kks'] = 0.01

# Create lines where each element is using something else
# There should always be a Marker at the beginning and at the end
# And then: Bend, Quadrupole, Sextupole, Octupole and Skew variants of those as well
names = ['Bend', 'Quad', 'Sext', 'Oct', 'SkewQ', 'SkewS', 'SkewOct']
elems = [
    env.new(names[0], xt.Bend, length=1, angle='kk*0.1', k0_from_h=True),
    env.new(names[1], xt.Quadrupole, length=1, k1='kk'),
    env.new(names[2], xt.Sextupole, length=1, k2='kk'),
    env.new(names[3], xt.Octupole, length=1, k3='kk'),
    env.new(names[4], xt.Quadrupole, length=1, k1='kk', k1s='kks'),
    env.new(names[5], xt.Sextupole, length=1, k2='kk', k2s='kks'),
    env.new(names[6], xt.Octupole, length=1, k3='kk', k3s='kks'),
]

start_elem = env.new('start', xt.Marker)
end_elem = env.new('end', xt.Marker)

for i in range(len(names)):
    print(f'--- Testing element: {names[i]} ---')
    line = env.new_line(components=[
        start_elem,
        elems[i],
        end_elem,
    ])

    line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, q0=1)
    beta0 = line.particle_ref.beta0[0]
    n_part = 1
    delta = 0
    init_coord = np.array([-1e-3, 1e-6, 1e-3, -1e-6, 0, dp2pt(delta, beta0)]) # MAD-NG coordinates
    part_order = ['x', 'px', 'y', 'py', 't', 'pt']

    param_vec = np.concatenate((init_coord, [env['kk'], env['kks']]))


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
            np=2, -- number of parameters
            pn={'kk', 'kks'}, -- parameter names
        }
        '''
        +
        coord_str
        +
        r'''
        MADX.kk = MADX.kk + X0.kk
        MADX.kks = MADX.kks + X0.kks

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

    line.track(particles)

    perturbed_coord = np.array([init_coord[0] * 0.9, init_coord[1] * 0.9, init_coord[2] * 1.1, init_coord[3] * 1.1, init_coord[4] * 1, init_coord[5] * 1])
    perturbed_param_vec = np.concatenate((perturbed_coord, [env['kk'], env['kks']]))


    delta = perturbed_param_vec - param_vec
    x_pert = tpsa.get_taylor_expansion_all(delta)

    particles = line.build_particles(x=perturbed_coord[0], px=perturbed_coord[1], y=perturbed_coord[2],
                                    py=perturbed_coord[3], zeta=perturbed_coord[4] * beta0, delta=perturbed_coord[5])
    line.track(particles)
    perturbed_particles_arr = np.array([particles.x[0], particles.px[0], particles.y[0],
                                    particles.py[0], particles.zeta[0], particles.delta[0]])

    print("Initial Particle: ", init_coord)
    print("Perturbed Particle: ", perturbed_coord)

    print(f"Difference between TPSA and Xsuite Track: {x_pert - perturbed_particles_arr}\n")
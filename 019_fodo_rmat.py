import xtrack as xt

import numpy as np
import matplotlib.pyplot as plt

import xtrack as xt

# Build a simple ring
colour = 'gray'

# Prepare the lattice and plot the beam
# =====================================

# plt.style.use('../../latex.mplstyle')

# Create an environment
env = xt.Environment()
env.vars.default_to_zero = True

# Build a simple ring
env['mq1'] = 0.5
env['mq2'] = 0.5
env['mqs1'] = 0.01

line = env.new_line(components=[
    env.new('START', xt.Marker),
    env.new('d0.1', xt.Drift, length=2.5),

    env.new('mqf.1', xt.Quadrupole, length=0.6, k1='mq1', k1s='mqs1'),
    env.new('d1.1',  xt.Drift, length=2.5),
    env.new('mk.1', xt.Marker),
    env.new('d1.2',  xt.Drift, length=2.5),

    env.new('mqd.1', xt.Quadrupole, length=0.6, k1='-mq1'),
    env.new('d3.1',  xt.Drift, length=5),

    env.new('mqf.2', xt.Quadrupole, length=0.6, k1='mq2'),
    env.new('d1.3',  xt.Drift, length=2.5),
    env.new('mk.2', xt.Marker),
    env.new('d1.4',  xt.Drift, length=2.5),

    env.new('mqd.2', xt.Quadrupole, length=0.6, k1='-mq2'),
    env.new('d3.2',  xt.Drift, length=2.5),
    env.new('END', xt.Marker),
])

kin_energy_0 = 50e6 # 50 MeV
line.particle_ref = xt.Particles(energy0=kin_energy_0 + xt.PROTON_MASS_EV, # total energy
                                 mass0=xt.PROTON_MASS_EV)

tw = line.twiss4d()

mng = line.to_madng()

opt = line.match(
    init=tw,
    solve=False,
    vary=xt.VaryList(['mq1', 'mqs1']),
    targets=[
        #xt.TargetSet(betx=2.0, at='END')
        xt.TargetRmatrix(start='START', end='END', r14=-0.09)
    ],
    use_tpsa=True
)
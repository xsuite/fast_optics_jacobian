import xtrack as xt

env = xt.Environment()
env.vars.default_to_zero = True
line = env.new_line(length=10, components=[
    env.new('corr1', xt.Multipole, isthick=True,
            knl=['kick_h_1'], ksl=['kick_v_1'], length=0.1, at=1),
    env.new('corr2', xt.Multipole, isthick=True,
            knl=['kick_h_2'], ksl=['kick_v_2'], length=0.1, at=2),
    env.new('corr3', xt.Multipole, isthick=True,
            knl=['kick_h_3'], ksl=['kick_v_3'], length=0.1, at=8),
    env.new('corr4', xt.Multipole, isthick=True,
            knl=['kick_h_4'], ksl=['kick_v_4'], length=0.1, at=9),
    env.new('mid', xt.Marker, at=5),
    env.new('end', xt.Marker, at=10)
    ])
line.set_particle_ref('proton', p0c=26e9)


mng = line.to_madng()

opt = line.match(
    solve=False,
    betx=1, bety=1,
    vary=xt.VaryList(['kick_h_1', 'kick_v_1',
                      'kick_h_2', 'kick_v_2',
                      'kick_h_3', 'kick_v_3',
                      'kick_h_4', 'kick_v_4']),
    targets=[
        xt.TargetSet(x=1e-3, y=-2e-3, px=0, py=0, at='mid'),
        xt.TargetSet(x=0, y=0, px=0, py=0, at='end'),
    ],
    use_tpsa=True
)

jac_ng = opt._err.get_jacobian(opt._err._get_x())
opt._err.call_counter = 0
opt.clear_log()
opt.solve(cleanup_madng_tpsa=False)

#tw = line.madng_twiss(betx=1, bety=1)

#tw.plot('x y')
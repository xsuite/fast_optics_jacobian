import numpy as np
from tpsa_util import TPSA

madx_src ='''

kf = 0.01;
kd = -0.01;

b1: sbend, l=1, angle=0.1;
q1: quadrupole, l=1, k1:=kf;
q2: quadrupole, l=1, k1:=kd;
c1: rfcavity, l=0.1, lag=180/360, volt=4, freq=400;
c2: rfcavity, l=0.1, lag=-20/360, volt=3, freq=400;
c3: rfcavity, l=0.1, lag=(180-20)/360, volt=3, freq=400;
mm: marker;
myseq: sequence, l=10;
    b1, at=1;
    q1, at=2.5;
    c1, at=4;
    c2, at=5;
    c3, at=6;
    q2, at=7.5;
    mm, at=10;
endsequence;
'''

# write to file
with open('temp_seq.madx', 'w') as fid:
    fid.write(madx_src)

# twiss with MAD-NG from madx file
import pymadng as pg
mng = pg.MAD()
mng.send(r'''
    MADX:load('temp_seq.madx')
    MADX.myseq.beam = MAD.beam{particle='positron', gamma=20000};
    MAD.element.rfcavity.method = 2;

    local tw = MAD.twiss{sequence=MADX.myseq}
    tw:print({'name', 's', 'beta11'})

    local obs_flag = MAD.element.flags.observed

    MADX.myseq:select(obs_flag, {list={'C1', 'MM'}})

    local X0 = MAD.damap {
        nv=6, -- number of variables
        mo=3, -- max order of variables
        np=2, -- number of parameters
        pn={'kf', 'kd'}, -- parameter names
    }
    X0.x = 1e-3
    MADX.kf = MADX.kf + X0.kf
    MADX.kd = MADX.kd + X0.kd
    trk, mflw = MAD.track{sequence=MADX.myseq, X0=X0, savemap=true}
    trk:print({'name', 's', 'x', 'px', 'y', 'py', 't', 'pt'})

    ! mflw[1]:print() -- map at the end
    ! py:send(mflw[1].x) -- send map back to python
    ! print(mflw[1].__td.nn) -- Monomial length
    ! print(mflw[1].__td.nv) -- Number of Variables
    ! print(MAD.tostring(mflw[1].__vn))

    local clearkeys in MAD.utility
    py:send(clearkeys(mflw[1].__vn), true) -- Send keys as a list (ordered)

    for i, v in ipairs(mflw[1].__vn) do
        py:send(mflw[1][v]) -- Send TPSAs over in order
    end
''')

tpsas = {k: mng.recv() for k in mng.recv()} # Create dict out of TPSAs

tpsa = TPSA(tpsas, num_variables=6) # Create TPSA object out of madng-dict

coeffs = tpsa.get_coeff('x', np.array([[1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0], [2,0,0,0,0,0,1,0]]))


mng.send(f"""
    ! trk['MM'].__map:print() -- map at MM

    local nf = MAD.gphys.normal(mflw[1]) -- Compute normal form

    print(MAD.tostring(nf))
    -- damaps --
    -- a: normalizing map (output)
    -- a0: map to go to fix point
    -- a1: map to go to CS basis
    -- a2: second order map
    -- m_n: normalized map
    -- m0: Backup one-turn map
    -- c: map to go to phasor basis
    -- ci: phasor inverse map

    local clearkeys in MAD.utility
    py:send(clearkeys(nf.a.__vn), true) -- Send keys as a list (ordered)

    for i, v in ipairs(nf.a.__vn) do
        py:send(nf.a[v]) -- Send TPSAs (Normal Forms) over in order
    end

    local B0 = MAD.gphys.map2bet(nf.a:real())
    print(B0.beta11)

    local a_re = nf.a:real()
    print(a_re.x:get("100000")^2 + a_re.x:get("010000")^2)
    print(a_re.x:get("1000001")^2)
""")

nfs = {k: mng.recv() for k in mng.recv()} # Create dict out of TPSAs
nf_tpsa = TPSA(nfs, num_variables=6) # Create TPSA object out of madng-dict
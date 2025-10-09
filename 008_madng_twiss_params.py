import xtrack as xt
from xtrack._temp import lhc_match as lm
import numpy as np
import time
import matplotlib.pyplot as plt

# Load LHC model
collider = xt.Environment.from_json(
    '../xtrack/test_data/hllhc15_thick/hllhc15_collider_thick.json')
collider.vars.load_madx(
    '../xtrack/test_data/hllhc15_thick/opt_round_150_1500.madx')

collider.build_trackers()

line = collider.lhcb1


mng = line.to_madng(sequence_name='lhcb1', keep_files=True, debug=True, redirect_stderr=True, stdout='log_madng.txt')

# Directly with send/recv start/end with initial conditions
mng.send("""abcd = twiss {sequence = lhcb1, range = '$start/$end', X0 = beta0 {betx = 1, bety = 1}}""")
tw1_ng = mng.abcd.to_df()

tw1_xsng = line.madng_twiss(beta11=1, beta22=1, xsuite_tw=False)

tw1_xs = line.twiss(betx=1, bety=1)

import xobjects as xo

xo.assert_allclose(tw1_xs['betx'], tw1_ng.beta11[:-1].to_list(), rtol=1e-8, atol=1e-5)
xo.assert_allclose(tw1_xs['bety'], tw1_ng.beta22[:-1].to_list(), rtol=1e-8, atol=1e-5)
xo.assert_allclose(tw1_xs['alfx'], tw1_ng.alfa11[:-1].to_list(), rtol=1e-8, atol=1e-5)
xo.assert_allclose(tw1_xs['alfy'], tw1_ng.alfa22[:-1].to_list(), rtol=1e-8, atol=1e-5)

assert len(tw1_xs.betx) == len(tw1_xsng.beta11_ng)
xo.assert_allclose(tw1_xs['betx'], tw1_xsng.beta11_ng, rtol=1e-8, atol=1e-5)
xo.assert_allclose(tw1_xs['bety'], tw1_xsng.beta22_ng, rtol=1e-8, atol=1e-5)
xo.assert_allclose(tw1_xs['alfx'], tw1_xsng.alfa11_ng, rtol=1e-8, atol=1e-5)
xo.assert_allclose(tw1_xs['alfy'], tw1_xsng.alfa22_ng, rtol=1e-8, atol=1e-5)

tw2_xs = line.twiss(start='s.ds.l8.b1', end='ip1', betx=100, bety=34)
tw2_xsng = line.madng_twiss(start='s.ds.l8.b1', end='ip1', beta11=100, beta22=34, xsuite_tw=False)

assert len(tw2_xs.betx) == len(tw2_xsng.beta11_ng)
xo.assert_allclose(tw2_xs['betx'], tw2_xsng.beta11_ng, rtol=1e-8, atol=1e-5)
xo.assert_allclose(tw2_xs['bety'], tw2_xsng.beta22_ng, rtol=1e-8, atol=1e-5)
xo.assert_allclose(tw2_xs['alfx'], tw2_xsng.alfa11_ng, rtol=1e-8, atol=1e-5)
xo.assert_allclose(tw2_xs['alfy'], tw2_xsng.alfa22_ng, rtol=1e-8, atol=1e-5)

tw3_xs = line.twiss(start='ip8', end='ip2', betx=1.5, bety=1.5)
tw3_xsng = line.madng_twiss(start='ip8', end='ip2', beta11=1.5, beta22=1.5, xsuite_tw=False)

assert len(tw3_xs.betx) == len(tw3_xsng.beta11_ng)
xo.assert_allclose(tw3_xs['betx'], tw3_xsng.beta11_ng, rtol=1e-8, atol=1e-5)
xo.assert_allclose(tw3_xs['bety'], tw3_xsng.beta22_ng, rtol=1e-8, atol=1e-5)
xo.assert_allclose(tw3_xs['alfx'], tw3_xsng.alfa11_ng, rtol=1e-8, atol=1e-5)
xo.assert_allclose(tw3_xs['alfy'], tw3_xsng.alfa22_ng, rtol=1e-8, atol=1e-5)
xo.assert_allclose(tw3_xs['x'], tw3_xsng.x_ng, rtol=1e-8, atol=1e-10)
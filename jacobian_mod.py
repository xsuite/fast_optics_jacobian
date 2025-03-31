import xdeps as xd
import numpy as np
import sympy

all_quad_sources = None
target_places = None
dkq_dvv = None

def get_dependency_derivatives():
    return all_quad_sources, target_places, dkq_dvv

# collider, opt
def get_dependency_derivatives(env, opt):
    class FakeQuad:
        pass

    dkq_dvv = {} # Derivatives of quadrupole strengths with respect to the knobs
    a = sympy.var("a")
    for ivv in range(len(opt.vary)):
        vv = opt.vary[ivv].name

        k1 = []
        myelems = {}
        for dd in env.ref_manager.find_deps([env.vars[vv]]):
            if dd.__class__.__name__ == "AttrRef" and dd._key == "k1":
                k1.append((dd._owner._key, dd._expr))
                myelems[dd._owner._key] = FakeQuad()

        fdef = env.ref_manager.mk_fun("myfun", a=env.vars[vv])
        gbl = {
            "vars": env.ref_manager.containers["vars"]._owner.copy(),
            "element_refs": myelems,
        }
        lcl = {}
        exec(fdef, gbl, lcl)
        fff = lcl["myfun"]

        fff(a)
        dk1_dvv = {}
        for kk, expr in k1:
            dd = gbl["element_refs"][kk].k1.diff(a)
            dk1_dvv[kk] = dd

        dkq_dvv[vv] = dk1_dvv

    all_quad_sources = set()
    for vv in dkq_dvv.keys():
        all_quad_sources.update(dkq_dvv[vv].keys())

    target_places = set()
    for tt in opt.targets:
        assert isinstance(tt.tar, tuple)
        target_places.add(tt.tar[1])
    return all_quad_sources, target_places, dkq_dvv
############################################################### here new twiss needed

def get_jac(env, opt, all_quad_sources, target_places, dkq_dvv, tw0):
    twiss_derivs = {}
    for qqnn in all_quad_sources:
        twiss_derivs[qqnn] = {}
        for tt in target_places:
            twiss_derivs[qqnn][tt] = tw0.get_twiss_param_derivative(src=qqnn, observation=tt)

            # Refer to k1 instead of k1l
            for nn in twiss_derivs[qqnn][tt].keys():
                twiss_derivs[qqnn][tt][nn] *= env[qqnn].length

    jac_estim = np.zeros((len(opt.targets), len(opt.vary)))
    for itt, tt in enumerate(opt.targets):

        assert isinstance(tt.tar, tuple)

        tar_quantity = tt.tar[0]
        tar_place = tt.tar[1]
        tar_weight = tt.weight

        for ivv in range(len(opt.vary)):

            vv = opt.vary[ivv].name

            quad_names = dkq_dvv[vv].keys()

            dtar_dvv = 0
            for qqnn in quad_names:
                dtar_dvv += twiss_derivs[qqnn][tar_place]['d'+tar_quantity] * dkq_dvv[vv][qqnn]

            dtar_dvv *= tar_weight

            jac_estim[itt, ivv] = dtar_dvv

def get_jacobian(self, x, f0=None):
    x = np.array(x).copy()
    steps = self._knobs_to_x(self.steps_for_jacobian)
    # get twiss
    tw0 = self.actions[0].line.twiss()
    print(f"TWISS = f{tw0}")

xd.optimize.optimize.MeritFunctionForMatch.get_jacobian = get_jacobian
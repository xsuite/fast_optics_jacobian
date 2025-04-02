import xdeps as xd
import numpy as np
import sympy

all_quad_sources = None
target_places = None
dkq_dvv = None

def get_dependency_derivatives():
    return all_quad_sources, target_places, dkq_dvv

# collider, opt
def get_dependency_derivatives(line, opt):
    class FakeQuad:
        pass

    dkq_dvv = {} # Derivatives of quadrupole strengths with respect to the knobs
    a = sympy.var("a")
    for ivv in range(len(opt.vary)):
        vv = opt.vary[ivv].name

        k1 = []
        myelems = {}
        for dd in line.ref_manager.find_deps([line.vars[vv]]):
            if dd.__class__.__name__ == "AttrRef" and dd._key == "k1":
                k1.append((dd._owner._key, dd._expr))
                myelems[dd._owner._key] = FakeQuad()

        fdef = line.ref_manager.mk_fun("myfun", a=line.vars[vv])
        gbl = {
            "vars": line.ref_manager.containers["vars"]._owner.copy(),
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

    tt0 = line.get_table()
    n_points = 10
    insertions = []
    points = {}
    for qqnn in all_quad_sources:
        assert qqnn in line.element_names
        qq_start = tt0['s_start', qqnn]
        qq_end = tt0['s_end', qqnn]
        s_slice = np.linspace(qq_start, qq_end, n_points)
        points[qqnn] = []
        for ii, ss in enumerate(s_slice):
            nn_point = qqnn + f'_p{ii}'
            insertions.append(line.new(nn_point, 'marker', at=ss)) ############## env needed!!
            points[qqnn].append(nn_point)
    line.insert(insertions)

    target_places = set()
    for tt in opt.targets:
        assert isinstance(tt.tar, tuple)
        target_places.add(tt.tar[1])
    return all_quad_sources, target_places, dkq_dvv, points
############################################################### here new twiss needed

def get_jac(line, opt, all_quad_sources, target_places, dkq_dvv, points):
    tw0 = line.twiss()
    twiss_derivs = {}
    for qqnn in all_quad_sources:
        twiss_derivs[qqnn] = {}
    for tt in target_places:
        twiss_derivs[qqnn][tt] = {}
        for qqnn_p in points[qqnn]:
            twiss_derivs[qqnn][tt][qqnn_p] = tw0.get_twiss_param_derivative(src=qqnn_p, observation=tt)

        # Refer to k1 instead of k1l
            for nn in twiss_derivs[qqnn][tt][qqnn_p].keys():
                twiss_derivs[qqnn][tt][qqnn_p][nn] *= line[qqnn].length

        # Take the mean of all points preserving the keys
        mean_values = {}
        for qqnn_p in points[qqnn]:
            for nn, val in twiss_derivs[qqnn][tt][qqnn_p].items():
                mean_values.setdefault(nn, []).append(val)

        twiss_derivs[qqnn][tt] = {nn: np.mean(val) for nn, val in mean_values.items()}

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

    return jac_estim

def get_jacobian(self, x, f0=None):
    if len(self.targets) == 12 and len(self.vary) == 20:
        x = np.array(x).copy()
        steps = self._knobs_to_x(self.steps_for_jacobian)
        # get twiss
        env_line = self.actions[0].line
        global all_quad_sources, target_places, dkq_dvv
        if all_quad_sources is None or target_places is None or dkq_dvv is None:
            all_quad_sources, target_places, dkq_dvv, points = get_dependency_derivatives(env_line, self)
        jacobian = get_jac(env_line, self, all_quad_sources, target_places, dkq_dvv, points)
        return jacobian
    else:
        if hasattr(self, "_force_jacobian"):
            return self._force_jacobian
        x = np.array(x).copy()
        steps = self._knobs_to_x(self.steps_for_jacobian)
        assert len(x) == len(steps)
        if f0 is None:
            f0 = self(x)
        if np.isscalar(f0):
            jac = np.zeros((1, len(x)))
        else:
            jac = np.zeros((len(f0), len(x)))
        mask_input = self.mask_input
        for ii in range(len(x)):
            if not mask_input[ii]:
                continue
            x[ii] += steps[ii]
            jac[:, ii] = (self(x, check_limits=False) - f0) / steps[ii]
            x[ii] -= steps[ii]

        self._last_jac = jac
        return jac


xd.optimize.optimize.MeritFunctionForMatch.get_jacobian = get_jacobian
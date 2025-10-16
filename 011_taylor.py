import xtrack as xt
import numpy as np
import math

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

class TPSA:
    def __init__(self, tpsa_dict: dict, num_variables: int = None):
        self.tpsa_dict = tpsa_dict
        self.monom_length = len(tpsa_dict)
        self.keys = list(tpsa_dict.keys())
        self.order = max(np.sum(tpsa_dict[k][0], axis=1).max() for k in tpsa_dict)
        if num_variables is None:
            self.num_variables = sum(1 for k in self.keys if len(self.tpsa_dict[k][1]) > 1)
        else:
            self.num_variables = num_variables

    def get_coeff(self, key: str, derived_var_arr: np.ndarray) -> float | list[float]:
        assert key in self.keys, f"Key {key} not in TPSA dict"

        if derived_var_arr.ndim == 1:
            coeff_index = np.where(np.all(derived_var_arr == self.tpsa_dict[key][0], axis=1))[0]
            if coeff_index.size == 0:
                print(f"WARNING: No coefficient found for key {key} with monomial {derived_var_arr} not found.")
                return 0.0
            elif coeff_index.size == 1:
                return self.tpsa_dict[key][1][coeff_index[0]]
        else:
            assert derived_var_arr.ndim == 2, "Only 1D or 2D arrays supported"
            coeff_indices = [np.where(np.all(i == self.tpsa_dict[key][0], axis=1))[0] for i in derived_var_arr]
            coeff_values = [self.tpsa_dict[key][1][i[0]] if i.size > 0 else 0.0 for i in coeff_indices]
            return coeff_values

    def get_taylor_expansion_key(self, key: str, delta: np.ndarray) -> float:
        """
        Evaluate the Taylor expansion for a given key at specified variable values.

        Parameters:
        -----------
        key: str
            The key in the TPSA dict to evaluate.
        var_values: np.ndarray
            A 1D array of variable values at which to evaluate the Taylor expansion.
        init_values: np.ndarray
            A 1D array of initial variable values (the expansion point).

        Returns:
        --------
        float
            The evaluated Taylor expansion value.
        """
        assert key in self.keys, f"Key {key} not in TPSA dict"
        assert delta.ndim == 1, "delta must be a 1D array"
        assert delta.size == self.monom_length, f"delta must have size {self.monom_length}"

        total = 0.0
        tpsas_key = self.tpsa_dict[key]
        for i in range(tpsas_key[0].shape[0]):
            monom = tpsas_key[0][i]
            coeff = tpsas_key[1][i]
            order = int(np.sum(monom))
            term = 1/math.factorial(order) * coeff
            for j in range(self.monom_length):
                if monom[j] != 0:
                    term *= (delta[j]) ** monom[j]
            total += term
        return total

    def get_taylor_expansion_all(self, delta: np.ndarray) -> np.ndarray:
        """
        Evaluate the Taylor expansion for all keys at specified variable values.

        Parameters:
        -----------
        var_values: np.ndarray
            A 1D array of variable values at which to evaluate the Taylor expansion.
        init_values: np.ndarray
            A 1D array of initial variable values (the expansion point).

        Returns:
        --------
        dict
            A dictionary with keys as in the TPSA dict and values as the evaluated Taylor expansion values.
        """
        assert delta.ndim == 1, "delta must be a 1D array"
        assert delta.size == self.monom_length, f"delta must have size {self.monom_length}"

        return np.array([self.get_taylor_expansion_key(k, delta) for k in self.keys[:self.num_variables]])

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
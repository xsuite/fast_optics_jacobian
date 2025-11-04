import numpy as np
import math

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

    def get_coeff(self, key: str, derived_var_arr: np.ndarray, verbose: str = False) -> float | list[float]:
        assert key in self.keys, f"Key {key} not in TPSA dict"

        if derived_var_arr.ndim == 1:
            coeff_index = np.where(np.all(derived_var_arr == self.tpsa_dict[key][0], axis=1))[0]
            if coeff_index.size == 0:
                if verbose:
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
        delta: np.ndarray
            The difference between initial and perturbed coordinates to evaluate the Taylor expansion at.

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
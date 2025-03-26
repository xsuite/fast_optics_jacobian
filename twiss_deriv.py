import numpy as np
import xtrack as xt


def get_twiss_param_derivative(self, src, observation):
        """
        Returns the derivatives of Twiss parameters between two elements
        with respect to the quadrupole strength.

        Parameters
        ----------
        start : int or str
            Index or name of the element at which the computation starts.
        end : int or str
            Index or name of the element at which the computation ends.

        Returns
        -------
        derivatives : dict
            Dictionary containing the derivatives of Twiss parameters.
        """

        R_matrix = self.get_R_matrix(start=src, end=observation)

        betx_start = self['betx', src]
        bety_start = self['bety', src]
        alfx_start = self['alfx', src]
        alfy_start = self['alfy', src]
        dx_start = self['dx', src]
        dy_start = self['dy', src]

        dbetx = -2 * R_matrix[0, 0] * R_matrix[0, 1] * betx_start + 2 * R_matrix[0, 1]**2 * alfx_start
        dbety = 2 * R_matrix[2, 2] * R_matrix[2, 3] * bety_start - 2 * R_matrix[2, 3]**2 * alfy_start
        dalfx = betx_start * (R_matrix[0, 1] * R_matrix[1, 0] + R_matrix[0, 0] * R_matrix[1, 1])\
                    - 2 * alfx_start * R_matrix[0, 1] * R_matrix[1, 1]
        dalfy = - bety_start * (R_matrix[2, 3] * R_matrix[3, 2] + R_matrix[2, 2] * R_matrix[3, 3])\
                    + 2 * alfy_start * R_matrix[2, 3] * R_matrix[3, 3]
        dmux = 1 / (2 * np.pi) * (R_matrix[0, 1]**2 * betx_start / (R_matrix[0, 1]**2 + (R_matrix[0, 0] * betx_start - R_matrix[0, 1] * alfx_start)**2))
        dmuy = 1 / (2 * np.pi) * (-R_matrix[2, 3]**2 * bety_start / (R_matrix[2, 3]**2 + (R_matrix[2, 2] * bety_start - R_matrix[2, 3] * alfy_start)**2))
        ddx = -R_matrix[0, 1] * dx_start
        ddpx = -R_matrix[1, 1] * dx_start
        ddy = R_matrix[2, 3] * dy_start
        ddpy = R_matrix[3, 3] * dy_start

        derivatives = {
            'dbetx': dbetx,
            'dbety': dbety,
            'dalfx': dalfx,
            'dalfy': dalfy,
            'dmux': dmux,
            'dmuy': dmuy,
            'ddx': ddx,
            'ddpx': ddpx,
            'ddy': ddy,
            'ddpy': ddpy
        }

        return derivatives

xt.twiss.TwissTable.get_twiss_param_derivative = get_twiss_param_derivative
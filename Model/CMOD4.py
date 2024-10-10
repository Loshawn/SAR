import numpy as np
from tqdm import tqdm

class CMOD4():

    def __init__(self) -> None:
        self.C = np.array([
            0, -2.301523, -1.632686, 0.761210, 1.156619, 0.595955, -0.293819,
            -1.015244, 0.342175, -0.500786, 0.014430, 0.002484, 0.074450,
            0.004023, 0.148810, 0.089286, -0.006667, 3.000000, -10.000000
        ])
        self.theta2br = [
            1.075, 1.075, 1.075, 1.072, 1.069, 1.066, 1.056, 1.030, 1.004,
            0.979, 0.967, 0.958, 0.949, 0.941, 0.934, 0.927, 0.923, 0.930,
            0.937, 0.944, 0.955, 0.967, 0.978, 0.998, 0.998, 1.009, 1.021,
            1.033, 1.042, 1.050, 1.054, 1.053, 1.052, 1.047, 1.038, 1.028,
            1.056, 1.016, 1.002, 0.989, 0.965, 0.941, 0.929, 0.929, 0.929
        ]
        self.br = None

    def _forward(self, v, phi, theta):
        b0, b1, b2, b3 = self._get_B(v, theta)
        sigma0 = b0 * np.power(
            (1 + b1 * np.cos(phi) + b3 * np.tanh(b2) * np.cos(2 * phi)), 1.6)

        return sigma0

    def _get_B(self, V, theta):
        x = (theta - 40) / 25
        P = [1, x, (3 * np.power(x, 2) - 1) / 2]
        Alpha = self.C[1] * P[0] + self.C[2] * P[1] + self.C[3] * P[2]
        GAM = self.C[4] * P[0] + self.C[5] * P[1] + self.C[6] * P[2]
        Beta = self.C[7] * P[0] + self.C[8] * P[1] + self.C[9] * P[2]

        f2 = np.tanh(2.5 * (x + 0.35)) - 0.61 * (x + 0.35)

        b1 = self.C[10] * P[0] + self.C[11] * V + (self.C[12] * P[0] +
                                                   self.C[13] * V) * f2
        b2 = self.C[14] * P[0] + self.C[15] * (1 + P[1]) * V
        b3 = 0.42 * (1 + self.C[16] * (self.C[17] + x) * (self.C[18] + V))

        y = V + Beta
        f1 = np.ones_like(y) * -10
        f1[(y>1e-10) & (y <= 5)] = np.log10(y[(y>1e-10) & (y <= 5)])
        f1[y > 5] = np.sqrt(y[y > 5]) / 3.2
        b0 = self.br * 10**(Alpha + GAM * f1)
        return b0, b1, b2, b3

    def _get_br(self, theta):
        br = np.empty_like(theta)
        theta_int = np.floor(theta)
        for i in range(16, 61):
            br[theta_int == i] = self.theta2br[i - 16]
        return br

    def inverse(self, sigma0_obs, phi, incidence, iterations=10):
        print("##----  Model CMOD4 ----##\n")
        sigma0_mask = np.ma.masked_equal(sigma0_obs, 0)
        incidence_mask = np.ma.masked_array(incidence, mask=sigma0_mask.mask)
        self.br = self._get_br(incidence_mask)
        
        phi_mask = np.ma.masked_array(phi, mask=sigma0_mask.mask)
        V = np.array([20]) * np.ones(sigma0_obs.shape)

        step = 10
        print("Begin the inversion ...\n")
        for iterno in tqdm(range(iterations)):
            sigma0_calc = self._forward(V, phi_mask, incidence_mask)
            ind = sigma0_calc - sigma0_mask > 0
            V = V + step
            V[ind] = V[ind] - 2 * step
            step /= 2
        V = np.ma.masked_array(V, mask=sigma0_mask.mask)
        print("\nDone .\n")
        return V

__all__ = ['CMOD4']

if __name__ == "__main__":
    model = CMOD4()
    sigma0 = np.linspace(0,0.5,100)
    phi = np.zeros_like(sigma0)
    inc = np.linspace(16,60,100)
    print(inc)
    model.inverse(sigma0, phi, inc)



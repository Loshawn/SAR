import numpy as np
from tqdm import tqdm

class CMOD():

    def __init__(self, C):
        self.C = C

    def _forward(self, v, phi, theta):
        b0, b1, b2 = self._get_B(v, theta)
        sigma0 = b0 * np.power(
            (1 + b1 * np.cos(phi) + b2 * np.cos(2 * phi)), 1.6)

        return sigma0

    def _get_B(self, v, theta):
        x = (theta - 40) / 25
        A0 = self.C[1] + self.C[2] * x + self.C[3] * np.power(
            x, 2) + self.C[4] * np.power(x, 3)
        A1 = self.C[5] + self.C[6] * x
        A2 = self.C[7] + self.C[8] * x

        GAM = self.C[9] + self.C[10] * x + self.C[11] * np.power(x, 2)
        S0 = self.C[12] + self.C[13] * x

        s = A2 * v

        alpha = S0 * (1 - self._g(S0))
        f = self._g(s)
        slS0 = (s < S0)
        f[slS0] = ((s / S0)**alpha * self._g(S0))[slS0]

        b0 = 10**(A0 + A1 * v) * f**GAM
        b1_b = self.C[14] * (1 + x) - self.C[15] * v * (
            0.5 + x - np.tanh(4 * (x + self.C[16] + self.C[17] * v)))
        b1_a = 1 + np.exp(0.34 * (v - self.C[18]))
        b1 = b1_b / b1_a
        # B2
        y0 = self.C[19]
        n = self.C[20]
        a = y0 - (y0 - 1) / n
        b = 1 / (n * (y0 - 1)**(n - 1))
        v0 = self.C[21] + self.C[22] * x + self.C[23] * np.power(x, 2)
        d1 = self.C[24] + self.C[25] * x + self.C[26] * np.power(x, 2)
        d2 = self.C[27] + self.C[28] * x

        v2 = y = (v + v0) / v0
        v2[y < y0] = a + b * (y[y < y0] - 1)**n

        b2 = (-d1 + d2 * v2) * np.exp(-v2)

        return b0, b1, b2

    def _g(self, s):
        return 1 / (1 + np.exp(-s))

    def inverse(self, sigma0_obs, phi, incidence, iterations=10):
        sigma0_mask = np.ma.masked_equal(sigma0_obs, 0)
        incidence_mask = np.ma.masked_equal(incidence, 0)
        phi_mask = np.ma.masked_array(phi, mask=sigma0_mask.mask)

        V = np.array([10]) * np.ones(sigma0_obs.shape)

        step = 10

        print("Begin the inversion ...\n")
        for iterno in tqdm(range(iterations)):
            sigma0_calc = self._forward(V, phi_mask, incidence_mask)
            ind = sigma0_calc - sigma0_mask > 0
            V = V + step
            V[ind] = V[ind] - 2 * step
            step /= 2
        V = np.ma.masked_array(V, mask=sigma0_mask.mask)
        print("\nDone.\n")
        return V


class CMOD5_N(CMOD):

    def __init__(self):
        C = [
            0, -0.6878, -0.7957, 0.3380, -0.1728, 0.0000, 0.0040, 0.1103,
            0.0159, 6.7329, 2.7713, -2.2885, 0.4971, -0.7250, 0.0450, 0.0066,
            0.3222, 0.0120, 22.7000, 2.0813, 3.0000, 8.3659, -3.3428, 1.3236,
            6.2437, 2.3893, 0.3249, 4.1590, 1.6930
        ]
        super().__init__(C)

    def inverse(self, sigma0_obs, phi, incidence, iterations=10):
        print("##----  Model CMOD5 ----##\n")
        return super().inverse(sigma0_obs, phi, incidence, iterations)


class CMOD5(CMOD):

    def __init__(self):
        C = [
            0, -0.688, -0.793, 0.3380, -0.173, 0.0000, 0.0040, 0.111, 0.0162,
            6.340, 2.57, -2.180, 0.40, -0.6, 0.0450, 0.007, 0.330, 0.0120,
            22.0, 1.95, 3.0000, 8.39, -3.44, 1.36, 5.35, 1.99, 0.29, 3.80, 1.53
        ]
        super().__init__(C)

    def inverse(self, sigma0_obs, phi, incidence, iterations=10):
        print("##----  Model CMOD5.N ----##\n")
        return super().inverse(sigma0_obs, phi, incidence, iterations)
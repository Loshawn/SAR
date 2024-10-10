import numpy as np
from tqdm import tqdm

class CMOD_IFR2():

    def __init__(self) -> None:
        self.C = np.array([
            0, -2.437597, -1.5670307, 0.3708242, -0.040590, 0.40464678,
            0.188397, -0.027262, 0.064650, 0.054500, 0.086350, 0.055100,
            -0.058450, -0.096100, 0.412754, 0.121785, -0.024333, 0.072163,
            -0.062954, 0.015958, -0.069514, -0.062945, 0.035538, 0.023049,
            0.074654, 0.014713
        ])

    def _forward(self, v, phi, theta):
        b0, b1, b2 = self._get_B(v, theta)
        sigma0 = 10**b0 * (1 + b1 * np.cos(phi) +
                           np.tanh(b2) * np.cos(2 * phi))

        return sigma0

    def _get_B(self, v, theta):
        X = (theta - 36) / 19
        P1 = X
        P2 = (3 * X**2 - 1) / 2
        P3 = X * (5 * X**2 - 3) / 2
        V1 = (2 * v - 28) / 22
        V2 = 2 * V1**2 - 1
        V3 = (2 * V1**2 - 1) * V1
        Y = (2 * theta - 76) / 40
        Q1 = Y
        Q2 = 2 * Y**2 - 1

        Alpha = self.C[1] + self.C[2] * P1 + self.C[3] * P2 + self.C[4] * P3
        Beta = self.C[5] + self.C[6] * P1 + self.C[7] * P2

        B0 = Alpha + Beta * np.sqrt(v)
        B1 = self.C[8] + self.C[9] * V1 + (
            self.C[10] + self.C[11] * V1) * Q1 + (self.C[12] +
                                                  self.C[13] * V1) * Q2
        B2 = self.C[14] + self.C[15] * Q1 + self.C[16] * Q2 + (
            self.C[17] + self.C[18] * Q1 + self.C[19] * Q2) * V1 + (
                self.C[20] + self.C[21] * Q1 + self.C[22] * Q2) * V2 + (
                    self.C[23] + self.C[24] * Q1 + self.C[25] * Q2) * V3
        return B0, B1, B2

    def inverse(self, sigma0_obs, phi, incidence, iterations=10):
        print("##----  Model CMOD IFR2 ----##\n")
        sigma0_mask = np.ma.masked_equal(sigma0_obs, 0)
        incidence_mask = np.ma.masked_invalid(np.ma.masked_equal(incidence, 0))
        phi_mask = np.ma.masked_array(phi, mask=sigma0_mask.mask)
        V = np.array([10]) * np.ones(sigma0_obs.shape)

        step = 10
        print("Begin the inversion ...\n")
        for iterno in tqdm(range(iterations), ncols=100):
            sigma0_calc = self._forward(V, phi_mask, incidence_mask)
            ind = sigma0_calc - sigma0_mask > 0
            V = V + step
            V[ind] = V[ind] - 2 * step
            step /= 2
        V = np.ma.masked_array(V, mask=sigma0_mask.mask)
        print("\nDone.\n")
        return V

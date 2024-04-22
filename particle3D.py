import numpy as np
import constants as ct


class Particle:
    def __init__(
        self,
        omegas,
        gamma,
        B,
        coupling,
        eta_detection=1,
        radius=147e-9,
        rho=2200,
        T=293,
    ):
        self.__omega_x__ = omegas[0]
        self.__omega_y__ = omegas[1]
        self.__omega_z__ = omegas[2]
        self.__gamma__ = gamma
        self.T = T
        self.A = np.array([[0, 0, 0, self.__omega_x__, 0, 0],
                           [0, 0, 0, 0, self.__omega_y__, 0],
                           [0, 0, 0, 0, 0, self.__omega_z__],
                           [-self.__omega_x__, 0, 0, -self.__gamma__, 0, 0],
                           [0, -self.__omega_y__, 0, 0, -self.__gamma__, 0],
                           [0, 0, -self.__omega_z__, 0, 0, -self.__gamma__]])
        self.B = B
        self.C = np.array([[1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0]])
        self.G = np.array([[0], [0], [0], [1], [1], [1]])
        self.eta_det = eta_detection
        self.backaction = np.sqrt(4 * np.pi * coupling)
        self._m_ = rho * 4 * np.pi * np.power(radius, 3) / 3
        self.zp_x = np.sqrt(ct.hbar / (2 * omegas[0] * self._m_))
        self.zp_px = np.sqrt(omegas[0] * ct.hbar * self._m_ / 2)
        self.zp_y = np.sqrt(ct.hbar / (2 * omegas[1] * self._m_))
        self.zp_py = np.sqrt(omegas[1] * ct.hbar * self._m_ / 2)
        self.zp_z = np.sqrt(ct.hbar / (2 * omegas[2] * self._m_))
        self.zp_pz = np.sqrt(omegas[2] * ct.hbar * self._m_ / 2)
        self.thermal_force_std_x = (
            np.sqrt(4 * self.__gamma__ * self._m_ * ct.kb * T) / self.zp_px
        )
        self.thermal_force_std_y = (
            np.sqrt(4 * self.__gamma__ * self._m_ * ct.kb * T) / self.zp_py
        )
        self.thermal_force_std_z = (
            np.sqrt(4 * self.__gamma__ * self._m_ * ct.kb * T) / self.zp_pz
        )
        self.backaction_std_x = self.backaction / self.zp_px
        self.backaction_std_y = self.backaction / self.zp_py
        self.backaction_std_z = self.backaction / self.zp_pz

    def step(self, states, control=0.0, delta_t=50e-2, bypass_noise=False):
        thermal_force_x = self.thermal_force_std_x * np.random.normal()
        thermal_force_y = self.thermal_force_std_y * np.random.normal()
        thermal_force_z = self.thermal_force_std_z * np.random.normal()
        backaction_force_x = self.backaction_std_x * (
            np.sqrt(self.eta_det) * np.random.normal() +
            np.sqrt(1 - self.eta_det) * np.random.normal()
        )
        backaction_force_y = self.backaction_std_y * (
            np.sqrt(self.eta_det) * np.random.normal() +
            np.sqrt(1 - self.eta_det) * np.random.normal()
        )
        backaction_force_z = self.backaction_std_z * (
            np.sqrt(self.eta_det) * np.random.normal() +
            np.sqrt(1 - self.eta_det) * np.random.normal()
        )

        if bypass_noise:
            stoch_force = np.zeros((6, 1))
        else:
            stoch_force = np.array([[0],
                                    [0],
                                    [0],
                                    [thermal_force_x+backaction_force_x],
                                    [thermal_force_y+backaction_force_y],
                                    [thermal_force_z+backaction_force_z]])

        state_dot = np.matmul(self.A, states) + self.B @ control
        states = states + state_dot * delta_t + self.G * np.sqrt(delta_t) * stoch_force
        return states

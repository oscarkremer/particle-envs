import numpy as np
import constants as ct


class Particle:
    def __init__(
        self,
        omega,
        gamma,
        coupling,
        eta_detection=1,
        radius=147e-9,
        rho=2200,
        T=293,
    ):
        """
        Create instance of a particle, setting the A and B matrix for
        the dynamics.
        :param omega: Resonance angular frequency of the particle's motion
        :param gamma: Damping coefficient of the medium - influences
        the drag force and stochastic force related to the residual gas.
        :param coupling: Term that defines the intensity of the backaction
        stochastic force, such that the uncertainty principle is satisfied.
        :param eta_detection: Detection efficiency defines how far from the
        quantum limit the simulated detection is.
        :param radius: Particle's radius used to compute its mass.
        :param rho: Particle's density, pre-set to 2200 kg/m^3, that is
        the Silica density.
        :param T: Room temperature used to define the gas temperature inside
        the vacuum chamber, which affects the variance of the stochastic force.
        """
        self.__omega__ = omega
        self.__gamma__ = gamma
        self.T = T
        self.A = np.array([[0, self.__omega__], [-self.__omega__, -self.__gamma__]])
        self.B = np.array([[0], [1]]).astype(float)
        self.G = np.array([[0], [1]]).astype(float)
        self.backaction = np.sqrt(4 * np.pi * coupling)
        self.eta_det = eta_detection
        self._m_ = rho * 4 * np.pi * np.power(radius, 3) / 3
        self.zp_x = np.sqrt(ct.hbar / (2 * omega * self._m_))
        self.zp_p = np.sqrt(omega * ct.hbar * self._m_ / 2)
        self.nl = ct.hbar * self.__omega__ / (ct.kb * self.T)
        self.C = np.array([[1, 0]]).astype(float)
        self.thermal_force_std = (
            np.sqrt(4 * self.__gamma__ * self._m_ * ct.kb * T) / self.zp_p
        )
        self.backaction_std = self.backaction / self.zp_p

    def __backaction_fluctuation__(self):
        """
        Create the backaction force term such that it will be a combination
        of two gaussian stochastic processes.
        :return: Value of the stochastic quantum force for a
        specific time instant.
        """
        return self.backaction_std * (
            np.sqrt(self.eta_det) * np.random.normal()
            + np.sqrt(1 - self.eta_det) * np.random.normal()
        )

    def step(self, states, control=0.0, delta_t=50e-2):
        """
        Run one step of the dynamics
        :param states: current state vector
        :control: control signal (for this case can be a scalar or 1x1 array
        :delta_t: timestep used for the simulation
        ...
        :return: new state vector
        """
        if states.size > 2:
            raise ValueError(
                "States size for this specific system is equal to two \
                (position and velocity)"
            )
        backaction_force = self.__backaction_fluctuation__()
        thermal_force = self.thermal_force_std * np.random.normal()
        state_dot = np.matmul(self.A, states) + self.B * control
        states = (
            states
            + state_dot * delta_t
            + self.G * np.sqrt(delta_t) * (thermal_force - backaction_force)
        )
        return states

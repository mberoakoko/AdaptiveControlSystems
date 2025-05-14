import dataclasses

from typing import NamedTuple
import abc

import numpy as np


class ForcingSignal(NamedTuple):
    t: np.ndarray
    u: np.ndarray


@dataclasses.dataclass(frozen=True)
class AbsstractSignal(abc.ABC):
    dt: float
    t_final: float

    def _sim_time(self) -> np.ndarray:
        return np.linspace(0, self.t_final, round(self.t_final / self.dt))

    @abc.abstractmethod
    def generate_signal(self) -> ForcingSignal:
        raise NotImplementedError("this function is not implemented")


@dataclasses.dataclass(frozen=True)
class SisoStepSignal(AbsstractSignal):
    delay: float

    def generate_signal(self) -> ForcingSignal:
        sim_time: np.ndarray = self._sim_time()
        u: np.ndarray = np.ones_like(sim_time)
        u[u < self.delay] = 0
        return ForcingSignal(
            t=sim_time,
            u=u
        )


@dataclasses.dataclass(frozen=True)
class SisoRampSignal(SisoStepSignal):
    slope: float

    def generate_signal(self) -> ForcingSignal:
        step_forcing_signal = super().generate_signal()
        ramp_forcing = self.slope * step_forcing_signal.u
        return ForcingSignal(
            step_forcing_signal.t,
            ramp_forcing
        )




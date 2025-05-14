import control
import dataclasses
import abc

import numpy as np


@dataclasses.dataclass
class AbstractModel(abc.ABC):
    @abc.abstractmethod
    def update(self, t: float, x: np.ndarray, u: np.ndarray, params: dict) -> np.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def output(self, t: float, x: np.ndarray, u: np.ndarray, params: dict) -> np.ndarray:
        raise NotImplementedError


@dataclasses.dataclass
class SimpleModel(AbstractModel):
    w: float

    def update(self, t: float, x: np.ndarray, u: np.ndarray, params: dict) -> np.ndarray:
        return self.w @ x + u

    def output(self, t: float, x: np.ndarray, u: np.ndarray, params: dict) -> np.ndarray:
        return x

    def generate(self) -> control.NonlinearIOSystem:
        return control.NonlinearIOSystem(
            self.update, self.output, name="SimpleSystem",
            inputs=("u",),
            outputs=("x",),
            states=("x",)
        )

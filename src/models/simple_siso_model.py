import control
import dataclasses
import abc

import numpy as np


class AbstractModel(abc.ABC):
    @abc.abstractmethod
    def __update(self, t: float, x: np.ndarray, u: np.ndarray, params: dict) -> np.ndarray:
        ...

    @abc.abstractmethod
    def __output(self, t: float, x: np.ndarray, u: np.ndarray, params: dict) -> np.ndarray:
        ...


@dataclasses.dataclass
class SimpleModel(AbstractModel):
    w: float

    def __update(self, t: float, x: np.ndarray, u: np.ndarray, params: dict) -> np.ndarray:
        return self.w @ x + u

    def __output(self, t: float, x: np.ndarray, u: np.ndarray, params: dict) -> np.ndarray:
        return x

    def generate(self) -> control.NonlinearIOSystem:
        return control.NonlinearIOSystem(
            self.__update, self.__output, name="SimpleSystem",
            inputs=("u",),
            outputs=("x",),
            states=("x",)
        )

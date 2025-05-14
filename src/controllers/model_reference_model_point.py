import control
import numpy as np
import dataclasses

from src.models.simple_siso_model import AbstractModel


@dataclasses.dataclass
class SisoReferenceAdaptiveControl(AbstractModel):
    gamma: float
    alpha: float

    def update(self, t: float, x: np.ndarray, u: np.ndarray, params: dict) -> np.ndarray:
        x_, c = u
        return self.gamma * x_ * (x_ - c)

    def output(self, t: float, x: np.ndarray, u: np.ndarray, params: dict) -> np.ndarray:
        w_hat = x
        x_, c = u
        return -self.alpha * (x_ - c) - w_hat * x

    def generate_controller(self) -> control.NonlinearIOSystem:
        return  control.NonlinearIOSystem(
            self.update, self.output, name=self.__class__.__name__,
            inputs=("x", "c"), outputs=("u", ),
            states=("w_hat", )
        )


class SisoReferenceModelAdaptiveControl(AbstractModel):
    gamma: float
    alpha: float

    def update(self, t: float, x: np.ndarray, u: np.ndarray, params: dict) -> np.ndarray:
        w_hat, x_r = x
        x_, c = u
        return np.array([
            self.gamma * x * (x - x_r),
            -self.alpha * (x_r - c)
        ])

    def output(self, t: float, x: np.ndarray, u: np.ndarray, params: dict) -> np.ndarray:
        w_hat, x_r = x
        x_, c = u
        return -self.alpha * (x_ - c) - w_hat * x

    def generate_controllers(self) -> control.NonlinearIOSystem:
        return control.NonlinearIOSystem(
            self.update, self.output, name=self.__class__.__name__,
            inputs=("x", "c"), outputs=("u", ),
            states=("w_hat", "x_r")
        )


if __name__ == "__main__":
    sin_ref_controller = SisoReferenceAdaptiveControl(
        gamma=0.1,
        alpha=0.1
    ).generate_controller()
    print(sin_ref_controller)
    print(sin_ref_controller.dynamics(1, np.array([1]), u=np.array([0, 1])))

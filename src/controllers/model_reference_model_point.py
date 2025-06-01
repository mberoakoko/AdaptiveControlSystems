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


@dataclasses.dataclass
class SpringMassDamperWithParametricUncertainty(AbstractModel):
    mass: float
    alpha: float
    beta: float

    def update(self, t: float, x: np.ndarray, u: np.ndarray, params: dict) -> np.ndarray:
        print(f"x_state => {x}")
        x_1, x_2 = x
        b, m, a = self.beta, self.mass, self.alpha
        u = u[0] # to avaoid broadcasting
        return np.array([
            x_2,
            (b/m) * x_2 ** 3 + (a /m ) * x_1 - u
        ])

    def output(self, t: float, x: np.ndarray, u: np.ndarray, params: dict) -> np.ndarray:
        return x

    def generate_system(self) -> control.NonlinearIOSystem:
        return control.NonlinearIOSystem(
            self.update, self.output, name= self.__class__.__name__,
            inputs=("u", ), outputs=("x", "x_dot"),
            states=("x", "x_dot")
        )

@dataclasses.dataclass
class LinearisedLQRController(AbstractModel):
    spring_mass_damper_sys: control.NonlinearIOSystem
    Q: np.ndarray
    R: np.ndarray
    k_1: np.ndarray = dataclasses.field(init=False)
    k_2: np.ndarray = dataclasses.field(init=False)

    def __post_init__(self):
        self.__generate_gains()

    def __control_law(self,x: np.ndarray, c: np.ndarray):
        print(f"x state {x=} | {x.shape=}")
        # return -self.k_1 @ x # + self.k_2 @ c
        return x
    def __generate_gains(self) -> None:
        linearized_system = control.linearize(
            sys=self.spring_mass_damper_sys,
            xeq=[0, 0],
            ueq=[0]
        )
        a, b, c = linearized_system.A, linearized_system.B, linearized_system.C
        self.k_1, _, _ = control.lqr(linearized_system, self.Q, self.R)
        expr_1: np.ndarray = np.linalg.inv((a - b @ self.k_1))
        expr_2: np.ndarray = c @ expr_1
        expr_3: np.ndarray = expr_2 @ b
        print(f"{c=}\n")
        print(f"{b.shape=}\n")
        print(f"{expr_1=}")
        print(f"{expr_2=}")
        print(f"{expr_3=}")
        print(f"{expr_3.flatten()=}\n")
        print(f"{self.k_1.shape=}")
        # self.k_2 = np.linalg.inv(expr_3)


    def update(self, t: float, x: np.ndarray, u: np.ndarray, params: dict) -> np.ndarray:
        pass

    def output(self, t: float, x: np.ndarray, u: np.ndarray, params: dict) -> np.ndarray:
        return self.__control_law(x, u)

    def generate_system(self) -> control.NonlinearIOSystem:
        return control.NonlinearIOSystem(
            None, outfcn=self.output,
            inputs=("x", "x_d0t", "c", ), outputs=("u", ),
        )

def create_closed_loop_spring_mass_damper_system(mass: float, alpha: float, beta: float,
                                                 q: np.ndarray, r: np.ndarray) -> control.NonlinearIOSystem:
    plant: control.NonlinearIOSystem = SpringMassDamperWithParametricUncertainty(
        mass=mass,
        alpha=alpha,
        beta=beta
    ).generate_system()

    controller = control.NonlinearIOSystem = LinearisedLQRController(
        spring_mass_damper_sys=plant,
        Q=q,
        R=r
    ).generate_system()

    return  control.interconnect([plant, controller], inplist=["c",], outlist=["x", "x_dot"])


if __name__ == "__main__":
    sin_ref_controller = SisoReferenceAdaptiveControl(
        gamma=0.1,
        alpha=0.1
    ).generate_controller()
    print(sin_ref_controller)
    print(sin_ref_controller.dynamics(1, np.array([1]), u=np.array([0, 1])))

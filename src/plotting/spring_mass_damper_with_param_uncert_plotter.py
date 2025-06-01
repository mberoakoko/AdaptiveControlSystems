from src.controllers.model_reference_model_point import create_closed_loop_spring_mass_damper_system
from src.simulators.model_reference_model_point_sim import spring_mass_system_with_parametized_uncertinty_simuulation
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

matplotlib.use("TkAgg")

plt.rcParams.update(
    {
        "font.size": 7,
        "figure.dpi": 200
    }
)
plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle')

def plot_system_with_lqr_control(mass: float, alpha: float, beta: float, q: np.ndarray, r: np.ndarray) -> None:
    closed_loop_sys = create_closed_loop_spring_mass_damper_system(mass, alpha, beta, q, r)
    print(closed_loop_sys)
    # sim_result = spring_mass_system_with_parametized_uncertinty_simuulation(
    #     closed_loop_sys, dt=0.01, t_final=10
    # )



if __name__ == "__main__":
    plot_system_with_lqr_control(10, 1, 1, np.eye(2), np.eye(1))
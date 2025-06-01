import control
import numpy as np
from src.simulators.signal_generators import SisoStepSignal, ForcingSignal

def spring_mass_system_with_parametized_uncertinty_simuulation(system: control.NonlinearIOSystem,
                                                               dt: float , t_final: float) -> control.TimeResponseData:
    signal: ForcingSignal = SisoStepSignal(dt=dt, t_final=t_final, delay=1).generate_signal()
    print(signal.u.shape)
    sim_result = control.input_output_response(
        system,
        X0=np.array([0, 0]),
        T=signal.t,
        U=signal.u,
    )
    return sim_result
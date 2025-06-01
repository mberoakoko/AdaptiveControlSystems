import unittest

import control
import numpy as np

from src.controllers.model_reference_model_point import (
    SisoReferenceAdaptiveControl,
    SisoReferenceModelAdaptiveControl,
    SpringMassDamperWithParametricUncertainty,
    LinearisedLQRController,
)


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)  # add assertion here


test_reference_point_controller: control.NonlinearIOSystem = SisoReferenceAdaptiveControl(gamma=0.1,
                                                                                          alpha=0.1).generate_controller()

test_reference_model_controller: control.NonlinearIOSystem = SisoReferenceAdaptiveControl(gamma=0.1,
                                                                                          alpha=0.1).generate_controller()

test_spring_mass_damper_with_parametric_uncertainty: control.NonlinearIOSystem = (
    SpringMassDamperWithParametricUncertainty(
        mass=1,
        alpha=1,
        beta=1
    ).generate_system()
)

test_linearized_spring_mass_damper_with_param_uncertainty_controller: control.NonlinearIOSystem = (
    LinearisedLQRController(
        spring_mass_damper_sys=test_spring_mass_damper_with_parametric_uncertainty,
        Q=np.eye(2),
        R=np.eye(1)
    ).generate_system()
)


class SisoReferenceAdaptiveControlTest(unittest.TestCase):

    def test_dynamics(self):
        self.assertIsNotNone(
            obj=test_reference_point_controller.dynamics(
                t=1,
                x=np.array([1]),
                u=np.array([0, 1])
            ),
            msg="Dynamics Output a value that isnt none"
        )

    def test_output(self):
        self.assertIsNotNone(
            obj=test_reference_point_controller.output(
                t=1,
                x=np.array([1]),
                u=np.array([0, 1])
            ),
            msg="Dynamics Output a value that isnt none"
        )


class SisoReferenceModelAdaptiveControl(unittest.TestCase):

    def test_dynamics(self):
        self.assertIsNotNone(
            obj=test_reference_model_controller.dynamics(
                t=1,
                x=np.array([1, 1]),
                u=np.array([0, 1])
            ),
            msg="Dynamics Output a value that isnt none"
        )

    def test_output(self):
        self.assertIsNotNone(
            obj=test_reference_model_controller.dynamics(
                t=1,
                x=np.array([1, 1]),
                u=np.array([0, 1])
            ),
            msg="Dynamics Output a value that isnt none"
        )

class SpringMassDamperWithParameterUncertaintyTest(unittest.TestCase):

    def test_dynamics(self):
        self.assertIsNotNone(
            obj=test_spring_mass_damper_with_parametric_uncertainty.dynamics(
                t=1,
                x=np.array([1, 1]),
                u=np.array([1])
            )
        )

    def test_output(self):
        self.assertIsNotNone(
            obj=test_spring_mass_damper_with_parametric_uncertainty.output(
                t=1,
                x=np.array([1, 1]),
                u=np.array([1])
            )
        )

class LinearizedLQRControllerTest(unittest.TestCase):

    def test_output(self):
        self.assertIsNotNone(
            obj=test_linearized_spring_mass_damper_with_param_uncertainty_controller.output(
                t=1,
                x=np.array([0, 0]),
                u=np.array([0])
            )
        )

if __name__ == '__main__':
    unittest.main()

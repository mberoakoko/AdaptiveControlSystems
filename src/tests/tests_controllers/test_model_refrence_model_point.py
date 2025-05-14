import unittest

import control
import numpy as np

from src.controllers.model_reference_model_point import SisoReferenceAdaptiveControl, SisoReferenceModelAdaptiveControl


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)  # add assertion here


test_reference_point_controller: control.NonlinearIOSystem = SisoReferenceAdaptiveControl(gamma=0.1,
                                                                                          alpha=0.1).generate_controller()

test_reference_model_controller: control.NonlinearIOSystem = SisoReferenceAdaptiveControl(gamma=0.1,
                                                                                          alpha=0.1).generate_controller()
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


if __name__ == '__main__':
    unittest.main()

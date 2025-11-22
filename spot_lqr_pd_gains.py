# external/spot/spot_lqr_pd_gains.py
#
# Use continuous-time LQR on a simple double-integrator model
#   x = [position_error; velocity_error]
#   xdot = A x + B u
#       with A = [[0, 1],
#                 [0, 0]],
#            B = [[0],
#                 [1]]
#
# to design scalar PD gains (Kp, Kd) such that:
#   u = -Kp * position_error - Kd * velocity_error
#
# You can then plug these gains into Drake's PdControllerGains for
# each joint actuator (e.g., in SpotModel).
#
# Usage (standalone):
#   python3 spot_lqr_pd_gains.py
#
# It will print Kp and Kd. You can copy them into SpotModel(joint_kp=..., joint_kd=...).
#
# Usage (from other code):
#   from spot_lqr_pd_gains import design_pd_gains_from_lqr, apply_lqr_pd_gains_to_plant
#   kp, kd = design_pd_gains_from_lqr()
#   apply_lqr_pd_gains_to_plant(plant, kp, kd)

from __future__ import annotations

from typing import Tuple

import numpy as np
from pydrake.all import (
    LinearQuadraticRegulator,
    PdControllerGains,
    JointActuatorIndex,
    MultibodyPlant,
)


def design_pd_gains_from_lqr(
    q_weight: float = 100.0,
    v_weight: float = 10.0,
    torque_weight: float = 1.0,
) -> Tuple[float, float]:
    """
    Design scalar PD gains (Kp, Kd) via continuous-time LQR on a
    unit-mass double integrator:

        x = [position_error; velocity_error]
        xdot = A x + B u

    with:

        A = [[0, 1],
             [0, 0]],
        B = [[0],
             [1]]

    and quadratic cost:

        J = ∫ (xᵀ Q x + uᵀ R u) dt

    where:

        Q = diag(q_weight, v_weight)
        R = [torque_weight]

    Returns:
        Kp, Kd such that the LQR feedback law
            u = -K x  (with x = [e_q; e_v])
        is equivalent to
            u = -Kp * e_q - Kd * e_v
    """
    # Double integrator dynamics
    A = np.array([[0.0, 1.0],
                  [0.0, 0.0]])
    B = np.array([[0.0],
                  [1.0]])

    Q = np.diag([q_weight, v_weight])
    R = np.array([[torque_weight]])

    # Drake's continuous-time LQR returns K such that u = -K x
    K, _ = LinearQuadraticRegulator(A, B, Q, R)

    # K is shape (1, 2). Extract scalar gains.
    Kp = float(K[0, 0])
    Kd = float(K[0, 1])

    return Kp, Kd


def apply_lqr_pd_gains_to_plant(
    plant: MultibodyPlant,
    Kp: float,
    Kd: float,
) -> None:
    """
    Apply the same PD gains (Kp, Kd) to every joint actuator in `plant`
    using Drake's PdControllerGains.

    This assumes that the plant's actuators are configured to use simple
    joint-level PD controllers (as is the case for Spot in the
    underactuated repo when PdControllerGains are set).
    """
    gains = PdControllerGains(p=Kp, d=Kd)
    num_actuators = plant.num_actuators()

    for i in range(num_actuators):
        actuator = plant.get_joint_actuator(JointActuatorIndex(i))
        actuator.set_controller_gains(gains)

    print(
        f"[LQR-PD] Applied PD gains to {num_actuators} actuators: "
        f"Kp = {Kp:.3f}, Kd = {Kd:.3f}"
    )


if __name__ == "__main__":
    # Standalone usage: design PD gains and print them.
    Kp, Kd = design_pd_gains_from_lqr(
        q_weight=100.0,
        v_weight=10.0,
        torque_weight=1.0,
    )

    print("[LQR-PD] Designed PD gains from double-integrator LQR:")
    print(f"  Kp = {Kp:.6f}")
    print(f"  Kd = {Kd:.6f}")
    print(
        "\nYou can now plug these into SpotModel, e.g.:\n"
        "    model = SpotModel(time_step=1e-3,\n"
        "                     enable_joint_pd=True,\n"
        "                     joint_kp=Kp,\n"
        "                     joint_kd=Kd)\n"
    )

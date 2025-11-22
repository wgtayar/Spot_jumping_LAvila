# Spot standing simulation with joint-level PD on Spot’s actuated joints.
#########################################################################
# Usage:
#   - To import disturbance helper from another script:
#       from spot_standing_sim import apply_first_joint_position_disturbance
#   - Run with no disturbance (just stand there):
#       python3 spot_standing_sim.py
#   - Run with a disturbance at 15 s, magnitude 0.6 rad:
#       python3 spot_standing_sim.py --disturbance_time 15.0 --disturbance_magnitude 0.6
#
# If the robot twitches and recovers, the PD is working. If it doesn’t move
# at all, either the gains are zero or the robot has achieved true inner peace.


import argparse
import numpy as np
from pydrake.all import (
    JointActuatorIndex,
    Simulator,
    StartMeshcat,
)

from spot_model import SpotModel


def build_joint_desired_state(plant, plant_context, model_instance, desired_state_port):
    port_dim = desired_state_port.size()

    q_full = plant.GetPositions(plant_context)
    v_full = plant.GetVelocities(plant_context)

    q_des_list = []
    v_des_list = []

    for i in range(plant.num_actuators()):
        actuator = plant.get_joint_actuator(JointActuatorIndex(i))
        if actuator.model_instance() != model_instance:
            continue

        joint = actuator.joint()
        nq_j = joint.num_positions()
        nv_j = joint.num_velocities()

        q_start = joint.position_start()
        v_start = joint.velocity_start()

        q_des_list.extend(q_full[q_start : q_start + nq_j])
        v_des_list.extend(v_full[v_start : v_start + nv_j])

    q_des = np.array(q_des_list)
    v_des = np.array(v_des_list)

    x_des = np.concatenate([q_des, v_des])

    assert x_des.shape[0] == port_dim, (
        f"Desired state length {x_des.shape[0]} does not match port size {port_dim}"
    )

    return x_des


def apply_first_joint_position_disturbance(
    plant,
    plant_context,
    model_instance,
    magnitude: float = 0.2,
) -> str | None:

    q_full = plant.GetPositions(plant_context)
    disturbed_joint_name = None

    for i in range(plant.num_actuators()):
        actuator = plant.get_joint_actuator(JointActuatorIndex(i))
        if actuator.model_instance() != model_instance:
            continue
        joint = actuator.joint()
        q_start = joint.position_start()
        q_full[q_start] += magnitude
        disturbed_joint_name = joint.name()
        break  # only perturb the first actuated joint

    plant.SetPositions(plant_context, q_full)
    return disturbed_joint_name


def run_sim(disturbance_time: float | None, disturbance_magnitude: float):
    meshcat = StartMeshcat()

    # Enable the built-in joint PD controllers.
    model = SpotModel(
        time_step=1e-3,
        enable_joint_pd=True,
        joint_kp=200.0,
        joint_kd=20.0,
    )
    diagram = model.build_robot_diagram(meshcat=meshcat)

    simulator = Simulator(diagram)
    root_context = simulator.get_mutable_context()

    plant = model.plant
    model_instance = model.model_instance
    plant_context = plant.GetMyMutableContextFromRoot(root_context)

    desired_state_port = plant.get_desired_state_input_port(model_instance)

    x_des = build_joint_desired_state(
        plant, plant_context, model_instance, desired_state_port
    )

    # Fix the desired state input
    desired_state_port.FixValue(plant_context, x_des)

    simulator.Initialize()
    simulator.set_target_realtime_rate(1.0)

    dt = 0.02  # simulation step


    # python3 spot_standing_sim.py --disturbance_time 5.0 --disturbance_magnitude 0.1
    if disturbance_time is None:
        disturbance_applied = True  # never apply
        print("Running simulation with NO disturbance. Press Ctrl+C to stop.")
    else:
        disturbance_applied = False
        print(
            f"Running simulation. Disturbance will be applied at t = "
            f"{disturbance_time:.2f} s (magnitude = {disturbance_magnitude} rad)."
        )
        print("Press Ctrl+C to stop.")

    try:
        while True:
            current_time = simulator.get_context().get_time()
            target_time = current_time + dt

            if (not disturbance_applied) and (target_time >= disturbance_time):
                simulator.AdvanceTo(disturbance_time)

                # Refresh contexts after advancing
                root_context = simulator.get_mutable_context()
                plant_context = plant.GetMyMutableContextFromRoot(root_context)

                joint_name = apply_first_joint_position_disturbance(
                    plant,
                    plant_context,
                    model_instance,
                    magnitude=disturbance_magnitude,
                )

                if joint_name is not None:
                    print(
                        f"Applied disturbance at t = {disturbance_time:.2f} s "
                        f"to joint '{joint_name}' (+{disturbance_magnitude} rad)."
                    )
                else:
                    print("Warning: could not find an actuated joint to disturb.")

                disturbance_applied = True

            else:
                simulator.AdvanceTo(target_time)

    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")


def main():
    parser = argparse.ArgumentParser(
        description="Spot standing simulation with optional PD disturbance."
    )
    parser.add_argument(
        "--disturbance_time",
        type=float,
        default=None,
        help="Time [s] at which to apply a joint disturbance. "
             "If omitted, no disturbance is applied.",
    )
    parser.add_argument(
        "--disturbance_magnitude",
        type=float,
        default=0.2,
        help="Disturbance magnitude in radians (default: 0.2).",
    )
    args = parser.parse_args()

    run_sim(
        disturbance_time=args.disturbance_time,
        disturbance_magnitude=args.disturbance_magnitude,
    )


if __name__ == "__main__":
    main()

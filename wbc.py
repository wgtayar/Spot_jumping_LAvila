##### Whole Body Controller given foot forces #####
import numpy as np
from pydrake.all import (
    RobotDiagramBuilder,
    PidController,
    DiscreteContactApproximation,
    AddDefaultVisualization,
    Simulator,
    StartMeshcat,
)
PARENT_FOLDER = "jump_in_place_forces" # where forces got backed out
from underactuated import ConfigureParser

fl = np.load(f"{PARENT_FOLDER}/front_left_forces.npy")
fr = np.load(f"{PARENT_FOLDER}/front_right_forces.npy")
rl = np.load(f"{PARENT_FOLDER}/rear_left_forces.npy")
rr = np.load(f"{PARENT_FOLDER}/rear_right_forces.npy")

robot_builder = RobotDiagramBuilder(time_step=1e-4)

parser = robot_builder.parser()
ConfigureParser(parser)
parser.AddModelsFromUrl("package://underactuated/models/spot/spot.dmd.yaml")
parser.AddModelsFromUrl("package://underactuated/models/littledog/ground.urdf")
plant = robot_builder.plant()
plant.set_discrete_contact_approximation(DiscreteContactApproximation.kLagged)
plant.Finalize()

builder = robot_builder.builder()

nq = plant.num_positions()
nv = plant.num_velocities()


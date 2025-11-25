# external/spot/spot_multi_step_playback.py
#
# Script to compute a multi-step walking trajectory using gait_optimization
# and play it back in Meshcat as a kinematic animation.

from __future__ import annotations

import numpy as np
import time
import matplotlib.pyplot as plt

from pydrake.all import (
    Simulator,
    Parser,
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    StartMeshcat,
    MeshcatVisualizer,
    PiecewisePolynomial,
)

from spot_lqr_standing import (
    get_default_standing_state,
)
from spot_footstep_plan import (
    generate_straight_footstep_plan,
    get_leg_names,
)
from gait_optimization import gait_optimization, UNDERACTUATED_JOINT_NAMES

from underactuated import ConfigureParser


def build_optimization_plant():
    """
    Build a plant specifically for gait optimization.
    This needs to capture the model instance index.
    
    Returns:
        diagram, plant, spot_model_instance
    """
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = Parser(plant)
    
    ConfigureParser(parser)
    
    # Load Spot and capture the model instance
    (spot,) = parser.AddModelsFromUrl(
        "package://underactuated/models/spot/spot.dmd.yaml"
    )
    parser.AddModelsFromUrl(
        "package://underactuated/models/littledog/ground.urdf"
    )
    
    plant.Finalize()
    diagram = builder.Build()
    
    return diagram, plant, spot


def get_foot_positions(plant, plant_context):
    """
    Get the world positions of the four foot frames (lower leg frames).
    
    Args:
        plant: MultibodyPlant instance
        plant_context: Context for the plant
        
    Returns:
        foot_positions: numpy array of shape (4, 3) with positions in order:
                       [front_left, front_right, rear_left, rear_right]
    """
    world_frame = plant.world_frame()
    
    # Define the four foot frame names in order (using foot_center frames)
    foot_frame_names = [
        "front_left_foot_center",
        "front_right_foot_center",
        "rear_left_foot_center",
        "rear_right_foot_center"
    ]
    
    foot_positions = np.zeros((4, 3))
    
    for i, frame_name in enumerate(foot_frame_names):
        frame = plant.GetFrameByName(frame_name)
        p_WF = plant.CalcPointsPositions(
            plant_context,
            frame,
            np.array([[0.0], [0.0], [0.0]]),
            world_frame
        )
        foot_positions[i, :] = p_WF.flatten()
    
    return foot_positions


def compute_multi_step_trajectory(num_steps: int = 4):
    """
    Compute a multi-step walking trajectory using gait_optimization.
    
    Args:
        num_steps: Number of footsteps to execute
    
    Returns:
        t_sol: Time samples (numpy array)
        q_sol: Joint trajectory (PiecewisePolynomial)
        v_sol: Velocity trajectory (PiecewisePolynomial)
        q0: Initial joint configuration (numpy array)
    """
    print("=" * 80)
    print(f"Computing Multi-Step Trajectory ({num_steps} steps)")
    print("=" * 80)
    
    # Build plant for optimization (need model instance)
    print("\nBuilding optimization plant...")
    diagram, plant, spot_model = build_optimization_plant()
    
    # Create simulator and contexts
    simulator = Simulator(diagram)
    root_context = simulator.get_mutable_context()
    plant_context = plant.GetMyMutableContextFromRoot(root_context)
    
    # Set standing state
    print("Setting nominal standing state...")
    q_star, v_star = get_default_standing_state(plant)
    plant.SetPositions(plant_context, q_star)
    plant.SetVelocities(plant_context, v_star)
    
    # Save the initial configuration
    q0 = plant.GetPositions(plant_context).copy()
    
    # Get initial foot positions
    print("Getting initial foot positions...")
    foot_pos_array = get_foot_positions(plant, plant_context)
    
    # Compute ground height from mean foot z position (for information)
    ground_height = np.mean(foot_pos_array[:, 2])
    print(f"\nGround height (mean foot z): {ground_height:.4f}")
    
    print("\nInitial foot positions (our ordering: FL, FR, RL, RR):")
    leg_names = get_leg_names()
    for i, name in enumerate(leg_names):
        pos = foot_pos_array[i, :]
        print(f"  {name:25s}: [{pos[0]:8.4f}, {pos[1]:8.4f}, {pos[2]:8.4f}]")
    
    # Map our leg_index to gait_optimization index
    # Our ordering: FL (0), FR (1), RL (2), RR (3)
    # gait_optimization ordering (foot_frame list): FL (0), FR (1), RL (2), RR (3)
    # They are the SAME ordering, so identity mapping!
    our_to_opt_index = {
        0: 0,  # front_left -> FL (index 0)
        1: 1,  # front_right -> FR (index 1)
        2: 2,  # rear_left -> RL (index 2)
        3: 3,  # rear_right -> RR (index 3)
    }
    
    # Storage for trajectory segments
    segment_times = []
    segment_q_traj = []
    segment_v_traj = []
    all_torque_data = []  # Store torque data from each phase
    t_offset = 0.0
    
    # Set box_height (ground level)
    box_height = 0.016
    
    # Step parameters
    step_length = 0.15  # meters per step
    
    # Trot gait: diagonal pairs move together
    # Our leg indices: 0=FL, 1=FR, 2=RL, 3=RR
    # Phase 0: FL + RR swing together (indices 0, 3)
    # Phase 1: FR + RL swing together (indices 1, 2)
    gait_phases = [(0, 3), (1, 2)]  # (front_left, rear_right), (front_right, rear_left)
    
    print("\n" + "=" * 80)
    print("Optimizing trajectory for each phase (diagonal pair trot)...")
    print("=" * 80)
    
    # Loop over each phase, swinging diagonal pairs
    for step_num in range(num_steps):
        print(f"\n--- Phase {step_num + 1}/{num_steps} ---")
        
        # Print current foot positions before planning
        print(f"  Current foot positions:")
        for i, name in enumerate(leg_names):
            pos = foot_pos_array[i, :]
            print(f"    {name:15s}: x={pos[0]:7.4f}, y={pos[1]:7.4f}, z={pos[2]:7.4f}")
        
        # Select which diagonal pair moves in this phase
        swing_legs_our = gait_phases[step_num % len(gait_phases)]
        
        # Compute target positions for both swing legs
        swing_leg_names = [leg_names[li] for li in swing_legs_our]
        target_positions = []
        for leg_idx in swing_legs_our:
            target_pos = foot_pos_array[leg_idx].copy()
            target_pos[0] += step_length  # Move forward in x
            target_positions.append(target_pos)
        
        print(f"  Selected diagonal pair from gait phase:")
        for leg_idx, leg_name, target_pos in zip(swing_legs_our, swing_leg_names, target_positions):
            print(f"    Leg {leg_idx} ({leg_name}): target x={target_pos[0]:.4f}, "
                  f"y={target_pos[1]:.4f}, z={target_pos[2]:.4f}")
        
        # Build next_foot array for gait_optimization
        # gait_optimization expects shape (4, 2) with ordering: FL, FR, RL, RR (same as ours!)
        # Start with current positions (x, y only - ground plane coordinates)
        next_foot = np.zeros((4, 2))
        next_foot[0, :] = [foot_pos_array[0, 0], foot_pos_array[0, 1]]  # FL = front_left
        next_foot[1, :] = [foot_pos_array[1, 0], foot_pos_array[1, 1]]  # FR = front_right
        next_foot[2, :] = [foot_pos_array[2, 0], foot_pos_array[2, 1]]  # RL = rear_left
        next_foot[3, :] = [foot_pos_array[3, 0], foot_pos_array[3, 1]]  # RR = rear_right
        
        # Map both swing leg indices to gait_optimization indices
        swing_feet_opt = [our_to_opt_index[li] for li in swing_legs_our]
        
        # Update both swing feet with their target positions
        for leg_idx, target_pos in zip(swing_legs_our, target_positions):
            opt_idx = our_to_opt_index[leg_idx]
            next_foot[opt_idx, 0] = target_pos[0]  # Update x
            next_foot[opt_idx, 1] = target_pos[1]  # Update y
        
        opt_leg_names = ["front_left (FL)", "front_right (FR)", 
                        "rear_left (RL)", "rear_right (RR)"]
        swing_leg_names_opt = [opt_leg_names[idx] for idx in swing_feet_opt]
        print(f"  Swing feet (diagonal pair): {', '.join(swing_leg_names_opt)}")
        
        # Debug: print body height and foot positions
        current_q = plant.GetPositions(plant_context)
        print(f"  Debug - Body height (q[6]): {current_q[6]:.4f}")
        print(f"  Debug - next_foot array:")
        for i, name in enumerate(opt_leg_names):
            swing_marker = " [SWING]" if i in swing_feet_opt else " [STANCE]"
            print(f"    {name}: x={next_foot[i, 0]:.4f}, y={next_foot[i, 1]:.4f}{swing_marker}")
        
        # Call gait_optimization for this phase (with diagonal pair)
        t_step, q_step, v_step, q_end, torque_data = gait_optimization(
            plant,
            plant_context,
            spot_model,
            next_foot,
            swing_feet_opt,
            box_height
        )
        
        print(f"  Optimization complete: duration = {t_step[-1] - t_step[0]:.4f} s")
        
        # Store segment trajectories and time arrays
        segment_times.append(t_step.copy())
        segment_q_traj.append(q_step)
        segment_v_traj.append(v_step)
        
        # Store torque data with time offset for stitching
        torque_data['phase'] = step_num + 1
        torque_data['time_offset'] = t_offset if step_num > 0 else 0.0
        all_torque_data.append(torque_data)
        
        # Update time offset for next segment
        t_offset += (t_step[-1] - t_step[0])
        
        # Update plant state to end of this step
        plant.SetPositions(plant_context, q_end)
        plant.SetVelocities(plant_context, v_step.value(t_step[-1]).flatten())
        
        # Recompute foot positions for next iteration
        foot_pos_array = get_foot_positions(plant, plant_context)
        
        print(f"  Updated foot positions after step:")
    
    print("\n" + "=" * 80)
    print("All steps optimized! Stitching trajectory segments...")
    print("=" * 80)
    
    # Stitch trajectories: concatenate times with proper offsets, skip duplicate endpoints
    t_sol_list = []
    q_samples_list = []
    v_samples_list = []
    cumulative_time = 0.0
    
    for i, (t_seg, q_traj, v_traj) in enumerate(zip(segment_times, segment_q_traj, segment_v_traj)):
        if i == 0:
            # First segment: include all samples
            t_sol_list.append(t_seg)
            q_samples = np.column_stack([q_traj.value(t) for t in t_seg])
            v_samples = np.column_stack([v_traj.value(t) for t in t_seg])
        else:
            # Subsequent segments: skip first sample (duplicates last of previous segment)
            t_sol_list.append(t_seg[1:] + cumulative_time)
            q_samples = np.column_stack([q_traj.value(t) for t in t_seg[1:]])
            v_samples = np.column_stack([v_traj.value(t) for t in t_seg[1:]])
        
        q_samples_list.append(q_samples)
        v_samples_list.append(v_samples)
        
        # Update cumulative time for next segment
        cumulative_time += (t_seg[-1] - t_seg[0])
    
    # Concatenate all time samples and trajectory samples
    t_sol = np.concatenate(t_sol_list)
    q_all = np.concatenate(q_samples_list, axis=1)
    v_all = np.concatenate(v_samples_list, axis=1)
    
    # Create new piecewise polynomials over the full time horizon
    q_sol = PiecewisePolynomial.FirstOrderHold(t_sol, q_all)
    v_sol = PiecewisePolynomial.FirstOrderHold(t_sol, v_all)
    
    print(f"  Total trajectory duration: {t_sol[-1]:.4f} s")
    print(f"  Total time samples: {len(t_sol)}")
    
    return t_sol, q_sol, v_sol, q0, all_torque_data


def build_visualization_diagram():
    """
    Build a diagram with Meshcat visualization for kinematic playback.
    
    Returns:
        diagram: Built diagram
        plant: MultibodyPlant instance
        meshcat: Meshcat instance
        visualizer: MeshcatVisualizer instance
    """
    print("\n" + "=" * 80)
    print("Building visualization diagram...")
    print("=" * 80)
    
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = Parser(plant)
    
    ConfigureParser(parser)
    
    # Load Spot and ground
    parser.AddModelsFromUrl(
        "package://underactuated/models/spot/spot.dmd.yaml"
    )
    parser.AddModelsFromUrl(
        "package://underactuated/models/littledog/ground.urdf"
    )
    
    plant.Finalize()
    
    # Start Meshcat and add visualizer
    meshcat = StartMeshcat()
    visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat=meshcat)
    
    diagram = builder.Build()
    
    print(f"  Visualization diagram built")
    print(f"  Meshcat URL: {meshcat.web_url()}")
    
    return diagram, plant, meshcat, visualizer


def playback_trajectory(t_sol, q_sol, diagram, plant, meshcat, visualizer):
    """
    Play back the trajectory in Meshcat as a kinematic animation.
    
    Args:
        t_sol: Time samples (numpy array)
        q_sol: Joint trajectory (PiecewisePolynomial)
        diagram: Diagram with visualization
        plant: MultibodyPlant instance
        meshcat: Meshcat instance
        visualizer: MeshcatVisualizer instance
    """
    print("\n" + "=" * 80)
    print("Playing back trajectory in Meshcat...")
    print("=" * 80)
    
    # Create contexts ONCE at the start (reuse in the loop)
    root_context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(root_context)
    
    # Start recording using the visualizer
    visualizer.StartRecording()
    
    # Set up playback parameters
    t_start = t_sol[0]
    t_end = t_sol[-1]
    num_frames = 200  # More frames for longer trajectory
    time_samples = np.linspace(t_start, t_end, num_frames)
    playback_rate = 1.0  # Real-time playback
    frame_delay = (t_end - t_start) / num_frames * playback_rate
    
    print(f"  Trajectory duration: {t_end - t_start:.4f} s")
    print(f"  Number of frames: {num_frames}")
    print(f"  Frame delay: {frame_delay:.4f} s")
    print(f"  Playback rate: {playback_rate}x")
    print(f"\nPlaying trajectory...")
    
    # Get a foot frame for debugging
    front_left_foot = plant.GetFrameByName("front_left_foot_center")
    world_frame = plant.world_frame()
    
    # Playback loop
    for i, t in enumerate(time_samples):
        # Set the diagram time for this frame
        root_context.SetTime(float(t))
        
        # Evaluate trajectory at this time
        q = q_sol.value(t).flatten()
        
        # Set plant positions on the VISUALIZATION plant
        plant.SetPositions(plant_context, q)
        
        # Publish to Meshcat (this sends the pose update)
        diagram.ForcedPublish(root_context)
        
        # Delay for visualization
        time.sleep(frame_delay)
        
        # Progress indicator with debug info
        if (i + 1) % 40 == 0 or i == num_frames - 1 or i == 0:
            progress = (i + 1) / num_frames * 100
            # Debug: print configuration changes
            foot_pos = plant.CalcPointsPositions(
                plant_context,
                front_left_foot,
                np.array([[0.0], [0.0], [0.0]]),
                world_frame
            ).flatten()
            print(f"  Progress: {progress:.1f}% (frame {i + 1}/{num_frames}) | "
                  f"t = {t:.3f} s | "
                  f"body x = {q[4]:.3f} | "
                  f"FL foot z = {foot_pos[2]:.3f}")
    
    # Stop and publish recording using the visualizer
    visualizer.StopRecording()
    visualizer.PublishRecording()
    
    print("\n" + "=" * 80)
    print("Playback complete!")
    print("=" * 80)
    print(f"Recording published to Meshcat")
    print(f"You can replay it using the Meshcat controls")
    print(f"Meshcat URL: {meshcat.web_url()}")


def print_underactuation_report(all_torque_data: list):
    """
    Print a comprehensive report on underactuation verification.
    
    Args:
        all_torque_data: List of torque data dicts from each phase
    """
    print("\n" + "=" * 80)
    print("UNDERACTUATION VERIFICATION REPORT")
    print("=" * 80)
    
    if not all_torque_data or not all_torque_data[0].get('underactuated_indices'):
        print("\n  No underactuation constraints were applied.")
        print("  All joints are fully actuated.")
        return
    
    underactuated_names = all_torque_data[0].get('underactuated_names', [])
    underactuated_indices = all_torque_data[0].get('underactuated_indices', [])
    
    print(f"\n  Underactuated Joints: {underactuated_names}")
    print(f"  Velocity Indices: {underactuated_indices}")
    print(f"  Constraint: τ = 0 (no motor torque at these joints)")
    print()
    
    # Table header
    print(f"  {'Phase':<8} {'Duration (s)':<14} {'Max |τ_passive| (N⋅m)':<24} {'Status':<10}")
    print("  " + "-" * 60)
    
    overall_max = 0.0
    total_duration = 0.0
    
    for data in all_torque_data:
        phase = data.get('phase', '?')
        max_tau = data.get('max_violation', 0.0)
        times = data.get('times', np.array([]))
        duration = times[-1] - times[0] if len(times) > 1 else 0.0
        
        overall_max = max(overall_max, max_tau)
        total_duration += duration
        
        status = "✓ PASS" if max_tau < 1e-3 else "✗ FAIL"
        print(f"  {phase:<8} {duration:<14.4f} {max_tau:<24.6f} {status:<10}")
    
    print("  " + "-" * 60)
    print(f"  {'TOTAL':<8} {total_duration:<14.4f} {overall_max:<24.6f}")
    
    print("\n" + "=" * 80)
    if overall_max < 1e-3:
        print("  ✓ UNDERACTUATION VERIFIED!")
        print("    The trajectory is dynamically feasible with passive (unactuated) joints.")
        print(f"    Maximum torque at passive joints: {overall_max:.6f} N⋅m (effectively zero)")
    else:
        print("  ⚠ WARNING: Underactuation constraints may be violated!")
        print(f"    Maximum torque at passive joints: {overall_max:.6f} N⋅m")
    print("=" * 80)


def plot_torque_comparison(all_torque_data: list, save_path: str = None):
    """
    Create plots comparing actuated vs underactuated joint torques.
    
    Args:
        all_torque_data: List of torque data dicts from each phase
        save_path: Optional path to save the figure
    """
    if not all_torque_data:
        print("No torque data to plot.")
        return
    
    # Stitch together all phases
    all_times = []
    all_tau = []
    cumulative_time = 0.0
    
    for data in all_torque_data:
        times = data.get('times', np.array([]))
        tau_all = data.get('tau_all', np.array([]))
        
        if len(times) == 0 or len(tau_all) == 0:
            continue
        
        # Offset times for stitching
        all_times.append(times + cumulative_time)
        all_tau.append(tau_all)
        cumulative_time = all_times[-1][-1]
    
    if not all_times:
        print("No valid torque data to plot.")
        return
    
    times = np.concatenate(all_times)
    tau = np.vstack(all_tau)
    
    underactuated_indices = all_torque_data[0].get('underactuated_indices', [])
    underactuated_names = all_torque_data[0].get('underactuated_names', [])
    joint_names = all_torque_data[0].get('joint_names', [])
    
    # Get actuated indices (all indices not in underactuated)
    nv = tau.shape[1]
    actuated_indices = [i for i in range(nv) if i not in underactuated_indices]
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle('Joint Torque Analysis: Actuated vs Underactuated Joints', fontsize=14, fontweight='bold')
    
    # Color scheme
    actuated_colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(actuated_indices)))
    underactuated_color = 'red'
    
    # === Subplot 1: All actuated joint torques ===
    ax1 = axes[0]
    ax1.set_title('Actuated Joint Torques (motors active)', fontsize=12)
    
    for i, idx in enumerate(actuated_indices):
        name = joint_names[idx] if idx < len(joint_names) else f"joint_{idx}"
        # Only label a few for clarity
        label = name if i < 6 else None
        ax1.plot(times, tau[:, idx], color=actuated_colors[i % len(actuated_colors)], 
                 alpha=0.7, linewidth=1, label=label)
    
    ax1.set_ylabel('Torque (N⋅m)')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.legend(loc='upper right', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # === Subplot 2: Underactuated joint torques (should be ~0) ===
    ax2 = axes[1]
    ax2.set_title(f'Underactuated Joint Torques (PASSIVE - should be ≈ 0): {underactuated_names}', 
                  fontsize=12, color='red')
    
    if underactuated_indices:
        for i, idx in enumerate(underactuated_indices):
            name = underactuated_names[i] if i < len(underactuated_names) else f"passive_{idx}"
            ax2.plot(times, tau[:, idx], color=underactuated_color, 
                     linewidth=2, label=name, alpha=0.8)
        
        # Add shaded region for "acceptable" range
        max_acceptable = 0.001  # 1 mN⋅m
        ax2.axhspan(-max_acceptable, max_acceptable, color='green', alpha=0.2, 
                    label=f'Acceptable range (±{max_acceptable*1000:.1f} mN⋅m)')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Set y-axis limits to zoom into the small values
        max_val = np.max(np.abs(tau[:, underactuated_indices])) * 1.5
        max_val = max(max_val, 0.01)  # At least show ±10 mN⋅m
        ax2.set_ylim(-max_val, max_val)
    else:
        ax2.text(0.5, 0.5, 'No underactuated joints specified', 
                 transform=ax2.transAxes, ha='center', va='center', fontsize=12)
    
    ax2.set_ylabel('Torque (N⋅m)')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # === Subplot 3: Comparison - one actuated vs one underactuated ===
    ax3 = axes[2]
    ax3.set_title('Direct Comparison: Sample Actuated vs Underactuated Joint', fontsize=12)
    
    # Pick one actuated knee for comparison (e.g., front_right_knee)
    # Index 11 is typically front_right_knee in the velocity ordering
    comparison_actuated_idx = None
    for idx in actuated_indices:
        if idx < len(joint_names) and 'knee' in joint_names[idx]:
            comparison_actuated_idx = idx
            break
    
    if comparison_actuated_idx is None and actuated_indices:
        comparison_actuated_idx = actuated_indices[0]
    
    if comparison_actuated_idx is not None:
        actuated_name = joint_names[comparison_actuated_idx] if comparison_actuated_idx < len(joint_names) else f"joint_{comparison_actuated_idx}"
        ax3.plot(times, tau[:, comparison_actuated_idx], 'b-', 
                 linewidth=2, label=f'{actuated_name} (ACTUATED)', alpha=0.8)
    
    if underactuated_indices:
        passive_idx = underactuated_indices[0]
        passive_name = underactuated_names[0] if underactuated_names else f"joint_{passive_idx}"
        ax3.plot(times, tau[:, passive_idx], 'r-', 
                 linewidth=2, label=f'{passive_name} (PASSIVE)', alpha=0.8)
    
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Torque (N⋅m)')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n  Torque plot saved to: {save_path}")
    
    plt.show()


def plot_underactuation_detail(all_torque_data: list, save_path: str = None):
    """
    Create a detailed plot focused on underactuated joint torques.
    
    Args:
        all_torque_data: List of torque data dicts from each phase
        save_path: Optional path to save the figure
    """
    if not all_torque_data:
        print("No torque data to plot.")
        return
    
    underactuated_indices = all_torque_data[0].get('underactuated_indices', [])
    if not underactuated_indices:
        print("No underactuated joints to plot.")
        return
    
    underactuated_names = all_torque_data[0].get('underactuated_names', [])
    
    # Stitch together all phases
    all_times = []
    all_tau_underact = []
    phase_boundaries = [0.0]
    cumulative_time = 0.0
    
    for data in all_torque_data:
        times = data.get('times', np.array([]))
        tau_underact = data.get('tau_underactuated', np.array([]))
        
        if len(times) == 0 or len(tau_underact) == 0:
            continue
        
        all_times.append(times + cumulative_time)
        all_tau_underact.append(tau_underact)
        cumulative_time = all_times[-1][-1]
        phase_boundaries.append(cumulative_time)
    
    if not all_times:
        print("No valid underactuated torque data to plot.")
        return
    
    times = np.concatenate(all_times)
    tau_underact = np.vstack(all_tau_underact)
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f'Underactuation Verification: {underactuated_names}', 
                 fontsize=14, fontweight='bold')
    
    colors = ['red', 'darkred', 'orangered', 'crimson']
    
    # === Top: Time series ===
    ax1 = axes[0]
    ax1.set_title('Passive Joint Torques Over Time (Should be ≈ 0)', fontsize=12)
    
    for i in range(tau_underact.shape[1]):
        name = underactuated_names[i] if i < len(underactuated_names) else f"passive_{i}"
        ax1.plot(times, tau_underact[:, i], color=colors[i % len(colors)], 
                 linewidth=1.5, label=name, alpha=0.9)
    
    # Add phase boundaries
    for boundary in phase_boundaries[1:-1]:
        ax1.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    
    # Add acceptable region
    ax1.axhspan(-0.001, 0.001, color='green', alpha=0.15, label='Acceptable (±1 mN⋅m)')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    ax1.set_ylabel('Torque (N⋅m)')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Auto-scale with some padding
    max_abs = np.max(np.abs(tau_underact)) * 1.3
    max_abs = max(max_abs, 0.005)  # At least ±5 mN⋅m visible
    ax1.set_ylim(-max_abs, max_abs)
    
    # === Bottom: Histogram of torque values ===
    ax2 = axes[1]
    ax2.set_title('Distribution of Passive Joint Torques', fontsize=12)
    
    all_values = tau_underact.flatten()
    
    # Histogram
    n, bins, patches = ax2.hist(all_values * 1000, bins=50, color='red', alpha=0.7, 
                                 edgecolor='darkred', label='Torque samples')
    
    # Add vertical lines for statistics
    mean_val = np.mean(all_values) * 1000
    std_val = np.std(all_values) * 1000
    max_val = np.max(np.abs(all_values)) * 1000
    
    ax2.axvline(x=mean_val, color='blue', linestyle='-', linewidth=2, 
                label=f'Mean: {mean_val:.4f} mN⋅m')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
    
    # Acceptable region
    ax2.axvspan(-1, 1, color='green', alpha=0.2, label='Acceptable (±1 mN⋅m)')
    
    ax2.set_xlabel('Torque (mN⋅m)')
    ax2.set_ylabel('Count')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Add statistics text box
    stats_text = (f'Statistics:\n'
                  f'  Mean: {mean_val:.4f} mN⋅m\n'
                  f'  Std Dev: {std_val:.4f} mN⋅m\n'
                  f'  Max |τ|: {max_val:.4f} mN⋅m\n'
                  f'  Samples: {len(all_values)}')
    
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n  Underactuation detail plot saved to: {save_path}")
    
    plt.show()


def plot_joint_angles(all_torque_data: list, t_sol, q_sol, save_path: str = None):
    """
    Plot joint angle trajectories to show how passive vs actuated joints move.
    
    This helps visualize that passive joints DO move, but without torque.
    """
    if not all_torque_data:
        return
    
    # Sample the trajectory
    num_samples = 200
    times = np.linspace(t_sol[0], t_sol[-1], num_samples)
    
    # Get joint names
    joint_names = list(all_torque_data[0].get('joint_names', []))
    underactuated_names = all_torque_data[0].get('underactuated_names', [])
    
    if not joint_names:
        return
    
    # Sample joint positions over time
    # Joint angles in q are at indices 7-18 (after quaternion[4] + position[3])
    # Joint names from velocity view are at indices 6-17
    q_samples = np.zeros((num_samples, len(joint_names)))
    for i, t in enumerate(times):
        q_full = q_sol.value(t).flatten()
        # Map velocity indices to position indices
        for j in range(len(joint_names)):
            if j < 6:  # Skip body velocities (wx, wy, wz, vx, vy, vz)
                continue
            pos_idx = j + 1  # Offset for position vs velocity indexing
            if pos_idx < len(q_full):
                q_samples[i, j] = q_full[pos_idx]
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle('Joint Motion Analysis: Passive Joints MOVE but need NO TORQUE', 
                 fontsize=14, fontweight='bold')
    
    # Define joint groups
    knee_joints = ['front_left_knee', 'front_right_knee', 'rear_left_knee', 'rear_right_knee']
    
    # Plot 1: All knee angles with passive/actuated distinction
    ax1 = axes[0]
    ax1.set_title('Knee Joint Angles: PASSIVE vs ACTUATED\n(Both move, but passive needs no motor torque!)', 
                  fontsize=12, fontweight='bold')
    
    colors = {'front_left_knee': 'red', 'front_right_knee': 'orangered',
              'rear_left_knee': 'blue', 'rear_right_knee': 'darkblue'}
    
    for name in knee_joints:
        if name in joint_names:
            idx = joint_names.index(name)
            is_passive = name in underactuated_names
            linewidth = 3 if is_passive else 1.5
            linestyle = '-' if is_passive else '--'
            label = f"{name} {'[PASSIVE - τ=0]' if is_passive else '[ACTUATED - τ≠0]'}"
            ax1.plot(times, np.degrees(q_samples[:, idx]), 
                    linestyle=linestyle, linewidth=linewidth, 
                    color=colors.get(name, 'gray'), label=label, alpha=0.9)
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Joint Angle (degrees)')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([times[0], times[-1]])
    
    # Plot 2: Passive joint motion vs torque (key insight!)
    ax2 = axes[1]
    ax2.set_title('KEY INSIGHT: Passive Joint Moves ~20° with ZERO Torque', 
                  fontsize=12, fontweight='bold', color='darkgreen')
    
    # Get passive knee angle
    if underactuated_names and underactuated_names[0] in joint_names:
        knee_name = underactuated_names[0]
        knee_idx = joint_names.index(knee_name)
        knee_angles = np.degrees(q_samples[:, knee_idx])
        
        # Plot angle on primary axis
        line1, = ax2.plot(times, knee_angles, 'r-', linewidth=2.5, 
                          label=f'{knee_name} angle (deg)')
        ax2.set_ylabel('Joint Angle (degrees)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Add torque on secondary axis
        ax2_torque = ax2.twinx()
        
        # Collect all torque data for this joint
        all_tau_times = []
        all_tau_values = []
        cumulative_time = 0.0
        for data in all_torque_data:
            if 'times' in data and 'tau_underactuated' in data:
                phase_times = data['times']
                phase_torques = data['tau_underactuated']
                if len(phase_torques) > 0 and len(phase_times) > 0:
                    all_tau_times.extend(phase_times + cumulative_time)
                    # Get first underactuated joint torque
                    all_tau_values.extend([t[0] * 1000 if len(t) > 0 else 0 for t in phase_torques])
                    cumulative_time = phase_times[-1] + cumulative_time
        
        if all_tau_times:
            line2, = ax2_torque.plot(all_tau_times, all_tau_values, 'b-', 
                                     linewidth=1.5, alpha=0.7, label=f'{knee_name} torque (mN⋅m)')
            ax2_torque.axhline(y=0, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
            ax2_torque.fill_between(all_tau_times, -1, 1, alpha=0.15, color='green')
            ax2_torque.set_ylabel('Joint Torque (mN⋅m)', color='blue')
            ax2_torque.tick_params(axis='y', labelcolor='blue')
            ax2_torque.set_ylim([-5, 5])
            
            # Combined legend
            ax2.legend([line1, line2], [f'{knee_name} angle (deg)', f'{knee_name} torque (mN⋅m)'],
                      loc='upper right', fontsize=9)
        
        ax2.set_xlabel('Time (s)')
        ax2.set_xlim([times[0], times[-1]])
        
        # Add annotation
        mid_idx = len(times) // 2
        angle_range = np.max(knee_angles) - np.min(knee_angles)
        ax2.annotate(f'Motion: {angle_range:.1f}° range\nTorque: ≈0 mN⋅m', 
                    xy=(times[mid_idx], knee_angles[mid_idx]),
                    xytext=(times[mid_idx] - 2, np.max(knee_angles) + 3),
                    fontsize=11, color='darkgreen', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='darkgreen', alpha=0.7),
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Side-by-side comparison of actuated vs passive
    ax3 = axes[2]
    ax3.set_title('Motion Comparison: Both joints move, only actuated one needs torque', fontsize=12)
    
    # Find an actuated knee
    actuated_knee = None
    for name in ['rear_left_knee', 'rear_right_knee']:
        if name in joint_names and name not in underactuated_names:
            actuated_knee = name
            break
    
    passive_knee = underactuated_names[0] if underactuated_names else None
    
    if actuated_knee and passive_knee and actuated_knee in joint_names and passive_knee in joint_names:
        act_idx = joint_names.index(actuated_knee)
        pas_idx = joint_names.index(passive_knee)
        
        ax3.plot(times, np.degrees(q_samples[:, act_idx]), 'b-', 
                 linewidth=2, label=f'{actuated_knee} [ACTUATED]', alpha=0.8)
        ax3.plot(times, np.degrees(q_samples[:, pas_idx]), 'r-', 
                 linewidth=2, label=f'{passive_knee} [PASSIVE]', alpha=0.8)
        
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Joint Angle (degrees)')
        ax3.legend(loc='upper right', fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim([times[0], times[-1]])
        
        # Add text box explaining the difference
        explanation = ("Both joints move similarly during walking.\n"
                      "ACTUATED: requires 50-100+ N⋅m motor torque\n"
                      "PASSIVE: requires 0 N⋅m (moves freely)")
        ax3.text(0.02, 0.98, explanation, transform=ax3.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n  Joint angle analysis saved to: {save_path}")
    
    plt.show()


def print_motion_explanation():
    """Print an explanation of what underactuation means visually."""
    print("\n" + "=" * 80)
    print("UNDERSTANDING UNDERACTUATION IN THE VISUALIZATION")
    print("=" * 80)
    print("""
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  IMPORTANT: Passive joints DO move - they just move WITHOUT motor power │
  └─────────────────────────────────────────────────────────────────────────┘
  
  What you're seeing in the simulation:
  
  ✓ The front knees (passive) ARE bending and extending
  ✓ This motion is driven by:
    • Gravity pulling the lower leg down
    • Inertial forces from body/hip motion
    • Ground reaction forces pushing up through the leg
  
  ✓ The rear knees (actuated) ALSO bend and extend
  ✓ Their motion is driven by:
    • Motor torque (can be 50-100+ N⋅m as shown in plots)
    • Plus the same passive forces
  
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  The KEY difference:                                                     │
  │                                                                          │
  │  PASSIVE JOINT:  motion = f(gravity, inertia, contacts)                 │
  │                  torque ≈ 0 N⋅m (no motor!)                             │
  │                                                                          │
  │  ACTUATED JOINT: motion = f(gravity, inertia, contacts, motor_torque)   │
  │                  torque = 50-100+ N⋅m (motor working!)                  │
  └─────────────────────────────────────────────────────────────────────────┘
  
  See the plots:
  • torque_comparison.png     - Shows actuated joints need high torques
  • underactuation_detail.png - Shows passive joints have near-zero torque  
  • joint_angle_analysis.png  - Shows joints move but passive ones need no torque
""")


def main():
    """
    Main function: compute multi-step trajectory and play it back in Meshcat.
    """
    # Compute the multi-step trajectory
    t_sol, q_sol, v_sol, q0, all_torque_data = compute_multi_step_trajectory(num_steps=10)
    
    # Print comprehensive underactuation report
    print_underactuation_report(all_torque_data)
    
    # Generate plots
    print("\n" + "=" * 80)
    print("Generating Torque Analysis Plots...")
    print("=" * 80)
    
    plot_torque_comparison(all_torque_data, save_path='/home/hassan/Underactuated-Biped/external/spot/torque_comparison.png')
    plot_underactuation_detail(all_torque_data, save_path='/home/hassan/Underactuated-Biped/external/spot/underactuation_detail.png')
    plot_joint_angles(all_torque_data, t_sol, q_sol, save_path='/home/hassan/Underactuated-Biped/external/spot/joint_angle_analysis.png')
    
    # Print explanation of what underactuation means visually
    print_motion_explanation()
    
    # Build visualization diagram
    diagram, plant, meshcat, visualizer = build_visualization_diagram()
    
    # Set initial configuration in visualization
    print("\nSetting initial configuration in visualization...")
    root_context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(root_context)
    
    # Set initial configuration
    plant.SetPositions(plant_context, q0)
    diagram.ForcedPublish(root_context)
    
    print("\nWaiting 2 seconds before starting playback...")
    time.sleep(2.0)
    
    # Play back the trajectory
    playback_trajectory(t_sol, q_sol, diagram, plant, meshcat, visualizer)
    
    print("\n" + "=" * 80)
    print("Script complete!")
    print("=" * 80)
    print("The Meshcat window will remain open.")
    print("Press Ctrl+C to exit.")
    
    # Keep the script running so Meshcat stays open
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    main()

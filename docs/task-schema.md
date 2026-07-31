# Task Schema v1

Farpoint tasks are defined by a small YAML schema that describes the language instruction, scene setup, objects, camera capture, recording policy, output location, and success criteria.

The current schema version is:

```yaml
schema_version: "task.v1"
```

## Required Fields

```yaml
schema_version: "task.v1"
name: isaac_cube_scene
language_instruction: "Observe a red cube falling onto a ground plane."
frames: 60
record_every_n_frames: 1
scene:
  ground:
    enabled: true
    path: "/World/GroundPlane"
  lighting:
    type: "distant"
    path: "/World/DistantLight"
    intensity: 300
objects:
  cube:
    type: "dynamic_cube"
    path: "/World/RedCube"
    color: [1.0, 0.0, 0.0]
    position: [0.0, 0.0, 1.0]
    scale: [0.25, 0.25, 0.25]
    size: 1.0
camera:
  enabled: true
  position: [2.2, -3.0, 1.8]
  target: [0.0, 0.0, 0.25]
  resolution: [1280, 720]
  preview_frames: [0, 15, 30, 59]
  rt_subframes: 8
success_criteria:
  min_recorded_frames: 60
  final_cube_z:
    min: 0.10
    max: 0.15
output:
  root: "/workspace/project/outputs/episodes"
```

## Semantics

- `schema_version`: identifies the task contract. The first supported version is `task.v1`.
- `name`: stable task name used in reports and outputs.
- `language_instruction`: human-readable task instruction.
- `frames`: number of simulation frames to step.
- `record_every_n_frames`: frame stride for trajectory recording.
- `scene`: global scene configuration such as ground plane and lighting.
- `objects`: task objects. Current examples support one or more `dynamic_cube` objects.
- `camera`: preview camera and render settings for report playback.
- `success_criteria`: checks that determine whether the episode is a PASS.
- `output.root`: container-mounted episode output root.

## Deterministic Randomization

The UR10e + Robotiq task adds deterministic XY randomization:

```yaml
randomization:
  enabled: true
  pick_object:
    x: [0.94, 1.00]
    y: [0.22, 0.28]
  target_zone:
    x: [0.73, 0.79]
    y: [-0.09, -0.03]
  min_pick_target_separation: 0.20
  max_sampling_attempts: 100
```

`FARPOINT_EPISODE_SEED` selects the sample. The seed and resolved positions are written to both `metadata.json` and `metrics.json`, so an episode can be reproduced without inferring state from rendered frames.

## Success Criteria

The first task validates:

- `recorded_frames >= min_recorded_frames`
- `final_cube_z.min <= final_cube_position.z <= final_cube_z.max`

These checks are written to `metrics.json`:

```json
{
  "success": true,
  "success_checks": {
    "final_cube_z": true,
    "recorded_frames": true
  }
}
```

Future tasks should extend `success_criteria` instead of hard-coding PASS logic in reports.

## Multi-object Tabletop Example

`examples/isaac_tabletop_scene` uses the same schema version with a richer object map:

```yaml
name: isaac_tabletop_scene
scene:
  robot_pedestal:
    path: "/World/RobotPedestal"
    color: [0.32, 0.36, 0.38]
    position: [-0.55, 0.0, 0.26]
    scale: [0.28, 0.28, 0.26]
  table:
    path: "/World/Table"
    color: [0.55, 0.45, 0.34]
    position: [0.0, 0.0, 0.35]
    scale: [1.4, 0.9, 0.10]
objects:
  red_block:
    type: "dynamic_cube"
  blue_block:
    type: "dynamic_cube"
  green_block:
    type: "dynamic_cube"
success_criteria:
  min_recorded_frames: 90
  all_objects_min_z: 0.35
  all_objects_max_abs_xy: 0.85
```

The tabletop trajectory records an `objects` map per frame, with each object's position, orientation, linear velocity, and angular velocity. Episode reports render multi-object z-position traces when this format is present.

## Robot Arm Motion Example

`examples/isaac_robot_arm_scene` verifies that a local industrial robot USD asset can load as a real Isaac Sim articulation and be driven by `ArticulationController` joint position targets:

```yaml
name: isaac_robot_arm_scene
robot:
  name: ur10e
  prim_path: "/World/UR10e"
  asset_path: "/isaac-sim/exts/isaacsim.asset.transformer.rules/data/tests/ur10e/ur10e.usd"
success_criteria:
  min_recorded_frames: 60
  min_robot_prim_count: 10
  min_articulation_dofs: 6
  min_controlled_joints: 6
  min_joint_motion_degrees: 10.0
  min_preview_images: 5
```

The robot arm scene may include `scene.robot_pedestal` to mount the arm above the work surface and keep the robot visible in replay frames. The robot arm trajectory records joint names, commanded joint position targets, measured joint positions, degree-converted joint positions, and end-effector marker positions per frame. The scene also adds a debug visualization layer with an end-effector marker, RGB axes, a target marker, and joint command bars so controller behavior is visible even when the source robot asset is visually dark. This is a bridge toward pick-and-place: the platform now verifies that robot assets, workcell geometry, articulation control, moving robot telemetry, preview rendering, resource telemetry, phase markers, and episode reports work together before adding planners.

The robot arm scene also supports a scripted pick-and-place scaffold:

```yaml
scene:
  pick_object:
    path: "/World/PickObject"
    color: [0.90, 0.28, 0.08]
    position: [0.58, 0.04, 0.47]
    scale: [0.11, 0.11, 0.11]
success_criteria:
  max_pick_place_distance: 0.08
```

This scaffold remains useful for lightweight visualization tests, but the production manipulation example is now `examples/isaac_ur10e_robotiq_scene`.

## UR10e + Robotiq Pick-and-Place

The production example loads the Isaac Sim UR10e USD with its Robotiq 2F-85 variant and PhysX articulation. RMPflow supplies arm joint targets, the articulation controller drives the robot, and the grasp is represented by a solver-level fixed joint created only after the end effector reaches the configured tolerance.

The controller includes:

- A bounded grasp retry window while holding the grasp pose.
- A high-level transfer trajectory.
- A hover-align-descend placement trajectory.
- Release-time XY feedback using the observed attached-object position.
- A stable post-release observation window.

Failures are classified from failed acceptance checks into `infrastructure`, `motion_planning`, `grasp`, `pickup`, `transport`, `release`, `placement`, `settling`, or `evaluation`. `metrics.json` stores `failure_category`, `failure_reason`, and `failed_checks`.

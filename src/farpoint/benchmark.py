import copy
import math
import random


FAILURE_RULES = (
    (
        "infrastructure",
        "simulation_infrastructure_failure",
        {
            "recorded_frames",
            "robot_loaded",
            "robot_prim_count",
            "articulation_controller_initialized",
            "articulation_dofs",
            "controlled_joints",
            "preview_images",
            "robotiq_asset",
            "mimic_joint_prims",
            "real_grasp_body",
            "dataset_valid",
            "dataset_observations",
        },
    ),
    (
        "perception",
        "rgbd_pose_estimation_exceeded_tolerance",
        {"rgbd_perception", "perception_xy_error"},
    ),
    (
        "motion_planning",
        "motion_plan_did_not_reach_the_grasp_pose",
        {"joint_motion"},
    ),
    (
        "grasp",
        "gripper_did_not_form_a_valid_grasp",
        {
            "gripper_motion",
            "grasp_attach_distance",
            "grasp_created",
            "bilateral_contact",
            "no_grasp_joint",
        },
    ),
    (
        "pickup",
        "object_was_not_lifted_high_enough",
        {"lift_height"},
    ),
    (
        "transport",
        "object_was_not_held_stably_during_transport",
        {"object_attached_frames", "grasp_rigidity", "transport_contact"},
    ),
    (
        "release",
        "object_was_not_released",
        {"final_released"},
    ),
    (
        "placement",
        "object_finished_outside_the_target_tolerance",
        {
            "final_inside_target_zone",
            "final_target_xy_distance",
            "final_object_height",
            "max_final_object_height",
        },
    ),
    (
        "settling",
        "object_did_not_remain_stable_after_release",
        {"release_settle_frames", "post_release_stability"},
    ),
)


def _sample_xy(rng, bounds):
    return [
        round(rng.uniform(float(bounds["x"][0]), float(bounds["x"][1])), 6),
        round(rng.uniform(float(bounds["y"][0]), float(bounds["y"][1])), 6),
    ]


def _xy_distance(a, b):
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def randomize_task(task, seed):
    randomized = copy.deepcopy(task)
    config = randomized.get("randomization", {})
    if not config.get("enabled", False):
        return randomized, {
            "enabled": False,
            "seed": int(seed),
            "pick_object_xy": list(randomized["scene"]["pick_object"]["position"][:2]),
            "target_zone_xy": list(randomized["scene"]["target_zone"]["position"][:2]),
        }

    rng = random.Random(int(seed))
    pick_xy = _sample_xy(rng, config["pick_object"])
    target_xy = None
    min_separation = float(config.get("min_pick_target_separation", 0.0))
    max_attempts = int(config.get("max_sampling_attempts", 100))
    for _ in range(max_attempts):
        candidate = _sample_xy(rng, config["target_zone"])
        if _xy_distance(pick_xy, candidate) >= min_separation:
            target_xy = candidate
            break
    if target_xy is None:
        raise ValueError(
            f"unable to sample pick and target positions at least {min_separation:.3f} m apart"
        )

    randomized["scene"]["pick_object"]["position"][:2] = pick_xy
    randomized["scene"]["target_zone"]["position"][:2] = target_xy
    return randomized, {
        "enabled": True,
        "seed": int(seed),
        "pick_object_xy": pick_xy,
        "target_zone_xy": target_xy,
        "pick_target_separation": round(_xy_distance(pick_xy, target_xy), 6),
        "config": copy.deepcopy(config),
    }


def classify_failure(success_checks):
    failed_checks = sorted(name for name, passed in success_checks.items() if not passed)
    if not failed_checks:
        return {
            "failure_category": None,
            "failure_reason": None,
            "failed_checks": [],
        }

    failed_set = set(failed_checks)
    for category, reason, check_names in FAILURE_RULES:
        if failed_set.intersection(check_names):
            return {
                "failure_category": category,
                "failure_reason": reason,
                "failed_checks": failed_checks,
            }
    return {
        "failure_category": "evaluation",
        "failure_reason": "one_or_more_acceptance_checks_failed",
        "failed_checks": failed_checks,
    }

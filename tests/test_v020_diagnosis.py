from copy import deepcopy
from pathlib import Path

from farpoint.v020_diagnosis import build_v020_failure_diagnosis
from farpoint.v020_plan import build_v020_plan, load_v020_config


ROOT = Path(__file__).resolve().parents[1]


def test_v020_failure_diagnosis_stratifies_every_frozen_axis():
    config = load_v020_config(
        ROOT / "configs/variations/so101_v020_nominal300.json",
        project_root=ROOT,
    )
    plan = build_v020_plan(
        config,
        project_root=ROOT,
        plan_id="diagnosis-test",
        mode="pad-pilot",
        pad_dimensions_m=[0.09, 0.09, 0.01],
    )
    trial = plan["trials"][8]
    manifest = {
        "plan_sha256": plan["plan_sha256"],
        "git_commit": "a" * 40,
        "execution_status": "ABORTED",
        "selected_variations": {},
        "attempts": [
            {
                "attempt_id": "attempt-0",
                "episode_id": "episode-0",
                "variation_id": trial["variation_id"],
                "success": False,
                "dataset_valid": True,
                "failure_category": "oracle",
                "failure_reason": "bilateral_contact_lost:bilateral_settle",
            }
        ],
    }
    diagnosis = build_v020_failure_diagnosis(plan, deepcopy(manifest))
    assert diagnosis["failure_count"] == 1
    assert diagnosis["failure_counts"]["failure_class"] == {"bilateral_contact_lost": 1}
    assert diagnosis["failure_counts"]["object_variant"] == {"red-40mm-40g": 1}
    assert diagnosis["failure_counts"]["target_profile"] == {"target-b": 1}
    assert diagnosis["failure_counts"]["camera_profile"] == {"front-x-positive": 1}
    failure = diagnosis["failures"][0]
    assert failure["lhs_strata"] == trial["sampler"]["resolved"]["strata"]
    assert failure["cube_pose"]["yaw_degrees"] == trial["object_yaw_degrees"]

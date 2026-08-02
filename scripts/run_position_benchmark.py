#!/usr/bin/env python3
"""Run a contract pilot or the immutable 75-trial cube position benchmark."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from farpoint.episode_metadata import normalize_episode_metadata_v2  # noqa: E402
from farpoint.formal_benchmark import (  # noqa: E402
    abort_run_state,
    append_completed_trial,
    append_infrastructure_attempt,
    build_formal_manifest,
    build_release_selection,
    finish_run_state,
    infrastructure_retry_allowed,
    new_run_state,
    selected_trials,
    validate_formal_plan,
    validate_resume_state,
)
from farpoint.position_pilot import audit_pilot_episode, find_episode  # noqa: E402
from farpoint.position_plan import load_position_plan  # noqa: E402
from farpoint.release_spec import load_release_spec  # noqa: E402


DEFAULT_PLAN = (
    PROJECT_ROOT
    / "configs"
    / "plans"
    / "farpoint_v1_3_cube_position_expanded_candidate.json"
)
EXAMPLE_PATH = "examples/isaac_perception_contact_scene"
BENCHMARKS_ROOT = PROJECT_ROOT / "outputs" / "benchmarks"
EPISODES_ROOT = PROJECT_ROOT / "outputs" / "episodes"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def checked_git_revision(expected: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", expected):
        raise ValueError("--git-commit must be a full lowercase Git SHA")
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()
    if actual != expected:
        raise ValueError(f"checked-out Git revision is {actual}, expected {expected}")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True
    ).strip()
    if dirty:
        raise ValueError("formal benchmark requires a clean Git worktree")


def image_digest(image: str) -> str:
    result = subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{index .RepoDigests 0}}"],
        check=True,
        capture_output=True,
        text=True,
    )
    digest = result.stdout.strip().partition("@")[-1]
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise ValueError(f"could not resolve an immutable image digest for {image}")
    return digest


def runtime_directory(root: Path, benchmark_id: str, trial_id: str, attempt: int) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = root / benchmark_id / f"{stamp}_{trial_id}_a{attempt}"
    for name in ("cache", "compute", "config", "data", "logs", "pkg", "hub"):
        directory = path / name
        directory.mkdir(parents=True, exist_ok=True)
        directory.chmod(0o777)
    return path


def prepare_container_output_directories() -> None:
    for directory in (
        PROJECT_ROOT / "outputs",
        EPISODES_ROOT,
        EPISODES_ROOT / "_resources",
        EPISODES_ROOT / "_phases",
    ):
        directory.mkdir(parents=True, exist_ok=True)
        directory.chmod(0o777)


def audit_episode(
    episode: Path,
    trial: dict,
    *,
    plan: dict,
    git_commit: str,
    simulator_image_digest: str,
    dataset_episode_index: int,
) -> dict:
    audit = audit_pilot_episode(
        episode,
        trial,
        plan_sha256=plan["plan_sha256"],
        episode_root=EPISODES_ROOT,
    )
    errors = list(audit.get("errors", []))
    checks = dict(audit.get("checks", {}))
    try:
        metadata = read_json(episode / "metadata.json")
        metrics = read_json(episode / "metrics.json")
    except (OSError, ValueError) as error:
        errors.append(f"episode_json:{error}")
        checks["v2_metadata"] = False
        return {
            **audit,
            "status": "completed",
            "success": False,
            "dataset_valid": False,
            "checks": checks,
            "errors": errors,
            "accepted": False,
            "failure_category": "evaluation",
            "failure_reason": ", ".join(errors),
        }
    try:
        normalized = normalize_episode_metadata_v2(
            metadata,
            metrics,
            split=trial["split"],
            dataset_episode_index=dataset_episode_index,
            trial_id=trial["trial_id"],
        )
        provenance = normalized["provenance"]
        provenance_valid = (
            provenance["git_commit"] == git_commit
            and provenance["config_sha256"] == plan["config_sha256"]
            and provenance["simulator_image_digest"] == simulator_image_digest
        )
        variation_valid = (
            normalized["variation"]["variation_id"] == trial["variation_id"]
            and normalized["variation"]["cell_id"] == trial["cell_id"]
            and normalized["variation"]["slot"] == trial["slot"]
        )
    except (KeyError, TypeError, ValueError) as error:
        provenance_valid = False
        variation_valid = False
        errors.append(f"v2_metadata:{error}")
    checks["v2_metadata"] = not any(error.startswith("v2_metadata:") for error in errors)
    checks["provenance"] = provenance_valid
    checks["variation_contract"] = variation_valid
    for name in ("v2_metadata", "provenance", "variation_contract"):
        if not checks[name] and name not in errors:
            errors.append(name)
    accepted = not errors
    failure_category = metrics.get("failure_category")
    failure_reason = metrics.get("failure_reason")
    if not accepted and not failure_category:
        failure_category = "evaluation"
        failure_reason = ", ".join(errors)
    return {
        **audit,
        "status": "completed",
        "success": accepted,
        "dataset_valid": bool(metrics.get("dataset_valid")),
        "checks": checks,
        "errors": errors,
        "accepted": accepted,
        "failure_category": failure_category,
        "failure_reason": failure_reason,
    }


def refresh_reports(state_path: Path, episode: Path | None = None) -> None:
    if episode is not None:
        subprocess.run(
            [
                sys.executable,
                "scripts/build_episode_report.py",
                str(episode),
                "--output-dir",
                str(PROJECT_ROOT / "outputs" / "reports" / episode.name),
            ],
            cwd=PROJECT_ROOT,
            check=False,
        )
    subprocess.run(
        [sys.executable, "scripts/build_benchmark_report.py", str(state_path)],
        cwd=PROJECT_ROOT,
        check=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("pilot", "formal"))
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--benchmark-id")
    parser.add_argument("--git-commit", required=True)
    parser.add_argument(
        "--image",
        default=os.environ.get("ISAAC_SIM_IMAGE", "nvcr.io/nvidia/isaac-sim:6.0.0"),
    )
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=Path.home() / ".cache" / "farpoint" / "isaac-sim" / "benchmark-runs",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--pilot-trial-id",
        action="append",
        default=[],
        help="run an explicit frozen-plan trial in pilot mode; repeat as needed",
    )
    parser.add_argument("--cooldown-seconds", type=int, default=10)
    parser.add_argument("--retry-cooldown-seconds", type=int, default=180)
    parser.add_argument("--run-timeout-seconds", type=int, default=900)
    parser.add_argument("--startup-timeout-seconds", type=int, default=300)
    parser.add_argument("--max-infrastructure-attempts", type=int, default=3)
    args = parser.parse_args()
    if min(
        args.run_timeout_seconds,
        args.startup_timeout_seconds,
        args.max_infrastructure_attempts,
    ) < 1:
        parser.error("timeouts and max infrastructure attempts must be positive")
    if min(args.cooldown_seconds, args.retry_cooldown_seconds) < 0:
        parser.error("cooldown values cannot be negative")
    if args.mode == "formal" and args.pilot_trial_id:
        parser.error("--pilot-trial-id is not allowed in formal mode")

    checked_git_revision(args.git_commit)
    prepare_container_output_directories()
    plan = load_position_plan(args.plan)
    validate_formal_plan(plan)
    trials = selected_trials(plan, args.mode, args.pilot_trial_id)
    digest = image_digest(args.image)
    short_sha = args.git_commit[:7]
    prefix = "cube_position_formal" if args.mode == "formal" else "cube_position_contract_pilot"
    benchmark_id = args.benchmark_id or (
        f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%d')}_{short_sha}"
    )
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", benchmark_id):
        parser.error("benchmark ID contains unsupported characters")
    benchmark_dir = BENCHMARKS_ROOT / benchmark_id
    state_path = benchmark_dir / "run-state.json"
    manifest_path = benchmark_dir / "manifest.json"
    if manifest_path.exists():
        print(f"BENCHMARK_ALREADY_FINAL: {manifest_path}")
        return 0
    if state_path.exists():
        if not args.resume:
            parser.error(f"{benchmark_id} already exists; pass --resume")
        state = read_json(state_path)
        validate_resume_state(
            state,
            mode=args.mode,
            git_commit=args.git_commit,
            image_digest=digest,
            plan=plan,
            pilot_trial_ids=args.pilot_trial_id,
        )
        state["execution_status"] = "RUNNING"
        state.pop("failure_reason", None)
        state.pop("finished_at", None)
    else:
        state = new_run_state(
            benchmark_id=benchmark_id,
            mode=args.mode,
            git_commit=args.git_commit,
            image=args.image,
            image_digest=digest,
            plan=plan,
            pilot_trial_ids=args.pilot_trial_id,
        )
    write_json(state_path, state)
    relative_plan = args.plan.resolve().relative_to(PROJECT_ROOT.resolve())
    completed_ids = {trial["trial_id"] for trial in state.get("trials", [])}

    for ordinal, trial in enumerate(trials, start=1):
        if trial["trial_id"] in completed_ids:
            print(f"SKIP {trial['trial_id']}: already recorded", flush=True)
            continue
        episode = find_episode(EPISODES_ROOT, benchmark_id, trial["trial_id"])
        if episode:
            print(f"ADOPT {trial['trial_id']}: {episode.name}", flush=True)
        else:
            for attempt_number in range(1, args.max_infrastructure_attempts + 1):
                if state["completed_trials"] or attempt_number > 1:
                    delay = (
                        args.retry_cooldown_seconds
                        if attempt_number > 1
                        else args.cooldown_seconds
                    )
                    if delay:
                        time.sleep(delay)
                runtime = runtime_directory(
                    args.runtime_root,
                    benchmark_id,
                    trial["trial_id"],
                    attempt_number,
                )
                run_id = runtime.name
                attempt = append_infrastructure_attempt(
                    state,
                    trial,
                    attempt_number=attempt_number,
                    run_id=run_id,
                )
                write_json(state_path, state)
                print(
                    f"RUN {ordinal}/{len(trials)} {trial['trial_id']} "
                    f"seed={trial['seed']} attempt={attempt_number}",
                    flush=True,
                )
                command = [
                    "bash",
                    "scripts/run_remote_isaac_example.sh",
                    EXAMPLE_PATH,
                    args.image,
                    str(runtime),
                    run_id,
                    str(trial["seed"]),
                    benchmark_id,
                    "0",
                    "",
                    str(relative_plan),
                    trial["trial_id"],
                    "0",
                    args.git_commit,
                    plan["config_sha256"],
                    digest,
                ]
                run_env = os.environ.copy()
                run_env["FARPOINT_RUN_TIMEOUT_SECONDS"] = str(args.run_timeout_seconds)
                run_env["FARPOINT_STARTUP_TIMEOUT_SECONDS"] = str(
                    args.startup_timeout_seconds
                )
                result = subprocess.run(command, cwd=PROJECT_ROOT, env=run_env, check=False)
                episode = find_episode(EPISODES_ROOT, benchmark_id, trial["trial_id"])
                attempt.update(
                    {
                        "finished_at": utc_now(),
                        "return_code": result.returncode,
                        "episode_id": episode.name if episode else None,
                        "status": "EPISODE_RECORDED" if episode else "INFRASTRUCTURE_FAILURE",
                    }
                )
                write_json(state_path, state)
                if not infrastructure_retry_allowed(episode.name if episode else None):
                    break
                print(
                    f"INFRASTRUCTURE_RETRY {trial['trial_id']} attempt={attempt_number}",
                    flush=True,
                )
        if episode is None:
            reason = f"infrastructure attempts exhausted for {trial['trial_id']}"
            abort_run_state(state, reason)
            write_json(state_path, state)
            refresh_reports(state_path)
            print(f"BENCHMARK_ABORTED: {reason}", flush=True)
            return 2

        audited = audit_episode(
            episode,
            trial,
            plan=plan,
            git_commit=args.git_commit,
            simulator_image_digest=digest,
            dataset_episode_index=len(state["trials"]),
        )
        append_completed_trial(state, audited)
        write_json(state_path, state)
        refresh_reports(state_path, episode)
        print(
            f"TRIAL_{'PASS' if audited['success'] else 'FAIL'} "
            f"{trial['trial_id']} {episode.name}",
            flush=True,
        )

    finish_run_state(state)
    write_json(state_path, state)
    refresh_reports(state_path)
    if args.mode == "formal":
        manifest = build_formal_manifest(state, plan)
        write_json(manifest_path, manifest)
        if manifest["acceptance"]["accepted"]:
            selection = build_release_selection(
                manifest,
                dataset_id=load_release_spec()["dataset_id"],
            )
            write_json(benchmark_dir / "release-selection.json", selection)
        refresh_reports(manifest_path)
    print(
        f"POSITION_{args.mode.upper()} {'PASS' if state['accepted'] else 'FAIL'} "
        f"{state['passed_trials']}/{state['planned_trials']} {benchmark_dir}",
        flush=True,
    )
    return 0 if state["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

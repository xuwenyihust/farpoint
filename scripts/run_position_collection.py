#!/usr/bin/env python3
"""Run the resource-bounded Farpoint v1.3 balanced position collection."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from farpoint.formal_benchmark import (  # noqa: E402
    append_infrastructure_attempt,
    infrastructure_retry_allowed,
)
from farpoint.position_collection import (  # noqa: E402
    abort_collection,
    acceptance_snapshot,
    append_new_attempt,
    build_collection_manifest,
    build_collection_selection,
    complete_import_pilot,
    file_sha256,
    finish_collection,
    import_source_attempts,
    impossible_reason,
    load_collection_policy,
    new_collection_state,
    scheduled_trials,
    update_collection_progress,
    validate_collection_policy,
    validate_recovery_pilot_evidence,
    validate_resume_state,
)
from farpoint.position_plan import load_position_plan  # noqa: E402
from farpoint.release_spec import load_release_spec  # noqa: E402
from run_position_benchmark import (  # noqa: E402
    audit_episode,
    checked_git_revision,
    image_digest,
    prepare_container_output_directories,
    read_json,
    refresh_reports,
    runtime_directory,
    write_json,
)
from farpoint.position_pilot import find_episode  # noqa: E402


DEFAULT_POLICY = (
    PROJECT_ROOT / "configs" / "collections" / "farpoint_v1_3_cube_position_balanced.json"
)
BENCHMARKS_ROOT = PROJECT_ROOT / "outputs" / "benchmarks"
EPISODES_ROOT = PROJECT_ROOT / "outputs" / "episodes"
EXAMPLE_PATH = "examples/isaac_perception_contact_scene"
REQUIRED_SOURCE_ARTIFACTS = (
    "metadata.json",
    "metrics.json",
    "observations.jsonl",
    "phase_events.jsonl",
    "trajectory.jsonl",
    "labels.jsonl",
)
REQUIRED_SOURCE_DIRECTORIES = ("preview", "observations/rgb", "observations/depth")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_source_episode(episode_id: str, source_episode_roots: list[Path]) -> tuple[Path, Path]:
    for root in source_episode_roots:
        episode = root / episode_id
        if (
            episode.is_dir()
            and all((episode / relative).is_file() for relative in REQUIRED_SOURCE_ARTIFACTS)
            and all((episode / relative).is_dir() for relative in REQUIRED_SOURCE_DIRECTORIES)
        ):
            return episode, root
    raise ValueError(f"complete source episode artifacts are missing: {episode_id}")


def verify_source_artifacts(
    source_state: dict,
    source_episode_roots: list[Path],
    plan: dict,
    policy: dict,
) -> None:
    by_id = {trial["trial_id"]: trial for trial in plan["trials"]}
    source_type = policy["source"].get("type", "benchmark")
    records = (
        source_state.get("attempts") if source_type == "collection" else source_state.get("trials")
    ) or []
    for index, recorded in enumerate(records):
        episode, episode_root = resolve_source_episode(recorded["episode_id"], source_episode_roots)
        audited = audit_episode(
            episode,
            by_id[recorded["trial_id"]],
            plan=plan,
            git_commit=(
                recorded["source_git_commit"]
                if source_type == "collection"
                else policy["source"]["git_commit"]
            ),
            simulator_image_digest=policy["simulator_image_digest"],
            dataset_episode_index=index,
            episode_root=episode_root,
        )
        expected = {
            "trial_id": recorded["trial_id"],
            "episode_id": recorded["episode_id"],
            "success": recorded.get("outcome_success")
            if source_type == "collection"
            else recorded.get("success"),
            "dataset_valid": recorded.get("dataset_valid"),
        }
        for key, value in expected.items():
            if audited.get(key) != value:
                raise ValueError(
                    f"source artifact audit mismatch for {recorded['trial_id']}: {key}"
                )


def recorded_attempt_episode(state: dict, trial: dict) -> Path | None:
    attempt_id = trial.get("attempt_id", trial["trial_id"])
    for attempt in reversed(state.get("infrastructure_attempts", [])):
        if attempt.get("attempt_id") != attempt_id:
            continue
        episode_id = attempt.get("episode_id")
        if episode_id and (EPISODES_ROOT / episode_id).is_dir():
            return EPISODES_ROOT / episode_id
    return None


def link_imported_episodes(
    attempts: list[dict], source_episode_roots: list[Path], destination_root: Path
) -> None:
    destination_root.mkdir(parents=True, exist_ok=True)
    for attempt in attempts:
        if not attempt["selected_for_dataset"]:
            continue
        episode, _ = resolve_source_episode(attempt["episode_id"], source_episode_roots)
        source = episode.resolve()
        destination = destination_root / attempt["episode_id"]
        if destination.exists() or destination.is_symlink():
            if destination.resolve() != source:
                raise ValueError(f"imported episode destination conflict: {destination}")
            continue
        destination.symlink_to(source, target_is_directory=True)


def import_selection_manifest(
    collection_id: str, attempts: list[dict], source_episode_roots: list[Path]
) -> dict:
    episodes = []
    for attempt in attempts:
        if attempt["selected_for_dataset"]:
            episode, _ = resolve_source_episode(attempt["episode_id"], source_episode_roots)
            episodes.append(
                {
                    "episode_dir": str(episode.resolve()),
                    "trial_id": attempt["trial_id"],
                    "variation_id": attempt["variation_id"],
                    "split": attempt["dataset_split"],
                }
            )
    return {
        "schema_version": "farpoint.export-selection.v1",
        "dataset_id": load_release_spec()["dataset_id"],
        "collection_id": collection_id,
        "episodes": sorted(episodes, key=lambda row: row["trial_id"]),
    }


def refresh_collection_report(state_path: Path, episode: Path | None = None) -> None:
    refresh_reports(state_path, episode)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument(
        "--source-manifest",
        "--source-run-state",
        dest="source_manifest",
        type=Path,
        required=True,
    )
    parser.add_argument("--source-episode-root", type=Path, action="append", required=True)
    parser.add_argument("--recovery-pilot-state", type=Path)
    parser.add_argument("--collection-id")
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--import-only", action="store_true")
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=Path.home() / ".cache" / "farpoint" / "isaac-sim" / "collection-runs",
    )
    parser.add_argument("--cooldown-seconds", type=int, default=10)
    parser.add_argument("--retry-cooldown-seconds", type=int, default=180)
    parser.add_argument("--run-timeout-seconds", type=int, default=900)
    parser.add_argument("--startup-timeout-seconds", type=int, default=300)
    parser.add_argument("--max-infrastructure-attempts", type=int, default=3)
    args = parser.parse_args()
    if (
        min(
            args.run_timeout_seconds,
            args.startup_timeout_seconds,
            args.max_infrastructure_attempts,
        )
        < 1
    ):
        parser.error("timeouts and infrastructure attempts must be positive")
    if min(args.cooldown_seconds, args.retry_cooldown_seconds) < 0:
        parser.error("cooldown values cannot be negative")

    checked_git_revision(args.git_commit)
    policy = load_collection_policy(args.policy)
    policy_sha = file_sha256(args.policy)
    plan_path = PROJECT_ROOT / policy["position_plan"]
    plan = load_position_plan(plan_path)
    validate_collection_policy(policy, plan, PROJECT_ROOT)
    digest = image_digest(policy["simulator_image"])
    if digest != policy["simulator_image_digest"]:
        raise ValueError("local simulator image digest does not match collection policy")
    source_state = read_json(args.source_manifest)
    imported = import_source_attempts(source_state, policy, plan)
    verify_source_artifacts(source_state, args.source_episode_root, plan, policy)
    if policy.get("recovery_pilot"):
        if args.recovery_pilot_state is None:
            parser.error("recovery policy requires --recovery-pilot-state")
        validate_recovery_pilot_evidence(read_json(args.recovery_pilot_state), policy)
    elif args.recovery_pilot_state is not None:
        parser.error("--recovery-pilot-state requires a recovery policy")

    short_sha = args.git_commit[:7]
    collection_prefix = (
        "farpoint_v1_3_balanced_recovery_collection"
        if policy.get("recovery")
        else "farpoint_v1_3_balanced_collection"
    )
    collection_id = args.collection_id or (
        f"{collection_prefix}_{datetime.now(timezone.utc).strftime('%Y%m%d')}_{short_sha}"
    )
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", collection_id):
        parser.error("collection ID contains unsupported characters")
    collection_dir = BENCHMARKS_ROOT / collection_id
    state_path = collection_dir / "run-state.json"
    manifest_path = collection_dir / "manifest.json"
    if manifest_path.exists():
        print(f"COLLECTION_ALREADY_FINAL: {manifest_path}")
        return 0
    if state_path.exists():
        if not args.resume:
            parser.error(f"{collection_id} already exists; pass --resume")
        state = read_json(state_path)
        validate_resume_state(
            state,
            collection_id=collection_id,
            git_commit=args.git_commit,
            policy_sha256=policy_sha,
        )
        state["execution_status"] = "RUNNING"
        state["quality_status"] = "NOT_EVALUATED"
        state.pop("failure_reason", None)
        state.pop("finished_at", None)
        update_collection_progress(state, policy)
    else:
        state = new_collection_state(
            collection_id=collection_id,
            git_commit=args.git_commit,
            policy=policy,
            policy_sha256=policy_sha,
            imported_attempts=imported,
        )
    prepare_container_output_directories()
    link_imported_episodes(imported, args.source_episode_root, EPISODES_ROOT)
    write_json(state_path, state)
    refresh_collection_report(state_path)
    if args.import_only:
        complete_import_pilot(state, policy)
        write_json(state_path, state)
        write_json(
            collection_dir / "import-selection.json",
            import_selection_manifest(collection_id, imported, args.source_episode_root),
        )
        refresh_collection_report(state_path)
        print(
            f"COLLECTION_IMPORT_OK attempts={state['task_attempts']} "
            f"successes={state['task_successes']} selected={state['selected_episodes']}"
        )
        return 0

    relative_plan = plan_path.resolve().relative_to(PROJECT_ROOT.resolve())
    ordinal = 0
    while True:
        if acceptance_snapshot(state, policy)["accepted"]:
            break
        reason = impossible_reason(state, plan, policy)
        if reason:
            finish_collection(state, policy, failure_reason=reason)
            break
        candidates = scheduled_trials(state, plan, policy)
        if not candidates:
            break
        trial = candidates[0]
        ordinal += 1
        previous_episode = find_episode(EPISODES_ROOT, collection_id, trial["trial_id"])
        episode = recorded_attempt_episode(state, trial)
        if episode is None and not trial.get("recovery_retry"):
            episode = previous_episode
        if episode:
            print(
                f"ADOPT {trial.get('attempt_id', trial['trial_id'])}: {episode.name}",
                flush=True,
            )
        else:
            for attempt_number in range(1, args.max_infrastructure_attempts + 1):
                if state["task_attempts"] or attempt_number > 1:
                    delay = (
                        args.retry_cooldown_seconds if attempt_number > 1 else args.cooldown_seconds
                    )
                    if delay:
                        time.sleep(delay)
                runtime = runtime_directory(
                    args.runtime_root,
                    collection_id,
                    trial.get("attempt_id", trial["trial_id"]),
                    attempt_number,
                )
                infrastructure = append_infrastructure_attempt(
                    state,
                    trial,
                    attempt_number=attempt_number,
                    run_id=runtime.name,
                )
                infrastructure["attempt_id"] = trial.get("attempt_id", trial["trial_id"])
                write_json(state_path, state)
                print(
                    f"RUN {ordinal} {trial.get('attempt_id', trial['trial_id'])} "
                    f"seed={trial['seed']} attempt={attempt_number}",
                    flush=True,
                )
                command = [
                    "bash",
                    "scripts/run_remote_isaac_example.sh",
                    EXAMPLE_PATH,
                    policy["simulator_image"],
                    str(runtime),
                    runtime.name,
                    str(trial["seed"]),
                    collection_id,
                    "0",
                    "",
                    str(relative_plan),
                    trial["trial_id"],
                    "0",
                    args.git_commit,
                    policy["config_sha256"],
                    digest,
                ]
                env = os.environ.copy()
                env["FARPOINT_RUN_TIMEOUT_SECONDS"] = str(args.run_timeout_seconds)
                env["FARPOINT_STARTUP_TIMEOUT_SECONDS"] = str(args.startup_timeout_seconds)
                result = subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=False)
                latest = find_episode(EPISODES_ROOT, collection_id, trial["trial_id"])
                episode = (
                    latest
                    if latest and (previous_episode is None or latest.name != previous_episode.name)
                    else None
                )
                infrastructure.update(
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
                    "INFRASTRUCTURE_RETRY "
                    f"{trial.get('attempt_id', trial['trial_id'])} "
                    f"attempt={attempt_number}",
                    flush=True,
                )
        if episode is None:
            abort_collection(
                state,
                policy,
                f"infrastructure attempts exhausted for {trial['trial_id']}",
            )
            write_json(state_path, state)
            refresh_collection_report(state_path)
            return 2
        audited = audit_episode(
            episode,
            trial,
            plan=plan,
            git_commit=args.git_commit,
            simulator_image_digest=digest,
            dataset_episode_index=state["selected_episodes"],
        )
        append_new_attempt(state, trial, audited, policy)
        write_json(state_path, state)
        refresh_collection_report(state_path, episode)
        print(
            f"TRIAL_{'PASS' if audited['success'] else 'FAIL'} "
            f"{trial.get('attempt_id', trial['trial_id'])} {episode.name}",
            flush=True,
        )
    if state["execution_status"] == "RUNNING":
        reason = impossible_reason(state, plan, policy)
        finish_collection(state, policy, failure_reason=reason or "candidate_schedule_exhausted")
    write_json(state_path, state)
    manifest = build_collection_manifest(state, policy)
    write_json(manifest_path, manifest)
    if manifest["acceptance"]["accepted"]:
        selection = build_collection_selection(
            manifest, dataset_id=load_release_spec()["dataset_id"]
        )
        write_json(collection_dir / "release-selection.json", selection)
    refresh_collection_report(manifest_path)
    print(
        f"POSITION_COLLECTION {'PASS' if state['accepted'] else 'FAIL'} "
        f"selected={state['selected_episodes']} attempts={state['task_attempts']} "
        f"yield={state['task_yield']:.1%} {collection_dir}",
        flush=True,
    )
    return 0 if state["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

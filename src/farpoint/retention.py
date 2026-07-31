import json
import shutil
import uuid
from datetime import timedelta
from pathlib import Path

from .registry import EpisodeRegistry, read_json, utc_now


DEFAULT_POLICY = {
    "statuses": ["FAIL", "INCOMPLETE"],
    "minimum_age_hours": 24,
    "recovery_window_days": 7,
    "protect_pinned": True,
    "protect_benchmarks": True,
}


def append_jsonl(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


class RetentionManager:
    def __init__(self, registry):
        if not isinstance(registry, EpisodeRegistry):
            raise TypeError("registry must be an EpisodeRegistry")
        self.registry = registry
        self.layout = registry.layout

    def load_policy(self):
        path = self.layout.state / "retention-policy.json"
        if not path.exists():
            self.save_policy(DEFAULT_POLICY)
            return dict(DEFAULT_POLICY)
        policy = dict(DEFAULT_POLICY)
        policy.update(read_json(path))
        return policy

    def save_policy(self, policy):
        merged = dict(DEFAULT_POLICY)
        merged.update(policy)
        path = self.layout.state / "retention-policy.json"
        path.write_text(
            json.dumps(merged, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return merged

    def preview(self, policy=None):
        policy = dict(policy or self.load_policy())
        statuses = {str(value).upper() for value in policy["statuses"]}
        minimum_age = timedelta(hours=float(policy["minimum_age_hours"]))
        now = utc_now()
        candidates = []
        retained = []
        for row in self.registry.list_episodes(limit=100000):
            reasons = []
            started = row.get("finished_at") or row.get("updated_at")
            try:
                age = now - __import__("datetime").datetime.fromisoformat(
                    started.replace("Z", "+00:00")
                )
            except (AttributeError, ValueError):
                age = timedelta.max
            if row["status"] not in statuses:
                reasons.append(f"status={row['status']}")
            if row["status"] == "RUNNING":
                reasons.append("running")
            if age < minimum_age:
                reasons.append("younger_than_minimum_age")
            if policy.get("protect_pinned", True) and row.get("pinned"):
                reasons.append("pinned")
            if policy.get("protect_benchmarks", True) and row.get("benchmark_id"):
                reasons.append("benchmark")
            if not row.get("artifact_path"):
                reasons.append("no_artifact")
            item = {
                "episode_id": row["episode_id"],
                "status": row["status"],
                "size_bytes": row["size_bytes"],
                "reasons": reasons,
            }
            if reasons:
                retained.append(item)
            else:
                item["reason"] = (
                    f"status {row['status']} matched retention policy and age "
                    f"exceeded {policy['minimum_age_hours']} hours"
                )
                candidates.append(item)
        return {
            "dry_run": True,
            "policy": policy,
            "candidate_count": len(candidates),
            "candidate_bytes": sum(item["size_bytes"] for item in candidates),
            "candidates": candidates,
            "retained": retained,
        }

    def quarantine(self, episode_ids, actor="data-platform", reason="retention-policy"):
        rows = {
            row["episode_id"]: row
            for row in self.registry.list_episodes(limit=100000)
        }
        results = []
        for episode_id in episode_ids:
            row = rows.get(episode_id)
            if not row:
                results.append({"episode_id": episode_id, "status": "NOT_FOUND"})
                continue
            protection = self._protection_reason(row)
            if protection:
                results.append(
                    {
                        "episode_id": episode_id,
                        "status": "RETAINED",
                        "reason": protection,
                    }
                )
                continue
            source = Path(row["artifact_path"])
            quarantine_id = f"{episode_id}__{uuid.uuid4().hex[:8]}"
            target = self.layout.quarantine / quarantine_id
            manifest = {
                "schema_version": "quarantine.v1",
                "quarantine_id": quarantine_id,
                "episode_id": episode_id,
                "source_path": str(source),
                "quarantine_path": str(target),
                "quarantined_at": utc_now().isoformat(),
                "actor": actor,
                "reason": reason,
                "status_before": row["status"],
                "size_bytes": row["size_bytes"],
                "recovery_window_days": self.load_policy()["recovery_window_days"],
            }
            target.mkdir(parents=True, exist_ok=False)
            shutil.move(str(source), str(target / "artifact"))
            (target / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self._preserve_diagnostic(row)
            self._audit("QUARANTINE", manifest)
            results.append(
                {
                    "episode_id": episode_id,
                    "status": "QUARANTINED",
                    "quarantine_id": quarantine_id,
                }
            )
        self.registry.scan()
        return results

    def restore(self, quarantine_id, actor="data-platform"):
        container = self.layout.quarantine / quarantine_id
        manifest = read_json(container / "manifest.json")
        artifact = container / "artifact"
        destination = self.layout.episodes / manifest["episode_id"]
        if destination.exists():
            raise FileExistsError(f"restore target already exists: {destination}")
        shutil.move(str(artifact), str(destination))
        payload = {
            **manifest,
            "restored_at": utc_now().isoformat(),
            "restored_by": actor,
        }
        self._audit("RESTORE", payload)
        shutil.rmtree(container)
        self.registry.scan()
        return payload

    def list_quarantine(self):
        rows = []
        now = utc_now()
        for path in sorted(self.layout.quarantine.glob("*/manifest.json")):
            try:
                row = read_json(path)
            except (OSError, ValueError):
                continue
            quarantined_at = __import__("datetime").datetime.fromisoformat(
                row["quarantined_at"].replace("Z", "+00:00")
            )
            expires = quarantined_at + timedelta(
                days=float(row.get("recovery_window_days", 7))
            )
            row["expires_at"] = expires.isoformat()
            row["expired"] = now >= expires
            rows.append(row)
        return rows

    def purge_expired(self, actor="data-platform", execute=False):
        expired = [row for row in self.list_quarantine() if row["expired"]]
        if not execute:
            return {"dry_run": True, "expired": expired}
        results = []
        for row in expired:
            container = self.layout.quarantine / row["quarantine_id"]
            shutil.rmtree(container)
            payload = {**row, "purged_at": utc_now().isoformat(), "purged_by": actor}
            self._audit("PURGE", payload)
            results.append(payload)
        return {"dry_run": False, "purged": results}

    def pin(self, episode_id, reason, actor="data-platform"):
        path = self.layout.pins / f"{episode_id}.json"
        payload = {
            "episode_id": episode_id,
            "reason": reason,
            "actor": actor,
            "pinned_at": utc_now().isoformat(),
        }
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._audit("PIN", payload)
        self.registry.scan()
        return payload

    def unpin(self, episode_id, actor="data-platform"):
        path = self.layout.pins / f"{episode_id}.json"
        existed = path.exists()
        if existed:
            path.unlink()
        payload = {
            "episode_id": episode_id,
            "actor": actor,
            "unpinned_at": utc_now().isoformat(),
            "existed": existed,
        }
        self._audit("UNPIN", payload)
        self.registry.scan()
        return payload

    def _protection_reason(self, row):
        if row["status"] == "RUNNING":
            return "running episodes cannot be quarantined"
        if row.get("pinned"):
            return "pinned episodes cannot be quarantined"
        if row.get("benchmark_id"):
            return "benchmark episodes cannot be quarantined"
        if not row.get("artifact_path"):
            return "record has no artifact directory"
        return None

    def _preserve_diagnostic(self, row):
        root = self.layout.reports / "diagnostics"
        root.mkdir(parents=True, exist_ok=True)
        source = Path(row["artifact_path"])
        payload = {"registry": row}
        for name in ("metadata.json", "metrics.json"):
            try:
                payload[name.removesuffix(".json")] = read_json(source / name)
            except (OSError, ValueError):
                payload[name.removesuffix(".json")] = None
        (root / f"{row['episode_id']}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _audit(self, action, payload):
        append_jsonl(
            self.layout.audit_log,
            {
                "time": utc_now().isoformat(),
                "action": action,
                **payload,
            },
        )


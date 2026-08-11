# SO-101 v0.1.0 Live Campaign Dashboard

The v0.1.0 collector can publish a campaign without coupling the Dashboard to
an Isaac process. Pass `--campaign-root`, `--campaign-id`, and `--segment-id`
together. The collector then maintains these files in the campaign root:

- `status.json`, atomically replaced on every state change and at least every
  five seconds while the collector is running;
- `events.jsonl`, an append-only, monotonic `farpoint.collection-event.v1` log;
- `active-preview.jpg`, atomically replaced no more than once per second.

An attempt is published before Oracle execution. `attempt_completed` is
published after its durable run state and manifest update, and a successful,
dataset-valid attempt additionally emits `episode_completed`. Stopping a
partial or watchdog-paused invocation preserves all completed events and marks
the live segment `PAUSED`; it never removes selected episodes.

Start the Dashboard with one or more campaign roots:

```bash
python scripts/data_platform_server.py \
  --outputs-root /home/wenyixu/farpoint-data/so101/dashboard \
  --campaign-root /home/wenyixu/farpoint-data/so101/campaigns
```

`FARPOINT_CAMPAIGN_ROOTS` provides the same list using the platform path
separator. The server exposes:

- `GET /api/live-runs`
- `GET /api/collections`
- `GET /api/campaigns/{campaign_id}`
- `GET /api/campaigns/{campaign_id}/segments/{segment_id}`
- `GET /api/campaigns/{campaign_id}/active-preview`
- `GET /api/events` as same-origin server-sent events

A running campaign becomes `STALE` after 60 seconds without a heartbeat.
Collections always remain visible, including paused and failed campaigns.
Only campaigns explicitly marked `campaign_kind=formal` with
`execution_status=FINISHED` and `quality_status=PASS` are projected into the
Benchmarks tab. Older campaign contracts without the optional field remain
readable and are treated as non-formal evidence.

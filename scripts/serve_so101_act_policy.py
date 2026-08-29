#!/usr/bin/env python3
"""Serve one local LeRobot policy checkpoint to an isolated Isaac rollout container."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from farpoint.action_replay import ExpertActionReplay
from farpoint.policy_rollout import resolve_replan_interval
from farpoint.policy_training import file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--replan-interval-steps", type=int)
    parser.add_argument("--replay-manifest", type=Path)
    return parser.parse_args()


def load_policy(checkpoint: Path, replan_interval_steps: int | None):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    config.pretrained_path = str(checkpoint)
    config.device = "cuda"
    checkpoint_steps = int(config.n_action_steps)
    chunk_size = int(config.chunk_size)
    config.n_action_steps = resolve_replan_interval(
        replan_interval_steps,
        checkpoint_steps=checkpoint_steps,
        chunk_size=chunk_size,
    )
    if hasattr(config, "pretrained_backbone_weights"):
        config.pretrained_backbone_weights = None
    policy_class = get_policy_class(config.type)
    policy = policy_class.from_pretrained(
        checkpoint, config=config, local_files_only=True, strict=True
    ).to("cuda")
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config, pretrained_path=str(checkpoint)
    )
    declared_camera_features = {
        name for name in config.input_features if name.startswith("observation.images.")
    }
    camera_features = [
        feature
        for feature in (
            "observation.images.front",
            "observation.images.wrist",
        )
        if feature in declared_camera_features
    ]
    if set(camera_features) != declared_camera_features:
        raise RuntimeError(
            f"policy checkpoint declares unsupported camera features: {declared_camera_features}"
        )
    if not camera_features:
        raise RuntimeError("policy checkpoint declares no camera input features")
    return (
        policy,
        preprocessor,
        postprocessor,
        camera_features,
        {
            "policy_type": config.type,
            "chunk_size": chunk_size,
            "checkpoint_n_action_steps": checkpoint_steps,
            "replan_interval_steps": int(config.n_action_steps),
        },
    )


def reset_components(*components) -> None:
    for component in components:
        if hasattr(component, "reset"):
            component.reset()


def main() -> int:
    args = parse_args()
    model_file = args.checkpoint / "model.safetensors"
    model_sha256 = file_sha256(model_file)
    if model_sha256 != args.expected_model_sha256:
        raise RuntimeError("policy server checkpoint SHA256 mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("policy server requires CUDA")
    import importlib.metadata
    from lerobot.utils.control_utils import predict_action

    replay = ExpertActionReplay(args.replay_manifest) if args.replay_manifest else None
    if replay is None:
        policy, preprocessor, postprocessor, camera_features, execution = load_policy(
            args.checkpoint, args.replan_interval_steps
        )
        reset_components(policy, preprocessor, postprocessor)
    else:
        policy = preprocessor = postprocessor = None
        camera_features = replay.camera_features
        execution = {
            "source": "dataset_replay",
            "replan_interval_steps": args.replan_interval_steps,
            "replay_manifest_sha256": replay.manifest_sha256,
        }
    identity = {
        "status": "ready",
        "model_sha256": model_sha256,
        "lerobot_version": importlib.metadata.version("lerobot"),
        "policy_image_id": os.environ.get("FARPOINT_POLICY_IMAGE_ID", ""),
        "cuda_device": torch.cuda.get_device_name(0),
        "camera_features": camera_features,
        "action_execution": execution,
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *values):
            print("POLICY_SERVER " + format % values, flush=True)

        def response(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            return json.loads(self.rfile.read(length))

        def do_GET(self):
            if self.path == "/health":
                self.response(200, identity)
            else:
                self.response(404, {"error": "not_found"})

        def do_POST(self):
            if self.path == "/reset":
                payload = self.read_json()
                if replay is None:
                    reset_components(policy, preprocessor, postprocessor)
                else:
                    try:
                        replay.reset(str(payload["scene_id"]))
                    except (KeyError, ValueError) as error:
                        self.response(400, {"error": str(error)})
                        return
                self.response(200, {"status": "reset"})
                return
            if self.path != "/action":
                self.response(404, {"error": "not_found"})
                return
            payload = self.read_json()
            state = np.asarray(payload["state"], dtype=np.float32)
            if state.shape != (6,) or not np.all(np.isfinite(state)):
                self.response(400, {"error": "invalid_state"})
                return
            encoded_images = payload.get("images_jpeg")
            if encoded_images is None and "front_jpeg" in payload:
                encoded_images = {"observation.images.front": payload["front_jpeg"]}
            if not isinstance(encoded_images, dict) or set(encoded_images) != set(camera_features):
                self.response(400, {"error": "camera_feature_mismatch"})
                return
            observation = {"observation.state": state}
            for feature in camera_features:
                image_bytes = base64.b64decode(encoded_images[feature], validate=True)
                image = np.asarray(
                    Image.open(io.BytesIO(image_bytes)).convert("RGB"), dtype=np.uint8
                )
                if image.shape != (480, 640, 3):
                    self.response(400, {"error": "invalid_image_shape"})
                    return
                observation[feature] = image
            if replay is None:
                queue_depth_before = len(getattr(policy, "_action_queue", ()))
                action = predict_action(
                    observation,
                    policy,
                    torch.device("cuda"),
                    preprocessor,
                    postprocessor,
                    use_amp=False,
                    task=payload.get("task"),
                    robot_type="so101",
                )
                values = action.detach().cpu().numpy().reshape(-1)
                action_execution = {
                    "source": f"{policy.config.type}_policy",
                    "inference_refreshed": queue_depth_before == 0,
                    "queue_depth_before": queue_depth_before,
                    "queue_depth_after": len(getattr(policy, "_action_queue", ())),
                }
            else:
                values, action_execution = replay.next_action()
            if values.shape != (6,) or not np.all(np.isfinite(values)):
                self.response(500, {"error": "invalid_policy_action"})
                return
            self.response(
                200,
                {
                    "action": values.tolist(),
                    "execution": action_execution,
                },
            )

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(json.dumps(identity, sort_keys=True), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

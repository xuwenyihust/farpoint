#!/usr/bin/env python3
"""Serve one local ACT checkpoint to an isolated Isaac rollout container."""

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

from farpoint.policy_training import file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--expected-model-sha256", required=True)
    return parser.parse_args()


def load_policy(checkpoint: Path):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    config.pretrained_path = str(checkpoint)
    config.device = "cuda"
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
    return policy, preprocessor, postprocessor


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

    policy, preprocessor, postprocessor = load_policy(args.checkpoint)
    reset_components(policy, preprocessor, postprocessor)
    identity = {
        "status": "ready",
        "model_sha256": model_sha256,
        "lerobot_version": importlib.metadata.version("lerobot"),
        "policy_image_id": os.environ.get("FARPOINT_POLICY_IMAGE_ID", ""),
        "cuda_device": torch.cuda.get_device_name(0),
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
                reset_components(policy, preprocessor, postprocessor)
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
            image_bytes = base64.b64decode(payload["front_jpeg"], validate=True)
            front = np.asarray(Image.open(io.BytesIO(image_bytes)).convert("RGB"), dtype=np.uint8)
            if front.shape != (480, 640, 3):
                self.response(400, {"error": "invalid_image_shape"})
                return
            action = predict_action(
                {
                    "observation.state": state,
                    "observation.images.front": front,
                },
                policy,
                torch.device("cuda"),
                preprocessor,
                postprocessor,
                use_amp=False,
                task=payload.get("task"),
                robot_type="so101",
            )
            values = action.detach().cpu().numpy().reshape(-1)
            if values.shape != (6,) or not np.all(np.isfinite(values)):
                self.response(500, {"error": "invalid_policy_action"})
                return
            self.response(200, {"action": values.tolist()})

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

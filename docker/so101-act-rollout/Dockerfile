ARG BASE_IMAGE=farpoint-so101-isaaclab:3.0-beta2
FROM ${BASE_IMAGE}

ARG LEROBOT_VERSION=0.4.4

RUN /isaac-sim/python.sh -c "import importlib.metadata as m, pathlib; pathlib.Path('/opt/farpoint-rollout-constraints.txt').write_text('torch==' + m.version('torch') + '\ntorchvision==' + m.version('torchvision') + '\n')" \
    && /isaac-sim/python.sh -m pip install --no-cache-dir --pre \
      --constraint /opt/farpoint-rollout-constraints.txt \
      "lerobot==${LEROBOT_VERSION}" \
    && /isaac-sim/python.sh -m pip install --no-cache-dir --ignore-installed "pillow==12.1.1" \
    && /isaac-sim/python.sh -c "import av, importlib.metadata as m, PIL, torch; assert m.version('lerobot') == '${LEROBOT_VERSION}'; assert m.version('pillow') == '12.1.1'; assert '/isaac-sim/kit/python/lib/' in PIL.__file__; assert torch.version.cuda is not None; av.codec.Codec('libx264', 'w')" \
    && /isaac-sim/python.sh -m pip freeze > /opt/farpoint-rollout-python-freeze.txt

LABEL org.opencontainers.image.title="Farpoint SO-101 ACT rollout" \
      org.opencontainers.image.description="Isaac Lab 3.0 plus pinned LeRobot ACT inference" \
      org.opencontainers.image.licenses="Apache-2.0"

FROM nvcr.io/nvidia/pytorch:26.01-py3@sha256:38ed2ecb2c16d10677006d73fb0a150855d6ec81db8fc66e800b5ae92741007e

ARG LEROBOT_VERSION=0.4.4

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

RUN python -c "import importlib.metadata as m, pathlib; pathlib.Path('/opt/farpoint-training-constraints.txt').write_text('torch==' + m.version('torch') + '\\ntorchvision==' + m.version('torchvision') + '\\n')" \
    && python -m pip install --no-cache-dir --pre \
      --constraint /opt/farpoint-training-constraints.txt \
      "lerobot==${LEROBOT_VERSION}" \
    && python -c "import av, importlib.metadata as m, torch; from packaging.version import Version; assert m.version('lerobot') == '${LEROBOT_VERSION}'; assert Version(m.version('torch')) < Version('2.11.0'); assert torch.version.cuda is not None; av.codec.Codec('av1', 'r')" \
    && python -m pip freeze > /opt/farpoint-training-python-freeze.txt

LABEL org.opencontainers.image.title="Farpoint SO-101 LeRobot training" \
      org.opencontainers.image.description="Pinned LeRobot ACT training environment for DGX Spark" \
      org.opencontainers.image.licenses="Apache-2.0"

WORKDIR /workspace/project

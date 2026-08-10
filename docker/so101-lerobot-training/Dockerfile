FROM nvcr.io/nvidia/pytorch:26.03-py3

ARG LEROBOT_VERSION=0.4.4

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

COPY docker/so101-lerobot-training/constraints.txt /opt/farpoint-training-constraints.txt

RUN python -m pip install --no-cache-dir --pre \
      --constraint /opt/farpoint-training-constraints.txt \
      "lerobot==${LEROBOT_VERSION}" \
    && python -c "import av, importlib.metadata as m, torch; assert m.version('lerobot') == '${LEROBOT_VERSION}'; assert torch.version.cuda is not None; av.codec.Codec('av1', 'r')" \
    && python -m pip freeze > /opt/farpoint-training-python-freeze.txt

LABEL org.opencontainers.image.title="Farpoint SO-101 LeRobot training" \
      org.opencontainers.image.description="Pinned LeRobot ACT training environment for DGX Spark" \
      org.opencontainers.image.licenses="Apache-2.0"

WORKDIR /workspace/project

FROM nvcr.io/nvidia/isaac-sim:6.0.0

ARG ISAAC_LAB_REF=release/3.0.0-beta2
ENV ACCEPT_EULA=Y PRIVACY_CONSENT=Y PYTHONUNBUFFERED=1

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    git python3.12-dev libgl1-mesa-dev libx11-dev libxcursor-dev \
    libxi-dev libxinerama-dev libxrandr-dev ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
RUN git clone --depth 1 --branch ${ISAAC_LAB_REF} https://github.com/isaac-sim/IsaacLab.git
WORKDIR /workspace/IsaacLab
RUN ln -s /isaac-sim _isaac_sim && ./isaaclab.sh --install

WORKDIR /workspace/project
ENTRYPOINT ["/bin/bash", "-lc"]

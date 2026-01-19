#!/bin/bash
set -euo pipefail

CARLA_IMAGE="carlasim/carla:0.9.16"
docker pull "${CARLA_IMAGE}"

# 前台交互(-it) 或后台(-d)运行，可通过设置环境变量 DETACH 控制
EXTRA_ARGS="-it"
if [[ "${DETACH:-}" ]]; then
  EXTRA_ARGS="-d"
fi

# 如果用 sudo 调用，容器内仍然使用原始用户而非 root
HOST_UID=$(id -u)
HOST_GID=$(id -g)
if [[ "${HOST_UID}" -eq 0 && -n "${SUDO_UID:-}" ]]; then
  HOST_UID=${SUDO_UID}
  HOST_GID=${SUDO_GID:-${HOST_GID}}
fi

# -RenderOffScreen 可视需要添加到最后以无窗口渲染
docker run ${EXTRA_ARGS} \
  --runtime=nvidia \
  --net=host \
  --user="${HOST_UID}:${HOST_GID}" \
  --env="DISPLAY=${DISPLAY:-}" \
  --env=NVIDIA_VISIBLE_DEVICES=all \
  --env=NVIDIA_DRIVER_CAPABILITIES=all \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  "${CARLA_IMAGE}" bash CarlaUE4.sh -nosound

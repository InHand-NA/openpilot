#!/bin/bash
# 清理已停止的 Carla server 容器
# 用法: sudo bash cleanup_carla_containers.sh [--dry-run]

set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
  echo "[DRY RUN] 仅列出，不删除"
fi

# 查找已停止的 carla 相关容器（镜像名或容器名包含 carla）
STOPPED=$(docker ps -a --filter "status=exited" --filter "status=dead" --filter "status=created" \
  --format "{{.ID}}\t{{.Image}}\t{{.Names}}\t{{.Status}}" | grep -i carla || true)

if [[ -z "$STOPPED" ]]; then
  echo "没有找到已停止的 Carla 容器。"
  exit 0
fi

echo "找到以下已停止的 Carla 容器："
echo "---"
echo -e "ID\tIMAGE\tNAME\tSTATUS"
echo "$STOPPED"
echo "---"

COUNT=$(echo "$STOPPED" | wc -l)
echo "共 ${COUNT} 个容器"

if $DRY_RUN; then
  echo "[DRY RUN] 结束，未删除任何容器。"
  exit 0
fi

IDS=$(echo "$STOPPED" | awk '{print $1}')
echo "正在删除..."
echo "$IDS" | xargs docker rm -v
echo "已删除 ${COUNT} 个 Carla 容器。"

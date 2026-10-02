#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# NeuroSight AI — EC2 deploy script (run on the EC2 instance)
#  1. Installs Docker + Compose if missing (Amazon Linux & Ubuntu)
#  2. Logs into Docker Hub
#  3. Pulls the pre-built images and starts the stack
#
# Usage on EC2:
#   chmod +x deploy.sh
#   ./deploy.sh
#
# First run (one-time preparation):
#   cp .env.example .env && nano .env   # fill in AWS keys + JWT secret
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

cd "$(dirname "$0")"

DOCKER_USER="${DOCKER_USER:-madhusudhanchilumula}"
TAG="${TAG:-latest}"

echo "==> [1/4] Checking / installing Docker..."
if ! command -v docker >/dev/null 2>&1; then
  echo "    Docker not found. Installing..."
  if [ -f /etc/os-release ] && grep -qi "amzn" /etc/os-release; then
    sudo yum install -y docker
    sudo systemctl enable --now docker
  else
    curl -fsSL https://get.docker.com | sh
    sudo systemctl enable --now docker
  fi
  sudo usermod -aG docker "${USER}" || true
  echo "    Installed. (Re-login may be needed for docker group; run `newgrp docker` if access denied)"
else
  echo "    Docker $(docker --version | awk '{print $3}') found."
fi

echo "==> [2/4] Checking / installing Docker Compose..."
if ! command -v docker-compose >/dev/null 2>&1 && ! docker compose version >/dev/null 2>&1; then
  echo "    Installing docker compose plugin..."
  sudo mkdir -p /usr/local/lib/docker/cli-plugins
  sudo curl -SL "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-$(uname -m)" \
       -o /usr/local/lib/docker/cli-plugins/docker-compose
  sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
fi

if [ ! -f .env ]; then
  echo "!! Missing .env — create it first:"
  echo "   cp .env.example .env && nano .env"
  exit 1
fi

echo "==> [3/4] Logging into Docker Hub..."
docker login --username "${DOCKER_USER}"

echo "==> [4/4] Pulling images and starting stack..."
COMPOSE="docker compose"
if ! docker compose version >/dev/null 2>&1; then
  COMPOSE="docker-compose"
fi

$COMPOSE -f docker-compose.prod.yml --env-file .env pull
$COMPOSE -f docker-compose.prod.yml --env-file .env up -d

echo ""
echo "  Frontend:  http://$(hostname -I | awk '{print $1}')     (port 80)"
echo "  Backend :  http://$(hostname -I | awk '{print $1}'):5000 (health: /api/health)"
echo ""
echo "  Useful commands:"
echo "    docker compose -f docker-compose.prod.yml --env-file .env logs -f backend"
echo "    docker compose -f docker-compose.prod.yml --env-file .env down"
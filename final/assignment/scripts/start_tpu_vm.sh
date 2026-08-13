#!/usr/bin/env bash
set -euo pipefail

# Follows Stanford's working student-VM setup:
# https://github.com/stanfordhpccenter/me344/blob/master/googlecloud/setting-up-student-vm.md

ENV_FILE="$HOME/.me344.env"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "Run the Google Cloud setup block in the README first." >&2
  exit 1
fi
source "$ENV_FILE"

: "${PROJECT_ID:?Missing PROJECT_ID in $ENV_FILE}"
: "${ZONE:?Missing ZONE in $ENV_FILE}"

HOSTNUM=$(hostname -s | grep -oE '[0-9]+$' || true)
if [[ -z "$HOSTNUM" ]]; then
  echo "Could not find the student number at the end of hostname '$(hostname -s)'." >&2
  exit 1
fi

TPU_USER="student${HOSTNUM}"
TPU_NAME="tpu-${TPU_USER}"

if [[ -f "$HOME/.ssh/google_compute_engine" && -f "$HOME/.ssh/google_compute_engine.pub" ]]; then
  SSH_KEY="$HOME/.ssh/google_compute_engine"
elif [[ -f "$HOME/.ssh/id_ed25519" && -f "$HOME/.ssh/id_ed25519.pub" ]]; then
  SSH_KEY="$HOME/.ssh/id_ed25519"
else
  SSH_KEY="$HOME/.ssh/google_compute_engine"
  echo "Creating an SSH key for the TPU VM..."
  ssh-keygen -t ed25519 -f "$SSH_KEY" -N ""
fi
PUBKEY=$(<"${SSH_KEY}.pub")
printf -v PUBKEY_Q '%q' "$PUBKEY"

STARTUP_SCRIPT=$(mktemp /tmp/me344-tpu-startup.XXXXXX)
ENV_UPDATE=$(mktemp /tmp/me344-env.XXXXXX)
cleanup() {
  find "$STARTUP_SCRIPT" "$ENV_UPDATE" -delete 2>/dev/null || true
}
trap cleanup EXIT

cat > "$STARTUP_SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail
id -u "$TPU_USER" >/dev/null 2>&1 || useradd -m -s /bin/bash "$TPU_USER"
install -d -m 700 -o "$TPU_USER" -g "$TPU_USER" "/home/$TPU_USER/.ssh"
printf '%s\n' $PUBKEY_Q > "/home/$TPU_USER/.ssh/authorized_keys"
chmod 600 "/home/$TPU_USER/.ssh/authorized_keys"
chown "$TPU_USER:$TPU_USER" "/home/$TPU_USER/.ssh/authorized_keys"
printf '%s\n' '$TPU_USER ALL=(ALL) NOPASSWD:ALL' > "/etc/sudoers.d/$TPU_USER"
chmod 440 "/etc/sudoers.d/$TPU_USER"
printf '%s\n' always > /sys/kernel/mm/transparent_hugepage/enabled
EOF

if gcloud compute tpus tpu-vm describe "$TPU_NAME" \
  --project="$PROJECT_ID" --zone="$ZONE" >/dev/null 2>&1; then
  STATE=$(gcloud compute tpus tpu-vm describe "$TPU_NAME" \
    --project="$PROJECT_ID" --zone="$ZONE" --format='value(state)')
  if [[ "$STATE" == "STOPPED" ]]; then
    echo "Starting $TPU_NAME..."
    gcloud compute tpus tpu-vm update "$TPU_NAME" \
      --project="$PROJECT_ID" --zone="$ZONE" \
      --metadata-from-file="startup-script=$STARTUP_SCRIPT"
    gcloud compute tpus tpu-vm start "$TPU_NAME" \
      --project="$PROJECT_ID" --zone="$ZONE"
  else
    echo "Using existing TPU VM $TPU_NAME in state $STATE."
  fi
else
  echo "Creating TPU VM $TPU_NAME for $TPU_USER..."
  gcloud compute tpus tpu-vm create "$TPU_NAME" \
    --project="$PROJECT_ID" \
    --zone="$ZONE" \
    --accelerator-type=v5litepod-8 \
    --version=tpu-ubuntu2204-base \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --metadata-from-file="startup-script=$STARTUP_SCRIPT"
fi

echo "Waiting for the TPU VM and its startup script..."
TPU_IP=""
for _ in $(seq 1 60); do
  STATE=$(gcloud compute tpus tpu-vm describe "$TPU_NAME" \
    --project="$PROJECT_ID" --zone="$ZONE" --format='value(state)' 2>/dev/null || true)
  TPU_IP=$(gcloud compute tpus tpu-vm describe "$TPU_NAME" \
    --project="$PROJECT_ID" --zone="$ZONE" \
    --format='value(networkEndpoints[0].accessConfig.externalIp)' 2>/dev/null || true)
  if [[ "$STATE" == "READY" && -n "$TPU_IP" ]] && \
    ssh -i "$SSH_KEY" -o BatchMode=yes -o ConnectTimeout=5 \
      -o StrictHostKeyChecking=accept-new "$TPU_USER@$TPU_IP" true 2>/dev/null; then
    break
  fi
  sleep 5
done

if [[ "$STATE" != "READY" || -z "$TPU_IP" ]]; then
  echo "$TPU_NAME did not become ready. Check it with gcloud before retrying." >&2
  exit 1
fi
if ! ssh -i "$SSH_KEY" -o BatchMode=yes -o ConnectTimeout=10 \
  -o StrictHostKeyChecking=accept-new "$TPU_USER@$TPU_IP" true; then
  echo "The TPU VM is ready, but its student SSH account is not. Wait one minute and rerun this script." >&2
  exit 1
fi

awk '!/^export (TPU_USER|TPU_NAME|TPU_IP|TPU_SSH_KEY|ME344_TPU_NAME)=/' \
  "$ENV_FILE" > "$ENV_UPDATE"
{
  printf 'export TPU_USER=%q\n' "$TPU_USER"
  printf 'export TPU_NAME=%q\n' "$TPU_NAME"
  printf 'export TPU_IP=%q\n' "$TPU_IP"
  printf 'export TPU_SSH_KEY=%q\n' "$SSH_KEY"
  printf 'export ME344_TPU_NAME=%q\n' "$TPU_NAME"
} >> "$ENV_UPDATE"
mv "$ENV_UPDATE" "$ENV_FILE"

scp -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new \
  "$ENV_FILE" "$TPU_USER@$TPU_IP:~/.me344.env"

cleanup
trap - EXIT
echo "Connecting to $TPU_NAME. Keep this terminal open while Jupyter runs."
exec ssh -tt -i "$SSH_KEY" \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=6 \
  -L 8888:localhost:8888 \
  -L 8000:localhost:8000 \
  -L 7860:localhost:7860 \
  -L 6006:localhost:6006 \
  "$TPU_USER@$TPU_IP"

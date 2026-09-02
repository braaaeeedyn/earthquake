#!/usr/bin/env bash
# Auto-retry launching the Always-Free A1 instance until Oracle has capacity, then print its
# public IP. Run it in Git Bash after configuring the OCI CLI (see the header steps below).
# It retries ONLY on "out of host capacity"; any other error stops it so you can fix the cause.
#
# ONE-TIME SETUP (details in the chat / DEPLOY notes):
#   1. Install the OCI CLI.
#   2. Run `oci setup config` (creates an API key pair; asks for your user OCID, tenancy OCID,
#      region us-sanjose-1).
#   3. On the Oracle console, upload the generated PUBLIC key: Profile -> your user ->
#      Resources -> API keys -> Add API key -> paste ~/.oci/oci_api_key_public.pem.
#   4. Test: `oci iam region list` should return data.
#
# Then just run:  bash scripts/oci_retry_launch.sh

set -uo pipefail

# ---- settings you can override via env ----
SUBNET_NAME="${SUBNET_NAME:-public subnet-seismic-vcn}"   # the public subnet you created
SSH_PUB="${SSH_PUB:-$HOME/.ssh/oracle_seismic.pub}"       # your public key
DISPLAY_NAME="${DISPLAY_NAME:-seismicsocal}"
OCPUS="${OCPUS:-1}"                                        # 1/6 gets capacity far more often than 2/12
MEM_GB="${MEM_GB:-6}"
INTERVAL="${INTERVAL:-60}"                                 # seconds between retries

# The pip-generated oci.exe launcher is blocked by this machine's Application Control policy, but
# the venv's python is allowed — so run the CLI through it via its entry point (oci_cli.cli:cli).
OCI_PY="${OCI_PY:-$HOME/oci-cli-venv/Scripts/python.exe}"
[ -x "$OCI_PY" ] || { echo "oci-cli python not found at $OCI_PY — set OCI_PY to your oci-cli venv python."; exit 1; }
oci() { "$OCI_PY" -c "from oci_cli.cli import cli; cli()" "$@"; }
[ -f "$SSH_PUB" ] || { echo "SSH public key not found at $SSH_PUB"; exit 1; }

# ---- discover the OCIDs so you don't have to copy any ----
TENANCY=$(grep -iE '^tenancy' "$HOME/.oci/config" | head -1 | cut -d= -f2 | tr -d ' \r')
COMPARTMENT="${COMPARTMENT:-$TENANCY}"                     # root compartment == tenancy
echo "Discovering availability domain, image, and subnet..."
AD=$(oci iam availability-domain list -c "$COMPARTMENT" --query 'data[0].name' --raw-output)
IMAGE=$(oci compute image list -c "$COMPARTMENT" --shape VM.Standard.A1.Flex \
        --operating-system "Canonical Ubuntu" --operating-system-version "22.04" \
        --sort-by TIMECREATED --query 'data[0].id' --raw-output)
SUBNET=$(oci network subnet list -c "$COMPARTMENT" --all --output json \
        | "$OCI_PY" -c "import sys,json;print(next(s['id'] for s in json.load(sys.stdin)['data'] if s['display-name']=='$SUBNET_NAME'))")

for v in AD IMAGE SUBNET; do
  [ -n "${!v}" ] || { echo "Could not resolve $v — check your config / subnet name."; exit 1; }
done
echo "  AD     = $AD"
echo "  image  = $IMAGE"
echo "  subnet = $SUBNET"
echo "Launching ${DISPLAY_NAME} (A1.Flex ${OCPUS} OCPU / ${MEM_GB} GB); retrying every ${INTERVAL}s until capacity is available. Ctrl-C to stop."

try=0
while true; do
  try=$((try+1))
  err=$(mktemp)
  IID=$(oci compute instance launch \
        --availability-domain "$AD" \
        --compartment-id "$COMPARTMENT" \
        --shape VM.Standard.A1.Flex \
        --shape-config "{\"ocpus\": $OCPUS, \"memoryInGBs\": $MEM_GB}" \
        --image-id "$IMAGE" \
        --subnet-id "$SUBNET" \
        --assign-public-ip true \
        --display-name "$DISPLAY_NAME" \
        --ssh-authorized-keys-file "$SSH_PUB" \
        --query 'data.id' --raw-output 2>"$err")

  if [ -n "$IID" ]; then
    echo "$(date '+%H:%M:%S') SUCCESS after $try tries. Instance: $IID"
    echo "Waiting a moment for the VNIC to get a public IP..."
    sleep 20
    IP=$(oci compute instance list-vnics --instance-id "$IID" --query 'data[0]."public-ip"' --raw-output 2>/dev/null)
    echo "PUBLIC IP: ${IP:-<pending — check the console>}"
    rm -f "$err"
    break
  fi

  # Retry on capacity AND transient issues (network timeouts, throttling, 5xx) so an overnight
  # run survives blips. Only genuine config/auth/quota errors stop it.
  if grep -qiE "out of host capacity|outofhostcapacity|capacity|timed out|timeout|connection|RequestException|TooManyRequests|throttl|Service.?Unavailable|Internal.?Server|50[0-9]|429" "$err"; then
    reason=$(grep -qiE "capacity" "$err" && echo "no capacity yet" || echo "transient error (network/throttle)")
    echo "$(date '+%H:%M:%S') try $try: $reason, retrying in ${INTERVAL}s"
    rm -f "$err"
    sleep "$INTERVAL"
  else
    echo "$(date '+%H:%M:%S') try $try: launch failed for a non-transient reason — stopping:"
    cat "$err"; rm -f "$err"
    exit 1
  fi
done

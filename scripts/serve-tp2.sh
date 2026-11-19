#!/usr/bin/env bash
# serve-tp2.sh — EXPERIMENTAL: run Qwen3.8-Flash-Next TP=2 across two DGX Sparks.
#
# Primarily written by an AI agent (from issue #52, the x00byte and capetron
# dual-Spark recipes, MiaAI-Lab dual-Spark #73, the NVIDIA playbooks and the
# NCCL env docs) and NOT validated on a live
# pair; docs/MULTI-NODE-TP2.md carries the failure table and acceptance
# checks. flash/serve.sh stay single-node on purpose: a bridged, port-mapped
# container cannot join a multi-node mp group (cross-node NCCL and the torch
# rendezvous need the hosts' real fabric IPs, so the containers use
# --network host).
#
# Run on the HEAD node:
#   scripts/serve-tp2.sh [profile]        (default profile: default-tp2)
# PEER/MASTER_ADDR are set in the profile's fabric block; environment values
# still override, for ad-hoc runs.
#
# Both nodes need: this checkout at the same commit, the same image (ship it:
# docker save | zstd | ssh worker 'zstd -d | docker load'), the whole
# ~/.cache/huggingface rsynced across, and passwordless sudo for drop_caches.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:-default-tp2}"
PROF_FILE="$HERE/profiles/${PROFILE}.env"
[ -f "$PROF_FILE" ] || { echo "!! no profile $PROF_FILE — see: ./flash profiles"; exit 1; }
# Same semantics as flash's load_profile: variables ALREADY in the environment
# win; the profile only fills in what is unset (so `MASTER_ADDR=... scripts/
# serve-tp2.sh` still works for ad-hoc runs). No inline comments in profiles.
while IFS= read -r line || [ -n "$line" ]; do
  case "$line" in ''|'#'*) continue ;; esac
  key="${line%%=*}"; val="${line#*=}"; val="${val%\"}"; val="${val#\"}"
  if [ -z "${!key+x}" ]; then export "$key=$val"; fi
done < "$PROF_FILE"

TP_SIZE="${TP_SIZE:-2}"; [ "$TP_SIZE" = 2 ] || { echo "!! this launcher knows TP_SIZE=2 only (two Sparks, one GB10 each)"; exit 1; }
case "${MODE:-nvfp4}" in
  nvfp4) ;;
  hybrid|hybrid-mtp)
    echo "!! MODE=$MODE does not load at TP>1: mlp.shared_expert.* is 640 wide -> 320 per rank,"
    echo "   not divisible by the fp8 block_n=128 (issue #52). Use MODE=nvfp4, or the carveout"
    echo "   in docs/MULTI-NODE-TP2.md section 7.1."
    exit 1 ;;
esac
[ "${DRAFT_VOCAB:-1}" = 0 ] || { echo "!! DRAFT_VOCAB crashes at TP>1 (global ids vs the row-sharded lm_head, issue #52) — set DRAFT_VOCAB=0"; exit 1; }
[ "${PLE_MMAP:-0}" = 0 ] || { echo "!! the PLE mmap hook is untested under TP — set PLE_MMAP=0 (the table fits resident on a pair)"; exit 1; }
[ "${YARN:-0}" = 0 ] || echo "!! note: YaRN beyond native CTX has never been verified on TP — expect to re-doctor this"
ROCE_SUBNET="${ROCE_SUBNET:?ROCE_SUBNET=fabric subnet CIDR, the one shared by both RoCE interfaces}"
if [ -n "${NCCL_IB_GID_INDEX:-}" ]; then
  echo "!! note: NCCL_IB_GID_INDEX=$NCCL_IB_GID_INDEX is IGNORED by this lane."
  echo "   Pinned GID indices drift when one node reboots alone (MiaAI-Lab dual-Spark #73);"
  echo "   NCCL resolves the RoCEv2 GID per rank from NCCL_IB_ADDR_RANGE=$ROCE_SUBNET instead."
fi

PEER="${PEER:?set PEER=user@<worker-fabric-ip> in the profile fabric block}"
MASTER_ADDR="${MASTER_ADDR:?set MASTER_ADDR=<head fabric ip> in the profile fabric block}"
NAME="${NAME:-qwen38-flash-tp2}"; IMAGE="${IMAGE:-qwen38-flash-dgx:v0.33}"
MODEL="${MODEL:-nvidia/Qwen3.8-Flash-Next-NVFP4}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8-flash-next}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
PORT="${PORT:-18300}"; MASTER_PORT="${MASTER_PORT:-29500}"

# Same snapshot resolution as serve.sh (refs/main first; the whole cache is
# mounted because snapshot entries are RELATIVE symlinks into ../../blobs).
REPO_DIR="$HF_CACHE/hub/models--${MODEL//\//--}"
SNAP=""
for REF in main master; do
  REV="$(cat "$REPO_DIR/refs/$REF" 2>/dev/null || true)"
  [ -n "$REV" ] && [ -d "$REPO_DIR/snapshots/$REV" ] && { SNAP="$REPO_DIR/snapshots/$REV"; break; }
done
[ -n "$SNAP" ] || SNAP="$(ls -dt "$REPO_DIR"/snapshots/*/ 2>/dev/null | grep -v -- '-fp8hybrid' | head -1 | sed 's:/$::' || true)"
[ -n "$SNAP" ] || { echo "!! checkpoint not found under $REPO_DIR — ./flash setup first, then rsync the whole cache to $PEER"; exit 1; }
SNAP_IN="/hf/hub/models--${MODEL//\//--}/snapshots/$(basename "$SNAP")"

# Environment parity is a correctness property here: a drifted image on the
# worker shows up as the NCCL bootstrap error "Message truncated".
have_image() { docker image inspect "$IMAGE" >/dev/null 2>&1; }
have_image || { echo "!! $IMAGE not built here — ./flash setup"; exit 1; }
ssh "$PEER" "docker image inspect $IMAGE >/dev/null 2>&1" \
  || { echo "!! $IMAGE not on $PEER — ship it: docker save $IMAGE | zstd -T0 -3 | ssh $PEER 'zstd -d | docker load'"; exit 1; }
ssh "$PEER" "test -d '$REPO_DIR/snapshots/$(basename "$SNAP")'" \
  || { echo "!! snapshot not on $PEER at $REPO_DIR — rsync the whole $HF_CACHE across"; exit 1; }

# RoCEv2 GID indices are NOT stable: a one-sided reboot or an address
# add/remove leaves holes in a port's GID table and displaces the RoCE v2 IPv4
# entry (MiaAI-Lab dual-Spark issue #73). So this launcher never sets
# NCCL_IB_GID_INDEX — NCCL (>=2.21) resolves the v2 GID per rank from
# NCCL_IB_ADDR_FAMILY/ADDR_RANGE/ROCE_VERSION_NUM — but both nodes' tables are
# preflighted here so an out-of-range or empty match fails in one second with a
# named node, instead of a crash loop of rendezvous timeouts. $3 lets tests
# point at a fake sysfs tree.
GID_PROBE='hcas=$1; subnet=$2; root=${3:-/sys/class/infiniband}
ip=${subnet%%/*}; mask=${subnet##*/}; shift=$((32 - mask))
IFS=. read -r a b c d <<X
$ip
X
target=$((( (a<<24) + (b<<16) + (c<<8) + d ) >> shift))
found=""; fail=0
OLDIFS=$IFS; IFS=,
for hca in $hcas; do
  hit=""
  i=0
  while [ "$i" -le 31 ]; do
    t=$(cat "$root/$hca/ports/1/gid_attrs/types/$i" 2>/dev/null)
    if [ "$t" = "RoCE v2" ]; then
      g=$(cat "$root/$hca/ports/1/gids/$i" 2>/dev/null)
      case $g in
        0000:0000:0000:0000:0000:ffff:*)
          pair=${g##*ffff:}
          n=$((( 0x${pair%%:*} << 16 ) + 0x${pair##*:}))
          if [ $(( n >> shift )) -eq "$target" ]; then hit="$hca:$i"; break; fi
          ;;
      esac
    fi
    i=$((i + 1))
  done
  if [ -z "$hit" ]; then echo "no RoCE v2 IPv4 GID inside $subnet on $hca" >&2; fail=1; fi
  found="$found $hit"
done
IFS=$OLDIFS
[ $fail -eq 0 ] || exit 1
echo $found'
gid_check() {  # $1 = ssh target ('' = this node)
  local prefix=""
  [ -n "$1" ] && prefix="ssh $1 "
  ${prefix}sh -s -- "$NCCL_IB_HCA" "$ROCE_SUBNET" "${GID_ROOT:-}" <<< "$GID_PROBE"
}
echo ">> RoCEv2 GID preflight (want a v2 IPv4 GID inside $ROCE_SUBNET on: $NCCL_IB_HCA)"
HEAD_GIDS="$(gid_check "" )" \
  || { echo "!! head has no usable RoCE v2 GID — run: show_gids   (rebooting both nodes together repacks the tables)"; exit 1; }
WORKER_GIDS="$(gid_check "$PEER" )" \
  || { echo "!! worker has no usable RoCE v2 GID — run: ssh $PEER show_gids   (or reboot both nodes together)"; exit 1; }
echo "   head:$HEAD_GIDS   worker:$WORKER_GIDS   (indices may differ — that is fine, nothing pins them)"

EAGER_ARGS=(--enforce-eager); [ "${EAGER:-1}" = 1 ] || EAGER_ARGS=(-cc.cudagraph_mode=PIECEWISE)
# (piecewise note: serve.sh's splitting_ops list is not needed here — with the
#  mmap hook off the PLE custom op doesn't exist and vLLM's default attention-op
#  list applies.)
SPEC="\"method\":\"mtp\",\"num_speculative_tokens\":${MTP:-4}"
[ "${YARN:-0}" != 0 ] && SPEC="$SPEC,\"max_model_len\":${CTX}"

launch() {  # $1 = rank, $2 = ssh target ('' = this node)
  local rank="$1" host="$2"
  local docker_args=(run -d --name "$NAME" --restart unless-stopped --gpus all --ipc=host
    --network host --device=/dev/infiniband
    --ulimit memlock=-1:-1 --ulimit nofile=1048576:1048576 --cap-add IPC_LOCK
    -v "$HF_CACHE:/hf" -e HF_HOME=/hf -e HF_HUB_OFFLINE=1
    -e VLLM_PLE_MMAP=0 -e VLLM_USE_FLASHINFER_SAMPLER=1
    -e VLLM_QSA_DET_TOPK=1 -e VLLM_QSA_DET_LIB=/opt/llm/kernel-det/_C_det.so
    -e NCCL_NET=IB -e NCCL_IB_DISABLE=0
    -e "NCCL_IB_HCA=$NCCL_IB_HCA"
    -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET -e "NCCL_IB_ADDR_RANGE=$ROCE_SUBNET"
    -e "NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME" -e "GLOO_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME" -e "TP_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
    -e NCCL_NVLS_ENABLE=0 -e NCCL_CROSS_NIC=0 -e NCCL_IB_MERGE_NICS=0 -e NCCL_CUMEM_ENABLE=0
    -e NCCL_IGNORE_CPU_AFFINITY=1 -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 -e NCCL_DEBUG=WARN)
  local vllm_args=("$SNAP_IN" --served-model-name "$SERVED_MODEL_NAME" --host 0.0.0.0 --port "$PORT"
    --load-format safetensors --tensor-parallel-size 2 --nnodes 2 --node-rank "$rank"
    --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" --distributed-executor-backend mp
    --max-model-len "$CTX" --max-num-seqs "$SEQS" --gpu-memory-utilization "$GPU_MEM"
    --enable-chunked-prefill --max-num-batched-tokens "$BATCH_TOKENS"
    --kv-cache-dtype "$KV_DTYPE"
    --no-enable-flashinfer-autotune --enable-auto-tool-choice
    --tool-call-parser qwen3_coder --reasoning-parser qwen3
    --speculative-config "{$SPEC}" "${EAGER_ARGS[@]}")
  [ -n "${MAMBA_SSM_DTYPE:-}" ] && vllm_args+=(--mamba-ssm-cache-dtype "$MAMBA_SSM_DTYPE")
  [ "${PREFIX_CACHE:-1}" = 1 ] && vllm_args+=(--enable-prefix-caching) || vllm_args+=(--no-enable-prefix-caching)
  [ "$rank" != 0 ] && vllm_args+=(--headless)   # without it: "collective_rpc should not be called on follower node"
  if [ -z "$host" ]; then docker "${docker_args[@]}" "$IMAGE" "${vllm_args[@]}" ${EXTRA:-}
  else  # re-quote for the remote shell: vLLM's JSON braces would brace-expand remotely
    REMOTE="docker $(printf '%q ' "${docker_args[@]}") $IMAGE $(printf '%q ' "${vllm_args[@]}")"
    [ -n "${EXTRA:-}" ] && REMOTE+=" $EXTRA"
    ssh "$host" "$REMOTE"
  fi
}

echo ">> $NAME: TP=2 (MODE=$MODE ctx=$CTX seqs=$SEQS kv=$KV_DTYPE mtp=${MTP:-4} eager=${EAGER:-1} prefix_cache=${PREFIX_CACHE:-1}) head=$MASTER_ADDR:$MASTER_PORT peer=$PEER"
echo ">> dropping page caches on both nodes (warm cache starves the GB10 allocator mid-load)"
echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
ssh "$PEER" 'echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null'
docker rm -f "$NAME" >/dev/null 2>&1 || true
ssh "$PEER" "docker rm -f $NAME" >/dev/null 2>&1 || true
echo ">> worker (rank 1) first, then head (rank 0)"
launch 1 "$PEER"; launch 0 ""
echo ">> follow: docker logs -f $NAME   and   ssh $PEER docker logs -f $NAME"
echo ">> check transport: docker logs $NAME 2>&1 | grep -o 'via NET/[A-Z/]*' | sort | uniq -c   (NET/IB, not NET/Socket)"
echo ">> ready: export NAME=\"$NAME\" FLASH_API_BASE=\"http://$MASTER_ADDR:$PORT/v1\"  then ./flash wait && ./flash test"

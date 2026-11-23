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
  nvfp4|hybrid) ;;   # hybrid: -fp8hybrid layout validated after snapshot resolution (#52 carveout)
  hybrid-mtp)
    echo "!! MODE=hybrid-mtp (the NVFP4 MTP draft-experts graft) is not wired on this TP lane —"
    echo "   it builds on the hybrid layout but has never run across two nodes. Use MODE=nvfp4 or hybrid."
    exit 1 ;;
  *) echo "!! MODE must be nvfp4, hybrid or hybrid-mtp"; exit 1 ;;
esac
[ "${DRAFT_VOCAB:-1}" = 0 ] || { echo "!! DRAFT_VOCAB crashes at TP>1 (global ids vs the row-sharded lm_head, issue #52) — set DRAFT_VOCAB=0"; exit 1; }
[ "${PLE_MMAP:-0}" = 0 ] || { echo "!! the PLE mmap hook is untested under TP — set PLE_MMAP=0 (the table fits resident on a pair)"; exit 1; }
[ "${CTX:-262144}" -le 262144 ] || [ "${YARN:-0}" != 0 ] || { echo "!! CTX=${CTX:-262144} exceeds native 262144 without YaRN — set YARN=1 or lower CTX"; exit 1; }
[ "${YARN:-0}" = 0 ] || echo "!! note: YaRN is wired here (--hf-overrides factor 4 + draft max_model_len, same recipe as serve.sh) but has never RUN on TP — before trusting it, gate on retrieval at CTX + scripts/greedy-probe.sh"
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
# MODE=hybrid (issue #52 §2): serve the -fp8hybrid sibling layout. It loads at TP=2 only if
# the shared experts stayed bf16 (640 wide -> 320/rank trips vLLM's blockwise-fp8 loader,
# block_n=128). The INDEX is the ground truth for that — refuse on what the loader will
# actually see, not on what .prepared claims. Prepare: ./flash setup hybrid-tp2 (or
# TP_SIZE=2 scripts/prepare-hybrid.sh), then rsync the whole HF cache to the worker.
HYBRID_ENV=()
MODEL_DIR="$SNAP"
if [ "${MODE:-nvfp4}" = hybrid ]; then
  MODEL_DIR="${SNAP}-fp8hybrid"
  [ -f "$MODEL_DIR/.prepared" ] || { echo "!! hybrid checkpoint not prepared — ./flash setup hybrid-tp2,"
    echo "   or directly: TP_SIZE=2 scripts/prepare-hybrid.sh (one-time, ~10 min, +13 GB)"; exit 1; }
  [ -e "$MODEL_DIR/config.json" ] || { echo "!! the -fp8hybrid layout has no config.json: prepared from an"
    echo "   incomplete download (issue #17) — run TP_SIZE=2 scripts/prepare-hybrid.sh again, it repairs it"; exit 1; }
  IDX="$MODEL_DIR/model.safetensors.index.json"
  [ -f "$IDX" ] || { echo "!! no model.safetensors.index.json in $MODEL_DIR — re-run TP_SIZE=2 scripts/prepare-hybrid.sh"; exit 1; }
  if grep -q 'mlp\.shared_expert\.[a-z_]*\.weight_scale_inv' "$IDX"; then
    echo "!! this -fp8hybrid layout converted the shared experts to fp8 — vLLM will crash at load on TP=2"
    echo "   (#52). Rebuild with the carveout:"
    echo "   rm -rf '$MODEL_DIR'   &&   TP_SIZE=2 scripts/prepare-hybrid.sh"
    exit 1
  fi
  HYBRID_ENV=(-e VLLM_FP8_HYBRID=1 -e VLLM_USE_DEEP_GEMM=0)   # exactly serve.sh's hybrid env
fi
SNAP_IN="/hf/hub/models--${MODEL//\//--}/snapshots/$(basename "$MODEL_DIR")"

# reasoning_effort aliases (mirror of serve.sh EFFORT_ALIAS=1): the NVIDIA template 400s on
# high/max/minimal (Claude Code sends high). Rewrite the one effort-resolving line, publish the
# copy to BOTH hosts' cache, serve it via --chat-template bind-mounted in both containers.
TPL_ARGS=(); TPL_MNT=()
TPL_SRC="$SNAP/chat_template.jinja"
if [ "${EFFORT_ALIAS:-1}" = 1 ] && grep -qF "Supported types are xhigh (default), medium, and low." "$TPL_SRC" 2>/dev/null; then
  EFFORT_OLD="{%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}"
  EFFORT_NEW="{%- set resolved_reasoning_effort = {'high': 'xhigh', 'max': 'xhigh', 'minimal': 'low'}.get(reasoning_effort|default('xhigh'), reasoning_effort|default('xhigh')) %}"
  if [ "$(grep -cF "$EFFORT_OLD" "$TPL_SRC")" = 1 ]; then
    ALIAS_DIR="$HF_CACHE/qwen38-flash-dgx/chat-templates"; ALIAS="$ALIAS_DIR/$NAME.jinja"
    T=""; IFS= read -r -d '' T < "$TPL_SRC" || true   # keeps trailing newlines, unlike $(cat)
    if [ -n "$T" ] && mkdir -p "$ALIAS_DIR" 2>/dev/null \
       && printf '%s%s' '{#- serve-tp2.sh (EFFORT_ALIAS): effort line rewritten, high/max -> xhigh, minimal -> low. -#}' \
            "${T/"$EFFORT_OLD"/"$EFFORT_NEW"}" > "$ALIAS.tmp" 2>/dev/null && mv -f "$ALIAS.tmp" "$ALIAS"; then
      ssh "$PEER" "mkdir -p '$ALIAS_DIR'" && scp -q "$ALIAS" "$PEER:$ALIAS" \
        && { TPL_ARGS=(--chat-template /qwen38/chat_template.jinja); TPL_MNT=(-v "$ALIAS:/qwen38/chat_template.jinja:ro"); } \
        || echo "!! EFFORT_ALIAS: could not publish the template copy to $PEER; reasoning_effort high will 400"
    else echo "!! EFFORT_ALIAS: could not write the rewritten template copy under $ALIAS_DIR"; fi
  else echo "!! EFFORT_ALIAS: the effort line is not the expected one (exactly once); serving the stock template"; fi
fi

# Environment parity is a correctness property here: a drifted image on the
# worker shows up as the NCCL bootstrap error "Message truncated".
have_image() { docker image inspect "$IMAGE" >/dev/null 2>&1; }
have_image || { echo "!! $IMAGE not built here — ./flash setup"; exit 1; }
ssh "$PEER" "docker image inspect $IMAGE >/dev/null 2>&1" \
  || { echo "!! $IMAGE not on $PEER — ship it: docker save $IMAGE | zstd -T0 -3 | ssh $PEER 'zstd -d | docker load'"; exit 1; }
PEER_SNAP="$REPO_DIR/snapshots/$(basename "$MODEL_DIR")"
ssh "$PEER" "test -d '$PEER_SNAP'" \
  || { echo "!! $(basename "$MODEL_DIR") not on $PEER at $REPO_DIR — rsync the whole $HF_CACHE across"; exit 1; }
if [ "${MODE:-nvfp4}" = hybrid ]; then
  # A follower's synced cache does not gain new files retroactively: check the worker's
  # copy of the layout too, so a stale rsync fails here instead of at rank 1's load.
  ssh "$PEER" "test -f '$PEER_SNAP/model.safetensors.index.json' \
    && ! grep -q 'mlp\.shared_expert\.[a-z_]*\.weight_scale_inv' '$PEER_SNAP/model.safetensors.index.json'" \
    || { echo "!! the -fp8hybrid copy on $PEER is missing its index or still has fp8 shared experts (#52)"
         echo "   — rank 1 would crash at load. rsync the whole $HF_CACHE across and re-run"; exit 1; }
fi

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
# same published YaRN recipe serve.sh uses (factor 4 over the native 262144)
YARN_OVR='{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}'

launch() {  # $1 = rank, $2 = ssh target ('' = this node)
  local rank="$1" host="$2"
  # vLLM otherwise advertises the head's DEFAULT-ROUTE IP (the management NIC)
  # as mq_connect_ip for the headless<->head RPC, and the worker silently stalls
  # at startup if it has no route there. Pin it per rank to the fabric address.
  local host_ip="$MASTER_ADDR"
  [ "$rank" = 0 ] || host_ip="${PEER#*@}"
  local allow_long=0
  [ "${YARN:-0}" != 0 ] && allow_long=1
  local split_flag=0
  [ "${SPLIT_PREFILL:-0}" = 1 ] && split_flag=1
  local docker_args=(run -d --name "$NAME" --restart unless-stopped --gpus all --ipc=host
    --network host --device=/dev/infiniband
    --ulimit memlock=-1:-1 --ulimit nofile=1048576:1048576 --cap-add IPC_LOCK
    -v "$HF_CACHE:/hf" -e HF_HOME=/hf -e HF_HUB_OFFLINE=1
    -e "VLLM_HOST_IP=$host_ip" -e "VLLM_ALLOW_LONG_MAX_MODEL_LEN=$allow_long"
    -e VLLM_PLE_MMAP=0 -e VLLM_USE_FLASHINFER_SAMPLER=1
    -e "VLLM_QSA_EXACT_TOPK=${EXACT_TOPK:-0}" -e "VLLM_DGX_SPLIT_PREFILL=$split_flag"
    -e VLLM_QSA_DET_TOPK=1 -e VLLM_QSA_DET_LIB=/opt/llm/kernel-det/_C_det.so
    -e NCCL_NET=IB -e NCCL_IB_DISABLE=0
    -e "NCCL_IB_HCA=$NCCL_IB_HCA"
    -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET -e "NCCL_IB_ADDR_RANGE=$ROCE_SUBNET"
    -e "NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME" -e "GLOO_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME" -e "TP_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
    -e NCCL_NVLS_ENABLE=0 -e NCCL_CROSS_NIC=0 -e NCCL_IB_MERGE_NICS=0 -e NCCL_CUMEM_ENABLE=0
    -e NCCL_IGNORE_CPU_AFFINITY=1 -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 -e "NCCL_DEBUG=${NCCL_DEBUG:-WARN}")
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
  [ "${YARN:-0}" != 0 ] && vllm_args+=(--hf-overrides "$YARN_OVR")
  [ ${#TPL_MNT[@]} -gt 0 ] && docker_args+=("${TPL_MNT[@]}")
  [ ${#TPL_ARGS[@]} -gt 0 ] && vllm_args+=("${TPL_ARGS[@]}")
  [ ${#HYBRID_ENV[@]} -gt 0 ] && docker_args+=("${HYBRID_ENV[@]}")
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
ssh -t "$PEER" 'echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null'
docker rm -f "$NAME" >/dev/null 2>&1 || true
ssh "$PEER" "docker rm -f $NAME" >/dev/null 2>&1 || true
echo ">> worker (rank 1) first, then head (rank 0)"
launch 1 "$PEER"; launch 0 ""
echo ">> follow: docker logs -f $NAME   and   ssh $PEER docker logs -f $NAME"
echo ">> check transport: docker logs $NAME 2>&1 | grep -o 'via NET/[A-Z/]*' | sort | uniq -c   (NET/IB, not NET/Socket)"
echo ">> ready: export NAME=\"$NAME\" FLASH_API_BASE=127.0.0.1:$PORT  then ./flash wait && ./flash test   (FLASH_API_BASE = host:port only; remote clients: http://$MASTER_ADDR:$PORT/v1)"

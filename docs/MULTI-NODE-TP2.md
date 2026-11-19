# Qwen3.8-Flash-Next at TP=2 across two DGX Sparks — profile guide

> **Status: EXPERIMENTAL — this page, `profiles/default-tp2.env` and `scripts/serve-tp2.sh`
> were primarily written by an AI agent** (Hermes, 2026-11-19), assembled from
> [issue #52](https://github.com/blazux/qwen3.8-Flash-DGX/issues/52), the
> [x00byte dual-Spark recipe](https://github.com/x00byte/Qwen3.8-Flash-Dual-Spark-Recipe),
> the [capetron GB10 cluster guide](https://capetron.github.io/gb10-cluster-guide/serving-models/),
> the NVIDIA multi-node playbooks and vLLM's own docs. The launcher's command
> assembly is mock-verified (`bash -n` + stubbed `docker`/`ssh` dry-runs), but
> **no live two-Spark run has validated this checkout**, and it is deliberately
> outside `flash serve` and the repo's single-node validation suite. Treat every
> run on the pair as a validation session: run §3.7 end to end before trusting
> outputs.

This recipe runs `blazux/qwen3.8-Flash-DGX` as a two-node **tensor-parallel (TP=2)**
deployment over the ConnectX-7 / RoCE link. It is *not* a first-class repo mode —
`flash`/`serve.sh` are single-container tools — but the community work is mature:
[issue #52](https://github.com/blazux/qwen3.8-Flash-DGX/issues/52) (two Sparks, TP=2
over RoCE, with two TP-specific bugs), the
[x00byte dual-Spark recipe](https://github.com/x00byte/Qwen3.8-Flash-Dual-Spark-Recipe),
and the [capetron GB10 cluster guide](https://capetron.github.io/gb10-cluster-guide/serving-models/)
(measured TP2 numbers for this exact model). This guide folds them into a profile.

What you get (medians from the sources; each column is a different session, so treat
as a range): single-stream decode ~50 tok/s (vs ~36–43 on one Spark at `mtp-fast`),
about −20% TTFT, and a shared KV pool of 1.4–1.9M tokens (vs ~550k–1.1M). Aggregate
gain at 4+ clients is small (~+9% in #52). You buy latency and context, not throughput.

## 1. What breaks with this model + two Sparks (and the fix)

| # | Symptom | Cause | Fix |
|---|---------|-------|-----|
| 1 | NCCL "works" but prefill is *slower than one node*; `NCCL_DEBUG=INFO` shows `via NET/Socket` | The playbook/`run_cluster.sh` path doesn't pass the IB devices into the container, so NCCL silently falls back to TCP ([#52](https://github.com/blazux/qwen3.8-Flash-DGX/issues/52)) | `--device=/dev/infiniband --ulimit memlock=-1:-1 --cap-add=IPC_LOCK` plus the `NCCL_IB_*` env in §4; verify every channel says `NET/IB` |
| 2 | `ValueError: Weight output_partition_size = 320 is not divisible by weight block_n = 128` at load | `MODE=hybrid`: `mlp.shared_expert.*` is 640 wide → 320/rank, not divisible by the fp8 block size ([#52](https://github.com/blazux/qwen3.8-Flash-DGX/issues/52)) | Start with `MODE=nvfp4` (works). For hybrid, see §7.1 (exclude shared experts from the fp8 conversion) |
| 3 | `CUBLAS_STATUS_INTERNAL_ERROR` in `mtp.py:_dv_compute_logits` at first decode | `DRAFT_VOCAB=1` (`mtp-standard`, `default`) does `w.index_select(0, ids)` with global ids against a TP-row-sharded `lm_head` ([#52](https://github.com/blazux/qwen3.8-Flash-DGX/issues/52)) | `DRAFT_VOCAB=0` (`mtp-fast` already is). #52 includes a shard-aware fix if you want the +20% decode back |
| 4 | Garbled/incorrect output only on TP (inference from #52's setup notes) | The PLE-mmap hook (`VLLM_PLE_MMAP=1`, the single-Spark trick) gathers **global** rows on every rank; TP semantics for the resident path differ | Run with `VLLM_PLE_MMAP=0`. No mmap needed: each rank only holds ~1/2 of the non-table weights, so the 47.7 GiB n-gram table fits resident (drop caches first, §3.4) |
| 5 | After weight load: hang at first all-reduce / KV profiling, GPU pinned ~100%, no traceback | Cross-node all-reduce failure class on early dual-Spark stacks (vLLM 0.10–0.17 era, driver 580.126) — [forum thread](https://forums.developer.nvidia.com/t/nccl-all-reduce-deadlock-on-dual-dgx-spark-after-successful-channel-establishment-affects-both-vllm-and-trt-llm/366127); the reporter there got it working only after moving to the current stack with the `mp` executor (no Ray). #52 confirms the same class is gone on driver 580.178.04 / vLLM ≥ 0.31 | Update DGX OS/driver; use the `mp` backend as here. If a hang persists: `NCCL_DEBUG=INFO` and confirm `NET/IB` before blaming vLLM |
| 6 | `Call to ibv_reg_mr_iova2 failed with error Cannot allocate memory` | Kernel regression: `7.0.0-1019-nvidia` broke RoCE MR registration for some stacks ([forum thread](https://forums.developer.nvidia.com/t/dgx-spark-regression-kernel-7-0-0-1019-nvidia-causes-nccl-roce-ibv-reg-mr-iova2-enomem-6-17-0-1032-works/383023)); NVIDIA shipped a `kho=off` hotfix, and #52 ran on 7.0.0-1019 successfully, so this bites only with certain driver pairings | `uname -r` on both nodes; boot the known-good `6.17.0-1032-nvidia` or apply the latest DGX OS hotfix |
| 7 | Follower node dies at KV init: `AssertionError: collective_rpc should not be called on follower node` | Both containers running the full engine | Rank 1 launches with `--headless` ([x00byte §4](https://github.com/x00byte/Qwen3.8-Flash-Dual-Spark-Recipe)) |
| 8 | Node becomes unresponsive / vLLM exits 255 during post-load compile; `No available shared memory broadcast block found in 60 seconds` | torch.compile / CUDA-graph capture on multi-node GB10 ([x00byte §1](https://github.com/x00byte/Qwen3.8-Flash-Dual-Spark-Recipe); [capetron](https://capetron.github.io/gb10-cluster-guide/serving-models/) measured eager > piecewise at TP4; a forum post attributes a variant to CUDA-graph capture OOM during warmup) | First bring-up with `--enforce-eager` (`EAGER=1` in the profile below). #52 ran the repo's piecewise recipe successfully, so once it works, you can try `EAGER=0` — but only after a validated eager run |
| 9 | Bootstrap error `Message truncated: received N bytes instead of M` | NCCL version drift — same tag pulled on different days | Pin the **image digest** identically on both nodes (§3.2) |
| 10 | "no config.json" / dangling links in the container | HF snapshot dirs are **relative** symlinks into `../../blobs`; mounting only the snapshot breaks them | Mount the whole `$HF_CACHE` (what `serve.sh`/the launcher below do) |
| 11 | vLLM worker vanishes ~10% into a load, `OOMKilled=false`, no traceback | `earlyoom`'s default 10% MemAvailable floor kills the worker | `systemctl is-enabled earlyoom` on both boxes; disable or raise the floor |
| 12 | Load fails once per boot (`operation not permitted` on worker, gloo "connection closed" on head) | Un-root-caused first-load-after-boot defect, one NVFP4 2-node config ([capetron traps](https://capetron.github.io/gb10-cluster-guide/serving-models/)) | Reboot both nodes; keep stacks identical across reboots |
| 13 | `Call to ibv_modify_qp failed with 61 ... local GID ::` right after channels come up — often only after ONE node rebooted | Pinned `NCCL_IB_GID_INDEX` + unstable GID table indices: reboots and address add/remove leave holes that displace the RoCEv2 IPv4 entry, and the two nodes' (or two HCAs') tables disagree ([MiaAI-Lab dual-Spark #73](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks/issues/73)) | Don't pin the index: leave `NCCL_IB_GID_INDEX` unset and let NCCL ≥2.21 resolve the v2 GID per rank from `NCCL_IB_ADDR_FAMILY=AF_INET` + `NCCL_IB_ADDR_RANGE=$ROCE_SUBNET` + `NCCL_IB_ROCE_VERSION_NUM=2` (what `serve-tp2.sh` does; it also preflights both nodes' GID tables before launch) |
| 14 | Both nodes go quiet right after `Using ['PYNCCL'] all-reduce backends` — no weight-load lines, no error; head's earlier log shows `mq_connect_ip=<some management-NIC IP> (local)` | vLLM derives the headless↔head RPC address from the head's **default route**, not from `--master-addr`; rank 1 then dials an IP it may have no route to and the engines wait on each other | `serve-tp2.sh` exports `VLLM_HOST_IP` per rank (head: `MASTER_ADDR`, worker: the `PEER` host). Check with `ping <head's advertised mq_connect_ip>` from the worker; a box on a separate management LAN has no route — that's the point |

Model-side issues that are *not* TP-specific but still apply (you keep all the repo's
image patches, so you get the prefix-cache/Mamba fixes): [#73](https://github.com/blazux/qwen3.8-Flash-DGX/issues/73)
greedy divergence on prefix hits (see `prefix-cache-safe` / `SPLIT_PREFILL=1` —
**TP behavior of that workaround is untested**), [#31](https://github.com/blazux/qwen3.8-Flash-DGX/issues/31)
turn termination on a literal control token. The single-node launcher's
`EFFORT_ALIAS` template rewrite isn't wired into the TP launcher, so
`reasoning_effort: "high"` will 400 (same caveat #52 reported).

Parallelism limits for this checkpoint (all measured, [capetron](https://capetron.github.io/gb10-cluster-guide/serving-models/)):
TP2 is clean (640 → 320 per rank is fine); plain **TP4 fails to load**
(`NotImplementedError: Intermediate size padding … FLASHINFER_CUTLASS`) unless you add
`--enable-expert-parallel`; **TP6 impossible** (16 % 6); **pipeline parallel impossible**
at any node count (`hyper_connection_mixer` — the cross-layer hyper-connections are not
stage-sliceable in this vLLM). Two Sparks TP=2 is therefore the natural pairing.

## 2. Prerequisites

- 2× DGX Spark (GB10, 128 GB each — same size on both), DGX OS up to date, driver ≥ 580.178.
- One QSFP 200G DAC (or two, both rails) between the boxes. The repo recipe was
  validated with RoCE; over plain TCP, two nodes were *slower at prefill than one* (#52).
- ~140 GB free NVMe per box (weights) + 13 GB if you later prepare hybrid.
- Passwordless SSH head → worker; `docker` usable by your user on both.
- A checkout of this repo on **both** nodes, same commit.

## 3. Bring-up steps

### 3.1 Fabric (both nodes)

If the pair was configured with the NVIDIA Sync [Cluster Assistant](https://docs.nvidia.com/sync/latest/cluster-assistant.html),
the ConnectX-7 interfaces already carry **preset 10.100.x.x addresses** and there is
nothing to configure — just read them (`ip -br addr`) and copy yours into the profile's
fabric block (`MASTER_ADDR`, `PEER`, `ROCE_SUBNET`). Do not add a second subnet on top;
one consistent address per link is what NCCL resolves the RoCEv2 GID against.

Otherwise follow NVIDIA's [Connect Multiple DGX Sparks](https://build.nvidia.com/spark/connect-multiple-sparks)
playbook. The minimum that matters here (`<peer-fabric-ip>` = the other node's address
on the same link):

```shell
ip link set dev enp1s0f0np0 mtu 9000            # your iface name: `ls /sys/class/infiniband` + `ip a`
ip addr add 10.100.0.1/24 dev enp1s0f0np0       # 10.100.0.2 on the worker — any one IPv4
                                               # subnet shared by both CX-7 ifaces works
ping -c3 -M do -s 8000 <peer-fabric-ip>          # ~<1 ms, no fragmentation
ibv_devices                                      # expect 2 RoCE devs per box, e.g. rocep1s0f0 roceP2p1s0f0
show_gids                                        # optional eyeball only: there must be a v2 row with an
                                               # IPv4 inside your subnet on every HCA named in NCCL_IB_HCA.
                                               # Do NOT pin its index — see §1 #13; serve-tp2.sh resolves
                                               # it per rank and preflights both tables at launch
ib_write_bw -d rocep1s0f0                        # worker; head: ib_write_bw -d rocep1s0f0 <peer-ip>  -> ~98 Gb/s per rail
```

`NCCL_IB_HCA`/`NCCL_SOCKET_IFNAME` in the profile must match what you see here —
device names differ between Spark revisions and GX10 boards.

### 3.2 Image — build once, ship, pin the digest

```shell
# head node
./flash setup                    # builds qwen38-flash-dgx:v0.33 (all patches incl. the TP-relevant ones)
docker save qwen38-flash-dgx:v0.33 | zstd -T0 -3 | ssh worker 'zstd -d | docker load'
docker images --digests qwen38-flash-dgx      # record the digest; use it on BOTH nodes
```

Do not rely on the tag alone — see issue table #9. (Pulling the base on each node
also works, but takes ~10× longer over a slow uplink.)

### 3.3 Weights — same files, same paths, both nodes

```shell
./flash setup                    # downloads to ~/.cache/huggingface on the head
rsync -a --info=progress2 ~/.cache/huggingface/ worker:~/.cache/huggingface/
# rsync preserves the relative snapshot->blob symlinks; verify on the worker:
find ~/.cache/huggingface/hub -lname '*blobs*' -xtype l   # must print nothing
```

### 3.4 Host hygiene (both nodes, every boot)

```shell
uname -r                     # see issue table #6
systemctl is-enabled earlyoom && sudo systemctl disable --now earlyoom
free -g                      # MemAvailable >= ~105 GiB (nothing else loaded)
echo 3 | sudo tee /proc/sys/vm/drop_caches    # right before launch (§3.6); warm cache starves the GPU allocator mid-load
docker ps                                    # no other GPU containers
```

### 3.5 The profile — `profiles/default-tp2.env`

Shipped in this checkout (EXPERIMENTAL; primarily AI-authored — see the banner at
the top; it is `serve-tp2.sh`'s default profile). `./flash doctor default-tp2`
cross-checks it against the §1 failure table; edit
only the fabric block at the end (`MASTER_ADDR`, `PEER`,
`NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA`,
`ROCE_SUBNET`) to match §3.1 — there is deliberately no `NCCL_IB_GID_INDEX`:
pinned indices break on one-sided reboots (§1 #13), so the RoCEv2 GID is resolved
by address range instead. The settings that matter and
why: `MODE=nvfp4` (§1 #2), `DRAFT_VOCAB=0` (§1 #3), `PLE_MMAP=0` (§1 #4),
`CTX=262144` native (YaRN unverified on TP), `BATCH_TOKENS=8192` (larger measured
worse, #52), `EAGER=1` for the first bring-up (§1 #8), `GPU_MEM=0.80`
(≥0.85 leaves a 1–4% MemAvailable floor on the unified pool).

### 3.6 The launcher — `scripts/serve-tp2.sh`

Shipped in this checkout. `flash serve`/`serve.sh` refuse `TP_SIZE>1` on purpose:
one bridged, port-mapped container can't join a multi-node `mp` group (cross-node
NCCL and the torch rendezvous need the hosts' real fabric IPs, so these
containers use `--network host`), and serve.sh's fixed `-e VLLM_PLE_MMAP=1` is the
config §1 #4 says not to use. Run on the head node:

```shell
scripts/serve-tp2.sh [profile]      # profile defaults to default-tp2;
                                    # MASTER_ADDR/PEER come from its fabric block
```

It applies the §1 guards (hybrid/draft-vocab/mmap/YARN), checks image and
snapshot parity on the worker first (§1 #9/#10), resolves the RoCEv2 GID on BOTH
nodes' GID tables and refuses to start if either has none in `ROCE_SUBNET`
(§1 #13), drops the page cache on both
boxes, then starts **rank 1 (`--headless`) first, then rank 0** (§1 #7/#12). The
container assembly is `--network host --device=/dev/infiniband --ulimit
memlock=-1:-1 --ulimit nofile=1048576:1048576 --cap-add IPC_LOCK` plus the
NCCL-over-RoCE env block (`NCCL_NET=IB`, `NCCL_IB_HCA`, the
`NCCL_IB_ADDR_FAMILY/RANGE` pair — no pinned GID index, §1 #13 — the
`*_SOCKET_IFNAME` trio, `NCCL_CUMEM_ENABLE=0`, `NCCL_NVLS_ENABLE=0`), and the vLLM
line is the §2 table's flags (`--tensor-parallel-size 2 --nnodes 2 --node-rank …
--distributed-executor-backend mp`, MTP via `--speculative-config`,
`--enforce-eager` when `EAGER=1`). `EXTRA=...` appends verbatim vLLM flags.


### 3.7 Verify

```shell
# take NAME/MASTER_ADDR/PORT from the profile (wherever you launched from):
set -a; . profiles/default-tp2.env; set +a; NAME="${NAME:-qwen38-flash-tp2}"
# transport first: one channel line per connection
docker logs "$NAME" 2>&1 | grep -o 'via NET/[A-Z/]*' | sort | uniq -c     # NET/IB — good; NET/Socket — fix §3.1/§4
# API (head node):
curl -s "http://${MASTER_ADDR}:${PORT:-18300}/v1/models" | head -c 200
NAME="$NAME" FLASH_API_BASE="http://${MASTER_ADDR}:${PORT:-18300}/v1" ./flash wait
NAME="$NAME" FLASH_API_BASE="http://${MASTER_ADDR}:${PORT:-18300}/v1" ./flash test
scripts/greedy-probe.sh            # fresh vs repeat must agree (TP+prefix-cache canary)
```

Then watch both boxes for the first hour: `watch -n5 free -g` (head had ~6 GiB
MemAvailable left while serving in #52) and a needle test at ~37% prompt depth,
cold and on a cache hit.

## 4. Expected numbers (published, same model)

| | 1 Spark (`mtp-fast`) | 2 Sparks TP=2 (#52, v0.31, graphs) | 2 Sparks TP=2 (capetron, eager) | 2 Sparks TP=2 (x00byte, eager) |
|---|---|---|---|---|
| Decode, single stream | 42.6 tok/s sampled | 50.4–51.1 | 35.3 / 36.0 / 53.2 (prose/code/math) | 39–64 |
| Prefill, warm | 3,400–3,500 | 3,673 @10.6k | 2,303 | 2,932 @32k |
| TTFT 8k | 2.0–2.6 s | 2.02 | 3.9 s (8.9k) | — |
| 4–8 streams aggregate | 128 | 135 | 118 | 118–196 |
| KV pool | 500k–1.1M | 1.41M | 1.85M | 1.9M–3.16M |

Columns differ in recipe (YaRN 500k vs native 262k, graphs vs eager, fp8 vs bf16 KV),
so use them as envelopes, not promises.

## 5. Day-2 operations

- **Restart:** `scripts/serve-tp2.sh` recreates both containers; a stopped pair can't
  be `docker start`ed back into TP — the rendezvous is per-launch. Recreate instead.
- **One node reboots → reboot both** before the next launch if you hit issue table #12.
  GID-table drift from a one-sided reboot is now handled (§1 #13): nothing pins an
  index, and the launch preflight names the node if a table goes stale/empty.
- **Quality:** run the repo's `./flash test` and a tool-calling check after any stack
  change; #52 validated needles + first-token logprob agreement cold vs cached on TP2.

## 6. What you should NOT try (measured failures)

- `MODE=hybrid` unchanged at TP>1 (issue table #2), `DRAFT_VOCAB=1` unchanged (#3),
  YaRN >262k (untested on TP), `BATCH_TOKENS=32768` (#52: worse both metrics),
  TCP-only transport (#52: slower than one node), TP4 without `--enable-expert-parallel`,
  any PP split of this model.

## 7. Optional upgrades, in order of payoff

1. **Hybrid checkpoint on TP** (the +20% decode): run `prepare-hybrid.sh` with the
   shared experts excluded, per #52 — edit a copy of `tools/fp8_convert.py` `TARGETS`
   to drop the `mlp.shared_expert.*` line (they stay bf16; 1692 tensors convert
   instead of 1836), then serve the `-fp8hybrid` layout with `MODE=hybrid` and
   `VLLM_FP8_HYBRID=1 -e VLLM_USE_DEEP_GEMM=0`. (Inference: the repo's hybrid hook
   reads scales per tensor, so leaving one family unconverted only costs bandwidth.)
2. **Draft-vocab under TP** (#52 §1 shard-aware `_dv_compute_logits` patch — it was
   offered as a PR and never merged; port it into `src/` as patch 10b).
3. **fp8 KV (`KV_DTYPE=fp8_e4m3`)** for a bigger pool (capetron ran it at TP2; the
   repo's single-node 1M profiles use it with a quality cost — re-run retrieval checks).
4. **Both rails** (`NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0`): prefill +10–19% (#52),
   decode neutral. With Sync-preset pairs the two QSFP links often sit in different
   /24s — set `ROCE_SUBNET` to a supernet covering both (the GID preflight checks
   every HCA against that one range).
5. **Piecewise CUDA graphs** (`EAGER=0`) once eager runs are stable — #52 ran them at
   TP2 on v0.31. Any hang at capture: revert to eager, see issue table #8.
6. **Prefix-cache checkpoint workaround** (`SPLIT_PREFILL=1`, patch 24): the patch
   site (`mamba/abstract.py`) is per-rank and plausibly TP-safe, but nothing
   published has tested it on TP — treat as experimental.

## 8. Sources

- Repo: [issue #52](https://github.com/blazux/qwen3.8-Flash-DGX/issues/52) (setup, RoCE flags, the two TP bugs, benches), issues #2, #31, #73, docs/HOW-IT-WORKS.md, docs/PREFIX-CACHE-WORKAROUND.md.
- [x00byte/Qwen3.8-Flash-Dual-Spark-Recipe](https://github.com/x00byte/Qwen3.8-Flash-Dual-Spark-Recipe) — `mp`+`--headless` launch lines, enforce-eager lockup, PLE fp8 scale bug (day-zero dev image), caches/drop_caches, tool-eval 91/100.
- [capetron GB10 cluster guide](https://capetron.github.io/gb10-cluster-guide/serving-models/) — measured TP2/TP4+EP/6-node results for this model, NCCL env set, earlyoom, digest pinning, hyper-connection PP limit.
- [MiaAI-Lab dual-Spark issue #73](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks/issues/73) — GID-table index drift on one-sided reboots; why `NCCL_IB_GID_INDEX` must not be pinned. [NCCL env docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html) — `NCCL_IB_ADDR_FAMILY`/`ADDR_RANGE`/`ROCE_VERSION_NUM` drive selection only when `GID_INDEX` is unset.
- [vLLM parallelism docs](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/) — `--nnodes/--node-rank/--headless`, GPUDirect flags, `NET/IB` vs `NET/Socket` verification.
- NVIDIA: [Connect Multiple DGX Sparks](https://build.nvidia.com/spark/connect-multiple-sparks), [vLLM cluster playbook](https://build.nvidia.com/playbooks/vllm-cluster) ([eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker)), forum threads [366127](https://forums.developer.nvidia.com/t/nccl-all-reduce-deadlock-on-dual-dgx-spark-after-successful-channel-establishment-affects-both-vllm-and-trt-llm/366127) (deadlock, fixed by stack update) and [383023](https://forums.developer.nvidia.com/t/dgx-spark-regression-kernel-7-0-0-1019-nvidia-causes-nccl-roce-ibv-reg-mr-iova2-enomem-6-17-0-1032-works/383023) (kernel RoCE regression).

Items marked "(inference …)" or "untested on TP" are not measured claims.

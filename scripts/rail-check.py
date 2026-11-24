#!/usr/bin/env python3
"""Rail-traffic bracket: snapshot IB counters, watch until you generate traffic
elsewhere, re-snapshot when it quiets, report MiB per counter.

    python3 scripts/rail-check.py [max_wait_s]      # default 120

Counters are sampled at 1 Hz. The window closes IDLE_S seconds after the last
counter movement, or at max_wait_s — so it brackets exactly your burst.
For a prefill-shaped burst the prompt must be COLD (fresh text, or right after
a boot): a cached prompt moves a few MiB and reads as a flat rail.
"""
import sys, time

HCAS  = ["rocep1s0f0", "roceP2p1s0f0"]
IDLE_S, POLL_S = 5.0, 1.0
PATH  = "/sys/class/infiniband/{d}/ports/1/counters/port_{k}_data"

def snap():
    return {(d, k): int(open(PATH.format(d=d, k=k)).read().strip())
            for d in HCAS for k in ("xmit", "rcv")}

a = prev = snap()
max_wait = float(sys.argv[1]) if len(sys.argv) > 1 else 120.0
t0 = time.monotonic()
print(f"counters live — run your prompt now; closing {IDLE_S:.0f}s after traffic "
      f"goes quiet (hard cap {max_wait:.0f}s)")
saw_traffic, quiet_since = False, None
while True:
    time.sleep(POLL_S)
    cur = snap()
    if any(cur[k] != prev[k] for k in cur):
        saw_traffic, quiet_since = True, None
    elif quiet_since is None:
        quiet_since = time.monotonic()
    prev = cur
    if saw_traffic and quiet_since is not None and time.monotonic() - quiet_since >= IDLE_S:
        break
    if time.monotonic() - t0 >= max_wait:
        if not saw_traffic:
            print("!! no counter movement for the whole window — nothing was "
                  "measured (prompt served fully from cache? check the container's NCCL_IB_HCA)")
        break

b = snap()
for key in a:
    w = (1 << 64) if max(a[key], b[key]) >= (1 << 32) else (1 << 32)
    d = (b[key] - a[key]) % w
    print(f"{'.'.join(key):26} {d * 4 / 2**20:7.1f} MiB  "
          f"({'64-bit' if w == 1 << 64 else '32-bit'} counter)")
print("expect: this node's xmit roughly balanced across the two rails, "
      "each ~1.5-2 GiB per cold ~10k prefill; run on the worker to see rcv mirror it")

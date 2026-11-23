#!/usr/bin/env python3
# From https://github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound (tools/fp8_convert.py, Apache-2.0),
# by @Saren-Arterius. Included verbatim (minus this header) so scripts/prepare-hybrid.sh is self-contained.
# Regex TARGETS matches the RadixArk NVFP4 checkpoint tensor names as well (same HF layout).
# Local deviation (this repo): KEEP_SHARED_EXPERT_BF16=1 carves mlp.shared_expert.* out of the
# conversion — required for MODE=hybrid at TP>1 (640 wide -> 320/rank vs fp8 block_n=128,
# blazux/qwen3.8-Flash-DGX#52 §2; scripts/prepare-hybrid.sh sets it when TP_SIZE>1).
# Everything else stays verbatim.
"""Convert dense bf16 side-layers of the Intel Flash-Next checkpoint to
blockwise FP8-e4m3 (DeepSeek format: fp8 `weight` + fp32 `weight_scale_inv`,
block 128x128). In-place: affected shards are rewritten, originals -> .bf16.bak.

Only converts tensors whose both dims are divisible by 128 (all listed families
qualify); anything else is left untouched and reported.
"""
import json
import os
import re
import shutil
import sys

import torch
from safetensors.torch import load_file, save_file

ROOT = sys.argv[1] if len(sys.argv) > 1 else sys.exit("usage: fp8_convert.py <checkpoint_dir>")
BLOCK = 128
FP8_MAX = 448.0  # e4m3 max normal

SIDELAYERS = (
    r"linear_attn\.(in_proj_qkv|in_proj_z|out_proj)"
    r"|self_attn\.(q_proj|k_proj|v_proj|o_proj)"
)
# mlp.shared_expert.* is 640 wide: at TP=2 a rank gets 320 columns, which vLLM's
# blockwise-fp8 loader rejects ("output_partition_size = 320 is not divisible by
# block_n = 128", issue #52 §2). Left bf16 the layout loads at any TP, costing
# the shared-expert bandwidth (3 tensors x 48 layers = 144 unconverted; 1692 fp8
# tensors convert instead of 1836 — the carveout #52 served on a live TP=2 pair).
if os.environ.get("KEEP_SHARED_EXPERT_BF16") == "1":
    print("carveout: mlp.shared_expert.* stays bf16 (TP>1, issue #52)")
else:
    SIDELAYERS += r"|mlp\.shared_expert\.(gate_proj|up_proj|down_proj)"
TARGETS = re.compile(r"model\.language_model\.layers\.\d+\.(" + SIDELAYERS + r")\.weight$")


def block_quant(w: torch.Tensor):
    out, inn = w.shape
    assert out % BLOCK == 0 and inn % BLOCK == 0, w.shape
    wf = w.float().reshape(out // BLOCK, BLOCK, inn // BLOCK, BLOCK)
    absmax = wf.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-12)
    scale = absmax / FP8_MAX
    q = (wf / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    # roundtrip check on this tensor
    deq = q.float() * scale
    rel = ((deq - wf).abs().amax() / wf.abs().amax()).item()
    return (
        q.reshape(out, inn),
        scale.squeeze(1).squeeze(-1).contiguous(),  # [out/128, in/128] fp32
        rel,
    )


def main():
    idx_path = f"{ROOT}/model.safetensors.index.json"
    idx = json.load(open(idx_path))
    wm = idx["weight_map"]
    targets = {n: f for n, f in wm.items() if TARGETS.search(n)}
    by_file = {}
    for name, fname in targets.items():
        by_file.setdefault(fname, []).append(name)
    print(f"{len(targets)} tensors across {len(by_file)} shards")

    worst = 0.0
    for i, (fname, names) in enumerate(sorted(by_file.items())):
        path = f"{ROOT}/{fname}"
        tensors = load_file(path)
        for name in names:
            w = tensors.pop(name)
            if w.shape[0] % BLOCK or w.shape[1] % BLOCK:
                print(f"  SKIP (shape) {name} {tuple(w.shape)}")
                tensors[name] = w
                continue
            q, scale, rel = block_quant(w)
            worst = max(worst, rel)
            tensors[name] = q
            tensors[name.replace(".weight", ".weight_scale_inv")] = scale
            wm[name.replace(".weight", ".weight_scale_inv")] = fname
        shutil.move(path, path + ".bf16.bak")
        save_file(tensors, path)
        print(f"[{i + 1}/{len(by_file)}] {fname}: {len(names)} tensors")
    json.dump(idx, open(idx_path, "w"))
    print(f"done. worst per-tensor max rel err: {worst:.4f}")
    assert worst < 0.10, "fp8 roundtrip error unexpectedly large"


if __name__ == "__main__":
    sys.exit(main())

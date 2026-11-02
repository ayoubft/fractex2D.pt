"""Compute cost per model (review E6): params, FLOPs, latency, peak GPU memory.

Usage:
    python -m review.compute_cost TAG=RUN_DIR [TAG=RUN_DIR ...] [--out review_out/compute.json]
        [--override TAG:hydra.key=value ...]

Each model is rebuilt from its run's saved config with random weights (cost does not
depend on the weights). Input is a 256x256 tile with the run's in_channels.

  params        all parameters (and trainable ones)
  GFLOPs        torch.utils.flop_counter for one 256x256 image; that counter reports
                FLOPs = 2 x multiply-accumulates, so GMACs = GFLOPs / 2
  latency       forward only, eval mode, fp32, torch.no_grad, batch 1 and batch 8;
                30 warm-up passes, then the mean (and std) over 200 timed passes
                measured with CUDA events, in ms per batch and ms per image
  peak memory   torch.cuda.max_memory_allocated (weights included):
                inference = one batch-8 forward under no_grad;
                training  = one batch-8 forward + backward + optimiser step with the
                run's own loss and optimiser (the optimiser state is allocated by it)
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from hydra.utils import instantiate
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from infere import load_cfg  # noqa: E402


def measure(cfg, device, warmup=30, runs=200):
    model = instantiate(cfg.model).to(device)
    c = cfg.in_channels
    res = {"params_M": sum(p.numel() for p in model.parameters()) / 1e6,
           "trainable_params_M": sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6}

    model.eval()
    x1 = torch.rand(1, c, 256, 256, device=device)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(x1)
    res["gflops"] = fc.get_total_flops() / 1e9

    for bs in (1, 8):
        x = torch.rand(bs, c, 256, 256, device=device)
        with torch.no_grad():
            for _ in range(warmup):
                model(x)
            torch.cuda.synchronize()
            times = []
            for _ in range(runs):
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                model(x)
                e.record()
                torch.cuda.synchronize()
                times.append(s.elapsed_time(e))
        t = torch.tensor(times)
        res[f"latency_b{bs}_ms"] = float(t.mean())
        res[f"latency_b{bs}_ms_std"] = float(t.std())
        res[f"latency_b{bs}_ms_per_image"] = float(t.mean()) / bs

    x8 = torch.rand(8, c, 256, 256, device=device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        model(x8)
    torch.cuda.synchronize()
    res["peak_mem_infer_b8_MiB"] = torch.cuda.max_memory_allocated() / 2**20

    del model
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = instantiate(cfg.model).to(device).train()
    criterion = instantiate(cfg.loss)
    optimizer = instantiate(cfg.optimizer, model.parameters())
    y8 = torch.rand(8, 1, 256, 256, device=device)
    out = model(x8)
    loss = criterion(out, y8)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    res["peak_mem_train_b8_MiB"] = torch.cuda.max_memory_allocated() / 2**20
    del model, optimizer
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+", help="TAG=RUN_DIR")
    ap.add_argument("--override", nargs="*", default=[], help="TAG:hydra.key=value")
    ap.add_argument("--out", type=Path, default=Path("review_out/compute.json"))
    args = ap.parse_args()
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = False

    results = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "models": {}}
    for spec in args.models:
        tag, run_dir = spec.split("=", 1)
        ov = [o.split(":", 1)[1] for o in args.override if o.split(":", 1)[0] == tag]
        cfg = load_cfg(Path(run_dir), ov)
        results["models"][tag] = {"run_dir": run_dir, "overrides": ov, **measure(cfg, device)}
        print(tag, json.dumps(results["models"][tag], indent=1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print("Wrote", args.out)


if __name__ == "__main__":
    main()

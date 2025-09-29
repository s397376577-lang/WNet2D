import time, argparse, subprocess, re, torch
from wnet2d.models.wnet2d import build_model

def gpu_used_memory_via_nvidia_smi():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-compute-apps=used_memory", "--format=csv,noheader,nounits"]).decode()
        # Sum all processes on the first GPU
        nums = [int(x.strip()) for x in out.splitlines() if x.strip().isdigit()]
        if nums:
            return sum(nums) / 1024.0  # MB -> GB approx
    except Exception:
        pass
    return None

def measure(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if args.precision.lower()=="fp16" else torch.float32

    model = build_model(args.model, in_channels=3, num_classes=2).to(device)
    if dtype==torch.float16:
        model = model.half()
    model.eval()

    x = torch.randn(1, 3, args.size, args.size, device=device, dtype=dtype)

    # warmup
    for _ in range(args.warmup):
        with torch.no_grad():
            y = model(x)
    torch.cuda.synchronize()

    # latency (CUDA events + sync)
    starter, ender = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
    ms_list = []
    for _ in range(args.runs):
        torch.cuda.synchronize()
        starter.record()
        with torch.no_grad():
            y = model(x)
        ender.record()
        torch.cuda.synchronize()
        ms = starter.elapsed_time(ender)  # milliseconds
        ms_list.append(ms)

    mean_ms = sum(ms_list)/len(ms_list)
    fps = 1000.0/mean_ms

    # memory
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = model(x)
    torch.cuda.synchronize()
    alloc_gb = torch.cuda.max_memory_allocated()/ (1024**3)
    reserv_gb = torch.cuda.max_memory_reserved()/ (1024**3)
    used_gb = gpu_used_memory_via_nvidia_smi()

    if args.report == "allocated":
        mem_gb = alloc_gb
    elif args.report == "reserved":
        mem_gb = reserv_gb
    else:
        mem_gb = used_gb if used_gb is not None else reserv_gb

    print(f"Latency(mean over {args.runs}): {mean_ms:.2f} ms  |  FPS: {fps:.2f}")
    print(f"Peak GPU Memory ({args.report}): {mem_gb:.2f} GB   [allocated={alloc_gb:.2f}, reserved={reserv_gb:.2f}, nvidia-smi Used={used_gb if used_gb is not None else 'N/A'}]")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="wnet2d")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--precision", choices=["fp32","fp16"], default="fp32")
    p.add_argument("--runs", type=int, default=200)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--report", choices=["reserved","allocated","used"], default="reserved",
                   help="reserved (PyTorch peak reserved) | allocated (PyTorch peak allocated) | used (nvidia-smi)")
    args = p.parse_args()
    measure(args)

if __name__ == "__main__":
    main()

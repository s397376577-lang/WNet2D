import argparse, subprocess, torch
from wnet2d import build_wnet2d

# --- 与标杆一致：nvidia-smi 读取 "Used"（仅做参考，不作为默认口径） ---
def gpu_used_memory_via_nvidia_smi():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=used_memory", "--format=csv,noheader,nounits"]
        ).decode()
        nums = [int(x.strip()) for x in out.splitlines() if x.strip().isdigit()]
        if nums:
            return sum(nums) / 1024.0  # MB -> GB
    except Exception:
        pass
    return None

def measure(args):
    # 0) 固定环境与设备 —— 与标杆一致
    assert torch.cuda.is_available(), "需要在有 CUDA 的环境下测"
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True  # 与标杆脚本一致（如需绝对确定性可改为 False）

    # 1) 构建模型与输入（默认 FP32，BS=1，512×512）
    dtype = torch.float16 if args.precision.lower() == "fp16" else torch.float32
    model = build_wnet2d(in_channels=3, num_classes=1).eval().to(device)  # 论文口径：单通道概率图
    if dtype is torch.float16:
        model = model.half()

    x = torch.randn(1, 3, args.size, args.size, device=device, dtype=dtype)

    # 2) 预热 50 次（避免首次 CUDA 初始化偏差）—— 与标杆完全一致
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(x)
    torch.cuda.synchronize()

    # 3) Latency：CUDA events 计时 args.runs 次，取均值 —— 与标杆完全一致
    starter = torch.cuda.Event(enable_timing=True)
    ender   = torch.cuda.Event(enable_timing=True)
    ms_list = []
    with torch.no_grad():
        for _ in range(args.runs):
            torch.cuda.synchronize()
            starter.record()
            _ = model(x)
            ender.record()
            torch.cuda.synchronize()
            ms_list.append(starter.elapsed_time(ender))  # 毫秒

    mean_ms = sum(ms_list) / len(ms_list)
    fps = 1000.0 / mean_ms

    # 4) Peak GPU Memory：peak reserved（≈ nvidia-smi "Used"），与标杆一致
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = model(x)
    torch.cuda.synchronize()
    alloc_gb   = torch.cuda.max_memory_allocated() / (1024 ** 3)
    reserv_gb  = torch.cuda.max_memory_reserved()  / (1024 ** 3)
    used_gb    = gpu_used_memory_via_nvidia_smi()

    if args.report == "reserved":
        mem_gb = reserv_gb
    elif args.report == "allocated":
        mem_gb = alloc_gb
    else:  # "used"
        mem_gb = used_gb if used_gb is not None else reserv_gb

    print(f"Latency (mean over {args.runs}): {mean_ms:.2f} ms")
    print(f"FPS: {fps:.2f}")
    print(
        f"Peak GPU Memory ({args.report}): {mem_gb:.2f} GB   "
        f"[allocated={alloc_gb:.2f}, reserved={reserv_gb:.2f}, nvidia-smi Used={used_gb if used_gb is not None else 'N/A'}]"
    )

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="wnet2d")  # 目前未分支到多模型，这里保留占位
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    p.add_argument("--runs", type=int, default=200)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument(
        "--report", choices=["reserved", "allocated", "used"], default="reserved",
        help="reserved (PyTorch peak reserved) | allocated (PyTorch peak allocated) | used (nvidia-smi)"
    )
    args = p.parse_args()
    measure(args)

if __name__ == "__main__":
    main()

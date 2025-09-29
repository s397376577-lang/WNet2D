import os, yaml, csv, time
import numpy as np
import torch
from torch.utils.data import DataLoader
from wnet2d.models.wnet2d import build_model
from wnet2d.data.kvasir import KvasirDataset

def dice_and_iou(pred, target, eps=1e-6):
    # pred: N,C,H,W raw logits; target: N,1,H,W (0/1)
    prob = torch.softmax(pred, dim=1)[:,1]  # foreground prob
    pred_bin = (prob > 0.5).float()
    target = target[:,0]

    inter = (pred_bin*target).sum(dim=(1,2))
    union = pred_bin.sum(dim=(1,2)) + target.sum(dim=(1,2)) - inter
    dice = (2*inter + eps) / (pred_bin.sum(dim=(1,2)) + target.sum(dim=(1,2)) + eps)
    iou = (inter + eps) / (union + eps)
    return dice.cpu().numpy(), iou.cpu().numpy()

def hd95_asd_pixels(pred, target):
    # computes per-image 95HD and ASD in *pixels* if medpy available
    try:
        from medpy.metric.binary import hd, asd
    except Exception:
        return np.nan, np.nan
    pred = pred.astype(np.bool_)
    target = target.astype(np.bool_)
    if pred.sum() == 0 and target.sum() == 0:
        return 0.0, 0.0
    if pred.sum() == 0 or target.sum() == 0:
        # define as large distance
        return 95.0, 95.0
    return float(hd(pred, target, voxelspacing=None, connectivity=1, percentile=95)), float(asd(pred, target))

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()

    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = KvasirDataset(root=cfg["data_root"], img_dir=cfg.get("train_images","images"),
                       mask_dir=cfg.get("train_masks","masks"), transform=None)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    model = build_model(cfg.get("model","wnet2d"), in_channels=cfg.get("in_channels",3), num_classes=cfg.get("num_classes",2)).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    rows = []
    dices, jaccs, hd95s, asds = [], [], [], []

    with torch.no_grad():
        for i, (img, mask) in enumerate(loader):
            img = img.to(device)
            mask = mask.to(device)
            logits = model(img)
            d, j = dice_and_iou(logits, mask)
            prob = torch.softmax(logits, dim=1)[:,1].cpu().numpy()[0]
            pred_bin = (prob > 0.5).astype(np.uint8)
            gt = mask.cpu().numpy()[0,0].astype(np.uint8)
            hd95, asd_val = hd95_asd_pixels(pred_bin, gt)

            dices.append(d[0]); jaccs.append(j[0]); hd95s.append(hd95); asds.append(asd_val)
            rows.append({"idx": i, "dice": d[0], "iou": j[0], "hd95_px": hd95, "asd_px": asd_val})

    save_dir = cfg.get("save_dir","outputs/kvasir")
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "metrics_kvasir.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["idx","dice","iou","hd95_px","asd_px"])
        w.writeheader(); w.writerows(rows)

    print(f"Saved per-image metrics to {csv_path}")
    print("Summary (mean±sd for region; median[IQR] for distance):")
    import numpy as np
    print(f"Dice={np.mean(dices):.4f}±{np.std(dices):.4f}, IoU={np.mean(jaccs):.4f}±{np.std(jaccs):.4f}")
    print(f"95HD(px) median={np.median(hd95s):.2f} [IQR {np.percentile(hd95s,25):.2f}-{np.percentile(hd95s,75):.2f}]")
    print(f"ASD(px)  median={np.median(asds):.2f} [IQR {np.percentile(asds,25):.2f}-{np.percentile(asds,75):.2f}]")

if __name__ == "__main__":
    main()

import os, time, yaml, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tensorboardX import SummaryWriter

from wnet2d.models.wnet2d import build_model
from wnet2d.data.kvasir import KvasirDataset

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def dice_loss(pred, target, eps=1e-6):
    pred = torch.softmax(pred, dim=1)[:,1]  # assume class 1 is foreground for 2-class
    target = target[:,0] if target.shape[1] == 1 else target.argmax(1).float()
    inter = (pred*target).sum(dim=(1,2))
    union = pred.sum(dim=(1,2)) + target.sum(dim=(1,2)) + eps
    dice = (2*inter + eps) / union
    return 1 - dice.mean()

def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float32, enabled=False):
            logits = model(imgs)
            loss = dice_loss(logits, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)
        logits = model(imgs)
        loss = dice_loss(logits, masks)
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    args = parser.parse_args()

    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # dataset
    ds = KvasirDataset(root=cfg["data_root"], img_dir=cfg.get("train_images","images"),
                       mask_dir=cfg.get("train_masks","masks"), transform=None)
    val_ratio = cfg.get("val_split", 0.1)
    val_len = max(1, int(len(ds)*val_ratio))
    train_len = len(ds) - val_len
    train_ds, val_ds = random_split(ds, [train_len, val_len], generator=torch.Generator().manual_seed(cfg.get("seed",42)))

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=cfg.get("num_workers",4), pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                            num_workers=cfg.get("num_workers",4), pin_memory=True)

    # model
    model = build_model(cfg.get("model","wnet2d"),
                        in_channels=cfg.get("in_channels",3),
                        num_classes=cfg.get("num_classes",2))
    model = model.to(device)

    # optimizer
    if cfg.get("optimizer","adam").lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9, weight_decay=cfg.get("weight_decay",0.0), nesterov=True)
    else:
        optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay",0.0))

    # scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

    # training
    save_dir = cfg.get("save_dir","outputs/kvasir")
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(save_dir)

    best_val = float("inf")
    patience = cfg.get("early_stop_patience", 30)
    patience_cnt = 0

    for epoch in range(1, cfg["epochs"]+1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, None, device)
        val_loss = validate(model, val_loader, device)
        scheduler.step(val_loss)

        writer.add_scalar("loss/train", tr_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            patience_cnt = 0
            torch.save(model.state_dict(), os.path.join(save_dir, "wnet2d_kvasir.pth"))
        else:
            patience_cnt += 1

        print(f"[{epoch:03d}/{cfg['epochs']}] train={tr_loss:.4f} val={val_loss:.4f} best={best_val:.4f} patience={patience_cnt}/{patience}")
        if patience_cnt >= patience:
            print("Early stopping.")
            break

if __name__ == "__main__":
    main()

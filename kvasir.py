import os
from glob import glob
import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class KvasirDataset(Dataset):
    def __init__(self, root, split="train", transform=None, img_dir="images", mask_dir="masks"):
        self.root = root
        self.split = split
        self.transform = transform
        img_path = os.path.join(root, img_dir)
        mask_path = os.path.join(root, mask_dir)
        self.images = sorted(glob(os.path.join(img_path, "*.jpg")) + glob(os.path.join(img_path, "*.png")))
        self.masks  = [os.path.join(mask_path, os.path.basename(p).replace(".jpg",".png")) for p in self.images]
        # Fallback if mask names already align
        self.masks = [m if os.path.exists(m) else m.replace(".png",".jpg") for m in self.masks]
        assert len(self.images) == len(self.masks) and len(self.images) > 0, "No images/masks found"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = cv2.imread(self.images[idx], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            # some masks are RGB; convert to gray
            mask_rgb = cv2.imread(self.masks[idx], cv2.IMREAD_COLOR)
            mask = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2GRAY)

        # binarize (0/1); many Kvasir masks are 0/255
        mask = (mask > 127).astype(np.uint8)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]
        else:
            # default: convert to CHW float
            import numpy as np
            import torch
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2,0,1))  # CHW
            mask = mask.astype(np.float32)[None, ...]  # 1HW
            img = torch.from_numpy(img).float()
            mask = torch.from_numpy(mask).float()

        return img, mask

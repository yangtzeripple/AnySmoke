import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2

custom_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

class AnySmokeSegDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.pairs = []

        for sub in os.listdir(root_dir):
                subpath = os.path.join(root_dir, sub)
                img_dir  = os.path.join(subpath, 'images')
                mask_dir = os.path.join(subpath, 'masks')
                if not (os.path.isdir(img_dir) and os.path.isdir(mask_dir)):
                    continue

    

                mask_map = {
                    os.path.splitext(m)[0]: os.path.join(mask_dir, m)
                    for m in os.listdir(mask_dir)
                    if os.path.splitext(m)[1].lower() in IMG_EXTS
                }

                for img_name in os.listdir(img_dir):
                    base, ext = os.path.splitext(img_name)
                    if ext.lower() not in IMG_EXTS:
                        continue
                    if base in mask_map:
                        img_path  = os.path.join(img_dir, img_name)
                        mask_path = mask_map[base]
                        self.pairs.append((img_path, mask_path))

        if not self.pairs:
            raise RuntimeError(f"No image–mask pairs found under {root_dir!r}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        mask = (mask > 128).astype(np.uint8)

        if self.transform:
            image = self.transform(image)
            mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
            mask = torch.as_tensor((mask), dtype=torch.long)
        else:
            image = transforms.ToTensor()(image)
            mask = torch.as_tensor((mask), dtype=torch.long)

        return image, mask

if __name__ == "__main__":
    dataset = AnySmokeSegDataset(root_dir="./AnySmokeDataset/AnySmokeTrain", transform=custom_transform)
    print(f"Number of sample: {len(dataset)}")
    img, msk = dataset[0]
    print(f"Image tensor shape: {img.shape}, Mask tensor shape: {msk.shape}")



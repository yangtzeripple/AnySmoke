import os
import cv2
import torch
import copy
import numpy as np
from torch.utils.data import Dataset, DataLoader,SubsetRandomSampler
from torchvision import transforms
import segmentation_models_pytorch as smp
import torch.optim as optim
from segmentation_models_pytorch.utils.metrics import IoU, Fscore
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
import sys
sys.path.append('')
from utils import AnySmokeSegDataset

def compute_metrics(outputs, masks, threshold=0.5, eps=1e-6):
    """
    outputs: raw logits, shape (B,1,H,W)
    masks: ground truth in {0,1}, shape (B, H, W)
    """
    probs = torch.sigmoid(outputs).detach()
    preds = (probs > threshold).float()
    masks = masks.float()

    # flatten
    preds_flat = preds.view(-1)
    masks_flat = masks.view(-1)

    TP = (preds_flat * masks_flat).sum()
    FP = (preds_flat * (1 - masks_flat)).sum()
    FN = ((1 - preds_flat) * masks_flat).sum()

    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    iou = TP / (TP + FP + FN + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    mse = torch.mean((probs.view(-1) - masks_flat) ** 2)

    return {
        'iou': iou.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item(),
        'mse': mse.item()
    }

def evaluate(model, loader, device):
    model.eval()
    metrics_sum = {'iou':0, 'precision':0, 'recall':0, 'f1':0, 'mse':0}
    n_batches = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)

            batch_metrics = compute_metrics(outputs, masks)
            for k, v in batch_metrics.items():
                metrics_sum[k] += v
            n_batches += 1

    # 平均
    for k in metrics_sum:
        metrics_sum[k] /= n_batches
    return metrics_sum

def train():
    # 超参数
    num_epochs = 15
    batch_size = 32
    lr = 1e-4

    train_dir = "./AnySmokeDataset/AnySmokeTrain"
    val_dir   = "./AnySmokeDataset/AnySmokeTest"
    val_small_dir   = "./AnySmokeDataset/AnySmokeTestSmall"
    val_medium_dir   = "./AnySmokeDataset/AnySmokeTestMedium"
    val_large_dir   = "./AnySmokeDataset/AnySmokeTestLarge"

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_ds = AnySmokeSegDataset(train_dir, transform=transform)
    val_ds   = AnySmokeSegDataset(val_dir,   transform=transform)

    val_small_ds   = AnySmokeSegDataset(val_small_dir,   transform=transform)
    val_medium_ds   = AnySmokeSegDataset(val_medium_dir,   transform=transform)
    val_large_ds   = AnySmokeSegDataset(val_large_dir,   transform=transform)


    ### Ablation code start
    num_samples = len(train_ds)
    indices     = list(range(num_samples))
    np.random.seed(42)             # 固定 seed 保证可复现
    np.random.shuffle(indices)
    
    Scale = 0.2
    split = int(Scale * num_samples)
    print(Scale)
    train_idx = indices[:split]
    sampler = SubsetRandomSampler(train_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,    # 用 sampler 代替 shuffle
        num_workers=4
    )

    print(len(train_loader)*batch_size)

    # train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4)
    val_loader = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)

    val_small_loader   = DataLoader(val_small_ds,   batch_size=batch_size, shuffle=False, num_workers=4)
    val_medium_loader   = DataLoader(val_medium_ds,   batch_size=batch_size, shuffle=False, num_workers=4)
    val_large_loader   = DataLoader(val_large_ds,   batch_size=batch_size, shuffle=False, num_workers=4)

    # 模型 & 损失 & 优化器
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    )

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_val_iou = 0.0

    print("Start training ...")
    for epoch in tqdm(range(1, num_epochs+1)):
        model.train()
        running_loss = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device).unsqueeze(1).float()  # (B,1,H,W)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # 验证
        val_metrics = evaluate(model, val_loader, device)
        val_iou = val_metrics['iou']

        print(f"Epoch [{epoch}/{num_epochs}] "
              f"Train Loss: {avg_train_loss:.4f} "
              f"Val IoU: {val_iou:.4f}  "
              f"Precision: {val_metrics['precision']:.4f}  "
              f"Recall: {val_metrics['recall']:.4f}  "
              f"F1: {val_metrics['f1']:.4f}  "
              f"MSE: {val_metrics['mse']:.6f}")

        # 保存最优
        if val_iou > best_val_iou:
            best_val_iou = val_iou

        ## large
        val_large_metrics = evaluate(model, val_large_loader, device)
        val_large_iou = val_large_metrics['iou']
        print("large mask performance")
        print(
              f"Val IoU: {val_large_iou:.4f}  "
              f"Precision: {val_large_metrics['precision']:.4f}  "
              f"Recall: {val_large_metrics['recall']:.4f}  "
              f"F1: {val_large_metrics['f1']:.4f}  "
              f"MSE: {val_large_metrics['mse']:.6f}")

        ## medium
        val_medium_metrics = evaluate(model, val_medium_loader, device)
        val_medium_iou = val_medium_metrics['iou']
        print("medium mask performance")
        print(
              f"Val IoU: {val_medium_iou:.4f}  "
              f"Precision: {val_medium_metrics['precision']:.4f}  "
              f"Recall: {val_medium_metrics['recall']:.4f}  "
              f"F1: {val_medium_metrics['f1']:.4f}  "
              f"MSE: {val_medium_metrics['mse']:.6f}")

        ## small
        val_small_metrics = evaluate(model, val_small_loader, device)
        val_small_iou = val_small_metrics['iou']
        print("small mask performance")
        print(
              f"Val IoU: {val_small_iou:.4f}  "
              f"Precision: {val_small_metrics['precision']:.4f}  "
              f"Recall: {val_small_metrics['recall']:.4f}  "
              f"F1: {val_small_metrics['f1']:.4f}  "
              f"MSE: {val_small_metrics['mse']:.6f}")
        

    # 加载最佳权重
    print(f"Training complete. Best Val IoU: {best_val_iou:.4f}")

if __name__ == "__main__":
    train()

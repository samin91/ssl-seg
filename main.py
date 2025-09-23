import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torchvision.ops import FeaturePyramidNetwork
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from losses.Dice import DiceLoss
from utils.visualization import visualize_predictions 
from PIL import Image
import os
import argparse
from tqdm import tqdm
import csv
from torch.utils.tensorboard import SummaryWriter
import pdb

# ------------------------------
# 0. Utility function
# ------------------------------
# 0.1
# ------------------------------
def compute_dataset_stats(images_dir, split="train"):
    train_images = images_dir + "/" + split + '/images'
    imgs = [np.array(Image.open(os.path.join(train_images, f)).convert("L"), dtype=np.float32)/255.0
            for f in os.listdir(train_images)]
    imgs = np.stack(imgs, axis=0)
    mean = imgs.mean()
    std = imgs.std()
    return mean, std
# ------------------------------
# 0.2. Dice loss function
# ------------------------------
# class DiceLoss(nn.Module):
#     def __init__(self, smooth=1.0):
#         super().__init__()
#         self.smooth = smooth

#     def forward(self, logits, true, eps=1e-7):
#         """
#         logits: [B, C, H, W] - raw output from model
#         true: [B, H, W] - ground truth class indices
#         """
#         num_classes = logits.shape[1]
#         true_one_hot = F.one_hot(true, num_classes=num_classes)  # [B, H, W, C]
#         true_one_hot = true_one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]

#         probs = F.softmax(logits, dim=1)  # [B, C, H, W]

#         dims = (0, 2, 3)  # sum over batch and spatial dimensions
#         intersection = torch.sum(probs * true_one_hot, dims)
#         cardinality = torch.sum(probs + true_one_hot, dims)
#         dice_loss = 1.0 - ((2. * intersection + self.smooth) / (cardinality + self.smooth))
#         return dice_loss.mean()
# ------------------------------
# 1. Dataset
# ------------------------------
class FlameDataset(Dataset):
    def __init__(self, root, split="train", transform=None):
        self.root = root
        self.split = split
        self.transform = transform

        image_dir = os.path.join(root, split, "images")
        mask_dir = os.path.join(root, split, "masks")

        # filter out hidden files like .DS_Store
        self.images = sorted([f for f in os.listdir(image_dir) if not f.startswith('.')])
        self.masks = sorted([f for f in os.listdir(mask_dir) if not f.startswith('.')])
        #self.images = sorted(os.listdir(os.path.join(root, split, "images")))  # add sorted to ensure deterministic pairing and
        #self.masks = sorted(os.listdir(os.path.join(root, split, "masks")))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.split, "images", self.images[idx])
        mask_path = os.path.join(self.root, self.split, "masks", self.masks[idx])

        # Debug log (only for first few samples to avoid spam)
        if idx < 5:
            print(f"[{self.split}] Loading {img_path} and {mask_path}")

        # convert .tif images and masks to tensor - convert images to RGB
        img_greyscale = Image.open(img_path).convert("L")  # replicate grayscale -> RGB
        img_tensor_greyscale = T.ToTensor()(img_greyscale)
        img_rgb = img_tensor_greyscale.repeat(3, 1, 1) 
        mask = Image.open(mask_path).convert("L")  # assumes single-channel mask
        mask= T.ToTensor()(mask)

        if self.transform:
            image = self.transform(img_rgb)
       #mask = torch.as_tensor(np.array(mask), dtype=torch.long)

        return image, mask
# ------------------------------
# 2. Model with ResNet50 + FPN
# ------------------------------
class FPN_Segmentation(nn.Module):
    def __init__(self, num_classes, pretrain_type="imagenet", pretrain_path=None):
        super().__init__()

        # Load resnet50 backbone
        if pretrain_type == "imagenet":
            print("Using supervised ImageNet weights")
            resnet = resnet50(weights="IMAGENET1K_V1")
        else:
            print(f"Using custom ResNet50 with {pretrain_type} weights")
            resnet = resnet50(weights=None)

            if pretrain_type in ["simclr", "moco", "swav"]:
                assert pretrain_path is not None, f"--pretrain_path must be given for {pretrain_type}"
                pretrain_path = os.path.join(pretrain_path, pretrain_type + ".pth.tar")
                checkpoint = torch.load(pretrain_path, map_location="cpu")

                # VISSL checkpoints are nested
                if "classy_state_dict" in checkpoint:
                    state_dict = checkpoint["classy_state_dict"]["base_model"]["model"]
                else:
                    state_dict = checkpoint

                # Strip prefixes
                new_state_dict = {}
                for k, v in state_dict.items():
                    k = k.replace("module.", "")
                    if k.startswith("resnet."):
                        k = k[len("resnet."):]
                    new_state_dict[k] = v

                missing, unexpected = resnet.load_state_dict(new_state_dict, strict=False)
                print("Missing keys:", missing[:5], "..." if len(missing) > 5 else "")
                print("Unexpected keys:", unexpected[:5], "..." if len(unexpected) > 5 else "")

            elif pretrain_type == "none":
                print("Training from scratch")
        

        # Use layers as backbone features
        # resnet50 architecture and if the layers are named correctly
        self.body = nn.ModuleDict({
            "layer1": nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1),
            "layer2": resnet.layer2,
            "layer3": resnet.layer3,
            "layer4": resnet.layer4,
        })

        # Feature Pyramid Network
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=256
        )

        # Segmentation head: 1x1 conv + upsampling
        self.head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # Extract backbone features
        feats = {}
        for name, layer in self.body.items():
            x = layer(x)
            feats[name] = x

        # FPN output
        fpn_outs = self.fpn(feats)

        # Use highest resolution output from FPN (e.g. "layer1")
        out = fpn_outs["layer1"]

        # Segmentation head
        out = self.head(out)

        # Upsample to input size
        out = F.interpolate(out, scale_factor=4, mode="bilinear", align_corners=False)
        return out

# ------------------------------
# 3. Training loop
# ------------------------------
import numpy as np

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_pixels = 0

    train_loop = tqdm(dataloader,leave=False, desc="Training", dynamic_ncols=True, ascii=True)
    for imgs, masks in train_loop:
        imgs, masks = imgs.to(device), masks.to(device)
        masks = masks.squeeze(1).long()
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # ---------- Compute pixel-wise accuracy ----------
        preds = outputs.argmax(dim=1)  # [B, H, W]
        total_correct += (preds == masks).sum().item()
        total_pixels += masks.numel()

        train_loop.set_postfix(loss=loss.item())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_pixels
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_pixels = 0

    val_loop = tqdm(dataloader, leave=False, desc="Validation", dynamic_ncols=True, ascii=True)
    with torch.no_grad():
        for imgs, masks in val_loop:
            imgs, masks = imgs.to(device), masks.to(device)
            masks = masks.squeeze(1).long()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            total_loss += loss.item()

            # ---------- Compute pixel-wise accuracy ----------
            preds = outputs.argmax(dim=1)        # [B, H, W]
            total_correct += (preds == masks).sum().item()
            total_pixels += masks.numel()

            val_loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_pixels
    return avg_loss, accuracy


# ------------------------------
# 4. Putting it together
# ------------------------------
if __name__ == "__main__":
    
    args_list = [
    {"name": "--data_root", "type": str, "required": True},
    {"name": "--epochs", "type": int, "default": 20},
    {"name": "--batch_size", "type": int, "default": 4},
    {"name": "--lr", "type": float, "default": 1e-3},
    {"name": "--num_classes", "type": int, "default": 2},
    {"name": "--pretrain_type", "type": str, "default": "imagenet",
     "choices": ["none", "imagenet", "simclr", "moco", "swav"],
     "help": "Which pretrained weights to use"},
    {"name": "--pretrain_path", "type": str, "default": None,
     "help": "Path to self-supervised VISSL weights (.pth)"},
    {"name": "--log_csv", "type": str, "default": "training_log.csv"},
    {"name": "--log_tb", "type": str, "default": "runs/experiment1"}]

    parser = argparse.ArgumentParser()
    for arg in args_list:
        name = arg.pop("name")
        parser.add_argument(name, **arg)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on a {device}")

    mean, std = compute_dataset_stats(args.data_root, split="train")
    # transforms
    transform = T.Compose([
        #T.Resize((256, 256)),
        #T.ToTensor(), # we do this seperately for the images and mask 
        T.Normalize(mean=[mean], std=[std])
    ])

    # datasets
    train_ds = FlameDataset(args.data_root, split="train", transform=transform)
    val_ds = FlameDataset(args.data_root, split="val", transform=transform)
    test_ds = FlameDataset(args.data_root, split="test", transform=transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)

    # model
    model = FPN_Segmentation(num_classes=args.num_classes, 
                             pretrain_type=args.pretrain_type, 
                             pretrain_path=args.pretrain_path).to(device)
    # count model's trainable parameters 
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params:,}")
    # loss + optimizer
    #criterion = nn.CrossEntropyLoss() 
    #criterion = nn.BCEWithLogitsLoss()
    criterion = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999)) #adam optimizer

    # ------------------------------
    # Setup logging
    # ------------------------------
    # CSV logger
    csv_file = open(args.log_csv, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Epoch", "Train Loss", "Val Loss"])  # header

    # TensorBoard logger
    writer = SummaryWriter(args.log_tb)

    # ------------------------------
    # Train loop
    # ------------------------------
    for epoch in range(args.epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}: train loss={train_loss:.4f}, train acc={train_acc:.4f}, val loss={val_loss:.4f}, val acc={val_acc:.4f}")

        # Log to CSV
        csv_writer.writerow([epoch+1, train_loss, train_acc, val_loss, val_acc])
        csv_file.flush()  # ensure it's written to disk

        # Log to TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch+1)
        writer.add_scalar("Loss/val", val_loss, epoch+1)
        writer.add_scalar("Accuracy/train", train_acc, epoch+1)
        writer.add_scalar("Accuracy/val", val_acc, epoch+1)
    
    print("Training finished. Visualizing sample predictions on validation set...")
    visualize_predictions(model, test_ds, device, num_samples=3)

    # Close loggers
    csv_file.close()
    writer.close()



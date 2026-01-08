import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import copy
from dataclasses import dataclass
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from PIL import Image, ImageFile
from torch.utils.tensorboard import SummaryWriter
import warnings

# Suppress minor warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# Handle truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    from model_defs import PetDataset, get_model
except ImportError:
    raise ImportError("Could not import 'model_defs.py'.")

# =========================================================
# CONFIGURATION
# =========================================================
@dataclass
class Config:
    BASE_PATH: str = r"E:\Project\cat_dog\data"
    CAT_DIR: str = os.path.join(BASE_PATH, "Cat")
    DOG_DIR: str = os.path.join(BASE_PATH, "Dog")
    LOG_DIR: str = "runs"
    
    IMG_SIZE: int = 224
    BATCH_SIZE: int = 32
    EPOCHS: int = 30
    LR: float = 1e-4
    THRESHOLD: float = 0.5
    PATIENCE: int = 5
    SEED: int = 42
    NUM_WORKERS: int = 4
    
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = Config()

# =========================================================
# UTILITIES
# =========================================================
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def collate_fn_filter_none(batch):
    """Filters out corrupt images (None) from the batch."""
    batch = list(filter(lambda x: x[0] is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def get_file_paths(cat_dir, dog_dir):
    if not os.path.exists(cat_dir) or not os.path.exists(dog_dir):
        print(f"⚠️  Error: Directories not found.")
        return [], []

    cat_files = [os.path.join(cat_dir, f) for f in os.listdir(cat_dir) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
    dog_files = [os.path.join(dog_dir, f) for f in os.listdir(dog_dir) if f.lower().endswith(('jpg', 'jpeg', 'png'))]

    files = cat_files + dog_files
    labels = [0] * len(cat_files) + [1] * len(dog_files)
    
    print(f"📊 Dataset Stats: Cats: {len(cat_files)} | Dogs: {len(dog_files)} | Total: {len(files)}")
    return np.array(files), np.array(labels)

# =========================================================
# ENGINE: TRAIN & VAL LOOPS
# =========================================================
def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    
    for images, labels in pbar:
        images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE).unsqueeze(1)
        
        optimizer.zero_grad()
        
        with autocast(device_type="cuda", enabled=(cfg.DEVICE.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Metrics
        running_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(logits)
        preds = (probs >= cfg.THRESHOLD).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix(loss=loss.item())

    return running_loss / total, correct / total

def validate_one_epoch(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE).unsqueeze(1)
            
            logits = model(images)
            loss = criterion(logits, labels)
            
            running_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(logits)
            preds = (probs >= cfg.THRESHOLD).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    return running_loss / total, correct / total

# =========================================================
# MANAGER
# =========================================================
def run_training(model_name, train_loader, val_loader):
    print(f"\n🚀 Starting Training: {model_name}")
    
    # TensorBoard Writer
    writer = SummaryWriter(log_dir=os.path.join(cfg.LOG_DIR, model_name))
    
    model = get_model(model_name, cfg.DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.3)
    scaler = GradScaler(enabled=(cfg.DEVICE.type == "cuda"))

    best_loss = float('inf')
    best_weights = copy.deepcopy(model.state_dict())
    early_stop_count = 0

    for epoch in range(cfg.EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion)
        
        # Update Scheduler
        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Logging
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Val', val_acc, epoch)

        print(f"Epoch {epoch+1}/{cfg.EPOCHS} | "
              f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
              f"Acc: {train_acc:.4f}/{val_acc:.4f} | "
              f"LR: {prev_lr:.1e} -> {current_lr:.1e}")

        # Checkpoint & Early Stopping
        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, f"{model_name}_best.pth")
            early_stop_count = 0
            print(f"   ✅ New Best Model Saved (Loss: {best_loss:.4f})")
        else:
            early_stop_count += 1
            print(f"   ⏳ No improvement. Patience: {early_stop_count}/{cfg.PATIENCE}")

        if early_stop_count >= cfg.PATIENCE:
            print("🛑 Early stopping triggered.")
            break
            
    writer.close()
    print(f"🏁 Finished {model_name}. Best Val Loss: {best_loss:.4f}\n")

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    seed_everything(cfg.SEED)
    
    print("\n================ SYSTEM CHECK ================")
    print(f"Device: {cfg.DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("==============================================\n")

    files, labels = get_file_paths(cfg.CAT_DIR, cfg.DOG_DIR)

    if len(files) == 0:
        exit()

    X_train, X_val, y_train, y_val = train_test_split(files, labels, test_size=0.2, stratify=labels, random_state=cfg.SEED)

    train_tf = transforms.Compose([
        transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_tf = transforms.Compose([
        transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Note: Added collate_fn to dataloaders to handle corrupt images
    train_loader = DataLoader(
        PetDataset(X_train, y_train, train_tf), 
        batch_size=cfg.BATCH_SIZE, 
        shuffle=True, 
        num_workers=cfg.NUM_WORKERS,
        collate_fn=collate_fn_filter_none,
        pin_memory=True
    )

    val_loader = DataLoader(
        PetDataset(X_val, y_val, val_tf), 
        batch_size=cfg.BATCH_SIZE, 
        shuffle=False, 
        num_workers=cfg.NUM_WORKERS,
        collate_fn=collate_fn_filter_none,
        pin_memory=True
    )

    models_list = ["custom_cnn", "mobilenet_v2", "resnet18", "efficientnet_b0"]
    
    for model_name in models_list:
        run_training(model_name, train_loader, val_loader)
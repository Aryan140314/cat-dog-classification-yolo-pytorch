import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
from PIL import ImageFile
from tqdm import tqdm

# Ensure model_defs is accessible
try:
    from model_defs import PetDataset, get_model
except ImportError:
    raise ImportError("Ensure 'model_defs.py' is in the same directory.")

# =========================================================
# CONFIG
# =========================================================
BASE_PATH = r"E:\Project\cat_dog\data"
CAT_DIR = os.path.join(BASE_PATH, "Cat")
DOG_DIR = os.path.join(BASE_PATH, "Dog")

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
THRESHOLD = 0.5
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ImageFile.LOAD_TRUNCATED_IMAGES = True

# =========================================================
# OUTPUT DIRECTORIES
# =========================================================
OUTPUT_DIR = "evaluation_outputs"
CM_DIR = os.path.join(OUTPUT_DIR, "confusion_matrices")
ROC_DIR = os.path.join(OUTPUT_DIR, "roc_curves")

os.makedirs(CM_DIR, exist_ok=True)
os.makedirs(ROC_DIR, exist_ok=True)

# =========================================================
# DATASET UTILS
# =========================================================
def get_file_paths(cat_dir, dog_dir):
    if not os.path.exists(cat_dir) or not os.path.exists(dog_dir):
        print("⚠️ Data directories not found!")
        return [], []

    cat_files = [os.path.join(cat_dir, f) for f in os.listdir(cat_dir)
                 if f.lower().endswith(("jpg", "jpeg", "png"))]
    dog_files = [os.path.join(dog_dir, f) for f in os.listdir(dog_dir)
                 if f.lower().endswith(("jpg", "jpeg", "png"))]

    files = cat_files + dog_files
    labels = [0] * len(cat_files) + [1] * len(dog_files)
    return np.array(files), np.array(labels)

def collate_fn_filter_none(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

# =========================================================
# LOAD VALIDATION DATA
# =========================================================
print("Preparing Validation Data...")
files, labels = get_file_paths(CAT_DIR, DOG_DIR)

if len(files) == 0:
    print("No files found. Exiting.")
    exit()

# ⚠️ CRITICAL: Match train_test_split from train.py
_, X_val, _, y_val = train_test_split(
    files, 
    labels, 
    test_size=0.2, 
    stratify=labels, 
    random_state=SEED
)

print(f"Total Validation Images: {len(X_val)}")

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_loader = DataLoader(
    PetDataset(X_val, y_val, val_transform),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn_filter_none,
    pin_memory=True
)

# =========================================================
# EVALUATION ENGINE
# =========================================================
def evaluate_model(model_name):
    print(f"\n🔍 Evaluating {model_name}...")
    
    weight_path = f"{model_name}_best.pth"
    if not os.path.exists(weight_path):
        print(f"⚠️  Weights not found: {weight_path}. Skipping.")
        return None

    model = get_model(model_name, DEVICE)
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    model.eval()

    y_true, y_probs = [], []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Testing {model_name}"):
            images = images.to(DEVICE)
            
            logits = model(images)
            probs = torch.sigmoid(logits).squeeze()

            if probs.ndim == 0:
                probs = probs.unsqueeze(0)

            y_probs.extend(probs.cpu().numpy())
            y_true.extend(labels.numpy())

    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    y_pred = (y_probs >= THRESHOLD).astype(int)

    # ================= METRICS =================
    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_probs)
    except ValueError:
        auc = 0.5 

    # 1. Generate Dictionary for CSV
    report_dict = classification_report(y_true, y_pred, target_names=["Cat", "Dog"], output_dict=True)
    
    # 2. Generate String for Text File
    report_str = classification_report(y_true, y_pred, target_names=["Cat", "Dog"])
    
    # Save Text Report
    with open(os.path.join(OUTPUT_DIR, "classification_reports.txt"), "a") as f:
        f.write(f"\n{'='*30}\nMODEL: {model_name}\n{'='*30}\n")
        f.write(report_str)
        f.write("\n")

    print(f"   Accuracy : {acc:.4f}")
    print(f"   Cat F1   : {report_dict['Cat']['f1-score']:.4f}")
    print(f"   Dog F1   : {report_dict['Dog']['f1-score']:.4f}")

    # ================= PLOTS =================
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Cat", "Dog"], yticklabels=["Cat", "Dog"])
    plt.title(f"Confusion Matrix — {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(CM_DIR, f"{model_name}_confusion.png"))
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.title(f"ROC Curve — {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ROC_DIR, f"{model_name}_roc.png"))
    plt.close()

    # Return expanded metrics for CSV
    return {
        "Model": model_name,
        "Accuracy": acc,
        "AUC": auc,
        "Macro_F1": report_dict['macro avg']['f1-score'],
        "Cat_Precision": report_dict['Cat']['precision'],
        "Cat_Recall": report_dict['Cat']['recall'],
        "Cat_F1": report_dict['Cat']['f1-score'],
        "Dog_Precision": report_dict['Dog']['precision'],
        "Dog_Recall": report_dict['Dog']['recall'],
        "Dog_F1": report_dict['Dog']['f1-score']
    }

# =========================================================
# RUN EVALUATION
# =========================================================
if __name__ == "__main__":
    # Reset text report file
    open(os.path.join(OUTPUT_DIR, "classification_reports.txt"), "w").close()

    models = [
        "custom_cnn",
        "mobilenet_v2",
        "resnet18",
        "efficientnet_b0"
    ]

    results = []
    for model_name in models:
        res = evaluate_model(model_name)
        if res:
            results.append(res)

    if results:
        df = pd.DataFrame(results)
        
        # Reorder columns for readability
        cols = ["Model", "Accuracy", "AUC", "Macro_F1", 
                "Cat_Precision", "Cat_Recall", "Cat_F1", 
                "Dog_Precision", "Dog_Recall", "Dog_F1"]
        df = df[cols]
        
        csv_path = os.path.join(OUTPUT_DIR, "metrics_summary.csv")
        df.to_csv(csv_path, index=False)

        print("\n✅ Evaluation Complete!")
        print(f"📁 Summary CSV: {csv_path}")
        print(f"📄 Full Reports: {os.path.join(OUTPUT_DIR, 'classification_reports.txt')}")
        print(f"📊 Graphs saved in: {OUTPUT_DIR}/")
    else:
        print("\n❌ No models were evaluated.")
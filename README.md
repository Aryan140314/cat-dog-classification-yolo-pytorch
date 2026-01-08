```md
# 🐶🐱 Cat vs Dog Classification with YOLOv8 (PyTorch)

An end-to-end **Computer Vision project** for **Cat vs Dog image classification** using multiple **CNN architectures in PyTorch**, combined with **YOLOv8-based pet detection** and a **Streamlit web application**.

This repository includes:
- Training code
- Evaluation results (confusion matrix, ROC, metrics)
- Trained model weights
- Streamlit deployment app

---

## 📁 Project Structure

```

cat_dog/
├── app/
│   └── app.py                     # Streamlit application
│
├── scripts/
│   ├── train.py                   # Model training script
│   ├── evaluate.py                # Evaluation & metrics generation
│   └── model_defs.py              # Model architectures & dataset class
│
├── Models/                        # Trained model weights (.pth)
│   ├── custom_cnn_best.pth
│   ├── mobilenet_v2_best.pth
│   ├── resnet18_best.pth
│   └── efficientnet_b0_best.pth
│
├── evaluation_outputs/            # Evaluation results
│   ├── confusion_matrices/
│   ├── roc_curves/
│   ├── metrics_summary.csv
│   └── classification_reports.txt
│
├── data/                          # Dataset (Cats & Dogs)
│
├── requirements.txt
├── README.md
└── .gitignore

```

---

## 🧠 Models Implemented

- **Custom CNN** (from scratch with residual blocks)
- **MobileNetV2** (transfer learning)
- **ResNet18** (transfer learning)
- **EfficientNet-B0** (transfer learning)

All models are trained for **binary classification (Cat vs Dog)** using **BCEWithLogitsLoss**.

---

## 🔍 YOLOv8 Integration

- Uses **YOLOv8 Nano** for object detection
- Detects and counts **Cats 🐱 and Dogs 🐶**
- Integrated directly into the Streamlit dashboard
- Adjustable confidence threshold for detection sensitivity

---

## 📊 Evaluation Metrics

Evaluation is performed using a held-out validation set.

Generated metrics include:
- Accuracy
- Precision, Recall, F1-score
- ROC-AUC
- Confusion Matrix
- ROC Curve

All results are stored in:
```

evaluation_outputs/

````

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Aryan140314/cat-dog-classification-yolo-pytorch.git
cd cat-dog-classification-yolo-pytorch
````

---

### 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv .venv
.venv\Scripts\activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Run the Streamlit App

```bash
streamlit run app/app.py
```

---

## 🏋️ Training Models (Optional)

To retrain models from scratch:

```bash
python scripts/train.py
```

Trained models will be saved inside:

```
Models/
```

---

## 📈 Evaluate Models

```bash
python scripts/evaluate.py
```

Evaluation results will be saved inside:

```
evaluation_outputs/
```

---

## 🖥️ Streamlit App Features

* Upload an image for prediction
* Single-model inference
* Multi-model benchmarking
* Confidence & latency comparison
* YOLO-based pet detection and counting
* Interactive charts and tables

---

## 🧪 Tech Stack

* Python
* PyTorch
* TorchVision
* Streamlit
* YOLOv8 (Ultralytics)
* OpenCV
* NumPy, Pandas
* Scikit-learn
* Matplotlib, Seaborn, Plotly

---

## 🎯 Use Cases

* Computer Vision learning project
* Deep Learning portfolio project
* Academic / final-year project
* Resume & placement demonstrations

---

## ⚠️ Notes

* Model weights (`.pth`) are included in this repository.
* Large files may increase clone time.
* GPU is recommended for training, but inference works on CPU.

---

## 👤 Author

**Aryan Singh**
GitHub: [https://github.com/Aryan140314](https://github.com/Aryan140314)

---

## 📜 License

This project is licensed under the **MIT License**.

```

---

### ✅ What this README gives you
- Professional structure
- Recruiter-friendly
- Academic-ready
- Clear execution steps
- Proper documentation of YOLO + CNN + evaluation

If you want, next I can:
- simplify it for **college submission**
- make a **short README for recruiters**
- add **screenshots / GIF section**
- convert this into a **final-year project report**

Just tell me 👍
```


🐶🐱 Cat vs Dog Classification with YOLOv8 (PyTorch)

An end-to-end **Computer Vision project** for **Cat vs Dog image classification** using multiple **CNN architectures in PyTorch**, combined with **YOLOv8-based pet detection** and a **Streamlit web application**.

This repository includes:
- Training code
- Evaluation results (confusion matrix, ROC, metrics)
- Trained model weights
- Streamlit deployment app


🧠 Models Implemented

Custom CNN (from scratch with residual blocks)
MobileNetV2 (transfer learning)
ResNet18 (transfer learning)
EfficientNet-B0 (transfer learning)

All models are trained for **binary classification (Cat vs Dog)** using **BCEWithLogitsLoss**.



 🔍 YOLOv8 Integration

- Uses **YOLOv8 Nano** for object detection
- Detects and counts **Cats 🐱 and Dogs 🐶**
- Integrated directly into the Streamlit dashboard
- Adjustable confidence threshold for detection sensitivity



📊 Evaluation Metrics

Evaluation is performed using a held-out validation set.

Generated metrics include:
- Accuracy
- Precision, Recall, F1-score
- ROC-AUC
- Confusion Matrix
- ROC Curve

All results are stored in:

evaluation_outputs/

## 🖥️ Streamlit App Features

* Upload an image for prediction
* Single-model inference
* Multi-model benchmarking
* Confidence & latency comparison
* YOLO-based pet detection and counting
* Interactive charts and tables



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


```

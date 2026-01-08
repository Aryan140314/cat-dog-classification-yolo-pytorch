import sys
import os

sys.path.append(
os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "scripts")
    )
)

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import time
import cv2
import pandas as pd
import plotly.express as px
from ultralytics import YOLO


# Import your model definitions
try:
    from model_defs import get_model
except ImportError:
    st.error("⚠️ Critical Error: Could not import 'model_defs.py'.")
    st.stop()

# =========================================================
# CONFIG & PAGE SETUP
# =========================================================
st.set_page_config(
    page_title="Pet AI Vision Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Styling
st.markdown("""
    <style>
    .main { background-color: #f4f6f9; }
    .stMetric { background-color: white; padding: 15px; border-radius: 8px; border-left: 5px solid #4CAF50; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    h1, h2, h3 { color: #2c3e50; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: white; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    .stTabs [aria-selected="true"] { background-color: #e3f2fd; border-bottom: 2px solid #2196F3; }
    </style>
    """, unsafe_allow_html=True)

MODEL_NAMES = ["custom_cnn", "mobilenet_v2", "resnet18", "efficientnet_b0"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224

# =========================================================
# CACHED MODEL LOADERS
# =========================================================
@st.cache_resource
def load_classifier(model_name):
    """Loads a specific classification model."""
    weights_path = os.path.join(
    r"E:\Project\cat_dog\Models",
    f"{model_name}_best.pth"
)

    if not os.path.exists(weights_path):
        return None, f"Missing weights: {weights_path}"
    
    try:
        model = get_model(model_name, DEVICE)
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        model.eval()
        return model, None
    except Exception as e:
        return None, str(e)

@st.cache_resource
def load_yolo():
    """Loads YOLOv8 Nano model."""
    return YOLO('yolov8n.pt')

# =========================================================
# INFERENCE ENGINES
# =========================================================
def run_classification(model, image):
    """Runs binary classification (Cat vs Dog)."""
    # Preprocess
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Inference
    start_t = time.time()
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.sigmoid(logits).item()
    inference_time = (time.time() - start_t) * 1000

    # Logic
    if probs >= 0.5:
        label = "DOG"
        conf = probs
        emoji = "🐶"
    else:
        label = "CAT"
        conf = 1 - probs
        emoji = "🐱"

    return label, conf, emoji, inference_time

def run_yolo_detection(image, conf_threshold=0.25):
    """Runs object detection for Cats and Dogs."""
    model = load_yolo()
    results = model(image, conf=conf_threshold)
    result = results[0]

    detections = []
    cats, dogs = 0, 0
    
    # 15 = Cat, 16 = Dog in COCO dataset
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        
        if cls_id == 15:
            cats += 1
            detections.append({"Type": "Cat", "Confidence": conf})
        elif cls_id == 16:
            dogs += 1
            detections.append({"Type": "Dog", "Confidence": conf})

    # Generate Annotated Image
    annotated_img = result.plot()
    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    
    return annotated_img, cats, dogs, detections

# =========================================================
# UI: SIDEBAR
# =========================================================
with st.sidebar:
    st.title("🎛️ Settings")
    st.caption(f"Hardware Acceleration: **{DEVICE}**")
    
    st.divider()
    st.markdown("### 📊 Benchmark Settings")
    strict_mode = st.toggle("Strict Mode (90% Threshold)", value=True, help="Only accept predictions with >90% confidence.")
    
    st.markdown("### 👁️ YOLO Settings")
    yolo_conf = st.slider("Detection Sensitivity", 0.1, 1.0, 0.4, 0.05, help="Lower value detects more objects but may include errors.")

# =========================================================
# UI: MAIN CONTENT
# =========================================================
st.title("🧠 Pet Vision AI Dashboard")

# TAB LAYOUT
tab1, tab2 = st.tabs(["🔍 Breed Classification", "🔢 Pet Counter (YOLO)"])

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Load Image once
    image = Image.open(uploaded_file).convert("RGB")

    # =====================================================
    # TAB 1: CLASSIFICATION (Single & Benchmark)
    # =====================================================
    with tab1:
        c1, c2 = st.columns([1, 1.5])
        with c1:
            st.image(image, caption="Source Image", use_container_width=True)
        
        with c2:
            mode = st.radio("Analysis Mode:", ["Single Model Analysis", "Benchmark All Models"], horizontal=True)
            st.divider()

            if mode == "Single Model Analysis":
                # SINGLE MODEL LOGIC
                selected_model = st.selectbox("Choose Classifier", MODEL_NAMES, index=2)
                model, err = load_classifier(selected_model)
                
                if not model:
                    st.error(f"❌ {err}")
                else:
                    lbl, conf, emj, t = run_classification(model, image)
                    
                    # Strict Mode Check
                    if strict_mode and conf < 0.90:
                        st.warning(f"⚠️ **Uncertain Prediction**")
                        st.info(f"Model ({selected_model}) is only **{conf:.1%}** sure. (Threshold: 90%)")
                    else:
                        st.success(f"### It's a **{lbl}** {emj}")
                        st.metric("Confidence", f"{conf:.2%}", delta=f"{t:.1f}ms latency")
                        st.progress(conf)

            else:
                # BENCHMARK LOGIC
                st.subheader("🏆 Model Comparison")
                
                results_data = []
                progress = st.progress(0)
                
                for i, name in enumerate(MODEL_NAMES):
                    model, err = load_classifier(name)
                    if model:
                        lbl, conf, emj, t = run_classification(model, image)
                        # Normalize confidence: If Dog, keep as is. If Cat, invert for visualization (optional)
                        # Here we just show raw confidence for the predicted class
                        results_data.append({
                            "Model": name,
                            "Prediction": lbl,
                            "Confidence": conf,
                            "Latency (ms)": t
                        })
                    progress.progress((i + 1) / len(MODEL_NAMES))
                
                # Convert to DataFrame
                df = pd.DataFrame(results_data)
                
                # 1. Plotly Chart
                fig = px.bar(
                    df, x="Model", y="Confidence", 
                    color="Prediction", 
                    text_auto='.2%',
                    range_y=[0, 1.1],
                    title="Confidence Scores per Model",
                    color_discrete_map={"DOG": "#EF553B", "CAT": "#636EFA"}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 2. Detailed Table
                st.dataframe(
                    df.style.highlight_max(axis=0, subset=["Confidence"], color="#d4edda"), 
                    use_container_width=True
                )

    # =====================================================
    # TAB 2: YOLO COUNTING
    # =====================================================
    with tab2:
        col_y1, col_y2 = st.columns([1.5, 1])
        
        with st.spinner("Running YOLOv8 Object Detection..."):
            ann_img, n_cats, n_dogs, det_list = run_yolo_detection(image, conf_threshold=yolo_conf)
        
        with col_y1:
            st.image(ann_img, caption="AI Vision Output", use_container_width=True)
            
        with col_y2:
            st.subheader("Detection Stats")
            
            # Key Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Pets", n_cats + n_dogs)
            m2.metric("Cats 🐱", n_cats)
            m3.metric("Dogs 🐶", n_dogs)
            
            st.divider()
            
            # Detailed List
            if det_list:
                st.write("**Detailed Confidence Report:**")
                det_df = pd.DataFrame(det_list)
                # Format confidence as percentage
                det_df["Confidence"] = det_df["Confidence"].apply(lambda x: f"{x:.1%}")
                st.dataframe(det_df, use_container_width=True)
            else:
                st.warning("No animals detected at this sensitivity level.")
                st.info("💡 Try lowering the 'Detection Sensitivity' in the sidebar.")

else:
    # Empty State
    st.info("👈 Please upload an image to start the analysis.")
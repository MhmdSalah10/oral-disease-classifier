import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import plotly.express as px
import gdown

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

st.set_page_config(
    page_title="6-Class Image Classifier",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.0rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #475569;
        margin-bottom: 1.25rem;
    }
    .prediction-card {
        background: #EFF6FF;
        border-left: 5px solid #2563EB;
        padding: 1.25rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .pred-label {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1E40AF;
        margin: 0.2rem 0;
    }
    .conf-badge {
        display: inline-block;
        background-color: #1D4ED8;
        color: white;
        padding: 0.3rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.9rem;
        margin-top: 0.35rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

MODEL_PATH = Path("best_model.keras")
METADATA_PATH = Path("model_metadata.json")
DRIVE_MODEL_URL = "https://drive.google.com/uc?export=download&id=1GsTgzh5wDQEZw26bHxDu4wWvNNO83Fq5"


@st.cache_resource(show_spinner="Downloading model from Google Drive and loading resources...")
def load_classification_resources():
    metadata = {
        "class_names": ["Calculus", "Caries", "Gingivitis", "Hypodontia", "Mouth Ulcer", "Tooth Discoloration"],
        "img_size": [224, 224],
        "best_model_name": "EfficientNetB0",
        "best_model_filename": "best_model.keras",
        "preprocessing_note": "Raw RGB [0, 255] float32 tensor of shape (224, 224, 3).",
        "test_metrics": {
            "accuracy": None,
            "macro_precision": None,
            "macro_recall": None,
            "macro_f1": None,
            "weighted_f1": None,
        },
    }

    if METADATA_PATH.exists():
        try:
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                metadata.update(json.load(f))
        except Exception as e:
            st.sidebar.warning(f"Could not read model_metadata.json: {e}")

    # تحميل الموديل من جوجل درايف أوتوماتيكياً لو مش موجود محلياً
    if not MODEL_PATH.exists() and TF_AVAILABLE:
        try:
            st.info("Downloading model weights from Google Drive, please wait...")
            gdown.download(DRIVE_MODEL_URL, str(MODEL_PATH), quiet=False)
        except Exception as e:
            st.error(f"Failed to download model from Google Drive: {e}")

    model = None
    load_error = None
    if not TF_AVAILABLE:
        load_error = "TensorFlow is not installed in this environment."
    elif not MODEL_PATH.exists():
        load_error = f"Model file {MODEL_PATH} not found."
    else:
        try:
            model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
        except Exception as e:
            load_error = f"Error loading {MODEL_PATH}: {e}"

    return model, metadata, load_error


model, metadata, load_error = load_classification_resources()
class_names = metadata.get("class_names", [])
img_size = tuple(metadata.get("img_size", [224, 224]))
test_metrics = metadata.get("test_metrics", {})

with st.sidebar:
    st.title("Project Info")
    st.subheader("Selected Model")
    st.info(f"Architecture: {metadata.get('best_model_name', 'N/A')}")

    macro_f1_val = test_metrics.get("macro_f1")
    acc_val = test_metrics.get("accuracy")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric(
            "Test Macro F1",
            f"{macro_f1_val * 100:.2f}%" if isinstance(macro_f1_val, (int, float)) else "N/A",
        )
    with col_s2:
        st.metric(
            "Test Accuracy",
            f"{acc_val * 100:.2f}%" if isinstance(acc_val, (int, float)) else "N/A",
        )

    st.subheader(f"Supported Classes ({len(class_names)})")
    for idx, cname in enumerate(class_names):
        st.markdown(f"- `[{idx}]` {cname}")

    st.subheader("Input Specification")
    st.caption(metadata.get("preprocessing_note", "Raw RGB [0, 255] input."))
    st.caption(f"Resolution: {img_size[0]} x {img_size[1]} x 3 (RGB)")

st.markdown('<div class="main-header">Multi-Class Image Classification</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Upload an image to classify it across the supported categories.</div>',
    unsafe_allow_html=True,
)

if load_error:
    st.warning(f"Model Status: {load_error}")

uploaded_file = st.file_uploader(
    "Upload an image (JPG, JPEG, PNG, WEBP)",
    type=["jpg", "jpeg", "png", "webp"],
)


def preprocess_uploaded_image(pil_img: Image.Image, target_size: tuple) -> np.ndarray:
    img_rgb = pil_img.convert("RGB")
    img_resized = img_rgb.resize((target_size[1], target_size[0]), Image.Resampling.BILINEAR)
    img_array = np.asarray(img_resized, dtype=np.float32)
    return np.expand_dims(img_array, axis=0)


if uploaded_file is not None:
    try:
        pil_image = Image.open(uploaded_file)
    except Exception as e:
        st.error(f"Could not read the uploaded image: {e}")
        st.stop()

    col_left, col_right = st.columns([1, 1.35], gap="large")

    with col_left:
        st.subheader("Uploaded Image")
        st.image(pil_image, caption=uploaded_file.name, use_container_width=True)
        st.caption(f"Original: {pil_image.size[0]}x{pil_image.size[1]} ({pil_image.mode})")

    with col_right:
        st.subheader("Prediction Results")
        input_tensor = preprocess_uploaded_image(pil_image, img_size)

        if model is not None:
            with st.spinner("Running inference..."):
                preds = model.predict(input_tensor, verbose=0)[0]
                probs = np.asarray(preds, dtype=np.float64)
                if not np.isclose(np.sum(probs), 1.0, atol=1e-2):
                    exp_p = np.exp(probs - np.max(probs))
                    probs = exp_p / np.sum(exp_p)
        else:
            st.info("Demo Mode: best_model.keras is not loaded yet.")
            rng = np.random.default_rng(abs(hash(uploaded_file.name)) % (2**32))
            probs = rng.dirichlet(np.ones(len(class_names)))

        top_idx = int(np.argmax(probs))
        top_class = class_names[top_idx] if top_idx < len(class_names) else f"Class {top_idx}"
        top_conf = float(probs[top_idx]) * 100.0

        st.markdown(
            f"""
            <div class="prediction-card">
                <div style="font-size:0.85rem; color:#475569; text-transform:uppercase; font-weight:600;">
                    Predicted Class
                </div>
                <div class="pred-label">{top_class}</div>
                <div class="conf-badge">Confidence: {top_conf:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        df_probs = pd.DataFrame(
            {
                "Class": class_names,
                "Probability (%)": [round(float(p) * 100.0, 2) for p in probs],
            }
        ).sort_values(by="Probability (%)", ascending=True)

        fig = px.bar(
            df_probs,
            x="Probability (%)",
            y="Class",
            orientation="h",
            text="Probability (%)",
            color="Probability (%)",
            color_continuous_scale="Blues",
            range_x=[0, 105],
        )
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig.update_layout(
            height=340,
            margin=dict(l=10, r=30, t=20, b=10),
            xaxis_title="Probability (%)",
            yaxis_title="",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("View Probability Table"):
            st.dataframe(
                df_probs.sort_values(by="Probability (%)", ascending=False).reset_index(drop=True),
                use_container_width=True,
            )
else:
    st.info("Upload a JPG, JPEG, PNG, or WEBP image above to classify.")
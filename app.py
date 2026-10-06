import streamlit as st
import tensorflow as tf
import joblib
import numpy as np
from PIL import Image
import os

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Brain Tumor MRI Classification",
    page_icon="🧠",
    layout="centered"
)

# =====================================================
# PROJECT SETTINGS
# =====================================================

MODEL_PATH = "best_brain_tumor_model.joblib"

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "no_tumor",
    "pituitary"
]

IMG_SIZE = (128, 128)

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    return model


# =====================================================
# TITLE
# =====================================================

st.title("🧠 Brain Tumor MRI Image Classification")

st.write(
    "Upload a brain MRI image to predict the tumor category."
)

st.info(
    "This application is for educational/project purposes "
    "and should not be used as a medical diagnosis."
)

# =====================================================
# LOAD SAVED MODEL
# =====================================================

try:
    model = load_model()
    st.success("Model loaded successfully.")
except Exception as e:
    st.error("Unable to load the model.")
    st.error(str(e))
    st.stop()

# =====================================================
# IMAGE UPLOAD
# =====================================================

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

# =====================================================
# PREDICTION
# =====================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded MRI Image")

    st.image(
        image,
        caption="Uploaded MRI",
        use_container_width=True
    )

    # Resize
    resized_image = image.resize(IMG_SIZE)

    # Convert to NumPy
    image_array = np.array(resized_image).astype("float32")

    # Normalize
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Prediction
    prediction = model.predict(
        image_array,
        verbose=0
    )

    predicted_index = np.argmax(prediction[0])

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = prediction[0][predicted_index] * 100

    # =================================================
    # RESULT
    # =================================================

    st.subheader("Prediction Result")

    st.success(
        f"Predicted Class: {predicted_class.replace('_', ' ').title()}"
    )

    st.metric(
        "Confidence",
        f"{confidence:.2f}%"
    )

    # =================================================
    # ALL CLASS PROBABILITIES
    # =================================================

    st.subheader("Class Probabilities")

    for i, class_name in enumerate(CLASS_NAMES):

        probability = prediction[0][i] * 100

        st.write(
            f"{class_name.replace('_', ' ').title()}: "
            f"{probability:.2f}%"
        )

        st.progress(
            float(prediction[0][i])
        )
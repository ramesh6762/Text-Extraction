import streamlit as st
from PIL import Image
import numpy as np
import easyocr
import cv2

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Advanced OCR App", layout="wide")
st.title("🧠 Advanced Text Recognition (EasyOCR + Preprocessing)")

# -----------------------------
# Load EasyOCR
# -----------------------------
@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'], gpu=False)

reader = load_reader()

# -----------------------------
# Image Preprocessing Function
# -----------------------------
def preprocess_image(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Contrast Enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)

    # Denoise
    denoise = cv2.fastNlMeansDenoising(contrast, None, 30, 7, 21)

    # Sharpen
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    sharpen = cv2.filter2D(denoise, -1, kernel)

    return sharpen

# -----------------------------
# Upload Image
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((800, int(800 * image.height / image.width)))

    image_np = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, width="stretch")

    if st.button("🚀 Extract Text"):
        with st.spinner("Processing..."):

            # -----------------------------
            # Preprocess
            # -----------------------------
            processed = preprocess_image(image_np)

            # -----------------------------
            # Multi-pass OCR
            # -----------------------------
            results = []

            # Original
            results += reader.readtext(
                image_np,
                detail=1,
                paragraph=True,
                contrast_ths=0.05,
                text_threshold=0.6
            )

            # Processed
            results += reader.readtext(
                processed,
                detail=1,
                paragraph=True,
                contrast_ths=0.05,
                text_threshold=0.6
            )

            # Rotations (for vertical text)
            for angle in [90, 180, 270]:
                rotated = np.rot90(image_np, k=angle // 90)
                results += reader.readtext(rotated)

            # -----------------------------
            # Draw Results
            # -----------------------------
            annotated = image_np.copy()
            extracted_text = []

            for (bbox, text, conf) in results:
                if conf < 0.4:  # filter weak detections
                    continue

                top_left = tuple(map(int, bbox[0]))
                bottom_right = tuple(map(int, bbox[2]))

                cv2.rectangle(annotated, top_left, bottom_right, (0, 255, 0), 2)

                cv2.putText(
                    annotated,
                    f"{text} ({conf:.2f})",
                    (top_left[0], max(top_left[1] - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 255),
                    2
                )

                extracted_text.append((text, conf))

            # -----------------------------
            # Show Output
            # -----------------------------
            with col2:
                st.subheader("Detected Text")
                st.image(annotated, width="stretch")

            st.subheader("📄 Extracted Text")

            if extracted_text:
                unique_texts = list(set([t for t, _ in extracted_text]))

                for i, text in enumerate(unique_texts, 1):
                    st.write(f"{i}. {text}")

                st.download_button(
                    "⬇ Download Text",
                    data="\n".join(unique_texts),
                    file_name="text.txt"
                )
            else:
                st.warning("No text detected.")

import streamlit as st
from PIL import Image
import numpy as np
import easyocr
import cv2
 
# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="OCR App", layout="wide")
st.title("📝 Text Recognition (EasyOCR)")
 
# -----------------------------
# Load EasyOCR (Cached)
# -----------------------------
@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'], gpu=False)
 
reader = load_reader()
 
# -----------------------------
# Upload Image
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["png", "jpg", "jpeg"],
    key="upload1"
)
 
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
 
    # 🔥 Resize (prevents memory crash on cloud)
    image = image.resize((800, int(800 * image.height / image.width)))
 
    col1, col2 = st.columns(2)
 
    with col1:
        st.subheader("Original Image")
        st.image(image, width="stretch")
 
    # -----------------------------
    # OCR Button
    # -----------------------------
    if st.button("🚀 Extract Text"):
        with st.spinner("Running OCR..."):
            try:
                image_np = np.array(image)
 
                # Run OCR
                results = reader.readtext(image_np)
 
                # Draw boxes
                annotated = image_np.copy()
 
                extracted_text = []
 
                for (bbox, text, conf) in results:
                    top_left = tuple(map(int, bbox[0]))
                    bottom_right = tuple(map(int, bbox[2]))
 
                    # Draw rectangle
                    cv2.rectangle(annotated, top_left, bottom_right, (0, 255, 0), 2)
 
                    # Label
                    label = f"{text} ({conf:.0%})"
                    cv2.putText(
                        annotated,
                        label,
                        (top_left[0], max(top_left[1] - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (238, 130, 238),
                        2
                    )
 
                    extracted_text.append((text, conf))
 
                # -----------------------------
                # Show Annotated Image
                # -----------------------------
                with col2:
                    st.subheader("Detected Text")
                    st.image(annotated, width="stretch")
 
                # -----------------------------
                # Show Text Output
                # -----------------------------
                st.subheader("📄 Extracted Text")
 
                if extracted_text:
                    for i, (text, conf) in enumerate(extracted_text, 1):
                        color = "green" if conf > 0.75 else ("orange" if conf > 0.5 else "red")
                        st.markdown(
                            f"**{i}.** `{text}` — <span style='color:{color}'>{conf:.2f}</span>",
                            unsafe_allow_html=True
                        )
 
                    # Download button
                    full_text = "\n".join([t for t, _ in extracted_text])
                    st.download_button(
                        "⬇ Download Text",
                        data=full_text,
                        file_name="extracted_text.txt",
                        mime="text/plain"
                    )
                else:
                    st.warning("No text detected.")
 
            except Exception as e:
                st.error(f"❌ Error occurred: {e}")

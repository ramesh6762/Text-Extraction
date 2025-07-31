import streamlit as st
from PIL import Image
import numpy as np
import easyocr
import cv2

# Page config
st.set_page_config(page_title="OCR App")
st.title("Text Recognition")

# Upload image
uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"], key="upload1")

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_container_width=True)

    if st.button("Extract Text"):
        with st.spinner("Running OCR..."):
            reader = easyocr.Reader(['en'], gpu=False)
            result = reader.readtext(np.array(image))

            image_np = np.array(image)
            image_cv = image_np.copy()

            # Draw boxes and labels
            for (bbox, text, conf) in result:
                top_left = tuple(map(int, bbox[0]))
                bottom_right = tuple(map(int, bbox[2]))
                cv2.rectangle(image_cv, top_left, bottom_right, (0, 255, 0), 2)
                cv2.putText(image_cv, text, (top_left[0], top_left[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (238, 130, 238), 2)

        # Show result image
        st.subheader("Detected Text with Bounding Boxes:")
        st.image(image_cv, caption="Detected Text", use_container_width=True)

        # Show raw text
        st.subheader("Extracted Text:")
        for i, (_, text, conf) in enumerate(result):
            st.markdown(f"**{i+1}.** `{text}` (Confidence: `{conf:.2f}`)")

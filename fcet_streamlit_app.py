
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io

def fcet_contrast_enhancement(image: np.ndarray, alpha: float = 0.8) -> np.ndarray:
    L = 256
    g_k = np.zeros(L, dtype=np.int32)
    for value in image.ravel():
        g_k[value] += 1

    d_k = g_k / image.size
    C_k = np.cumsum(d_k)
    phi_k = C_k / C_k.max()

    identity = np.linspace(0, 1, L)
    T_k = alpha * phi_k + (1 - alpha) * identity

    T_k_scaled = (T_k * (L - 1)).astype(np.uint8)
    enhanced_image = T_k_scaled[image]

    return enhanced_image

def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image

st.set_page_config(page_title="FCET Image Enhancer", layout="wide")
st.title("🖼️ Feature-Preserving Contrast Enhancement Transform (FCET)")

uploaded_file = st.file_uploader("Upload an image (JPG, PNG, BMP, TIFF)", type=["jpg", "jpeg", "png", "bmp", "tiff"])

alpha = st.slider("Adjust Contrast Parameter α", min_value=0.0, max_value=1.0, value=0.8, step=0.01)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    grayscale_img = convert_to_grayscale(image_np)

    enhanced_img = fcet_contrast_enhancement(grayscale_img, alpha)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Grayscale Image")
        st.image(grayscale_img, channels="GRAY", use_column_width=True)

    with col2:
        st.subheader("FCET Enhanced Image")
        st.image(enhanced_img, channels="GRAY", use_column_width=True)

    buf = io.BytesIO()
    enhanced_pil = Image.fromarray(enhanced_img)
    enhanced_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button("📥 Download Enhanced Image", data=byte_im, file_name="fcet_enhanced.png", mime="image/png")

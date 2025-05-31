import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Apply Seaborn styling
sns.set_style("darkgrid")

# Core FCET Enhancement Function
def fcet_contrast_enhancement(image: np.ndarray, alpha: float = 0.8) -> tuple:
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

    return enhanced_image, g_k, d_k, T_k_scaled

def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image

# Streamlit UI
st.set_page_config(page_title="🔬 FCET Image Enhancer", layout="wide")
st.title("🖼️ Feature-Preserving Contrast Enhancement Transform (FCET)")
st.markdown("""
Welcome to the FCET Enhancer! This tool applies a contrast enhancement algorithm 
that preserves natural image features, ideal for grayscale image analysis (e.g., X-rays, MRI scans).
""")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png", "bmp", "tiff"])
alpha = st.slider("🎚️ Adjust Contrast Parameter α", min_value=0.0, max_value=1.0, value=0.8, step=0.01)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    grayscale_img = convert_to_grayscale(image_np)

    enhanced_img, g_k, d_k, T_k_scaled = fcet_contrast_enhancement(grayscale_img, alpha)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🖼️ Original Grayscale Image")
        st.image(grayscale_img, channels="GRAY", use_column_width=True)

    with col2:
        st.subheader("✨ FCET Enhanced Image")
        st.image(enhanced_img, channels="GRAY", use_column_width=True)

    # Download Option
    buf = io.BytesIO()
    enhanced_pil = Image.fromarray(enhanced_img)
    enhanced_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button("📥 Download Enhanced Image", data=byte_im, file_name="fcet_enhanced.png", mime="image/png")

    st.markdown("---")
    st.subheader("📊 Grey Level Distribution")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(g_k, label="Grey Level Density gₖ", color='blue')
    ax.set_title("Histogram of Grey Levels (gₖ)", fontsize=14)
    ax.set_xlabel("Grey Level k")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)

    st.subheader("📈 Normalized Density dₖ and Transformation Tₖ")

    df = pd.DataFrame({
        "Grey Level (k)": np.arange(256),
        "Density gₖ": g_k,
        "Normalized dₖ": d_k,
        "Transformed Tₖ": T_k_scaled
    })

    st.dataframe(df.head(20), use_container_width=True)  # Preview table

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(d_k, label="Normalized dₖ", color='green')
    ax2.plot(T_k_scaled / 255.0, label="Tₖ Scaled", color='orange')
    ax2.set_title("Contrast Transformation vs Normalized Density")
    ax2.set_xlabel("Grey Level k")
    ax2.set_ylabel("Value")
    ax2.legend()
    st.pyplot(fig2)

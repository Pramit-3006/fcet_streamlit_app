import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

# ---------- STYLING ----------
st.set_page_config(page_title="⚡ FCET Image Enhancer", layout="wide")
st.markdown(
    """
    <style>
    body {
        font-family: 'Segoe UI', sans-serif;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        background-color: #0E1117;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: bold;
    }
    .stSlider {
        padding: 10px 20px;
    }
    .stDownloadButton>button {
        background-color: #0096FF;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 25px;
    }
    .element-container:has(> .stImage) {
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
        padding: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- FCET LOGIC ----------
def fcet_contrast_enhancement(image: np.ndarray, alpha: float = 0.8):
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

    return enhanced_image, g_k, d_k, C_k, T_k

def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image

# ---------- SIDEBAR ----------
st.sidebar.title("⚙️ Controls")
uploaded_file = st.sidebar.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png", "bmp", "tiff"])
alpha = st.sidebar.slider("🎛️ Contrast Parameter α", 0.0, 1.0, 0.8, 0.01)

st.sidebar.markdown("---")
st.sidebar.markdown("✨ Developed with ❤️ for next-gen imaging")
st.sidebar.markdown("Made by **Pramit Pradhan**")

# ---------- MAIN INTERFACE ----------
st.title("🎨 FCET — Feature-Preserving Contrast Enhancement Transform")
st.markdown("Experience **AI-enhanced contrast** with precision tuning, analysis, and smooth visualization.")

if uploaded_file is not None:
    with st.spinner("🧠 Processing image..."):
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        grayscale_img = convert_to_grayscale(image_np)
        enhanced_img, g_k, d_k, C_k, T_k = fcet_contrast_enhancement(grayscale_img, alpha)

    st.markdown("## 🔍 Image Comparison")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original (Grayscale)**")
        st.image(grayscale_img, use_column_width=True, channels="GRAY", caption="Original Image")

    with col2:
        st.markdown("**Enhanced with FCET**")
        st.image(enhanced_img, use_column_width=True, channels="GRAY", caption="Enhanced Image")

    st.markdown("## 📊 Histogram Analysis")
    style.use('seaborn-darkgrid')
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(g_k, label='Original Histogram', color='skyblue')
    ax.plot(np.bincount(enhanced_img.ravel(), minlength=256), label='Enhanced Histogram', color='orange')
    ax.set_title("Histogram of Pixel Intensities", fontsize=14)
    ax.set_xlabel("Gray Level (0-255)")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)

    st.markdown("## 🧾 Detailed FCET Data")
    df = pd.DataFrame({
        "Gray Level (k)": np.arange(256),
        "g_k (Frequency)": g_k,
        "d_k (Density)": d_k,
        "C_k (Cumulative)": C_k,
        "T_k (Transform Fn)": T_k
    })
    st.dataframe(
        df.style.background_gradient(cmap="viridis", axis=0).format({
            "d_k (Density)": "{:.6f}",
            "C_k (Cumulative)": "{:.6f}",
            "T_k (Transform Fn)": "{:.6f}"
        }),
        height=400
    )

    # Download button
    buf = io.BytesIO()
    Image.fromarray(enhanced_img).save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button("📥 Download FCET Enhanced Image", data=byte_im, file_name="fcet_enhanced.png", mime="image/png")
else:
    st.info("👈 Upload an image to begin FCET transformation.")

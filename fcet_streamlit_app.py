import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io

sns.set_style("darkgrid")

# Multilingual support
def get_translation(language):
    translations = {
        "English": {
            "title": "🧠 FCET | Feature-Preserving Contrast Enhancement Transform",
            "purpose": "An AI-powered visual contrast enhancement tool ideal for grayscale MRI & CT scans.",
            "ideal_for": "Perfect for biomedical engineers, radiologists, and researchers.",
            "upload": "📥 Upload a grayscale medical image",
            "processing": "🔍 Processing the image using FCET algorithm...",
            "alpha_slider": "🎚️ Select alpha (Contrast Sensitivity)",
            "original_subheader": "Original Image",
            "enhanced_subheader": "Enhanced Image",
            "download": "⬇️ Download Enhanced Image",
            "histogram_title": "Histogram Comparison",
            "density_table": "gₖ, dₖ, Tₖ Transformation Table",
            "density_plot_title": "📈 Normalized Density dₖ vs Transformation Tₖ",
            "footer": "Crafted with ❤️ using Streamlit | MRI Sample by open source"
        },
        # Additional languages can be added here...
    }
    return translations.get(language, translations["English"])

# FCET Enhancement Logic
def fcet_contrast_enhancement(image, alpha=0.8):
    g_k = np.bincount(image.flatten(), minlength=256)
    total_pixels = image.size
    d_k = g_k / total_pixels
    T_k = 255 * (np.power(d_k, alpha) / np.max(np.power(d_k, alpha)))
    T_k_scaled = np.round(T_k).astype(np.uint8)
    enhanced_image = T_k_scaled[image]
    return enhanced_image, g_k, d_k, T_k_scaled

def convert_to_grayscale(image):
    if len(image.shape) == 3:
        return np.dot(image[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    return image

st.set_page_config(page_title="FCET MRI Enhancer", layout="wide")

st.markdown("""
<style>
    .css-1v0mbdj { max-width: 2000px; }
    .block-container { padding: 1rem 2rem 2rem 2rem; }
</style>
""", unsafe_allow_html=True)

# Language Selection
language = st.sidebar.selectbox("🌐 Select Language", ["English"])
t = get_translation(language)

# UI Elements
st.title(t["title"])
st.markdown(t["purpose"])
st.markdown(t["ideal_for"])
st.markdown("---")

uploaded_file = st.file_uploader(t["upload"], type=["jpg", "jpeg", "png", "bmp", "tiff"])

if uploaded_file is not None:
    st.markdown(t["processing"])
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    gray_image = convert_to_grayscale(image_np)

    alpha = st.slider(t["alpha_slider"], min_value=0.0, max_value=1.0, value=0.8, step=0.01)
    enhanced_image, g_k, d_k, T_k_scaled = fcet_contrast_enhancement(gray_image, alpha)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(t["original_subheader"])
        st.image(gray_image, clamp=True, use_column_width=True)

    with col2:
        st.subheader(t["enhanced_subheader"])
        st.image(enhanced_image, clamp=True, use_column_width=True)

    # Download
    enhanced_pil = Image.fromarray(enhanced_image)
    buf = io.BytesIO()
    enhanced_pil.save(buf, format="PNG")
    st.download_button(label=t["download"], data=buf.getvalue(), file_name="enhanced_fcet.png", mime="image/png")

    # Histogram plot
    st.markdown("## " + t["histogram_title"])
    fig, ax = plt.subplots()
    ax.hist(gray_image.ravel(), bins=256, color='gray', alpha=0.5, label='Original')
    ax.hist(enhanced_image.ravel(), bins=256, color='blue', alpha=0.5, label='Enhanced')
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)

    # Table
    st.markdown("## " + t["density_table"])
    df = pd.DataFrame({
        "Gray Level k": np.arange(256),
        "gₖ (Count)": g_k,
        "dₖ (Density)": d_k.round(6),
        "Tₖ": T_k_scaled
    })
    st.dataframe(df, height=300)

    # Density vs Tₖ Plot
    st.markdown("## " + t["density_plot_title"])
    fig2, ax2 = plt.subplots()
    ax2.plot(d_k, label="dₖ (Normalized Density)", color="black")
    ax2.plot(T_k_scaled / 255.0, label="Tₖ / 255", color="green")
    ax2.set_xlabel("Gray Level")
    ax2.set_ylabel("Value")
    ax2.legend()
    st.pyplot(fig2)

    st.markdown("---")
    st.markdown(f"#### {t['footer']}")

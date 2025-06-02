import streamlit as st
from PIL import Image
import numpy as np
from streamlit_image_comparison import image_comparison
from streamlit_drawable_canvas import st_canvas
import io

st.set_page_config(layout="wide")
st.title("FCET MRI Image Enhancement")

# Theme Toggle
theme = st.selectbox("Choose Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown("""
        <style>
        body { background-color: #1e1e1e; color: white; }
        .stApp { background-color: #1e1e1e; }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        body { background-color: white; color: black; }
        .stApp { background-color: white; }
        </style>
    """, unsafe_allow_html=True)

# Image Uploader
uploaded_file = st.file_uploader("Upload an MRI Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")
    image_resized = image.resize((512, 512))
    st.image(image_resized, caption="Preview (Downsampled)", use_column_width=True)

    # Side-by-side Comparison (dummy enhancement)
    enhanced_image = image_resized.point(lambda p: p * 1.2)  # Placeholder for FCET output

    st.subheader("Image Comparison")
    image_comparison(
        img1=image_resized,
        img2=enhanced_image,
        label1="Original",
        label2="Enhanced",
        width=700,
    )

    # ROI Selector using Sliders
    st.subheader("ROI Selector (Manual)")
    x = st.slider("X Coordinate", 0, 512, 100)
    y = st.slider("Y Coordinate", 0, 512, 100)
    w = st.slider("Width", 10, 256, 100)
    h = st.slider("Height", 10, 256, 100)

    roi = image_resized.crop((x, y, x + w, y + h))
    st.image(roi, caption="Selected ROI", use_column_width=True)

    # Optional enhancement on ROI
    enhanced_roi = roi.point(lambda p: p * 1.2)  # Placeholder for FCET
    st.image(enhanced_roi, caption="Enhanced ROI", use_column_width=True)

    # ROI Selector using Drawable Canvas
    st.subheader("Draw ROI with Mouse")
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=2,
        background_image=image_resized,
        update_streamlit=True,
        height=512,
        width=512,
        drawing_mode="rect",
    )

    if canvas_result.json_data:
        for obj in canvas_result.json_data["objects"]:
            left = int(obj["left"])
            top = int(obj["top"])
            width = int(obj["width"])
            height = int(obj["height"])
            drawn_roi = image_resized.crop((left, top, left + width, top + height))
            st.image(drawn_roi, caption="Drawn ROI", use_column_width=True)
            enhanced_drawn_roi = drawn_roi.point(lambda p: p * 1.2)
            st.image(enhanced_drawn_roi, caption="Enhanced Drawn ROI", use_column_width=True)
else:
    st.info("Upload an image to begin.")

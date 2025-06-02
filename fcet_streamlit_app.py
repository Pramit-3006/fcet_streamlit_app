import streamlit as st
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Alternative to OpenCV: Replace cv2 with PIL for grayscale conversion
def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        # Convert RGB to grayscale using PIL
        pil_image = Image.fromarray(image)
        gray_image = pil_image.convert("L")
        return np.array(gray_image)
    return image

# FCET Enhancement Function remains unchanged
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


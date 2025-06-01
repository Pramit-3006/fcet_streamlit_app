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

# Multilanguage dictionary for UI text
LANGUAGES = {
    "en": {
        "title": "🖼️✨ Feature-Preserving Contrast Enhancement Transform (FCET)",
        "purpose": "🎯 **Purpose:** Enhance medical and grayscale images with a human-friendly contrast technique.",
        "ideal_for": "💡 **Ideal for:** MRI, CT Scans, or any X-ray based imagery that requires fine detail preservation.",
        "adjust_alpha": "🔧 Adjust the alpha value to balance enhancement strength.",
        "upload": "📤 Upload an Image (JPG/PNG/BMP/TIFF)",
        "alpha_slider": "🎚️ Adjust Contrast Parameter α (0 = Original, 1 = Max Enhance)",
        "original_subheader": "🔍 Original Grayscale Image",
        "enhanced_subheader": "🌟 FCET Enhanced Image",
        "download": "📥 Download Enhanced Image",
        "histogram": "📊 Histogram: Grey Level Distribution",
        "histogram_title": "📉 Grey Level Histogram",
        "density_table": "📈 Density dₖ & Transformation Tₖ Table",
        "density_plot_title": "📈 Normalized Density vs Transformation Function",
        "processing": "📽️ Processing image, please wait...",
        "compare": "🧪 Compare Original & Enhanced",
        "language": "🌐 Select Language",
        "footer": "🚀 Developed with ❤️ for advanced grayscale image analysis.\n📬 For feedback, reach out at: `your-email@example.com`"
    },
    "es": {
        "title": "🖼️✨ Transformación de Mejora de Contraste Preservando Características (FCET)",
        "purpose": "🎯 **Propósito:** Mejorar imágenes médicas y en escala de grises con una técnica amigable de contraste.",
        "ideal_for": "💡 **Ideal para:** MRI, Tomografías o cualquier imagen de rayos X que requiera preservación de detalles finos.",
        "adjust_alpha": "🔧 Ajusta el valor alfa para balancear la intensidad de mejora.",
        "upload": "📤 Subir una imagen (JPG/PNG/BMP/TIFF)",
        "alpha_slider": "🎚️ Ajustar parámetro de contraste α (0 = Original, 1 = Máxima mejora)",
        "original_subheader": "🔍 Imagen en escala de grises original",
        "enhanced_subheader": "🌟 Imagen mejorada con FCET",
        "download": "📥 Descargar imagen mejorada",
        "histogram": "📊 Histograma: Distribución de niveles de gris",
        "histogram_title": "📉 Histograma de niveles de gris",
        "density_table": "📈 Tabla de densidad dₖ y transformación Tₖ",
        "density_plot_title": "📈 Densidad normalizada vs función de transformación",
        "processing": "📽️ Procesando imagen, por favor espere...",
        "compare": "🧪 Comparar imagen original y mejorada",
        "language": "🌐 Seleccionar idioma",
        "footer": "🚀 Desarrollado con ❤️ para análisis avanzado de imágenes en escala de grises.\n📬 Para comentarios, contáctanos en: `your-email@example.com`"
    },
    "hi": {
        "title": "🖼️✨ फीचर-प्रिज़र्विंग कंट्रास्ट एन्हांसमेंट ट्रांसफॉर्म (FCET)",
        "purpose": "🎯 **उद्देश्य:** चिकित्सा और ग्रेस्केल छवियों को मानव-मित्रवत कंट्रास्ट तकनीक से बेहतर बनाएं।",
        "ideal_for": "💡 **उपयुक्त:** MRI, CT स्कैन, या किसी भी एक्स-रे आधारित छवि के लिए जो सूक्ष्म विवरण संरक्षण की आवश्यकता हो।",
        "adjust_alpha": "🔧 एन्हांसमेंट की ताकत संतुलित करने के लिए अल्फा मान समायोजित करें।",
        "upload": "📤 एक छवि अपलोड करें (JPG/PNG/BMP/TIFF)",
        "alpha_slider": "🎚️ कंट्रास्ट पैरामीटर α समायोजित करें (0 = मूल, 1 = अधिकतम एन्हांस)",
        "original_subheader": "🔍 मूल ग्रेस्केल छवि",
        "enhanced_subheader": "🌟 FCET द्वारा संवर्धित छवि",
        "download": "📥 संवर्धित छवि डाउनलोड करें",
        "histogram": "📊 हिस्टोग्राम: ग्रे स्तर वितरण",
        "histogram_title": "📉 ग्रे स्तर हिस्टोग्राम",
        "density_table": "📈 डेंसिटी dₖ और ट्रांसफॉर्मेशन Tₖ तालिका",
        "density_plot_title": "📈 सामान्यीकृत डेंसिटी बनाम ट्रांसफॉर्मेशन फ़ंक्शन",
        "processing": "📽️ छवि संसाधित की जा रही है, कृपया प्रतीक्षा करें...",
        "compare": "🧪 मूल और संवर्धित छवि की तुलना करें",
        "language": "🌐 भाषा चुनें",
        "footer": "🚀 उन्नत ग्रेस्केल छवि विश्लेषण के लिए ❤️ के साथ विकसित।\n📬 प्रतिक्रिया के लिए संपर्क करें: `your-email@example.com`"
    },
    "ja": {
        "title": "🖼️✨ 特徴保持型コントラスト強調変換 (FCET)",
        "purpose": "🎯 **目的:** 医療用およびグレースケール画像を人に優しいコントラスト技術で強調します。",
        "ideal_for": "💡 **対象:** MRI、CTスキャン、または微細なディテール保持が必要なX線画像。",
        "adjust_alpha": "🔧 強調の強さを調整するためにアルファ値を調整してください。",
        "upload": "📤 画像をアップロード (JPG/PNG/BMP/TIFF)",
        "alpha_slider": "🎚️ コントラストパラメータ α を調整 (0 = 元画像, 1 = 最大強調)",
        "original_subheader": "🔍 元のグレースケール画像",
        "enhanced_subheader": "🌟 FCET 強調画像",
        "download": "📥 強調画像をダウンロード",
        "histogram": "📊 ヒストグラム: グレーレベル分布",
        "histogram_title": "📉 グレーレベルヒストグラム",
        "density_table": "📈 密度 dₖ と変換関数 Tₖ の表",
        "density_plot_title": "📈 正規化密度と変換関数の比較",
        "processing": "📽️ 画像を処理中です。お待ちください...",
        "compare": "🧪 元画像と強調画像を比較",
        "language": "🌐 言語を選択",
        "footer": "🚀 高度なグレースケール画像解析のために❤️で開発されました。\n📬 フィードバックはこちらへ: `your-email@example.com`"
    },
    "de": {
        "title": "🖼️✨ Merkmals-erhaltende Kontrastverstärkungstransformation (FCET)",
        "purpose": "🎯 **Zweck:** Verbesserung medizinischer und Graustufenbilder mit einer benutzerfreundlichen Kontrasttechnik.",
        "ideal_for": "💡 **Ideal für:** MRT, CT-Scans oder Röntgenbilder, die eine feine Detailerhaltung erfordern.",
        "adjust_alpha": "🔧 Passen Sie den Alpha-Wert an, um die Verstärkungsstärke zu balancieren.",
        "upload": "📤 Bild hochladen (JPG/PNG/BMP/TIFF)",
        "alpha_slider": "🎚️ Kontrastparameter α einstellen (0 = Original, 1 = Maximale Verstärkung)",
        "original_subheader": "🔍 Original Graustufenbild",
        "enhanced_subheader": "🌟 FCET verstärktes Bild",
        "download": "📥 Verstärktes Bild herunterladen",
        "histogram": "📊 Histogramm: Graustufenverteilung",
        "histogram_title": "📉 Graustufen-Histogramm",
        "density_table": "📈 Dichte dₖ & Transformation Tₖ Tabelle",
        "density_plot_title": "📈 Normalisierte Dichte vs Transformationsfunktion",
        "processing": "📽️ Bild wird verarbeitet, bitte warten...",
        "compare": "🧪 Original und verstärkt vergleichen",
        "language": "🌐 Sprache auswählen",
        "footer": "🚀 Entwickelt mit ❤️ für fortschrittliche Graustufenbildanalyse.\n📬 Für Feedback kontaktieren Sie: `your-email@example.com`"
    },
    "fr": {
        "title": "🖼️✨ Transformation d’Amélioration du Contraste Préservant les Caractéristiques (FCET)",
        "purpose": "🎯 **But:** Améliorer les images médicales et en niveaux de gris avec une technique de contraste conviviale.",
        "ideal_for": "💡 **Idéal pour:** IRM, scanners CT, ou toute image aux rayons X nécessitant une préservation fine des détails.",
        "adjust_alpha": "🔧 Ajustez la valeur alpha pour équilibrer la force de l’amélioration.",
        "upload": "📤 Téléchargez une image (JPG/PNG/BMP/TIFF)",
        "alpha_slider": "🎚️ Ajuster le paramètre de contraste α (0 = Original, 1 = Amélioration maximale)",
        "original_subheader": "🔍 Image en niveaux de gris originale",
        "enhanced_subheader": "🌟 Image améliorée par FCET",
        "download": "📥 Télécharger l’image améliorée",
        "histogram": "📊 Histogramme : Distribution des niveaux de gris",
        "histogram_title": "📉 Histogramme des niveaux de gris",
        "density_table": "📈 Tableau de densité dₖ & transformation Tₖ",
        "density_plot_title": "📈 Densité normalisée vs fonction de transformation",
        "processing": "📽️ Traitement de l’image, veuillez patienter...",
        "compare": "🧪 Comparer l’original et l’amélioré",
        "language": "🌐 Sélectionnez la langue",
        "footer": "🚀 Développé avec ❤️ pour l’analyse avancée des images en niveaux de gris.\n📬 Pour vos retours, contactez : `your-email@example.com`"
    }
}

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
st.set_page_config(page_title="🔬 FCET Image Enhancer", layout="wide", page_icon="🧠")

# Sidebar: Language selector
lang = st.sidebar.selectbox("🌐 Select Language ", options=["en", "es", "hi", "ja", "de", "fr"])
t = LANGUAGES[lang]

# Custom CSS Styling
st.markdown("""
    <style>
        .main {
            background-color: #f0f2f6;
            color: #262730;
        }
        h1, h2, h3, h4 {
            color: #36454F;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
        }
        .stSlider>div>div {
            background-color: #D1C4E9;
        }
    </style>
""", unsafe_allow_html=True)

# Title & Intro
st.title(t["title"])
st.markdown(f"""
{t['purpose']}

{t['ideal_for']}

{t['adjust_alpha']}

---
""")

# File uploader and slider
uploaded_file = st.file_uploader(t["upload"], type=["jpg", "jpeg", "png", "bmp", "tiff"])
alpha = st.slider(t["alpha_slider"], min_value=0.0, max_value=1.0, value=0.8, step=0.01)

if uploaded_file is not None:
    with st.spinner(t["processing"]):
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        grayscale_img = convert_to_grayscale(image_np)
        enhanced_img, g_k, d_k, T_k_scaled = fcet_contrast_enhancement(grayscale_img, alpha)

    # Two columns for original and enhanced images
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(t["original_subheader"])
        st.image(grayscale_img, channels="GRAY", use_column_width=True, caption=t["original_subheader"])

    with col2:
        st.subheader(t["enhanced_subheader"])
        st.image(enhanced_img, channels="GRAY", use_column_width=True, caption=t["enhanced_subheader"])

    # Image Comparison Slider with HTML/JS
    st.subheader(t["compare"])

    from streamlit.components.v1 import html
    import base64

    def pil_to_base64(img):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    original_b64 = pil_to_base64(Image.fromarray(grayscale_img))
    enhanced_b64 = pil_to_base64(Image.fromarray(enhanced_img))

    slider_html = f"""
    <style>
    .container {{
        position: relative;
        width: 100%;
        max-width: 700px;
        user-select:none;
    }}
    .image {{
        display: block;
        width: 100%;
        height: auto;
    }}
    .overlay {{
        position: absolute;
        top: 0;
        left: 0;
        width: 50%;
        overflow: hidden;
    }}
    .slider {{
        -webkit-appearance: none;
        width: 100%;
        height: 25px;
        background: #d3d3d3;
        outline: none;
        opacity: 0.7;
        transition: opacity .2s;
        margin-top: 10px;
        border-radius: 10px;
    }}
    .slider:hover {{
        opacity: 1;
    }}
    </style>
    <div class="container">
        <img src="data:image/png;base64,{original_b64}" class="image" />
        <div class="overlay" id="overlay">
            <img src="data:image/png;base64,{enhanced_b64}" class="image" />
        </div>
    </div>
    <input type="range" min="0" max="100" value="50" class="slider" id="slider" />
    <script>
    const slider = document.getElementById('slider');
    const overlay = document.getElementById('overlay');
    slider.oninput = function() {{
        overlay.style.width = this.value + '%';
    }}
    </script>
    """
    html(slider_html, height=400)

    # Download button for enhanced image
    buf = io.BytesIO()
    enhanced_pil = Image.fromarray(enhanced_img)
    enhanced_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(t["download"], data=byte_im, file_name="fcet_enhanced.png", mime="image/png")

    # Histogram and data tables
    st.markdown("---")
    st.subheader(t["histogram"])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(g_k, label="Grey Level Frequency gₖ", color='blue')
    ax.set_title(t["histogram_title"], fontsize=16, color='#2C3E50')
    ax.set_xlabel("Grey Level k")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)

    st.subheader(t["density_table"])

    df = pd.DataFrame({
        "Grey Level (k)": np.arange(256),
        "Density gₖ": g_k,
        "Normalized dₖ": d_k,
        "Transformed Tₖ": T_k_scaled
    })
    st.dataframe(df.head(20), use_container_width=True)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(d_k, label="Normalized dₖ", color='green')
    ax2.plot(T_k_scaled / 255.0, label="Scaled Tₖ", color='orange')
    ax2.set_title(t["density_plot_title"], fontsize=16)
    ax2.set_xlabel("Grey Level k")
    ax2.set_ylabel("Value")
    ax2.legend()
    st.pyplot(fig2)

    # Footer
    st.markdown("---")
    st.markdown(t["footer"])

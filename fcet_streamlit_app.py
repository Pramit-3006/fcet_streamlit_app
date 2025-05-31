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

# Language dictionary for UI texts
lang_dict = {
    "en": {
        "page_title": "🔬 FCET Image Enhancer",
        "title": "🖼️✨ Feature-Preserving Contrast Enhancement Transform (FCET)",
        "purpose": "🎯 **Purpose:** Enhance medical and grayscale images with a human-friendly contrast technique.",
        "ideal_for": "💡 **Ideal for:** MRI, CT Scans, or any X-ray based imagery that requires fine detail preservation.",
        "adjust_alpha": "🔧 Adjust the alpha value to balance enhancement strength.",
        "upload": "📤 Upload an Image (JPG/PNG/BMP/TIFF)",
        "alpha_slider": "🎚️ Adjust Contrast Parameter α (0 = Original, 1 = Max Enhance)",
        "original_img": "🔍 Original Grayscale Image",
        "enhanced_img": "🌟 FCET Enhanced Image",
        "download": "📥 Download Enhanced Image",
        "histogram": "📊 Histogram: Grey Level Distribution",
        "histogram_title": "📉 Grey Level Histogram",
        "grey_level": "Grey Level k",
        "frequency": "Frequency",
        "density_table": "📈 Density dₖ & Transformation Tₖ Table",
        "normalized_density": "Normalized dₖ",
        "transformed_Tk": "Transformed Tₖ",
        "norm_density_vs_trans": "📈 Normalized Density vs Transformation Function",
        "value": "Value",
        "footer": "🚀 Developed with ❤️ for advanced grayscale image analysis.\n📬 For feedback, reach out at: `your-email@example.com`"
    },
    "hi": {
        "page_title": "🔬 FCET छवि संवर्धक",
        "title": "🖼️✨ फीचर-प्रिज़र्विंग कंट्रास्ट संवर्धन ट्रांसफॉर्म (FCET)",
        "purpose": "🎯 **उद्देश्य:** चिकित्सा और ग्रेस्केल छवियों को मानव-अनुकूल कंट्रास्ट तकनीक से संवर्धित करें।",
        "ideal_for": "💡 **उपयुक्त:** MRI, CT स्कैन या किसी भी एक्स-रे छवि के लिए जिसमें सूक्ष्म विवरण की आवश्यकता हो।",
        "adjust_alpha": "🔧 संवर्धन शक्ति को संतुलित करने के लिए अल्फा मान समायोजित करें।",
        "upload": "📤 छवि अपलोड करें (JPG/PNG/BMP/TIFF)",
        "alpha_slider": "🎚️ कंट्रास्ट पैरामीटर α समायोजित करें (0 = मूल, 1 = अधिकतम संवर्धन)",
        "original_img": "🔍 मूल ग्रेस्केल छवि",
        "enhanced_img": "🌟 FCET संवर्धित छवि",
        "download": "📥 संवर्धित छवि डाउनलोड करें",
        "histogram": "📊 हिस्टोग्राम: ग्रे स्तर वितरण",
        "histogram_title": "📉 ग्रे स्तर हिस्टोग्राम",
        "grey_level": "ग्रे स्तर k",
        "frequency": "बारंबारता",
        "density_table": "📈 घनत्व dₖ और ट्रांसफॉर्मेशन Tₖ तालिका",
        "normalized_density": "सामान्यीकृत dₖ",
        "transformed_Tk": "परिवर्तित Tₖ",
        "norm_density_vs_trans": "📈 सामान्यीकृत घनत्व बनाम ट्रांसफॉर्मेशन फ़ंक्शन",
        "value": "मान",
        "footer": "🚀 उन्नत ग्रेस्केल छवि विश्लेषण के लिए ❤️ के साथ विकसित।\n📬 प्रतिक्रिया के लिए संपर्क करें: `your-email@example.com`"
    },
    "ja": {
        "page_title": "🔬 FCET画像エンハンサー",
        "title": "🖼️✨ 特徴保存コントラスト強調変換（FCET）",
        "purpose": "🎯 **目的:** 医療用およびグレースケール画像を人間に優しいコントラスト技術で強調します。",
        "ideal_for": "💡 **対象:** MRI、CTスキャン、または詳細な情報保持が必要なX線画像。",
        "adjust_alpha": "🔧 強調の強さを調整するためにアルファ値を調整してください。",
        "upload": "📤 画像をアップロード (JPG/PNG/BMP/TIFF)",
        "alpha_slider": "🎚️ コントラストパラメーターαを調整 (0 = 元画像, 1 = 最大強調)",
        "original_img": "🔍 元のグレースケール画像",
        "enhanced_img": "🌟 FCET強調画像",
        "download": "📥 強調画像をダウンロード",
        "histogram": "📊 ヒストグラム：グレーレベル分布",
        "histogram_title": "📉 グレーレベルヒストグラム",
        "grey_level": "グレーレベル k",
        "frequency": "頻度",
        "density_table": "📈 密度 dₖ と変換 Tₖ 表",
        "normalized_density": "正規化された dₖ",
        "transformed_Tk": "変換された Tₖ",
        "norm_density_vs_trans": "📈 正規化密度と変換関数の比較",
        "value": "値",
        "footer": "🚀 高度なグレースケール画像解析のために❤️で開発。\n📬 フィードバックは `your-email@example.com` まで"
    },
    "de": {
        "page_title": "🔬 FCET Bildverbesserer",
        "title": "🖼️✨ Merkmals-erhaltende Kontrastverbesserungstransformation (FCET)",
        "purpose": "🎯 **Zweck:** Verbesserung medizinischer und Graustufenbilder mit einer benutzerfreundlichen Kontrasttechnik.",
        "ideal_for": "💡 **Ideal für:** MRT, CT-Scans oder Röntgenbilder, die feine Details erfordern.",
        "adjust_alpha": "🔧 Passen Sie den Alpha-Wert an, um die Verstärkungsstärke zu steuern.",
        "upload": "📤 Bild hochladen (JPG/PNG/BMP/TIFF)",
        "alpha_slider": "🎚️ Kontrastparameter α einstellen (0 = Original, 1 = Maximale Verstärkung)",
        "original_img": "🔍 Original Graustufenbild",
        "enhanced_img": "🌟 FCET verbessertes Bild",
        "download": "📥 Verbessertes Bild herunterladen",
        "histogram": "📊 Histogramm: Graustufenverteilung",
        "histogram_title": "📉 Graustufen-Histogramm",
        "grey_level": "Graustufe k",
        "frequency": "Häufigkeit",
        "density_table": "📈 Dichte dₖ & Transformation Tₖ Tabelle",
        "normalized_density": "Normalisierte dₖ",
        "transformed_Tk": "Transformierte Tₖ",
        "norm_density_vs_trans": "📈 Normalisierte Dichte vs Transformationsfunktion",
        "value": "Wert",
        "footer": "🚀 Entwickelt mit ❤️ für fortschrittliche Graustufenbildanalyse.\n📬 Feedback an: `your-email@example.com`"
    },
    "fr": {
        "page_title": "🔬 Améliorateur d'image FCET",
        "title": "🖼️✨ Transformation d'amélioration du contraste préservant les caractéristiques (FCET)",
        "purpose": "🎯 **Objectif :** Améliorer les images médicales et en niveaux de gris avec une technique de contraste conviviale.",
        "ideal_for": "💡 **Idéal pour :** IRM, scanners CT, ou toute imagerie aux rayons X nécessitant une préservation fine des détails.",
        "adjust_alpha": "🔧 Ajustez la valeur alpha pour équilibrer la puissance d'amélioration.",
        "upload": "📤 Télécharger une image (JPG/PNG/BMP/TIFF)",
        "alpha_slider": "🎚️ Ajuster le paramètre de contraste α (0 = Original, 1 = Amélioration maximale)",
        "original_img": "🔍 Image en niveaux de gris originale",
        "enhanced_img": "🌟 Image améliorée FCET",
        "download": "📥 Télécharger l'image améliorée",
        "histogram": "📊 Histogramme : distribution des niveaux de gris",
        "histogram_title": "📉 Histogramme des niveaux de gris",
        "grey_level": "Niveau de gris k",
        "frequency": "Fréquence",
        "density_table": "📈 Tableau de densité dₖ et de transformation Tₖ",
        "normalized_density": "Densité normalisée dₖ",
        "transformed_Tk": "Transformation Tₖ",
        "norm_density_vs_trans": "📈 Densité normalisée vs fonction de transformation",
        "value": "Valeur",
        "footer": "🚀 Développé avec ❤️ pour une analyse avancée des images en niveaux de gris.\n📬 Pour des retours, contactez : `your-email@example.com`"
    }
}

# Core FCET Enhancement Function (same as you provided)
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
st.set_page_config(page_title=lang_dict["en"]["page_title"], layout="wide", page_icon="🧠")

# Language selection
language = st.selectbox("Select Language / भाषा चुनें / 言語を選択 / Sprache wählen / Choisir la langue", 
                        options=["English", "Hindi", "日本語", "Deutsch", "Français"],
                        index=0)

# Map selected language string to dict key
lang_map = {
    "English": "en",
    "Hindi": "hi",
    "日本語": "ja",
    "Deutsch": "de",
    "Français": "fr"
}
lang = lang_map[language]

# Apply custom CSS styles (optional)
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

# UI texts according to language
st.title(lang_dict[lang]["title"])
st.markdown(f"""
{lang_dict[lang]['purpose']}

{lang_dict[lang]['ideal_for']}

{lang_dict[lang]['adjust_alpha']}

---
""")

uploaded_file = st.file_uploader(lang_dict[lang]["upload"], type=["jpg", "jpeg", "png", "bmp", "tiff"])
alpha = st.slider(lang_dict[lang]["alpha_slider"], min_value=0.0, max_value=1.0, value=0.8, step=0.01)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    grayscale_img = convert_to_grayscale(image_np)

    enhanced_img, g_k, d_k, T_k_scaled = fcet_contrast_enhancement(grayscale_img, alpha)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(lang_dict[lang]["original_img"])
        st.image(grayscale_img, channels="GRAY", use_column_width=True, caption=lang_dict[lang]["original_img"])

    with col2:
        st.subheader(lang_dict[lang]["enhanced_img"])
        st.image(enhanced_img, channels="GRAY", use_column_width=True, caption=lang_dict[lang]["enhanced_img"])

    # Download Option
    buf = io.BytesIO()
    enhanced_pil = Image.fromarray(enhanced_img)
    enhanced_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(lang_dict[lang]["download"], data=byte_im, file_name="fcet_enhanced.png", mime="image/png")

    st.markdown("---")
    st.subheader(lang_dict[lang]["histogram"])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(g_k, label="gₖ", color='blue')
    ax.set_title(lang_dict[lang]["histogram_title"], fontsize=16, color='#2C3E50')
    ax.set_xlabel(lang_dict[lang]["grey_level"])
    ax.set_ylabel(lang_dict[lang]["frequency"])
    ax.legend()
    st.pyplot(fig)

    st.subheader(lang_dict[lang]["density_table"])

    df = pd.DataFrame({
        lang_dict[lang]["grey_level"]: np.arange(256),
        "gₖ": g_k,
        lang_dict[lang]["normalized_density"]: d_k,
        lang_dict[lang]["transformed_Tk"]: T_k_scaled
    })

    st.dataframe(df.head(20), use_container_width=True)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(d_k, label=lang_dict[lang]["normalized_density"], color='green')
    ax2.plot(T_k_scaled / 255.0, label=lang_dict[lang]["transformed_Tk"], color='orange')
    ax2.set_title(lang_dict[lang]["norm_density_vs_trans"], fontsize=16)
    ax2.set_xlabel(lang_dict[lang]["grey_level"])
    ax2.set_ylabel(lang_dict[lang]["value"])
    ax2.legend()
    st.pyplot(fig2)

    st.markdown(f"""
    ---
    {lang_dict[lang]['footer']}
    """)

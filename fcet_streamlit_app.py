import streamlit as st
import numpy as np
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
        "footer": "🚀 Developed with ❤️ for advanced grayscale image analysis.\n📬 For feedback, reach out at: pradhanpramit3006@gmail.com"
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
        "footer": "🚀 Desarrollado con ❤️ para análisis avanzado de imágenes en escala de grises.\n📬 Para comentarios, contáctanos en: pradhanpramit3006@gmail.com"
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
        "footer": "🚀 उन्नत ग्रेस्केल छवि विश्लेषण के लिए ❤️ के साथ विकसित।\n📬 प्रतिक्रिया के लिए संपर्क करें: pradhanpramit3006@gmail.com"
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
        "footer": "🚀 高度なグレースケール画像解析のために❤️で開発されました。\n📬 フィードバックはこちらへ: pradhanpramit3006@gmail.com"
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
        "footer": "🚀 Entwickelt mit ❤️ für fortschrittliche Graustufenbildanalyse.\n📬 Für Feedback kontaktieren Sie: pradhanpramit3006@gmail.com"
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
        "footer": "🚀 Développé avec ❤️ pour l’analyse avancée des images en niveaux de gris.\n📬 Pour vos retours, contactez : pradhanpramit3006@gmail.com"
    }
}


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


import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import joblib
from pathlib import Path


# Configuration de l'application
st.set_page_config(
    page_title="NeuroVision AI",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 NeuroVision AI")
st.subheader("Classification des images IRM cérébrales")

st.write(
    "Téléchargez une image IRM pour tester "
    "la prédiction du modèle CNN."
)


# Chemins des fichiers
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR.parent / "meilleur_modele.keras"
ENCODER_PATH = BASE_DIR.parent / "label_encoder.pkl"

TAILLE_IMAGE = (224, 224)

EXTENSIONS_AUTORISEES = {
    ".jpg", ".jpeg", ".png", ".bmp"
}

# Chargement du modèle et du LabelEncoder
@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Modèle introuvable : {MODEL_PATH}"
        )

    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_resource
def load_label_encoder():
    if not ENCODER_PATH.is_file():
        raise FileNotFoundError(
            f"LabelEncoder introuvable : {ENCODER_PATH}"
        )

    return joblib.load(ENCODER_PATH)


try:
    model = load_model()
    label_encoder = load_label_encoder()

except (FileNotFoundError, OSError, ValueError) as error:
    st.error(f"Erreur de chargement : {error}")
    st.stop()


# Fonction de prédiction

def predict_image(image_path):
    image_path = Path(image_path)

    # Vérifier le fichier
    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image introuvable : {image_path}"
        )

    # Vérifier l'extension
    if image_path.suffix.lower() not in EXTENSIONS_AUTORISEES:
        raise ValueError("Format d'image non autorisé.")

    # Lire l'image en couleur (BGR)
    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError("Impossible de lire cette image.")

    # BGR vers RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Redimensionner
    image = cv2.resize(
        image,
        TAILLE_IMAGE
    )

    # Normalisation [0, 1]
    image = image.astype("float32") / 255.0

    # Ajouter la dimension batch
    image = np.expand_dims(image, axis=0)

    # Prédiction
    probabilities = model.predict(
        image,
        verbose=0
    )[0]

    # Vérifier la compatibilité des classes
    if len(probabilities) != len(label_encoder.classes_):
        raise ValueError(
            "Le nombre de sorties du modèle ne correspond "
            "pas au nombre de classes du LabelEncoder."
        )

    # Index de la classe prédite
    predicted_index = int(np.argmax(probabilities))

    # Index vers nom de classe
    predicted_class = label_encoder.inverse_transform(
        [predicted_index]
    )[0]

    # Probabilité maximale
    confidence = float(probabilities[predicted_index])

    return predicted_class, confidence, probabilities


# Interface : upload de l'image

uploaded_file = st.file_uploader(
    "Choisir une image IRM",
    type=["jpg", "jpeg", "png", "bmp"]
)

# Affichage et prédiction

if uploaded_file is not None:

    st.image(
        uploaded_file,
        caption="Image IRM sélectionnée",
        use_container_width=True
    )

    if st.button(
        "Analyser l'image",
        type="primary",
        use_container_width=True
    ):
        temp_path = (
            BASE_DIR
            / f"temp_image{Path(uploaded_file.name).suffix.lower()}"
        )

        try:
            # Sauvegarder temporairement l'image
            temp_path.write_bytes(
                uploaded_file.getvalue()
            )

            # Prédire
            with st.spinner("Analyse en cours..."):
                predicted_class, confidence, probabilities = (
                    predict_image(temp_path)
                )

            # Résultat principal
            st.success(
                f"Classe prédite : {predicted_class}"
            )

            st.metric(
                label="Probabilité associée à la classe prédite",
                value=f"{confidence:.2%}"
            )

            # Probabilités de toutes les classes
            st.subheader("Probabilités par classe")

            results = {
                class_name: float(probability)
                for class_name, probability in zip(
                    label_encoder.classes_,
                    probabilities
                )
            }

            for class_name, probability in results.items():
                st.write(
                    f"**{class_name}** : {probability:.2%}"
                )

                st.progress(
                    float(np.clip(probability, 0.0, 1.0))
                )

        except (ValueError, FileNotFoundError, OSError) as error:
            st.error(str(error))

        except Exception:
            st.error(
                "Une erreur inattendue est survenue "
                "pendant la prédiction."
            )

        finally:
            # Supprimer le fichier temporaire
            if temp_path.exists():
                temp_path.unlink()


# Avertissement
st.divider()

st.caption(
    "Outil expérimental à but éducatif. "
    "Les prédictions ne constituent pas un diagnostic médical."
)
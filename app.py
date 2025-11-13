pip install openai streamlit

import streamlit as st
import pandas as pd
from openai import OpenAI
import os
import traceback


# ===============================
# CONFIGURACIÓN INICIAL
# ===============================
st.set_page_config(page_title="AI Music Recommender", layout="wide")
st.title("🎵 AI Music Recommender (FMA Dataset)")

# Cargar dataset
@st.cache_data
def load_data():
    return pd.read_csv("music_dataset.csv")

df = load_data()

# Mostrar vista previa
with st.expander("📊 Preview dataset"):
    st.dataframe(df.head(20))

# ===============================
# PREPARAR DATOS PARA EL MODELO
# ===============================
# Resumen breve para enviar al modelo
summary = (
    f"Dataset of {len(df)} songs from the Free Music Archive.\n"
    f"Genres: {', '.join(df['genre'].unique()[:10])}.\n"
    f"Features: tempo (avg {int(df['tempo'].mean())}), "
    f"energy (avg {df['energy'].mean():.2f}), "
    f"valence (avg {df['valence'].mean():.2f}), "
    f"acousticness (avg {df['acousticness'].mean():.2f}).\n"
)

# Select a few representative songs (sample)
sample_data = df.sample(50, random_state=42)[["track_name", "artist_name", "genre", "valence", "energy", "acousticness"]].to_string(index=False)


# ===============================
# INTERFAZ DE USUARIO
# ===============================
st.markdown("### 💬 Write your question about the dataset")

user_question = st.text_area(
    "For example: 'Recommend calm acoustic songs between 2010 and 2020'",
    height=100
)


st.sidebar.subheader("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")
api_key = st.sidebar.text_input("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", type="password")


if st.button("Ask the model"):
    if not api_key:
        st.warning("Please enter your OpenAI API key.")
    elif not user_question.strip():
        st.warning("Please write a question first.")
    else:
        # ===============================
        # CONSULTAR EL MODELO
        # ===============================
        client = OpenAI(api_key=api_key)

        st.info("Generating answer... please wait.")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a music data assistant. You won't be answering question that are not related to this topic. "
                        "Analyze the dataset description and answer questions using trends, genres, and features."
                    )
                },
                {
                    "role": "user",
                    "content": (summary + "\nHere are 50 sample songs from the dataset:\n" + sample_data + "\n\nQuestion: " + user_question
                    ),
                },
            ],
            temperature=0.3,
        )

        answer = response.choices[0].message.content
        st.success("✅ Model answer:")
        st.write(answer)



from utils import db_connect
engine = db_connect()

# your code here
import streamlit as st
from pickle import load
import pandas as pd

model = load(open("../models/decision_tree_classifier_optimized_42.sav", "rb"))
encoder = load(open("../models/species_encoder.sav", "rb"))

st.title("Predicción de especie de pingüino")

bill_length = st.slider("Longitud del pico (mm)", min_value=32.0, max_value=60.0, step=0.1)
bill_depth = st.slider("Profundidad del pico (mm)", min_value=13.0, max_value=22.0, step=0.1)
flipper_length = st.slider("Longitud de la aleta (mm)", min_value=170.0, max_value=235.0, step=1.0)
body_mass = st.slider("Masa corporal (g)", min_value=2700.0, max_value=6300.0, step=10.0)

if st.button("Predecir especie"):
    datos = pd.DataFrame(
        [[bill_length, bill_depth, flipper_length, body_mass]],
        columns=["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    )
    prediccion_numerica = model.predict(datos)[0]
    especie = encoder.inverse_transform([prediccion_numerica])[0]
    st.write("Especie predicha:", especie)
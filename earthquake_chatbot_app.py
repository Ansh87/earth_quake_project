import streamlit as st
import pandas as pd
import pydeck as pdk
import spacy
import joblib
import re

# Load NLP model and ML model
nlp = spacy.load("en_core_web_sm")
model = joblib.load("data/earthquake_magnitude_model.pkl")

# Load earthquake data
@st.cache_data
def load_data():
    df = pd.read_excel("data/Official Website of National Center of Seismology.xlsx", sheet_name='Sheet1')
    df_cleaned = df[1:].copy()
    df_cleaned.columns = df.iloc[0]
    df_cleaned = df_cleaned[1:]
    df_cleaned['Region'] = df_cleaned['Region'].astype(str)
    df_cleaned['Lat'] = pd.to_numeric(df_cleaned['Lat'], errors='coerce')
    df_cleaned['Long'] = pd.to_numeric(df_cleaned['Long'], errors='coerce')
    df_cleaned['Magnitude'] = pd.to_numeric(df_cleaned['Magnitude'], errors='coerce')
    df_cleaned['Depth'] = pd.to_numeric(df_cleaned['Depth'], errors='coerce')
    return df_cleaned.dropna(subset=['Lat', 'Long', 'Magnitude', 'Depth'])

df = load_data()

# UI
st.title("🌍 Earthquake Chatbot with NLP + ML")
user_input = st.text_input("Ask about earthquakes (e.g., 'Recent in Japan', 'Predict for 28.5, 77.2, 10')")

if user_input:
    # Try to extract lat, long, depth for prediction
    coords = re.findall(r"[-+]?\d*\.\d+|\d+", user_input.replace(",", " "))

    if len(coords) >= 3:
        try:
            lat, lon, depth = float(coords[0]), float(coords[1]), float(coords[2])
            input_df = pd.DataFrame({'Lat': [lat], 'Long': [lon], 'Depth': [depth]})
            magnitude = model.predict(input_df)[0]
            st.success(f"🧠 Predicted Magnitude for Lat:{lat}, Long:{lon}, Depth:{depth} → **{magnitude:.2f}**")
        except Exception as e:
                st.error(f"⚠️ Could not run prediction: {e}")
    else:
        # NLP-based Region Search
        doc = nlp(user_input)
        locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]

        if locations:
            region = locations[0]
            st.info(f"🔍 Interpreting query as region search: **{region}**")
            result = df[df['Region'].str.contains(region, case=False, na=False)]

            if not result.empty:
                st.success(f"Found {len(result)} earthquakes in {region}")
                st.dataframe(result[['Origin Time', 'Lat', 'Long', 'Depth', 'Magnitude', 'Location']].reset_index(drop=True))

                st.subheader("📍 Earthquake Map")
                st.pydeck_chart(pdk.Deck(
                    map_style='mapbox://styles/mapbox/light-v9',
                    initial_view_state=pdk.ViewState(
                        latitude=result['Lat'].mean(),
                        longitude=result['Long'].mean(),
                        zoom=4,
                        pitch=40,
                    ),
                    layers=[
                        pdk.Layer(
                            'ScatterplotLayer',
                            data=result,
                            get_position='[Long, Lat]',
                            get_color='[255, 0, 0, 160]',
                            get_radius='Magnitude * 20000',
                            pickable=True
                        ),
                    ],
                ))
            else:
                st.warning("No recent earthquakes found for that region.")
        else:
            st.warning("❓ I couldn't understand your query.")

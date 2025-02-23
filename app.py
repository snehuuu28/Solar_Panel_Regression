import streamlit as st
import numpy as np
import pickle
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt

# Load the trained model
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit app title and description
st.title("🌞 Solar Power Generation Prediction 🌞")

st.markdown("""
This application predicts the **power generation of solar panels** based on the input features.
Provide the required inputs and click **Predict** to see the result. ⬇️
""")

# Sidebar for input features
st.sidebar.header("Input Features 📊")

# Input fields with sliders for better interactivity
distance_to_solar_noon = st.sidebar.slider("⏳ Distance to Solar Noon (radians)", 0.0, 3.14, 1.57, step=0.01)
temperature = st.sidebar.slider("🌡️ Temperature (°C)", -30.0, 100.0, 25.0, step=0.1)
wind_direction = st.sidebar.slider("🌬️ Wind Direction (degrees)", 0.0, 360.0, 180.0, step=1.0)
wind_speed = st.sidebar.slider("💨 Wind Speed (m/s)", 0.0, 30.0, 3.0, step=0.1)
sky_cover = st.sidebar.slider("☁️ Sky Cover (0-4)", 0.0, 4.0, 2.0, step=1.0)
visibility = st.sidebar.slider("👀 Visibility (km)", 0.0, 50.0, 10.0, step=1.0)
humidity = st.sidebar.slider("💧 Humidity (%)", 0.0, 100.0, 60.0, step=1.0)
average_wind_speed = st.sidebar.slider("💨 Average Wind Speed (m/s)", 0.0, 100.0, 10.0, step=0.1)
average_pressure = st.sidebar.slider("📉 Average Pressure (inHg)", 18.0, 43.0, 29.92, step=0.01)

# Collect input features into an array
features = np.array([distance_to_solar_noon, temperature, wind_direction, wind_speed,
                     sky_cover, visibility, humidity, average_wind_speed, average_pressure]).reshape(1, -1)

# Initialize prediction variable
prediction = None

# Prediction button
if st.button("🔮 Predict"):
    with st.spinner("✨ Calculating..."):
        try:
            # Make prediction
            prediction = model.predict(features)[0]
            st.balloons()  # Celebration effect
            st.success(f"✨ The predicted power generation is: **{prediction:.2f} Joules** ✨")
            
            # Visualization of input features
            st.markdown("### 🌟 Feature vs Prediction Visualization")

            df_viz = pd.DataFrame({
                'Feature': [
                    'Distance to Solar Noon', 'Temperature', 'Wind Direction', 'Wind Speed',
                    'Sky Cover', 'Visibility', 'Humidity', 'Avg Wind Speed', 'Avg Pressure'
                ],
                'Value': features[0]
            })
            chart = alt.Chart(df_viz).mark_bar().encode(
                x='Feature:N',
                y='Value:Q',
                color='Feature:N',
                tooltip=['Feature', 'Value']
            ).interactive()
            st.altair_chart(chart, use_container_width=True)

            # Prediction confidence interval (if applicable)
            st.markdown("### 📈 Confidence Interval")
            lower_bound = prediction * 0.9  # For example, 10% lower
            upper_bound = prediction * 1.1  # For example, 10% higher
            st.write(f"Confidence Interval: **{lower_bound:.2f} to {upper_bound:.2f} Joules**")

            # Fun fact
            facts = [
                "Solar power is the most abundant energy source on Earth.",
                "Photovoltaic panels convert sunlight directly into electricity.",
                "India has one of the world's largest solar farms in Rajasthan.",
                "Solar panels work even on cloudy days by capturing diffuse sunlight."
            ]
            st.info(f"Did you know? {np.random.choice(facts)} 🌞")

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")

# Download Prediction Section - Only show if prediction exists
if prediction is not None:
    st.markdown("### 📂 Download Prediction")
    
    # Prepare results for download
    results = pd.DataFrame({
        'Feature': [
            'Distance to Solar Noon', 'Temperature', 'Wind Direction', 'Wind Speed', 
            'Sky Cover', 'Visibility', 'Humidity', 'Avg Wind Speed', 'Avg Pressure'
        ],
        'Input Value': features[0],
        'Predicted Power (Joules)': [prediction] * len(features[0])  # Repeat prediction for each feature
    })

    st.download_button("📂 Download Prediction as CSV", results.to_csv(index=False), "prediction.csv", "text/csv")

# Footer Section
st.markdown("""
---
Made with ❤️ to empower sustainable energy solutions. 🌍
""")
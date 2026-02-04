import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
from datetime import datetime

# Configuration
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="BCS Fault Predictor",
    page_icon="🚌",
    layout="wide"
)

# Custom CSS for aesthetics
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .status-normal { color: #28a745; font-weight: bold; font-size: 24px; }
    .status-fault { color: #dc3545; font-weight: bold; font-size: 24px; }
</style>
""", unsafe_allow_html=True)

st.title("🚌 Battery Cooling System (BCS) Fault Predictor")
st.markdown("Real-time predictive maintenance for electric bus fleets.")

# Sidebar for model status
with st.sidebar:
    st.header("Service Status")
    try:
        response = requests.get(f"{API_URL}/")
        if response.status_code == 200:
            st.success("API Backend: Connected")
            available_models = response.json().get("available_models", [])
            st.info(f"Available Models: {', '.join(available_models)}")
        else:
            st.error("API Backend: Error")
    except:
        st.warning("API Backend: Disconnected (Check if api.py is running)")
    
    st.divider()
    st.markdown("### About")
    st.info("This tool uses Random Forest models trained on historical sensor data to predict machinery faults before they occur.")

# Main content
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Sensor Inputs")
    with st.form("prediction_form"):
        manufacturer = st.selectbox(
            "Select Bus Manufacturer",
            ["Empire", "MBMT", "DHERADUN"],
            help="Choose the model trained for specific manufacturer signatures."
        )
        
        amax_temp = st.number_input("A Max Cell Temp (°C)", value=25.0, step=0.1)
        bmax_temp = st.number_input("B Max Cell Temp (°C)", value=22.0, step=0.1)
        therm1 = st.number_input("Thermistor 1 (°C)", value=21.0, step=0.1)
        therm2 = st.number_input("Thermistor 2 (°C)", value=21.0, step=0.1)
        
        submitted = st.form_submit_button("Predict Fault Status")

with col2:
    st.subheader("Analysis & Prediction")
    if submitted:
        payload = {
            "manufacturer": manufacturer,
            "amax_cell_temp": amax_temp,
            "bmax_cell_temp": bmax_temp,
            "thermistor1": therm1,
            "thermistor2": therm2
        }
        
        try:
            with st.spinner("Analyzing sensor signatures..."):
                response = requests.post(f"{API_URL}/predict", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                
                # Result Card
                fault = result["fault_detected"]
                prob = result["probability"]
                confidence = result["confidence_level"]
                
                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                if fault:
                    st.markdown(f'<p class="status-fault">⚠️ FAULT DETECTED</p>', unsafe_allow_html=True)
                    st.warning(f"High risk detected. Maintenance recommended soon.")
                else:
                    st.markdown(f'<p class="status-normal">✅ NORMAL OPERATION</p>', unsafe_allow_html=True)
                    st.success(f"System operating within normal parameters.")
                
                st.markdown(f"**Probability of Fault:** {prob:.1%}")
                st.markdown(f"**Confidence Level:** {confidence}")
                st.markdown(f"**Manufacturer Signature:** {manufacturer}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Visualization
                st.divider()
                st.subheader("Input Visualization")
                input_data = result["input_data"]
                df_viz = pd.DataFrame({
                    'Sensor': ['A Max Temp', 'B Max Temp', 'Thermistor 1', 'Thermistor 2'],
                    'Value': [input_data['amax_cell_temp'], input_data['bmax_cell_temp'], 
                              input_data['thermistor1'], input_data['thermistor2']]
                })
                
                fig = px.bar(df_viz, x='Sensor', y='Value', color='Value', 
                            color_continuous_scale='RdYlGn_r', height=300)
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.error(f"Error from API: {response.json().get('detail', 'Unknown error')}")
        except Exception as e:
            st.error(f"Failed to connect to backend: {e}")
    else:
        st.info("👈 Fill in the sensor readings and click 'Predict' to get results.")
        
        # Example ranges info
        with st.expander("Sensor Threshold Information"):
            st.markdown("""
            - **Normal Range**: 15°C - 30°C
            - **Warning Range**: 35°C - 45°C
            - **Critical Range**: > 50°C
            
            *Note: Fault signatures depend heavily on the manufacturer selected.*
            """)

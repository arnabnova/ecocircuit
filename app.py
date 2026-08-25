import streamlit as st
import pickle
import numpy as np
import requests
from PIL import Image
import io
import json

st.set_page_config(page_title="EcoCircuit", layout="wide", initial_sidebar_state="expanded")
st.title("♻️ EcoCircuit - E-Waste Device Disposal Assistant")

# Load models & encoders from artifacts
@st.cache_resource
def load_models():
    with open('artifacts/model_resale.pkl', 'rb') as f:
        rf_resale = pickle.load(f)
    with open('artifacts/model_repair_cost.pkl', 'rb') as f:
        rf_repair = pickle.load(f)
    with open('artifacts/model_recycle.pkl', 'rb') as f:
        rf_recycle = pickle.load(f)
    with open('artifacts/encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    with open('artifacts/features.pkl', 'rb') as f:
        features = pickle.load(f)
    return rf_resale, rf_repair, rf_recycle, encoders, features

rf_resale, rf_repair, rf_recycle, encoders, features = load_models()

# Condition mapping from CLIP to dataset labels
condition_map = {
    "like-new laptop": "No_Damage",
    "used laptop": "Minor",
    "damaged laptop": "Severe",
    "broken laptop": "Critical"
}

# Create 2 columns for input
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Device Details")
    brand = st.selectbox("Brand", encoders['Brand'].classes_)
    laptop_type = st.selectbox("Type", encoders['Laptop_Type'].classes_)
    processor = st.selectbox("Processor", encoders['Processor'].classes_)
    graphics = st.selectbox("Graphics", encoders['Graphics'].classes_)
    os = st.selectbox("OS", encoders['OS'].classes_)
    
    ram_gb = st.number_input("RAM (GB)", min_value=1, max_value=64, value=8)
    storage_gb = st.number_input("Storage (GB)", min_value=128, max_value=2048, value=256)
    display_inches = st.number_input("Display (inches)", min_value=10.0, max_value=17.0, value=13.3)
    device_age = st.number_input("Device Age (years)", min_value=0, max_value=20, value=3)

with col2:
    st.subheader("📷 Upload Photo")
    uploaded_file = st.file_uploader("Choose device photo", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", use_column_width=True)

# Analyze button
if st.button("🔍 Analyze Device", use_container_width=True):
    if not uploaded_file:
        st.error("❌ Please upload an image first")
    else:
        with st.spinner("Analyzing device..."):
            try:
                # Step 1: Call CLIP on HF Space
                st.info("📸 Analyzing device condition with AI...")
                image_bytes = uploaded_file.getvalue()
                files = {'data': (uploaded_file.name, image_bytes)}
                
                # Replace with your HF Space URL
                CLIP_API_URL = "https://your-username-ecocircuit-clip.hf.space/api/predict"
                clip_response = requests.post(CLIP_API_URL, files=files, timeout=60)
                
                if clip_response.status_code != 200:
                    st.error(f"❌ CLIP API error: {clip_response.status_code}")
                else:
                    clip_data = clip_response.json()
                    condition_label = clip_data['data'][0]  # e.g. "like-new laptop"
                    
                    # Step 2: Map condition
                    mapped_condition = condition_map.get(condition_label, "Minor")
                    
                    # Step 3: Build feature vector
                    features_list = []
                    feature_names = ['Brand', 'Laptop_Type', 'Processor', 'Ram_GB', 'Storage_GB', 
                                    'Display_Inches', 'Graphics', 'OS', 'Device_Age_Years', 'Image_Damage_Class']
                    
                    for feat in feature_names:
                        if feat in encoders:
                            if feat == 'Brand':
                                features_list.append(encoders[feat].transform([brand])[0])
                            elif feat == 'Laptop_Type':
                                features_list.append(encoders[feat].transform([laptop_type])[0])
                            elif feat == 'Processor':
                                features_list.append(encoders[feat].transform([processor])[0])
                            elif feat == 'Graphics':
                                features_list.append(encoders[feat].transform([graphics])[0])
                            elif feat == 'OS':
                                features_list.append(encoders[feat].transform([os])[0])
                            elif feat == 'Image_Damage_Class':
                                features_list.append(encoders[feat].transform([mapped_condition])[0])
                        else:
                            if feat == 'Ram_GB':
                                features_list.append(ram_gb)
                            elif feat == 'Storage_GB':
                                features_list.append(storage_gb)
                            elif feat == 'Display_Inches':
                                features_list.append(display_inches)
                            elif feat == 'Device_Age_Years':
                                features_list.append(device_age)
                    
                    X = np.array(features_list).reshape(1, -1)
                    
                    # Step 4: Predict with 3 models
                    st.info("🤖 Making predictions with ML models...")
                    resale_value = float(rf_resale.predict(X)[0])
                    repair_cost = float(rf_repair.predict(X)[0])
                    recycle_value = float(rf_recycle.predict(X)[0])
                    
                    # Step 5: Scoring rule
                    refurb_profit = max(0, resale_value - repair_cost)
                    
                    scores = {
                        'Resell': resale_value,
                        'Refurbish': refurb_profit,
                        'Recycle': recycle_value
                    }
                    
                    best_pathway = max(scores, key=scores.get)
                    best_value = scores[best_pathway]
                    
                    # Display Results
                    st.success("✓ Analysis Complete!")
                    
                    # Device Condition
                    st.subheader("🔍 Device Condition")
                    st.metric("AI Classification", condition_label)
                    
                    # Predictions
                    st.subheader("💰 Value Breakdown")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Resale Value", f"₹{resale_value:,.0f}")
                    col2.metric("Repair Cost", f"₹{repair_cost:,.0f}")
                    col3.metric("Recycle Value", f"₹{recycle_value:,.0f}")
                    
                    # Refurbishment Profit
                    st.subheader("📊 Refurbishment Analysis")
                    st.metric("Refurb Profit Margin", f"₹{refurb_profit:,.0f}", 
                              delta=f"Resale - Repair Cost")
                    
                    # Recommendation
                    st.subheader("🎯 Recommended Disposal Pathway")
                    if best_pathway == 'Resell':
                        st.success(f"**RESELL** - Best Value: ₹{best_value:,.0f}")
                        st.info("✓ This device is in good condition. Sell it as-is for maximum value.")
                    elif best_pathway == 'Refurbish':
                        st.success(f"**REFURBISH** - Best Value: ₹{best_value:,.0f}")
                        st.info(f"✓ Refurbish and resell. Repair cost (₹{repair_cost:,.0f}) is worth it for profit of ₹{best_value:,.0f}.")
                    else:
                        st.success(f"**RECYCLE** - Best Value: ₹{best_value:,.0f}")
                        st.info(f"✓ Best economics are in material recovery. Recycle value: ₹{best_value:,.0f}")
                    
                    # Summary Table
                    st.subheader("📈 Pathway Comparison")
                    summary_data = {
                        'Pathway': ['Resell', 'Refurbish', 'Recycle'],
                        'Value (₹)': [f"{scores['Resell']:,.0f}", f"{scores['Refurbish']:,.0f}", f"{scores['Recycle']:,.0f}"],
                        'Rank': ['1' if best_pathway == 'Resell' else '', 
                                '1' if best_pathway == 'Refurbish' else '', 
                                '1' if best_pathway == 'Recycle' else '']
                    }
                    st.table(summary_data)
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("Make sure the HF Space URL is correct and the API is running.")

# Sidebar info
with st.sidebar:
    st.subheader("ℹ️ About EcoCircuit")
    st.write("""
    EcoCircuit helps you find the best way to dispose of old electronics:
    - **Resell**: Sell working devices as-is
    - **Refurbish**: Repair and resell for profit
    - **Recycle**: Extract valuable materials
    
    Uses AI (CLIP) to detect device condition + ML models to predict values.
    """)
    
    st.subheader("🔧 Model Performance")
    st.metric("Resale Accuracy (R²)", "0.71")
    st.metric("Repair Cost Accuracy (R²)", "0.71")
    st.metric("Recycle Accuracy (R²)", "0.71")
    
    st.divider()
    st.caption("Made for EcoCircuit Hackathon 2026")
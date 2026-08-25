from flask import Flask, request, jsonify
import pickle
import numpy as np
import clip
import torch
from PIL import Image
import io
import base64
import os

app = Flask(__name__)

# Load models
rf_resale = pickle.load(open('model_resale.pkl', 'rb'))
rf_repair = pickle.load(open('model_repair_cost.pkl', 'rb'))
rf_recycle = pickle.load(open('model_recycle.pkl', 'rb'))
encoders = pickle.load(open('encoders.pkl', 'rb'))

# Load CLIP
device = "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

condition_map = {"like-new laptop": "No_Damage", "used laptop": "Minor", "damaged laptop": "Severe", "broken laptop": "Critical"}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Decode image from base64
        image_base64 = data.get('image')
        image = Image.open(io.BytesIO(base64.b64decode(image_base64)))
        
        # CLIP classification
        image_tensor = clip_preprocess(image).unsqueeze(0).to(device)
        labels = ["like-new laptop", "used laptop", "damaged laptop", "broken laptop"]
        text_tokens = clip.tokenize(labels).to(device)
        
        with torch.no_grad():
            image_features = clip_model.encode_image(image_tensor)
            text_features = clip_model.encode_text(text_tokens)
            logits = (image_features @ text_features.T).softmax(dim=-1)
        
        scores = logits[0].cpu().tolist()
        condition_label = labels[scores.index(max(scores))]
        mapped_condition = condition_map[condition_label]
        
        # Build features
        features_list = []
        feature_names = ['Brand', 'Laptop_Type', 'Processor', 'Ram_GB', 'Storage_GB', 'Display_Inches', 'Graphics', 'OS', 'Device_Age_Years', 'Image_Damage_Class']
        
        for feat in feature_names:
            if feat in encoders:
                if feat == 'Brand':
                    features_list.append(encoders[feat].transform([data['brand']])[0])
                elif feat == 'Laptop_Type':
                    features_list.append(encoders[feat].transform([data['laptop_type']])[0])
                elif feat == 'Processor':
                    features_list.append(encoders[feat].transform([data['processor']])[0])
                elif feat == 'Graphics':
                    features_list.append(encoders[feat].transform([data['graphics']])[0])
                elif feat == 'OS':
                    features_list.append(encoders[feat].transform([data['os']])[0])
                elif feat == 'Image_Damage_Class':
                    features_list.append(encoders[feat].transform([mapped_condition])[0])
            else:
                if feat == 'Ram_GB':
                    features_list.append(data['ram_gb'])
                elif feat == 'Storage_GB':
                    features_list.append(data['storage_gb'])
                elif feat == 'Display_Inches':
                    features_list.append(data['display_inches'])
                elif feat == 'Device_Age_Years':
                    features_list.append(data['device_age'])
        
        X = np.array(features_list).reshape(1, -1)
        
        # Predictions
        resale = float(rf_resale.predict(X)[0])
        repair = float(rf_repair.predict(X)[0])
        recycle = float(rf_recycle.predict(X)[0])
        
        # Scoring
        refurb_profit = max(0, resale - repair)
        recommendation = max({'Resell': resale, 'Refurbish': refurb_profit, 'Recycle': recycle}, 
                            key=lambda k: {'Resell': resale, 'Refurbish': refurb_profit, 'Recycle': recycle}[k])
        
        return jsonify({
            'condition': condition_label,
            'resale_value': resale,
            'repair_cost': repair,
            'recycle_value': recycle,
            'refurb_profit': refurb_profit,
            'recommendation': recommendation
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
"""
app.py - Application Flask pour le deploiement du modele ML
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import sys
import os

# Ajouter le repertoire src au path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from predict import (
    load_all_models, predict_churn, predict_cluster,
    get_cluster_description, get_churn_risk_description
)

app = Flask(__name__)

# Charger les modeles au demarrage
print("Chargement des modeles...")
models = load_all_models()
print("Modeles charges!")


@app.route('/')
def home():
    """Page d'accueil"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint pour les predictions
    """
    try:
        # Recuperer les donnees du formulaire
        data = request.form.to_dict()
        
        # Convertir les valeurs en float/int
        feature_values = {}
        for key, value in data.items():
            try:
                feature_values[key] = float(value)
            except:
                feature_values[key] = value
        
        # Creer un DataFrame
        df = pd.DataFrame([feature_values])
        
        # S'assurer que toutes les features sont presentes
        if 'feature_names' in models:
            expected_features = models['feature_names']
            for feat in expected_features:
                if feat not in df.columns:
                    df[feat] = 0
            df = df[expected_features]
        
        X = df.values
        
        # Predictions
        results = {}
        
        # Churn
        try:
            churn_pred, churn_prob = predict_churn(X, models)
            results['churn'] = {
                'prediction': int(churn_pred[0]),
                'probability': round(float(churn_prob[0]) * 100, 2),
                'risk_level': get_churn_risk_description(churn_prob[0])
            }
        except Exception as e:
            results['churn'] = {'error': str(e)}
        
        # Cluster
        try:
            cluster = predict_cluster(X, models)
            results['cluster'] = {
                'id': int(cluster[0]),
                'description': get_cluster_description(cluster[0])
            }
        except Exception as e:
            results['cluster'] = {'error': str(e)}
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    API endpoint pour les predictions (JSON)
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Creer un DataFrame
        df = pd.DataFrame([data])
        
        # S'assurer que toutes les features sont presentes
        if 'feature_names' in models:
            expected_features = models['feature_names']
            for feat in expected_features:
                if feat not in df.columns:
                    df[feat] = 0
            df = df[expected_features]
        
        X = df.values
        
        # Predictions
        results = {}
        
        # Churn
        try:
            churn_pred, churn_prob = predict_churn(X, models)
            results['churn_prediction'] = int(churn_pred[0])
            results['churn_probability'] = round(float(churn_prob[0]) * 100, 2)
            results['churn_risk'] = get_churn_risk_description(churn_prob[0])
        except Exception as e:
            results['churn_error'] = str(e)
        
        # Cluster
        try:
            cluster = predict_cluster(X, models)
            results['cluster'] = int(cluster[0])
            results['cluster_description'] = get_cluster_description(cluster[0])
        except Exception as e:
            results['cluster_error'] = str(e)
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Endpoint pour predictions en batch (fichier CSV)
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Lire le CSV
        df = pd.read_csv(file)
        
        # S'assurer que toutes les features sont presentes
        if 'feature_names' in models:
            expected_features = models['feature_names']
            for feat in expected_features:
                if feat not in df.columns:
                    df[feat] = 0
            
            # Garder une copie de l'original pour le resultat
            df_original = df.copy()
            df = df[expected_features]
        
        X = df.values
        
        # Predictions
        results_list = []
        
        for i in range(len(X)):
            row_result = {'index': i}
            
            try:
                churn_pred, churn_prob = predict_churn(X[i:i+1], models)
                row_result['churn_prediction'] = int(churn_pred[0])
                row_result['churn_probability'] = round(float(churn_prob[0]) * 100, 2)
            except:
                pass
            
            try:
                cluster = predict_cluster(X[i:i+1], models)
                row_result['cluster'] = int(cluster[0])
            except:
                pass
            
            results_list.append(row_result)
        
        return jsonify({
            'count': len(results_list),
            'results': results_list
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': list(models.keys())
    })


if __name__ == '__main__':
    import webbrowser
    import threading
    
    def open_browser():
        webbrowser.open('http://127.0.0.1:8080')
    
    # Open browser after 1.5 seconds
    threading.Timer(1.5, open_browser).start()
    
    print("\n" + "="*50)
    print("SERVEUR DEMARRE!")
    print("Ouvrez votre navigateur: http://127.0.0.1:8080")
    print("="*50 + "\n")
    
    app.run(debug=False, host='127.0.0.1', port=8080)

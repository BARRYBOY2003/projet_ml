from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from predict import (
    load_all_models, predict_churn, predict_cluster,
    get_cluster_description, get_churn_risk_description
)

app = Flask(__name__)

print("Chargement des modeles...")
models = load_all_models()
print("Modeles charges!")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()
        
        feature_values = {}
        for key, value in data.items():
            try:
                feature_values[key] = float(value)
            except:
                feature_values[key] = value
        
        df = pd.DataFrame([feature_values])
        
        if 'feature_names' in models:
            expected_features = models['feature_names']
            for feat in expected_features:
                if feat not in df.columns:
                    df[feat] = 0
            df = df[expected_features]
        
        X = df.values
        results = {}
        
        try:
            churn_pred, churn_prob = predict_churn(X, models)
            results['churn'] = {
                'prediction': int(churn_pred[0]),
                'probability': round(float(churn_prob[0]) * 100, 2),
                'risk_level': get_churn_risk_description(churn_prob[0])
            }
        except Exception as e:
            results['churn'] = {'error': str(e)}
        
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
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        df = pd.DataFrame([data])
        
        if 'feature_names' in models:
            expected_features = models['feature_names']
            for feat in expected_features:
                if feat not in df.columns:
                    df[feat] = 0
            df = df[expected_features]
        
        X = df.values
        results = {}
        
        try:
            churn_pred, churn_prob = predict_churn(X, models)
            results['churn_prediction'] = int(churn_pred[0])
            results['churn_probability'] = round(float(churn_prob[0]) * 100, 2)
            results['churn_risk'] = get_churn_risk_description(churn_prob[0])
        except Exception as e:
            results['churn_error'] = str(e)
        
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
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        df = pd.read_csv(file)
        
        if 'feature_names' in models:
            expected_features = models['feature_names']
            for feat in expected_features:
                if feat not in df.columns:
                    df[feat] = 0
            df_original = df.copy()
            df = df[expected_features]
        
        X = df.values
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
    return jsonify({
        'status': 'healthy',
        'models_loaded': list(models.keys())
    })


if __name__ == '__main__':
    import webbrowser
    import threading
    
    def open_browser():
        webbrowser.open('http://127.0.0.1:8080')
    
    threading.Timer(1.5, open_browser).start()
    
    print("\n" + "="*50)
    print("SERVEUR DEMARRE!")
    print("Ouvrez votre navigateur: http://127.0.0.1:8080")
    print("="*50 + "\n")
    
    app.run(debug=False, host='127.0.0.1', port=8080)

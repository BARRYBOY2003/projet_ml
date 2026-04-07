import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import load_model, MODELS_DIR
import warnings
warnings.filterwarnings('ignore')


def load_all_models():
    models = {}
    
    try:
        models['scaler'] = load_model('standard_scaler.joblib')
        print("  - StandardScaler charge")
    except:
        print("  ! StandardScaler non trouve")
    
    try:
        models['feature_names'] = load_model('feature_names.joblib')
        print("  - Feature names charges")
    except:
        print("  ! Feature names non trouves")
    
    try:
        models['pca'] = load_model('pca_model.joblib')
        print("  - PCA charge")
    except:
        print("  ! PCA non trouve")
    
    try:
        models['kmeans'] = load_model('kmeans_model.joblib')
        print("  - K-Means charge")
    except:
        print("  ! K-Means non trouve")
    
    try:
        models['classifier'] = load_model('best_classifier_churn.joblib')
        print("  - Classifier (Churn) charge")
    except:
        print("  ! Classifier non trouve")
    
    try:
        models['regressor'] = load_model('best_regressor.joblib')
        print("  - Regressor charge")
    except:
        print("  ! Regressor non trouve")
    
    try:
        models['label_encoder_country'] = load_model('label_encoder_country.joblib')
        print("  - Label Encoder (Country) charge")
    except:
        print("  ! Label Encoder non trouve")
    
    return models


def predict_churn(X, models):
    if 'classifier' not in models:
        raise ValueError("Classifier non charge")
    
    classifier = models['classifier']
    predictions = classifier.predict(X)
    
    try:
        probabilities = classifier.predict_proba(X)[:, 1]
    except:
        probabilities = predictions.astype(float)
    
    return predictions, probabilities


def predict_cluster(X, models):
    if 'kmeans' not in models:
        raise ValueError("K-Means non charge")
    
    kmeans = models['kmeans']
    clusters = kmeans.predict(X)
    return clusters


def predict_monetary(X, models):
    if 'regressor' not in models:
        raise ValueError("Regressor non charge")
    
    regressor = models['regressor']
    predictions = regressor.predict(X)
    return predictions


def get_cluster_description(cluster_id):
    descriptions = {
        0: "Clients Dormants - Faible activite, necessite reactivation",
        1: "Clients Potentiels - Engagement modere, opportunite de developpement",
        2: "Clients Fideles - Engagement regulier, maintenir la relation",
        3: "Champions - Meilleurs clients, programme VIP recommande"
    }
    return descriptions.get(cluster_id, f"Cluster {cluster_id}")


def get_churn_risk_description(probability):
    if probability < 0.2:
        return "Faible risque - Client fidele"
    elif probability < 0.4:
        return "Risque modere - Surveillance recommandee"
    elif probability < 0.6:
        return "Risque eleve - Actions preventives necessaires"
    else:
        return "Risque critique - Intervention urgente"


def predict_single_customer(customer_data, models, feature_names=None):
    if isinstance(customer_data, dict):
        df = pd.DataFrame([customer_data])
    elif isinstance(customer_data, pd.Series):
        df = pd.DataFrame([customer_data])
    else:
        df = customer_data
    
    if feature_names and 'feature_names' in models:
        expected_features = models['feature_names']
        for feat in expected_features:
            if feat not in df.columns:
                df[feat] = 0
        df = df[expected_features]
    
    X = df.values
    results = {'customer_data': customer_data}
    
    try:
        churn_pred, churn_prob = predict_churn(X, models)
        results['churn_prediction'] = int(churn_pred[0])
        results['churn_probability'] = float(churn_prob[0])
        results['churn_risk'] = get_churn_risk_description(churn_prob[0])
    except Exception as e:
        results['churn_error'] = str(e)
    
    try:
        cluster = predict_cluster(X, models)
        results['cluster'] = int(cluster[0])
        results['cluster_description'] = get_cluster_description(cluster[0])
    except Exception as e:
        results['cluster_error'] = str(e)
    
    try:
        monetary = predict_monetary(X, models)
        results['predicted_monetary'] = float(monetary[0])
    except Exception as e:
        results['monetary_error'] = str(e)
    
    return results


def batch_predict(df, models):
    results = df.copy()
    X = df.values
    
    try:
        churn_pred, churn_prob = predict_churn(X, models)
        results['Churn_Predicted'] = churn_pred
        results['Churn_Probability'] = churn_prob
        results['Churn_Risk'] = [get_churn_risk_description(p) for p in churn_prob]
    except Exception as e:
        print(f"Erreur prediction churn: {e}")
    
    try:
        clusters = predict_cluster(X, models)
        results['Cluster'] = clusters
        results['Cluster_Description'] = [get_cluster_description(c) for c in clusters]
    except Exception as e:
        print(f"Erreur prediction cluster: {e}")
    
    return results


def main():
    print("\n" + "="*60)
    print("CHARGEMENT DES MODELES")
    print("="*60)
    
    models = load_all_models()
    
    if not models:
        print("\nAucun modele trouve. Executez d'abord train_model.py")
        return
    
    print("\n" + "="*60)
    print("TEST DES PREDICTIONS")
    print("="*60)
    
    try:
        from utils import load_train_test_data
        X_train, X_test, y_train, y_test = load_train_test_data()
        
        print(f"\nTest sur {len(X_test)} echantillons...")
        
        results = batch_predict(X_test, models)
        
        print("\n--- Echantillon de predictions ---")
        display_cols = ['Churn_Predicted', 'Churn_Probability', 'Cluster']
        available_cols = [c for c in display_cols if c in results.columns]
        if available_cols:
            print(results[available_cols].head(10))
        
        if 'Churn_Predicted' in results.columns:
            print("\n--- Distribution des predictions de churn ---")
            print(results['Churn_Predicted'].value_counts(normalize=True))
        
        if 'Cluster' in results.columns:
            print("\n--- Distribution des clusters ---")
            print(results['Cluster'].value_counts().sort_index())
        
        print("\n--- Test prediction client unique ---")
        sample_customer = X_test.iloc[0]
        prediction = predict_single_customer(sample_customer, models)
        
        print(f"\nPredictions pour le client:")
        for key, value in prediction.items():
            if key != 'customer_data':
                print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"Erreur lors du test: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("TEST TERMINE")
    print("="*60)


if __name__ == "__main__":
    main()

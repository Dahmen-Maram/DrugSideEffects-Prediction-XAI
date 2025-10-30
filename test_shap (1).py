import pandas as pd
import numpy as np
import joblib
import shap
import warnings
from sklearn.multiclass import OneVsRestClassifier
import xgboost as xgb
from tqdm import tqdm
import time
import math

warnings.filterwarnings('ignore')
np.random.seed(42)

# Définir la classe FixedBlendingPredictor (tirée de votre code original)
class FixedBlendingPredictor:
    def __init__(self):
        self.base_models = {}
        self.meta_model = None
        self.feature_names = None
        self.top_drugs = None
        self.top_reactions = None
        self.per_label_thresholds = None

    def create_base_models(self):
        self.base_models = {
            'xgb1': OneVsRestClassifier(xgb.XGBClassifier(n_estimators=100, max_depth=8, learning_rate=0.1, random_state=42), n_jobs=-1),
            'rf1': OneVsRestClassifier(RandomForestClassifier(n_estimators=100, max_depth=15, random_state=44, n_jobs=-1), n_jobs=-1),
            'lr1': OneVsRestClassifier(LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=1000, random_state=46), n_jobs=-1),
        }

    def train_base_models(self, X_train, y_train):
        for name, model in tqdm(self.base_models.items(), desc='Training base models'):
            try:
                model.fit(X_train, y_train)
            except Exception as e:
                print(f"Erreur entraînement {name}: {e}")

    def create_meta_features(self, X, batch_size=1000):
        """
        Crée les méta-features en traitant X par batch pour afficher une progression
        et éviter les pics mémoire. Retourne np.hstack des probabilités par base model.
        """
        meta_features = []
        n_samples = X.shape[0]
        for name, model in tqdm(self.base_models.items(), desc='Generating meta-features'):
            parts = []
            start_model = time.perf_counter()
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                X_batch = X.iloc[start:end] if hasattr(X, 'iloc') else X[start:end]
                t0 = time.perf_counter()
                probas_batch = model.predict_proba(X_batch)
                t1 = time.perf_counter()
                parts.append(probas_batch)
                elapsed = t1 - t0
                remaining_batches = math.ceil((n_samples - end) / batch_size)
                est_remaining = remaining_batches * elapsed
                print(f"   {name}: processed {end}/{n_samples} samples (batch_time={elapsed:.2f}s, est_remain~{est_remaining:.1f}s)")
            model_time = time.perf_counter() - start_model
            print(f"--> {name} done in {model_time:.1f}s")
            meta_features.append(np.vstack(parts))
        if meta_features:
            return np.hstack(meta_features)
        else:
            raise ValueError("Aucune méta-feature générée")

    def train_meta_model(self, X_train, y_train):
        X_meta = self.create_meta_features(X_train)
        self.meta_model = xgb.XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=49)
        self.meta_model.fit(X_meta, y_train)

    def predict_proba_blended(self, X):
        X_meta = self.create_meta_features(X)
        return self.meta_model.predict_proba(X_meta)

    def predict(self, X, threshold=0.05, min_predictions=5):
        """
        Retourne (preds_binary, probabilities).
        - threshold : utilisé si self.per_label_thresholds is None
        - min_predictions : garantit au moins min_predictions labels par instance
          en complétant avec les labels les plus probables si nécessaire.
        """
        probabilities = self.predict_proba_blended(X)  # shape (n_samples, n_labels)

        if self.per_label_thresholds is None:
            preds = (probabilities > threshold).astype(int)
        else:
            preds = (probabilities > self.per_label_thresholds[np.newaxis, :]).astype(int)

        # compléter jusqu'à min_predictions par instance si besoin
        if min_predictions is not None and min_predictions > 0:
            n_samples, n_labels = probabilities.shape
            for i in range(n_samples):
                current_pos = int(preds[i].sum())
                if current_pos >= min_predictions:
                    continue
                sorted_idx = np.argsort(-probabilities[i])  # indices triés par prob décroissante
                for idx in sorted_idx:
                    if preds[i, idx] == 0:
                        preds[i, idx] = 1
                    if int(preds[i].sum()) >= min_predictions:
                        break

        return preds, probabilities

# Fonctions SHAP (inchangées)
def analyse_shap_ultra_simple(medicaments, age, sex, reaction_cible, blender, X_sample):
    """
    Version ultra-simplifiée et robuste de l'analyse SHAP
    """
    print(f"\n🎯 ANALYSE: {reaction_cible}")
    print("-" * 40)

    # Vérifier que la réaction existe
    if reaction_cible not in blender.top_reactions:
        print(f"❌ Réaction '{reaction_cible}' non trouvée dans les top réactions")
        return

    # Préparer les features du patient
    patient_features = {}
    drugs_up = [d.upper() for d in medicaments]
    for drug in blender.top_drugs:
        patient_features[f"drug_{drug}"] = 1 if drug in drugs_up else 0
    patient_features["age"] = age
    patient_features["sex"] = sex
    patient_features["nb_drugs"] = len(medicaments)
    patient_features["nb_reactions"] = 0
    patient_features["is_sider"] = 0

    # Créer le DataFrame
    patient_df = pd.DataFrame([patient_features])

    # Aligner les colonnes avec X_sample
    for col in X_sample.columns:
        if col not in patient_df.columns:
            patient_df[col] = 0
    patient_df = patient_df[X_sample.columns]

    # Obtenir l'index de la réaction
    idx_reaction = blender.top_reactions.index(reaction_cible)

    # Prédiction de probabilité
    try:
        preds, probas = blender.predict(patient_df, min_predictions=1)
        prediction_proba = probas[0][idx_reaction]
        print(f"📈 Probabilité prédite: {prediction_proba:.1%}")
    except:
        print("❌ Erreur lors de la prédiction de probabilité")
        return

    try:
        # SHAP pour cette réaction spécifique
        explainer = shap.TreeExplainer(blender.base_models['rf1'].estimators_[idx_reaction])
        shap_values = explainer.shap_values(patient_df)

        # DEBUG: Afficher le type et la forme pour comprendre
        print(f"🔧 Debug - Type SHAP: {type(shap_values)}, Forme: {getattr(shap_values, 'shape', 'No shape')}")

        # Gestion robuste des différents formats SHAP
        if isinstance(shap_values, list):
            # Format [shap_negative, shap_positive]
            if len(shap_values) == 2:
                shap_vals = shap_values[1]  # Classe positive
            else:
                shap_vals = shap_values[0]
        elif hasattr(shap_values, 'shape'):
            # Format array numpy
            if len(shap_values.shape) == 3:
                shap_vals = shap_values[0, :, 1] if shap_values.shape[2] == 2 else shap_values[0, :, 0]
            elif len(shap_values.shape) == 2:
                shap_vals = shap_values[0, :]
            else:
                shap_vals = shap_values
        else:
            print("❌ Format SHAP non reconnu")
            return

        # S'assurer que c'est un array 1D
        if hasattr(shap_vals, 'flatten'):
            shap_vals = shap_vals.flatten()

        # Créer le DataFrame des contributions
        contributions = []
        for i, col_name in enumerate(patient_df.columns):
            contribution_val = shap_vals[i] if i < len(shap_vals) else 0
            feature_name = col_name
            
            # Formater le nom de la feature pour l'affichage
            if feature_name.startswith('drug_'):
                drug_name = feature_name.replace('drug_', '')
                feature_value = patient_df[col_name].iloc[0]
                status = "PRÉSENT" if feature_value == 1 else "ABSENT"
                display_name = f"{drug_name} ({status})"
            elif feature_name == "age":
                feature_value = patient_df[col_name].iloc[0]
                display_name = f"âge (= {int(feature_value)})"
            elif feature_name == "sex":
                feature_value = patient_df[col_name].iloc[0]
                gender = "Homme" if feature_value == 1 else "Femme"
                display_name = f"sexe (= {int(feature_value)}: {gender})"
            elif feature_name == "nb_drugs":
                feature_value = patient_df[col_name].iloc[0]
                display_name = f"nb_médicaments (= {int(feature_value)})"
            elif feature_name == "nb_reactions":
                feature_value = patient_df[col_name].iloc[0]
                display_name = f"nb_réactions (= {int(feature_value)})"
            elif feature_name == "is_sider":
                feature_value = patient_df[col_name].iloc[0]
                display_name = f"is_sider (= {int(feature_value)})"
            else:
                feature_value = patient_df[col_name].iloc[0]
                display_name = f"{feature_name} (= {feature_value:.2f})"

            contributions.append({
                'feature': display_name,
                'contribution': contribution_val,
                'value': feature_value
            })

        contrib_df = pd.DataFrame(contributions)

        # Afficher les résultats
        print("\n📊 FACTEURS INFLUENTS:")

        # Facteurs qui augmentent le risque
        positives = contrib_df[contrib_df['contribution'] > 0.001].nlargest(8, 'contribution')
        if len(positives) > 0:
            print("  ➕ AUGMENTENT le risque:")
            for _, row in positives.iterrows():
                print(f"     {row['feature']}: +{row['contribution']:.3f}")

        # Facteurs qui réduisent le risque
        negatives = contrib_df[contrib_df['contribution'] < -0.001].nsmallest(8, 'contribution')
        if len(negatives) > 0:
            print("  ➖ RÉDUISENT le risque:")
            for _, row in negatives.iterrows():
                print(f"     {row['feature']}: {row['contribution']:.3f}")

        # Probabilité de base
        if hasattr(explainer, 'expected_value'):
            base_val = explainer.expected_value
            if isinstance(base_val, (list, np.ndarray)) and len(base_val) > 1:
                base_val = base_val[1]  # Classe positive
            print(f"\n🎲 Probabilité de base: {base_val:.3f}")
            print(f"📐 Impact total des features: {sum(shap_vals):.3f}")

    except Exception as e:
        print(f"❌ Erreur SHAP: {e}")
        import traceback
        traceback.print_exc()

def run_comparative_analysis_shap(blender, X_sample):
    """Exécute une analyse comparative entre différents cas avec SHAP"""
    print(f"\n🔬 COMPARAISON ENTRE CAS")
    print("=" * 50)

    # Cas de test variés
    test_cases = [
        {"drugs": ["ZEPBOUND"], "age": 45, "sex": 1, "label": "ZEPBOUND seul"},
        {"drugs": ["ZEPBOUND", "ASPIRIN"], "age": 45, "sex": 1, "label": "ZEPBOUND + ASPIRIN"},
        {"drugs": ["PREDNISONE"], "age": 85, "sex": 2, "label": "Patient âgé PREDNISONE"},
        {"drugs": ["METFORMIN", "LISINOPRIL", "ATORVASTATIN"], "age": 68, "sex": 1, "label": "Polymédication diabète"}
    ]

    for case in test_cases:
        print(f"\n📋 {case['label']}:")

        # Faire la prédiction
        patient_features = {}
        drugs_up = [d.upper() for d in case['drugs']]
        for drug in blender.top_drugs:
            patient_features[f"drug_{drug}"] = 1 if drug in drugs_up else 0
        patient_features["age"] = case['age']
        patient_features["sex"] = case['sex']
        patient_features["nb_drugs"] = len(case['drugs'])
        patient_features["nb_reactions"] = 0
        patient_features["is_sider"] = 0
        patient_df = pd.DataFrame([patient_features])

        # Aligner avec X_sample
        for col in X_sample.columns:
            if col not in patient_df.columns:
                patient_df[col] = 0
        patient_df = patient_df[X_sample.columns]

        # Prédictions
        preds, probas = blender.predict(patient_df, min_predictions=5)
        results_list = []
        for i, reaction in enumerate(blender.top_reactions):
            if preds[0, i] == 1:
                prob = probas[0][i]
                results_list.append({'reaction': reaction, 'probability': float(prob)})

        if results_list:
            results_list.sort(key=lambda x: x['probability'], reverse=True)
            main_pred = results_list[0]
            print(f"   Réaction principale: {main_pred['reaction']} ({main_pred['probability']:.1%})")

            # Analyser le facteur clé avec SHAP
            try:
                reaction_idx = blender.top_reactions.index(main_pred['reaction'])
                explainer = shap.TreeExplainer(blender.base_models['rf1'].estimators_[reaction_idx])
                shap_values = explainer.shap_values(patient_df)

                # Extraire les valeurs SHAP
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    shap_array = shap_values[1][0]  # Classe positive, premier échantillon
                elif hasattr(shap_values, 'shape') and len(shap_values.shape) == 3:
                    shap_array = shap_values[0, :, 1]  # Format (1, n_features, 2)
                else:
                    shap_array = shap_values[0] if hasattr(shap_values, '__getitem__') else shap_values

                # Trouver le facteur le plus influent
                if hasattr(shap_array, '__len__'):
                    max_impact_idx = np.argmax(np.abs(shap_array))
                    if max_impact_idx < len(patient_df.columns):
                        max_impact_feature = patient_df.columns[max_impact_idx]
                        max_impact_value = shap_array[max_impact_idx]

                        if max_impact_feature.startswith('drug_'):
                            drug_name = max_impact_feature.replace('drug_', '')
                            status = "PRÉSENT" if patient_df[max_impact_feature].iloc[0] == 1 else "ABSENT"
                            factor_info = f"{drug_name} ({status})"
                        elif max_impact_feature == "age":
                            factor_info = f"âge ({case['age']} ans)"
                        elif max_impact_feature == "sex":
                            factor_info = f"sexe ({'Homme' if case['sex'] == 1 else 'Femme'})"
                        elif max_impact_feature == "nb_drugs":
                            factor_info = f"nb_médicaments ({len(case['drugs'])})"
                        else:
                            factor_info = max_impact_feature

                        print(f"   Facteur clé: {factor_info} (impact: {max_impact_value:+.3f})")
                    else:
                        print(f"   Facteur clé: Âge ou autre facteur démographique")
                else:
                    print(f"   Facteur clé: Analyse complexe")
            except Exception as e:
                print(f"   Facteur clé: Analyse SHAP non disponible ({e})")
        else:
            print("   Aucune prédiction significative")

def explain_predictions_comprehensive(patient_drugs, predictions, blender, X_sample, age=50, sex=1):
    """Explique les prédictions avec SHAP en format complet"""
    print(f"\n🧪 PHASE 3: ANALYSE PRINCIPALE AVEC SHAP")
    print("=" * 50)
    print(f"💊 MÉDICAMENT TESTÉ: {', '.join(patient_drugs)}")
    print(f"👤 PATIENT: Âge {age}, Sexe {'Homme' if sex == 1 else 'Femme'}")
    
    print("\n🔍 ANALYSE SHAP - Version simplifiée et robuste")
    print("=" * 60)

    # Analyser chaque prédiction avec SHAP
    for i, pred in enumerate(predictions, 1):
        reaction = pred['reaction']
        probability = pred['probability']
        
        # Utiliser la fonction d'analyse SHAP ultra simple
        analyse_shap_ultra_simple(patient_drugs, age, sex, reaction, blender, X_sample)

# Script principal pour charger le modèle et exécuter l'analyse SHAP
if __name__ == "__main__":
    # Chemin vers le modèle sauvegardé
    model_path = "C:\\Users\\msi\\Desktop\\Data_mining\\blender_final_per_label_thresholds.pkl"

    # Charger le modèle et les métadonnées
    try:
        saved_data = joblib.load(model_path)
        blender = saved_data['predictor']
        top_drugs = saved_data['vocab']['drugs']
        top_reactions = saved_data['vocab']['reactions']
        per_label_thresholds = saved_data['per_label_thresholds']
        print(f"✅ Modèle chargé depuis {model_path}")
        print(f"   • Médicaments dans le vocabulaire: {len(top_drugs)}")
        print(f"   • Réactions dans le vocabulaire: {len(top_reactions)}")
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {e}")
        exit(1)

    # Créer un X_sample fictif pour l'alignement des colonnes (basé sur top_drugs et features)
    X_sample_data = []
    for i in range(50):  # Créer 50 échantillons fictifs pour SHAP
        features = {f"drug_{drug}": np.random.choice([0, 1]) for drug in top_drugs}
        features["age"] = float(np.random.randint(18, 95))
        features["sex"] = np.random.choice([1, 2])
        features["nb_drugs"] = sum([v for k, v in features.items() if k.startswith("drug_")])
        features["nb_reactions"] = 0
        features["is_sider"] = 0
        X_sample_data.append(features)
    X_sample = pd.DataFrame(X_sample_data)

    # Cas de test à analyser
    test_cases = [
        (["ZEPBOUND", "DUPIXENT"], 45, 1, "Polymédication"),
    ]

    # Exécuter l'analyse SHAP pour le cas de test
    for drugs, age, sex, desc in test_cases:
        print(f"\n--- {desc} ---")
        patient_features = {}
        drugs_up = [d.upper() for d in drugs]
        for drug in blender.top_drugs:
            patient_features[f"drug_{drug}"] = 1 if drug in drugs_up else 0
        patient_features["age"] = age
        patient_features["sex"] = sex
        patient_features["nb_drugs"] = len(drugs)
        patient_features["nb_reactions"] = 0
        patient_features["is_sider"] = 0
        patient_df = pd.DataFrame([patient_features])

        # Aligner les colonnes avec X_sample
        for col in X_sample.columns:
            if col not in patient_df.columns:
                patient_df[col] = 0
        patient_df = patient_df[X_sample.columns]

        # Faire les prédictions
        MIN_PREDICTIONS = 5
        preds, probas = blender.predict(patient_df, min_predictions=MIN_PREDICTIONS)
        results_list = []
        for i, reaction in enumerate(blender.top_reactions):
            if preds[0, i] == 1:
                prob = probas[0][i]
                confidence = '🔴 HAUTE' if prob > 0.7 else '🟡 MOYENNE' if prob > 0.4 else '🟢 FAIBLE'
                results_list.append({'reaction': reaction, 'probability': float(prob), 'confidence': confidence})

        results_list.sort(key=lambda x: x['probability'], reverse=True)
        
        # Afficher les 10 premières prédictions
        print("Top 10 prédictions:")
        for j, pred in enumerate(results_list[:10], 1):
            print(f"   {j}. {pred['reaction']:<25} {pred['probability']:.1%} {pred['confidence']}")

        # Analyse SHAP complète pour ce cas
        explain_predictions_comprehensive(drugs, results_list[:10], blender, X_sample, age=age, sex=sex)

    # Exécuter l'analyse comparative SHAP
    run_comparative_analysis_shap(blender, X_sample)

    print("\n✅ ANALYSE SHAP TERMINÉE!")
    print("=" * 50)
    print("\n💡 CE QUE CELA SIGNIFIE:")
    print("• Les valeurs POSITIVES augmentent le risque d'effet secondaire")
    print("• Les valeurs NÉGATIVES réduisent le risque")
    print("• L'impact est mesuré en log-odds (échelle logarithmique)")
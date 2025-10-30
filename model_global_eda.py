# pipeline_final_thresholds_all_drugs.py
# Pipeline final corrigé :
# - split BEFORE augmentation
# - augment only train (targeted + diversified)
# - vocab derived from train (option top_k_drugs)
# - top_reactions_limit option
# - per-label thresholds computed on internal validation
# - final retrain on full train
# - thresholding on test using per-label thresholds
# - predict(...) garantit au moins min_predictions labels par instance
#
# Dépendances: lxml, pandas, numpy, tqdm, scikit-learn, xgboost, joblib, shap

import lxml.etree as ET
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
from collections import Counter
import os
import json
import warnings
import joblib
import time
import math
import shap
import traceback
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
np.random.seed(42)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

# Configuration des styles pour les graphiques
sns.set(style="whitegrid", font_scale=1.05)

# ---------------------------
# EDA FUNCTIONS
# ---------------------------
def perform_comprehensive_eda(df, title_prefix="Dataset"):
    """
    Réalise une analyse exploratoire complète des données
    """
    print(f"\n{'='*60}")
    print(f"📊 ANALYSE EXPLORATOIRE ({title_prefix})")
    print(f"{'='*60}")
    
    # Informations de base
    print(f"📈 Taille du dataset: {len(df)} rapports")
    print(f"💊 Nombre total de médicaments uniques: {len(set([d for sub in df['drugs'] for d in sub]))}")
    print(f"🤒 Nombre total de réactions uniques: {len(set([r for sub in df['reactions'] for r in sub]))}")
    
    # 1. Distribution de l'âge
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 2, 1)
    age_data = df[df["age"].between(0, 120)]["age"]
    sns.histplot(age_data, bins=30, kde=True)
    plt.title(f"Distribution de l'âge - {title_prefix}")
    plt.xlabel("Âge")
    plt.ylabel("Nombre de patients")
    
    # 2. Répartition par sexe
    plt.subplot(2, 2, 2)
    sex_counts = df["sex"].value_counts()
    sex_labels = {1: 'Homme', 2: 'Femme'}
    sex_counts.index = [sex_labels.get(x, x) for x in sex_counts.index]
    sns.barplot(x=sex_counts.index, y=sex_counts.values)
    plt.title(f"Répartition par sexe - {title_prefix}")
    plt.ylabel("Nombre de patients")
    
    # 3. Top 20 médicaments
    plt.subplot(2, 2, 3)
    all_drugs = [d for sub in df["drugs"] for d in sub]
    drug_counter = Counter(all_drugs)
    top20_drugs = drug_counter.most_common(20)
    drugs, counts = zip(*top20_drugs)
    sns.barplot(x=list(counts), y=list(drugs))
    plt.title(f"Top 20 médicaments - {title_prefix}")
    plt.xlabel("Nombre d'apparitions")
    
    # 4. Top 20 réactions
    plt.subplot(2, 2, 4)
    all_reactions = [r for sub in df["reactions"] for r in sub]
    reaction_counter = Counter(all_reactions)
    top20_reactions = reaction_counter.most_common(20)
    reactions, counts = zip(*top20_reactions)
    sns.barplot(x=list(counts), y=list(reactions))
    plt.title(f"Top 20 réactions - {title_prefix}")
    plt.xlabel("Nombre d'apparitions")
    
    plt.tight_layout()
    plt.show()
    
    # 5. Distribution du nombre de médicaments par patient
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    df["nb_drugs"] = df["drugs"].map(len)
    sns.histplot(df["nb_drugs"], bins=20)
    plt.title(f"Nombre de médicaments par patient - {title_prefix}")
    plt.xlabel("Nombre de médicaments")
    plt.ylabel("Nombre de patients")
    
    # 6. Distribution du nombre de réactions par patient
    plt.subplot(1, 2, 2)
    df["nb_reactions"] = df["reactions"].map(len)
    sns.histplot(df["nb_reactions"], bins=20)
    plt.title(f"Nombre de réactions par patient - {title_prefix}")
    plt.xlabel("Nombre de réactions")
    plt.ylabel("Nombre de patients")
    
    plt.tight_layout()
    plt.show()
    
    # 7. Statistiques descriptives
    print(f"\n📊 STATISTIQUES DESCRIPTIVES:")
    print(f"   • Âge moyen: {age_data.mean():.1f} ans")
    print(f"   • Médiane d'âge: {age_data.median():.1f} ans")
    print(f"   • Écart-type âge: {age_data.std():.1f} ans")
    print(f"   • Médicaments moyens par patient: {df['nb_drugs'].mean():.2f}")
    print(f"   • Réactions moyennes par patient: {df['nb_reactions'].mean():.2f}")
    print(f"   • Patients avec polypharmacie (>4 médicaments): {(df['nb_drugs'] > 4).sum()} ({(df['nb_drugs'] > 4).mean()*100:.1f}%)")
    
    # 8. Analyse par source de données
    if 'data_source' in df.columns:
        plt.figure(figsize=(8, 5))
        source_counts = df['data_source'].value_counts()
        plt.pie(source_counts.values, labels=source_counts.index, autopct='%1.1f%%', startangle=90)
        plt.title(f"Répartition par source de données - {title_prefix}")
        plt.show()
        
        print(f"\n📁 RÉPARTITION PAR SOURCE:")
        for source, count in source_counts.items():
            print(f"   • {source}: {count} rapports ({count/len(df)*100:.1f}%)")

def analyze_drug_reaction_relationships(df, top_n=15):
    """
    Analyse les relations médicaments-réactions
    """
    print(f"\n🔗 ANALYSE DES RELATIONS MÉDICAMENTS-RÉACTIONS")
    
    # Top médicaments et leurs réactions associées
    all_drugs = [d for sub in df["drugs"] for d in sub]
    drug_counter = Counter(all_drugs)
    top_drugs = [drug for drug, _ in drug_counter.most_common(top_n)]
    
    # Pour chaque médicament top, trouver les réactions les plus fréquentes
    drug_reaction_matrix = {}
    
    for drug in top_drugs:
        # Filtrer les rapports contenant ce médicament
        drug_reports = df[df["drugs"].apply(lambda x: drug in x)]
        if len(drug_reports) > 0:
            # Réactions associées à ce médicament
            drug_reactions = [r for sub in drug_reports["reactions"] for r in sub]
            reaction_counter = Counter(drug_reactions)
            top_reactions = reaction_counter.most_common(5)
            drug_reaction_matrix[drug] = top_reactions
    
    # Afficher les résultats
    print(f"\n💊 TOP {top_n} MÉDICAMENTS ET LEURS RÉACTIONS ASSOCIÉES:")
    for i, (drug, reactions) in enumerate(drug_reaction_matrix.items(), 1):
        print(f"\n{i:2d}. {drug}:")
        for j, (reaction, count) in enumerate(reactions[:3], 1):
            percentage = (count / len(df[df["drugs"].apply(lambda x: drug in x)]) * 100)
            print(f"     {reaction} ({count} fois, {percentage:.1f}%)")
    
    return drug_reaction_matrix

def plot_feature_correlations(X, y, feature_names, top_features=20):
    """
    Visualise les corrélations entre les features les plus importantes
    """
    print(f"\n📈 ANALYSE DES CORRÉLATIONS ENTRE FEATURES")
    
    # Calculer l'importance moyenne des features
    if hasattr(X, 'columns'):
        feature_importance = pd.Series(np.mean(X.values, axis=0), index=X.columns)
    else:
        feature_importance = pd.Series(np.mean(X, axis=0), index=feature_names)
    
    # Sélectionner les top features
    top_feat_names = feature_importance.nlargest(min(top_features, len(feature_importance))).index
    
    if hasattr(X, 'columns'):
        X_top = X[top_feat_names]
    else:
        X_top = pd.DataFrame(X, columns=feature_names)[top_feat_names]
    
    # Matrice de corrélation
    plt.figure(figsize=(12, 10))
    corr_matrix = X_top.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap="RdBu_r", center=0, 
                square=True, annot=False, fmt=".2f")
    plt.title("Matrice de corrélation des top features")
    plt.tight_layout()
    plt.show()
    
    # Afficher les paires les plus corrélées
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
    
    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    print(f"\n🔗 TOP 10 PAIRES DE FEATURES LES PLUS CORRÉLÉES:")
    for i, (feat1, feat2, corr) in enumerate(corr_pairs[:10], 1):
        print(f"   {i:2d}. {feat1} ↔ {feat2}: {corr:.3f}")

# ---------------------------
# 1) PARSING FAERS (ULTRA PERMISSIF)
# ---------------------------
def parse_faers_xml_ultra_permissive(file_path, max_reports=None):
    print("📥 Début du parsing FAERS...")
    try:
        context = ET.iterparse(file_path, events=("end",), recover=True)
    except Exception as e:
        print(f"Erreur ouverture FAERS: {e}")
        return pd.DataFrame()

    reports = []
    count = 0
    for event, elem in tqdm(context, desc="Parsing XML FAERS"):
        if elem.tag == "safetyreport":
            try:
                report = {}
                report["report_id"] = elem.findtext("safetyreportid")
                patient = elem.find("patient")
                drugs, reactions = [], []
                age = 50
                sex = 1

                if patient is not None:
                    age_node = patient.find("patientonsetage")
                    if age_node is not None and age_node.text:
                        age_text = age_node.text.strip()
                        if re.match(r'^[-+]?\d*\.?\d+$', age_text):
                            try:
                                age = float(age_text)
                                age_unit_node = patient.find("patientonsetageunit")
                                if age_unit_node is not None and age_unit_node.text:
                                    unit = age_unit_node.text.strip()
                                    if unit in ["802", "months", "Months"]:
                                        age = round(age / 12, 1)
                                    elif unit in ["803", "weeks", "Weeks"]:
                                        age = round(age / 52, 1)
                                    elif unit in ["804", "days", "Days"]:
                                        age = round(age / 365, 1)
                                age = max(0, min(120, age))
                            except:
                                age = 50

                    sex_node = patient.find("patientsex")
                    if sex_node is not None and sex_node.text:
                        sex_text = sex_node.text.strip()
                        if sex_text in ["1", "male", "Male", "M", "m"]:
                            sex = 1
                        elif sex_text in ["2", "female", "Female", "F", "f"]:
                            sex = 2
                        else:
                            sex = 1

                    for r in patient.findall("reaction"):
                        reaction_fields = [
                            r.find("reactionmeddrapt"),
                            r.find("reactionmeddraversionpt"),
                            r.find("reactionterm")
                        ]
                        for field in reaction_fields:
                            if field is not None and field.text and field.text.strip():
                                reaction_text = field.text.strip().upper()
                                if len(reaction_text) >= 1:
                                    reactions.append(reaction_text)
                                    break

                    for d in patient.findall("drug"):
                        drug_fields = [
                            d.find("medicinalproduct"),
                            d.find("activesubstance/activesubstancename"),
                            d.find("druggenericname"),
                            d.find("drugcharacterization"),
                            d.find("drugadministrationroute")
                        ]
                        for field in drug_fields:
                            if field is not None and field.text and field.text.strip():
                                drug_text = field.text.strip().upper()
                                clean_drug = re.sub(r'[^A-Z0-9\s\-]', '', drug_text)
                                clean_drug = re.sub(r'\s+', ' ', clean_drug).strip()
                                if len(clean_drug) >= 1:
                                    drugs.append(clean_drug)
                                    break

                def ultra_norm(s):
                    if not s:
                        return None
                    s = s.upper().strip()
                    s = re.sub(r'\s+', ' ', s)
                    return s if len(s) >= 1 else None

                drugs = [ultra_norm(x) for x in drugs if ultra_norm(x)]
                reactions = [ultra_norm(x) for x in reactions if ultra_norm(x)]

                if drugs or reactions:
                    report["drugs"] = list(dict.fromkeys(drugs))
                    report["reactions"] = list(dict.fromkeys(reactions))
                    report["age"] = age
                    report["sex"] = sex
                    report["data_source"] = "FAERS"
                    reports.append(report)

                elem.clear()
                count += 1
                if max_reports and count >= max_reports:
                    break

            except Exception:
                continue

    print(f"✅ FAERS parsing terminé: {len(reports)} rapports valides")
    return pd.DataFrame(reports)

# ---------------------------
# 2) CHARGEMENT SIDER
# ---------------------------
def load_sider_fixed(sider_file_path):
    print("=== CHARGEMENT SIDER ===")
    try:
        with open(sider_file_path, "r", encoding="utf-8") as f:
            sider_data = json.load(f)
    except Exception as e:
        print(f"Erreur lecture SIDER: {e}")
        return pd.DataFrame()

    sider_samples = []
    for drug_info in tqdm(sider_data, desc="Processing SIDER"):
        try:
            drug_name = drug_info["drug_info"]["display_name"].upper()
            side_effects = []
            for effect in drug_info.get("side_effects", []):
                reaction_name = effect["side_effect"].strip().upper()
                if len(reaction_name) >= 2:
                    side_effects.append(reaction_name)
            if side_effects:
                for _ in range(2):
                    sample = {
                        "drugs": [drug_name],
                        "reactions": side_effects,
                        "age": float(max(18, min(80, np.random.normal(50, 15)))),
                        "sex": int(np.random.choice([1, 2])),
                        "data_source": "SIDER",
                        "report_id": f"SIDER_{drug_name}_{_}"
                    }
                    sider_samples.append(sample)
        except Exception:
            continue

    sider_df = pd.DataFrame(sider_samples)
    print(f"✅ SIDER chargé: {len(sider_df)} échantillons")
    return sider_df

# ---------------------------
# 3) AUGMENTATION CIBLÉE (ONLY TRAIN) - diversification
# ---------------------------
def targeted_augment_train_only(df_combined, min_count_rare=20, rare_multiplier=4, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    # strat flag: presence of any reaction
    strat_flag = df_combined['reactions'].apply(lambda r: 1 if len(r) > 0 else 0).astype(int)

    df_train, df_test = train_test_split(df_combined, test_size=test_size, random_state=random_state, stratify=strat_flag)
    all_drugs_train = [d for sub in df_train['drugs'] for d in sub]
    drug_counts = Counter(all_drugs_train)
    rare_drugs = {drug for drug, c in drug_counts.items() if c < min_count_rare}

    print(f"🔧 TRAIN size: {len(df_train)}, TEST size: {len(df_test)}. Rare drugs in train (<{min_count_rare}): {len(rare_drugs)}")

    augmented = []
    for idx, row in tqdm(df_train.iterrows(), total=len(df_train), desc='Augment Train'):
        augmented.append(row.copy())
        present_rare = [d for d in row['drugs'] if d in rare_drugs]
        repeats = rare_multiplier if present_rare else 1

        for _i in range(repeats - 1):
            new = row.copy()
            # jitter age
            if pd.notna(new.get('age')):
                new['age'] = float(max(18, min(95, round(new['age'] + np.random.randint(-4, 5)))))
            # flip sex sometimes
            if np.random.random() < 0.12:
                new['sex'] = 3 - int(new.get('sex', 1))
            # mutate reactions sometimes: remove or duplicate one element or shuffle
            if new['reactions'] and np.random.random() < 0.25:
                rlist = new['reactions'][:]
                if np.random.random() < 0.5 and len(rlist) > 1:
                    rlist.pop(np.random.randint(0, len(rlist)))
                else:
                    rlist = rlist + [rlist[np.random.randint(0, len(rlist))]]
                new['reactions'] = list(dict.fromkeys(rlist))
            # occasionally add a co-med (pick a popular drug)
            if np.random.random() < 0.08:
                popular = [d for d, c in drug_counts.most_common(50)]
                if popular:
                    add_dr = np.random.choice(popular)
                    if add_dr not in new['drugs']:
                        new['drugs'] = list(dict.fromkeys(new['drugs'] + [add_dr]))
            augmented.append(new)

    train_aug_df = pd.DataFrame(augmented).reset_index(drop=True)

    # Build signatures of train augmented rows to remove identical items from test
    def row_signature(r):
        return (tuple(sorted(r['drugs'])), tuple(sorted(r['reactions'])), round(float(r.get('age', 50)), 1), int(r.get('sex', 1)))

    train_sigs = set(row_signature(r) for idx, r in train_aug_df.iterrows())

    # filter test to remove rows whose signature is in train_sigs (prevent leakage)
    test_filtered_rows = []
    removed = 0
    for idx, r in df_test.iterrows():
        if row_signature(r) in train_sigs:
            removed += 1
            continue
        test_filtered_rows.append(r)
    test_clean_df = pd.DataFrame(test_filtered_rows).reset_index(drop=True)
    print(f"✅ Après nettoyage, TEST size clean: {len(test_clean_df)} (removed {removed} potential dupes)")

    return train_aug_df, test_clean_df

# ---------------------------
# 4) FEATURES FROM VOCAB
# ---------------------------
def create_features_from_vocab(df, all_drugs, top_reactions, dataset_name="DATA"):
    X_list = []
    reactions_list = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f'Features {dataset_name}'):
        try:
            features = {}
            drug_count = 0
            for drug in all_drugs:
                present = 1 if drug in row['drugs'] else 0
                features[f"drug_{drug}"] = present
                drug_count += present
            if drug_count == 0:
                continue
            features['age'] = row.get('age', 50)
            features['sex'] = row.get('sex', 1)
            features['nb_drugs'] = len(row['drugs'])
            features['nb_reactions'] = len(row['reactions'])
            features['is_sider'] = 1 if row.get('data_source', '').upper() == 'SIDER' else 0
            X_list.append(features)
            reactions_list.append(row['reactions'])
        except Exception:
            continue

    X = pd.DataFrame(X_list).fillna(0)
    mlb = MultiLabelBinarizer(classes=top_reactions)
    Y = mlb.fit_transform(reactions_list)
    return X, Y

# ---------------------------
# 5) BLENDING PREDICTOR
# ---------------------------
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

# ---------------------------
# 6) FONCTIONS SHAP AMÉLIORÉES
# ---------------------------
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

def analyze_with_shap_simple(blender, X_test, sample_size=50):
    """Analyse SHAP simplifiée et robuste pour FixedBlendingPredictor"""
    print("\n🔍 ANALYSE SHAP - Version simplifiée et robuste")
    print("=" * 60)

    try:
        # Préparer les données pour SHAP
        if sample_size < len(X_test):
            X_sample = X_test.sample(n=sample_size, random_state=42)
        else:
            X_sample = X_test

        print("✅ SHAP importé avec succès")
        print("📊 Calcul des valeurs SHAP...")

        # Utiliser le modèle Random Forest de base pour SHAP
        rf_model = blender.base_models['rf1']

        # Calcul SHAP pour toutes les classes
        explainer = shap.TreeExplainer(rf_model.estimators_[0])  # Premier classifieur RF
        shap_values = explainer.shap_values(X_sample)

        print(f"🔧 Debug - Type SHAP: {type(shap_values)}, Forme: {np.array(shap_values).shape if hasattr(shap_values, 'shape') else 'N/A'}")

        return explainer, shap_values, X_sample

    except ImportError:
        print("❌ SHAP n'est pas installé. Installation: pip install shap")
        return None, None, None
    except Exception as e:
        print(f"❌ Erreur analyse SHAP: {e}")
        return None, None, None

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

# ---------------------------
# 7) PIPELINE MAIN (with per-label thresholds)
# ---------------------------
def run_pipeline_final(faers_xml_path, sider_json_path,
                       max_faers_reports=30000,
                       min_count_rare=20, rare_multiplier=4,
                       test_size=0.2, random_state=42,
                       top_reactions_limit=500, top_k_drugs=None):
    print("🎯 DÉMARRAGE SYSTÈME AVEC FILTRAGE RELÂCHÉ")
    print("=" * 50)
    print("🚀 LANCEMENT PIPELINE HYBRIDE OPTIMISÉ - FILTRAGE RELÂCHÉ")
    print("=" * 60)

    # Chargement des données
    sider_df = load_sider_fixed(sider_json_path)
    faers_df = parse_faers_xml_ultra_permissive(faers_xml_path, max_reports=max_faers_reports)

    if len(sider_df) == 0 and len(faers_df) == 0:
        raise RuntimeError("Aucune donnée SIDER ni FAERS trouvée")

    combined_df = pd.concat([sider_df, faers_df], ignore_index=True)
    print(f"📊 Données brutes - SIDER: {len(sider_df)}, FAERS: {len(faers_df)}, TOTAL: {len(combined_df)}")

    # ==================== EDA SUR LES DONNÉES BRUTES ====================
    print("\n📊 PHASE EDA: ANALYSE DES DONNÉES BRUTES")
    perform_comprehensive_eda(combined_df, "Données Brutes (FAERS + SIDER)")
    analyze_drug_reaction_relationships(combined_df, top_n=15)

    # split BEFORE augmentation and augment only train
    train_aug, test_clean = targeted_augment_train_only(combined_df, min_count_rare=min_count_rare,
                                                        rare_multiplier=rare_multiplier, test_size=test_size,
                                                        random_state=random_state)

    # ==================== EDA SUR LES DONNÉES D'ENTRAÎNEMENT ====================
    print("\n📊 PHASE EDA: ANALYSE DES DONNÉES D'ENTRAÎNEMENT")
    perform_comprehensive_eda(train_aug, "Données d'Entraînement (Augmentées)")

    # ==================== EDA SUR LES DONNÉES DE TEST ====================
    print("\n📊 PHASE EDA: ANALYSE DES DONNÉES DE TEST")
    perform_comprehensive_eda(test_clean, "Données de Test")

    # Derive vocab from train_aug
    drug_counts_train = Counter([d for sub in train_aug['drugs'] for d in sub])
    all_drugs = sorted(list({d for sub in train_aug['drugs'] for d in sub}))
    if top_k_drugs is not None:
        top_k = [d for d, _ in drug_counts_train.most_common(top_k_drugs)]
        all_drugs = sorted(top_k)
        print(f"🔧 Limitation vocab drugs -> top {top_k_drugs} (used {len(all_drugs)})")

    all_reactions_counts = Counter([r for sub in train_aug['reactions'] for r in sub])
    if top_reactions_limit is not None:
        top_reactions = [r for r, _ in all_reactions_counts.most_common(top_reactions_limit)]
    else:
        top_reactions = list(all_reactions_counts.keys())
    print(f"🔤 Vocab (train-based): drugs={len(all_drugs)}, reactions={len(top_reactions)} (top {top_reactions_limit})")

    # Create features for train_aug and test_clean using same vocab
    X_train_full, Y_train_full = create_features_from_vocab(train_aug, all_drugs, top_reactions, dataset_name="TRAIN_FULL")
    X_test, Y_test = create_features_from_vocab(test_clean, all_drugs, top_reactions, dataset_name="TEST")

    # align test columns
    for col in X_train_full.columns:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[X_train_full.columns]

    print(f"✅ Shapes: X_train_full={X_train_full.shape}, Y_train_full={Y_train_full.shape}, X_test={X_test.shape}, Y_test={Y_test.shape}")

    # ==================== EDA SUR LES FEATURES ====================
    print("\n📊 PHASE EDA: ANALYSE DES FEATURES")
    plot_feature_correlations(X_train_full, Y_train_full, X_train_full.columns.tolist(), top_features=20)

    # --- compute per-label thresholds using an internal split on TRAIN_FULL ---
    X_train_inner, X_val_inner, y_train_inner, y_val_inner = train_test_split(
        X_train_full, Y_train_full, test_size=0.15, random_state=random_state, stratify=(Y_train_full.sum(axis=1) > 0).astype(int)
    )
    print(f"🔬 Calibration thresholds: inner train {len(X_train_inner)}, val {len(X_val_inner)}")

    # Train blender on inner train
    blender_inner = FixedBlendingPredictor()
    blender_inner.create_base_models()
    blender_inner.train_base_models(X_train_inner, y_train_inner)
    blender_inner.train_meta_model(X_train_inner, y_train_inner)

    # Compute probabilities on val
    probas_val = blender_inner.predict_proba_blended(X_val_inner)  # (n_val, n_labels)

    # For each label, find best threshold maximizing F1 on val
    per_label_thresholds = []
    n_labels = probas_val.shape[1]
    print("🔎 Calculating per-label thresholds (this may take a while)...")
    for j in range(n_labels):
        best_t = 0.05
        best_f1 = -1.0
        probs_j = probas_val[:, j]
        true_j = y_val_inner[:, j]
        candidates = np.concatenate([np.linspace(0.01, 0.1, 10), np.linspace(0.12, 0.5, 10)])
        for t in candidates:
            pred_j = (probs_j > t).astype(int)
            f1 = f1_score(true_j, pred_j, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        per_label_thresholds.append(best_t)
    per_label_thresholds = np.array(per_label_thresholds)
    print("✅ Per-label thresholds calculated (sample):", per_label_thresholds[:10])

    # --- Retrain final blender on full train (to have final model trained on all train data) ---
    blender_final = FixedBlendingPredictor()
    blender_final.create_base_models()
    blender_final.train_base_models(X_train_full, Y_train_full)
    blender_final.train_meta_model(X_train_full, Y_train_full)
    blender_final.feature_names = X_train_full.columns.tolist()
    blender_final.top_drugs = all_drugs
    blender_final.top_reactions = top_reactions
    blender_final.per_label_thresholds = per_label_thresholds

    # Evaluate on test using per-label thresholds
    probas_test = blender_final.predict_proba_blended(X_test)
    preds_test = (probas_test > per_label_thresholds[np.newaxis, :]).astype(int)
    micro_f1 = f1_score(Y_test, preds_test, average='micro', zero_division=0)
    exact_match = accuracy_score(Y_test, preds_test)

    # baseline
    simple = OneVsRestClassifier(xgb.XGBClassifier(n_estimators=100, random_state=42), n_jobs=-1)
    simple.fit(X_train_full, Y_train_full)
    y_pred_simple = simple.predict(X_test)
    f1_simple = f1_score(Y_test, y_pred_simple, average='micro', zero_division=0)
    improvement = ((micro_f1 - f1_simple) / f1_simple) * 100 if f1_simple > 0 else 0

    results = {
        'micro_f1': micro_f1,
        'exact_match': exact_match,
        'improvement_vs_simple': improvement,
        'n_train': len(X_train_full),
        'n_test': len(X_test),
        'n_drugs_vocab': len(all_drugs),
        'n_reactions_vocab': len(top_reactions)
    }

    print("\n🎯 PERFORMANCE FINALE (with per-label thresholds):")
    print(f"   • F1-micro: {micro_f1:.4f}")
    print(f"   • Exact match: {exact_match:.4f}")
    print(f"   • Amélioration vs simple: {improvement:+.1f}%")

    return blender_final, results, train_aug, test_clean, per_label_thresholds, X_train_full, X_test, Y_test

# ---------------------------
# 8) MAIN
# ---------------------------
if __name__ == "__main__":
    faers_xml_path = "combined_ADR25Q2 (1).xml"
    sider_json_path = "donnees_sider_completes (3).json"

    blender, results, train_aug, test_clean, per_label_thresholds, X_train_full, X_test, Y_test = run_pipeline_final(
        faers_xml_path, sider_json_path,
        max_faers_reports=30000,
        min_count_rare=20,
        rare_multiplier=4,
        test_size=0.2,
        random_state=42,
        top_reactions_limit=500,   # <-- set to None to include toutes les réactions (mais lent)
        top_k_drugs=None           # <-- set to int (e.g. 1000) to limit drug features for speed/memory
    )

    print("\n📌 Résultats summary:")
    for k, v in results.items():
        print(f"   {k}: {v}")

    # quick manual tests (assure au moins min_predictions par patient)
    MIN_PREDICTIONS = 5

    test_cases = [
        (["ZEPBOUND", "DUPIXENT"], 45, 1, "Polymédication"),
    ]

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

        # align
        for col in blender.feature_names:
            if col not in patient_df.columns:
                patient_df[col] = 0
        patient_df = patient_df[blender.feature_names]

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

    # Analyse SHAP complète
    explainer, shap_values, X_sample = analyze_with_shap_simple(blender, X_test, sample_size=50)
    
    if explainer is not None and shap_values is not None:
        # Exécuter l'analyse comparative
        run_comparative_analysis_shap(blender, X_sample)

        # Exécuter l'analyse SHAP complète pour le cas de test
        test_case = test_cases[0]  # Prenons le premier cas de test
        drugs, age, sex, desc = test_case
        
        # Obtenir les prédictions pour ce cas
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
        for col in blender.feature_names:
            if col not in patient_df.columns:
                patient_df[col] = 0
        patient_df = patient_df[blender.feature_names]
        
        preds, probas = blender.predict(patient_df, min_predictions=MIN_PREDICTIONS)
        results_list = [
            {'reaction': blender.top_reactions[i], 'probability': float(probas[0][i])}
            for i in range(len(blender.top_reactions)) if preds[0, i] == 1
        ]
        results_list.sort(key=lambda x: x['probability'], reverse=True)
        
        # Exécuter l'analyse SHAP complète
        explain_predictions_comprehensive(drugs, results_list[:10], blender, X_sample, age=age, sex=sex)

    print("\n✅ ANALYSE TERMINÉE!")
    print("=" * 50)
    print("\n💡 CE QUE CELA SIGNIFIE:")
    print("• Les valeurs POSITIVES augmentent le risque d'effet secondaire")
    print("• Les valeurs NÉGATIVES réduisent le risque")
    print("• L'impact est mesuré en log-odds (échelle logarithmique)")

    print(f"\n📊 RAPPORT FINAL:")
    print("=" * 30)
    print(f"• Performance F1: {results['micro_f1']:.4f}")
    print(f"• Médicaments modélisés: {results['n_drugs_vocab']}")
    print(f"• Effets secondaires: {results['n_reactions_vocab']}")
    print(f"• Explicabilité: ✅ SHAP intégré")

    # Save final model + vocab + thresholds
    try:
        out = {
            'predictor': blender,
            'results': results,
            'vocab': {'drugs': blender.top_drugs, 'reactions': blender.top_reactions},
            'per_label_thresholds': per_label_thresholds
        }
        joblib.dump(out, 'blender_final_per_label_thresholds.pkl')
        print("💾 Modèle sauvegardé: blender_final_per_label_thresholds.pkl")
    except Exception as e:
        print(f"Erreur sauvegarde: {e}")
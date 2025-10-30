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
# Dépendances: lxml, pandas, numpy, tqdm, scikit-learn, xgboost, joblib

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

warnings.filterwarnings('ignore')
np.random.seed(42)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

# ---------------------------
# 1) PARSING FAERS (ULTRA PERMISSIF)
# ---------------------------
def parse_faers_xml_ultra_permissive(file_path, max_reports=None):
    print("📥 Début du parsing FAERS (version ultra-permissive)...")
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
# 6) PIPELINE MAIN (with per-label thresholds)
# ---------------------------
def run_pipeline_final(faers_xml_path, sider_json_path,
                       max_faers_reports=30000,
                       min_count_rare=20, rare_multiplier=4,
                       test_size=0.2, random_state=42,
                       top_reactions_limit=500, top_k_drugs=None):
    print("🚀 LANCEMENT PIPELINE FINAL (split BEFORE augmentation + per-label thresholds)")

    sider_df = load_sider_fixed(sider_json_path)
    faers_df = parse_faers_xml_ultra_permissive(faers_xml_path, max_reports=max_faers_reports)

    if len(sider_df) == 0 and len(faers_df) == 0:
        raise RuntimeError("Aucune donnée SIDER ni FAERS trouvée")

    combined_df = pd.concat([sider_df, faers_df], ignore_index=True)
    print(f"📊 Données brutes - SIDER: {len(sider_df)}, FAERS: {len(faers_df)}, TOTAL: {len(combined_df)}")

    # split BEFORE augmentation and augment only train
    train_aug, test_clean = targeted_augment_train_only(combined_df, min_count_rare=min_count_rare,
                                                        rare_multiplier=rare_multiplier, test_size=test_size,
                                                        random_state=random_state)

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

    return blender_final, results, train_aug, test_clean, per_label_thresholds

# ---------------------------
# 7) MAIN
# ---------------------------
if __name__ == "__main__":
    faers_xml_path = "combined_ADR25Q2 (1).xml"
    sider_json_path = "donnees_sider_completes (3).json"

    blender, results, train_aug, test_clean, per_label_thresholds = run_pipeline_final(
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
        (["METFORMIN", "INSULIN"], 52, 1, "Diabète"),
        (["ZEPBOUND"], 42, 2, "Obésité"),
        (["DUPIXENT"], 35, 1, "Dermatite")
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
        for j, pred in enumerate(results_list[:10], 1):
            print(f"   {j}. {pred['reaction']:<25} {pred['probability']:.1%} {pred['confidence']}")

    # Save final model + vocab + thresholds
    try:
        out = {
            'predictor': blender,
            'results': results,
            'vocab': {'drugs': blender.top_drugs, 'reactions': blender.top_reactions},
            'per_label_thresholds': per_label_thresholds
        }
        joblib.dump(out, 'blender_final_per_label_thresholds.pkl')
        print("\n💾 Modèle sauvegardé: blender_final_per_label_thresholds.pkl")
    except Exception as e:
        print(f"Erreur sauvegarde: {e}")

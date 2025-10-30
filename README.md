# DrugSideEffects-Prediction-XAI
Ce projet développe une solution complète de pharmacovigilance assistée par IA, combinant Machine Learning, OCR, Explicabilité (XAI) et probabilistic sampling pour prédire les effets indésirables médicamenteux (EIM) à partir de profils patients et d’ordonnances médicales.

Il intègre un pipeline de bout en bout :

Extraction et fusion de données FAERS + Web Scraping

Prétraitement et modélisation (Blending/Stacking + probabilistic sampling pour corriger le déséquilibre des classes)

Analyse explicative via SHAP

Lecture automatique d’ordonnances (YOLOv8 + TrOCR)

Interface utilisateur interactive (Streamlit)

🔬 Objectif : renforcer la sécurité patient et l’aide à la décision clinique grâce à une IA transparente, robuste et explicable.

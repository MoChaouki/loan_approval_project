# Importation des bibliothèques nécessaires
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les données
data = pd.read_csv('donnees_pret_avance.csv', encoding='latin1')

# Afficher un aperçu des données
print("Aperçu des données :")
print(data.head())

# Informations générales sur le dataset
print("\nInformations sur le dataset :")
print(data.info())

# Statistiques descriptives
print("\nStatistiques descriptives :")
print(data.describe())

###################################
# Gestion des valeurs manquantes
###################################

# Remplacer les valeurs manquantes pour les colonnes numériques
cols_num = ['Age', 'Revenu', 'Montant_Pret', 'Score_Credit', 'Nb_Prets_Existant']
for col in cols_num:
    data[col] = data[col].fillna(data[col].median())  # Remplacement par la médiane

# Remplacer les valeurs manquantes pour les colonnes catégoriques
cols_cat = ['Statut_Emploi', 'Niveau_Education']
for col in cols_cat:
    data[col] = data[col].fillna(data[col].mode()[0])  # Remplacement par le mode

# Remplacer les valeurs manquantes par la valeur la plus fréquente (mode)
data['Statut_Marital'] = data['Statut_Marital'].fillna(data['Statut_Marital'].mode()[0])

# Vérification après remplacement
print("\nValeurs manquantes après traitement :")
print(data.isnull().sum())

###################################
# Encodage des variables catégorielles
###################################

# Encodage One-Hot Encoding
cols_cat = ['Statut_Emploi', 'Niveau_Education', 'Statut_Marital']
data_encoded = pd.get_dummies(data, columns=cols_cat, drop_first=True)

# Vérification de l'encodage
print("\nAperçu des données après encodage :")
print(data_encoded.head())

###################################
# Ingénierie des caractéristiques
###################################

# Création de la variable 'Ratio_Endettement'
data_encoded['Ratio_Endettement'] = data_encoded['Montant_Pret'] / data_encoded['Revenu']

# Remplacer les valeurs infinies et NaN
data_encoded['Ratio_Endettement'].replace([float('inf'), -float('inf')], 0)
data_encoded['Ratio_Endettement'].fillna(0)

# Vérification de la nouvelle variable
print("\nAperçu des données après ajout de Ratio_Endettement :")
print(data_encoded[['Montant_Pret', 'Revenu', 'Ratio_Endettement']].head())

###################################
# Division en ensembles d’entraînement et de test
###################################

# Séparer la cible (y) et les variables explicatives (X)
X = data_encoded.drop('Pret_Approuve', axis=1)  # Toutes les colonnes sauf la cible
y = data_encoded['Pret_Approuve']  # La colonne cible

# Division en 70% entraînement et 30% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Vérification des tailles des ensembles
print("\nTaille de l'ensemble d'entraînement :", X_train.shape)
print("Taille de l'ensemble de test :", X_test.shape)

###################################
# Traitement du Déséquilibre des Classes
###################################

# Analyser la distribution de la variable cible
print("Répartition des classes dans Pret_Approuve :")
print(y.value_counts())

# Visualiser la distribution
sns.countplot(x=y, palette="viridis")
plt.title("Distribution des classes dans Pret_Approuve")
plt.xlabel("Pret_Approuve (0 : Refusé, 1 : Approuvé)")
plt.ylabel("Nombre d'observations")
plt.show()

# Maintenant Nous allons appliquer au moins deux techniques de rééchantillonnage #
# Option 1 : Sous-échantillonnage

from imblearn.under_sampling import RandomUnderSampler
# Sous-échantillonnage
under_sampler = RandomUnderSampler(random_state=42)
X_under, y_under = under_sampler.fit_resample(X_train, y_train)

# Vérification
print("\nRépartition des classes après sous-échantillonnage :")
print(y_under.value_counts())

## Option 2 : Sur-échantillonnage

from imblearn.over_sampling import RandomOverSampler
# Sur-échantillonnage
over_sampler = RandomOverSampler(random_state=42)
X_over, y_over = over_sampler.fit_resample(X_train, y_train)

# Vérification
print("\nRépartition des classes après sur-échantillonnage :")
print(y_over.value_counts())

## Option 3 : SMOTE

from imblearn.over_sampling import SMOTE
# Appliquer SMOTE
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_train, y_train)

# Vérification de la répartition des classes
print("\nRépartition des classes après SMOTE :")
print(y_smote.value_counts())

###################################
# Modélisation
###################################

"""
On peut maintenant les Entraîner sur les trois datasets équilibrés.

Datasets utilisés :
- Sous-échantillonné.
- SMOTE.
"""

# Importation des bibliothèques
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np

# Initialiser les modèles
models = {
    "Régression Logistique": LogisticRegression(max_iter=1000, random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42),
    "Perceptron Multicouche": MLPClassifier(hidden_layer_sizes=(50, 30), max_iter=2000, learning_rate_init=0.01, random_state=42)
}

# Jeux de données : Sous-échantillonnage et SMOTE
datasets = {
    "Sous-échantillonnage": (X_under, y_under),
    "SMOTE": (X_smote, y_smote)
}

# Grille d'hyperparamètres pour XGBoost
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}

# Fonction pour standardiser les données
def standardize_data(X_train):
    scaler = StandardScaler()
    return scaler.fit_transform(X_train)

# Fonction pour tracer les courbes ROC et Precision-Recall
def plot_curves(y_true, y_proba, model_name, dataset_name):
    # Courbe ROC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC {model_name} ({dataset_name})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Courbe ROC - {model_name} - {dataset_name}")
    plt.legend()
    plt.show()
    
    # Courbe Precision-Recall
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_proba)
    plt.figure()
    plt.plot(recall_vals, precision_vals, label=f"PR {model_name} ({dataset_name})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Courbe Precision-Recall - {model_name} - {dataset_name}")
    plt.legend()
    plt.show()

# Entraîner chaque modèle sur les jeux de données sélectionnés
print("### Comparaison des modèles pour Sous-échantillonnage et SMOTE ###\n")

for model_name, model in models.items():
    print(f"\nModèle : {model_name}")
    for dataset_name, (X_res, y_res) in datasets.items():
        print(f"\n### Entraîment sur : {dataset_name} ###")
        
        # Standardisation des données
        X_res_standardized = standardize_data(X_res)
        
        if model_name == "XGBoost":
            # XGBoost : Optimisation des hyperparamètres
            print(f"\nOptimisation des hyperparamètres pour XGBoost sur {dataset_name}...")
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', verbose=2)
            grid_search.fit(X_res_standardized, y_res)
            best_model = grid_search.best_estimator_
            print(f"Meilleurs paramètres pour XGBoost ({dataset_name}) : {grid_search.best_params_}")
            print(f"Meilleur score : {grid_search.best_score_:.4f}")
        else:
            # Validation croisee pour LogisticRegression et MLPClassifier
            best_model = model
            best_model.fit(X_res_standardized, y_res)
            
        # Prédiction
        y_pred = best_model.predict(X_res_standardized)
        if hasattr(best_model, "predict_proba"):
            y_proba = best_model.predict_proba(X_res_standardized)[:, 1]
        else:
            y_proba = None
        
        # Calcul des métriques
        precision = precision_score(y_res, y_pred)
        recall = recall_score(y_res, y_pred)
        f1 = f1_score(y_res, y_pred)
        auc = roc_auc_score(y_res, y_proba) if y_proba is not None else None
        
        print(f"{dataset_name} - Précision : {precision:.4f} | Rappel : {recall:.4f} | F1-Score : {f1:.4f} | AUC : {auc:.4f}")
        
        # Tracer les courbes ROC et Precision-Recall
        if y_proba is not None:
            plot_curves(y_res, y_proba, model_name, dataset_name)

###########################
# Interprétation et Explicabilité
###########################

# Vérification et conversion des types de données
print("\nTypes de données avant SHAP :")
print(X_res.dtypes)

# Conversion en float64 si nécessaire
X_res = X_res.astype(float)
print("\nTypes de données après conversion :")
print(X_res.dtypes)

# SHAP Explainer
import shap
import matplotlib.pyplot as plt

# Identifiez le meilleur modèle
best_xgb = grid_search.best_estimator_

# Importance des variables avec XGBoost (Gini Importance)
plt.figure(figsize=(10, 6))
plt.barh(X_res.columns, best_xgb.feature_importances_)
plt.xlabel("Importance des variables")
plt.title(f"Importance des variables - XGBoost ({dataset_name})")
plt.show()

# SHAP Values
explainer = shap.Explainer(best_xgb, X_res)  # Utilisez les données converties
shap_values = explainer(X_res)

# Visualiser les SHAP values
shap.summary_plot(shap_values, X_res)


####################################
# Sauvegarder le Meilleur Modèle
####################################

import joblib

# Sauvegarder le modèle XGBoost
best_xgb = grid_search.best_estimator_
joblib.dump(best_xgb, 'meilleur_modele_xgb.pkl')
print("Le meilleur modèle XGBoost a été sauvegardé avec succès !")

####### Charger le Modèle et Prédire les Probabilités
# Charger le modèle sauvegardé
def charger_modele():
    import joblib
    model = joblib.load('meilleur_modele_xgb.pkl')
    return model

# Charger le modèle sauvegardé
model = joblib.load("meilleur_modele_xgb.pkl")

# Liste des colonnes attendues par le modèle (les colonnes utilisées lors de l'entraînement)
colonnes_modele = [
    'Age', 'Revenu', 'Montant_Pret', 'Score_Credit', 'Nb_Prets_Existant',
    'Statut_Emploi_Employ', 'Statut_Emploi_Etudiant', 'Statut_Emploi_Indépendant',
    'Statut_Emploi_Retrait', 'Niveau_Education_Licence', 'Niveau_Education_Master',
    'Niveau_Education_Secondaire', 'Statut_Marital_Divorcé', 'Statut_Marital_Marié',
    'Statut_Marital_Veuf', 'Ratio_Endettement'
]

# Fonction pour préparer les caractéristiques du client
def preparer_features_client(caracteristiques_client, colonnes_modele):
    """
    Prépare les caractéristiques du client pour le modèle :
    - Crée un DataFrame avec les colonnes attendues
    - Remplit les colonnes manquantes avec des valeurs par défaut (0)
    """
    # Créer un DataFrame à partir des caractéristiques client
    features_df = pd.DataFrame([caracteristiques_client])
    
    # Ajouter les colonnes manquantes avec la valeur 0
    for col in colonnes_modele:
        if col not in features_df:
            features_df[col] = 0  # Colonnes manquantes remplies avec 0
    
    # Réorganiser les colonnes dans l'ordre attendu par le modèle
    features_df = features_df[colonnes_modele]
    return features_df

# Exemple de caractéristiques d'un client
caracteristiques_client = {
    'Age': 45,
    'Revenu': 60000,
    'Montant_Pret': 20000,
    'Score_Credit': 750,
    'Nb_Prets_Existant': 2,
    'Statut_Emploi_Employ': 1,
    'Niveau_Education_Licence': 0,
    'Statut_Marital_Veuf': 0,
    'Ratio_Endettement': 0.33
}

# Préparer les caractéristiques du client
features_df = preparer_features_client(caracteristiques_client, colonnes_modele)

# Prédire la probabilité d'approbation du prêt
probabilite = model.predict_proba(features_df)[:, 1]

# Afficher le résultat
print("Caractéristiques du client :")
print(features_df)
print(f"\nProbabilité d'approbation du prêt : {probabilite[0]:.2f}")
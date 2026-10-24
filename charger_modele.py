import mlflow
import mlflow.sklearn      # pour recharger un modèle scikit-learn
import pandas as pd

# 1. Retrouver le meilleur run (même logique que lire_runs.py)
runs = mlflow.search_runs(experiment_names=["churn-prediction"],
                          order_by=["metrics.f1 DESC"])
run_id = runs.iloc[0]["run_id"]

# 2. Recharger le modèle : "runs:/<id>/model" = le dossier 'model' de ce run
model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
print("Modele recharge :", type(model).__name__)
print("Nombre de variables attendues :", model.n_features_in_)

# 3. Nouveau client (abonnement Standard, Premium = False)
client = pd.DataFrame([{
    "age": 35, "salaire": 2500, "anciennete": 1, "consommation": 10,
    "abonnement_Premium": False, "abonnement_Standard": True,
}])

# 4. Garder les colonnes connues du modèle, dans le bon ordre
client = client[list(model.feature_names_in_)]

# 5. Prédire : 1 = le client va partir, 0 = il reste
print("Le client va-t-il partir ? ->", "OUI" if model.predict(client)[0] == 1 else "NON")
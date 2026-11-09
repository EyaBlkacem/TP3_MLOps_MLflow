import mlflow                       # module principal de MLflow
from mlflow import MlflowClient     # le client qui gère le Registry

# Se connecter à la même base que le TP3
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Retrouver le meilleur run (f1 décroissant)
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    order_by=["metrics.f1 DESC"])
best = runs.iloc[0]
run_id = best["run_id"]
print("Meilleur run :", best["tags.mlflow.runName"], "| id :", run_id[:8], "...")

# Enregistrer le modèle de ce run sous le nom churn-model (crée une version)
result = mlflow.register_model(f"runs:/{run_id}/model", "churn-model")
print("Modele enregistre : churn-model | version", result.version)

# Poser l'alias « champion » sur cette version
client = MlflowClient()
client.set_registered_model_alias("churn-model", "champion", result.version)
print("Alias 'champion' -> version", result.version)
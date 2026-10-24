# ===== 1. Importer les bibliothèques =====
import pandas as pd                      # lire le CSV
import mlflow                            # module principal de MLflow
import mlflow.sklearn                    # partie dédiée à scikit-learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
# 4 façons de mesurer la qualité du modèle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ===== 2. Charger les données (une seule fois) =====
df = pd.read_csv("churn_data.csv")       # 500 clients
y = df["churn"]                          # 1 = part, 0 = reste

# ===== 3. Préparer les données pour une liste de colonnes =====
def preparer(colonnes):
    # get_dummies : 'abonnement' (texte) -> colonnes 0/1 si présente
    X = pd.get_dummies(df[colonnes], drop_first=True)
    # 80 % apprentissage / 20 % test, même découpage à chaque fois
    return train_test_split(X, y, test_size=0.2, random_state=42)

# ===== 4. Lancer UN essai et l'enregistrer dans MLflow =====
def lancer_essai(nom, n_estimators, max_depth, colonnes):
    X_train, X_test, y_train, y_test = preparer(colonnes)

    with mlflow.start_run(run_name=nom):          # ouvre un run au nom de l'essai
        model = RandomForestClassifier(
            n_estimators=n_estimators,            # nombre d'arbres
            max_depth=max_depth,                  # profondeur maximale
            random_state=42)                      # résultat reproductible
        model.fit(X_train, y_train)               # entraînement
        pred = model.predict(X_test)              # prédictions sur les clients de test

        # Paramètres : les réglages de cet essai
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("features", ",".join(colonnes))

        # Métriques : les résultats (zero_division=0 évite une erreur
        # si le modèle ne prédit aucun départ)
        mlflow.log_metric("accuracy", accuracy_score(y_test, pred))
        mlflow.log_metric("precision", precision_score(y_test, pred, zero_division=0))
        mlflow.log_metric("recall", recall_score(y_test, pred, zero_division=0))
        mlflow.log_metric("f1", f1_score(y_test, pred, zero_division=0))

        # Artifact : le modèle sauvegardé dans ce run
        mlflow.sklearn.log_model(model, name="model",
                                 serialization_format="cloudpickle")

        print(nom, "-> accuracy :", round(accuracy_score(y_test, pred), 3),
              "| f1 :", round(f1_score(y_test, pred, zero_division=0), 3))

# ===== 5. Programme principal =====
mlflow.set_experiment("churn-prediction")         # même expérience qu'avant
TOUT = ["age", "salaire", "abonnement", "anciennete", "consommation"]

lancer_essai("essai-1", 100, 5, TOUT)             # 100 arbres, profondeur 5, tout
lancer_essai("essai-2", 50, 3, TOUT)              # modèle plus simple
lancer_essai("essai-3", 100, 5,                   # sans la colonne salaire
             ["age", "abonnement", "anciennete", "consommation"])
# Réflexion

## TP3 : suivi des expériences avec MLflow

1. **Quel est le meilleur modèle ? Peux-tu le prouver ?**
   Le meilleur modèle est l'essai 3 (f1 = 0.889), à égalité avec l'essai 1.
   Je le prouve dans MLflow en triant les runs par f1 : le résultat est daté
   et conservé. L'essai 2 a une bonne accuracy (0.96) mais un f1 de 0.333 :
   il rate presque tous les clients qui partent. À égalité, je garde l'essai 3,
   plus simple, car il n'utilise pas la colonne salaire.

2. **Peux-tu redonner exactement le modèle de l'essai 1 ?**
   Oui. Chaque run garde son propre modèle, que je recharge avec
   `mlflow.sklearn.load_model("runs:/<id>/model")` sans réentraîner.

3. **Un camarade peut-il reproduire mon travail sur une autre machine ?**
   En partie. Avec Git il récupère le code, et les paramètres et versions
   des bibliothèques sont notés dans les runs. Mais `mlflow.db` et `mlruns/`
   restent sur ma machine, et l'environnement n'est pas identique.
   Docker (TP5-6) réglera ce problème.
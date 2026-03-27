Master Context : Projet A-JEPA (Action-conditional JEPA)
1. Vision GlobaleObjectif : 
    Développer un modèle de monde (World Model) capable de prédire l'intention et le futur d'un geste robotique dans un espace latent.
    Algorithme : A-JEPA (Action-conditional Joint-Embedding Predictive Architecture). Le modèle apprend à prédire la représentation de l'état futur $s_{t+1}$ en fonction de l'état actuel $s_t$ et d'une action $a_t$.
    Philosophie : Apprentissage auto-supervisé non-génératif. On ne cherche pas à reconstruire les valeurs brutes des capteurs, mais à capturer la structure sémantique du mouvement.
2. Spécifications TechniquesDomaine de données : 
    Séries temporelles multivariées issues de la robotique (Proprioception).
    Source de données : Écosystème LeRobot (Hugging Face), privilégiant des datasets de manipulation avec des degrés de liberté (DoF) définis.
    Stack Technologique : Python avec le framework JAX (utilisation probable d'Equinox pour la structure).
    Architecture :
        Encodeur : Transforme des fenêtres temporelles de capteurs en vecteurs latents.
        Prédicteur Conditionnel : Un bloc (potentiellement de type Transformer léger) qui prédit le futur latent conditionné par l'action.
        Cible (Target) : Mise à jour via une moyenne mobile exponentielle (EMA) pour éviter l'effondrement dimensionnel (Collapse).
3. Stratégies de ModélisationMasquage : 
    Utilisation d'un masquage par blocs temporels (Context-Target) pour forcer la compréhension de la dynamique à long terme.
    Espace Latent : 
        - Arbitrage itératif entre Bottleneck (pour l'abstraction/locomotion) et Over-parameterization (pour la précision/manipulation fine).
        - **Régularisation VICReg** : Utilisation de termes de Variance, Invariance et Covariance pour structurer l'espace latent et empêcher l'effondrement (Collapse) sans nécessiter de paires négatives explicites.
    Inférence : Exploration d'une double prédiction (état intermédiaire + état final) pour stabiliser le chemin et limiter le "drift".
4. Évaluation & ValidationMéthode primaire : 
    Linear Probing. Geler l'encodeur et entraîner une couche linéaire pour classifier ou régresser des tâches sémantiques.
    Indicateurs de succès : Capacité du modèle à linéariser les concepts de mouvement et à regrouper (cluster) des intentions similaires.

5. Current Structure:

    1 JEPA/
    2 ├── data/
    3 │   └── stats.pkl
    4 ├── src/
    5 │   ├── compute_stats.py
    6 │   ├── dataloader.py
    7 │   ├── load_data.py
    8 │   ├── load_data_raw.py
    9 │   └── main.py
   10 ├── MASTERCONTEXT.md
   11 ├── pyproject.toml
   12 └── README.md



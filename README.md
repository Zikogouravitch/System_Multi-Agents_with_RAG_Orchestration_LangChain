# PharmaGuardian AI — Assistant Pharmaceutique Professionnel

> Système multi-agents RAG spécialisé en information pharmaceutique au Maroc.  
> Basé sur **LangChain**, **LlamaIndex**, **Ollama (Gemma3:4b)** et des embeddings multilingues.

---

## Structure du projet

```
.
├── dataset/
│   ├── main.py              # À exécuter EN PREMIER pour générer medicaments.jsonl
│   ├── cleaner.py           # Nettoyage des données brutes
│   ├── scraper.py           # Scraping des données médicaments
│   └── medicaments.jsonl    # Base de données générée (créée par main.py)
├── multi-Agent.py           # Point d'entrée principal du système multi-agents
├── env.example              # Variables d'environnement à configurer
├── requirements.txt         # Dépendances Python
└── README.md
```

---

##  Prérequis

- Python **3.11**
- [Ollama](https://ollama.com/) installé et le modèle `gemma3:4b` disponible :
  ```bash
  ollama pull gemma3:4b
  ```
- Connexion internet (premier lancement uniquement, pour télécharger les embeddings)

---

##  Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/Zikogouravitch/System_Multi-Agents_with_RAG_Orchestration_LangChain
cd cd System_Multi-Agents
```

### 2. Créer un environnement virtuel et installer les dépendances

```bash
python -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configurer les variables d'environnement

```bash
cp env.example .env
# Éditez .env selon votre configuration
```

---

##  Étape obligatoire — Générer la base de données

**Avant tout lancement du système**, il faut générer le fichier `medicaments.jsonl` en exécutant le script `main.py` situé dans le dossier `dataset/` :

```bash
cd dataset
python main.py
```

Ce script effectue le scraping et le nettoyage des données, puis génère le fichier :

```
dataset/medicaments.jsonl
```

> Sans ce fichier, le système multi-agents ne pourra pas démarrer.

---

##  Lancement du système multi-agents

Une fois `medicaments.jsonl` généré, revenez à la racine du projet et lancez :

```bash
python multi-Agent.py
```

Au **premier démarrage**, le système va :
1. Télécharger le modèle d'embedding (`paraphrase-multilingual-MiniLM-L12-v2`) → sauvegardé dans `./pharma_embedding_cache/`
2. Construire l'index RAG LlamaIndex → sauvegardé dans `./pharma_index_cache/`

Ces étapes ne se produisent **qu'une seule fois** ; les lancements suivants utilisent le cache local.

---

---

##  Architecture multi-agents

```
Utilisateur
    │
    ▼
[Détecteur d'intention]  ← fuzz matching + vecteurs
    │
    ├── salutation / remerciement / inconnu → Réponse directe
    │
    └── médicament / symptôme
            │
            ▼
      [Agent 1] Recherche vectorielle cosinus sur les indications (JSONL)
            │
            ▼
      [Agent 2] Recherche sémantique RAG — LlamaIndex (top-k=3)
            │
            ▼
      [Agent 3] Vérification sécurité — détection INJECTABLE / VACCIN / PERFUSION
            │
            ▼
      [Agent 4] Rédaction finale — LangChain Chain → réponse professionnelle
```

---

##  Configuration

Les paramètres principaux se trouvent en tête de `multi-Agent.py` :

| Paramètre | Valeur par défaut | Description |
|---|---|---|
| `MODEL_NAME` | `gemma3:4b` | Modèle Ollama utilisé |
| `DATA_FILE` | `dataset/medicaments.jsonl` | Chemin vers la base de données |
| `EMBEDDING_DIR` | `./pharma_embedding_cache` | Cache du modèle d'embedding |
| `INDEX_DIR` | `./pharma_index_cache` | Cache de l'index RAG |
| `SIMILARITY_THRESHOLD` | `0.45` | Seuil de similarité cosinus |

---

##  Sécurité

Le système détecte automatiquement les termes sensibles (`INJECTABLE`, `VACCIN`, `PERFUSION`) et génère une alerte invitant l'utilisateur à consulter un professionnel de santé.

---


Voir `requirements.txt` pour la liste complète avec les versions.

---

##  Licence

Projet à usage interne / éducatif. Les données médicales sont issues de sources publiques marocaines.  
**Ce système ne remplace pas l'avis d'un pharmacien ou d'un médecin.**
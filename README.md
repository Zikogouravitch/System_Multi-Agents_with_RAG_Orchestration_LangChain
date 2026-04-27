# PharmaGuardian AI

Assistant pharmaceutique intelligent basé sur une architecture multi-agents LangChain + pipeline RAG LlamaIndex, tournant entièrement en local via Ollama.

## Prérequis

- Python 3.11
- [Ollama](https://ollama.com) installé et en cours d'exécution
- Modèle Ollama téléchargé :

cmd:ollama pull gemma3:1b

## Installation

cmd:pip install langchain langchain-ollama llama-index llama-index-embeddings-huggingface rapidfuzz sentence-transformers

## Structure du projet

.
├── pharma_guardian.py          # Script principal
├── medicaments.jsonl           # Base de données privée 
├── pharma_embedding_cache/     # Cache embedding HuggingFace (auto-généré)
└── pharma_index_cache/         # Cache index RAG LlamaIndex (auto-généré)




## Format de `medicaments.jsonl`

Chaque ligne est un objet JSON représentant un médicament :

```json
{
  "drug_name": "DOLIPRANE 1000MG",
  "presentation": "Comprimé",
  "manufacturer": "Sanofi",
  "dosage": {"adulte": "1g toutes les 6h", "enfant": "selon poids"},
  "composition": ["PARACETAMOL"],
  "therapeutic_class": "Analgésique antipyrétique",
  "status": "Sans ordonnance",
  "price": {"ppv": "12.50"},
  "indications": ["fièvre", "douleur", "mal de tête"]
}
```
## Lancement

cmd:python pharma_guardian.py


Au premier démarrage, le système :
1. télécharge le modèle d'embedding `all-MiniLM-L6-v2` (une seule fois)
2. construit et persiste l'index RAG à partir du fichier JSONL (une seule fois)

Les démarrages suivants chargent directement les caches.



## Architecture

Le système repose sur **4 agents ReAct LangChain** orchestrés séquentiellement, avec un pipeline RAG LlamaIndex pour l'accès aux données privées.

### Pipeline RAG (LlamaIndex)

| Étape | Détail |
|---|---|
| Ingestion | Chargement JSONL ligne par ligne, 2 documents par médicament (fiche complète + indications) |
| Indexation | `VectorStoreIndex` avec embeddings `all-MiniLM-L6-v2` |
| Stockage | Persistance sur disque dans `pharma_index_cache/` |
| Retrieval | Recherche sémantique `top-k=3`, mode `compact` |
| Augmentation | Contexte RAG injecté dans le prompt de l'Agent 4 |

### Agents

| Agent | Rôle | Outil | Raisonnement |
|---|---|---|---|
| Agent 1 | Détection de symptôme | `outil_detection_symptome` | ReAct |
| Agent 2 | Recherche RAG | `outil_recherche_rag` | ReAct |
| Agent 3 | Vérification sécurité | `outil_securite` | ReAct |
| Agent 4 | Rédaction finale | — (LangChain chain) | Chain-of-thought |

### Flux d'orchestration

```
Requête utilisateur
       │
       ▼
 Détecteur d'intention
       │
  ┌────┴────────────────────────────┐
  │ salutation / remerciement /     │
  │ trop_court / inconnu            │
  │ → réponse directe (sans LLM)    │
  └─────────────────────────────────┘
       │ symptome / medicament
       ▼
  Agent 1 — détection symptôme (ReAct)
       │ symptom
       ▼
  Agent 2 — recherche RAG (ReAct)
       │ context
       ▼
  Agent 3 — vérification sécurité (ReAct)
       │ safety
       ▼
  Agent 4 — rédaction finale (Chain + augmentation RAG)
       │
       ▼
  Réponse (3-4 phrases, français)
```

### Détecteur d'intention

Classe chaque requête en 6 catégories sans appel LLM :

| Intention | Déclencheur | Action |
|---|---|---|
| `trop_court` | < 3 caractères | Réponse directe |
| `salutation` | Mots-clés (bonjour, hi…) | Réponse directe |
| `remerciement` | Mots-clés (merci, ok…) | Réponse directe |
| `inconnu` | Aucune correspondance | Réponse directe |
| `symptome` | Fuzzy match ≥ 80 sur SYMPTOM_MAP | Pipeline agents |
| `medicament` | Fuzzy match ≥ 82 sur les noms | Pipeline agents |

---

## Symptômes pris en charge

| Symptôme | Molécules recommandées |
|---|---|
| fièvre | PARACETAMOL, IBUPROFENE |
| toux sèche | DEXTROMETHORPHANE |
| toux grasse | AMBROXOL |
| allergie | CETIRIZINE |
| douleur | PARACETAMOL, IBUPROFENE |
| reflux | OMEPRAZOLE |
| mal de tête | PARACETAMOL, IBUPROFENE |
| rhume | PARACETAMOL, PSEUDOEPHEDRINE |
| inflammation | IBUPROFENE |
| brûlure estomac | OMEPRAZOLE |
| nausée | DOMPERIDONE |
| infection | AMOXICILLINE |

---

## Termes de sécurité surveillés

L'Agent 3 bloque et alerte si le contexte RAG contient : `INJECTABLE`, `VACCIN`, `PERFUSION`.

---

## Configuration

Toutes les constantes modifiables sont en haut du fichier :

```python
MODEL_NAME      = "gemma3:1b"         # modèle Ollama
DATA_FILE       = "medicaments.jsonl" # base de données
EMBEDDING_DIR   = "./pharma_embedding_cache"
INDEX_DIR       = "./pharma_index_cache"
MODEL_EMBEDDING = "sentence-transformers/all-MiniLM-L6-v2"
```

Pour utiliser un modèle plus puissant : remplacer `gemma3:1b` par `gemma3:4b`, `llama3.2`, etc.

---

## Limitations

- Le système répond uniquement sur la base des médicaments présents dans `medicaments.jsonl`.
- Il ne se substitue pas à un avis médical ou pharmaceutique professionnel.
- Les agents ReAct dépendent de la capacité du modèle à suivre le format Thought/Action/Observation — les petits modèles (1b) peuvent parfois dévier du format.
import os
import json
import re
import unicodedata
import warnings
warnings.filterwarnings("ignore")

from rapidfuzz import fuzz

from sentence_transformers import SentenceTransformer

# ── LangChain ──────────────────────────────────────────────
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool

# ── LlamaIndex ────────────────────────────────────────────
from llama_index.core import (
    VectorStoreIndex, Document, Settings,
    StorageContext, load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

# ==========================================================
# CONFIG
# ==========================================================

MODEL_NAME      = "gemma3:4b"
DATA_FILE       = "medicaments.jsonl"
EMBEDDING_DIR   = "./pharma_embedding_cache"
INDEX_DIR       = "./pharma_index_cache"
MODEL_EMBEDDING = "sentence-transformers/all-MiniLM-L6-v2"


SYMPTOM_MAP = {
    "fievre":          ["PARACETAMOL", "IBUPROFENE"],
    "toux seche":      ["DEXTROMETHORPHANE"],
    "toux grasse":     ["AMBROXOL"],
    "allergie":        ["CETIRIZINE"],
    "douleur":         ["PARACETAMOL", "IBUPROFENE"],
    "reflux":          ["OMEPRAZOLE"],
    "mal de tete":     ["PARACETAMOL", "IBUPROFENE"],
    "rhume":           ["PARACETAMOL", "PSEUDOEPHEDRINE"],
    "inflammation":    ["IBUPROFENE"],
    "brulure estomac": ["OMEPRAZOLE"],
    "nausee":          ["DOMPERIDONE"],
    "infection":       ["AMOXICILLINE"],
}

BANNED_TERMS  = ["INJECTABLE", "VACCIN", "PERFUSION"]
SALUTATIONS   = {"hello","bonjour","bonsoir","salut","hi","hey","bonne nuit","salam","slt","bjr"}
REMERCIEMENTS = {"merci","thanks","thank you","super","parfait","nickel","ok","d accord"}

def normalize_text(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def load_data():
    data = []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def format_med(m: dict) -> str:
    return (
        f"Nom: {m.get('drug_name','Non disponible')}\n"
        f"Presentation: {m.get('presentation','Non disponible')}\n"
        f"Fabricant: {m.get('manufacturer','Non disponible')}\n"
        f"Dosage: {m.get('dosage',{})}\n"
        f"Composition: {', '.join(m.get('composition',[]))}\n"
        f"Classe therapeutique: {m.get('therapeutic_class','Non disponible')}\n"
        f"Statut: {m.get('status','Non disponible')}\n"
        f"Prix: {m.get('price',{}).get('ppv','N/A')} MAD\n"
        f"Indications: {', '.join(m.get('indications',[])) if m.get('indications') else 'Non specifie'}"
    )

# ==========================================================
# INITIALISATION LLMs + EMBEDDINGS
# ==========================================================

Settings.llm = Ollama(model=MODEL_NAME, request_timeout=120)

if os.path.exists(EMBEDDING_DIR) and os.listdir(EMBEDDING_DIR):
    print("Embedding trouvé en cache - chargement local...")
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_DIR)
else:
    print("Téléchargement embedding (une seule fois)...")
    SentenceTransformer(MODEL_EMBEDDING).save(EMBEDDING_DIR)
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_DIR)
    print(f"Embedding sauvegardé dans '{EMBEDDING_DIR}'")

# ==========================================================
# PIPELINE RAG — INGESTION JSONL (LlamaIndex)
# ==========================================================

print("Chargement données privées (JSONL)...")
data = load_data()

if os.path.exists(INDEX_DIR) and os.listdir(INDEX_DIR):
    print("Index RAG trouvé en cache - chargement rapide...")
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    index = load_index_from_storage(storage_context)
else:
    print("Premier démarrage - construction de l'index RAG (une seule fois)...")
    documents = []
    for med in data:
        
        documents.append(Document(
            text=format_med(med),
            metadata={"source": "jsonl", "drug_name": med.get("drug_name", "")}
        ))
        if med.get("indications"):
            
            documents.append(Document(
                text=f"Medicament {med['drug_name']} indique pour : {', '.join(med['indications'])}",
                metadata={"source": "jsonl_indications", "drug_name": med.get("drug_name", "")}
            ))
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    index.storage_context.persist(persist_dir=INDEX_DIR)
    print(f"Index RAG sauvegardé dans '{INDEX_DIR}'")

rag_query_engine = index.as_query_engine(similarity_top_k=3, response_mode="compact")

llm = OllamaLLM(model=MODEL_NAME, timeout=120)

# ==========================================================
# DÉTECTEUR D'INTENTION
# ==========================================================

def detect_intent(query: str) -> str:
    q= normalize_text(query)
    tokens = set(q.split())
    if len(q) < 3:
        return "trop_court"
    if q in SALUTATIONS or tokens & SALUTATIONS:
        return "salutation"
    if q in REMERCIEMENTS or tokens & REMERCIEMENTS:
        return "remerciement"
    for symptom in SYMPTOM_MAP:
        if fuzz.token_set_ratio(q, normalize_text(symptom)) >= 80:
            return "symptome"
    for med in data:
        name  = normalize_text(med["drug_name"])
        score = max(fuzz.partial_ratio(q, name), fuzz.token_set_ratio(q, name))
        if score >= 82:
            return "medicament"
    return "inconnu"

# ==========================================================
# OUTILS LANGCHAIN — un outil dédié par agent
# ==========================================================

@tool
def outil_detection_symptome(texte: str) -> str:
    """
    [Agent 1] Détecte le symptôme principal dans le texte utilisateur
    et recommande un médicament depuis la base privée JSONL via matching fuzzy.
    """
    q = normalize_text(texte)
    best, best_score = None, 0

    for symptom in SYMPTOM_MAP:
        score = fuzz.token_set_ratio(q, normalize_text(symptom))
        if score > best_score:
            best_score = score
            best = symptom

    if best_score >= 80:
        molecules = SYMPTOM_MAP[best]
        for mol in molecules:
            for med in data:
                compo = [normalize_text(c) for c in med.get("composition", [])]
                if normalize_text(mol) in compo:
                    return (
                        f"Symptôme détecté : {best}\n"
                        f"Médicament recommandé : {med['drug_name']}\n"
                        f"Composition : {', '.join(med.get('composition', []))}"
                    )
        return f"Symptôme détecté : {best}. Molécules recommandées : {', '.join(molecules)}."

    return "Aucun symptôme reconnu dans le texte."


@tool
def outil_recherche_rag(requete: str) -> str:
    """
    [Agent 2] Recherche sémantique dans la base privée via LlamaIndex RAG.
    Embeddings all-MiniLM-L6-v2, retrieval top-k=3.
    """
    result = rag_query_engine.query(requete)
    return str(result) if result else "Aucun médicament pertinent trouvé dans la base RAG."


@tool
def outil_securite(contexte: str) -> str:
    """
    [Agent 3] Vérifie si le contexte contient des termes sensibles.
    Termes surveillés : INJECTABLE, VACCIN, PERFUSION.
    """
    txt = contexte.lower()
    for term in BANNED_TERMS:
        if fuzz.partial_ratio(term.lower(), txt) >= 80:
            return (
                f"ALERTE SÉCURITÉ : terme sensible détecté ({term}). "
                "Consultez un professionnel de santé avant toute utilisation."
            )
    return "Aucune alerte de sécurité détectée."

# ==========================================================
# TEMPLATE ReAct PARTAGÉ
# ==========================================================

REACT_TEMPLATE = """
Tu es {role}

Tu as accès aux outils suivants :
{tools}

Noms des outils disponibles : {tool_names}

Format de raisonnement ReAct (obligatoire) :
Question: la question à traiter
Thought: réfléchis à ce que tu dois faire
Action: le nom de l'outil à utiliser
Action Input: l'entrée pour l'outil
Observation: le résultat de l'outil
... (répète Thought/Action/Observation si nécessaire, max 3 fois)
Thought: j'ai la réponse finale
Final Answer: ta réponse finale structurée

Question: {input}
{agent_scratchpad}
"""

REACT_PROMPT = PromptTemplate.from_template(REACT_TEMPLATE)

# ==========================================================
# FACTORY D'AGENTS ReAct
# ==========================================================

def make_agent(role: str, tools: list, max_iter: int = 5) -> AgentExecutor:
    """
    Crée un agent ReAct LangChain avec :
    - un rôle défini (persona / system prompt)
    - des outils dédiés à sa mission
    - une capacité de raisonnement chain-of-thought (ReAct)
    """
    prompt = REACT_PROMPT.partial(role=role)
    agent  = create_react_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=max_iter,
        return_intermediate_steps=False,
    )

# ==========================================================
# AGENT 1 — SYMPTÔMES (ReAct)
# ==========================================================

def create_agent_symptomes() -> AgentExecutor:
    return make_agent(
        role=(
            "un agent spécialiste en détection de symptômes pharmaceutiques. "
            "Ta mission : identifier le symptôme principal dans la question "
            "et trouver le médicament correspondant dans la base privée JSONL. "
            "Utilise outil_detection_symptome pour effectuer cette recherche."
        ),
        tools=[outil_detection_symptome],
    )

# ==========================================================
# AGENT 2 — RECHERCHE RAG (ReAct)
# ==========================================================

def create_agent_rag() -> AgentExecutor:
    return make_agent(
        role=(
            "un agent expert en recherche pharmaceutique via base de données privée. "
            "Ta mission : effectuer une recherche sémantique RAG (LlamaIndex) "
            "pour retrouver les informations précises sur le médicament demandé. "
            "Utilise outil_recherche_rag avec la question enrichie du symptôme connu."
        ),
        tools=[outil_recherche_rag],
    )

# ==========================================================
# AGENT 3 — SÉCURITÉ (ReAct)
# ==========================================================

def create_agent_securite() -> AgentExecutor:
    return make_agent(
        role=(
            "un agent pharmacovigilance chargé de la sécurité des prescriptions. "
            "Ta mission : analyser le contexte RAG et détecter "
            "tout terme sensible (INJECTABLE, VACCIN, PERFUSION). "
            "Utilise outil_securite en lui passant le contexte RAG disponible."
        ),
        tools=[outil_securite],
    )

# ==========================================================
# AGENT 4 — RÉDACTEUR FINAL (LangChain Chain)
# ==========================================================

WRITER_PROMPT = PromptTemplate.from_template("""
Tu es PharmaGuardian AI, assistant pharmaceutique professionnel au Maroc.

QUESTION UTILISATEUR : {question}

=== CONTEXTE AUGMENTÉ (données privées RAG) ===
SYMPTÔME IDENTIFIÉ   : {symptom}
DONNÉES MÉDICAMENT   : {context}
RAPPORT SÉCURITÉ     : {safety}
===============================================

INSTRUCTIONS :
- Réponds UNIQUEMENT sur la base des données fournies ci-dessus (base privée RAG).
- Si DONNÉES MÉDICAMENT contient un médicament : cite son nom exact, sa posologie et ses indications.
- Si SYMPTÔME est renseigné et un médicament est trouvé : lie-les explicitement.
- Si danger détecté dans RAPPORT SÉCURITÉ : recommande impérativement de consulter un médecin.
- Réponse courte (3-4 phrases max), professionnelle, en français.
- Ne jamais inventer un médicament absent des données.
- Si aucune information pertinente : dis-le clairement et oriente vers un pharmacien.

RÉPONSE :
""")

def agent_writer(question: str, symptom: str, context: str, safety: str) -> str:
    chain = WRITER_PROMPT | llm
    return chain.invoke({
        "question": question,
        "symptom":  symptom  or "Aucun symptôme détecté",
        "context":  context  or "Aucun médicament trouvé dans la base.",
        "safety":   safety   or "Aucune vérification effectuée.",
    })

# ==========================================================
# ORCHESTRATEUR
# ==========================================================

def orchestrator(query: str) -> str:
    intent = detect_intent(query)

    if intent == "trop_court":
        return "Pouvez-vous préciser votre question ?"

    if intent == "salutation":
        return (
            "Bonjour ! Je suis PharmaGuardian AI.\n"
            "Décrivez vos symptômes ou le nom d'un médicament "
            "et je vous fournirai une information pharmaceutique fiable."
        )

    if intent == "remerciement":
        return "Avec plaisir ! N'hésitez pas si vous avez d'autres questions."

    if intent == "inconnu":
        return (
            "Je suis spécialisé en information pharmaceutique.\n"
            "Posez-moi une question sur un médicament ou décrivez vos symptômes."
        )

    print("\nAgent 1 : détection symptôme...")
    symptom = ""
    try:
        result1 = create_agent_symptomes().invoke({"input": query})
        symptom = result1.get("output", "")
    except Exception as e:
        print(f"Agent 1 erreur : {e}")

    print("\nAgent 2 : recherche RAG...")
    context = ""
    try:
        rag_input = f"{query}. Symptôme connu : {symptom}" if symptom else query
        result2   = create_agent_rag().invoke({"input": rag_input})
        context   = result2.get("output", "")
    except Exception as e:
        print(f"   Agent 2 erreur : {e}")

    if not symptom and not context:
        return (
            "Je n'ai pas trouvé d'information correspondante dans ma base de données. "
            "Reformulez votre question ou consultez un pharmacien."
        )

    print("\nAgent 3 : vérification sécurité...")
    safety = "Vérification sécurité indisponible."
    try:
        result3 = create_agent_securite().invoke({"input": context or query})
        safety  = result3.get("output", safety)
    except Exception as e:
        print(f"Agent 3 erreur : {e}")

    print("\n Agent 4 : rédaction finale...")
    response = agent_writer(query, symptom, context, safety)
    print("Réponse rédigée.")

    return response

# ==========================================================
# main
# ==========================================================

if __name__ == "__main__":
    print("=" * 60)
    print("   PharmaGuardian AI - Assistant Pharmaceutique Intelligent")
    print("   Tapez 'exit' pour quitter")
    print("=" * 60)

    while True:
        q = input("\nQuestion : ").strip()
        if not q:
            continue
        if normalize_text(q) in ["exit", "quit"]:
            print("Au revoir !")
            break

        rep = orchestrator(q)
        print("\n" + "-" * 60)
        print("Réponse :")
        print(rep)
        print("-" * 60)
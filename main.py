import os
import re
import pickle
import logging
import networkx as nx
import numpy as np
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except Exception:
    HAS_GEMINI = False

from typing import List, Dict, Optional
from pydantic import BaseModel, Field, ConfigDict
from langgraph.graph import StateGraph, END
from dataclasses import dataclass
from PyPDF2 import PdfReader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------
# Data Models
# ----------------------------
@dataclass
class Entity:
    name: str
    type: str
    description: str
    embedding: Optional[List[float]] = None  # instead of np.ndarray

@dataclass
class Relationship:
    source: str
    target: str
    relationship: str
    description: str
    confidence: float = 1.0

class GraphState(BaseModel):
    pdf_paths: List[str] = []
    query: Optional[str] = None
    text_chunks: List[str] = []
    entities: Dict[str, Entity] = {}
    relationships: List[Relationship] = []
    graph: Optional[nx.Graph] = None
    answer: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

# ----------------------------
# Graph RAG with LangGraph
# ----------------------------
class GraphRAGLangGraph:
    def __init__(self):
        self.workflow = StateGraph(GraphState)

        # Add nodes (placeholders, customize with your logic)
        self.workflow.add_node("extract_text", self._extract_text)
        self.workflow.add_node("chunk_text", self._chunk_text)
        self.workflow.add_node("extract_entities", self._extract_entities)
        self.workflow.add_node("extract_relationships", self._extract_relationships)
        self.workflow.add_node("build_graph", self._build_graph)
        self.workflow.add_node("answer_query", self._answer_query)

        # Define edges
        self.workflow.set_entry_point("extract_text")
        self.workflow.add_edge("extract_text", "chunk_text")
        self.workflow.add_edge("chunk_text", "extract_entities")
        self.workflow.add_edge("extract_entities", "extract_relationships")
        self.workflow.add_edge("extract_relationships", "build_graph")
        self.workflow.add_edge("build_graph", "answer_query")
        self.workflow.add_edge("answer_query", END)

        self.app = self.workflow.compile()

        # Configure Google Gemini (if available) and instantiate a hardcoded model
        # Require Gemini-only operation. Fail fast with explicit errors if the client or API key
        # is not available so users won't silently fall back to local extraction.
        if not HAS_GEMINI:
            raise RuntimeError("google.generativeai is not installed. Install it and ensure it is importable to use Gemini-only mode.")

        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable is not set. Set it to use Gemini-only mode.")

        try:
            genai.configure(api_key=gemini_api_key)
        except Exception as e:
            raise RuntimeError(f"Failed to configure Gemini client: {e}")

        # Hardcode the model name as requested
        model_name = "gemini-1.5-flash"
        try:
            self.model = genai.GenerativeModel(model_name)
            logger.info(f"Instantiated Gemini model '{model_name}'")
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate Gemini model '{model_name}': {e}")

    # ------------------------
    # Helper: call Gemini safely
    # ------------------------
    def _call_gemini(self, prompt: str) -> Optional[str]:
        if self.model is None:
            raise RuntimeError("Gemini model is not configured")
        try:
            resp = self.model.generate_content(prompt)
            # response may expose .text or nested content depending on client; be defensive
            text = getattr(resp, "text", None)
            if text is None:
                return str(resp)
            return text
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            raise

    def _extract_json_from_text(self, text: str) -> Optional[dict]:
        """
        Try to locate and parse the first valid JSON object or array in a block of text.
        This is defensive against model outputs that include extra commentary before/after JSON.
        Returns a parsed JSON object (dict or list) or None.
        """
        import json

        # Fast attempt: try to parse the whole text
        try:
            return json.loads(text)
        except Exception:
            pass

        # Scan for balanced braces for objects
        stack = 0
        start = None
        for i, ch in enumerate(text):
            if ch == '{':
                if stack == 0:
                    start = i
                stack += 1
            elif ch == '}' and stack > 0:
                stack -= 1
                if stack == 0 and start is not None:
                    candidate = text[start:i+1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        # continue scanning
                        start = None

        # Scan for balanced brackets for arrays
        stack = 0
        start = None
        for i, ch in enumerate(text):
            if ch == '[':
                if stack == 0:
                    start = i
                stack += 1
            elif ch == ']' and stack > 0:
                stack -= 1
                if stack == 0 and start is not None:
                    candidate = text[start:i+1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        start = None

        return None

    # ------------------------
    # Node Implementations
    # ------------------------
    def _extract_text(self, state: GraphState) -> GraphState:
        pdf_text = ""
        for pdf_path in state.pdf_paths:
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                pdf_text += page.extract_text()
        state.text_chunks = [pdf_text]
        return state

    def _chunk_text(self, state: GraphState) -> GraphState:
        # Better chunking: split by sentence boundaries but keep reasonable length windows
        raw = state.text_chunks[0]
        # Normalize whitespace
        raw = re.sub(r"\s+", " ", raw).strip()
        # Split into sentences using simple punctuation heuristic
        sentences = re.split(r'(?<=[.!?])\s+', raw)
        # Group sentences into chunks of ~500-1000 chars to preserve context
        chunks = []
        cur = []
        cur_len = 0
        for s in sentences:
            if not s.strip():
                continue
            cur.append(s.strip())
            cur_len += len(s)
            if cur_len > 800:
                chunks.append(" ".join(cur))
                cur = []
                cur_len = 0
        if cur:
            chunks.append(" ".join(cur))
        state.text_chunks = chunks if chunks else [raw[:2000]]
        return state

    def _extract_entities(self, state: GraphState) -> GraphState:
        text = " ".join(state.text_chunks)

        # Try Gemini structured extraction
        prompt = f"""
        Extract entities from the text below and return valid JSON with the key "entities", an array of objects with fields: name, type, description.

        Text:
        {text}
        """
        gem_text = self._call_gemini(prompt)
        extracted = []
        if not gem_text:
            raise RuntimeError("Gemini returned no text for entity extraction")

        parsed = self._extract_json_from_text(gem_text)
        if isinstance(parsed, dict):
            entities = parsed.get("entities")
            if entities and isinstance(entities, list):
                for ent in entities:
                    if not isinstance(ent, dict):
                        continue
                    name = ent.get("name")
                    if name:
                        extracted.append(Entity(name=name.strip(), type=ent.get("type", "Unknown"), description=ent.get("description", "")))
        if not extracted:
            raise RuntimeError("Gemini did not return any structured entities. Ensure the model prompt and permissions are correct.")

        # Post-process: normalize entity names, dedupe, and keep frequencies
        def normalize_name(n: str) -> str:
            return re.sub(r"\s+", " ", n).strip()

        freq = {}
        for chunk in state.text_chunks:
            low = chunk.lower()
            for e in extracted:
                n = normalize_name(e.name)
                if n == "":
                    continue
                count = low.count(n.lower())
                if count > 0:
                    freq[n] = freq.get(n, 0) + count

        # Merge extracted entities by normalized name, prefer longest name as canonical
        ents_by_norm = {}
        for e in extracted:
            n = normalize_name(e.name)
            if n == "":
                continue
            existing = ents_by_norm.get(n)
            if not existing or len(e.name) > len(existing.name):
                ents_by_norm[n] = e

        # Filter out very short tokens and noise
        cleaned = {}
        for n, e in ents_by_norm.items():
            if len(n) < 2 or re.match(r"^[\W_]+$", n):
                continue
            e.name = n
            # attach frequency
            e.description = (e.description or "") + f" (mentions: {freq.get(n,0)})"
            cleaned[n] = e

        state.entities = cleaned
        return state

    def _normalize_name(self, n: str) -> str:
        """Normalize an entity/display name for matching purposes."""
        return re.sub(r"\s+", " ", (n or "")).strip()

    def _parse_mentions_from_desc(self, desc: str) -> int:
        m = re.search(r"mentions:\s*(\d+)", desc or "")
        return int(m.group(1)) if m else 0

    def _merge_states(self, existing: GraphState, new: GraphState) -> GraphState:
        """Merge two GraphState objects into a new GraphState.

        Simple normalized-name matching + substring merging is used. When names match or are
        substrings, we merge metadata and sum mention counts. Relationships are remapped to
        the canonical names and deduplicated (keep highest confidence).
        """
        merged = GraphState()

        # Start from existing entities (canonical)
        canonical: Dict[str, Entity] = {self._normalize_name(n): e for n, e in existing.entities.items()} if existing and existing.entities else {}

        # Map normalized -> canonical key (string used as key in canonical dict)
        norm_to_canon: Dict[str, str] = {k: k for k in canonical.keys()}

        # Merge new entities
        for name, ent in (new.entities or {}).items():
            n_norm = self._normalize_name(name)
            if not n_norm:
                continue

            # exact normalized match
            if n_norm in canonical:
                target_key = n_norm
            else:
                # substring match: prefer longer canonical or new name
                target_key = None
                for k in list(canonical.keys()):
                    if k in n_norm or n_norm in k:
                        # choose the longer string as canonical
                        target_key = k if len(k) >= len(n_norm) else n_norm
                        break

            if target_key is None:
                # new canonical entity
                canonical[n_norm] = Entity(name=ent.name, type=ent.type, description=ent.description, embedding=ent.embedding)
                norm_to_canon[n_norm] = n_norm
            else:
                # merge into existing canonical
                if target_key not in canonical:
                    # create if target_key is the new normalized name
                    canonical[target_key] = Entity(name=ent.name, type=ent.type, description=ent.description, embedding=ent.embedding)
                else:
                    exist_ent = canonical[target_key]
                    # prefer longer display name
                    if len(ent.name or "") > len(exist_ent.name or ""):
                        exist_ent.name = ent.name
                    # prefer non-empty type
                    if (not exist_ent.type or exist_ent.type == "Unknown") and ent.type:
                        exist_ent.type = ent.type
                    # sum mentions if present in descriptions
                    exist_mentions = self._parse_mentions_from_desc(exist_ent.description)
                    new_mentions = self._parse_mentions_from_desc(ent.description)
                    total_mentions = exist_mentions + new_mentions
                    # combine descriptions (keep existing plus any new unique text)
                    combined_desc = exist_ent.description or ""
                    if ent.description and ent.description not in combined_desc:
                        combined_desc = (combined_desc + "; " + ent.description).strip()
                    combined_desc = combined_desc + f" (mentions: {total_mentions})"
                    exist_ent.description = combined_desc
                    # TODO: embedding merge policy (keep existing unless missing)
                    if exist_ent.embedding is None and ent.embedding is not None:
                        exist_ent.embedding = ent.embedding
                norm_to_canon[n_norm] = target_key

        # Build merged.entities keyed by canonical display name
        merged.entities = {}
        for canon_norm, ent in canonical.items():
            # use entity.name (already set to a display name) as key
            key = ent.name or canon_norm
            merged.entities[key] = ent

        # Merge relationships: take existing plus new, remap names to canonical keys
        rel_map: Dict[tuple, Relationship] = {}

        def find_canonical(name: str) -> Optional[str]:
            if not name:
                return None
            n = self._normalize_name(name)
            # direct mapping
            if n in norm_to_canon:
                canon = norm_to_canon[n]
                ent_obj = canonical.get(canon)
                return ent_obj.name if ent_obj else canon
            # fallback: try direct canonical keys
            for k in canonical.keys():
                if k in n or n in k:
                    ent_obj = canonical.get(k)
                    return ent_obj.name if ent_obj else k
            return name

        # Add existing relationships
        for r in (existing.relationships or []):
            s = find_canonical(r.source)
            t = find_canonical(r.target)
            if not s or not t:
                continue
            key = tuple(sorted((s, t)))
            existing_r = rel_map.get(key)
            if not existing_r or r.confidence > existing_r.confidence:
                rel_map[key] = Relationship(source=s, target=t, relationship=r.relationship, description=r.description, confidence=r.confidence)

        # Add new relationships (remapped)
        for r in (new.relationships or []):
            s = find_canonical(r.source)
            t = find_canonical(r.target)
            if not s or not t:
                continue
            key = tuple(sorted((s, t)))
            existing_r = rel_map.get(key)
            if not existing_r or r.confidence > existing_r.confidence:
                rel_map[key] = Relationship(source=s, target=t, relationship=r.relationship, description=r.description, confidence=r.confidence)

        merged.relationships = list(rel_map.values())

        # Rebuild graph from merged entities/relationships
        G = nx.Graph()
        for ent in merged.entities.values():
            G.add_node(ent.name, type=ent.type, description=ent.description)
        for rel in merged.relationships:
            if rel.source in G and rel.target in G:
                G.add_edge(rel.source, rel.target, relationship=rel.relationship, description=rel.description, confidence=rel.confidence)
        merged.graph = G

        # Keep new text chunks appended for context
        merged.text_chunks = (existing.text_chunks or []) + (new.text_chunks or [])
        return merged

    def _extract_relationships(self, state: GraphState) -> GraphState:
        text = " ".join(state.text_chunks)
        relationships: List[Relationship] = []

        # Attempt Gemini-based relationship inference
        entities_list = list(state.entities.keys())
        if not entities_list:
            raise RuntimeError("No entities available for relationship inference; ensure entity extraction succeeded.")

        prompt = f"""
        Given the following text and the list of entities, infer relationships between entities. Return valid JSON with key "relationships" which is an array of objects with fields: source, target, relationship, description, confidence (0-1).

        Text:
        {text}

        Entities:
        {entities_list}
        """
        gem_text = self._call_gemini(prompt)
        if not gem_text:
            raise RuntimeError("Gemini returned no text for relationship inference")
        parsed = self._extract_json_from_text(gem_text)
        if isinstance(parsed, dict):
            rels = parsed.get("relationships")
            if rels and isinstance(rels, list):
                for r in rels:
                    if not isinstance(r, dict):
                        continue
                    src = r.get("source")
                    tgt = r.get("target")
                    if src and tgt:
                        try:
                            conf = float(r.get("confidence", 1.0))
                        except Exception:
                            conf = 1.0
                        relationships.append(Relationship(source=src.strip(), target=tgt.strip(), relationship=r.get("relationship", "related_to"), description=r.get("description", ""), confidence=conf))
        if not relationships:
            raise RuntimeError("Gemini did not return any relationships. Ensure the model prompt and permissions are correct.")

        # Deduplicate relationships (keep highest confidence)
        unique = {}
        for r in relationships:
            key = tuple(sorted((r.source, r.target)))
            if key in unique:
                if r.confidence > unique[key].confidence:
                    unique[key] = r
            else:
                unique[key] = r

        state.relationships = list(unique.values())
        return state

    def _build_graph(self, state: GraphState) -> GraphState:
        G = nx.Graph()
        for ent in state.entities.values():
            G.add_node(ent.name, type=ent.type, description=ent.description)
        for rel in state.relationships:
            if rel.source in G and rel.target in G:
                G.add_edge(rel.source, rel.target, relationship=rel.relationship, description=rel.description, confidence=rel.confidence)
        state.graph = G
        return state

    def retrieve_context(self, state: GraphState) -> str:
        # Build a short context string from graph, entities, relationships, and text samples
        parts = []
        if state.entities:
            parts.append("Entities:\n" + ", ".join(list(state.entities.keys())[:50]))
        if state.relationships:
            parts.append("Relationships:\n" + ", ".join([f"{r.source}->{r.target}({r.relationship})" for r in state.relationships[:50]]))
        # Add first few text chunks as context
        if state.text_chunks:
            sample = "\n\n".join(state.text_chunks[:3])
            parts.append("Text sample:\n" + sample[:2000])
        return "\n\n".join(parts) if parts else "No context available."

    def generate_answer(self, state: GraphState, query: str) -> str:
        context = self.retrieve_context(state)
        prompt = f"""
        Based on the following context from a knowledge graph and relevant documents, answer the user's question.

        Context:
        {context}

        Question: {query}

        Please provide a comprehensive answer based on the context provided. If the context doesn't contain enough information to answer the question completely, indicate what information is missing.
        """
        gem_text = self._call_gemini(prompt)
        if gem_text:
            return gem_text
        # Fallback: simple answer using entities
        if state.graph and state.graph.nodes:
            nodes = list(state.graph.nodes)[:10]
            return f"The document discusses these entities: {', '.join(nodes)}. I couldn't produce a detailed answer with Gemini." 
        return "No context available to answer the question."

    def _answer_query(self, state: GraphState) -> GraphState:
        query = state.query or ""
        # Gemini-only answer generation
        state.answer = self.generate_answer(state, query)
        return state

    # ------------------------
    # Run
    # ------------------------
    def run(self, pdfs: List[str], query: str, save_path: Optional[str] = None) -> str:
        pdfs = [os.path.join("data", os.path.basename(pdf)) for pdf in pdfs]  # Ensure correct relative path
        state = GraphState(pdf_paths=pdfs, query=query)
        final_state = self.app.invoke(state)

        # Ensure final_state is a GraphState (coerce dicts returned by the workflow)
        final_state = self._coerce_to_state(final_state)

        answer = getattr(final_state, "answer", None)
        if answer is None and isinstance(final_state, dict):
            answer = final_state.get("answer", "")

        # If a save_path is provided and the file exists, attempt an incremental merge
        if save_path:
            if os.path.exists(save_path):
                try:
                    existing = self.load_graph(save_path)
                    merged = self._merge_states(existing, final_state)
                    # save merged graph
                    self.save_graph(merged, save_path)
                except Exception as e:
                    logger.warning(f"Failed to merge existing graph at {save_path}: {e}. Overwriting.")
                    self.save_graph(final_state, save_path)
            else:
                self.save_graph(final_state, save_path)

        return answer

    # ------------------------
    # Save / Load
    # ------------------------
    def save_graph(self, state_obj, filepath: str = "knowledge_graph.pkl"):
        try:
            entities = {}
            relationships = []
            graph = None
            text_chunks = []

            if hasattr(state_obj, "entities"):
                entities = state_obj.entities
            elif isinstance(state_obj, dict):
                entities = state_obj.get("entities", {})

            if hasattr(state_obj, "relationships"):
                relationships = state_obj.relationships
            elif isinstance(state_obj, dict):
                relationships = state_obj.get("relationships", [])

            if hasattr(state_obj, "graph"):
                graph = state_obj.graph
            elif isinstance(state_obj, dict):
                graph = state_obj.get("graph", None)

            if hasattr(state_obj, "text_chunks"):
                text_chunks = state_obj.text_chunks
            elif isinstance(state_obj, dict):
                text_chunks = state_obj.get("text_chunks", [])

            if graph is None:
                logger.error("No graph in state; cannot save")
                return

            graph_data_dict = nx.node_link_data(graph)
            graph_data = {
                "graph": graph_data_dict,
                "entities": {
                    name: {
                        "name": ent.name if hasattr(ent, "name") else ent["name"],
                        "type": ent.type if hasattr(ent, "type") else ent.get("type"),
                        "description": ent.description if hasattr(ent, "description") else ent.get("description"),
                        "embedding": (ent.embedding.tolist() if hasattr(ent, "embedding") and ent.embedding is not None
                                      else (ent.get("embedding") if isinstance(ent, dict) else None))
                    }
                    for name, ent in entities.items()
                },
                "relationships": [
                    {
                        "source": r.source if hasattr(r, "source") else r.get("source"),
                        "target": r.target if hasattr(r, "target") else r.get("target"),
                        "relationship": r.relationship if hasattr(r, "relationship") else r.get("relationship"),
                        "description": r.description if hasattr(r, "description") else r.get("description"),
                        "confidence": r.confidence if hasattr(r, "confidence") else r.get("confidence", 0.0)
                    }
                    for r in relationships
                ],
                "text_chunks": text_chunks
            }

            with open(filepath, "wb") as f:
                pickle.dump(graph_data, f)
            logger.info(f"Graph saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save graph: {e}")

    def load_graph(self, filepath: str) -> GraphState:
        with open(filepath, "rb") as f:
            graph_data = pickle.load(f)
        # If a raw dict was saved, convert it into a GraphState
        return self._graph_data_to_state(graph_data)

    def _graph_data_to_state(self, graph_data: dict) -> GraphState:
        """Convert on-disk graph_data dict (as produced by save_graph) into GraphState."""
        state = GraphState()
        try:
            state.graph = nx.node_link_graph(graph_data["graph"])
        except Exception as e:
            # If graph key is missing or malformed, leave graph None and continue
            logger.warning(f"Failed to reconstruct NetworkX graph from saved data: {e}")
            state.graph = None

        state.entities = {}
        for name, data in (graph_data.get("entities") or {}).items():
            embedding = None
            if data.get("embedding") is not None:
                try:
                    embedding = np.array(data.get("embedding"))
                except Exception:
                    embedding = data.get("embedding")
            state.entities[name] = Entity(
                name=data.get("name", name),
                type=data.get("type", ""),
                description=data.get("description", ""),
                embedding=embedding
            )

        state.relationships = []
        for r in graph_data.get("relationships", []):
            state.relationships.append(Relationship(
                source=r.get("source"),
                target=r.get("target"),
                relationship=r.get("relationship"),
                description=r.get("description", ""),
                confidence=r.get("confidence", 0.0)
            ))

        state.text_chunks = graph_data.get("text_chunks", [])
        return state

    def _coerce_to_state(self, obj) -> GraphState:
        """Coerce a workflow return value (GraphState or dict) into GraphState."""
        if isinstance(obj, GraphState):
            return obj
        if isinstance(obj, dict):
            # The workflow may return a dict with keys similar to GraphState; try to map them
            state = GraphState()
            # map simple fields
            state.pdf_paths = obj.get("pdf_paths", [])
            state.query = obj.get("query")
            state.text_chunks = obj.get("text_chunks", [])

            # entities may already be dict-of-Entity-like objects or plain dicts
            ents = {}
            for k, v in (obj.get("entities") or {}).items():
                if isinstance(v, Entity):
                    ents[k] = v
                elif isinstance(v, dict):
                    embedding = None
                    if v.get("embedding") is not None:
                        try:
                            embedding = np.array(v.get("embedding"))
                        except Exception:
                            embedding = v.get("embedding")
                    ents[k] = Entity(name=v.get("name", k), type=v.get("type", ""), description=v.get("description", ""), embedding=embedding)
            state.entities = ents

            rels = []
            for r in (obj.get("relationships") or []):
                if isinstance(r, Relationship):
                    rels.append(r)
                elif isinstance(r, dict):
                    rels.append(Relationship(source=r.get("source"), target=r.get("target"), relationship=r.get("relationship"), description=r.get("description", ""), confidence=r.get("confidence", 0.0)))
            state.relationships = rels

            # graph may be present as NetworkX graph or as dict
            g = obj.get("graph")
            if isinstance(g, nx.Graph):
                state.graph = g
            elif isinstance(g, dict):
                try:
                    state.graph = nx.node_link_graph(g)
                except Exception:
                    state.graph = None

            state.answer = obj.get("answer")
            return state

        raise TypeError("Cannot coerce object to GraphState")


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    rag = GraphRAGLangGraph()
    # Read queries from queries.txt (one per non-empty line)
    queries_path = os.path.join(os.getcwd(), "queries.txt")
    if not os.path.exists(queries_path):
        print(f"queries.txt not found at {queries_path}; create the file with one query per line.")
    else:
        with open(queries_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines()]
        queries = [q for q in lines if q]
        if not queries:
            print("queries.txt is empty. Add one query per line.")
        # use a single cumulative KG file
        cumulative_path = os.path.join(os.getcwd(), "kg.pkl")
        for i, q in enumerate(queries, start=1):
            try:
                answer = rag.run(["/data/sample.pdf"], q, save_path=cumulative_path)
                print(f"Query {i}: {q}\nAnswer: {answer}\nMerged graph saved to {cumulative_path}\n")
                # Load and print a short summary of the cumulative graph
                restored = rag.load_graph(cumulative_path)
                print(f"Cumulative entities ({len(restored.entities)}):", list(restored.entities.keys())[:20])
                print(f"Cumulative relationships ({len(restored.relationships)}):", [(r.source, r.target) for r in restored.relationships[:20]])
            except Exception as e:
                print(f"Failed to process query '{q}': {e}")

from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import normalize


class SimilarityService:

    def __init__(self):
        print("🚀 EMBEDDING SIMILARITY SERVICE INITIALIZED")

        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.df = None
        self.index = None
        self.embeddings = None

        self._build_index()

    # ─────────────────────────────────────────────────────────────
    # BUILD INDEX
    # ─────────────────────────────────────────────────────────────
    def _build_index(self):
        csv_path = Path("data/tickets.csv")

        if not csv_path.exists():
            print("❌ CSV not found")
            return

        df = pd.read_csv(csv_path)

        # Keep only resolved tickets
        df = df[df["status"].isin(["resolved", "closed"])].copy()
        df = df.dropna(subset=["title", "description", "resolution"])
        df = df.reset_index(drop=True)

        if df.empty:
            print("⚠️ No resolved tickets found")
            return

        self.df = df

        # Title weighted 3x
        texts = (
            df["title"] + " " +
            df["title"] + " " +
            df["title"] + " " +
            df["description"]
        ).tolist()

        print("🔄 Generating embeddings...")

        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        ).astype("float32")

        # Normalize (IMPORTANT for cosine similarity)
        self.embeddings = normalize(embeddings)

        # FAISS inner product index
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)

        print(f"✅ Index built with {len(df)} tickets")

    # ─────────────────────────────────────────────────────────────
    # FIND SIMILAR TICKETS
    # ─────────────────────────────────────────────────────────────
    def find_similar(
        self,
        title: str,
        description: str,
        predicted_category: str = None,
        top_k: int = 5
    ):
        if self.index is None:
            print("❌ Index not ready")
            return []

        query = f"{title} {title} {title} {description}"

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True
        ).astype("float32")

        query_embedding = normalize(query_embedding)

        scores, indices = self.index.search(query_embedding, top_k * 5)

        results = []
        seen = set()

        query_text = (title + " " + description).lower()

        for idx, score in zip(indices[0], scores[0]):

            if idx < 0:
                continue

            row = self.df.iloc[idx]

            resolution = str(row.get("resolution", ""))
            if resolution in seen:
                continue

            seen.add(resolution)

            row_text = (
                str(row.get("title", "")) + " " +
                str(row.get("description", ""))
            ).lower()

            score = float(score)

            # ───────────── CATEGORY BOOST ─────────────
            if predicted_category and row.get("category") == predicted_category:
                score *= 1.2

            # ───────────── DOMAIN BOOSTS 🔥 ─────────────
            if "database" in query_text and "database" in row_text:
                score += 0.15

            if "connection pool" in query_text and "connection pool" in row_text:
                score += 0.15

            if "timeout" in query_text and "timeout" in row_text:
                score += 0.05

            # ───────────── PENALIZE WRONG DOMAIN ─────────────
            if "database" in query_text and "api" in row_text:
                score -= 0.10

            # ───────────── CLAMP ─────────────
            score = max(0.0, min(score, 1.0))

            # ───────────── FILTER WEAK MATCHES ─────────────
            if score < 0.35:
                continue

            results.append({
                "ticket_id": str(row.get("ticket_id")),
                "title": str(row.get("title")),
                "similarity_score": round(score, 4),
                "category": str(row.get("category")),
                "priority": str(row.get("priority")),
                "resolution": resolution,
                "resolution_time_minutes": int(row.get("resolution_time_minutes", 0)),
            })

            if len(results) >= top_k:
                break

        return results

    # ─────────────────────────────────────────────────────────────
    # ADD NEW TICKET (LEARNING)
    # ─────────────────────────────────────────────────────────────
    def add_ticket(self, title: str, description: str, row_dict: dict):
        """
        Add new resolved ticket to index without rebuilding
        """

        text = f"{title} {title} {title} {description}"

        embedding = self.model.encode(
            [text],
            convert_to_numpy=True
        ).astype("float32")

        embedding = normalize(embedding)

        # Add to FAISS
        self.index.add(embedding)

        # Update memory
        self.embeddings = np.vstack([self.embeddings, embedding])

        # Update dataframe
        self.df = pd.concat([self.df, pd.DataFrame([row_dict])], ignore_index=True)

        print("✅ New ticket added to similarity index")
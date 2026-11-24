from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class SimilarityResult:
    representative_index: int
    similar_indices: List[int]
    representative_text: str
    similar_texts: List[str]
    average_similarity: float
    cluster_size: int


class SimilarityService:
    """
    Semantic similarity service for finding repeated/opinion clusters.

    Uses sentence-transformers/all-MiniLM-L6-v2 for efficient embeddings.
    """

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    DEFAULT_MIN_SIMILARITY = 0.75
    DEFAULT_MIN_CLUSTER_SIZE = 2

    def __init__(
        self,
        device: str = None,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
        embedding_model: Optional[SentenceTransformer] = None
    ):
        self.device = device or ("cuda" if __import__('torch').cuda.is_available() else "cpu")
        self.min_similarity = min_similarity
        self._model = embedding_model

    def _load_model(self):
        """Lazy load the embedding model."""
        if self._model is None:
            self._model = SentenceTransformer(self.MODEL_NAME, device=self.device)

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get sentence embeddings for a list of texts."""
        self._load_model()
        return self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def find_most_repeated_opinion(
        self,
        texts: List[str],
        min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE
    ) -> SimilarityResult:
        """
        Find the most repeated/similar opinion cluster.

        Uses semantic similarity (not exact matching) to group similar opinions.
        Returns the largest cluster of semantically similar comments.

        Args:
            texts: List of comment texts
            min_cluster_size: Minimum comments to form a cluster

        Returns:
            SimilarityResult with cluster details
        """
        if len(texts) < min_cluster_size:
            return SimilarityResult(
                representative_index=0,
                similar_indices=[0],
                representative_text=texts[0] if texts else "",
                similar_texts=texts[:1],
                average_similarity=1.0,
                cluster_size=len(texts)
            )

        embeddings = self.get_embeddings(texts)

        similarities = np.dot(embeddings, embeddings.T)
        norms = np.linalg.norm(embeddings, axis=1)
        similarity_matrix = similarities / np.outer(norms, norms)

        np.fill_diagonal(similarity_matrix, 0)

        best_cluster = self._find_best_cluster(similarity_matrix, min_cluster_size)

        if best_cluster is None:
            return SimilarityResult(
                representative_index=0,
                similar_indices=[0],
                representative_text=texts[0],
                similar_texts=[texts[0]],
                average_similarity=1.0,
                cluster_size=1
            )

        rep_idx, similar_indices, avg_sim = best_cluster

        return SimilarityResult(
            representative_index=rep_idx,
            similar_indices=similar_indices,
            representative_text=texts[rep_idx],
            similar_texts=[texts[i] for i in similar_indices],
            average_similarity=avg_sim,
            cluster_size=len(similar_indices)
        )

    def _find_best_cluster(
        self,
        similarity_matrix: np.ndarray,
        min_cluster_size: int
    ) -> Tuple[int, List[int], float] | None:
        """
        Find the best cluster using a greedy approach.

        For each comment, find all comments above similarity threshold.
        Return the cluster with the most members (and highest avg similarity as tiebreaker).
        """
        n = similarity_matrix.shape[0]
        best_cluster = None
        best_size = 0
        best_avg_sim = 0.0

        for i in range(n):
            similar_mask = similarity_matrix[i] >= self.min_similarity
            similar_indices = np.where(similar_mask)[0].tolist()
            similar_indices.append(i)
            similar_indices = sorted(set(similar_indices))

            if len(similar_indices) < min_cluster_size:
                continue

            cluster_similarities = similarity_matrix[i][similar_indices]
            avg_sim = float(cluster_similarities.mean())

            if len(similar_indices) > best_size or (
                len(similar_indices) == best_size and avg_sim > best_avg_sim
            ):
                best_size = len(similar_indices)
                best_avg_sim = avg_sim
                best_cluster = (i, similar_indices, avg_sim)

        return best_cluster

    def find_opinion_clusters(
        self,
        texts: List[str],
        min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE
    ) -> List[SimilarityResult]:
        """
        Find all significant opinion clusters (not just the largest).

        Useful for showing multiple prevalent opinions.
        """
        if len(texts) < min_cluster_size:
            return []

        embeddings = self.get_embeddings(texts)

        similarities = np.dot(embeddings, embeddings.T)
        norms = np.linalg.norm(embeddings, axis=1)
        similarity_matrix = similarities / np.outer(norms, norms)

        np.fill_diagonal(similarity_matrix, 0)

        clusters = []
        used = set()

        for i in range(len(texts)):
            if i in used:
                continue

            similar_mask = similarity_matrix[i] >= self.min_similarity
            similar_indices = np.where(similar_mask)[0].tolist()
            similar_indices.append(i)
            similar_indices = sorted(set(similar_indices))

            if len(similar_indices) < min_cluster_size:
                continue

            cluster_similarities = similarity_matrix[i][similar_indices]
            avg_sim = float(cluster_similarities.mean())

            clusters.append(SimilarityResult(
                representative_index=i,
                similar_indices=similar_indices,
                representative_text=texts[i],
                similar_texts=[texts[j] for j in similar_indices],
                average_similarity=avg_sim,
                cluster_size=len(similar_indices)
            ))

            used.update(similar_indices)

        clusters.sort(key=lambda c: (c.cluster_size, c.average_similarity), reverse=True)

        return clusters


def create_similarity_service(
    device: str = None,
    min_similarity: float = 0.75,
    embedding_model: Optional[SentenceTransformer] = None
) -> SimilarityService:
    """Factory function to create similarity service."""
    return SimilarityService(device, min_similarity, embedding_model)
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer


@dataclass
class SentimentResult:
    label: str
    score: float
    stars: int
    all_scores: Dict[str, float]


class SentimentService:
    """
    Sentiment analysis using pretrained transformer model.

    Uses cardiffnlp/twitter-roberta-base-sentiment-latest which provides:
    - Negative, Neutral, Positive labels
    - Good performance on social media text
    """

    MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    LABELS = ["Negative", "Neutral", "Positive"]

    STAR_MAPPING = {
        "Negative": (1, 2),
        "Neutral": 3,
        "Positive": (4, 5)
    }

    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = None
        self._model = None
        self._embedding_model = None

    def _load_model(self):
        """Lazy load the sentiment model."""
        if self._model is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.MODEL_NAME
            ).to(self.device)
            self._model.eval()

    def _load_embedding_model(self):
        """Lazy load sentence transformer for semantic similarity."""
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(
                'sentence-transformers/all-MiniLM-L6-v2',
                device=self.device
            )

    def analyze(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of a single text.

        Returns:
            SentimentResult with label, confidence score, star rating, and all scores
        """
        self._load_model()

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            probs = probs.cpu().numpy()[0]

        label_idx = int(np.argmax(probs))
        label = self.LABELS[label_idx]
        score = float(probs[label_idx])

        all_scores = {self.LABELS[i]: float(probs[i]) for i in range(3)}

        stars = self._label_to_stars(label, score)

        return SentimentResult(
            label=label,
            score=score,
            stars=stars,
            all_scores=all_scores
        )

    def analyze_batch(self, texts: List[str], batch_size: int = 32) -> List[SentimentResult]:
        """
        Analyze sentiment for multiple texts efficiently using batching.
        """
        self._load_model()

        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            inputs = self._tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self._model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                probs = probs.cpu().numpy()

            for prob in probs:
                label_idx = int(np.argmax(prob))
                label = self.LABELS[label_idx]
                score = float(prob[label_idx])
                all_scores = {self.LABELS[j]: float(prob[j]) for j in range(3)}
                stars = self._label_to_stars(label, score)

                results.append(SentimentResult(
                    label=label,
                    score=score,
                    stars=stars,
                    all_scores=all_scores
                ))

        return results

    def _label_to_stars(self, label: str, confidence: float) -> int:
        """
        Convert sentiment label + confidence to 1-5 star rating.

        Mapping:
        - Negative: 1-2 stars (more confident = 1 star)
        - Neutral: 3 stars
        - Positive: 4-5 stars (more confident = 5 stars)
        """
        if label == "Negative":
            return 1 if confidence > 0.7 else 2
        elif label == "Neutral":
            return 3
        else:
            return 5 if confidence > 0.7 else 4

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get sentence embeddings for semantic similarity."""
        self._load_embedding_model()
        embeddings = self._embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings

    @property
    def embedding_model(self) -> Optional[SentenceTransformer]:
        """Get the shared embedding model instance."""
        self._load_embedding_model()
        return self._embedding_model

    def find_most_similar_opinion(self, texts: List[str], min_similarity: float = 0.75) -> Tuple[int, List[int], float]:
        """
        Find the most repeated/similar opinion using semantic similarity.

        Returns:
            (representative_index, similar_indices, avg_similarity)
        """
        if len(texts) < 2:
            return (0, [0], 1.0)

        embeddings = self.get_embeddings(texts)

        similarities = np.dot(embeddings, embeddings.T)
        norms = np.linalg.norm(embeddings, axis=1)
        similarity_matrix = similarities / np.outer(norms, norms)

        np.fill_diagonal(similarity_matrix, 0)

        cluster_scores = similarity_matrix.mean(axis=1)
        best_idx = int(np.argmax(cluster_scores))

        similar_indices = np.where(similarity_matrix[best_idx] >= min_similarity)[0].tolist()
        similar_indices.append(best_idx)
        similar_indices = sorted(set(similar_indices))

        avg_sim = float(similarity_matrix[best_idx][similar_indices].mean()) if similar_indices else 0.0

        return (best_idx, similar_indices, avg_sim)


def create_sentiment_service(device: str = None) -> SentimentService:
    """Factory function to create sentiment service."""
    return SentimentService(device)
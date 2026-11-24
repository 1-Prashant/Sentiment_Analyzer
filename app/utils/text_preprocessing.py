import re
import html
from typing import List


def clean_text(text: str) -> str:
    """
    Clean comment text while preserving sentiment-relevant information.

    Operations:
    - Decode HTML entities
    - Remove URLs
    - Remove excessive whitespace
    - Preserve emojis, punctuation, and case (important for sentiment)
    """
    if not text:
        return ""

    text = html.unescape(text)

    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    text = re.sub(r'\s+', ' ', text)

    text = text.strip()

    return text


def remove_spam_patterns(text: str) -> str:
    """
    Remove common spam patterns while preserving genuine sentiment.
    """
    if not text:
        return ""

    patterns = [
        r'(?i)\b(subscribe|sub|like|comment|share|follow)\b',
        r'(?i)\b(check out my|check my|my channel|my video)\b',
        r'(?i)\b(first|second|third|early gang)\b',
        r'(?i)\b(bot|spam|scam)\b',
    ]

    for pattern in patterns:
        text = re.sub(pattern, '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text


def preprocess_comment(text: str, remove_spam: bool = False) -> str:
    """
    Full preprocessing pipeline for a single comment.

    Args:
        text: Raw comment text
        remove_spam: Whether to apply spam pattern removal

    Returns:
        Cleaned text ready for sentiment analysis
    """
    text = clean_text(text)

    if remove_spam:
        text = remove_spam_patterns(text)

    return text


def preprocess_comments(texts: List[str], remove_spam: bool = False) -> List[str]:
    """Preprocess a list of comments."""
    return [preprocess_comment(t, remove_spam) for t in texts]


def is_meaningful_comment(text: str, min_length: int = 3) -> bool:
    """
    Check if a comment has meaningful content for sentiment analysis.

    Filters out:
    - Empty or whitespace-only
    - Very short comments (likely just emojis or single words)
    - Comments that are only punctuation/emojis
    """
    if not text or not text.strip():
        return False

    cleaned = clean_text(text)

    if len(cleaned) < min_length:
        return False

    alnum_count = sum(1 for c in cleaned if c.isalnum())
    if alnum_count < 2:
        return False

    return True
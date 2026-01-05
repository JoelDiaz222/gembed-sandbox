"""
TPCx-AI Review Data Loader

Provides functions to load and sample review texts from the TPCx-AI benchmark
dataset for use in embedding benchmarks. Falls back to synthetic data if 
the TPCx-AI data file is not available.
"""

import csv
import random
from pathlib import Path
from typing import List, Tuple

DATA_DIR = Path(__file__).parent
REVIEW_FILE = DATA_DIR / "Review.psv"

# Cache for loaded reviews
_cached_reviews: List[Tuple[int, int, str]] = []
_cached_legitimate: List[Tuple[int, int, str]] = []
_using_synthetic: bool = False


def _generate_synthetic_reviews(n: int) -> List[Tuple[int, int, str]]:
    """Generate synthetic review data as fallback."""
    categories = ["Electronics", "Home & Garden", "Sports", "Books", "Clothing",
                  "Toys", "Health", "Automotive", "Food", "Office"]
    adjectives = ["great", "excellent", "good", "okay", "poor", "amazing", "terrible"]
    templates = [
        "This {adj} product exceeded my expectations. {detail}",
        "I bought this for my {use} and it works {adj}. {detail}",
        "After using this for a month, I can say it's {adj}. {detail}",
        "{adj} quality for the price. {detail}",
        "The {category} item arrived quickly. Overall {adj} experience. {detail}",
    ]
    details = [
        "Would recommend to others.",
        "Shipping was fast.",
        "Build quality is solid.",
        "Easy to set up and use.",
        "Customer service was helpful.",
        "Works as described.",
        "Better than similar products I've tried.",
    ]
    
    reviews = []
    for i in range(n):
        template = random.choice(templates)
        text = template.format(
            adj=random.choice(adjectives),
            category=random.choice(categories).lower(),
            use=random.choice(["home", "office", "travel", "daily use"]),
            detail=random.choice(details)
        )
        reviews.append((i, 0, text))  # All synthetic reviews are "legitimate" (spam=0)
    return reviews


def load_reviews(legitimate_only: bool = True) -> List[Tuple[int, int, str]]:
    """
    Load and cache all reviews from the TPCx-AI dataset.
    Falls back to synthetic data if Review.psv is not available.
    
    Args:
        legitimate_only: If True, return only non-spam reviews (default)
    
    Returns:
        List of tuples (ID, spam, text)
    """
    global _cached_reviews, _cached_legitimate, _using_synthetic
    
    # Load all reviews if not cached
    if not _cached_reviews:
        if REVIEW_FILE.exists():
            with open(REVIEW_FILE, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='|')
                next(reader)  # Skip header: "ID"|"spam"|"text"
                _cached_reviews = [(int(r[0]), int(r[1]), r[2]) for r in reader]
            # Pre-filter legitimate reviews
            _cached_legitimate = [r for r in _cached_reviews if r[1] == 0]
            _using_synthetic = False
        else:
            # Generate synthetic fallback data
            print("Note: Review.psv not found, using synthetic data. "
                  "See data/README.md for TPCx-AI data setup.")
            _cached_reviews = _generate_synthetic_reviews(10000)
            _cached_legitimate = _cached_reviews  # All synthetic are legitimate
            _using_synthetic = True
    
    return _cached_legitimate if legitimate_only else _cached_reviews


def is_using_synthetic() -> bool:
    """Check if synthetic data is being used instead of TPCx-AI data."""
    load_reviews()  # Ensure data is loaded
    return _using_synthetic


def get_review_texts(n: int, shuffle: bool = True, legitimate_only: bool = True) -> List[str]:
    """
    Get n review texts for benchmarking.
    
    Args:
        n: Number of reviews to return
        shuffle: If True, randomly sample; if False, take first n
        legitimate_only: If True, return only non-spam reviews (default)
        
    Returns:
        List of review text strings
    """
    reviews = load_reviews(legitimate_only=legitimate_only)
    if shuffle:
        sampled = random.sample(reviews, min(n, len(reviews)))
    else:
        sampled = reviews[:n]
    return [r[2] for r in sampled]


def get_reviews_with_labels(n: int, shuffle: bool = True, legitimate_only: bool = True) -> List[Tuple[str, int]]:
    """
    Get n reviews with their spam labels.
    
    Args:
        n: Number of reviews to return
        shuffle: If True, randomly sample; if False, take first n
        legitimate_only: If True, return only non-spam reviews (default)
        
    Returns:
        List of tuples (text, spam_label)
    """
    reviews = load_reviews(legitimate_only=legitimate_only)
    if shuffle:
        sampled = random.sample(reviews, min(n, len(reviews)))
    else:
        sampled = reviews[:n]
    return [(r[2], r[1]) for r in sampled]


def make_inputs(n: int) -> List[str]:
    """
    Drop-in replacement for synthetic make_inputs() functions.
    
    Args:
        n: Number of input texts to return
        
    Returns:
        List of review text strings
    """
    return get_review_texts(n, shuffle=True)

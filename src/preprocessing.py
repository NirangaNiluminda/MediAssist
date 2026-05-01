"""
Text pre-processing pipeline for clinical notes.
Owner: Person A (Stage 2)
"""
import re
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import List

# Run once: nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')

_nlp = None
_stop_words = None
_lemmatizer = WordNetLemmatizer()


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def _get_stopwords():
    global _stop_words
    if _stop_words is None:
        _stop_words = set(stopwords.words("english"))
    return _stop_words


def clean_text(text: str) -> str:
    """Remove boilerplate characters while preserving medical terms."""
    # TODO (Person A): tune regex — preserve dosage numbers, drug names, brackets
    text = re.sub(r"\s+", " ", text)          # collapse whitespace
    text = re.sub(r"[^\w\s\.\,\-\/]", "", text)  # strip special chars except medical punctuation
    return text.strip()


def tokenize(text: str) -> List[str]:
    """Tokenize using spaCy."""
    # TODO (Person A): decide whether to keep sentence boundaries
    nlp = _get_nlp()
    doc = nlp(text)
    return [token.text for token in doc]


def lemmatize(tokens: List[str]) -> List[str]:
    """Lemmatize a token list with NLTK WordNetLemmatizer."""
    # TODO (Person A): compare NLTK vs spaCy lemma — spaCy may be better for medical text
    return [_lemmatizer.lemmatize(t.lower()) for t in tokens]


def remove_stopwords(tokens: List[str]) -> List[str]:
    """Remove English stopwords. Keeps medical negations (no, not, without)."""
    keep = {"no", "not", "without", "nor"}   # critical in clinical context
    sw = _get_stopwords() - keep
    return [t for t in tokens if t.lower() not in sw]


def preprocess(text: str) -> dict:
    """Full pipeline — returns dict with all intermediate results."""
    # TODO (Person A): wire into ner_module.py and summarizer.py
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    lemmas = lemmatize(tokens)
    filtered = remove_stopwords(lemmas)
    return {
        "cleaned": cleaned,
        "tokens": tokens,
        "lemmas": lemmas,
        "filtered": filtered,
    }

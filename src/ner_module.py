"""
Medical Named Entity Recognition using scispaCy.
Owner: Person A (Stage 2)
"""
import spacy
from typing import Dict, List

# Requires: pip install scispacy + en_core_sci_md wheel
# See README for install command.

_ner_model = None


def load_ner_model(model_name: str = "en_core_sci_md"):
    """Load scispaCy model (cached after first call)."""
    global _ner_model
    if _ner_model is None:
        # TODO (Person A): optionally add EntityLinker for UMLS CUIs
        # nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
        _ner_model = spacy.load(model_name)
    return _ner_model


def extract_entities(text: str, nlp=None) -> Dict[str, List[str]]:
    """
    Extract medical entities from raw text.

    Returns:
        {
            "diseases":  [...],
            "drugs":     [...],
            "symptoms":  [...],
            "other":     [...]
        }
    """
    if nlp is None:
        nlp = load_ner_model()

    doc = nlp(text)

    # TODO (Person A): refine label mapping — scispaCy en_core_sci_md uses broad ENTITY labels.
    # BC5CDR has CHEMICAL and DISEASE; map accordingly after eval on that dataset.
    entities: Dict[str, List[str]] = {
        "diseases": [],
        "drugs": [],
        "symptoms": [],
        "other": [],
    }

    label_map = {
        "DISEASE": "diseases",
        "CHEMICAL": "drugs",
        "SYMPTOM": "symptoms",   # may not exist in base model — add via pattern matcher
    }

    for ent in doc.ents:
        bucket = label_map.get(ent.label_, "other")
        entities[bucket].append(ent.text)

    # Deduplicate while preserving order
    for key in entities:
        seen = set()
        deduped = []
        for e in entities[key]:
            if e.lower() not in seen:
                seen.add(e.lower())
                deduped.append(e)
        entities[key] = deduped

    return entities


def entities_to_json(text: str, nlp=None) -> dict:
    """Convenience wrapper returning entities + source text."""
    return {"text": text, "entities": extract_entities(text, nlp)}

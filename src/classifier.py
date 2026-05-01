"""
BioBERT few-shot triage classification (urgent / routine / follow-up).
Owner: Person B (Stage 2)
"""
from typing import List, Optional
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

DEFAULT_MODEL = "dmis-lab/biobert-base-cased-v1.2"
LABELS = ["urgent", "routine", "follow-up"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

_model = None
_tokenizer = None


def load_classifier(checkpoint_path: Optional[str] = None, model_name: str = DEFAULT_MODEL):
    """Load BioBERT classifier. Prefer fine-tuned checkpoint if available."""
    global _model, _tokenizer
    from pathlib import Path
    source = checkpoint_path if checkpoint_path and Path(checkpoint_path).exists() else model_name
    _tokenizer = AutoTokenizer.from_pretrained(source)
    _model = AutoModelForSequenceClassification.from_pretrained(
        source,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    _model.eval()
    return _model, _tokenizer


def _tokenize_fn(batch, tokenizer, max_len=512):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_len)


def train_classifier(
    train_records: List[dict],   # [{"text": ..., "label": "urgent"}, ...]
    val_records: List[dict],
    n_examples: Optional[int] = None,
    model_name: str = DEFAULT_MODEL,
    output_dir: str = "models/biobert_triage",
    epochs: int = 3,
    batch_size: int = 16,
):
    """
    Fine-tune BioBERT for triage classification.
    Set n_examples to run few-shot sweep (10 / 50 / 100 / 200 / None=all).
    TODO (Person B): call from 03_BioBERT_Classification.ipynb with n sweep loop.
    """
    if n_examples:
        train_records = train_records[:n_examples]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    def encode(records):
        ds = Dataset.from_list([{"text": r["text"], "labels": LABEL2ID[r["label"]]} for r in records])
        return ds.map(lambda b: _tokenize_fn(b, tokenizer), batched=True)

    train_ds = encode(train_records)
    val_ds = encode(val_records)

    args = TrainingArguments(
        output_dir=f"{output_dir}_n{n_examples or 'all'}",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available(),
        logging_steps=20,
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds, tokenizer=tokenizer)
    trainer.train()
    return trainer


def predict(text: str, model=None, tokenizer=None) -> dict:
    """Return predicted label and per-class probabilities."""
    if model is None or tokenizer is None:
        model, tokenizer = load_classifier()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
    pred_id = int(np.argmax(probs))

    return {
        "label": ID2LABEL[pred_id],
        "confidence": float(probs[pred_id]),
        "probabilities": {ID2LABEL[i]: float(p) for i, p in enumerate(probs)},
    }

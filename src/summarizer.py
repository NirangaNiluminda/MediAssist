"""
Abstractive summarization using fine-tuned BART / T5.
Owner: Person A (Stage 3)
"""
from pathlib import Path
from typing import Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
import torch

DEFAULT_MODEL = "facebook/bart-base"
MAX_INPUT_LEN = 1024
MAX_TARGET_LEN = 128


def load_summarizer(checkpoint_path: Optional[str] = None, model_name: str = DEFAULT_MODEL):
    """Load tokenizer + model from a fine-tuned checkpoint or the base HF model."""
    source = checkpoint_path if checkpoint_path and Path(checkpoint_path).exists() else model_name
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForSeq2SeqLM.from_pretrained(source)
    model.eval()
    return model, tokenizer


def prepare_dataset(records: list, tokenizer, max_input: int = MAX_INPUT_LEN, max_target: int = MAX_TARGET_LEN) -> Dataset:
    """
    Tokenize a list of {"input": ..., "summary": ...} dicts into a HF Dataset.
    TODO (Person A): call this in 02_Summarization_BART.ipynb after loading MeQSum.
    """
    def tokenize_fn(batch):
        model_inputs = tokenizer(
            batch["input"], max_length=max_input, truncation=True, padding="max_length"
        )
        labels = tokenizer(
            text_target=batch["summary"], max_length=max_target, truncation=True, padding="max_length"
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    ds = Dataset.from_list(records)
    return ds.map(tokenize_fn, batched=True, remove_columns=["input", "summary"])


def train_summarizer(
    model_name: str,
    train_ds: Dataset,
    val_ds: Dataset,
    output_dir: str = "models/bart_meqsum",
    epochs: int = 3,
    batch_size: int = 8,
):
    """Fine-tune BART/T5. Run inside 02_Summarization_BART.ipynb on GPU/Colab."""
    # TODO (Person A): add fp16=True when running on GPU; set fp16=False for CPU testing
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=2,
        logging_steps=50,
        predict_with_generate=True,
    )

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
    )
    trainer.train()
    return trainer


def summarize(text: str, model, tokenizer, max_input: int = MAX_INPUT_LEN, max_new_tokens: int = 128) -> str:
    """Generate a summary for a single input text."""
    inputs = tokenizer(text, return_tensors="pt", max_length=max_input, truncation=True)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
    return tokenizer.decode(ids[0], skip_special_tokens=True)

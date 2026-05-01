# Model Download & Training Instructions

Model weights are **not committed** to the repo (gitignored). Download or train them locally.

## Fine-tuned BART Summarizer
Train in `notebooks/02_Summarization_BART.ipynb` (Person A).
Checkpoint saved to: `models/bart_meqsum/`

Or load the base model directly (no fine-tuning):
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
```

## BioBERT Triage Classifier
Train in `notebooks/03_BioBERT_Classification.ipynb` (Person B).
Checkpoint saved to: `models/biobert_triage_n<N>/`

Or load the base model:
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("dmis-lab/biobert-base-cased-v1.2", num_labels=3)
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
```

## FastText Medical Embeddings (optional)
Train in `notebooks/01_NLP_Pipeline.ipynb` (Person A) on the PubMed corpus.
Saved to: `models/fasttext_med.bin`

Or use pre-trained BioWordVec:
```bash
# Download from: https://github.com/ncbi-nlp/BioWordVec
```

## RAG FAISS Index
Build in `notebooks/04_Prompt_Engineering.ipynb` (Person B) once PubMed corpus is available.
```python
from src.rag_utils import build_index
build_index("data/processed/pubmed_corpus.txt")
# Saves: models/rag_index.faiss + models/rag_chunks.pkl
```

# Dataset Download Instructions

Do NOT commit raw data files to the repository.

## Datasets Used

### 1. MeQSum (Summarization QA)
```python
from datasets import load_dataset
ds = load_dataset("abachaa/meqsum")
```

### 2. PubMed Summarization
```python
from datasets import load_dataset
ds = load_dataset("ccdv/pubmed-summarization", "section")
```

### 3. BC5CDR (Medical NER)
```python
from datasets import load_dataset
ds = load_dataset("bigbio/bc5cdr", "bc5cdr_bigbio_ner")
```

### 4. MedNLI (Classification)
Requires PhysioNet MIMIC credentialing: https://physionet.org/content/mednli/
Optional — BC5CDR is sufficient if access unavailable.

## Folder Structure
```
data/
├── raw/          ← downloaded dataset files (gitignored)
├── processed/    ← cleaned/split datasets
└── README.md     ← this file
```

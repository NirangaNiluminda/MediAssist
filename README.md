# MediAssist: Intelligent Medical Document Summarization & Symptom Analysis

Advanced AI – Group Project 2026

## Project Overview

An AI-powered pipeline that ingests clinical notes and medical documents, extracts key medical entities, generates concise patient summaries, and answers domain-specific questions using prompt engineering.

## AI Techniques Implemented

1. **NLP Pipeline** — spaCy/scispaCy NER for medical entity extraction
2. **LLM Fine-tuning** — BART/T5 for abstractive summarization
3. **Transfer Learning** — BioBERT few-shot triage classification
4. **Prompt Engineering** — Few-shot, Chain-of-Thought, RAG-style QA

## Setup

```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download spaCy & scispaCy models
python -m spacy download en_core_web_sm
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz

# 4. Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```

## Running the Notebooks

| Notebook | Description |
|---|---|
| `notebooks/01_NLP_Pipeline.ipynb` | Text pre-processing + NER evaluation on BC5CDR |
| `notebooks/02_Summarization_BART.ipynb` | BART fine-tuning + ROUGE evaluation |
| `notebooks/03_BioBERT_Classification.ipynb` | Few-shot triage classification |
| `notebooks/04_Prompt_Engineering.ipynb` | Prompt strategy comparison |

## Dataset Download

See [data/README.md](data/README.md) for dataset download instructions.

## Running the Demo

```bash
streamlit run app.py
```

## Environment

- Python 3.10+
- GPU recommended for fine-tuning (Google Colab / local CUDA)

## Team

| Member | Role |
|---|---|
| Member 1 | NLP Pipeline & NER |
| Member 2 | BART Fine-tuning |
| Member 3 | BioBERT Classification |
| Member 4 | Prompt Engineering & Demo |
| Member 5 | Report & Literature Review |

# Advanced AI – Final Project 2026: Comprehensive Project Plan

> **Assignment:** Advanced Artificial Intelligence – Group Project
> **Duration:** 4–6 Weeks
> **Total Marks:** 100 Points

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Selected Domain & Problem Statement](#2-selected-domain--problem-statement)
3. [AI Techniques Selected (Minimum 3)](#3-ai-techniques-selected-minimum-3)
4. [Dataset Plan](#4-dataset-plan)
5. [System Architecture](#5-system-architecture)
6. [Week-by-Week Timeline](#6-week-by-week-timeline)
7. [Report Structure & Writing Plan](#7-report-structure--writing-plan)
8. [Evaluation & Metrics Plan](#8-evaluation--metrics-plan)
9. [Video Presentation Plan](#9-video-presentation-plan)
10. [Submission Checklist](#10-submission-checklist)
11. [Team Responsibilities](#11-team-responsibilities)
12. [Grading Coverage Map](#12-grading-coverage-map)

---

## 1. Project Overview

### Project Title
**MediAssist: An Intelligent Medical Document Summarization & Symptom Analysis System**

### Objective
Build a functional AI-powered system that:
- Ingests clinical notes and medical documents
- Extracts key symptoms, diagnoses, and medications using NLP
- Generates concise patient summaries using a fine-tuned LLM (T5/BART)
- Answers domain-specific questions using prompt engineering (few-shot, chain-of-thought)
- Demonstrates at least **four** Advanced AI techniques from the course module

### Why This Project?
- Directly addresses a high-impact real-world problem (healthcare overload, documentation burden)
- Multiple public datasets are available (MIMIC-III, MeQSum, PubMed)
- Covers **NLP**, **Transformer-based LLMs**, **Transfer Learning**, and **Prompt Engineering** — all required course techniques
- Produces a **demonstrable working product** (web-based or CLI demo)

---

## 2. Selected Domain & Problem Statement

### Domain: Healthcare

### Problem Statement
Healthcare professionals spend up to 35% of their time on documentation. Medical records are lengthy, unstructured, and time-consuming to review. There is a critical need for:
1. Automatic summarization of patient notes into concise discharge summaries
2. Extraction of key medical entities (symptoms, medications, diagnoses)
3. A QA interface to query documents in natural language

### Motivation
- Patient safety depends on accurate information transfer between clinicians
- NLP and LLMs have demonstrated strong performance on clinical text tasks
- Public datasets (MIMIC-III notes, MeQSum consumer health QA) make this feasible academically

---

## 3. AI Techniques Selected (Minimum 3)

The project implements **four** techniques to maximize marks on the *Use of AI Techniques (20 pts)* criterion.

### Technique 1 — Natural Language Processing (NLP)
**What:** Text pre-processing pipeline + Named Entity Recognition (NER) for medical entities
**How:**
- Tokenization, stopword removal, lemmatization using `spaCy` and `NLTK`
- Word embeddings via **FastText** trained on clinical corpus (or pre-trained BioWordVec)
- NER using `scispaCy` (en_core_sci_md model) to extract: diseases, drugs, symptoms, dosages
- Post-processing: entity normalization to UMLS/SNOMED codes (optional, bonus)

**Output:** Structured JSON of extracted medical entities per document

---

### Technique 2 — Transformer-based LLMs (Fine-tuning / Transfer Learning)
**What:** Fine-tune a pre-trained seq2seq model for abstractive medical summarization
**How:**
- Base model: **BART-base** or **T5-small** from Hugging Face
- Dataset: MeQSum (consumer health QA) + subset of MIMIC-III discharge summaries
- Fine-tuning: Hugging Face `Trainer` API with mixed precision (fp16) on GPU
- Evaluation: ROUGE-1, ROUGE-2, ROUGE-L vs baseline (extractive summarization with BertSum)

**Output:** Fine-tuned summarization model producing 3–5 sentence summaries

---

### Technique 3 — Prompt Engineering
**What:** Use a large general-purpose LLM (GPT-3.5-turbo via API or Llama-3 locally) with systematic prompt strategies
**How:**
- **Few-shot prompting:** Provide 3–5 example QA pairs in the prompt for medical question answering
- **Chain-of-thought (CoT) prompting:** Instruct the model to reason step-by-step before answering
- **In-context learning:** Dynamically retrieve relevant document chunks (RAG-style) to inject into prompt
- Prompt variants evaluated systematically: zero-shot vs few-shot vs CoT vs RAG+few-shot

**Output:** Comparative analysis of prompt strategies on a held-out QA test set

---

### Technique 4 — Transfer Learning (Few-shot / Zero-shot Classification)
**What:** Use **BioBERT** (pre-trained on PubMed) for zero-shot/few-shot medical text classification
**How:**
- Task: Classify clinical notes into triage categories (urgent / routine / follow-up)
- Fine-tune BioBERT with only 50–200 labelled examples (few-shot regime)
- Compare against: full fine-tune baseline, zero-shot with MNLI (NLI-based classification)
- Framework: Hugging Face `transformers` + `datasets`

**Output:** Classification accuracy curves showing few-shot vs full-data learning

---

## 4. Dataset Plan

| Dataset | Source | Task | Size |
|---|---|---|---|
| MeQSum | Hugging Face (`abachaa/meqsum`) | Summarization QA | ~1,000 pairs |
| PubMed Abstracts | Hugging Face (`ccdv/pubmed-summarization`) | Abstractive summarization | ~133k |
| BC5CDR | Hugging Face (`bigbio/bc5cdr`) | Medical NER (chemicals, diseases) | ~1,500 docs |
| MedNLI | PhysioNet (requires MIMIC credentialing) | NLI / Classification | ~14k pairs |
| Custom clinical notes (synthetic) | Generated via GPT or public templates | Demo / test | ~200 samples |

> **Note on Proprietary Data:** All datasets are publicly available. If MIMIC access is not obtained in time, the PubMed and MeQSum datasets alone are sufficient for all three core tasks.

### Data Collection Steps
1. Download MeQSum and PubMed datasets via `datasets.load_dataset()`
2. Download BC5CDR for NER evaluation
3. Apply for PhysioNet MIMIC-III access (if time permits — not blocking)
4. Document all data sources, licenses, and preprocessing steps in the report

---

## 5. System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    MediAssist Pipeline                   │
│                                                          │
│  [Input: Raw Medical Text / Clinical Notes]              │
│              │                                           │
│              ▼                                           │
│  ┌─────────────────────┐                                 │
│  │  NLP Pre-processing │  ← spaCy, NLTK, FastText       │
│  │  - Tokenize         │                                 │
│  │  - NER (scispaCy)   │                                 │
│  │  - Entity Extract   │                                 │
│  └──────────┬──────────┘                                 │
│             │                                            │
│    ┌────────┴─────────────────┐                          │
│    │                          │                          │
│    ▼                          ▼                          │
│  ┌──────────────┐    ┌────────────────────┐              │
│  │ Fine-tuned   │    │  Prompt Engineering│              │
│  │ BART/T5      │    │  (GPT / Llama-3)   │              │
│  │ Summarizer   │    │  Few-shot + CoT    │              │
│  └──────┬───────┘    └────────┬───────────┘              │
│         │                     │                          │
│         └─────────┬───────────┘                          │
│                   ▼                                      │
│         ┌─────────────────┐                              │
│         │ BioBERT Triage  │  ← Transfer Learning         │
│         │ Classifier      │                              │
│         └────────┬────────┘                              │
│                  │                                       │
│                  ▼                                       │
│         [Output: Summary + Entities + Triage Label]      │
│         [Demo: Streamlit Web UI / CLI]                   │
└──────────────────────────────────────────────────────────┘
```

### Tech Stack
| Component | Tool/Framework |
|---|---|
| Language | Python 3.10+ |
| NLP | spaCy, scispaCy, NLTK |
| Embeddings | FastText, Hugging Face |
| LLM Fine-tuning | Hugging Face Transformers, PEFT/LoRA |
| Prompt Engineering | OpenAI API / Ollama (Llama-3 local) |
| Evaluation | ROUGE, F1, sklearn metrics |
| Demo | Streamlit |
| Notebooks | Jupyter |
| Version Control | Git + GitHub |

---

## 6. Week-by-Week Timeline

> Assumes a **5-week** project window. Adjust if 6 weeks are available.

### Week 1 — Setup, Literature & Data
**Goal:** Research foundation, environment setup, dataset acquisition

| Task | Owner | Output |
|---|---|---|
| Read 8–10 relevant papers on clinical NLP & medical summarization | All | Literature notes |
| Set up shared GitHub repo with folder structure | Team lead | Repo ready |
| Install all dependencies (`requirements.txt`) | All | Working environment |
| Download and explore all datasets (EDA) | Data sub-team | EDA notebook |
| Draft Problem Statement & Executive Summary (report sections 1–2) | Writer | Draft sections |
| Define evaluation metrics for each module | All | Metrics doc |

**Deliverable:** Literature Review draft + EDA notebook

---

### Week 2 — NLP Pipeline & Embeddings (Technique 1)
**Goal:** Build and evaluate the pre-processing and NER pipeline

| Task | Owner | Output |
|---|---|---|
| Build text pre-processing pipeline (tokenize, clean, lemmatize) | NLP sub-team | `preprocessing.py` |
| Train/load FastText embeddings on medical corpus | NLP sub-team | Embeddings model |
| Implement scispaCy NER for diseases, drugs, symptoms | NLP sub-team | `ner_module.py` |
| Evaluate NER on BC5CDR (Precision, Recall, F1) | NLP sub-team | NER results |
| Write report sections: Methodology (NLP), Implementation Details | Writer | Draft sections 4–5 |

**Deliverable:** `01_NLP_Pipeline.ipynb` — reproducible NER notebook with results

---

### Week 3 — LLM Fine-tuning & Transfer Learning (Techniques 2 & 4)
**Goal:** Fine-tune BART/T5 and BioBERT; evaluate both models

| Task | Owner | Output |
|---|---|---|
| Prepare summarization dataset (train/val/test split) | ML sub-team | Cleaned dataset |
| Fine-tune BART-base on MeQSum + PubMed (Hugging Face Trainer) | ML sub-team | Fine-tuned model |
| Evaluate with ROUGE-1/2/L; compare with extractive baseline | ML sub-team | Results table |
| Few-shot fine-tune BioBERT for triage classification | ML sub-team | BioBERT model |
| Plot few-shot learning curve (10, 50, 100, 200 samples) | ML sub-team | Learning curve chart |
| Write report sections: Model Training, Experimental Setup | Writer | Draft sections |

**Deliverable:** `02_Summarization_BART.ipynb` + `03_BioBERT_Classification.ipynb`

---

### Week 4 — Prompt Engineering & System Integration (Technique 3)
**Goal:** Design and evaluate prompt strategies; integrate all modules into demo

| Task | Owner | Output |
|---|---|---|
| Design zero-shot, few-shot, CoT, and RAG prompt templates | Prompt sub-team | `prompts/` folder |
| Run systematic evaluation on held-out QA test set | Prompt sub-team | Prompt comparison table |
| Build Streamlit demo UI integrating all 4 modules | Demo sub-team | `app.py` |
| Write README with setup instructions and dependencies | Team lead | `README.md` |
| Record initial demo walkthrough (draft) | All | Draft video |
| Write report sections: Prompt Engineering, Results, Metrics | Writer | Draft sections 6–7 |

**Deliverable:** `04_Prompt_Engineering.ipynb` + working `app.py` demo

---

### Week 5 — Evaluation, Report Writing & Submission
**Goal:** Complete all evaluation, polish report, record final video, submit

| Task | Owner | Output |
|---|---|---|
| Run final evaluation for all modules; compile all results tables | All | Final results |
| Write Discussion section (limitations, challenges, lessons) | Writer | Section 8 |
| Write Conclusion & Future Work | Writer | Section 9 |
| Format all references in APA 7th edition | Writer | References list |
| Finalize report (Times New Roman 12pt, 1.5 spacing, 1-inch margins) | All | Final report PDF |
| Final code cleanup, comments, and documentation | All | Clean codebase |
| Record final 5-minute video presentation | All | Video link |
| Upload video to YouTube/Drive; add link to report | Team lead | Shareable link |
| Final submission package review | All | Submission-ready |

**Deliverable:** All submission artifacts ready — report, code, video, demo

---

## 7. Report Structure & Writing Plan

Follow the required structure. **Max 20 pages** (excluding references and appendices).

| # | Section | Pages | Content Notes | Assigned To |
|---|---|---|---|---|
| 1 | Executive Summary | 1 | One-page overview: problem, approach, key results | Team Lead |
| 2 | Problem Statement & Motivation | 1–1.5 | Why this problem matters; healthcare context; gap analysis | Writer A |
| 3 | Literature Review | 2–3 | 8–10 papers: clinical NLP, BERT in healthcare, T5 summarization, prompt engineering | Writer B |
| 4 | Methodology | 3–4 | Detail each of the 4 AI techniques; pipeline diagram; design decisions | ML Team |
| 5 | Implementation Details | 2–3 | Architecture, algorithms, frameworks, hyperparameters, hardware used | ML Team |
| 6 | Experimental Setup & Results | 2–3 | Datasets, splits, training configs, result tables, figures | Data Team |
| 7 | Evaluation & Performance Metrics | 1–2 | ROUGE, F1, accuracy, BERTScore; comparison with baselines | ML Team |
| 8 | Discussion | 1–1.5 | Limitations, failure cases, ethical considerations, challenges | All |
| 9 | Conclusion & Future Work | 0.5–1 | Summary of achievements; 3–5 specific future directions | Team Lead |
| 10 | References | Excluded | APA 7th edition; min 10 references | Writer A/B |
| 11 | Appendices | Excluded | Code snippets, extra charts, team contribution table | All |

### Formatting Checklist
- [ ] Font: Times New Roman, 12pt
- [ ] Line spacing: 1.5
- [ ] Margins: 1 inch on all sides
- [ ] Page numbers included
- [ ] Section headings present
- [ ] References in APA 7th edition format (see https://apastyle.apa.org)
- [ ] Total body ≤ 20 pages

---

## 8. Evaluation & Metrics Plan

Address all **100 marks** explicitly.

### Summarization (BART/T5)
- **ROUGE-1, ROUGE-2, ROUGE-L** — standard summarization metrics
- **BERTScore** — semantic similarity to reference summaries
- **Human evaluation** — 20 randomly sampled summaries rated for fluency + factuality (1–5)

### NER (spaCy/scispaCy)
- **Precision, Recall, F1** per entity type (disease, drug, symptom)
- Evaluated on BC5CDR test set

### Classification (BioBERT)
- **Accuracy, F1-macro, Confusion Matrix**
- **Few-shot learning curve** (plot F1 vs number of training examples)
- Baseline: Majority class + TF-IDF + Logistic Regression

### Prompt Engineering
- **Exact Match (EM)** and **F1** on MedQA test set
- Compare: zero-shot, few-shot (3-shot, 5-shot), CoT, RAG+few-shot
- Report: win rates and qualitative error analysis

### Baselines to Beat
| Module | Baseline | Target |
|---|---|---|
| Summarization | Extractive (BertSum) | ROUGE-L > 35 |
| NER | spaCy default en_core_web_sm | F1 > 80% on BC5CDR |
| Classification | TF-IDF + LR | F1-macro > 75% |
| QA via Prompting | Zero-shot GPT | 10%+ gain with few-shot/CoT |

---

## 9. Video Presentation Plan

**Duration:** Maximum 5 minutes
**Format:** Screen recording + face cam (optional) + voiceover
**All team members must speak**

### Suggested Structure (5 minutes)

| Segment | Duration | Speaker | Content |
|---|---|---|---|
| Introduction | 0:30 | Member 1 | Project title, domain, team intro |
| Problem Statement | 0:45 | Member 2 | Why this matters; the specific problem |
| Approach & Techniques | 1:15 | Member 3 | Walkthrough of the 4 AI techniques used |
| Live Demo | 1:30 | Member 1 | Run the Streamlit app with a real example |
| Results & Key Findings | 0:30 | Member 2 | Show 2–3 key result tables/charts |
| Conclusion | 0:30 | Member 3/All | Summary, limitations, future work |

### Video Checklist
- [ ] Duration ≤ 5 minutes
- [ ] Covers: problem, approach, results, demo
- [ ] All team members participate with speaking roles
- [ ] Uploaded to YouTube (unlisted) or Google Drive (shareable link)
- [ ] Link included in the final report

---

## 10. Submission Checklist

### Code Package
- [ ] Well-commented Python code / Jupyter notebooks
- [ ] `README.md` with:
  - [ ] Project description
  - [ ] Setup instructions (Python version, virtualenv/conda)
  - [ ] `requirements.txt` or `environment.yml`
  - [ ] How to run each notebook
  - [ ] Dataset download instructions
- [ ] Notebooks have markdown cells explaining each major step
- [ ] Code is reproducible with provided datasets (set random seeds)
- [ ] No hardcoded absolute file paths

### Notebooks Structure
```
project/
├── README.md
├── requirements.txt
├── app.py                          ← Streamlit demo
├── notebooks/
│   ├── 01_NLP_Pipeline.ipynb
│   ├── 02_Summarization_BART.ipynb
│   ├── 03_BioBERT_Classification.ipynb
│   └── 04_Prompt_Engineering.ipynb
├── src/
│   ├── preprocessing.py
│   ├── ner_module.py
│   ├── summarizer.py
│   ├── classifier.py
│   └── prompt_templates.py
├── data/
│   └── README.md                   ← Data download instructions only (no raw data committed)
├── models/
│   └── README.md                   ← Model download instructions
└── results/
    ├── rouge_scores.csv
    ├── ner_results.csv
    └── prompt_comparison.csv
```

### Report Package
- [ ] PDF of final report
- [ ] ≤ 20 pages (body, excluding references and appendices)
- [ ] Times New Roman, 12pt, 1.5 line spacing, 1-inch margins
- [ ] All sections present per required structure
- [ ] Appendix includes team contribution breakdown

### Video
- [ ] ≤ 5 minutes
- [ ] Shareable link confirmed working

---

## 11. Team Responsibilities

> Assign names based on your actual team. Adjust roles as needed.

| Role | Responsibilities | Techniques Owned |
|---|---|---|
| **Team Lead / Integrator** | Repo management, integration, README, video coordination | Overall pipeline, Streamlit demo |
| **NLP Engineer** | Pre-processing pipeline, FastText embeddings, scispaCy NER | Technique 1 (NLP) |
| **ML Engineer** | BART/T5 fine-tuning, BioBERT few-shot classification, evaluation | Techniques 2 & 4 |
| **Prompt Engineer** | Prompt design, CoT, few-shot, RAG implementation, QA evaluation | Technique 3 |
| **Report Writer** | Literature review, report formatting, APA citations, proofreading | Sections 2, 3, 8, 9 |

### Team Contribution Table (For Appendix)
| Member | Contribution | % |
|---|---|---|
| Member 1 | NLP pipeline, NER, preprocessing | 20% |
| Member 2 | BART fine-tuning, summarization evaluation | 20% |
| Member 3 | BioBERT few-shot classification | 20% |
| Member 4 | Prompt engineering, QA evaluation, demo UI | 20% |
| Member 5 | Report writing, literature review, video editing | 20% |

---

## 12. Grading Coverage Map

Explicit mapping of project work to every rubric criterion:

| Criteria | Points | How This Project Earns Full Marks |
|---|---|---|
| **Problem Relevance & Scope** | 10 | Healthcare domain is high-impact; problem (clinical note overload) is well-defined; scope (4 modules) is appropriately ambitious but achievable |
| **Use of AI Techniques** | 20 | Implements all 4 required technique categories: NLP (Technique 1), Transformer LLMs with fine-tuning (Technique 2), Transfer Learning (Technique 4), Prompt Engineering (Technique 3) |
| **Model Implementation & Creativity** | 20 | Clean, modular Python code; innovative RAG+CoT prompt strategy; integration of 4 separate modules into one pipeline; Streamlit demo |
| **Evaluation & Performance Metrics** | 15 | ROUGE, BERTScore, F1, accuracy, learning curves, prompt comparison tables, baseline comparisons for every module |
| **Final Report Quality** | 15 | Follows exact required structure; correct formatting (TNR 12pt, 1.5 spacing, 1-inch margins); APA 7th citations; professional writing |
| **Working Demo/Product** | 15 | Streamlit app demonstrates full pipeline; video shows live demo; all notebooks are reproducible |
| **Team Collaboration & Presentation** | 5 | All team members present in video; contribution table in appendix shows equal distribution; good video quality |
| **Total** | **100** | |

---

## Quick Reference: Key Resources

| Resource | URL |
|---|---|
| Hugging Face Datasets | https://huggingface.co/datasets |
| Kaggle Datasets | https://www.kaggle.com/datasets |
| UCI ML Repository | https://archive.ics.uci.edu/ml |
| OpenML | https://www.openml.org |
| APA Style Guide | https://apastyle.apa.org |
| scispaCy Models | https://allenai.github.io/scispacy |
| BioBERT (HuggingFace) | dmis-lab/biobert-base-cased-v1.2 |
| BART (HuggingFace) | facebook/bart-base |
| T5 (HuggingFace) | google/t5-small |

---

*Generated: 2026-03-15 | Advanced AI – Semester 7*

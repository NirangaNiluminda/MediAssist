# MediAssist — Stage-wise Execution Plan (2-Person Parallel)

> **Goal:** Ship the full project (4 AI techniques + Streamlit demo + report + video) with two engineers working in parallel.
> **Roles:**
> - **Person A — NLP & Summarization track** (Techniques 1 & 2)
> - **Person B — Classification & Prompt track** (Techniques 3 & 4)
>
> Both share report writing and the final demo integration.

---

## Parallelization Map (at a glance)

```
Stage 0 ── Setup ───────────────────────── (both, together)
Stage 1 ── Data + EDA ──────┬─ A: summarization/NER data
                            └─ B: classification/QA data
Stage 2 ── Core modules ────┬─ A: preprocessing + scispaCy NER
                            └─ B: BioBERT few-shot classifier
Stage 3 ── Heavy training ──┬─ A: BART/T5 fine-tune + ROUGE
                            └─ B: Prompt engineering (zero/few/CoT/RAG)
Stage 4 ── Integration ──────────────────── (both: Streamlit app)
Stage 5 ── Report + Video ──┬─ A: Methodology, Results
                            └─ B: Lit review, Discussion, References
```

Sync points: end of every stage — 30-min review call to merge branches and unblock.

---

## Repository Layout (target)

```
MediAssist/
├── README.md
├── EXECUTION_PLAN.md          ← this file
├── PROJECT_PLAN.md
├── requirements.txt
├── app.py                     ← Streamlit demo
├── notebooks/
│   ├── 00_EDA.ipynb
│   ├── 01_NLP_Pipeline.ipynb
│   ├── 02_Summarization_BART.ipynb
│   ├── 03_BioBERT_Classification.ipynb
│   └── 04_Prompt_Engineering.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── ner_module.py
│   ├── summarizer.py
│   ├── classifier.py
│   ├── prompt_templates.py
│   └── rag_utils.py
├── data/                      ← raw + processed (gitignored)
├── models/                    ← saved checkpoints (gitignored)
└── results/
    ├── rouge_scores.csv
    ├── ner_results.csv
    ├── classification_results.csv
    └── prompt_comparison.csv
```

### Branching strategy
- `main` — always working
- `feat/a-nlp-pipeline`, `feat/a-summarizer` — Person A
- `feat/b-classifier`, `feat/b-prompt` — Person B
- Open a PR at the end of each stage; the other person reviews before merge.

---

## Stage 0 — Setup (Day 1, ~3 hours, both together)

**Goal:** Identical, working environment on both machines.

### Steps
1. **Clone the repo** on both machines.
   ```bash
   git clone <repo-url> && cd MediAssist
   ```
2. **Create the Python environment** (Python 3.10).
   ```bash
   python3.10 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. **Install spaCy + scispaCy models.**
   ```bash
   python -m spacy download en_core_web_sm
   pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz
   python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"
   ```
4. **Smoke test** — both run a 3-line script importing `spacy`, `transformers`, `streamlit`. Commit nothing if it fails.
5. **Create folder skeleton & empty modules** (the directory layout above). Push to `main`.
6. **Set up Hugging Face & OpenAI accounts** (both):
   - `huggingface-cli login`
   - Add `OPENAI_API_KEY` to a local `.env` (gitignored). If no API budget, use **Ollama + Llama-3-8B** locally instead — same prompts, free.
7. **Pick the compute target:** Google Colab (free T4) for fine-tuning if no local GPU. Decide and document in README.

**Exit criteria:** Both machines run `python -c "import torch; print(torch.cuda.is_available())"` successfully and `streamlit hello` opens.

---

## Stage 1 — Data Acquisition + EDA (Days 2–3, ~1 day)

**Goal:** Every dataset downloaded, inspected, split.

### Person A — Summarization & NER data
1. **Download MeQSum** (consumer health QA summarization).
   ```python
   from datasets import load_dataset
   ds = load_dataset("sumedh/MeQSum")
   ```
2. **Download PubMed summarization** subset (~10k slice, full is too big).
   ```python
   ds = load_dataset("ccdv/pubmed-summarization", "document", split="train[:10000]")
   ```
3. **Download BC5CDR** for NER eval.
   ```python
   ds = load_dataset("bigbio/bc5cdr", "bc5cdr_bigbio_kb")
   ```
4. **In `notebooks/00_EDA.ipynb` (section A):** plot length distributions, missing values, sample 5 examples per dataset.
5. **Commit cleaned splits** as `data/processed/{summ_train,summ_val,summ_test}.jsonl`.

### Person B — Classification & QA data
1. **Build a triage labelling rule** for clinical notes (urgent/routine/follow-up). Use:
   - **MedNLI** if you can get PhysioNet access; otherwise
   - Synthetic: take MeQSum questions and weak-label by keyword rules → 600 examples; have Person A spot-check 50.
2. **MedQA dataset** for QA evaluation:
   ```python
   ds = load_dataset("bigbio/med_qa", "med_qa_en_source")
   ```
3. **In `notebooks/00_EDA.ipynb` (section B):** class balance plots, token length histograms, sample 5 QA pairs.
4. **Commit splits** as `data/processed/{triage_train,triage_val,triage_test}.csv` and `data/processed/medqa_test.jsonl`.

**Sync point:** Both walk through the EDA notebook together, agree on text-length cutoffs (e.g., truncate to 512 tokens for BERT models, 1024 for BART).

**Exit criteria:** All 5 dataset splits committed with sizes documented in `data/README.md`.

---

## Stage 2 — Core Modules (Days 4–7, ~4 days, fully parallel)

### Person A — NLP Pipeline + NER (`src/preprocessing.py`, `src/ner_module.py`)

#### Steps
1. **`preprocessing.py`** — implement these functions:
   - `clean_text(s) -> str` — lowercasing (preserve casing for NER input separately), whitespace, punctuation rules
   - `tokenize(s) -> List[str]` — spaCy tokenizer
   - `lemmatize(tokens) -> List[str]`
   - `remove_stopwords(tokens) -> List[str]`
2. **`ner_module.py`** — implement:
   - `load_ner_model() -> Language` (loads `en_core_sci_md`)
   - `extract_entities(text) -> Dict[str, List[str]]` returning `{"diseases": [...], "drugs": [...], "symptoms": [...]}`
   - Use scispaCy `LinkerComponent` to attach UMLS CUIs (bonus marks).
3. **FastText embeddings** (optional but in the plan):
   - Train on the cleaned PubMed subset:
     ```bash
     python -c "import fasttext; m = fasttext.train_unsupervised('data/processed/pubmed_corpus.txt', model='skipgram', dim=100); m.save_model('models/fasttext_med.bin')"
     ```
4. **`notebooks/01_NLP_Pipeline.ipynb`**:
   - End-to-end demo on 3 sample notes
   - **Eval on BC5CDR test set:** report per-entity Precision / Recall / F1
   - **Baseline comparison:** same metrics with default `en_core_web_sm`
5. **Save** `results/ner_results.csv` with columns `entity_type, precision, recall, f1, model`.

**Definition of done:** F1 > 0.80 on diseases & chemicals on BC5CDR; notebook runs top-to-bottom without errors.

### Person B — BioBERT Few-shot Classifier (`src/classifier.py`)

#### Steps
1. **`src/classifier.py`** — implement:
   - `train_classifier(train_df, n_examples, model_name="dmis-lab/biobert-base-cased-v1.2") -> Trainer`
   - `predict(text, model) -> str`
2. **`notebooks/03_BioBERT_Classification.ipynb`**:
   - Load `triage_train`
   - Sweep `n = [10, 50, 100, 200, all]` and train BioBERT with HF `Trainer` (3 epochs, lr 2e-5, batch 16, fp16)
   - Plot **F1-macro vs n** as the few-shot learning curve → save `results/few_shot_curve.png`
   - **Baselines:**
     - Majority class
     - TF-IDF + Logistic Regression (sklearn)
     - Zero-shot via `facebook/bart-large-mnli` NLI prompts
3. **Save** `results/classification_results.csv` with columns `method, n_train, accuracy, f1_macro, f1_weighted`.

**Definition of done:** BioBERT (n=200) F1-macro > 0.75 and beats TF-IDF baseline; learning curve plot included.

**Sync point:** End of Stage 2, merge both PRs into `main`. Person A reviews B's PR, vice versa.

---

## Stage 3 — Heavy Training (Days 8–12, ~5 days, parallel — needs GPU)

### Person A — BART/T5 Summarization (`src/summarizer.py`)

#### Steps
1. **`src/summarizer.py`** — implement:
   - `prepare_dataset(jsonl_path, tokenizer, max_input=1024, max_target=128) -> Dataset`
   - `train_summarizer(model_name="facebook/bart-base", train_ds, val_ds, output_dir) -> Trainer`
   - `summarize(text, model, tokenizer) -> str`
2. **`notebooks/02_Summarization_BART.ipynb`** on Colab T4:
   - Fine-tune BART-base on MeQSum (3 epochs, batch 8, lr 3e-5, fp16, gradient_accumulation=2)
   - Generate summaries on `summ_test`
   - Compute **ROUGE-1, ROUGE-2, ROUGE-L** and **BERTScore**
   - **Baseline:** extractive — `sumy` LexRank or `bertsum`-style first-3-sentence baseline
   - **Human eval:** 20 random samples, both team members rate fluency + factuality 1–5; average
3. **Save**:
   - Checkpoint to `models/bart_meqsum/` (or upload to HF Hub)
   - `results/rouge_scores.csv` with columns `model, rouge1, rouge2, rougeL, bertscore_f1`

**Definition of done:** ROUGE-L > 35 on MeQSum test; checkpoint loadable in `app.py`.

### Person B — Prompt Engineering + RAG (`src/prompt_templates.py`, `src/rag_utils.py`)

#### Steps
1. **`src/prompt_templates.py`** — define 4 templates:
   - `ZERO_SHOT` — plain instruction
   - `FEW_SHOT` — 3–5 in-context examples drawn from MedQA train
   - `COT` — "Let's think step by step…" reasoning prefix
   - `RAG_FEW_SHOT` — retrieved top-k chunks + few-shot examples
2. **`src/rag_utils.py`** — implement a small retriever:
   - Index `pubmed_corpus.txt` chunks with `sentence-transformers/all-MiniLM-L6-v2` → FAISS
   - `retrieve(query, k=3) -> List[str]`
3. **`notebooks/04_Prompt_Engineering.ipynb`**:
   - Loop over `medqa_test` (subsample 200 questions to keep API costs bounded)
   - For each prompt strategy, call:
     - GPT-3.5-turbo (`openai` SDK) **OR** local Llama-3-8B via Ollama (`requests` to `localhost:11434`)
   - Score with **Exact Match** and token-F1
   - **Qualitative analysis:** pick 5 questions where CoT helps and 5 where it hurts
4. **Save** `results/prompt_comparison.csv` with columns `strategy, em, f1, n`.

**Definition of done:** few-shot/CoT shows ≥10% F1 gain over zero-shot; one comparison table in notebook.

**Sync point:** Mid-stage check on Day 10 — share preliminary numbers in Slack/chat. Adjust hyperparameters if metrics are off.

---

## Stage 4 — Integration: Streamlit Demo (Days 13–14, ~2 days, both together)

**Goal:** A single `app.py` that runs the full pipeline on user-pasted text.

### Steps
1. **Decide UI layout** (sketch on paper first):
   - Sidebar: model toggles, prompt strategy selector
   - Main: textarea input → "Analyze" button → tabbed output (Entities | Summary | Triage | Q&A)
2. **`app.py` skeleton** (Person A drafts; Person B fills tabs):
   ```python
   import streamlit as st
   from src.ner_module import load_ner_model, extract_entities
   from src.summarizer import summarize, load_summarizer
   from src.classifier import predict, load_classifier
   from src.prompt_templates import answer_question

   @st.cache_resource
   def load_all():
       return load_ner_model(), load_summarizer(), load_classifier()
   ```
3. **Wire each tab:**
   - **Entities tab** (A): call `extract_entities`, render with `st.dataframe`
   - **Summary tab** (A): call `summarize`, show diff vs original length
   - **Triage tab** (B): call `predict`, show class + confidence bar
   - **Q&A tab** (B): text input for question, dropdown for prompt strategy, call selected template
4. **Cache models** with `@st.cache_resource` so the app loads once.
5. **Run locally** `streamlit run app.py` — both test on 3 different sample notes.
6. **Polish:** add a "Load example" button with 2 pre-canned clinical notes.

**Definition of done:** End-to-end demo works on a fresh machine after `pip install -r requirements.txt && streamlit run app.py`.

---

## Stage 5 — Report + Video + Submission (Days 15–18, ~4 days)

### Report writing — split as below
| Section | Owner |
|---|---|
| 1. Executive Summary | Both (last) |
| 2. Problem Statement & Motivation | B |
| 3. Literature Review (8–10 papers) | B |
| 4. Methodology | A |
| 5. Implementation Details | A |
| 6. Experimental Setup & Results | A |
| 7. Evaluation & Performance Metrics | A |
| 8. Discussion (limitations, ethics) | B |
| 9. Conclusion & Future Work | B |
| 10. References (APA 7) | B |
| 11. Appendix (contribution table, extra plots) | Both |

### Steps
1. **Day 15–16:** First draft of all sections in a shared Google Doc / Overleaf project.
2. **Day 17:** Cross-review — A edits B's sections and vice versa. Format: TNR 12pt, 1.5 line spacing, 1-inch margins, page numbers, ≤20 pages body.
3. **Day 17 evening: Record video** (≤5 min):
   - 0:00–0:30 Intro (both speak)
   - 0:30–1:15 Problem (B)
   - 1:15–2:30 Approach + 4 techniques (A & B alternating)
   - 2:30–4:00 Live demo from `app.py` (A driving, B narrating)
   - 4:00–4:30 Results (A)
   - 4:30–5:00 Conclusion (both)
   - Tools: OBS Studio (free) → upload to YouTube unlisted → drop link in report.
4. **Day 18:** Final pass — proofread, verify all CSVs are committed in `results/`, run `app.py` on a clean venv, tag a release `git tag v1.0`.

### Submission checklist
- [ ] PDF of report (≤20 pages body, APA refs, formatted)
- [ ] Code repo zipped or GitHub link
- [ ] All 5 notebooks run top-to-bottom
- [ ] `results/` populated
- [ ] Video link working (unlisted YouTube or shareable Drive)
- [ ] Team contribution table in appendix
- [ ] No hardcoded absolute paths in code
- [ ] No raw datasets committed (only `data/README.md` instructions)

---

## Daily Standup Template (5 minutes, async over chat)

```
Yesterday: <what you finished>
Today:     <what you're doing>
Blocked:   <anything you need from the other person>
ETA risk:  <on track | slipping | blocked>
```

---

## Risk Register & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| No GPU access | Medium | Use Google Colab free T4; downgrade BART-base → T5-small if OOM |
| OpenAI API budget runs out | Medium | Switch to local Ollama + Llama-3-8B for prompt eval |
| MIMIC-III access denied | High | Skip — MeQSum + PubMed cover all required tasks |
| BioBERT few-shot underperforms | Low | Fall back to full-data fine-tune; still demonstrates Transfer Learning |
| Streamlit integration breaks | Low | Keep CLI fallback in `src/main.py` that runs the same pipeline |
| One person falls sick | Medium | All notebooks must be runnable independently; pair-review at every stage gate |

---

## Quick Stage-Owner Lookup

| Stage | Person A focus | Person B focus | Deliverable |
|---|---|---|---|
| 0 — Setup | Env + repo skeleton | HF/OpenAI accounts | Working envs |
| 1 — Data | Summ/NER datasets | Classification/QA datasets | `00_EDA.ipynb` |
| 2 — Core | `preprocessing.py`, `ner_module.py`, `01_NLP_Pipeline.ipynb` | `classifier.py`, `03_BioBERT_Classification.ipynb` | NER F1 + few-shot curve |
| 3 — Training | `summarizer.py`, `02_Summarization_BART.ipynb` | `prompt_templates.py`, `rag_utils.py`, `04_Prompt_Engineering.ipynb` | ROUGE + prompt comparison |
| 4 — Integration | App skeleton + Entities/Summary tabs | Triage/Q&A tabs | `app.py` |
| 5 — Report+Video | Sections 4–7 | Sections 2,3,8,9,10 | PDF + video link |

---

*Total elapsed time: ~18 working days (≈3.5 weeks at 5 days/week).*
*Generated: 2026-05-01*

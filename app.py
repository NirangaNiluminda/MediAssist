"""
MediAssist — Streamlit Demo
Integrates all 4 AI modules into a single web UI.
Owner: Both (Stage 4)
Run: streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="MediAssist", page_icon="🏥", layout="wide")

# ---------------------------------------------------------------------------
# Lazy model loading (cached so they load only once per session)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading NER model...")
def get_ner():
    from src.ner_module import load_ner_model
    return load_ner_model()


@st.cache_resource(show_spinner="Loading summarizer...")
def get_summarizer():
    from src.summarizer import load_summarizer
    return load_summarizer(checkpoint_path="models/bart_meqsum")


@st.cache_resource(show_spinner="Loading classifier...")
def get_classifier():
    from src.classifier import load_classifier
    return load_classifier(checkpoint_path="models/biobert_triage")


# ---------------------------------------------------------------------------
# Pre-canned sample notes for quick demo
# ---------------------------------------------------------------------------

SAMPLES = {
    "Sample 1 — Chest pain": (
        "Patient is a 58-year-old male presenting with acute chest pain radiating to the left arm, "
        "diaphoresis, and shortness of breath for the past 2 hours. History of hypertension and type 2 "
        "diabetes managed with Metformin 500 mg twice daily and Lisinopril 10 mg once daily. ECG shows "
        "ST-elevation in leads II, III, aVF. Troponin I elevated at 2.4 ng/mL. Impression: STEMI. "
        "Initiated aspirin 325 mg, heparin infusion, and cath lab activated."
    ),
    "Sample 2 — Respiratory infection": (
        "A 34-year-old female presents with productive cough for 5 days, fever up to 38.9°C, and right-sided "
        "pleuritic chest pain. Decreased breath sounds and dullness to percussion at the right base. "
        "CXR confirms right lower lobe consolidation. WBC 14,200 with left shift. Diagnosis: community-acquired "
        "pneumonia. Prescribed Amoxicillin-Clavulanate 875 mg twice daily for 7 days."
    ),
}

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("MediAssist")
    st.caption("Advanced AI – Group Project 2026")
    st.divider()

    sample_choice = st.selectbox("Load a sample note", ["(paste your own)"] + list(SAMPLES.keys()))
    st.divider()

    prompt_strategy = st.selectbox(
        "Q&A prompt strategy",
        ["few_shot", "zero_shot", "cot", "rag_few_shot"],
        help="Strategy used in the Q&A tab",
    )
    st.divider()
    st.caption("Models used:\n- scispaCy en_core_sci_md\n- BART-base (fine-tuned)\n- BioBERT (few-shot)\n- GPT-3.5 / Llama-3")

# ---------------------------------------------------------------------------
# Main input area
# ---------------------------------------------------------------------------

st.title("MediAssist: Medical Document Analysis")

default_text = SAMPLES.get(sample_choice, "") if sample_choice != "(paste your own)" else ""
clinical_note = st.text_area(
    "Clinical Note",
    value=default_text,
    height=200,
    placeholder="Paste a clinical note or discharge summary here…",
)

analyze = st.button("Analyze", type="primary", disabled=not clinical_note.strip())

if analyze and clinical_note.strip():
    tab_entities, tab_summary, tab_triage, tab_qa = st.tabs(
        ["Entities", "Summary", "Triage", "Q&A"]
    )

    # ── Tab 1: NER Entities (Person A) ──────────────────────────────────────
    with tab_entities:
        st.subheader("Extracted Medical Entities")
        with st.spinner("Running NER…"):
            from src.ner_module import extract_entities
            nlp = get_ner()
            entities = extract_entities(clinical_note, nlp=nlp)

        for category, items in entities.items():
            if items:
                st.markdown(f"**{category.capitalize()}**")
                st.write(", ".join(items))
            else:
                st.markdown(f"**{category.capitalize()}** — none detected")

    # ── Tab 2: Abstractive Summary (Person A) ────────────────────────────────
    with tab_summary:
        st.subheader("Abstractive Summary")
        with st.spinner("Generating summary…"):
            from src.summarizer import summarize
            model, tokenizer = get_summarizer()
            summary = summarize(clinical_note, model, tokenizer)

        st.success(summary)
        orig_words = len(clinical_note.split())
        summ_words = len(summary.split())
        st.caption(f"Compressed {orig_words} → {summ_words} words ({100 - round(summ_words/orig_words*100)}% reduction)")

    # ── Tab 3: Triage Classification (Person B) ──────────────────────────────
    with tab_triage:
        st.subheader("Triage Classification")
        with st.spinner("Classifying…"):
            from src.classifier import predict
            clf_model, clf_tok = get_classifier()
            result = predict(clinical_note, model=clf_model, tokenizer=clf_tok)

        label_colors = {"urgent": "red", "routine": "green", "follow-up": "orange"}
        color = label_colors.get(result["label"], "grey")
        st.markdown(f"### :{color}[{result['label'].upper()}]")
        st.caption(f"Confidence: {result['confidence']:.1%}")

        st.divider()
        st.caption("Class probabilities")
        for lbl, prob in result["probabilities"].items():
            st.progress(prob, text=f"{lbl}: {prob:.1%}")

    # ── Tab 4: Q&A (Person B) ─────────────────────────────────────────────
    with tab_qa:
        st.subheader("Medical Q&A")
        question = st.text_input("Ask a question about this note…", placeholder="What medication was prescribed?")

        if st.button("Ask", disabled=not question.strip()):
            with st.spinner(f"Answering ({prompt_strategy})…"):
                from src.prompt_templates import answer_question
                from src.rag_utils import retrieve

                kwargs = {}
                if prompt_strategy == "rag_few_shot":
                    # TODO (Person B): swap corpus path after building FAISS index
                    try:
                        context = retrieve(question)
                        kwargs["context"] = context
                    except Exception:
                        st.warning("RAG index not built yet — falling back to few_shot.")
                        prompt_strategy = "few_shot"

                answer = answer_question(question, strategy=prompt_strategy, **kwargs)

            st.info(f"**Q:** {question}")
            st.success(f"**A:** {answer}")
            st.caption(f"Strategy: `{prompt_strategy}`")

elif not clinical_note.strip() and analyze:
    st.warning("Please enter a clinical note first.")

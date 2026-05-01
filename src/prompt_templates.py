"""
Prompt templates for medical QA: zero-shot, few-shot, CoT, RAG+few-shot.
Owner: Person B (Stage 3)
"""
import os
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Template definitions
# ---------------------------------------------------------------------------

ZERO_SHOT = """\
You are a medical assistant. Answer the following clinical question accurately and concisely.

Question: {question}
Answer:"""

FEW_SHOT = """\
You are a medical assistant. Here are some example questions and answers:

{examples}

Now answer this question:
Question: {question}
Answer:"""

COT = """\
You are a medical assistant. Think step by step before giving your final answer.

Question: {question}

Let's think step by step:
1."""

RAG_FEW_SHOT = """\
You are a medical assistant. Use the following retrieved context to help answer the question.

Context:
{context}

Examples:
{examples}

Question: {question}
Answer:"""

# ---------------------------------------------------------------------------
# Few-shot example bank  (add more in 04_Prompt_Engineering.ipynb)
# ---------------------------------------------------------------------------

DEFAULT_EXAMPLES = [
    {
        "question": "What is the first-line treatment for type 2 diabetes?",
        "answer": "Metformin is the recommended first-line pharmacological treatment alongside lifestyle modifications.",
    },
    {
        "question": "What are common symptoms of pneumonia?",
        "answer": "Common symptoms include fever, cough with phlegm, shortness of breath, and chest pain.",
    },
    {
        "question": "What does elevated troponin indicate?",
        "answer": "Elevated troponin levels suggest myocardial injury, most commonly acute myocardial infarction.",
    },
]


def _format_examples(examples: List[dict]) -> str:
    return "\n\n".join(
        f"Question: {e['question']}\nAnswer: {e['answer']}" for e in examples
    )


# ---------------------------------------------------------------------------
# LLM call helper — supports OpenAI API or local Ollama
# ---------------------------------------------------------------------------

def _call_llm(prompt: str, model: str = "gpt-3.5-turbo", max_tokens: int = 256) -> str:
    """
    Call OpenAI API if OPENAI_API_KEY is set, else fall back to local Ollama.
    TODO (Person B): test both paths; set OLLAMA_MODEL in .env for local use.
    """
    api_key = os.getenv("OPENAI_API_KEY")

    if api_key:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()

    # Fallback: Ollama local (run: ollama pull llama3)
    import requests
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3")
    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": ollama_model, "prompt": prompt, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


# ---------------------------------------------------------------------------
# Public strategy functions
# ---------------------------------------------------------------------------

def answer_zero_shot(question: str, **kwargs) -> str:
    prompt = ZERO_SHOT.format(question=question)
    return _call_llm(prompt, **kwargs)


def answer_few_shot(question: str, examples: Optional[List[dict]] = None, **kwargs) -> str:
    ex = examples or DEFAULT_EXAMPLES
    prompt = FEW_SHOT.format(examples=_format_examples(ex), question=question)
    return _call_llm(prompt, **kwargs)


def answer_cot(question: str, **kwargs) -> str:
    prompt = COT.format(question=question)
    return _call_llm(prompt, **kwargs)


def answer_rag_few_shot(question: str, context: str, examples: Optional[List[dict]] = None, **kwargs) -> str:
    ex = examples or DEFAULT_EXAMPLES
    prompt = RAG_FEW_SHOT.format(context=context, examples=_format_examples(ex), question=question)
    return _call_llm(prompt, **kwargs)


STRATEGY_MAP = {
    "zero_shot": answer_zero_shot,
    "few_shot": answer_few_shot,
    "cot": answer_cot,
    "rag_few_shot": answer_rag_few_shot,
}


def answer_question(question: str, strategy: str = "few_shot", **kwargs) -> str:
    """Unified entry point used by app.py."""
    fn = STRATEGY_MAP.get(strategy)
    if fn is None:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from: {list(STRATEGY_MAP)}")
    return fn(question, **kwargs)

import os
import re
import requests
from typing import List, Dict, Any

from rapidfuzz import fuzz, process
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

API_URL = "https://november7-730026606190.europe-west1.run.app/messages/"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QA_MODEL = os.getenv("QA_MODEL", "gpt-4o-mini")

# ---------------- Fetch messages safely ----------------
def fetch_messages(limit: int = 200) -> List[Dict[str, Any]]:
    """
    Safe message fetcher with retry and fallback.
    Only tries to fetch first few pages to avoid timeout / 403 / 401.
    """

    headers = {
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (AuroraQA/1.0)"
    }

    all_msgs = []
    skip = 0
    page_size = 100

    for _ in range(3):  # retry up to 3 times
        try:
            resp = requests.get(
                API_URL,
                params={"skip": skip, "limit": page_size},
                headers=headers,
                timeout=30,  # <-- increased timeout
            )
            if resp.status_code == 403:
                print("Forbidden: API may be rate-limiting. Returning partial results.")
                return all_msgs

            if resp.status_code == 401:
                print("Unauthorized or blocked: returning messages retrieved so far.")
                return all_msgs

            resp.raise_for_status()
            data = resp.json()
            items = data.get("items", [])

            if not items:
                break

            all_msgs.extend(items)
            skip += page_size

            if skip >= limit:
                break

        except Exception as e:
            print(f"Retry fetch failed due to: {e}")
            continue

    # Normalize
    for m in all_msgs:
        m["member_id"] = m.pop("user_id", None)
        m["member_name"] = m.pop("user_name", None)
        m["text"] = m.pop("message", "") or ""

    print(f"Fetched {len(all_msgs)} messages after retry strategy.")
    return all_msgs

# ---------------- Retrieval ----------------
def _msg_text(m: Dict[str, Any]) -> str:
    return f"{m.get('member_name','')} :: {m.get('text','')}"

def retrieve_top(messages: List[Dict[str, Any]], question: str, k: int = 24) -> List[Dict[str, Any]]:
    if not messages:
        return []

    corpus = [(_msg_text(m), m) for m in messages]

    strings = [c[0] for c in corpus]
    results = process.extract(question, strings, scorer=fuzz.token_set_ratio, limit=min(k, len(strings)))

    ranked = []
    for _, score, idx in results:
        if score >= 40:
            ranked.append(corpus[idx][1])

    if not ranked:
        ranked = [m for _, m in corpus[:k]]

    return ranked[:k]

# ---------------- Build context ----------------
def format_context(snippets: List[Dict[str, Any]], max_chars: int = 8000) -> str:
    lines = []
    used = 0

    for m in snippets:
        name = m.get("member_name")
        ts = m.get("timestamp")
        mid = m.get("id")
        txt = re.sub(r"\s+", " ", m.get("text", "")).strip()

        line = f"- {name} | {ts} | {mid}: {txt}"
        if used + len(line) > max_chars:
            break
        lines.append(line)
        used += len(line)

    return "\n".join(lines)

# ---------------- LLM Answer ----------------
def llm_answer(question: str, context: str) -> str:
    if not OPENAI_API_KEY:
        return "Error: OPENAI_API_KEY missing"

    client = OpenAI(api_key=OPENAI_API_KEY)

    system_prompt = (
        "You are a helpful assistant that answers questions strictly using the provided message context. "
        "If the answer is not in the context, respond exactly with 'I don’t know'. "
        "Do not hallucinate or invent details."
    )

    user_prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

    resp = client.chat.completions.create(
        model=QA_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=200,
        temperature=0.1,
    )

    return resp.choices[0].message.content.strip()

# ---------------- Pipeline ----------------
def route(question: str, messages: List[Dict[str, Any]]) -> str:
    snippets = retrieve_top(messages, question, k=24)
    context = format_context(snippets)
    return llm_answer(question, context)